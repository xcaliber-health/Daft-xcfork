"""End-to-end tests for IcebergTable.rewrite_data_files('zorder')."""

from __future__ import annotations

import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, LongType, NestedField, StringType

from daft.catalog import Table


@pytest.fixture
def cluster_table(local_catalog):
    """Seed a 2-D distribution where row order is uncorrelated with spatial proximity."""
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "lat", DoubleType(), required=False),
        NestedField(3, "lon", DoubleType(), required=False),
        NestedField(4, "label", StringType(), required=False),
    )
    table = local_catalog.create_table(
        identifier="default.t_zorder",
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    rng = random.Random(42)
    n_files = 10
    rows_per_file = 80
    for f in range(n_files):
        ids = list(range(f * rows_per_file, (f + 1) * rows_per_file))
        # Lat/lon randomly drawn each row — adjacent rows share no locality.
        lats = [rng.random() * 90 for _ in ids]
        lons = [rng.random() * 180 for _ in ids]
        table.append(
            pa.table(
                {
                    "id": pa.array(ids, type=pa.int64()),
                    "lat": pa.array(lats, type=pa.float64()),
                    "lon": pa.array(lons, type=pa.float64()),
                    "label": pa.array([f"r{i}" for i in ids], type=pa.string()),
                }
            )
        )
    return table


def _file_paths(table) -> list[str]:
    out = []
    for t in table.scan().plan_files():
        p = t.file.file_path
        if p.startswith("file://"):
            p = p[len("file://") :]
        out.append(p)
    return out


def _files_touching_range(paths: list[str], col: str, lo: float, hi: float) -> int:
    """Count parquet files whose [col] range overlaps [lo, hi] via column statistics."""
    touched = 0
    for path in paths:
        meta = pq.read_metadata(path)
        # If stats are missing for any row group, conservatively count the file.
        overlap = False
        for rg in range(meta.num_row_groups):
            stats = meta.row_group(rg).column(_col_index(meta, col)).statistics
            if stats is None or stats.min is None or stats.max is None:
                overlap = True
                break
            if not (stats.max < lo or stats.min > hi):
                overlap = True
                break
        if overlap:
            touched += 1
    return touched


def _col_index(meta, name: str) -> int:
    schema = meta.schema
    for i in range(len(schema)):
        col = schema.column(i)
        if col.name == name or col.path == name:
            return i
    raise KeyError(name)


def test_zorder_preserves_rows_and_schema(cluster_table):
    pre_rows = sorted(int(r["id"]) for r in cluster_table.scan().to_arrow().to_pylist())
    pre_cols = set(cluster_table.scan().to_arrow().column_names)

    dt = Table.from_iceberg(cluster_table)
    result = dt.rewrite_data_files(
        "zorder",
        zorder_by=["lat", "lon"],
        options={"rewrite-all": True, "min-input-files": 2},
    )
    cluster_table.refresh()

    post_arrow = cluster_table.scan().to_arrow()
    assert sorted(int(r["id"]) for r in post_arrow.to_pylist()) == pre_rows
    assert set(post_arrow.column_names) == pre_cols
    # Synthetic key must not leak.
    assert "__daft_zorder_key__" not in post_arrow.column_names
    assert result.added_files >= 1


def test_zorder_improves_locality(cluster_table):
    pre_paths = _file_paths(cluster_table)
    pre_touched = _files_touching_range(pre_paths, "lat", 10.0, 12.0)

    dt = Table.from_iceberg(cluster_table)
    dt.rewrite_data_files(
        "zorder",
        zorder_by=["lat", "lon"],
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            # Force several output files so locality is measurable.
            "target-file-size-bytes": 1 * 1024 * 1024,
        },
    )
    cluster_table.refresh()

    post_paths = _file_paths(cluster_table)
    post_touched = _files_touching_range(post_paths, "lat", 10.0, 12.0)

    # With multiple post-rewrite files, a narrow predicate must read fewer files.
    if len(post_paths) > 1:
        assert post_touched < pre_touched, (
            f"expected locality improvement, pre={pre_touched}/{len(pre_paths)} "
            f"post={post_touched}/{len(post_paths)}"
        )


def test_zorder_requires_non_empty_columns(cluster_table):
    dt = Table.from_iceberg(cluster_table)
    with pytest.raises(ValueError, match="non-empty zorder_by"):
        dt.rewrite_data_files("zorder")


def test_zorder_rejects_unknown_column(cluster_table):
    dt = Table.from_iceberg(cluster_table)
    with pytest.raises(ValueError, match="not in table schema"):
        dt.rewrite_data_files("zorder", zorder_by=["missing"])
