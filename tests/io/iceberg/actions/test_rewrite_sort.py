"""End-to-end tests for IcebergTable.rewrite_data_files('sort')."""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

from daft.catalog import Table


@pytest.fixture
def shuffled_table(local_catalog):
    """Seed an unpartitioned table with reverse-and-interleaved data across many files."""
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "label", StringType(), required=False),
        NestedField(3, "bucket", LongType(), required=False),
    )
    table = local_catalog.create_table(
        identifier="default.t_sort",
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    # Write 10 small files in deliberately reversed/interleaved id order so a global
    # sort visibly reorders rows.
    n_files = 10
    rows_per_file = 6
    total = n_files * rows_per_file
    for f in range(n_files):
        ids = list(range(total - 1 - f, -1, -n_files))  # interleaved descending
        table.append(
            pa.table(
                {
                    "id": pa.array(ids, type=pa.int64()),
                    "label": pa.array([f"row-{i}" for i in ids], type=pa.string()),
                    "bucket": pa.array([i % 3 for i in ids], type=pa.int64()),
                }
            )
        )
    return table


def _read_parquet_sorted(path: str, by: str, descending: bool) -> bool:
    import pyarrow.parquet as pq

    tbl = pq.read_table(path)
    vals = tbl.column(by).to_pylist()
    if descending:
        return all(a >= b for a, b in zip(vals, vals[1:]))
    return all(a <= b for a, b in zip(vals, vals[1:]))


def _output_data_paths(table) -> list[str]:
    out = []
    for t in table.scan().plan_files():
        p = t.file.file_path
        if p.startswith("file://"):
            p = p[len("file://") :]
        out.append(p)
    return out


def test_sort_ascending_per_file_monotonic(shuffled_table):
    pre_ids = sorted(int(r["id"]) for r in shuffled_table.scan().to_arrow().to_pylist())

    dt = Table.from_iceberg(shuffled_table)
    result = dt.rewrite_data_files(
        "sort",
        sort_order=[("id", "asc", "nulls-last")],
        options={
            "target-file-size-bytes": 16 * 1024 * 1024,
            "min-input-files": 2,
            "rewrite-all": True,
        },
    )
    shuffled_table.refresh()
    post_ids = sorted(
        int(r["id"]) for r in shuffled_table.scan().to_arrow().to_pylist()
    )
    assert post_ids == pre_ids
    assert result.added_files >= 1

    for path in _output_data_paths(shuffled_table):
        assert _read_parquet_sorted(
            path, "id", descending=False
        ), f"output file {path!r} is not ascending in id"


def test_sort_descending_per_file_monotonic(shuffled_table):
    dt = Table.from_iceberg(shuffled_table)
    dt.rewrite_data_files(
        "sort",
        sort_order=[("id", "desc", "nulls-first")],
        options={"rewrite-all": True, "min-input-files": 2},
    )
    shuffled_table.refresh()
    for path in _output_data_paths(shuffled_table):
        assert _read_parquet_sorted(path, "id", descending=True)


def test_sort_requires_non_empty_order(shuffled_table):
    dt = Table.from_iceberg(shuffled_table)
    with pytest.raises(ValueError, match="non-empty sort_order"):
        dt.rewrite_data_files("sort")


def test_sort_rejects_unknown_column(shuffled_table):
    dt = Table.from_iceberg(shuffled_table)
    with pytest.raises(ValueError, match="not in table schema"):
        dt.rewrite_data_files("sort", sort_order=[("missing", "asc", "nulls-last")])


def test_sort_multi_column(shuffled_table):
    dt = Table.from_iceberg(shuffled_table)
    dt.rewrite_data_files(
        "sort",
        sort_order=[
            ("bucket", "asc", "nulls-last"),
            ("id", "desc", "nulls-first"),
        ],
        options={"rewrite-all": True, "min-input-files": 2},
    )
    shuffled_table.refresh()
    # bucket is the primary key — assert non-decreasing across each file, and id
    # within each bucket region is non-increasing.
    import pyarrow.parquet as pq

    for path in _output_data_paths(shuffled_table):
        tbl = pq.read_table(path)
        buckets = tbl.column("bucket").to_pylist()
        ids = tbl.column("id").to_pylist()
        assert all(
            a <= b for a, b in zip(buckets, buckets[1:])
        ), f"bucket not asc in {path}"
        for b_val in set(buckets):
            ids_in_bucket = [ids[i] for i, bv in enumerate(buckets) if bv == b_val]
            assert all(
                a >= b for a, b in zip(ids_in_bucket, ids_in_bucket[1:])
            ), f"id not desc within bucket={b_val} in {path}"
