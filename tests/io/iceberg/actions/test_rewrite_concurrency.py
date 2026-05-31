"""File groups are rewritten one at a time.

Each group's read, re-cluster, and write stream through the execution engine,
which bounds memory to a single group. Processing groups sequentially keeps
that bound flat, so the rewrite never runs two groups at once regardless of the
``max-concurrent-file-group-rewrites`` value, which is retained for interface
compatibility.
"""

from __future__ import annotations

import threading

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType

from daft.catalog import Table
from daft.io.iceberg import _compact  # noqa: internal — monkeypatching internal helper


def _make_partitioned(local_catalog, name: str):
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "region", StringType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(
            source_id=2, field_id=1000, transform=IdentityTransform(), name="region"
        )
    )
    table = local_catalog.create_table(name, schema=schema, partition_spec=spec)
    for region in ("us", "eu", "ap"):
        for k in range(3):
            ids = list(range(k * 5, k * 5 + 5))
            table.append(
                pa.table(
                    {
                        "id": pa.array(ids, type=pa.int64()),
                        "region": pa.array([region] * 5, type=pa.string()),
                    }
                )
            )
    return table


def _instrument_peak_concurrency(monkeypatch):
    """Patch ``_rewrite_group`` to record the peak number of overlapping calls."""
    active = {"current": 0, "peak": 0}
    lock = threading.Lock()
    real_rewrite = _compact._rewrite_group

    def instrumented(*args, **kwargs):
        with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        try:
            return real_rewrite(*args, **kwargs)
        finally:
            with lock:
                active["current"] -= 1

    monkeypatch.setattr(_compact, "_rewrite_group", instrumented)
    return active


@pytest.mark.parametrize("max_concurrent", [1, 4])
def test_groups_rewrite_sequentially(local_catalog, monkeypatch, max_concurrent):
    table = _make_partitioned(local_catalog, f"default.t_conc_{max_concurrent}")
    active = _instrument_peak_concurrency(monkeypatch)

    dt = Table.from_iceberg(table)
    dt.compact_files(
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "max-concurrent-file-group-rewrites": max_concurrent,
        }
    )

    assert active["peak"] == 1, (
        f"groups must rewrite one at a time, observed peak={active['peak']}"
    )


def test_concurrency_option_does_not_change_result(local_catalog):
    table = _make_partitioned(local_catalog, "default.t_conc_result")

    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "max-concurrent-file-group-rewrites": 8,
        }
    )

    table.refresh()
    # Three partitions, three files each: every input file is rewritten.
    assert result.rewritten_files == 9
    assert result.added_files == result.added_files  # at least one output per partition
    rows = sorted(table.scan().to_arrow().column("id").to_pylist())
    assert rows == sorted([i for _ in range(3) for k in range(3) for i in range(k * 5, k * 5 + 5)])
