"""Concurrent file-group rewrite respects max-concurrent-file-group-rewrites."""

from __future__ import annotations

import threading
import time

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import LongType, NestedField, StringType

from daft.catalog import Table
from daft.io.iceberg import _compact


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


def test_max_concurrent_drives_thread_pool(local_catalog, monkeypatch):
    table = _make_partitioned(local_catalog, "default.t_conc_par")

    active = {"current": 0, "peak": 0}
    lock = threading.Lock()
    real_rewrite = _compact._rewrite_group

    def slow_rewrite(*args, **kwargs):
        with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        try:
            time.sleep(0.15)
            return real_rewrite(*args, **kwargs)
        finally:
            with lock:
                active["current"] -= 1

    monkeypatch.setattr(_compact, "_rewrite_group", slow_rewrite)

    dt = Table.from_iceberg(table)
    dt.compact_files(
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "max-concurrent-file-group-rewrites": 4,
        }
    )
    assert active["peak"] >= 2, f"expected >= 2 concurrent groups, peak={active['peak']}"


def test_max_concurrent_one_is_sequential(local_catalog, monkeypatch):
    table = _make_partitioned(local_catalog, "default.t_conc_seq")

    active = {"current": 0, "peak": 0}
    lock = threading.Lock()
    real_rewrite = _compact._rewrite_group

    def slow_rewrite(*args, **kwargs):
        with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        try:
            time.sleep(0.05)
            return real_rewrite(*args, **kwargs)
        finally:
            with lock:
                active["current"] -= 1

    monkeypatch.setattr(_compact, "_rewrite_group", slow_rewrite)

    dt = Table.from_iceberg(table)
    dt.compact_files(
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "max-concurrent-file-group-rewrites": 1,
        }
    )
    assert active["peak"] == 1, f"expected sequential execution, peak={active['peak']}"
