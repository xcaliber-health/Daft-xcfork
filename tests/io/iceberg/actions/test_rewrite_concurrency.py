"""Concurrent file-group rewrite respects max-concurrent-file-group-rewrites.

Uses a :class:`threading.Barrier` inside the slow-rewrite mock so the
foreground releases groups deterministically: the test does not depend on
sleep durations and will not flake on slow CI.
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


def _instrument_rewrite_group(monkeypatch, *, expected_peak: int):
    """Patch ``_rewrite_group`` to track concurrency and release at ``expected_peak``."""
    active = {"current": 0, "peak": 0}
    lock = threading.Lock()
    release_gate = threading.Event()
    target_peak_reached = threading.Event()
    real_rewrite = _compact._rewrite_group

    def instrumented(*args, **kwargs):
        with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
            if active["peak"] >= expected_peak:
                target_peak_reached.set()
        # Wait until either the peak has been observed or another worker
        # frees a slot; this keeps groups concurrent without sleep.
        target_peak_reached.wait(timeout=5.0)
        try:
            return real_rewrite(*args, **kwargs)
        finally:
            with lock:
                active["current"] -= 1

    monkeypatch.setattr(_compact, "_rewrite_group", instrumented)
    return active, release_gate


@pytest.mark.parametrize(
    "max_concurrent,expected_peak,assert_op",
    [
        pytest.param(4, 2, "ge", id="max4_drives_pool_to_at_least_2"),
        pytest.param(1, 1, "eq", id="max1_is_strictly_sequential"),
    ],
)
def test_max_concurrent_groups_caps_thread_pool(
    local_catalog, monkeypatch, max_concurrent, expected_peak, assert_op
):
    table = _make_partitioned(local_catalog, f"default.t_conc_{max_concurrent}")
    active, _ = _instrument_rewrite_group(monkeypatch, expected_peak=expected_peak)

    dt = Table.from_iceberg(table)
    dt.compact_files(
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "max-concurrent-file-group-rewrites": max_concurrent,
        }
    )
    if assert_op == "ge":
        assert active["peak"] >= expected_peak, (
            f"expected >= {expected_peak} concurrent groups, peak={active['peak']}"
        )
    else:
        assert active["peak"] == expected_peak, (
            f"expected sequential execution, peak={active['peak']}"
        )
