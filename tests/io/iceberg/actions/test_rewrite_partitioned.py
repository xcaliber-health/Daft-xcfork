"""Partition-tuple correctness for rewrite_data_files across Iceberg transforms."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.expressions import EqualTo
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import (
    BucketTransform,
    DayTransform,
    IdentityTransform,
    TruncateTransform,
)
from pyiceberg.types import (
    LongType,
    NestedField,
    StringType,
    TimestampType,
)

from daft.catalog import Table


@pytest.fixture(scope="function")
def catalog(tmp_path):
    cat = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path}/c.db",
        warehouse=f"file://{tmp_path}",
    )
    cat.create_namespace("default")
    yield cat
    cat.engine.dispose()


def _scan_data_files(table) -> list[Any]:
    return [t.file for t in table.scan().plan_files()]


def _partition_tuples(table) -> list[tuple]:
    out = []
    for f in _scan_data_files(table):
        rec = f.partition
        # Iceberg Record is positional; flatten via tuple(rec) when available
        try:
            out.append(tuple(rec))
        except TypeError:
            out.append(rec)
    return out


def _read_ids(table) -> list[int]:
    return sorted(int(r["id"]) for r in table.scan().to_arrow().to_pylist())


def test_identity_partition_each_output_has_correct_region(catalog):
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "region", StringType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(
            source_id=2, field_id=1000, transform=IdentityTransform(), name="region"
        )
    )
    table = catalog.create_table(
        "default.t_identity", schema=schema, partition_spec=spec
    )
    # Append 4 small files into each of 3 regions.
    for region in ("us", "eu", "ap"):
        for k in range(4):
            ids = list(range(k * 10, k * 10 + 5))
            table.append(
                pa.table(
                    {
                        "id": pa.array(ids, type=pa.int64()),
                        "region": pa.array([region] * 5, type=pa.string()),
                    }
                )
            )
    pre_ids = _read_ids(table)

    dt_table = Table.from_iceberg(table)
    result = dt_table.compact_files(
        options={"rewrite-all": True, "min-input-files": 2}
    )
    table.refresh()

    # Each output must carry exactly one non-empty partition tuple equal to its region.
    parts = _partition_tuples(table)
    assert parts, "expected at least one output file"
    assert all(len(p) == 1 for p in parts), f"expected 1-tuple partitions, got {parts}"
    regions = {p[0] for p in parts}
    assert regions == {"us", "eu", "ap"}, f"missing/extra regions: {regions}"

    # Partition pruning must still return correct rows.
    us_rows = list(
        table.scan(row_filter=EqualTo("region", "us")).to_arrow().to_pylist()
    )
    assert us_rows, "partition filter returned no rows after rewrite"
    assert all(r["region"] == "us" for r in us_rows)

    # Full scan preserves data.
    assert _read_ids(table) == pre_ids
    assert result.rewritten_files > 0
    assert result.added_files > 0


def test_bucket_partition_each_output_tagged_with_bucket(catalog):
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "label", StringType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(
            source_id=1,
            field_id=1000,
            transform=BucketTransform(num_buckets=4),
            name="id_bucket",
        )
    )
    table = catalog.create_table(
        "default.t_bucket", schema=schema, partition_spec=spec
    )
    # Seed enough distinct ids to hit multiple buckets.
    for chunk in range(6):
        rows = list(range(chunk * 10, chunk * 10 + 10))
        table.append(
            pa.table(
                {
                    "id": pa.array(rows, type=pa.int64()),
                    "label": pa.array([f"r{i}" for i in rows], type=pa.string()),
                }
            )
        )
    pre_ids = _read_ids(table)

    dt_table = Table.from_iceberg(table)
    result = dt_table.compact_files(
        options={"rewrite-all": True, "min-input-files": 2}
    )
    table.refresh()

    parts = _partition_tuples(table)
    assert parts, "expected at least one output file"
    # All bucket values must be ints in [0, num_buckets).
    bucket_values = [p[0] for p in parts]
    assert all(
        isinstance(b, int) and 0 <= b < 4 for b in bucket_values
    ), f"bucket values out of range: {bucket_values}"
    # Multiple buckets exercised.
    assert len(set(bucket_values)) >= 2

    # Data integrity.
    assert _read_ids(table) == pre_ids
    assert result.added_files > 0


def test_truncate_partition_each_output_tagged_with_truncated_prefix(catalog):
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "name", StringType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(
            source_id=2,
            field_id=1000,
            transform=TruncateTransform(width=3),
            name="name_trunc",
        )
    )
    table = catalog.create_table(
        "default.t_trunc", schema=schema, partition_spec=spec
    )
    for prefix in ("alpha", "beta", "gamma"):
        for k in range(3):
            ids = list(range(k * 10, k * 10 + 4))
            table.append(
                pa.table(
                    {
                        "id": pa.array(ids, type=pa.int64()),
                        "name": pa.array(
                            [f"{prefix}-{i}" for i in ids], type=pa.string()
                        ),
                    }
                )
            )

    dt_table = Table.from_iceberg(table)
    dt_table.compact_files(options={"rewrite-all": True, "min-input-files": 2})
    table.refresh()

    parts = _partition_tuples(table)
    assert parts
    trunc_values = {p[0] for p in parts}
    assert trunc_values == {"alp", "bet", "gam"}, f"got: {trunc_values}"


def test_day_partition_each_output_tagged_with_day_int(catalog):
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "event_ts", TimestampType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(
            source_id=2,
            field_id=1000,
            transform=DayTransform(),
            name="event_day",
        )
    )
    table = catalog.create_table("default.t_day", schema=schema, partition_spec=spec)
    days = [
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 2),
        dt.datetime(2024, 1, 3),
    ]
    for d in days:
        for k in range(3):
            ids = list(range(k * 10, k * 10 + 4))
            table.append(
                pa.table(
                    {
                        "id": pa.array(ids, type=pa.int64()),
                        "event_ts": pa.array(
                            [d] * 4, type=pa.timestamp("us")
                        ),
                    }
                )
            )

    dt_table = Table.from_iceberg(table)
    dt_table.compact_files(options={"rewrite-all": True, "min-input-files": 2})
    table.refresh()

    parts = _partition_tuples(table)
    assert parts
    # day transform yields integer days-since-epoch.
    day_values = {p[0] for p in parts}
    epoch = dt.date(1970, 1, 1)
    expected = {(d.date() - epoch).days for d in days}
    assert day_values == expected, f"got {day_values}, expected {expected}"
