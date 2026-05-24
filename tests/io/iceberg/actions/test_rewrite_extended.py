"""Coverage for max-files-to-rewrite, min/max file size, and partial-progress failure paths."""

from __future__ import annotations

import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.exceptions import CommitFailedException

from daft.catalog import Table
from daft.io.iceberg import RewriteFailedException


def _file_count(table) -> int:
    return len(list(table.scan().plan_files()))


def _make_partitioned(local_catalog, name: str, n_partitions: int = 4, files_per_part: int = 2):
    import pyarrow as pa
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import LongType, NestedField, StringType

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
    for r in range(n_partitions):
        region = f"r{r}"
        for k in range(files_per_part):
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


def test_max_files_to_rewrite_caps_planner(local_catalog):
    # Force one group per partition by partitioning; 4 partitions × 2 files = 8 files.
    table = _make_partitioned(local_catalog, "default.t_cap", n_partitions=4, files_per_part=2)
    assert _file_count(table) == 8

    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "max-files-to-rewrite": 4,
        }
    )
    # Cap honored: at most 4 input files removed in this rewrite.
    assert result.rewritten_files <= 4


def test_min_file_size_makes_inputs_well_sized(make_tiny_table):
    # If min-file-size-bytes is below every input's size, no file is undersized
    # so the planner has no work.
    table = make_tiny_table(name="default.t_minsize", n_files=8, rows_per_file=3)
    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={
            "min-input-files": 2,
            # Inputs are tiny (<1KB); set lower=1 so they're "well-sized".
            "min-file-size-bytes": 1,
            "max-file-size-bytes": 5 * 1024 * 1024 * 1024,
        }
    )
    assert result.rewritten_files == 0
    assert result.added_files == 0


def test_max_failed_commits_raises_when_exceeded(local_catalog, monkeypatch):
    table = _make_partitioned(
        local_catalog, "default.t_maxfail", n_partitions=4, files_per_part=2
    )

    txn_type = type(table.transaction())
    calls = {"n": 0}

    def always_fail(self):
        calls["n"] += 1
        raise CommitFailedException(f"forced fail #{calls['n']}")

    monkeypatch.setattr(txn_type, "commit_transaction", always_fail)

    dt = Table.from_iceberg(table)
    with pytest.raises(RewriteFailedException) as exc_info:
        dt.compact_files(
            options={
                "rewrite-all": True,
                "min-input-files": 2,
                "partial-progress.enabled": True,
                "partial-progress.max-commits": 4,
                "partial-progress.max-failed-commits": 1,
            }
        )
    assert "max-failed-commits" in str(exc_info.value)


def test_failed_data_files_counted_in_result(local_catalog, monkeypatch):
    table = _make_partitioned(
        local_catalog, "default.t_failfiles", n_partitions=4, files_per_part=2
    )

    txn_type = type(table.transaction())
    real_commit_transaction = txn_type.commit_transaction
    state = {"calls": 0}

    def selective_fail(self):
        state["calls"] += 1
        # Fail the first batch entirely (num-retries+1 = 5 attempts), allow the second through.
        if state["calls"] <= 5:
            raise CommitFailedException("forced first-batch fail")
        return real_commit_transaction(self)

    monkeypatch.setattr(txn_type, "commit_transaction", selective_fail)

    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "partial-progress.enabled": True,
            "partial-progress.max-commits": 2,
            "partial-progress.max-failed-commits": 2,
        }
    )
    assert result.failed_groups >= 1
    assert result.failed_data_files >= result.failed_groups
