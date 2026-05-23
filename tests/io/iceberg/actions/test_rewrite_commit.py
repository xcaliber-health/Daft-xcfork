"""Commit-hardening tests for rewrite_data_files: OCC retry, partial-progress, idempotency."""

from __future__ import annotations

import logging
import random
import string

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.exceptions import CommitFailedException
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.schema import Schema
from pyiceberg.table import Transaction
from pyiceberg.types import LongType, NestedField, StringType

from daft.catalog import Table
from daft.io.iceberg._compact import RewriteConflict


def _random_strings(n: int, width: int, rng: random.Random) -> list[str]:
    """Generate `n` random length-`width` ASCII strings (poorly compressible).

    Used to inflate parquet file size past the bin-packer's group cap without needing
    a partitioned table (which the current writer doesn't yet support).
    """
    alphabet = string.ascii_letters + string.digits
    return ["".join(rng.choices(alphabet, k=width)) for _ in range(n)]


def _make_multifile_table(
    local_catalog, name: str, n_files: int = 4, rows_per_file: int = 25_000
):
    """Create an unpartitioned table with `n_files` parquet files each ~700KB.

    With target/max-group set to 1 MiB (the validator minimum), two of these files
    can't share a bin-pack group, so the planner emits one group per file — giving
    exactly `n_files` groups for partial-progress to slice.
    """
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "blob", StringType(), required=False),
    )
    table = local_catalog.create_table(
        identifier=name,
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    rng = random.Random(17)
    for f in range(n_files):
        ids = list(range(f * rows_per_file, (f + 1) * rows_per_file))
        blobs = _random_strings(rows_per_file, 32, rng)
        table.append(
            pa.table(
                {
                    "id": pa.array(ids, type=pa.int64()),
                    "blob": pa.array(blobs, type=pa.string()),
                }
            )
        )
    return table


_MULTI_OPTS = {
    "rewrite-all": True,
    "min-input-files": 2,
    "target-file-size-bytes": 1024 * 1024,
    "max-file-group-size-bytes": 1024 * 1024,
}


def _snapshot_count(table) -> int:
    table.refresh()
    return len(list(table.metadata.snapshots or []))


def _patch_commit(monkeypatch, behavior):
    """Wrap `Transaction.commit_transaction` so `behavior(call_index, real_commit, self)` runs each time."""
    real_commit = Transaction.commit_transaction
    counter = {"n": 0}

    def wrapper(self):
        counter["n"] += 1
        return behavior(counter["n"], real_commit, self)

    monkeypatch.setattr(Transaction, "commit_transaction", wrapper)
    return counter


def test_occ_retry_succeeds_on_transient_conflict(make_tiny_table, monkeypatch):
    table = make_tiny_table(name="default.t_occ_ok", n_files=6, rows_per_file=3)
    pre_snaps = _snapshot_count(table)

    def behavior(n, real, tx):
        if n == 1:
            raise CommitFailedException("simulated transient conflict")
        return real(tx)

    counter = _patch_commit(monkeypatch, behavior)

    dt = Table.from_iceberg(table)
    result = dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})

    assert counter["n"] == 2, "expected one failure + one success"
    assert result.commits == 1
    assert result.rewritten_files == 6
    assert _snapshot_count(table) == pre_snaps + 1


def test_occ_retry_exhausts_and_raises(make_tiny_table, monkeypatch):
    from daft.io.iceberg._compact import RewriteFailedException

    table = make_tiny_table(name="default.t_occ_exhaust", n_files=6, rows_per_file=3)
    pre_snaps = _snapshot_count(table)

    def behavior(n, real, tx):
        raise CommitFailedException(f"always fails (attempt {n})")

    counter = _patch_commit(monkeypatch, behavior)

    dt = Table.from_iceberg(table)
    with pytest.raises(RewriteFailedException) as exc_info:
        dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})

    assert "partial-progress" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, CommitFailedException)
    assert counter["n"] == 4, "expected exactly _COMMIT_MAX_ATTEMPTS attempts"
    assert _snapshot_count(table) == pre_snaps, "no snapshot must land on full failure"


def test_occ_aborts_when_input_files_vanish(make_tiny_table, monkeypatch):
    table = make_tiny_table(name="default.t_occ_conflict", n_files=4, rows_per_file=3)
    table_type = type(table)
    original_scan = table_type.scan

    # After the first commit attempt fails, hide one input file from subsequent
    # `table.scan().plan_files()` calls — simulating a concurrent rewrite that
    # already consumed it.
    state = {"failed_once": False, "vanished_path": None}

    def behavior(n, real, tx):
        if n == 1:
            # Capture a path we want to "vanish" from the table.
            files = [t.file.file_path for t in original_scan(table).plan_files()]
            state["vanished_path"] = files[0]
            state["failed_once"] = True
            raise CommitFailedException("first attempt")
        return real(tx)

    _patch_commit(monkeypatch, behavior)

    def filtering_scan(self, **kwargs):
        scan = original_scan(self, **kwargs)
        if state["failed_once"]:
            real_plan = scan.plan_files
            vanished = state["vanished_path"]

            def filtered():
                return [t for t in real_plan() if t.file.file_path != vanished]

            scan.plan_files = filtered  # type: ignore[method-assign]
        return scan

    monkeypatch.setattr(table_type, "scan", filtering_scan)

    dt = Table.from_iceberg(table)
    with pytest.raises(RewriteConflict, match="input files vanished"):
        dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})


def test_partial_progress_creates_n_commits(local_catalog):
    # 4 large files → 4 singleton planner groups → 2 batches of 2 groups (max-commits=2).
    table = _make_multifile_table(local_catalog, "default.t_pp_n", n_files=4)
    pre_snaps = _snapshot_count(table)

    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={
            **_MULTI_OPTS,
            "partial-progress.enabled": True,
            "partial-progress.max-commits": 2,
        }
    )

    assert result.commits == 2
    assert _snapshot_count(table) == pre_snaps + 2
    assert len(result.snapshot_ids) == 2
    assert result.failed_groups == 0


def test_partial_progress_failed_batch_aggregated(local_catalog, monkeypatch, caplog):
    table = _make_multifile_table(local_catalog, "default.t_pp_fail", n_files=4)
    pre_snaps = _snapshot_count(table)

    def behavior(n, real, tx):
        # First batch (call 1) succeeds. Second batch (calls 2..5) all raise → exhaust.
        if n == 1:
            return real(tx)
        raise CommitFailedException(f"batch-2 always fails (call {n})")

    _patch_commit(monkeypatch, behavior)

    dt = Table.from_iceberg(table)
    with caplog.at_level(logging.WARNING, logger="daft.io.iceberg._compact"):
        result = dt.compact_files(
            options={
                **_MULTI_OPTS,
                "partial-progress.enabled": True,
                "partial-progress.max-commits": 2,
            }
        )

    assert result.commits == 1, "only the first batch should have committed"
    assert result.failed_groups > 0
    assert _snapshot_count(table) == pre_snaps + 1
    assert any(
        "orphan outputs" in rec.message for rec in caplog.records
    ), f"expected an orphan-outputs WARNING; got: {[r.message for r in caplog.records]}"


def test_idempotent_replay_across_partial_progress(local_catalog):
    table = _make_multifile_table(local_catalog, "default.t_pp_idemp", n_files=4)
    rid = "pp-replay-me"
    common_opts = {
        **_MULTI_OPTS,
        "partial-progress.enabled": True,
        "partial-progress.max-commits": 2,
        "rewrite-id": rid,
    }

    dt = Table.from_iceberg(table)
    r1 = dt.compact_files(options=common_opts)
    snaps_after_first = _snapshot_count(table)
    assert r1.commits == 2

    r2 = dt.compact_files(options=common_opts)
    assert (
        _snapshot_count(table) == snaps_after_first
    ), "replay must not create new snapshots"
    assert r2.rewrite_id == rid
    assert r2.commits == r1.commits == 2
    assert sorted(r2.snapshot_ids) == sorted(r1.snapshot_ids)
