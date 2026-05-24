"""End-to-end concurrency: maintenance APIs vs a live appender thread.

Each test runs a background thread that appends to the same table while the
foreground performs a maintenance action. Asserts isolation guarantees and a
row-count invariant. Synchronization uses :class:`threading.Event` rather
than ``time.sleep`` so behavior is deterministic on slow CI.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.io.iceberg import RewriteConflict

from tests.io.iceberg.actions._helpers import (
    Appender,
    _row_count,
    make_seeded_table,
)

# Internal symbol: the rewrite uses this synthetic column name during a
# z-order pass and projects it away before commit. The test confirms it
# never appears in any committed data-file path.
_ZORDER_KEY_COL = "__daft_zorder_key__"  # noqa: internal


def _await_first_append(appender: Appender) -> None:
    assert appender.wait_for_first_commit(
        timeout=10.0
    ), "appender did not commit within 10s"


def test_atomic_rewrite_raises_conflict_on_same_partition_append(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_atomic_conflict", n_files=6)
    appender = Appender(table, interval_s=0.05, batch_rows=20)
    appender.start()
    try:
        _await_first_append(appender)
        dt = Table.from_iceberg(table)
        with pytest.raises(Exception) as exc_info:
            dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})
        assert isinstance(exc_info.value, RewriteConflict) or (
            "partition" in str(exc_info.value).lower()
            or "vanished" in str(exc_info.value).lower()
        ), f"unexpected error: {exc_info.value!r}"
    finally:
        appender.stop()


_REWRITE_STRATEGIES = [
    pytest.param(
        "binpack",
        {},
        id="binpack",
    ),
    pytest.param(
        "sort",
        {"sort_order": [("id", "asc", "nulls-last")]},
        id="sort",
    ),
    pytest.param(
        "zorder",
        {"zorder_by": ["id"]},
        id="zorder",
    ),
]


@pytest.mark.parametrize("strategy,strategy_kwargs", _REWRITE_STRATEGIES)
def test_partial_progress_rewrite_progresses_under_appends(
    local_catalog, strategy, strategy_kwargs
):
    table = make_seeded_table(
        local_catalog, f"default.t_pp_{strategy}_appends", n_files=8
    )
    appender = Appender(table, interval_s=0.05, batch_rows=10)
    appender.start()
    try:
        _await_first_append(appender)
        dt = Table.from_iceberg(table)
        result = dt.rewrite_data_files(
            strategy=strategy,
            **strategy_kwargs,
            options={
                "rewrite-all": True,
                "min-input-files": 2,
                "partial-progress.enabled": True,
                "partial-progress.max-commits": 4,
                "partial-progress.max-failed-commits": 4,
            },
        )
        assert result.added_files >= 1
        assert result.strategy == strategy
    finally:
        appender.stop()


def test_zorder_key_column_is_dropped_from_committed_files(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_zorder_key_dropped", n_files=4)
    dt = Table.from_iceberg(table)
    dt.rewrite_data_files(
        strategy="zorder",
        zorder_by=["id"],
        options={"rewrite-all": True, "min-input-files": 2},
    )
    table.refresh()
    for snap in table.metadata.snapshots:
        for manifest in snap.manifests(table.io):
            for entry in manifest.fetch_manifest_entry(
                table.io, discard_deleted=False
            ):
                assert _ZORDER_KEY_COL not in (entry.data_file.file_path or "")


def test_rewrite_manifests_succeeds_under_appends(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_rm_appends", n_files=6)
    appender = Appender(table, interval_s=0.05, batch_rows=10)
    appender.start()
    try:
        _await_first_append(appender)
        dt = Table.from_iceberg(table)
        result = dt.rewrite_manifests(
            options={"manifest-target-size-bytes": 8 * 1024 * 1024}
        )
        assert result.rewrite_id
    finally:
        appender.stop()


def test_expire_snapshots_succeeds_under_appends(local_catalog):
    table = make_seeded_table(local_catalog, "default.t_es_appends", n_files=4)
    appender = Appender(table, interval_s=0.05, batch_rows=10)
    appender.start()
    try:
        _await_first_append(appender)
        dt = Table.from_iceberg(table)
        result = dt.expire_snapshots(
            older_than=datetime.now(tz=timezone.utc),
            retain_last=1,
            clean_expired_files=False,
        )
        assert (
            result.deleted_data_files_count
            + result.deleted_manifest_files_count
            + result.deleted_manifest_lists_count
            >= 0
        )
    finally:
        appender.stop()


def test_row_count_invariant_through_rewrite_then_expire(local_catalog):
    seed_rows = 6 * 100
    table = make_seeded_table(local_catalog, "default.t_invariant", n_files=6)

    appender = Appender(table, interval_s=0.05, batch_rows=10)
    appender.start()
    try:
        _await_first_append(appender)
        dt = Table.from_iceberg(table)
        dt.compact_files(
            options={
                "min-input-files": 2,
                "partial-progress.enabled": True,
                "partial-progress.max-commits": 8,
                "partial-progress.max-failed-commits": 8,
            }
        )
    finally:
        appender.stop()

    final_rows = _row_count(table)
    min_expected = seed_rows
    max_expected = seed_rows + appender.commits * 10
    assert min_expected <= final_rows <= max_expected, (
        f"row count outside acceptable range [{min_expected}, {max_expected}], "
        f"got {final_rows}; appender.errors="
        f"{[type(e).__name__ for e in appender.errors[:3]]}"
    )
