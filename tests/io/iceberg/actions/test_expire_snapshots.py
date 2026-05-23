"""End-to-end tests for IcebergTable.expire_snapshots()."""

from __future__ import annotations

import os
import threading
import time

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.exceptions import CommitFailedException

from daft.catalog import Table
from daft.io.iceberg._expire import ExpireResult, _resolve_expired_ids


def _snapshot_count(table) -> int:
    return len(table.metadata.snapshots)


def _snapshot_ids(table) -> list[int]:
    return [s.snapshot_id for s in table.metadata.snapshots]


def _data_file_paths(table) -> set[str]:
    return {t.file.file_path for t in table.scan().plan_files()}


def _strip_scheme(p: str) -> str:
    return p[len("file://") :] if p.startswith("file://") else p


def _all_files_exist(paths) -> bool:
    return all(os.path.exists(_strip_scheme(p)) for p in paths)


def _any_file_missing(paths) -> bool:
    return any(not os.path.exists(_strip_scheme(p)) for p in paths)


def test_older_than_removes_old_keeps_new(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_older", n_files=5, rows_per_file=3)
    snaps = list(table.metadata.snapshots)
    assert len(snaps) == 5
    # Cut between snapshot 2 (idx 1) and snapshot 3 (idx 2).
    cutoff_ms = (snaps[1].timestamp_ms + snaps[2].timestamp_ms) // 2

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(older_than=cutoff_ms)

    table.refresh()
    remaining = _snapshot_ids(table)
    assert snaps[0].snapshot_id not in remaining
    assert snaps[1].snapshot_id not in remaining
    assert snaps[2].snapshot_id in remaining
    assert snaps[-1].snapshot_id in remaining
    assert isinstance(result, ExpireResult)


def test_retain_last_keeps_n_most_recent(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_retain", n_files=6, rows_per_file=3)
    pre_ids = _snapshot_ids(table)
    assert len(pre_ids) == 6

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(retain_last=3)

    table.refresh()
    post_ids = _snapshot_ids(table)
    assert len(post_ids) == 3
    # The three retained must be the three most-recent.
    assert set(post_ids) == set(pre_ids[-3:])


def test_explicit_snapshot_ids_bypass_retention(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_ids", n_files=5, rows_per_file=3)
    snaps = list(table.metadata.snapshots)
    target = snaps[1].snapshot_id  # not protected (not current)

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(snapshot_ids=[target])

    table.refresh()
    assert target not in _snapshot_ids(table)


def test_branch_head_protected(local_catalog, simple_schema):
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table = local_catalog.create_table(
        "default.t_exp_branch",
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    for k in range(4):
        table.append(
            pa.table(
                {
                    "id": pa.array(list(range(k * 5, k * 5 + 5)), type=pa.int64()),
                    "label": pa.array([f"r{i}" for i in range(k * 5, k * 5 + 5)]),
                }
            )
        )
    snap0 = table.metadata.snapshots[0].snapshot_id

    with table.manage_snapshots() as ms:
        ms.create_branch(snapshot_id=snap0, branch_name="protect_me")

    # Try to expire everything: snapshot 0 must survive because the branch refs it.
    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(older_than=int(time.time() * 1000) + 60_000)

    table.refresh()
    surviving = set(_snapshot_ids(table))
    assert snap0 in surviving


def test_explicit_protected_id_rejected(local_catalog, simple_schema):
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table = local_catalog.create_table(
        "default.t_exp_reject",
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    table.append(
        pa.table({"id": pa.array([1], type=pa.int64()), "label": pa.array(["a"])})
    )
    snap = table.current_snapshot().snapshot_id

    with table.manage_snapshots() as ms:
        ms.create_branch(snapshot_id=snap, branch_name="prot")

    dt_table = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="protected"):
        dt_table.expire_snapshots(snapshot_ids=[snap])


def test_min_snapshots_to_keep_floors_retain_last(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_floor", n_files=6, rows_per_file=2)
    with table.transaction() as tx:
        tx.set_properties(**{"history.expire.min-snapshots-to-keep": "5"})
    table.refresh()

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(retain_last=2)

    table.refresh()
    # Floor wins: 5 (not 2) snapshots must remain.
    assert len(table.metadata.snapshots) >= 5


def test_gc_enabled_false_refuses(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_gc", n_files=3, rows_per_file=2)
    with table.transaction() as tx:
        tx.set_properties(**{"gc.enabled": "false"})

    dt_table = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="gc.enabled"):
        dt_table.expire_snapshots(retain_last=1)


def test_clean_expired_files_false_is_metadata_only(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_metaonly", n_files=4, rows_per_file=3)
    # Capture all data file paths that exist before expire.
    snaps = list(table.metadata.snapshots)
    all_paths: set[str] = set()
    for s in snaps:
        for m in s.manifests(table.io):
            for e in m.fetch_manifest_entry(table.io, discard_deleted=False):
                all_paths.add(e.data_file.file_path)

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(retain_last=1, clean_expired_files=False)

    table.refresh()
    assert len(table.metadata.snapshots) == 1
    assert result == ExpireResult()  # all-zero counts
    # No files removed from disk.
    assert _all_files_exist(all_paths)


def test_result_counts_match_actual_deletions(make_tiny_table):
    # Append-only tables don't orphan data files (later snapshots still reach them via
    # inherited manifests), but each expired snapshot orphans its own manifest list.
    table = make_tiny_table(name="default.t_exp_counts", n_files=4, rows_per_file=3)

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(retain_last=1)

    table.refresh()
    surviving = _data_file_paths(table)
    assert _all_files_exist(surviving)
    # 4 appends → 1 retained → 3 expired snapshots → 3 manifest-list orphans.
    assert result.deleted_manifest_lists_count == 3
    # No data files removed because nothing was overwritten or deleted.
    assert result.deleted_data_files_count == 0


def test_result_counts_after_overwrite(make_tiny_table):
    """Overwrite produces data-file orphans once the prior snapshots are expired."""
    table = make_tiny_table(name="default.t_exp_overwrite", n_files=3, rows_per_file=3)
    overwritten = _data_file_paths(table)
    table.overwrite(
        pa.table(
            {
                "id": pa.array([999], type=pa.int64()),
                "label": pa.array(["final"]),
            }
        )
    )
    table.refresh()

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(retain_last=1)

    table.refresh()
    surviving = _data_file_paths(table)
    # Surviving file is the post-overwrite one.
    assert _all_files_exist(surviving)
    # Overwritten files have been physically deleted.
    assert _any_file_missing(overwritten)
    assert result.deleted_data_files_count >= len(overwritten)


def test_idempotent_rerun_is_noop(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_idem", n_files=4, rows_per_file=3)
    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(retain_last=1)

    table.refresh()
    result2 = dt_table.expire_snapshots(retain_last=1)
    assert result2 == ExpireResult()


def test_parallel_delete_observable(make_tiny_table, monkeypatch):
    table = make_tiny_table(name="default.t_exp_parallel", n_files=6, rows_per_file=3)
    active = {"current": 0, "peak": 0}
    lock = threading.Lock()
    real_delete = type(table.io).delete

    def slow_delete(self, location):
        with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        try:
            time.sleep(0.1)
            return real_delete(self, location)
        finally:
            with lock:
                active["current"] -= 1

    monkeypatch.setattr(type(table.io), "delete", slow_delete)

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(
        retain_last=1,
        options={"max-concurrent-deletes": 4},
    )
    assert (
        active["peak"] >= 2
    ), f"expected >= 2 concurrent deletes, peak={active['peak']}"


def test_notfound_during_delete_is_success(make_tiny_table, monkeypatch):
    table = make_tiny_table(name="default.t_exp_nf", n_files=4, rows_per_file=3)
    real_delete = type(table.io).delete
    state = {"fail_path": None}

    def flaky(self, location):
        path_str = (
            location if isinstance(location, str) else getattr(location, "location", "")
        )
        if state["fail_path"] is None:
            state["fail_path"] = path_str
            raise FileNotFoundError(path_str)
        return real_delete(self, location)

    monkeypatch.setattr(type(table.io), "delete", flaky)

    dt_table = Table.from_iceberg(table)
    result = dt_table.expire_snapshots(retain_last=1)
    # FileNotFoundError on the first delete was suppressed; rest proceeded.
    total = (
        result.deleted_data_files_count
        + result.deleted_manifest_files_count
        + result.deleted_manifest_lists_count
    )
    assert total >= 1


def test_commit_conflict_retries(make_tiny_table, monkeypatch):
    table = make_tiny_table(name="default.t_exp_commit", n_files=4, rows_per_file=3)

    # Reach into pyiceberg's ExpireSnapshots.commit and fail the first call only.
    from pyiceberg.table.update.snapshot import ExpireSnapshots as _PyExpireSnapshots

    real_commit = _PyExpireSnapshots.commit
    calls = {"n": 0}

    def flaky_commit(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CommitFailedException("transient")
        return real_commit(self)

    monkeypatch.setattr(_PyExpireSnapshots, "commit", flaky_commit)

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots(retain_last=1)

    table.refresh()
    assert calls["n"] >= 2
    assert len(table.metadata.snapshots) == 1


def test_retain_last_below_one_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_bad", n_files=2, rows_per_file=2)
    dt_table = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="retain_last"):
        dt_table.expire_snapshots(retain_last=0)


def test_no_args_falls_back_to_max_age_property(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_noargs", n_files=4, rows_per_file=2)
    # Configure a very short max-age so every snapshot is older than the threshold.
    with table.transaction() as tx:
        tx.set_properties(**{"history.expire.max-snapshot-age-ms": "1"})
    table.refresh()

    dt_table = Table.from_iceberg(table)
    dt_table.expire_snapshots()

    table.refresh()
    # The current snapshot is protected by the main ref; everything else expired.
    assert len(table.metadata.snapshots) == 1


def test_resolve_unit_combination_of_knobs(make_tiny_table):
    table = make_tiny_table(name="default.t_exp_resolve", n_files=5, rows_per_file=2)
    snaps = list(table.metadata.snapshots)
    older_than = snaps[2].timestamp_ms  # expires 0 and 1
    explicit = [snaps[3].snapshot_id]  # extra
    expired = _resolve_expired_ids(
        table=table,
        older_than=older_than,
        retain_last=None,
        snapshot_ids=explicit,
        protected_ids={snaps[-1].snapshot_id},
    )
    assert snaps[0].snapshot_id in expired
    assert snaps[1].snapshot_id in expired
    assert snaps[3].snapshot_id in expired
    assert snaps[-1].snapshot_id not in expired  # protected
