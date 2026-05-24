"""End-to-end tests for IcebergTable.rewrite_manifests()."""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

from daft.catalog import Table
from daft.io.iceberg import RewriteManifestsResult

from tests.io.iceberg.actions._helpers import read_ids as _read_ids


def _current_manifest_count(table) -> int:
    snap = table.metadata.current_snapshot()
    if snap is None:
        return 0
    return len(list(snap.manifests(table.io)))


def _small_target_opts():
    return {"manifest-target-size-bytes": 64 * 1024 * 1024, "manifest-min-count-to-merge": 2}


def test_many_small_manifests_merge_into_one(make_tiny_table):
    table = make_tiny_table(
        name="default.t_rm_merge", n_files=8, rows_per_file=2
    )
    assert _current_manifest_count(table) == 8
    pre_ids = _read_ids(table)

    dt = Table.from_iceberg(table)
    result = dt.rewrite_manifests(options=_small_target_opts())

    assert isinstance(result, RewriteManifestsResult)
    assert result.rewritten_manifests_count == 8
    assert result.added_manifests_count == 1
    assert result.snapshot_id is not None
    assert result.bytes_rewritten > 0
    assert result.bytes_added > 0
    assert result.bytes_added < result.bytes_rewritten

    table.refresh()
    assert _current_manifest_count(table) == 1
    assert _read_ids(table) == pre_ids


def test_data_integrity_preserved(make_tiny_table):
    table = make_tiny_table(name="default.t_rm_data", n_files=5, rows_per_file=4)
    pre_ids = _read_ids(table)

    dt = Table.from_iceberg(table)
    dt.rewrite_manifests(options=_small_target_opts())

    table.refresh()
    assert _read_ids(table) == pre_ids


def test_idempotent_replay(make_tiny_table):
    table = make_tiny_table(name="default.t_rm_idemp", n_files=6, rows_per_file=2)
    dt = Table.from_iceberg(table)
    r1 = dt.rewrite_manifests(options=_small_target_opts())
    assert r1.snapshot_id is not None
    table.refresh()

    # Manually replay: pretend nothing was committed in between by reconstructing
    # the same hash. The second call should hit the snapshot-summary cache.
    r2 = dt.rewrite_manifests(options=_small_target_opts())
    assert r2.snapshot_id is None or r2.snapshot_id == r1.snapshot_id


def test_already_balanced_is_no_op(make_tiny_table):
    table = make_tiny_table(name="default.t_rm_noop", n_files=8, rows_per_file=2)
    dt = Table.from_iceberg(table)
    # First call collapses 8 → 1.
    dt.rewrite_manifests(options=_small_target_opts())
    table.refresh()
    snap_id_after_first = table.metadata.current_snapshot().snapshot_id
    assert _current_manifest_count(table) == 1

    # Second call should detect the table is already balanced and not commit.
    r = dt.rewrite_manifests(options=_small_target_opts())
    table.refresh()
    assert table.metadata.current_snapshot().snapshot_id == snap_id_after_first
    assert r.snapshot_id is None or r.snapshot_id == snap_id_after_first


def test_invalid_spec_id_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_rm_spec", n_files=2, rows_per_file=2)
    dt = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="spec_id="):
        dt.rewrite_manifests(spec_id=42)


def test_invalid_branch_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_rm_branch_bad", n_files=2, rows_per_file=2)
    dt = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="does not exist"):
        dt.rewrite_manifests(branch="nonexistent")


def test_gc_enabled_false_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_rm_gc", n_files=2, rows_per_file=2)
    with table.transaction() as tx:
        tx.set_properties(**{"gc.enabled": "false"})
    table.refresh()
    dt = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="gc.enabled=false"):
        dt.rewrite_manifests(options=_small_target_opts())


def test_invalid_target_size_rejected(make_tiny_table):
    table = make_tiny_table(name="default.t_rm_bad_size", n_files=2, rows_per_file=2)
    dt = Table.from_iceberg(table)
    with pytest.raises(ValueError, match="manifest-target-size-bytes"):
        dt.rewrite_manifests(options={"manifest-target-size-bytes": 0})


def test_branch_scoping_does_not_touch_main(local_catalog, simple_schema):
    table = local_catalog.create_table(
        "default.t_rm_branch_scope",
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    for k in range(4):
        table.append(
            pa.table(
                {
                    "id": pa.array([k], type=pa.int64()),
                    "label": pa.array([f"r{k}"], type=pa.string()),
                }
            )
        )
    table.refresh()
    main_snapshot_id = table.metadata.current_snapshot().snapshot_id

    table.manage_snapshots().create_branch(
        snapshot_id=main_snapshot_id, branch_name="dev"
    ).commit()
    table.refresh()

    # Add more snapshots on `dev` only so it has its own manifest chain.
    for k in range(4, 8):
        table.append(
            pa.table(
                {
                    "id": pa.array([k], type=pa.int64()),
                    "label": pa.array([f"r{k}"], type=pa.string()),
                }
            ),
            branch="dev",
        )
    table.refresh()

    main_manifest_paths_before = {
        m.manifest_path
        for m in table.snapshot_by_name("main").manifests(table.io)
    }

    dt = Table.from_iceberg(table)
    dt.rewrite_manifests(branch="dev", options=_small_target_opts())

    table.refresh()
    main_manifest_paths_after = {
        m.manifest_path
        for m in table.snapshot_by_name("main").manifests(table.io)
    }
    assert main_manifest_paths_after == main_manifest_paths_before


def test_snapshot_summary_carries_daft_props(make_tiny_table):
    table = make_tiny_table(name="default.t_rm_summary", n_files=6, rows_per_file=2)
    dt = Table.from_iceberg(table)
    result = dt.rewrite_manifests(options=_small_target_opts())
    table.refresh()
    snap = table.metadata.current_snapshot()
    props = snap.summary.additional_properties
    assert props.get("daft.rewrite-id") == result.rewrite_id
    assert props.get("daft.rewrite-strategy") == "manifests"
    assert props.get("daft.input-manifests") == "6"
    assert props.get("daft.output-manifests") == str(result.added_manifests_count)
    # Data totals carried over from parent snapshot.
    assert int(props.get("total-data-files", 0)) == 6
