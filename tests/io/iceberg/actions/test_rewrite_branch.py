"""Rewrite against a non-default branch ref commits to that branch only."""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table


def test_rewrite_on_branch_leaves_main_untouched(local_catalog, simple_schema):
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC

    table = local_catalog.create_table(
        "default.t_branch",
        schema=simple_schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    for k in range(6):
        table.append(
            pa.table(
                {
                    "id": pa.array(list(range(k * 5, k * 5 + 5)), type=pa.int64()),
                    "label": pa.array([f"r{i}" for i in range(k * 5, k * 5 + 5)]),
                }
            )
        )
    main_snap_before = table.current_snapshot().snapshot_id

    # Create a new branch pointing at the current snapshot.
    branch_name = "feature_x"
    with table.manage_snapshots() as ms:
        ms.create_branch(snapshot_id=main_snap_before, branch_name=branch_name)

    # Sanity: branch exists at the same snapshot.
    assert table.snapshot_by_name(branch_name).snapshot_id == main_snap_before

    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        branch=branch_name,
        options={"rewrite-all": True, "min-input-files": 2},
    )
    table.refresh()

    # Main is untouched.
    assert table.current_snapshot().snapshot_id == main_snap_before
    # Branch advanced to a new snapshot.
    new_branch_snap = table.snapshot_by_name(branch_name).snapshot_id
    assert new_branch_snap != main_snap_before
    assert result.commits >= 1
    assert result.snapshot_ids == [new_branch_snap]
