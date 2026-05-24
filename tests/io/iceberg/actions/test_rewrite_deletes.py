"""Delete-handling tests for rewrite_data_files."""

from __future__ import annotations

import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.daft import _iceberg as _rust_iceberg


def test_equality_deletes_refuse_via_planner():
    """The Rust planner short-circuits the moment a candidate carries equality deletes."""
    candidates = [
        {
            "path": "/x/a.parquet",
            "size_bytes": 1024,
            "partition_key": "{}",
            "partition_spec_id": 0,
            "positional_delete_paths": [],
            "has_equality_deletes": True,
        },
    ]
    with pytest.raises(_rust_iceberg.EqualityDeletesPresentError):
        _rust_iceberg.plan_file_groups_py(candidates, {"min-input-files": 2}, 0)


def test_positional_deletes_are_merged(make_tiny_table):
    """Run rewrite on a table after a partial delete; deleted rows must not reappear."""
    from pyiceberg.expressions import EqualTo

    table = make_tiny_table(name="default.t_pos_del", n_files=8, rows_per_file=4)
    # Delete a specific row by id; pyiceberg emits a delete via overwrite.
    table.delete(EqualTo("id", 5))
    table.refresh()
    pre_ids = sorted(int(r["id"]) for r in table.scan().to_arrow().to_pylist())
    assert 5 not in pre_ids

    dt = Table.from_iceberg(table)
    result = dt.compact_files(options={"rewrite-all": True, "min-input-files": 2})

    table.refresh()
    post_ids = sorted(int(r["id"]) for r in table.scan().to_arrow().to_pylist())
    assert post_ids == pre_ids
    assert result.added_files >= 1
