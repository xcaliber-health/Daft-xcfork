"""Smoke + helper tests for `remove-dangling-deletes`."""

from __future__ import annotations

import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.io.iceberg._compact import _remove_dangling_deletes  # noqa: internal


def test_remove_dangling_deletes_option_is_safe_noop(make_tiny_table):
    table = make_tiny_table(name="default.t_dangling", n_files=6, rows_per_file=3)
    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "remove-dangling-deletes": True,
        }
    )
    table.refresh()
    # No deletes to sweep → removed_delete_files unchanged from main rewrite path.
    assert result.removed_delete_files == 0
    assert result.added_files >= 1


def test_remove_dangling_deletes_helper_on_clean_table(make_tiny_table):
    table = make_tiny_table(name="default.t_dangling_helper", n_files=4, rows_per_file=2)
    removed = _remove_dangling_deletes(table, branch=None)
    assert removed == 0
