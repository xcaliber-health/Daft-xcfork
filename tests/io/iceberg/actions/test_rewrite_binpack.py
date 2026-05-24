"""End-to-end tests for IcebergTable.rewrite_data_files('binpack') / compact_files()."""

from __future__ import annotations

import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table

from tests.io.iceberg.actions._helpers import (
    read_ids as _read_ids,
    scan_file_count as _scan_file_count,
)


def test_binpack_reduces_file_count(make_tiny_table):
    table = make_tiny_table(name="default.t_binpack", n_files=12, rows_per_file=5)
    pre_count = _scan_file_count(table)
    assert pre_count == 12

    dt = Table.from_iceberg(table)
    result = dt.compact_files(
        options={
            "target-file-size-bytes": 64 * 1024 * 1024,
            "min-input-files": 2,
            "rewrite-all": True,
        }
    )

    table.refresh()
    post_count = _scan_file_count(table)
    assert (
        post_count < pre_count
    ), f"expected fewer files, got {pre_count} -> {post_count}"
    assert result.rewritten_files == pre_count
    assert result.added_files == post_count
    assert result.commits == 1
    assert result.rewrite_id
    # Data is preserved.
    assert _read_ids(table) == list(range(12 * 5))


def test_binpack_below_min_input_files_is_noop(make_tiny_table):
    table = make_tiny_table(name="default.t_min", n_files=3, rows_per_file=2)
    pre = _read_ids(table)

    dt = Table.from_iceberg(table)
    result = dt.rewrite_data_files(
        "binpack",
        options={
            "min-input-files": 5,
        },
    )

    table.refresh()
    assert result.rewritten_files == 0
    assert result.added_files == 0
    assert result.commits == 0
    assert _read_ids(table) == pre


def test_binpack_idempotent_replay(make_tiny_table):
    table = make_tiny_table(name="default.t_idemp", n_files=6, rows_per_file=3)
    dt = Table.from_iceberg(table)
    rid = "replay-me"
    r1 = dt.compact_files(
        options={"rewrite-all": True, "min-input-files": 2, "rewrite-id": rid}
    )
    table.refresh()
    r2 = dt.compact_files(
        options={"rewrite-all": True, "min-input-files": 2, "rewrite-id": rid}
    )
    assert r1.rewrite_id == rid == r2.rewrite_id
    assert r2.commits == r1.commits
    assert r2.rewritten_files == r1.rewritten_files
    assert r2.added_files == r1.added_files
    assert r2.bytes_rewritten == r1.bytes_rewritten
    assert r2.bytes_added == r1.bytes_added
    assert r2.snapshot_ids == r1.snapshot_ids
