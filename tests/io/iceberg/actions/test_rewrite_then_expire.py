"""End-to-end tests covering rewrite_data_files followed by expire_snapshots."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table

from tests.io.iceberg.actions._helpers import (
    read_ids as _read_ids,
    scan_file_count as _scan_file_count,
    scan_paths as _scan_paths,
    strip_scheme as _strip_scheme,
)


def test_rewrite_then_expire_cleans_orphan_data_files(make_tiny_table):
    table = make_tiny_table(
        name="default.t_rewrite_expire", n_files=8, rows_per_file=5
    )
    pre_paths = _scan_paths(table)
    pre_ids = _read_ids(table)
    assert _scan_file_count(table) == 8

    dt = Table.from_iceberg(table)
    rewrite_result = dt.compact_files(
        options={
            "target-file-size-bytes": 64 * 1024 * 1024,
            "min-input-files": 2,
            "rewrite-all": True,
        }
    )
    assert rewrite_result.commits == 1
    assert rewrite_result.rewritten_files == 8

    table.refresh()
    post_rewrite_paths = _scan_paths(table)
    for p in pre_paths:
        assert os.path.exists(_strip_scheme(p))
    assert pre_paths.isdisjoint(post_rewrite_paths)

    expire_result = dt.expire_snapshots(retain_last=1)

    table.refresh()
    assert _read_ids(table) == pre_ids
    assert expire_result.deleted_data_files_count >= 8
    for p in pre_paths:
        assert not os.path.exists(_strip_scheme(p))


def test_compacted_files_land_in_standard_data_layout(make_tiny_table):
    table = make_tiny_table(
        name="default.t_rewrite_paths", n_files=6, rows_per_file=4
    )
    base = table.location().rstrip("/") + "/data"

    dt = Table.from_iceberg(table)
    dt.compact_files(
        options={
            "target-file-size-bytes": 64 * 1024 * 1024,
            "min-input-files": 2,
            "rewrite-all": True,
        }
    )
    table.refresh()

    paths = _scan_paths(table)
    assert paths
    for p in paths:
        bare = _strip_scheme(p)
        assert bare.startswith(_strip_scheme(base) + "/")
        assert "__daft_rewrite__" not in bare


def test_full_maintenance_cycle(make_tiny_table):
    """rewrite_data_files -> rewrite_manifests -> expire_snapshots -> remove_orphan_files."""
    import time as _time
    from datetime import datetime, timezone

    table = make_tiny_table(name="default.t_full_cycle", n_files=8, rows_per_file=5)
    pre_ids = _read_ids(table)
    pre_paths = _scan_paths(table)

    dt = Table.from_iceberg(table)

    r1 = dt.compact_files(
        options={
            "target-file-size-bytes": 64 * 1024 * 1024,
            "min-input-files": 2,
            "rewrite-all": True,
        }
    )
    assert r1.commits == 1
    table.refresh()

    r2 = dt.rewrite_manifests(
        options={"manifest-target-size-bytes": 64 * 1024 * 1024, "manifest-min-count-to-merge": 2}
    )
    assert r2.snapshot_id is not None
    table.refresh()

    r3 = dt.expire_snapshots(retain_last=1)
    assert r3.deleted_data_files_count >= 8
    table.refresh()

    # Plant a true orphan and a sleep so backdating mtime predates the cutoff
    # without using allow-recent.
    base = _strip_scheme(table.location())
    planted = os.path.join(base, "data", "_full_cycle_stray.parquet")
    os.makedirs(os.path.dirname(planted), exist_ok=True)
    with open(planted, "wb") as f:
        f.write(b"orphan")
    backdated = _time.time() - 60
    os.utime(planted, (backdated, backdated))

    r4 = dt.remove_orphan_files(
        older_than=datetime.now(tz=timezone.utc),
        options={"allow-recent": True},
    )
    assert r4.orphan_files_count == 1
    assert r4.deleted_files_count == 1
    assert not os.path.exists(planted)

    # Table is still queryable end-to-end.
    table.refresh()
    assert _read_ids(table) == pre_ids
    for p in pre_paths:
        assert not os.path.exists(_strip_scheme(p))
