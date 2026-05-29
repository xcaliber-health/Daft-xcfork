"""Conflict-isolation option for rewrite_data_files.

Serializable isolation (the default) rejects a rewrite when a concurrent
writer adds a file to a partition the rewrite touches. Snapshot isolation
permits such concurrent appends and rejects only when one of the rewrite's
own input files is removed.

Each test injects exactly one foreign operation from inside ``_rewrite_group``
-- after the plan snapshot is taken but before the commit -- so the outcome is
deterministic and does not depend on a background thread's cadence.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pyiceberg")

from daft.catalog import Table
from daft.io.iceberg import RewriteConflict
from daft.io.iceberg import _compact  # noqa: internal — monkeypatching internal helper

from tests.io.iceberg.actions._helpers import _row_count, make_seeded_table

_FOREIGN_ROWS = 20


def _inject_once_around_rewrite(monkeypatch, action, *, before: bool) -> dict[str, int]:
    """Run ``action(table)`` exactly once around the first ``_rewrite_group`` call.

    ``action`` receives the live table and performs the foreign operation that
    must land between the rewrite plan and the commit. When ``before`` is true
    it fires ahead of the group read; otherwise it fires after the outputs are
    produced, which is required when the operation removes the group's own input
    files. Returns a counter dict so callers can assert the injection fired.
    """
    state = {"fired": 0}
    real_rewrite_group = _compact._rewrite_group

    def instrumented(*args: Any, **kwargs: Any):
        first = state["fired"] == 0
        if first and before:
            state["fired"] = 1
            action(kwargs["table"])
        out = real_rewrite_group(*args, **kwargs)
        if first and not before:
            state["fired"] = 1
            action(kwargs["table"])
        return out

    monkeypatch.setattr(_compact, "_rewrite_group", instrumented)
    return state


def _append_foreign_rows(table: Any) -> None:
    table.refresh()
    table.append(
        pa.table(
            {
                "id": pa.array(list(range(5_000_000, 5_000_000 + _FOREIGN_ROWS)), type=pa.int64()),
                "label": pa.array(["foreign"] * _FOREIGN_ROWS, type=pa.string()),
            }
        )
    )


def _delete_all_seed_rows(table: Any) -> None:
    table.refresh()
    table.delete(delete_filter="label = 'seed'")


def test_snapshot_isolation_allows_concurrent_partition_append(local_catalog, monkeypatch):
    # Arrange
    table = make_seeded_table(local_catalog, "default.t_iso_snapshot_ok", n_files=6)
    seed_rows = _row_count(table)
    state = _inject_once_around_rewrite(monkeypatch, _append_foreign_rows, before=True)
    dt = Table.from_iceberg(table)

    # Act
    result = dt.rewrite_data_files(
        strategy="binpack",
        options={
            "rewrite-all": True,
            "min-input-files": 2,
            "conflict-isolation": "snapshot",
        },
    )

    # Assert: rewrite committed, the foreign append survived, no rows lost.
    assert state["fired"] == 1
    assert result.added_files >= 1
    assert _row_count(table) == seed_rows + _FOREIGN_ROWS


def test_serializable_isolation_rejects_concurrent_partition_append(local_catalog, monkeypatch):
    # Arrange
    table = make_seeded_table(local_catalog, "default.t_iso_serializable_conflict", n_files=6)
    state = _inject_once_around_rewrite(monkeypatch, _append_foreign_rows, before=True)
    dt = Table.from_iceberg(table)

    # Act / Assert: default isolation rejects the same concurrent append.
    with pytest.raises(RewriteConflict):
        dt.rewrite_data_files(
            strategy="binpack",
            options={"rewrite-all": True, "min-input-files": 2},
        )
    assert state["fired"] == 1


def test_snapshot_isolation_still_rejects_vanished_inputs(local_catalog, monkeypatch):
    # Arrange: the foreign op removes the very files the rewrite is replacing.
    table = make_seeded_table(local_catalog, "default.t_iso_snapshot_vanished", n_files=6)
    state = _inject_once_around_rewrite(monkeypatch, _delete_all_seed_rows, before=False)
    dt = Table.from_iceberg(table)

    # Act / Assert: snapshot isolation does not mask a vanished-input conflict.
    with pytest.raises(RewriteConflict):
        dt.rewrite_data_files(
            strategy="binpack",
            options={
                "rewrite-all": True,
                "min-input-files": 2,
                "conflict-isolation": "snapshot",
            },
        )
    assert state["fired"] == 1


@pytest.mark.parametrize("bad_value", ["serial", "SNAPSHOT", "", "none"])
def test_invalid_conflict_isolation_value_rejected(local_catalog, bad_value):
    # Arrange
    table = make_seeded_table(local_catalog, f"default.t_iso_bad_{abs(hash(bad_value))}", n_files=2)
    dt = Table.from_iceberg(table)

    # Act / Assert
    with pytest.raises(ValueError, match="conflict-isolation"):
        dt.rewrite_data_files(options={"conflict-isolation": bad_value})
