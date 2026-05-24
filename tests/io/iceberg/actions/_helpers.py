"""Shared helpers for Iceberg action tests.

Importable from any test file in this directory. Keep this module pure
Python (no pyiceberg imports at top level beyond what the helpers
themselves need at call time).
"""

from __future__ import annotations

import threading
from typing import Any

import pyarrow as pa


def strip_scheme(path: str) -> str:
    """Strip a leading ``file://`` from a URI so :mod:`os.path` can read it."""
    return path[len("file://") :] if path.startswith("file://") else path


def scan_file_count(table: Any) -> int:
    """Return the number of live data files reachable from the current snapshot."""
    return len(list(table.scan().plan_files()))


def scan_paths(table: Any) -> set[str]:
    """Return the set of live data file paths reachable from the current snapshot."""
    return {t.file.file_path for t in table.scan().plan_files()}


def read_ids(table: Any) -> list[int]:
    """Return the sorted ``id`` column of the table's live rows."""
    return sorted(int(r["id"]) for r in table.scan().to_arrow().to_pylist())


def snapshot_count(table: Any) -> int:
    """Return the number of snapshots in the table's metadata log."""
    return len(table.metadata.snapshots or [])


def _row_count(table: Any) -> int:
    table.refresh()
    return table.scan().to_arrow().num_rows


class Appender(threading.Thread):
    """Background thread that calls ``table.append`` on a fixed cadence.

    Exposes :attr:`first_commit_event` so tests can wait for the first
    append to land deterministically rather than relying on ``time.sleep``.
    """

    def __init__(self, table: Any, *, interval_s: float, batch_rows: int = 10) -> None:
        super().__init__(daemon=True)
        self._table = table
        self._interval = interval_s
        self._batch = batch_rows
        self._stop_event = threading.Event()
        self.first_commit_event = threading.Event()
        self.commits = 0
        self.errors: list[BaseException] = []

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_event.set()
        self.join(timeout=timeout)

    def wait_for_first_commit(self, timeout: float = 10.0) -> bool:
        return self.first_commit_event.wait(timeout=timeout)

    def run(self) -> None:
        next_id = 1_000_000
        while not self._stop_event.is_set():
            try:
                self._table.refresh()
                self._table.append(
                    pa.table(
                        {
                            "id": pa.array(
                                list(range(next_id, next_id + self._batch)),
                                type=pa.int64(),
                            ),
                            "label": pa.array(
                                ["live"] * self._batch, type=pa.string()
                            ),
                        }
                    )
                )
                self.commits += 1
                if not self.first_commit_event.is_set():
                    self.first_commit_event.set()
                next_id += self._batch
            except BaseException as exc:  # noqa: BLE001
                self.errors.append(exc)
            self._stop_event.wait(self._interval)


def make_seeded_table(
    catalog: Any,
    name: str,
    *,
    n_files: int = 6,
    rows_per_file: int = 100,
) -> Any:
    """Create an unpartitioned ``(id, label)`` table with ``n_files`` appends."""
    from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField, StringType

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "label", StringType(), required=False),
    )
    table = catalog.create_table(
        identifier=name,
        schema=schema,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
    )
    for i in range(n_files):
        start = i * rows_per_file
        table.append(
            pa.table(
                {
                    "id": pa.array(
                        list(range(start, start + rows_per_file)), type=pa.int64()
                    ),
                    "label": pa.array(["seed"] * rows_per_file, type=pa.string()),
                }
            )
        )
    return table
