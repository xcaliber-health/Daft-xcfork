"""Expire snapshots: resolve the kept set, commit the metadata change, delete unreachable files."""

from __future__ import annotations

import datetime as _dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from daft.io.iceberg._common import (
    DEFAULT_DELETE_BACKOFF_BASE_SECONDS,
    DEFAULT_DELETE_NUM_RETRIES,
    DEFAULT_MAX_CONCURRENT_DELETES,
    DEFAULT_MAX_CONCURRENT_MANIFEST_READS,
    CommitRetryExhausted,
    commit_with_retry,
    delete_files,
    is_not_found,
    validate_gc_enabled,
)

if TYPE_CHECKING:
    from pyiceberg.manifest import ManifestFile
    from pyiceberg.table import Table as PyIcebergTable

logger = logging.getLogger(__name__)


MAX_SNAPSHOT_AGE_MS_KEY = "history.expire.max-snapshot-age-ms"
MIN_SNAPSHOTS_TO_KEEP_KEY = "history.expire.min-snapshots-to-keep"

_DEFAULT_MAX_SNAPSHOT_AGE_MS = 5 * 24 * 60 * 60 * 1000
_DEFAULT_MIN_SNAPSHOTS_TO_KEEP = 1


@dataclass(frozen=True)
class ExpireResult:
    """Summary of an expire_snapshots invocation.

    Parameters
    ----------
    deleted_data_files_count
        Number of data files removed.
    deleted_position_delete_files_count
        Number of position-delete files removed.
    deleted_equality_delete_files_count
        Number of equality-delete files removed.
    deleted_manifest_files_count
        Number of manifest files removed.
    deleted_manifest_lists_count
        Number of manifest-list files removed.
    deleted_statistics_files_count
        Number of statistics (and partition-statistics) files removed.
    """

    deleted_data_files_count: int = 0
    deleted_position_delete_files_count: int = 0
    deleted_equality_delete_files_count: int = 0
    deleted_manifest_files_count: int = 0
    deleted_manifest_lists_count: int = 0
    deleted_statistics_files_count: int = 0


class ExpireSnapshotsFailedException(RuntimeError):
    """Raised when expire_snapshots cannot commit or make forward progress."""


def run(
    table: PyIcebergTable,
    *,
    older_than: _dt.datetime | int | None = None,
    retain_last: int | None = None,
    snapshot_ids: list[int] | None = None,
    clean_expired_files: bool = True,
    stream_results: bool = False,
    options: dict[str, Any] | None = None,
) -> ExpireResult:
    opts = options or {}
    max_concurrent_deletes = int(
        opts.get("max-concurrent-deletes", DEFAULT_MAX_CONCURRENT_DELETES)
    )
    max_concurrent_manifest_reads = int(
        opts.get(
            "max-concurrent-manifest-reads", DEFAULT_MAX_CONCURRENT_MANIFEST_READS
        )
    )
    delete_num_retries = int(
        opts.get("delete-num-retries", DEFAULT_DELETE_NUM_RETRIES)
    )
    delete_backoff_base = float(
        opts.get("delete-backoff-base-seconds", DEFAULT_DELETE_BACKOFF_BASE_SECONDS)
    )

    validate_gc_enabled(table)

    if older_than is None and retain_last is None and not snapshot_ids:
        max_age_ms = int(
            table.properties.get(MAX_SNAPSHOT_AGE_MS_KEY, _DEFAULT_MAX_SNAPSHOT_AGE_MS)
        )
        older_than = int(time.time() * 1000) - max_age_ms

    if retain_last is not None and retain_last < 1:
        raise ValueError(f"retain_last must be >= 1, got {retain_last!r}")

    original_metadata = table.metadata
    protected_ids = _protected_snapshot_ids(table)

    if snapshot_ids:
        _validate_explicit_snapshot_ids(table, snapshot_ids, protected_ids)

    expired_ids = _resolve_expired_ids(
        table=table,
        older_than=older_than,
        retain_last=retain_last,
        snapshot_ids=snapshot_ids,
        protected_ids=protected_ids,
    )

    if not expired_ids:
        return ExpireResult()

    _commit_expire(table, expired_ids)

    if not clean_expired_files:
        return ExpireResult()

    table.refresh()
    updated_metadata = table.metadata

    kept_snapshots = list(updated_metadata.snapshots)
    expired_snapshots = [
        s for s in original_metadata.snapshots if s.snapshot_id in expired_ids
    ]

    kept_paths, _ = _collect_paths(
        table=table,
        snapshots=kept_snapshots,
        statistics_files=updated_metadata.statistics,
        partition_statistics_files=updated_metadata.partition_statistics,
        max_concurrent_manifest_reads=max_concurrent_manifest_reads,
    )

    if stream_results:
        candidates: Iterable[tuple[str, str]] = _stream_candidate_paths(
            table=table,
            snapshots=expired_snapshots,
            statistics_files=original_metadata.statistics,
            partition_statistics_files=original_metadata.partition_statistics,
            expired_ids=expired_ids,
            max_concurrent_manifest_reads=max_concurrent_manifest_reads,
        )
        to_delete: Iterable[tuple[str, str]] = (
            (path, kind) for path, kind in candidates if path not in kept_paths
        )
    else:
        candidate_paths, _ = _collect_paths(
            table=table,
            snapshots=expired_snapshots,
            statistics_files=[
                s for s in original_metadata.statistics if s.snapshot_id in expired_ids
            ],
            partition_statistics_files=[
                s
                for s in original_metadata.partition_statistics
                if s.snapshot_id in expired_ids
            ],
            max_concurrent_manifest_reads=max_concurrent_manifest_reads,
        )
        to_delete = [(p, k) for p, k in candidate_paths.items() if p not in kept_paths]

    counts, _failed = delete_files(
        table=table,
        to_delete=to_delete,
        max_concurrent_deletes=max_concurrent_deletes,
        num_retries=delete_num_retries,
        backoff_base=delete_backoff_base,
        op_name="expire_snapshots",
    )
    return ExpireResult(
        deleted_data_files_count=counts.get(_KIND_DATA, 0),
        deleted_position_delete_files_count=counts.get(_KIND_POS_DELETE, 0),
        deleted_equality_delete_files_count=counts.get(_KIND_EQ_DELETE, 0),
        deleted_manifest_files_count=counts.get(_KIND_MANIFEST, 0),
        deleted_manifest_lists_count=counts.get(_KIND_MANIFEST_LIST, 0),
        deleted_statistics_files_count=counts.get(_KIND_STATS, 0),
    )


def _protected_snapshot_ids(table: PyIcebergTable) -> set[int]:
    from pyiceberg.table.refs import SnapshotRefType

    return {
        ref.snapshot_id
        for ref in table.metadata.refs.values()
        if ref.snapshot_ref_type in (SnapshotRefType.BRANCH, SnapshotRefType.TAG)
    }


def _validate_explicit_snapshot_ids(
    table: PyIcebergTable, snapshot_ids: list[int], protected_ids: set[int]
) -> None:
    known = {s.snapshot_id for s in table.metadata.snapshots}
    missing = [sid for sid in snapshot_ids if sid not in known]
    if missing:
        raise ValueError(f"snapshot_ids do not exist: {missing!r}")
    illegal = [sid for sid in snapshot_ids if sid in protected_ids]
    if illegal:
        raise ValueError(
            f"snapshot_ids are protected by a branch/tag ref and cannot be expired: {illegal!r}"
        )


def _resolve_expired_ids(
    *,
    table: PyIcebergTable,
    older_than: _dt.datetime | int | None,
    retain_last: int | None,
    snapshot_ids: list[int] | None,
    protected_ids: set[int],
) -> set[int]:
    """Combine all three knobs into a final expiry set, honoring table-property floors."""
    candidates: set[int] = set()

    if snapshot_ids:
        candidates.update(snapshot_ids)

    if older_than is not None:
        older_than_ms = _to_epoch_millis(older_than)
        for s in table.metadata.snapshots:
            if s.timestamp_ms < older_than_ms:
                candidates.add(s.snapshot_id)

    if retain_last is not None:
        min_keep = max(
            retain_last,
            int(
                table.properties.get(
                    MIN_SNAPSHOTS_TO_KEEP_KEY, _DEFAULT_MIN_SNAPSHOTS_TO_KEEP
                )
            ),
        )
        kept = _most_recent_main_snapshot_ids(table, min_keep)
        for s in table.metadata.snapshots:
            if s.snapshot_id not in kept:
                candidates.add(s.snapshot_id)
    else:
        min_keep = int(
            table.properties.get(
                MIN_SNAPSHOTS_TO_KEEP_KEY, _DEFAULT_MIN_SNAPSHOTS_TO_KEEP
            )
        )
        if min_keep > 0:
            kept = _most_recent_main_snapshot_ids(table, min_keep)
            candidates -= kept

    candidates -= protected_ids
    return candidates


def _most_recent_main_snapshot_ids(table: PyIcebergTable, n: int) -> set[int]:
    """Return the IDs of the N most-recent snapshots on the table's current ref chain."""
    from pyiceberg.table.snapshots import ancestors_of

    current = table.metadata.current_snapshot()
    if current is None:
        return set()
    out: list[int] = []
    for snap in ancestors_of(current, table.metadata):
        out.append(snap.snapshot_id)
        if len(out) >= n:
            break
    return set(out)


def _to_epoch_millis(value: _dt.datetime | int) -> int:
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return int(value.timestamp() * 1000)
    return int(value)


def _commit_expire(table: PyIcebergTable, expired_ids: set[int]) -> None:
    """Commit the snapshot-expiry metadata change with bounded OCC retry."""
    state = {"ids": sorted(expired_ids)}
    sentinel = object()

    def _attempt(_: int) -> object:
        if not state["ids"]:
            return sentinel
        table.maintenance.expire_snapshots().by_ids(state["ids"]).commit()
        return sentinel

    def _on_conflict(t: PyIcebergTable) -> object | None:
        known = {s.snapshot_id for s in t.metadata.snapshots}
        protected = _protected_snapshot_ids(t)
        state["ids"] = [sid for sid in state["ids"] if sid in known and sid not in protected]
        if not state["ids"]:
            return sentinel
        return None

    try:
        commit_with_retry(
            table,
            _attempt,
            op_name="expire_snapshots",
            on_conflict=_on_conflict,
        )
    except CommitRetryExhausted as exc:
        raise ExpireSnapshotsFailedException(
            "expire_snapshots: metadata commit could not land within the retry budget"
        ) from exc


_KIND_DATA = "data"
_KIND_POS_DELETE = "pos_delete"
_KIND_EQ_DELETE = "eq_delete"
_KIND_MANIFEST = "manifest"
_KIND_MANIFEST_LIST = "manifest_list"
_KIND_STATS = "stats"


def _collect_paths(
    *,
    table: PyIcebergTable,
    snapshots: Iterable[Any],
    statistics_files: Iterable[Any],
    partition_statistics_files: Iterable[Any],
    max_concurrent_manifest_reads: int,
) -> tuple[dict[str, str], int]:
    """Return ``{path: kind}`` for every file reachable from ``snapshots`` and statistics."""
    paths: dict[str, str] = {}
    manifest_files: list[Any] = []

    for snap in snapshots:
        ml = getattr(snap, "manifest_list", None)
        if ml:
            paths.setdefault(ml, _KIND_MANIFEST_LIST)
        try:
            for m in snap.manifests(table.io):
                manifest_files.append(m)
                paths.setdefault(m.manifest_path, _KIND_MANIFEST)
        except Exception as exc:
            if not is_not_found(exc):
                raise
            logger.warning(
                "expire_snapshots: manifest list missing for snapshot %s (%s); skipping",
                getattr(snap, "snapshot_id", "?"),
                ml,
            )

    for entry_path, kind in _read_manifest_entries(
        table=table,
        manifests=manifest_files,
        max_workers=max_concurrent_manifest_reads,
    ):
        paths.setdefault(entry_path, kind)

    for s in statistics_files:
        paths.setdefault(s.statistics_path, _KIND_STATS)
    for s in partition_statistics_files:
        paths.setdefault(s.statistics_path, _KIND_STATS)

    return paths, len(manifest_files)


def _read_manifest_entries(
    *,
    table: PyIcebergTable,
    manifests: list[ManifestFile],
    max_workers: int,
) -> Iterator[tuple[str, str]]:
    """Yield ``(path, kind)`` for every live data/delete file across ``manifests``."""
    from pyiceberg.manifest import DataFileContent

    def _read_one(m: ManifestFile) -> list[tuple[str, str]]:
        # discard_deleted=True so DELETED entries in a kept snapshot's manifest
        # do not pin retired data files into kept_paths.
        try:
            entries = m.fetch_manifest_entry(table.io, discard_deleted=True)
        except Exception as exc:
            if not is_not_found(exc):
                raise
            logger.warning(
                "expire_snapshots: manifest missing on object store: %s; skipping",
                m.manifest_path,
            )
            return []
        out: list[tuple[str, str]] = []
        for e in entries:
            f = e.data_file
            content = getattr(f, "content", DataFileContent.DATA)
            if content == DataFileContent.POSITION_DELETES:
                kind = _KIND_POS_DELETE
            elif content == DataFileContent.EQUALITY_DELETES:
                kind = _KIND_EQ_DELETE
            else:
                kind = _KIND_DATA
            out.append((f.file_path, kind))
        return out

    if not manifests:
        return iter(())

    workers = max(1, min(max_workers, len(manifests)))
    if workers == 1:
        for m in manifests:
            yield from _read_one(m)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in pool.map(_read_one, manifests):
            yield from batch


def _stream_candidate_paths(
    *,
    table: PyIcebergTable,
    snapshots: Iterable[Any],
    statistics_files: Iterable[Any],
    partition_statistics_files: Iterable[Any],
    expired_ids: set[int],
    max_concurrent_manifest_reads: int,
) -> Iterator[tuple[str, str]]:
    """Yield candidate paths snapshot-by-snapshot for memory-bounded expiration."""
    for snap in snapshots:
        ml = getattr(snap, "manifest_list", None)
        if ml:
            yield ml, _KIND_MANIFEST_LIST
        try:
            ms = list(snap.manifests(table.io))
        except Exception as exc:
            if not is_not_found(exc):
                raise
            logger.warning(
                "expire_snapshots: manifest list missing for snapshot %s (%s); skipping",
                getattr(snap, "snapshot_id", "?"),
                ml,
            )
            continue
        for m in ms:
            yield m.manifest_path, _KIND_MANIFEST
        yield from _read_manifest_entries(
            table=table, manifests=ms, max_workers=max_concurrent_manifest_reads
        )

    for s in statistics_files:
        if s.snapshot_id in expired_ids:
            yield s.statistics_path, _KIND_STATS
    for s in partition_statistics_files:
        if s.snapshot_id in expired_ids:
            yield s.statistics_path, _KIND_STATS
