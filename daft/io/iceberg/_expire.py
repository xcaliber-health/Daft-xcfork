"""expire_snapshots orchestration: snapshot-set resolution, metadata commit, file cleanup."""

from __future__ import annotations

import datetime as _dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator

if TYPE_CHECKING:
    from pyiceberg.manifest import ManifestFile
    from pyiceberg.table import Table as PyIcebergTable

logger = logging.getLogger(__name__)


GC_ENABLED_KEY = "gc.enabled"
MAX_SNAPSHOT_AGE_MS_KEY = "history.expire.max-snapshot-age-ms"
MIN_SNAPSHOTS_TO_KEEP_KEY = "history.expire.min-snapshots-to-keep"

_DEFAULT_MAX_SNAPSHOT_AGE_MS = 5 * 24 * 60 * 60 * 1000  # 5 days
_DEFAULT_MIN_SNAPSHOTS_TO_KEEP = 1

_DEFAULT_MAX_CONCURRENT_DELETES = 4
_DEFAULT_MAX_CONCURRENT_MANIFEST_READS = 4
_DEFAULT_DELETE_NUM_RETRIES = 3
_DEFAULT_DELETE_BACKOFF_BASE_SECONDS = 0.1
_COMMIT_MAX_ATTEMPTS = 4
_COMMIT_BACKOFF_BASE_SECONDS = 0.1


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
    """Expire snapshots could not commit or could not make forward progress."""


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
        opts.get("max-concurrent-deletes", _DEFAULT_MAX_CONCURRENT_DELETES)
    )
    max_concurrent_manifest_reads = int(
        opts.get(
            "max-concurrent-manifest-reads", _DEFAULT_MAX_CONCURRENT_MANIFEST_READS
        )
    )
    delete_num_retries = int(
        opts.get("delete-num-retries", _DEFAULT_DELETE_NUM_RETRIES)
    )
    delete_backoff_base = float(
        opts.get("delete-backoff-base-seconds", _DEFAULT_DELETE_BACKOFF_BASE_SECONDS)
    )

    _validate_gc_enabled(table)

    if older_than is None and retain_last is None and not snapshot_ids:
        # Fall back to table property defaults so callers get sensible behavior
        # from a bare expire_snapshots() invocation.
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

    return _delete_files(
        table=table,
        to_delete=to_delete,
        max_concurrent_deletes=max_concurrent_deletes,
        num_retries=delete_num_retries,
        backoff_base=delete_backoff_base,
    )


def _validate_gc_enabled(table: PyIcebergTable) -> None:
    raw = table.properties.get(GC_ENABLED_KEY, "true")
    if str(raw).strip().lower() in {"false", "0", "no"}:
        raise ValueError(
            f"expire_snapshots refuses to run: table property {GC_ENABLED_KEY}=false. "
            "Set it to true to permit physical file deletion."
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
        # Even without explicit retain_last, the min-snapshots floor still applies as a guard
        # so we never strand the table below the configured minimum.
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
    """Atomic metadata commit with bounded retry on CAS conflict."""
    from pyiceberg.exceptions import CommitFailedException

    last_err: Exception | None = None
    ids = sorted(expired_ids)
    for attempt in range(_COMMIT_MAX_ATTEMPTS):
        try:
            table.maintenance.expire_snapshots().by_ids(ids).commit()
            return
        except CommitFailedException as e:
            last_err = e
            if attempt < _COMMIT_MAX_ATTEMPTS - 1:
                time.sleep(_COMMIT_BACKOFF_BASE_SECONDS * (2**attempt))
                table.refresh()
                # Re-validate that the targeted IDs still exist and aren't protected after refresh.
                known = {s.snapshot_id for s in table.metadata.snapshots}
                protected = _protected_snapshot_ids(table)
                ids = [sid for sid in ids if sid in known and sid not in protected]
                if not ids:
                    return
                continue
    assert last_err is not None
    raise ExpireSnapshotsFailedException(
        f"expire_snapshots: metadata commit failed after {_COMMIT_MAX_ATTEMPTS} attempts"
    ) from last_err


# ---------- reachability ----------

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
    """Return ``{path -> kind}`` for every file reachable from ``snapshots`` plus statistics."""
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
        except FileNotFoundError:
            # A prior expire may already have removed a manifest list; skip.
            continue

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
    """Yield (path, kind) for every data/delete file across a list of manifests."""
    from pyiceberg.manifest import DataFileContent

    def _read_one(m: ManifestFile) -> list[tuple[str, str]]:
        try:
            entries = m.fetch_manifest_entry(table.io, discard_deleted=False)
        except FileNotFoundError:
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
    """Stream candidate paths snapshot-by-snapshot for memory-bounded expiration."""
    for snap in snapshots:
        ml = getattr(snap, "manifest_list", None)
        if ml:
            yield ml, _KIND_MANIFEST_LIST
        try:
            ms = list(snap.manifests(table.io))
        except FileNotFoundError:
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


# ---------- delete loop ----------


_DELETE_CHUNK_SIZE = 256


def _delete_files(
    *,
    table: PyIcebergTable,
    to_delete: Iterable[tuple[str, str]],
    max_concurrent_deletes: int,
    num_retries: int,
    backoff_base: float,
) -> ExpireResult:
    """Delete files in chunked parallel batches.

    Accepts any iterable. Materializes only one chunk at a time, so streaming
    callers (`stream_results=True`) stay memory-bounded at ``_DELETE_CHUNK_SIZE``.
    """
    counts = {
        _KIND_DATA: 0,
        _KIND_POS_DELETE: 0,
        _KIND_EQ_DELETE: 0,
        _KIND_MANIFEST: 0,
        _KIND_MANIFEST_LIST: 0,
        _KIND_STATS: 0,
    }
    io = table.io

    def _delete_one(item: tuple[str, str]) -> tuple[str, str, bool]:
        path, kind = item
        for attempt in range(num_retries + 1):
            try:
                io.delete(path)
                return path, kind, True
            except FileNotFoundError:
                return path, kind, True
            except Exception as exc:
                if _is_not_found(exc):
                    return path, kind, True
                if attempt < num_retries:
                    time.sleep(backoff_base * (2**attempt))
                    continue
                logger.warning("expire_snapshots: failed to delete %s: %r", path, exc)
                return path, kind, False
        return path, kind, False

    workers = max(1, max_concurrent_deletes)
    iterator = iter(to_delete)

    if workers == 1:
        for item in iterator:
            _, kind, ok = _delete_one(item)
            if ok:
                counts[kind] += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                chunk: list[tuple[str, str]] = []
                for _ in range(_DELETE_CHUNK_SIZE):
                    try:
                        chunk.append(next(iterator))
                    except StopIteration:
                        break
                if not chunk:
                    break
                for _, kind, ok in pool.map(_delete_one, chunk):
                    if ok:
                        counts[kind] += 1

    return ExpireResult(
        deleted_data_files_count=counts[_KIND_DATA],
        deleted_position_delete_files_count=counts[_KIND_POS_DELETE],
        deleted_equality_delete_files_count=counts[_KIND_EQ_DELETE],
        deleted_manifest_files_count=counts[_KIND_MANIFEST],
        deleted_manifest_lists_count=counts[_KIND_MANIFEST_LIST],
        deleted_statistics_files_count=counts[_KIND_STATS],
    )


def _is_not_found(exc: BaseException) -> bool:
    """True for object-store NotFound errors that should be treated as success."""
    name = type(exc).__name__
    if name in {"FileNotFoundError", "NoSuchKey", "ObjectNotFound", "BlobNotFound"}:
        return True
    msg = str(exc).lower()
    return "not found" in msg or "no such key" in msg or "404" in msg
