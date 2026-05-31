"""Delete files under the table location that no snapshot still references.

Lists physical files under the table root, subtracts the union of files
reachable from every snapshot, and deletes the remainder. The set-difference
between listed and reachable paths is delegated to a Rust helper; metadata
traversal, path normalization, and deletion run in Python.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from daft.io.iceberg._common import (
    DEFAULT_DELETE_BACKOFF_BASE_SECONDS,
    DEFAULT_DELETE_NUM_RETRIES,
    DEFAULT_MAX_CONCURRENT_DELETES,
    delete_files,
    validate_gc_enabled,
)

if TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable

logger = logging.getLogger(__name__)


DEFAULT_OLDER_THAN_MS = 3 * 24 * 60 * 60 * 1000
MIN_AGE_MS = 24 * 60 * 60 * 1000
DEFAULT_MAX_CONCURRENT_LIST = 4
DEFAULT_SAMPLE_LIMIT = 1000

_VALID_PREFIX_MODES = frozenset({"error", "delete", "ignore"})
_SCHEME_ALIASES = {"s3a": "s3", "s3n": "s3"}
_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+\-.]*)://")
_ORPHAN_KIND = "file"


@dataclass(frozen=True)
class RemoveOrphanResult:
    """Summary of a remove_orphan_files invocation.

    Parameters
    ----------
    orphan_files_count
        Number of files identified as orphans (present in listing, absent
        from the table's reachable set).
    deleted_files_count
        Number of orphans successfully deleted. Equals ``orphan_files_count``
        unless ``dry_run=True`` or some deletes failed.
    sample_paths
        Up to ``sample-limit`` orphan paths in listing order. Useful for
        operator review when running with ``dry_run=True``.
    skipped_prefix_mismatch_count
        Number of listed files dropped from the candidate set because their
        scheme/authority did not match any reachable path and
        ``prefix_mismatch_mode`` was ``"ignore"``.
    failed_deletes
        Orphans whose delete exhausted the per-file retry budget.
    """

    orphan_files_count: int = 0
    deleted_files_count: int = 0
    sample_paths: list[str] = field(default_factory=list)
    skipped_prefix_mismatch_count: int = 0
    failed_deletes: int = 0


class PrefixMismatchError(ValueError):
    """Raised when listed files use a scheme/authority absent from reachable paths."""


def run(
    table: PyIcebergTable,
    *,
    older_than: _dt.datetime | int | None = None,
    location: str | None = None,
    dry_run: bool = False,
    prefix_mismatch_mode: str = "error",
    stream_results: bool = False,
    options: dict[str, Any] | None = None,
) -> RemoveOrphanResult:
    if prefix_mismatch_mode not in _VALID_PREFIX_MODES:
        raise ValueError(
            f"prefix_mismatch_mode must be one of {sorted(_VALID_PREFIX_MODES)}, "
            f"got {prefix_mismatch_mode!r}"
        )

    opts = options or {}
    max_concurrent_list = int(
        opts.get("max-concurrent-list", DEFAULT_MAX_CONCURRENT_LIST)
    )
    max_concurrent_deletes = int(
        opts.get("max-concurrent-deletes", DEFAULT_MAX_CONCURRENT_DELETES)
    )
    delete_num_retries = int(
        opts.get("delete-num-retries", DEFAULT_DELETE_NUM_RETRIES)
    )
    delete_backoff_base = float(
        opts.get("delete-backoff-base-seconds", DEFAULT_DELETE_BACKOFF_BASE_SECONDS)
    )
    sample_limit = int(opts.get("sample-limit", DEFAULT_SAMPLE_LIMIT))
    allow_recent = bool(opts.get("allow-recent", False))

    validate_gc_enabled(table)

    older_than_ms = _resolve_older_than_ms(older_than, allow_recent=allow_recent)

    base_location = _resolve_location(table, location)

    canon = _build_canonicalizer(opts)
    reachable = _reachable_paths(table, canon)

    listed_iter = _list_files(
        base_location,
        older_than_ms=older_than_ms,
        max_workers=max_concurrent_list,
        canon=canon,
    )

    candidates, mismatched = _apply_prefix_mode(
        listed_iter,
        reachable=reachable,
        mode=prefix_mismatch_mode,
        canon=canon,
    )

    if not stream_results:
        candidates = list(candidates)

    orphans = _compute_orphans(candidates, reachable)

    sample = orphans[:sample_limit]
    if dry_run:
        return RemoveOrphanResult(
            orphan_files_count=len(orphans),
            deleted_files_count=0,
            sample_paths=sample,
            skipped_prefix_mismatch_count=mismatched,
            failed_deletes=0,
        )

    counts, failed = delete_files(
        table=table,
        to_delete=((p, _ORPHAN_KIND) for p in orphans),
        max_concurrent_deletes=max_concurrent_deletes,
        num_retries=delete_num_retries,
        backoff_base=delete_backoff_base,
        op_name="remove_orphan_files",
    )
    deleted = counts.get(_ORPHAN_KIND, 0)
    return RemoveOrphanResult(
        orphan_files_count=len(orphans),
        deleted_files_count=deleted,
        sample_paths=sample,
        skipped_prefix_mismatch_count=mismatched,
        failed_deletes=failed,
    )


def _resolve_older_than_ms(
    older_than: _dt.datetime | int | None, *, allow_recent: bool
) -> int:
    now_ms = int(time.time() * 1000)
    if older_than is None:
        return now_ms - DEFAULT_OLDER_THAN_MS

    if isinstance(older_than, _dt.datetime):
        if older_than.tzinfo is None:
            older_than = older_than.replace(tzinfo=_dt.timezone.utc)
        cutoff_ms = int(older_than.timestamp() * 1000)
    else:
        cutoff_ms = int(older_than)

    if not allow_recent and cutoff_ms > now_ms - MIN_AGE_MS:
        raise ValueError(
            "remove_orphan_files: refusing to use a cutoff newer than 24 hours ago "
            "(risk of deleting files written by concurrent jobs). "
            "Pass options={'allow-recent': True} only in tests."
        )
    return cutoff_ms


def _resolve_location(table: PyIcebergTable, location: str | None) -> str:
    table_loc = table.location().rstrip("/")
    if location is None:
        return table_loc
    loc = location.rstrip("/")
    if _canonical(loc) != _canonical(table_loc) and not _canonical(loc).startswith(
        _canonical(table_loc) + "/"
    ):
        raise ValueError(
            f"location={location!r} is not a subpath of table.location()={table.location()!r}"
        )
    return loc


def _reachable_paths(table: PyIcebergTable, canon: _PathCanonicalizer) -> set[str]:
    """Return canonical paths of every file the table metadata still references.

    Includes data and delete files (across all snapshots), manifest files,
    manifest lists, statistics and partition-statistics files, every recorded
    metadata.json (current + log), and the table's own metadata pointer file.
    """
    paths: set[str] = set()

    try:
        data_files_tbl = table.inspect.all_files()
        for p in data_files_tbl.column("file_path").to_pylist():
            if p:
                paths.add(canon.canonical(p))
    except Exception as exc:
        logger.warning("remove_orphan_files: inspect.all_files failed: %r", exc)

    try:
        manifests_tbl = table.inspect.all_manifests()
        for p in manifests_tbl.column("path").to_pylist():
            if p:
                paths.add(canon.canonical(p))
    except Exception as exc:
        logger.warning("remove_orphan_files: inspect.all_manifests failed: %r", exc)

    for snap in table.metadata.snapshots:
        ml = getattr(snap, "manifest_list", None)
        if ml:
            paths.add(canon.canonical(ml))

    for s in getattr(table.metadata, "statistics", []) or []:
        paths.add(canon.canonical(s.statistics_path))
    for s in getattr(table.metadata, "partition_statistics", []) or []:
        paths.add(canon.canonical(s.statistics_path))

    for entry in getattr(table.metadata, "metadata_log", []) or []:
        paths.add(canon.canonical(entry.metadata_file))
    current_md = getattr(table, "metadata_location", None)
    if current_md:
        paths.add(canon.canonical(current_md))

    return paths


def _list_files(
    location: str, *, older_than_ms: int, max_workers: int, canon: _PathCanonicalizer
) -> Iterator[str]:
    """Yield canonical paths under ``location`` modified before ``older_than_ms``.

    Walks the underlying filesystem (PyArrow) starting from the immediate child
    directories of ``location`` in parallel. Files at the top level are walked
    inline. Mtime-newer-than-cutoff files are skipped to avoid racing live writers.
    Each path is canonicalized so it compares equal to a reachable path that
    names the same location through an equivalent scheme or authority.
    """
    import pyarrow.fs as pafs

    fs, base = pafs.FileSystem.from_uri(location)
    scheme = canon.scheme_of(canon.canonical(location))

    try:
        children = fs.get_file_info(pafs.FileSelector(base, recursive=False))
    except Exception as exc:
        logger.warning("remove_orphan_files: top-level listing failed at %s: %r", base, exc)
        return

    top_files = [c for c in children if c.type == pafs.FileType.File]
    top_dirs = [c.path for c in children if c.type == pafs.FileType.Directory]

    for info in top_files:
        if int(info.mtime_ns / 1_000_000) < older_than_ms:
            yield canon.canonical(_scheme_join(scheme, info.path))

    if not top_dirs:
        return

    workers = max(1, min(max_workers, len(top_dirs)))

    def _walk(dir_path: str) -> list[str]:
        out: list[str] = []
        try:
            entries = fs.get_file_info(pafs.FileSelector(dir_path, recursive=True))
        except Exception as exc:
            logger.warning(
                "remove_orphan_files: subtree listing failed at %s: %r", dir_path, exc
            )
            return out
        for info in entries:
            if info.type != pafs.FileType.File:
                continue
            if int(info.mtime_ns / 1_000_000) >= older_than_ms:
                continue
            out.append(canon.canonical(_scheme_join(scheme, info.path)))
        return out

    if workers == 1:
        for dir_path in top_dirs:
            yield from _walk(dir_path)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in pool.map(_walk, top_dirs):
            yield from batch


def _apply_prefix_mode(
    listed: Iterator[str], *, reachable: set[str], mode: str, canon: _PathCanonicalizer
) -> tuple[Iterable[str], int]:
    """Filter listed paths by scheme/authority match against the reachable set.

    Parameters
    ----------
    listed
        Iterator over canonical paths returned by the file lister.
    reachable
        Canonical paths the table metadata still references.
    mode
        ``"error"`` raises if any listed path's scheme/authority is absent
        from the reachable set's prefixes; ``"delete"`` passes mismatches
        through as candidates; ``"ignore"`` drops mismatches.
    canon
        Canonicalizer used to derive the scheme/authority prefix of each path.

    Returns
    -------
    tuple of (candidates, skipped_count)
        ``candidates`` is the filtered iterable; ``skipped_count`` is the
        number of mismatches dropped under ``"ignore"``.
    """
    reachable_prefixes = {canon.prefix(p) for p in reachable}
    skipped = 0
    mismatches: list[str] = []
    out: list[str] = []
    for path in listed:
        if canon.prefix(path) in reachable_prefixes:
            out.append(path)
            continue
        if mode == "delete":
            out.append(path)
        elif mode == "ignore":
            skipped += 1
        else:
            mismatches.append(path)
    if mode == "error" and mismatches:
        sample = mismatches[:5]
        raise PrefixMismatchError(
            f"remove_orphan_files: {len(mismatches)} listed file(s) use a "
            f"scheme/authority not present in the table's reachable set. "
            f"Sample: {sample!r}. Pass prefix_mismatch_mode='delete' or "
            f"'ignore' to override."
        )
    return out, skipped


def _compute_orphans(candidates: Iterable[str], reachable: set[str]) -> list[str]:
    """Return ``candidates - reachable`` using the Rust helper when available."""
    listed = list(candidates)
    if not listed:
        return []
    try:
        from daft.daft import _iceberg as _iceberg_native
    except ImportError:
        _iceberg_native = None
    if _iceberg_native is not None and hasattr(_iceberg_native, "orphan_diff_py"):
        return _iceberg_native.orphan_diff_py(listed, list(reachable))
    return [p for p in listed if p not in reachable]


@dataclass(frozen=True)
class _PathCanonicalizer:
    """Map equivalent schemes and authorities to one canonical spelling.

    Two paths that name the same physical location through different but
    declared-equivalent schemes (such as ``s3a`` and ``s3``) or authorities
    (such as a direct host and an endpoint alias) canonicalize to the same
    string, so comparing reachable and listed paths does not flag a live file as
    an orphan merely because the two sides spell its location differently.
    """

    scheme_aliases: dict[str, str]
    authority_aliases: dict[str, str]

    def canonical(self, path: str) -> str:
        m = _SCHEME_RE.match(path)
        if not m:
            return path
        scheme = m.group(1).lower()
        scheme = self.scheme_aliases.get(scheme, scheme)
        rest = path[m.end() :]
        slash = rest.find("/")
        if slash < 0:
            authority, body = rest, ""
        else:
            authority, body = rest[:slash], rest[slash:]
        authority = self.authority_aliases.get(authority, authority)
        return f"{scheme}://{authority}{body}"

    def scheme_of(self, canon: str) -> str:
        idx = canon.find("://")
        return canon[:idx] if idx >= 0 else ""

    def prefix(self, canon: str) -> str:
        idx = canon.find("://")
        if idx < 0:
            return ""
        rest = canon[idx + 3 :]
        slash = rest.find("/")
        authority = rest if slash < 0 else rest[:slash]
        return f"{canon[:idx]}://{authority}"


_DEFAULT_CANONICALIZER = _PathCanonicalizer(scheme_aliases=_SCHEME_ALIASES, authority_aliases={})


def _build_canonicalizer(opts: dict[str, Any]) -> _PathCanonicalizer:
    """Build a path canonicalizer from the equivalence options.

    The default scheme equivalences are always applied; caller-supplied
    equivalences extend them.
    """
    scheme_aliases = dict(_SCHEME_ALIASES)
    for key, value in (opts.get("equal-schemes") or {}).items():
        scheme_aliases[str(key).lower()] = str(value).lower()
    authority_aliases = {
        str(key): str(value) for key, value in (opts.get("equal-authorities") or {}).items()
    }
    return _PathCanonicalizer(scheme_aliases=scheme_aliases, authority_aliases=authority_aliases)


def _canonical(path: str) -> str:
    return _DEFAULT_CANONICALIZER.canonical(path)


def _scheme_of(canon: str) -> str:
    return _DEFAULT_CANONICALIZER.scheme_of(canon)


def _scheme_join(scheme: str, body: str) -> str:
    body = body.lstrip("/") if scheme not in {"", "file"} else body
    if not scheme:
        return body
    if scheme == "file":
        return f"file://{body}" if body.startswith("/") else f"file:///{body}"
    return f"{scheme}://{body}"


def _prefix(canon_path: str) -> str:
    return _DEFAULT_CANONICALIZER.prefix(canon_path)
