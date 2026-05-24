"""Shared primitives for the Iceberg maintenance surface.

Centralizes the ``gc.enabled`` gate, the object-store NotFound detector, the
chunked parallel file-delete loop, and the optimistic-concurrency commit
retry helper. Each maintenance operation reads its retry policy from table
properties through :func:`commit_with_retry` so the four APIs behave
uniformly under contention.
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable, Hashable, Iterable, TypeVar

if TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable

logger = logging.getLogger(__name__)


GC_ENABLED_KEY = "gc.enabled"

DEFAULT_MAX_CONCURRENT_DELETES = 4
DEFAULT_MAX_CONCURRENT_MANIFEST_READS = 4
DEFAULT_DELETE_NUM_RETRIES = 3
DEFAULT_DELETE_BACKOFF_BASE_SECONDS = 0.1
DELETE_CHUNK_SIZE = 256

COMMIT_NUM_RETRIES_KEY = "commit.retry.num-retries"
COMMIT_MIN_WAIT_MS_KEY = "commit.retry.min-wait-ms"
COMMIT_MAX_WAIT_MS_KEY = "commit.retry.max-wait-ms"
COMMIT_TOTAL_TIMEOUT_MS_KEY = "commit.retry.total-timeout-ms"

COMMIT_DEFAULT_NUM_RETRIES = 4
COMMIT_DEFAULT_MIN_WAIT_MS = 100
COMMIT_DEFAULT_MAX_WAIT_MS = 60_000
COMMIT_DEFAULT_TOTAL_TIMEOUT_MS = 1_800_000

# Retained for callers still importing the legacy names; behavior matches the
# new helper's defaults when no table property overrides them.
COMMIT_MAX_ATTEMPTS = COMMIT_DEFAULT_NUM_RETRIES
COMMIT_BACKOFF_BASE_SECONDS = COMMIT_DEFAULT_MIN_WAIT_MS / 1000.0


_NOT_FOUND_EXCEPTION_NAMES = frozenset(
    {"FileNotFoundError", "NoSuchKey", "ObjectNotFound", "BlobNotFound"}
)
_NOT_FOUND_MESSAGE_SUBSTRINGS = (
    "not found",
    "no such key",
    "nosuchkey",
    "no such file",
    "resource_not_found",
    "404",
)


def is_not_found(exc: BaseException) -> bool:
    """Return True if ``exc`` represents an object-store NotFound result.

    Matches by exception class name and by case-insensitive substring of
    ``str(exc)``. Covers ``FileNotFoundError`` and the wrapped ``OSError``
    that pyarrow raises for S3 ``AWS Error RESOURCE_NOT_FOUND``.
    """
    if type(exc).__name__ in _NOT_FOUND_EXCEPTION_NAMES:
        return True
    msg = str(exc).lower()
    return any(s in msg for s in _NOT_FOUND_MESSAGE_SUBSTRINGS)


def validate_gc_enabled(table: PyIcebergTable) -> None:
    """Refuse to run if the ``gc.enabled`` table property is explicitly false.

    Parameters
    ----------
    table
        Iceberg table whose properties are read.

    Raises
    ------
    ValueError
        If ``table.properties[gc.enabled]`` resolves to false/0/no.
    """
    raw = table.properties.get(GC_ENABLED_KEY, "true")
    if str(raw).strip().lower() in {"false", "0", "no"}:
        raise ValueError(
            f"refusing to run: table property {GC_ENABLED_KEY}=false. "
            "Set it to true to permit physical file deletion."
        )


K = TypeVar("K", bound=Hashable)


def delete_files(
    *,
    table: PyIcebergTable,
    to_delete: Iterable[tuple[str, K]],
    max_concurrent_deletes: int = DEFAULT_MAX_CONCURRENT_DELETES,
    num_retries: int = DEFAULT_DELETE_NUM_RETRIES,
    backoff_base: float = DEFAULT_DELETE_BACKOFF_BASE_SECONDS,
    op_name: str = "iceberg-op",
) -> tuple[dict[K, int], int]:
    """Delete files in chunked parallel batches.

    Parameters
    ----------
    table
        Iceberg table whose ``io.delete`` performs the deletions.
    to_delete
        Iterable of ``(path, kind)`` pairs. ``kind`` is an opaque hashable
        used only as the key in the returned counts.
    max_concurrent_deletes
        Upper bound on the worker pool size.
    num_retries
        Per-file retry budget on transient errors. NotFound results are
        treated as success on the first attempt.
    backoff_base
        Exponential-backoff base in seconds: attempt ``n`` sleeps for
        ``backoff_base * 2**n`` seconds before the next try.
    op_name
        Used only in the warning log emitted on terminal delete failures.

    Returns
    -------
    tuple of (counts, failed)
        ``counts`` maps each ``kind`` to the number of successful deletes for
        that kind. ``failed`` is the total number of paths that exhausted the
        retry budget.
    """
    counts: dict[K, int] = {}
    failed = 0
    io = table.io

    def _delete_one(item: tuple[str, K]) -> tuple[str, K, bool]:
        path, kind = item
        for attempt in range(num_retries + 1):
            try:
                io.delete(path)
                return path, kind, True
            except FileNotFoundError:
                return path, kind, True
            except Exception as exc:
                if is_not_found(exc):
                    return path, kind, True
                if attempt < num_retries:
                    time.sleep(backoff_base * (2**attempt))
                    continue
                logger.warning("%s: failed to delete %s: %r", op_name, path, exc)
                return path, kind, False
        return path, kind, False

    workers = max(1, max_concurrent_deletes)
    iterator = iter(to_delete)

    if workers == 1:
        for item in iterator:
            _, kind, ok = _delete_one(item)
            if ok:
                counts[kind] = counts.get(kind, 0) + 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                chunk: list[tuple[str, K]] = []
                for _ in range(DELETE_CHUNK_SIZE):
                    try:
                        chunk.append(next(iterator))
                    except StopIteration:
                        break
                if not chunk:
                    break
                for _, kind, ok in pool.map(_delete_one, chunk):
                    if ok:
                        counts[kind] = counts.get(kind, 0) + 1
                    else:
                        failed += 1

    return counts, failed


T = TypeVar("T")


class CommitRetryExhausted(RuntimeError):
    """Raised when an Iceberg commit cannot land within the configured retry budget.

    Parameters
    ----------
    op_name
        Identifier of the maintenance operation that gave up.
    attempts
        Number of commit attempts made before exhaustion.
    elapsed_ms
        Wall time in milliseconds spent across attempts and waits.
    """

    def __init__(self, op_name: str, attempts: int, elapsed_ms: int) -> None:
        super().__init__(
            f"{op_name}: commit retry budget exhausted after {attempts} "
            f"attempt(s), {elapsed_ms}ms elapsed"
        )
        self.op_name = op_name
        self.attempts = attempts
        self.elapsed_ms = elapsed_ms


def _read_retry_policy(table: PyIcebergTable) -> tuple[int, float, float, float]:
    """Resolve commit-retry parameters from table properties.

    Returns
    -------
    tuple
        ``(num_retries, min_wait_s, max_wait_s, total_timeout_s)`` with bounds
        clamped to non-negative values.
    """
    props = table.properties
    num_retries = max(
        0,
        int(props.get(COMMIT_NUM_RETRIES_KEY, COMMIT_DEFAULT_NUM_RETRIES)),
    )
    min_wait_ms = max(
        0,
        int(props.get(COMMIT_MIN_WAIT_MS_KEY, COMMIT_DEFAULT_MIN_WAIT_MS)),
    )
    max_wait_ms = max(
        min_wait_ms,
        int(props.get(COMMIT_MAX_WAIT_MS_KEY, COMMIT_DEFAULT_MAX_WAIT_MS)),
    )
    total_timeout_ms = max(
        0,
        int(props.get(COMMIT_TOTAL_TIMEOUT_MS_KEY, COMMIT_DEFAULT_TOTAL_TIMEOUT_MS)),
    )
    return (
        num_retries,
        min_wait_ms / 1000.0,
        max_wait_ms / 1000.0,
        total_timeout_ms / 1000.0,
    )


def commit_with_retry(
    table: PyIcebergTable,
    attempt_fn: Callable[[int], T],
    *,
    op_name: str,
    on_conflict: Callable[[PyIcebergTable], T | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    rng: Callable[[], float] = random.random,
) -> T:
    """Invoke ``attempt_fn`` with optimistic-concurrency retry.

    The loop retries on optimistic-concurrency commit failures using
    exponential backoff with full jitter, bounded by both the attempt count
    and the total wall-time budget read from the table properties
    ``commit.retry.num-retries``, ``commit.retry.min-wait-ms``,
    ``commit.retry.max-wait-ms`` and ``commit.retry.total-timeout-ms``.

    Parameters
    ----------
    table
        The table whose properties supply the retry policy and whose
        ``refresh()`` is called between attempts.
    attempt_fn
        Callable receiving the zero-based attempt index. Returns the result
        of the operation on success.
    op_name
        Operation label used in the exhausted-retry exception.
    on_conflict
        Optional callback invoked after ``table.refresh()`` between attempts.
        May return a non-``None`` value to short-circuit the loop (e.g. when
        the work is now a no-op or a prior attempt's commit already landed).
    sleep, monotonic, rng
        Injection points to make the helper deterministic in tests.

    Returns
    -------
    T
        The value returned by ``attempt_fn`` on a successful attempt, or the
        value returned by ``on_conflict`` if it short-circuits.

    Raises
    ------
    CommitRetryExhausted
        When the retry budget is exhausted without a successful commit.
    """
    from pyiceberg.exceptions import CommitFailedException

    num_retries, min_wait_s, max_wait_s, total_timeout_s = _read_retry_policy(table)
    start = monotonic()
    last_err: BaseException | None = None
    attempts_made = 0

    for attempt in range(num_retries + 1):
        attempts_made = attempt + 1
        try:
            return attempt_fn(attempt)
        except CommitFailedException as exc:
            last_err = exc
            elapsed = monotonic() - start
            if attempt >= num_retries or elapsed >= total_timeout_s:
                break
            base = min(max_wait_s, min_wait_s * (2**attempt)) if min_wait_s > 0 else 0.0
            wait = base * (1.0 + rng()) if base > 0 else 0.0
            remaining = max(0.0, total_timeout_s - elapsed)
            if wait > remaining:
                wait = remaining
            if wait > 0:
                sleep(wait)
            table.refresh()
            if on_conflict is not None:
                short_circuit = on_conflict(table)
                if short_circuit is not None:
                    return short_circuit

    elapsed_ms = int((monotonic() - start) * 1000)
    raise CommitRetryExhausted(
        op_name=op_name,
        attempts=attempts_made,
        elapsed_ms=elapsed_ms,
    ) from last_err
