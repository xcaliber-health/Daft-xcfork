"""Iceberg `rewrite_data_files` orchestration.

Drives candidate enumeration via pyiceberg, file-group planning via the Rust crate,
per-group read+write via Daft's existing read/write pipeline, and atomic commit via
pyiceberg's overwrite snapshot.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from daft.daft import _iceberg as _rust_iceberg

if TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable

logger = logging.getLogger(__name__)

# Equality-delete refusal mirrors Iceberg's RewriteDataFilesSparkAction semantics:
# applying equality deletes during compaction would require an equality-delete reader,
# which is not implemented. Users must rewrite equality deletes into positional
# deletes (or apply them) before compacting.
EqualityDeletesPresent = _rust_iceberg.EqualityDeletesPresentError


class RewriteConflict(RuntimeError):
    """Raised when a concurrent transaction removed input files mid-retry.

    Output files have already been written to the data root but cannot safely be
    committed — another rewrite already replaced part of our input. The message
    lists the orphaned output paths so the operator can clean them up.
    """


SUPPORTED_STRATEGIES = ("binpack", "sort", "zorder")
_VALID_SORT_DIRECTIONS = {"asc", "desc"}
_VALID_NULL_ORDERS = {"nulls-first", "nulls-last"}
_ZORDER_KEY_COL = "__daft_zorder_key__"
SNAPSHOT_PROP_REWRITE_ID = "daft.rewrite-id"
SNAPSHOT_PROP_STRATEGY = "daft.rewrite-strategy"
SNAPSHOT_PROP_INPUT_FILES = "daft.rewrite-input-files"
SNAPSHOT_PROP_OUTPUT_FILES = "daft.rewrite-output-files"
SNAPSHOT_PROP_BATCH = "daft.rewrite-batch"

_COMMIT_MAX_ATTEMPTS = 4
_COMMIT_BACKOFF_BASE_SECONDS = 0.1


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of a `rewrite_data_files` call.

    `rewritten_files` and `bytes_rewritten` count data files removed; `added_files` and
    `bytes_added` count data files written. `failed_groups` is non-zero only under
    partial-progress mode when one or more batched commits exhausted retries.
    """

    strategy: str
    rewritten_files: int
    added_files: int
    bytes_rewritten: int
    bytes_added: int
    removed_delete_files: int
    failed_groups: int
    commits: int
    snapshot_ids: list[int] = field(default_factory=list)
    rewrite_id: str = ""


def run(
    table: PyIcebergTable,
    strategy: str,
    sort_order: list[tuple[str, str, str]] | None,
    zorder_by: list[str] | None,
    where: str | Any | None,
    branch: str | None,
    options: dict[str, Any] | None,
) -> RewriteResult:
    """Execute a `rewrite_data_files` action against `table`.

    Single-commit path. Partial-progress and OCC retry land in M4.
    """
    from pyiceberg.expressions import AlwaysTrue
    from pyiceberg.manifest import DataFileContent

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {SUPPORTED_STRATEGIES}, got {strategy!r}"
        )
    parsed_sort_order: list[tuple[str, bool, bool]] | None = None
    parsed_zorder_by: list[str] | None = None
    if strategy == "sort":
        parsed_sort_order = _parse_sort_order(sort_order, table)
    elif strategy == "zorder":
        parsed_zorder_by = _parse_zorder_columns(zorder_by, table)

    raw_options = options or {}
    normalized = _rust_iceberg.validate_options_py(raw_options)

    row_filter = where if where is not None else AlwaysTrue()
    scan_kwargs: dict[str, Any] = {"row_filter": row_filter}
    if branch is not None:
        scan_kwargs["snapshot_id"] = table.snapshot_for_ref(branch).snapshot_id  # type: ignore[union-attr]
    scan = table.scan(**scan_kwargs)
    plan_files = list(scan.plan_files())

    candidates: list[dict[str, Any]] = []
    plan_by_path: dict[str, Any] = {}
    eq_delete_files: list[str] = []
    for task in plan_files:
        path = task.file.file_path
        pos_deletes: list[str] = []
        has_eq = False
        for d in task.delete_files:
            if d.content == DataFileContent.POSITION_DELETES:
                pos_deletes.append(d.file_path)
            elif d.content == DataFileContent.EQUALITY_DELETES:
                has_eq = True
                eq_delete_files.append(d.file_path)
        candidates.append(
            {
                "path": path,
                "size_bytes": int(task.file.file_size_in_bytes),
                "partition_key": _stable_partition_key(task.file.partition),
                "partition_spec_id": int(task.file.spec_id),
                "positional_delete_paths": pos_deletes,
                "has_equality_deletes": has_eq,
            }
        )
        plan_by_path[path] = task

    if eq_delete_files:
        raise EqualityDeletesPresent(
            f"equality deletes present in files: {sorted(set(eq_delete_files))}"
        )

    current_spec_id = int(table.spec().spec_id)
    groups = _rust_iceberg.plan_file_groups_py(candidates, raw_options, current_spec_id)

    rewrite_id = _resolve_rewrite_id(
        table, branch, strategy, normalized, candidates, raw_options
    )
    cached = _lookup_idempotent_result(table, rewrite_id, strategy)
    if cached is not None:
        logger.info(
            "rewrite_data_files: idempotency hit on rewrite_id=%s; skipping", rewrite_id
        )
        return cached

    if not groups:
        return RewriteResult(
            strategy=strategy,
            rewritten_files=0,
            added_files=0,
            bytes_rewritten=0,
            bytes_added=0,
            removed_delete_files=0,
            failed_groups=0,
            commits=0,
            snapshot_ids=[],
            rewrite_id=rewrite_id,
        )

    outputs: list[_GroupOutput] = []
    for idx, group in enumerate(groups):
        outputs.append(
            _rewrite_group(
                table=table,
                group=group,
                group_index=idx,
                normalized_options=normalized,
                strategy=strategy,
                sort_order=parsed_sort_order,
                zorder_by=parsed_zorder_by,
            )
        )

    return _commit(
        table=table,
        outputs=outputs,
        plan_by_path=plan_by_path,
        rewrite_id=rewrite_id,
        strategy=strategy,
        normalized_options=normalized,
    )


@dataclass
class _GroupOutput:
    input_data_files: list[str]
    input_positional_delete_files: list[str]
    data_files: list[Any]  # pyiceberg.manifest.DataFile
    bytes_added: int
    bytes_rewritten: int


_ZORDER_SUPPORTED_TYPES = {
    "int",
    "long",
    "boolean",
    "float",
    "double",
    "date",
    "timestamp",
    "timestamptz",
    "string",
    "binary",
    "decimal",
}


def _parse_zorder_columns(
    zorder_by: list[str] | None,
    table: PyIcebergTable,
) -> list[str]:
    """Validate zorder_by columns and reject unsupported types (nested, uuid, fixed)."""
    if not zorder_by:
        raise ValueError("strategy='zorder' requires a non-empty zorder_by")
    schema = table.schema()
    by_name = {f.name: f for f in schema.fields}
    out: list[str] = []
    for c in zorder_by:
        if c not in by_name:
            raise ValueError(f"zorder column {c!r} not in table schema")
        type_str = str(by_name[c].field_type).lower()
        # Strip parameters: "decimal(10,2)" -> "decimal", "timestamptz" stays.
        bare = type_str.split("(")[0].strip()
        if bare not in _ZORDER_SUPPORTED_TYPES:
            raise ValueError(
                f"zorder column {c!r} has unsupported type {type_str!r}; "
                f"supported: {sorted(_ZORDER_SUPPORTED_TYPES)}"
            )
        if c == _ZORDER_KEY_COL:
            raise ValueError(
                f"column name {_ZORDER_KEY_COL!r} is reserved by the z-order rewrite"
            )
        out.append(c)
    return out


def _parse_sort_order(
    sort_order: list[tuple[str, str, str]] | None,
    table: PyIcebergTable,
) -> list[tuple[str, bool, bool]]:
    """Validate sort_order and return ``[(column_name, descending, nulls_first)]``."""
    if not sort_order:
        raise ValueError("strategy='sort' requires a non-empty sort_order")
    schema = table.schema()
    schema_names = {f.name for f in schema.fields}
    parsed: list[tuple[str, bool, bool]] = []
    for item in sort_order:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(
                f"sort_order entries must be (column, asc|desc, nulls-first|nulls-last); got {item!r}"
            )
        col, direction, null_order = item
        if col not in schema_names:
            raise ValueError(f"sort column {col!r} not in table schema")
        if direction not in _VALID_SORT_DIRECTIONS:
            raise ValueError(
                f"sort direction must be one of {_VALID_SORT_DIRECTIONS}, got {direction!r}"
            )
        if null_order not in _VALID_NULL_ORDERS:
            raise ValueError(
                f"null order must be one of {_VALID_NULL_ORDERS}, got {null_order!r}"
            )
        parsed.append((col, direction == "desc", null_order == "nulls-first"))
    return parsed


def _rewrite_group(
    *,
    table: PyIcebergTable,
    group: dict[str, Any],
    group_index: int,
    normalized_options: dict[str, Any],
    strategy: str,
    sort_order: list[tuple[str, bool, bool]] | None,
    zorder_by: list[str] | None,
) -> _GroupOutput:
    """Read this group's input files, apply strategy transform, write field-id-aware parquet.

    Sort and z-order share this code path with binpack — the only difference is the
    per-group transform applied before splitting into target-sized chunks.
    """
    from daft.expressions import col as col_expr
    from daft.expressions.expressions import ExpressionsProjection
    from daft.io.writer import IcebergWriter
    from daft.recordbatch import MicroPartition

    input_paths = [f["path"] for f in group["files"]]
    input_delete_paths_nested = [f["positional_delete_paths"] for f in group["files"]]
    flat_delete_paths = sorted({p for sub in input_delete_paths_nested for p in sub})
    bytes_rewritten = sum(int(f["size_bytes"]) for f in group["files"])

    target_size = int(normalized_options["target-file-size-bytes"])
    output_root = _group_output_root(table, group, group_index)

    arrow_table = _to_arrow_for_paths(table, input_paths)
    mp = MicroPartition.from_arrow(arrow_table)

    if strategy == "sort":
        assert sort_order is not None
        sort_exprs = ExpressionsProjection(
            [col_expr(name) for (name, _, _) in sort_order]
        )
        descending = [d for (_, d, _) in sort_order]
        nulls_first = [nf for (_, _, nf) in sort_order]
        mp = mp.sort(sort_exprs, descending=descending, nulls_first=nulls_first)
    elif strategy == "zorder":
        assert zorder_by is not None
        mp = _apply_zorder(mp, arrow_table, zorder_by, normalized_options)

    # Split into roughly target-size-aligned chunks. Uses input bytes as the size estimate;
    # output bytes track this closely when compression behavior matches the inputs. After a
    # sort, slicing the sorted MicroPartition contiguously preserves the within-file order.
    n_chunks = max(1, (bytes_rewritten + target_size - 1) // target_size)
    chunks = _split_micropartition(mp, int(n_chunks))

    spec = table.spec()
    schema = table.schema()
    properties = table.properties

    data_files: list[Any] = []
    bytes_added = 0
    for idx, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        writer = IcebergWriter(
            root_dir=output_root,
            file_idx=idx,
            schema=schema,
            properties=properties,
            partition_spec_id=spec.spec_id,
            partition_values=None,
            io_config=None,
        )
        writer.write(chunk)
        rb = writer.close()
        df_list = rb.to_pylist()
        for entry in df_list:
            df_ = entry.get("data_file") if isinstance(entry, dict) else entry
            if df_ is None:
                continue
            data_files.append(df_)
            bytes_added += int(getattr(df_, "file_size_in_bytes", 0))

    return _GroupOutput(
        input_data_files=input_paths,
        input_positional_delete_files=flat_delete_paths,
        data_files=data_files,
        bytes_added=bytes_added,
        bytes_rewritten=bytes_rewritten,
    )


def _apply_zorder(
    mp: Any,
    arrow_table: Any,
    zorder_by: list[str],
    normalized_options: dict[str, Any],
) -> Any:
    """Build the synthetic z-order key, sort by it, drop it.

    Returns a MicroPartition with the same schema as the input (key column projected away).
    """
    from daft.expressions import col as col_expr
    from daft.expressions.expressions import ExpressionsProjection
    from daft.recordbatch import MicroPartition

    var_len = int(normalized_options["var-length-contribution"])
    max_out = int(normalized_options["max-output-size"])
    key_array = _rust_iceberg.build_zorder_key_py(
        [arrow_table.column(c).combine_chunks() for c in zorder_by],
        var_len,
        max_out,
    )

    augmented_arrow = arrow_table.append_column(_ZORDER_KEY_COL, key_array)
    augmented = MicroPartition.from_arrow(augmented_arrow)
    sort_exprs = ExpressionsProjection([col_expr(_ZORDER_KEY_COL)])
    sorted_mp = augmented.sort(sort_exprs, descending=False, nulls_first=True)

    keep_cols = [c for c in arrow_table.column_names if c != _ZORDER_KEY_COL]
    return sorted_mp.eval_expression_list(
        ExpressionsProjection([col_expr(c) for c in keep_cols])
    )


def _split_micropartition(mp: Any, n: int) -> list[Any]:
    """Split a MicroPartition into roughly `n` evenly sized slices (row-wise)."""
    total = len(mp)
    if n <= 1 or total <= 1:
        return [mp]
    step = (total + n - 1) // n
    out = []
    for start in range(0, total, step):
        out.append(mp.slice(start, min(start + step, total)))
    return out


def _to_arrow_for_paths(table: PyIcebergTable, paths: list[str]):
    """Read selected files (with positional deletes applied) into a single Arrow table."""
    from pyiceberg.io.pyarrow import ArrowScan

    scan = table.scan()
    plan_files = list(scan.plan_files())
    wanted = {p for p in paths}
    tasks = [t for t in plan_files if t.file.file_path in wanted]
    if not tasks:
        raise ValueError(f"no plan_files matched: {paths!r}")
    arrow_scan = ArrowScan(
        table_metadata=table.metadata,
        io=table.io,
        projected_schema=scan.projection(),
        row_filter=scan.row_filter,
        case_sensitive=scan.case_sensitive,
    )
    return arrow_scan.to_table(tasks)


def _group_output_root(
    table: PyIcebergTable, group: dict[str, Any], group_index: int
) -> str:
    base = table.location().rstrip("/")
    suffix = uuid.uuid4().hex[:12]
    # Sit alongside existing data; an extra subdir keeps writes isolated.
    return f"{base}/data/__daft_rewrite__/{group_index}-{suffix}"


def _path_protocol(p: str) -> str | None:
    if "://" in p:
        return p.split("://", 1)[0]
    return None


def _read_parquet_metadata(path: str):
    from daft.dependencies import pq

    return pq.read_metadata(path)


def _stable_partition_key(record: Any) -> str:
    """Canonical JSON encoding of an Iceberg partition Record (sorted by field name)."""
    if record is None:
        return "{}"
    try:
        items = {}
        for name in dir(record):
            if name.startswith("_"):
                continue
            try:
                v = getattr(record, name)
            except Exception:
                continue
            if callable(v):
                continue
            items[name] = _json_safe(v)
        return json.dumps(items, sort_keys=True, default=str)
    except Exception:
        return json.dumps(str(record))


def _json_safe(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _record_from_partition_key(table: PyIcebergTable, partition_key: str):
    """Reconstruct an Iceberg partition Record from a candidate's `partition_key`.

    For unpartitioned tables this returns an empty Record. Partitioned tables look up
    the field order from the current spec and fill in values by name.
    """
    from pyiceberg.typedef import Record as IcebergRecord

    spec = table.spec()
    if not spec.fields:
        return IcebergRecord()
    payload = json.loads(partition_key) if partition_key else {}
    if not isinstance(payload, dict):
        return IcebergRecord()
    values = [payload.get(f.name) for f in spec.fields]
    try:
        return IcebergRecord(*values)
    except TypeError:
        return IcebergRecord(**{f.name: payload.get(f.name) for f in spec.fields})


def _resolve_rewrite_id(
    table: PyIcebergTable,
    branch: str | None,
    strategy: str,
    normalized_options: dict[str, Any],
    candidates: list[dict[str, Any]],
    raw_options: dict[str, Any],
) -> str:
    explicit = raw_options.get("rewrite-id")
    if explicit:
        return str(explicit)
    payload = {
        "table_uuid": str(table.metadata.table_uuid),
        "branch": branch or "main",
        "strategy": strategy,
        "options": {k: normalized_options[k] for k in sorted(normalized_options)},
        "files": sorted(c["path"] for c in candidates),
    }
    h = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return h[:16]


def _summary_as_dict(summary: Any) -> dict[str, str]:
    """pyiceberg's Summary is dict-like but resists ``dict(s)``; flatten via attribute access."""
    if summary is None:
        return {}
    out: dict[str, str] = {}
    op = getattr(summary, "operation", None)
    if op is not None:
        out["operation"] = str(op.value if hasattr(op, "value") else op)
    extra = getattr(summary, "additional_properties", None)
    if extra:
        out.update({str(k): str(v) for k, v in extra.items()})
    return out


def _lookup_idempotent_result(
    table: PyIcebergTable,
    rewrite_id: str,
    strategy: str,
) -> RewriteResult | None:
    """Reconstruct a `RewriteResult` from any prior snapshots tagged with `rewrite_id`.

    Under partial-progress, one logical rewrite spans N snapshots — we aggregate every
    match so a replay short-circuits even when the original run committed batched.
    Look-back window: last 50 snapshots (configurable in a follow-up).
    """
    snapshots = list(table.metadata.snapshots or [])
    matches: list[tuple[Any, dict[str, str]]] = []
    for snap in snapshots[-50:]:
        summary = _summary_as_dict(snap.summary)
        if summary.get(SNAPSHOT_PROP_REWRITE_ID) == rewrite_id:
            matches.append((snap, summary))
    if not matches:
        return None
    rewritten_total = sum(int(s.get(SNAPSHOT_PROP_INPUT_FILES, 0)) for _, s in matches)
    added_total = sum(int(s.get(SNAPSHOT_PROP_OUTPUT_FILES, 0)) for _, s in matches)
    strat = matches[-1][1].get(SNAPSHOT_PROP_STRATEGY, strategy)
    return RewriteResult(
        strategy=strat,
        rewritten_files=rewritten_total,
        added_files=added_total,
        # bytes_* and removed_delete_files are intentionally not persisted to the
        # snapshot summary (would balloon it for large rewrites); replay reconstructs
        # only what the summary carries.
        bytes_rewritten=0,
        bytes_added=0,
        removed_delete_files=0,
        failed_groups=0,
        commits=len(matches),
        snapshot_ids=[int(s.snapshot_id) for s, _ in matches],
        rewrite_id=rewrite_id,
    )


def _find_batch_snapshot(
    table: PyIcebergTable, rewrite_id: str, batch_label: str
) -> Any | None:
    """Locate a snapshot already committed for `(rewrite_id, batch_label)`, if any."""
    snapshots = list(table.metadata.snapshots or [])
    for snap in reversed(snapshots[-50:]):
        summary = _summary_as_dict(snap.summary)
        if (
            summary.get(SNAPSHOT_PROP_REWRITE_ID) == rewrite_id
            and summary.get(SNAPSHOT_PROP_BATCH) == batch_label
        ):
            return snap
    return None


def _commit(
    *,
    table: PyIcebergTable,
    outputs: list[_GroupOutput],
    plan_by_path: dict[str, Any],
    rewrite_id: str,
    strategy: str,
    normalized_options: dict[str, Any],
) -> RewriteResult:
    """Dispatch single-commit or partial-progress commit based on options."""
    if not outputs:
        return RewriteResult(
            strategy=strategy,
            rewritten_files=0,
            added_files=0,
            bytes_rewritten=0,
            bytes_added=0,
            removed_delete_files=0,
            failed_groups=0,
            commits=0,
            snapshot_ids=[],
            rewrite_id=rewrite_id,
        )
    if normalized_options.get("partial-progress.enabled"):
        return _commit_partial(
            table=table,
            outputs=outputs,
            plan_by_path=plan_by_path,
            rewrite_id=rewrite_id,
            strategy=strategy,
            max_commits=int(normalized_options["partial-progress.max-commits"]),
        )
    return _commit_single(
        table=table,
        outputs=outputs,
        plan_by_path=plan_by_path,
        rewrite_id=rewrite_id,
        strategy=strategy,
    )


def _commit_single(
    *,
    table: PyIcebergTable,
    outputs: list[_GroupOutput],
    plan_by_path: dict[str, Any],
    rewrite_id: str,
    strategy: str,
) -> RewriteResult:
    """Atomic single-transaction commit with OCC retry. Re-raises on exhaustion."""
    result, err = _commit_batch(
        table=table,
        batch=outputs,
        plan_by_path=plan_by_path,
        rewrite_id=rewrite_id,
        strategy=strategy,
        batch_label=None,
    )
    if result is None:
        assert err is not None
        raise err
    return result


def _commit_partial(
    *,
    table: PyIcebergTable,
    outputs: list[_GroupOutput],
    plan_by_path: dict[str, Any],
    rewrite_id: str,
    strategy: str,
    max_commits: int,
) -> RewriteResult:
    """Split outputs into ≤ max_commits contiguous batches and commit each independently.

    A batch that exhausts its retry budget or hits a `RewriteConflict` is logged with
    its orphan output paths and counted in `failed_groups`; remaining batches still
    commit. Group order is preserved (planner already sorted by `rewrite-job-order`).
    """
    n_batches = min(max(1, int(max_commits)), len(outputs))
    chunk_size = (len(outputs) + n_batches - 1) // n_batches
    batches = [outputs[i : i + chunk_size] for i in range(0, len(outputs), chunk_size)]
    n_actual = len(batches)

    agg_rewritten = 0
    agg_added = 0
    agg_in_bytes = 0
    agg_out_bytes = 0
    agg_removed_deletes = 0
    failed_groups = 0
    snapshot_ids: list[int] = []

    for idx, batch in enumerate(batches):
        label = f"{idx + 1}/{n_actual}"
        result, err = _commit_batch(
            table=table,
            batch=batch,
            plan_by_path=plan_by_path,
            rewrite_id=rewrite_id,
            strategy=strategy,
            batch_label=label,
        )
        if result is None:
            orphan_paths = _orphan_output_paths(batch)
            logger.warning(
                "rewrite_data_files: batch %s of %s failed after retries (%s); "
                "orphan outputs: %s",
                label,
                n_actual,
                type(err).__name__ if err else "unknown",
                orphan_paths,
            )
            failed_groups += len(batch)
            continue
        agg_rewritten += result.rewritten_files
        agg_added += result.added_files
        agg_in_bytes += result.bytes_rewritten
        agg_out_bytes += result.bytes_added
        agg_removed_deletes += result.removed_delete_files
        snapshot_ids.extend(result.snapshot_ids)

    return RewriteResult(
        strategy=strategy,
        rewritten_files=agg_rewritten,
        added_files=agg_added,
        bytes_rewritten=agg_in_bytes,
        bytes_added=agg_out_bytes,
        removed_delete_files=agg_removed_deletes,
        failed_groups=failed_groups,
        commits=len(snapshot_ids),
        snapshot_ids=snapshot_ids,
        rewrite_id=rewrite_id,
    )


def _commit_batch(
    *,
    table: PyIcebergTable,
    batch: list[_GroupOutput],
    plan_by_path: dict[str, Any],
    rewrite_id: str,
    strategy: str,
    batch_label: str | None,
) -> tuple[RewriteResult | None, Exception | None]:
    """Atomically overwrite this batch's inputs with its outputs; retry on OCC conflict.

    Returns `(result, None)` on success, `(None, last_exception)` after exhausting
    `_COMMIT_MAX_ATTEMPTS`. A `RewriteConflict` (input files vanished mid-retry) is
    raised inline — partial-progress callers must catch it.
    """
    from pyiceberg.exceptions import CommitFailedException

    all_data_files = [df_ for o in batch for df_ in o.data_files]
    input_paths = sorted({p for o in batch for p in o.input_data_files})
    delete_files_consumed = sorted(
        {p for o in batch for p in o.input_positional_delete_files}
    )
    total_in = sum(o.bytes_rewritten for o in batch)
    total_out = sum(o.bytes_added for o in batch)

    snapshot_props: dict[str, str] = {
        SNAPSHOT_PROP_REWRITE_ID: rewrite_id,
        SNAPSHOT_PROP_STRATEGY: strategy,
        SNAPSHOT_PROP_INPUT_FILES: str(len(input_paths)),
        SNAPSHOT_PROP_OUTPUT_FILES: str(len(all_data_files)),
    }
    if batch_label is not None:
        snapshot_props[SNAPSHOT_PROP_BATCH] = batch_label

    def _success_result(snapshot_id: int) -> RewriteResult:
        return RewriteResult(
            strategy=strategy,
            rewritten_files=len(input_paths),
            added_files=len(all_data_files),
            bytes_rewritten=total_in,
            bytes_added=total_out,
            removed_delete_files=len(delete_files_consumed),
            failed_groups=0,
            commits=1,
            snapshot_ids=[int(snapshot_id)] if snapshot_id else [],
            rewrite_id=rewrite_id,
        )

    last_err: Exception | None = None
    for attempt in range(_COMMIT_MAX_ATTEMPTS):
        table.refresh()

        # If a prior attempt's commit actually landed despite a thrown exception,
        # detect via the snapshot summary and short-circuit. Scope by batch_label
        # under partial-progress so we don't confuse our batch with a sibling.
        if batch_label is None:
            cached = _lookup_idempotent_result(table, rewrite_id, strategy)
            if cached is not None and cached.commits >= 1:
                return cached, None
        else:
            existing = _find_batch_snapshot(table, rewrite_id, batch_label)
            if existing is not None:
                return _success_result(int(existing.snapshot_id)), None

        # After at least one failure, ensure the input files still exist. If a
        # concurrent rewrite removed any, our outputs would orphan stale rows.
        if attempt > 0:
            live = {t.file.file_path for t in table.scan().plan_files()}
            missing = [p for p in input_paths if p not in live]
            if missing:
                orphans = _orphan_output_paths(batch)
                raise RewriteConflict(
                    f"input files vanished during retry: {missing!r}; "
                    f"orphan outputs: {orphans!r}"
                )

        try:
            tx = table.transaction()
            update = tx.update_snapshot(snapshot_properties=snapshot_props)
            with update.overwrite() as ow:
                for p in input_paths:
                    task = plan_by_path[p]
                    ow.delete_data_file(task.file)
                for df_ in all_data_files:
                    ow.append_data_file(df_)
            tx.commit_transaction()
            table.refresh()
            snap = table.current_snapshot()
            snapshot_id = int(snap.snapshot_id) if snap else 0
            return _success_result(snapshot_id), None
        except CommitFailedException as e:
            last_err = e
            if attempt < _COMMIT_MAX_ATTEMPTS - 1:
                time.sleep(_COMMIT_BACKOFF_BASE_SECONDS * (2**attempt))
            continue

    return None, last_err


def _orphan_output_paths(batch: list[_GroupOutput]) -> list[str]:
    """Best-effort extraction of output file paths from a batch's `DataFile` objects."""
    out: list[str] = []
    for o in batch:
        for df_ in o.data_files:
            path = getattr(df_, "file_path", None)
            if path:
                out.append(str(path))
    return out
