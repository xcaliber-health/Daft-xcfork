"""Compact or re-cluster data files: enumerate candidates, plan groups, read+write outputs, commit atomically."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from daft.daft import _iceberg as _rust_iceberg
from daft.io.iceberg._common import (
    CommitRetryExhausted,
    commit_with_retry,
)

if TYPE_CHECKING:
    from pyiceberg.table import Table as PyIcebergTable

logger = logging.getLogger(__name__)

# Equality deletes are rejected up front; users must apply them before compacting.
EqualityDeletesPresent = _rust_iceberg.EqualityDeletesPresentError


class RewriteConflict(RuntimeError):
    """Concurrent writer removed input files mid-retry; outputs are orphaned."""


SUPPORTED_STRATEGIES = ("binpack", "sort", "zorder")
_VALID_SORT_DIRECTIONS = {"asc", "desc"}
_VALID_NULL_ORDERS = {"nulls-first", "nulls-last"}
_ZORDER_KEY_COL = "__daft_zorder_key__"
SNAPSHOT_PROP_REWRITE_ID = "daft.rewrite-id"
SNAPSHOT_PROP_STRATEGY = "daft.rewrite-strategy"
SNAPSHOT_PROP_INPUT_FILES = "daft.rewrite-input-files"
SNAPSHOT_PROP_OUTPUT_FILES = "daft.rewrite-output-files"
SNAPSHOT_PROP_BATCH = "daft.rewrite-batch"
SNAPSHOT_PROP_MAINTENANCE_OP = "daft.maintenance.op"
SNAPSHOT_PROP_MAINTENANCE_OP_VALUE = "rewrite-data-files"

WRITE_TARGET_FILE_SIZE_BYTES_KEY = "write.target-file-size-bytes"

_STARTING_SEQ_NUMBER_WARNED = False


def _warn_unused_starting_sequence_number_once() -> None:
    global _STARTING_SEQ_NUMBER_WARNED
    if _STARTING_SEQ_NUMBER_WARNED:
        return
    _STARTING_SEQ_NUMBER_WARNED = True
    import warnings

    warnings.warn(
        "rewrite_data_files: option `use-starting-sequence-number` is accepted but "
        "currently a no-op; default sequencing applies.",
        stacklevel=3,
    )


@dataclass(frozen=True)
class RewriteResult:
    """Summary of a rewrite_data_files invocation.

    Parameters
    ----------
    strategy
        The strategy applied: ``"binpack"``, ``"sort"``, or ``"zorder"``.
    rewritten_files
        Number of input data files removed by the rewrite.
    added_files
        Number of output data files written.
    bytes_rewritten
        Total size in bytes of the removed data files.
    bytes_added
        Total size in bytes of the written data files.
    removed_delete_files
        Number of positional delete files consumed during read, plus any deletes
        dropped by ``remove-dangling-deletes`` post-processing.
    failed_groups
        Number of file groups whose batched commit exhausted retries. Non-zero
        only when ``partial-progress.enabled=true``.
    commits
        Number of snapshots produced. Always ``1`` in atomic mode; up to
        ``partial-progress.max-commits`` otherwise.
    snapshot_ids
        Snapshot IDs created by this call, in commit order.
    rewrite_id
        Stable identifier used for idempotent replay.
    failed_data_files
        Same as ``failed_groups`` but counted at file granularity.
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
    failed_data_files: int = 0


class RewriteFailedException(RuntimeError):
    """Rewrite could not make forward progress."""


def run(
    table: PyIcebergTable,
    strategy: str,
    sort_order: list[tuple[str, str, str]] | None,
    zorder_by: list[str] | None,
    where: str | Any | None,
    branch: str | None,
    options: dict[str, Any] | None,
) -> RewriteResult:
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

    raw_options = dict(options or {})
    # Fall back to the table property when the caller did not pass an explicit
    # target file size; this lets writers and the rewriter agree on output size
    # without restating it at every callsite.
    if "target-file-size-bytes" not in raw_options:
        prop = table.properties.get(WRITE_TARGET_FILE_SIZE_BYTES_KEY)
        if prop is not None:
            raw_options["target-file-size-bytes"] = int(prop)
    normalized = _rust_iceberg.validate_options_py(raw_options)

    if "use-starting-sequence-number" in raw_options:
        _warn_unused_starting_sequence_number_once()

    row_filter = where if where is not None else AlwaysTrue()
    scan_kwargs: dict[str, Any] = {"row_filter": row_filter}
    if branch is not None:
        starting_snapshot = table.snapshot_by_name(branch)
    else:
        starting_snapshot = table.current_snapshot()
    starting_snapshot_id: int | None = (
        int(starting_snapshot.snapshot_id) if starting_snapshot is not None else None
    )
    if starting_snapshot_id is not None:
        scan_kwargs["snapshot_id"] = starting_snapshot_id
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

    max_concurrent = max(
        1, int(normalized.get("max-concurrent-file-group-rewrites", 1))
    )
    outputs = _rewrite_groups_concurrent(
        table=table,
        groups=groups,
        normalized_options=normalized,
        strategy=strategy,
        sort_order=parsed_sort_order,
        zorder_by=parsed_zorder_by,
        max_workers=max_concurrent,
    )

    result = _commit(
        table=table,
        outputs=outputs,
        plan_by_path=plan_by_path,
        rewrite_id=rewrite_id,
        strategy=strategy,
        normalized_options=normalized,
        branch=branch,
        starting_snapshot_id=starting_snapshot_id,
    )

    if normalized.get("remove-dangling-deletes"):
        removed = _remove_dangling_deletes(table, branch=branch)
        if removed:
            result = _augment_result_with_dangling(result, removed)
    return result


def _rewrite_groups_concurrent(
    *,
    table: PyIcebergTable,
    groups: list[dict[str, Any]],
    normalized_options: dict[str, Any],
    strategy: str,
    sort_order: list[tuple[str, bool, bool]] | None,
    zorder_by: list[str] | None,
    max_workers: int,
) -> list[_GroupOutput]:
    from concurrent.futures import ThreadPoolExecutor

    if max_workers <= 1 or len(groups) <= 1:
        return [
            _rewrite_group(
                table=table,
                group=g,
                normalized_options=normalized_options,
                strategy=strategy,
                sort_order=sort_order,
                zorder_by=zorder_by,
            )
            for g in groups
        ]

    def _one(g: dict[str, Any]) -> _GroupOutput:
        return _rewrite_group(
            table=table,
            group=g,
            normalized_options=normalized_options,
            strategy=strategy,
            sort_order=sort_order,
            zorder_by=zorder_by,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        outputs = list(pool.map(_one, groups))
    return outputs


def _augment_result_with_dangling(
    result: RewriteResult, removed_delete_files: int
) -> RewriteResult:
    return RewriteResult(
        strategy=result.strategy,
        rewritten_files=result.rewritten_files,
        added_files=result.added_files,
        bytes_rewritten=result.bytes_rewritten,
        bytes_added=result.bytes_added,
        removed_delete_files=result.removed_delete_files + removed_delete_files,
        failed_groups=result.failed_groups,
        commits=result.commits,
        snapshot_ids=result.snapshot_ids,
        rewrite_id=result.rewrite_id,
        failed_data_files=result.failed_data_files,
    )


@dataclass
class _GroupOutput:
    input_data_files: list[str]
    input_positional_delete_files: list[str]
    data_files: list[Any]  # Iceberg DataFile
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
    normalized_options: dict[str, Any],
    strategy: str,
    sort_order: list[tuple[str, bool, bool]] | None,
    zorder_by: list[str] | None,
) -> _GroupOutput:
    from daft.expressions import col as col_expr
    from daft.expressions.expressions import ExpressionsProjection
    from daft.io.writer import IcebergWriter
    from daft.recordbatch import MicroPartition

    input_paths = [f["path"] for f in group["files"]]
    input_delete_paths_nested = [f["positional_delete_paths"] for f in group["files"]]
    flat_delete_paths = sorted({p for sub in input_delete_paths_nested for p in sub})
    bytes_rewritten = sum(int(f["size_bytes"]) for f in group["files"])

    target_size = int(normalized_options["target-file-size-bytes"])
    compression_factor = float(normalized_options.get("compression-factor", 1.0))
    output_root = _group_output_root(table)

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

    estimated_output_bytes = max(1, int(bytes_rewritten * compression_factor))
    n_chunks = max(1, (estimated_output_bytes + target_size - 1) // target_size)
    chunks = _split_micropartition(mp, int(n_chunks))

    output_spec_id = int(group["output_spec_id"])
    output_spec = table.specs()[output_spec_id]
    schema = table.schema()
    properties = table.properties

    partition_values_rb = _build_partition_values_recordbatch(mp, output_spec, schema)

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
            partition_spec_id=output_spec_id,
            partition_values=partition_values_rb,
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


def _build_partition_values_recordbatch(
    mp: Any,
    output_spec: Any,
    schema: Any,
) -> Any | None:
    from daft.expressions.expressions import ExpressionsProjection
    from daft.io.iceberg.iceberg_write import partition_field_to_expr
    from daft.recordbatch import RecordBatch as DaftRecordBatch

    if not getattr(output_spec, "fields", None):
        return None
    if len(mp) == 0:
        return None
    # Group is partition-homogeneous; one row is representative of the whole group.
    head_mp = mp.slice(0, 1)
    proj = ExpressionsProjection(
        [partition_field_to_expr(f, schema) for f in output_spec.fields]
    )
    transformed = head_mp.eval_expression_list(proj)
    return DaftRecordBatch.from_arrow_table(transformed.to_arrow())


def _apply_zorder(
    mp: Any,
    arrow_table: Any,
    zorder_by: list[str],
    normalized_options: dict[str, Any],
) -> Any:
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
    total = len(mp)
    if n <= 1 or total <= 1:
        return [mp]
    step = (total + n - 1) // n
    out = []
    for start in range(0, total, step):
        out.append(mp.slice(start, min(start + step, total)))
    return out


# Use a field-id-aware Arrow scan so schema evolution resolves correctly and
# positional delete files are applied during read.
def _to_arrow_for_paths(table: PyIcebergTable, paths: list[str]):
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


def _group_output_root(table: PyIcebergTable) -> str:
    return f"{table.location().rstrip('/')}/data"


# Partition records are positional with no named attrs; iterate the tuple values.
def _stable_partition_key(record: Any) -> str:
    if record is None:
        return "[]"
    try:
        values = [_json_safe(v) for v in tuple(record)]
        return json.dumps(values, default=str)
    except TypeError:
        return json.dumps(str(record))


def _json_safe(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


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
    snapshots = list(table.metadata.snapshots or [])
    matches: list[tuple[Any, dict[str, str]]] = []
    # Last 50 snapshots is a pragmatic window covering even multi-batch partial-progress runs.
    for snap in snapshots[-50:]:
        summary = _summary_as_dict(snap.summary)
        if summary.get(SNAPSHOT_PROP_REWRITE_ID) == rewrite_id:
            matches.append((snap, summary))
    if not matches:
        return None
    rewritten_total = sum(int(s.get(SNAPSHOT_PROP_INPUT_FILES, 0)) for _, s in matches)
    added_total = sum(int(s.get(SNAPSHOT_PROP_OUTPUT_FILES, 0)) for _, s in matches)
    bytes_rewritten_total = sum(
        int(s.get("removed-files-size", 0)) for _, s in matches
    )
    bytes_added_total = sum(int(s.get("added-files-size", 0)) for _, s in matches)
    strat = matches[-1][1].get(SNAPSHOT_PROP_STRATEGY, strategy)
    return RewriteResult(
        strategy=strat,
        rewritten_files=rewritten_total,
        added_files=added_total,
        bytes_rewritten=bytes_rewritten_total,
        bytes_added=bytes_added_total,
        removed_delete_files=0,
        failed_groups=0,
        commits=len(matches),
        snapshot_ids=[int(s.snapshot_id) for s, _ in matches],
        rewrite_id=rewrite_id,
    )


def _find_batch_snapshot(
    table: PyIcebergTable, rewrite_id: str, batch_label: str
) -> Any | None:
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
    branch: str | None,
    starting_snapshot_id: int | None,
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
            max_failed_commits=int(
                normalized_options.get(
                    "partial-progress.max-failed-commits",
                    normalized_options["partial-progress.max-commits"],
                )
            ),
            branch=branch,
            starting_snapshot_id=starting_snapshot_id,
        )
    return _commit_single(
        table=table,
        outputs=outputs,
        plan_by_path=plan_by_path,
        rewrite_id=rewrite_id,
        strategy=strategy,
        branch=branch,
        starting_snapshot_id=starting_snapshot_id,
    )


def _commit_single(
    *,
    table: PyIcebergTable,
    outputs: list[_GroupOutput],
    plan_by_path: dict[str, Any],
    rewrite_id: str,
    strategy: str,
    branch: str | None,
    starting_snapshot_id: int | None,
) -> RewriteResult:
    result, err = _commit_batch(
        table=table,
        batch=outputs,
        plan_by_path=plan_by_path,
        rewrite_id=rewrite_id,
        strategy=strategy,
        batch_label=None,
        branch=branch,
        starting_snapshot_id=starting_snapshot_id,
    )
    if result is None:
        assert err is not None
        if isinstance(err, CommitRetryExhausted):
            raise RewriteFailedException(
                "rewrite_data_files: atomic commit could not land within the "
                "retry budget. To tolerate concurrent writers, set "
                "options={'partial-progress.enabled': True}."
            ) from err
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
    max_failed_commits: int,
    branch: str | None,
    starting_snapshot_id: int | None,
) -> RewriteResult:
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
    failed_data_files = 0
    failed_batches = 0
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
            branch=branch,
            starting_snapshot_id=starting_snapshot_id,
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
            failed_data_files += sum(len(o.input_data_files) for o in batch)
            failed_batches += 1
            continue
        agg_rewritten += result.rewritten_files
        agg_added += result.added_files
        agg_in_bytes += result.bytes_rewritten
        agg_out_bytes += result.bytes_added
        agg_removed_deletes += result.removed_delete_files
        snapshot_ids.extend(result.snapshot_ids)

    if failed_batches > max_failed_commits:
        raise RewriteFailedException(
            f"rewrite_data_files: {failed_batches} of {n_actual} batches failed "
            f"(threshold partial-progress.max-failed-commits={max_failed_commits}). "
            f"{len(snapshot_ids)} commit(s) landed; orphan outputs may need cleanup."
        )

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
        failed_data_files=failed_data_files,
    )


def _commit_batch(
    *,
    table: PyIcebergTable,
    batch: list[_GroupOutput],
    plan_by_path: dict[str, Any],
    rewrite_id: str,
    strategy: str,
    batch_label: str | None,
    branch: str | None,
    starting_snapshot_id: int | None,
) -> tuple[RewriteResult | None, Exception | None]:
    all_data_files = [df_ for o in batch for df_ in o.data_files]
    input_paths = sorted({p for o in batch for p in o.input_data_files})
    delete_files_consumed = sorted(
        {p for o in batch for p in o.input_positional_delete_files}
    )
    total_in = sum(o.bytes_rewritten for o in batch)
    total_out = sum(o.bytes_added for o in batch)
    touched_partitions = {
        _stable_partition_key(plan_by_path[p].file.partition) for p in input_paths
    }

    snapshot_props: dict[str, str] = {
        SNAPSHOT_PROP_MAINTENANCE_OP: SNAPSHOT_PROP_MAINTENANCE_OP_VALUE,
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

    def _check_idempotent_replay(t: PyIcebergTable) -> RewriteResult | None:
        if batch_label is None:
            cached = _lookup_idempotent_result(t, rewrite_id, strategy)
            if cached is not None and cached.commits >= 1:
                return cached
            return None
        existing = _find_batch_snapshot(t, rewrite_id, batch_label)
        if existing is not None:
            return _success_result(int(existing.snapshot_id))
        return None

    def _attempt(_: int) -> RewriteResult:
        table.refresh()
        cached = _check_idempotent_replay(table)
        if cached is not None:
            return cached
        _validate_no_overlap(
            table,
            starting_snapshot_id=starting_snapshot_id,
            input_paths=input_paths,
            touched_partitions=touched_partitions,
            batch=batch,
            rewrite_id=rewrite_id,
        )
        tx = table.transaction()
        update_kwargs: dict[str, Any] = {"snapshot_properties": snapshot_props}
        if branch is not None:
            update_kwargs["branch"] = branch
        update = tx.update_snapshot(**update_kwargs)
        with update.overwrite() as ow:
            for p in input_paths:
                task = plan_by_path[p]
                ow.delete_data_file(task.file)
            for df_ in all_data_files:
                ow.append_data_file(df_)
        tx.commit_transaction()
        table.refresh()
        snap = (
            table.snapshot_by_name(branch)
            if branch is not None
            else table.current_snapshot()
        )
        snapshot_id = int(snap.snapshot_id) if snap else 0
        return _success_result(snapshot_id)

    def _on_conflict(t: PyIcebergTable) -> RewriteResult | None:
        return _check_idempotent_replay(t)

    try:
        return (
            commit_with_retry(
                table,
                _attempt,
                op_name="rewrite_data_files",
                on_conflict=_on_conflict,
            ),
            None,
        )
    except CommitRetryExhausted as exc:
        return None, exc
    except RewriteConflict as exc:
        return None, exc


def _validate_no_overlap(
    table: PyIcebergTable,
    *,
    starting_snapshot_id: int | None,
    input_paths: list[str],
    touched_partitions: set[Any],
    batch: list[_GroupOutput],
    rewrite_id: str,
) -> None:
    """Reject the commit if foreign writes since the plan snapshot affect this batch.

    Raises :class:`RewriteConflict` when either (a) one of this batch's input
    files is no longer reachable from the current head, or (b) a foreign
    snapshot committed after ``starting_snapshot_id`` added a data file in a
    partition this batch is rewriting. Snapshots produced by the same rewrite
    (matched by ``daft.rewrite-id``) are excluded so partial-progress batches
    do not collide with their own predecessors.
    """
    head = table.current_snapshot()
    if head is None or starting_snapshot_id is None or int(head.snapshot_id) == int(starting_snapshot_id):
        _raise_if_inputs_vanished(table, input_paths, batch)
        return

    ancestry: list[Any] = []
    snap = head
    visited: set[int] = set()
    while snap is not None and int(snap.snapshot_id) != int(starting_snapshot_id):
        sid = int(snap.snapshot_id)
        if sid in visited:
            break
        visited.add(sid)
        ancestry.append(snap)
        parent_id = getattr(snap, "parent_snapshot_id", None)
        snap = table.metadata.snapshot_by_id(parent_id) if parent_id is not None else None

    for s in ancestry:
        if _snapshot_rewrite_id(s) == rewrite_id:
            continue
        added = _added_data_files(s, table)
        for df_ in added:
            partition_key = _stable_partition_key(df_.partition)
            if partition_key in touched_partitions:
                orphans = _orphan_output_paths(batch)
                raise RewriteConflict(
                    f"snapshot {int(s.snapshot_id)} added a data file in "
                    f"partition {partition_key!r} after the rewrite plan was "
                    f"taken; orphan outputs: {orphans!r}"
                )

    _raise_if_inputs_vanished(table, input_paths, batch)


def _raise_if_inputs_vanished(
    table: PyIcebergTable,
    input_paths: list[str],
    batch: list[_GroupOutput],
) -> None:
    live = {t.file.file_path for t in table.scan().plan_files()}
    missing = [p for p in input_paths if p not in live]
    if missing:
        orphans = _orphan_output_paths(batch)
        raise RewriteConflict(
            f"input files vanished before commit: {missing!r}; "
            f"orphan outputs: {orphans!r}"
        )


def _snapshot_rewrite_id(snapshot: Any) -> str | None:
    summary = _summary_as_dict(snapshot.summary)
    return summary.get(SNAPSHOT_PROP_REWRITE_ID)


def _added_data_files(snapshot: Any, table: PyIcebergTable) -> list[Any]:
    """Return data files added by ``snapshot`` (status ADDED).

    A manifest's ``added_snapshot_id`` identifies the single snapshot that
    contributed new entries to it. Manifests with a different
    ``added_snapshot_id`` cannot contain ADDED entries for the snapshot we
    are inspecting, so we skip them before performing any per-entry I/O.
    """
    from pyiceberg.manifest import ManifestEntryStatus

    out: list[Any] = []
    try:
        manifests = snapshot.manifests(table.io)
    except Exception:
        return out
    sid = int(snapshot.snapshot_id)
    for m in manifests:
        if int(getattr(m, "added_snapshot_id", -1)) != sid:
            continue
        try:
            entries = m.fetch_manifest_entry(table.io, discard_deleted=False)
        except Exception:
            continue
        for entry in entries:
            if int(entry.status) == int(ManifestEntryStatus.ADDED):
                out.append(entry.data_file)
    return out


def _remove_dangling_deletes(table: PyIcebergTable, branch: str | None) -> int:
    """Drop delete files whose sequence number is at or below the partition's
    minimum data-file sequence number.

    A delete with no live data file at or after its sequence number can never
    apply to anything, so removing it is safe. Commits a single snapshot. Returns
    the number of delete files removed.
    """
    from pyiceberg.manifest import DataFileContent

    table.refresh()
    snap = (
        table.snapshot_by_name(branch)
        if branch is not None
        else table.current_snapshot()
    )
    if snap is None:
        return 0

    min_data_seq: dict[tuple[int, str], int] = {}
    delete_entries: dict[tuple[int, str], list[tuple[Any, int]]] = {}
    for manifest in snap.manifests(table.io):
        for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
            data_file = entry.data_file
            seq = entry.sequence_number if entry.sequence_number is not None else 0
            key = (
                int(data_file.spec_id),
                _stable_partition_key(data_file.partition),
            )
            if data_file.content == DataFileContent.DATA:
                cur = min_data_seq.get(key)
                if cur is None or seq < cur:
                    min_data_seq[key] = seq
            else:
                delete_entries.setdefault(key, []).append((data_file, seq))

    to_remove: list[Any] = []
    for key, entries in delete_entries.items():
        min_seq = min_data_seq.get(key)
        for df_, seq in entries:
            # min_seq is None when the partition holds only delete files.
            if min_seq is None or seq <= min_seq:
                to_remove.append(df_)

    if not to_remove:
        return 0

    tx = table.transaction()
    update_kwargs: dict[str, Any] = {
        "snapshot_properties": {
            "daft.rewrite-dangling-deletes-removed": str(len(to_remove)),
        }
    }
    if branch is not None:
        update_kwargs["branch"] = branch
    with tx.update_snapshot(**update_kwargs).overwrite() as ow:
        for df_ in to_remove:
            ow.delete_data_file(df_)
    tx.commit_transaction()
    table.refresh()
    return len(to_remove)


def _orphan_output_paths(batch: list[_GroupOutput]) -> list[str]:
    out: list[str] = []
    for o in batch:
        for df_ in o.data_files:
            path = getattr(df_, "file_path", None)
            if path:
                out.append(str(path))
    return out
