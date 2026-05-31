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

    from daft.daft import IOConfig
    from daft.dataframe import DataFrame

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

# Conflict-isolation level for the commit-time overlap check.
#
# ``serializable`` (default) rejects the commit if any foreign snapshot added a
# data file in a partition the rewrite touches since the plan was taken.
# ``snapshot`` only rejects when one of the rewrite's own input files was
# removed, allowing concurrent appends of *new* files into the same partition to
# coexist with the rewrite. ``snapshot`` is safe only when no concurrent process
# deletes data from the touched partitions (e.g. an append-only writer).
CONFLICT_ISOLATION_KEY = "conflict-isolation"
CONFLICT_ISOLATION_SERIALIZABLE = "serializable"
CONFLICT_ISOLATION_SNAPSHOT = "snapshot"
_VALID_CONFLICT_ISOLATIONS = (
    CONFLICT_ISOLATION_SERIALIZABLE,
    CONFLICT_ISOLATION_SNAPSHOT,
)

def _parse_conflict_isolation(raw_options: dict[str, Any]) -> str:
    """Pop and validate the conflict-isolation option from ``raw_options``.

    The key is removed in place so it never reaches the option validator, which
    only recognizes planning options. Returns the validated isolation level,
    defaulting to ``serializable`` when the option is absent.
    """
    value = raw_options.pop(CONFLICT_ISOLATION_KEY, CONFLICT_ISOLATION_SERIALIZABLE)
    if value not in _VALID_CONFLICT_ISOLATIONS:
        raise ValueError(
            f"{CONFLICT_ISOLATION_KEY} must be one of {_VALID_CONFLICT_ISOLATIONS}, "
            f"got {value!r}"
        )
    return value


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
    # Pop orchestration-only options before the planner validator, which rejects
    # keys it does not recognize.
    conflict_isolation = _parse_conflict_isolation(raw_options)
    # Fall back to the table property when the caller did not pass an explicit
    # target file size; this lets writers and the rewriter agree on output size
    # without restating it at every callsite.
    if "target-file-size-bytes" not in raw_options:
        prop = table.properties.get(WRITE_TARGET_FILE_SIZE_BYTES_KEY)
        if prop is not None:
            raw_options["target-file-size-bytes"] = int(prop)
    normalized = _rust_iceberg.validate_options_py(raw_options)

    if "use-starting-sequence-number" in raw_options:
        raise ValueError(
            "rewrite_data_files: option `use-starting-sequence-number` is not "
            "supported. Output files are assigned sequence numbers at commit "
            "time; remove the option to proceed."
        )

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

    io_config = _io_config_for_table(table)
    outputs = _rewrite_groups(
        table=table,
        groups=groups,
        plan_by_path=plan_by_path,
        snapshot_id=starting_snapshot_id,
        io_config=io_config,
        normalized_options=normalized,
        strategy=strategy,
        sort_order=parsed_sort_order,
        zorder_by=parsed_zorder_by,
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
        conflict_isolation=conflict_isolation,
    )

    if normalized.get("remove-dangling-deletes"):
        removed = _remove_dangling_deletes(table, branch=branch)
        if removed:
            result = _augment_result_with_dangling(result, removed)
    return result


def _io_config_for_table(table: PyIcebergTable) -> IOConfig:
    """Resolve object-store access configuration for reading and writing a table.

    Prefers the configuration recorded on the table, falling back to the process
    default when none is set.
    """
    from daft.context import get_context
    from daft.io.iceberg._iceberg import (
        _convert_iceberg_file_io_properties_to_io_config,
    )

    io_config = _convert_iceberg_file_io_properties_to_io_config(table.io.properties)
    if io_config is not None:
        return io_config
    return get_context().daft_planning_config.default_io_config


def _rewrite_groups(
    *,
    table: PyIcebergTable,
    groups: list[dict[str, Any]],
    plan_by_path: dict[str, Any],
    snapshot_id: int | None,
    io_config: IOConfig,
    normalized_options: dict[str, Any],
    strategy: str,
    sort_order: list[tuple[str, bool, bool]] | None,
    zorder_by: list[str] | None,
) -> list[_GroupOutput]:
    """Rewrite each file group in turn through the streaming engine.

    Groups are processed sequentially: within a group the read, optional
    re-clustering, and write all stream through the execution engine, which
    bounds peak memory to the engine's budget rather than the group's full
    decompressed size. Running groups one at a time keeps that bound flat.
    """
    return [
        _rewrite_group(
            table=table,
            group=g,
            plan_by_path=plan_by_path,
            snapshot_id=snapshot_id,
            io_config=io_config,
            normalized_options=normalized_options,
            strategy=strategy,
            sort_order=sort_order,
            zorder_by=zorder_by,
        )
        for g in groups
    ]


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
    plan_by_path: dict[str, Any],
    snapshot_id: int | None,
    io_config: IOConfig,
    normalized_options: dict[str, Any],
    strategy: str,
    sort_order: list[tuple[str, bool, bool]] | None,
    zorder_by: list[str] | None,
) -> _GroupOutput:
    """Read one group's files, optionally re-cluster, and write target-sized outputs.

    The read, sort or z-order, and write all flow through the streaming execution
    engine, so peak memory is bounded by the engine's budget rather than the
    group's full decompressed size. The written files are returned as metadata for
    the caller to commit; nothing is committed here.
    """
    input_paths = [f["path"] for f in group["files"]]
    input_delete_paths_nested = [f["positional_delete_paths"] for f in group["files"]]
    flat_delete_paths = sorted({p for sub in input_delete_paths_nested for p in sub})
    bytes_rewritten = sum(int(f["size_bytes"]) for f in group["files"])

    target_size = int(normalized_options["target-file-size-bytes"])
    output_spec_id = int(group["output_spec_id"])

    df = _group_dataframe(
        table=table,
        input_paths=input_paths,
        plan_by_path=plan_by_path,
        snapshot_id=snapshot_id,
        io_config=io_config,
    )

    if strategy == "sort":
        assert sort_order is not None
        df = df.sort(
            [name for (name, _, _) in sort_order],
            desc=[descending for (_, descending, _) in sort_order],
            nulls_first=[nulls_first for (_, _, nulls_first) in sort_order],
        )
    elif strategy == "zorder":
        assert zorder_by is not None
        df = _apply_zorder(df, zorder_by, normalized_options)

    data_files = _collect_data_files(
        df=df,
        table=table,
        io_config=io_config,
        target_size=target_size,
        output_spec_id=output_spec_id,
    )
    bytes_added = sum(int(getattr(d, "file_size_in_bytes", 0)) for d in data_files)

    return _GroupOutput(
        input_data_files=input_paths,
        input_positional_delete_files=flat_delete_paths,
        data_files=data_files,
        bytes_added=bytes_added,
        bytes_rewritten=bytes_rewritten,
    )


def _group_dataframe(
    *,
    table: PyIcebergTable,
    input_paths: list[str],
    plan_by_path: dict[str, Any],
    snapshot_id: int | None,
    io_config: IOConfig,
) -> DataFrame:
    """Build a lazy frame over exactly the group's data files.

    The frame reads each file with the table's read schema (resolving field ids)
    and applies any positional delete files during the read, matching a normal
    table read but restricted to this group.
    """
    from daft import runners
    from daft.daft import ScanOperatorHandle, StorageConfig
    from daft.dataframe import DataFrame
    from daft.io.iceberg.iceberg_scan import IcebergFileGroupScanOperator
    from daft.logical.builder import LogicalPlanBuilder

    tasks = [plan_by_path[path] for path in input_paths]
    multithreaded_io = runners.get_or_create_runner().name != "ray"
    storage_config = StorageConfig(multithreaded_io, io_config)
    operator = IcebergFileGroupScanOperator(
        table, snapshot_id=snapshot_id, storage_config=storage_config, tasks=tasks
    )
    handle = ScanOperatorHandle.from_python_scan_operator(operator)
    builder = LogicalPlanBuilder.from_tabular_scan(scan_operator=handle)
    return DataFrame(builder)


def _apply_zorder(
    df: DataFrame,
    zorder_by: list[str],
    normalized_options: dict[str, Any],
) -> DataFrame:
    """Cluster rows along a space-filling curve over the given columns.

    A single ordered key is derived from the columns and the frame is sorted by
    it, then the key is dropped so the output schema matches the input. The key is
    computed as a streaming expression so no full copy of the group is held.
    """
    from daft.expressions import col as col_expr
    from daft.io.iceberg._zorder import zorder_key

    var_len = int(normalized_options["var-length-contribution"])
    max_out = int(normalized_options["max-output-size"])
    key = zorder_key([col_expr(c) for c in zorder_by], var_len, max_out)
    return (
        df.with_column(_ZORDER_KEY_COL, key)
        .sort(_ZORDER_KEY_COL, desc=False, nulls_first=True)
        .exclude(_ZORDER_KEY_COL)
    )


def _collect_data_files(
    *,
    df: DataFrame,
    table: PyIcebergTable,
    io_config: IOConfig,
    target_size: int,
    output_spec_id: int,
) -> list[Any]:
    """Write the frame's rows as target-sized data files and return their metadata.

    The write streams through the engine, rolling a new file each time the target
    size is reached and partitioning rows by the chosen spec. The destination is
    not committed; the returned descriptors are handed to the commit step.
    """
    from daft.dataframe import DataFrame

    write_builder = df._builder.write_iceberg(
        table,
        io_config,
        target_file_size_bytes=target_size,
        partition_spec_id=output_spec_id,
    )
    write_df = DataFrame(write_builder)
    write_df.collect()
    result = write_df.to_pydict()
    data_files = result.get("data_file", [])
    return [data_file for data_file in data_files if data_file is not None]


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
    conflict_isolation: str,
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
            conflict_isolation=conflict_isolation,
        )
    return _commit_single(
        table=table,
        outputs=outputs,
        plan_by_path=plan_by_path,
        rewrite_id=rewrite_id,
        strategy=strategy,
        branch=branch,
        starting_snapshot_id=starting_snapshot_id,
        conflict_isolation=conflict_isolation,
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
    conflict_isolation: str,
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
        conflict_isolation=conflict_isolation,
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
    conflict_isolation: str,
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
            conflict_isolation=conflict_isolation,
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
    conflict_isolation: str,
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
            isolation=conflict_isolation,
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
    isolation: str = CONFLICT_ISOLATION_SERIALIZABLE,
) -> None:
    """Reject the commit if foreign writes since the plan snapshot affect this batch.

    Under ``serializable`` isolation, raises :class:`RewriteConflict` when either
    (a) one of this batch's input files is no longer reachable from the current
    head, or (b) a foreign snapshot committed after ``starting_snapshot_id``
    added a data file in a partition this batch is rewriting. Under ``snapshot``
    isolation, only condition (a) is enforced: concurrent additions of new files
    to a touched partition are permitted, so the check is safe to use only when
    no concurrent process deletes data from those partitions. Snapshots produced
    by the same rewrite (matched by ``daft.rewrite-id``) are excluded so
    partial-progress batches do not collide with their own predecessors.
    """
    if isolation == CONFLICT_ISOLATION_SNAPSHOT:
        _raise_if_inputs_vanished(table, input_paths, batch)
        return

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
