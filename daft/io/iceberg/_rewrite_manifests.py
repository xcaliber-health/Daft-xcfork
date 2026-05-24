"""Repack live manifest entries into target-sized manifests.

Reads the current snapshot's manifests, retains only live entries, and writes
a fresh manifest set sized to ``manifest-target-size-bytes``. Output entries
are clustered by ``(spec_id, partition_key_tuple)`` so a partition-scoped
query reads at most one manifest per partition. Commits a single REPLACE
snapshot whose existing-manifests set is the new manifests plus any
untouched manifests; data and delete files are unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import uuid as _uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from daft.io.iceberg._common import (
    CommitRetryExhausted,
    commit_with_retry,
    validate_gc_enabled,
)

if TYPE_CHECKING:
    from pyiceberg.manifest import ManifestFile
    from pyiceberg.table import Table as PyIcebergTable

logger = logging.getLogger(__name__)


MANIFEST_TARGET_SIZE_KEY = "commit.manifest.target-size-bytes"
MANIFEST_MIN_COUNT_TO_MERGE_KEY = "commit.manifest.min-count-to-merge"
_DEFAULT_MANIFEST_TARGET_SIZE_BYTES = 8 * 1024 * 1024
_DEFAULT_MIN_COUNT_TO_MERGE = 100
_ROLL_FACTOR = 1.2

SNAPSHOT_PROP_REWRITE_ID = "daft.rewrite-id"
SNAPSHOT_PROP_REWRITE_STRATEGY = "daft.rewrite-strategy"
SNAPSHOT_PROP_REWRITE_STRATEGY_VALUE = "manifests"
SNAPSHOT_PROP_MAINTENANCE_OP = "daft.maintenance.op"
SNAPSHOT_PROP_MAINTENANCE_OP_VALUE = "rewrite-manifests"
SNAPSHOT_PROP_SPEC_ID = "daft.spec-id"
SNAPSHOT_PROP_INPUT_MANIFESTS = "daft.input-manifests"
SNAPSHOT_PROP_OUTPUT_MANIFESTS = "daft.output-manifests"
SNAPSHOT_PROP_INPUT_BYTES = "daft.input-manifest-bytes"
SNAPSHOT_PROP_OUTPUT_BYTES = "daft.output-manifest-bytes"


@dataclass(frozen=True)
class RewriteManifestsResult:
    """Summary of a rewrite_manifests invocation.

    Parameters
    ----------
    rewritten_manifests_count
        Number of input manifest files whose entries were repacked.
    added_manifests_count
        Number of output manifest files written.
    bytes_rewritten
        Total ``length`` of the input manifests that were repacked.
    bytes_added
        Total ``length`` of the output manifests.
    rewrite_id
        Stable identifier used for idempotent replay.
    snapshot_id
        The REPLACE snapshot's ID, or ``None`` when nothing needed rewriting.
    """

    rewritten_manifests_count: int = 0
    added_manifests_count: int = 0
    bytes_rewritten: int = 0
    bytes_added: int = 0
    rewrite_id: str = ""
    snapshot_id: int | None = None


class RewriteManifestsFailedException(RuntimeError):
    """Raised when rewrite_manifests cannot commit after the retry budget."""


def run(
    table: PyIcebergTable,
    *,
    spec_id: int | None = None,
    branch: str | None = None,
    use_caching: bool = False,
    options: dict[str, Any] | None = None,
) -> RewriteManifestsResult:
    from pyiceberg.table.snapshots import Operation

    opts = options or {}
    target_size_bytes = int(
        opts.get(
            "manifest-target-size-bytes",
            table.properties.get(
                MANIFEST_TARGET_SIZE_KEY, _DEFAULT_MANIFEST_TARGET_SIZE_BYTES
            ),
        )
    )
    min_count_to_merge = int(
        opts.get(
            "manifest-min-count-to-merge",
            table.properties.get(
                MANIFEST_MIN_COUNT_TO_MERGE_KEY, _DEFAULT_MIN_COUNT_TO_MERGE
            ),
        )
    )
    if target_size_bytes <= 0:
        raise ValueError(
            f"manifest-target-size-bytes must be > 0, got {target_size_bytes!r}"
        )

    _ = use_caching

    validate_gc_enabled(table)

    resolved_spec_id = _resolve_spec_id(table, spec_id)
    target_branch = _resolve_branch(table, branch)

    plan = _plan(
        table=table,
        spec_id=resolved_spec_id,
        target_branch=target_branch,
        target_size_bytes=target_size_bytes,
        min_count_to_merge=min_count_to_merge,
    )

    if plan.rewrite_id:
        cached = _lookup_idempotent_result(table, plan.rewrite_id)
        if cached is not None:
            return cached

    if not plan.matching_manifests or plan.no_op_reason is not None:
        logger.info(
            "rewrite_manifests: no-op (%s)",
            plan.no_op_reason or "no manifests matched the spec filter",
        )
        return RewriteManifestsResult(
            rewrite_id=plan.rewrite_id,
            snapshot_id=None,
        )

    plan_box = {"plan": plan}

    def _attempt(_: int) -> RewriteManifestsResult:
        return _commit_attempt(
            table=table,
            plan=plan_box["plan"],
            target_branch=target_branch,
            operation=Operation.REPLACE,
            target_size_bytes=target_size_bytes,
        )

    def _on_conflict(t: PyIcebergTable) -> RewriteManifestsResult | None:
        cached = _lookup_idempotent_result(t, plan_box["plan"].rewrite_id)
        if cached is not None:
            return cached
        new_plan = _plan(
            table=t,
            spec_id=resolved_spec_id,
            target_branch=target_branch,
            target_size_bytes=target_size_bytes,
            min_count_to_merge=min_count_to_merge,
        )
        if not new_plan.matching_manifests or new_plan.no_op_reason is not None:
            return RewriteManifestsResult(
                rewrite_id=new_plan.rewrite_id,
                snapshot_id=None,
            )
        plan_box["plan"] = new_plan
        return None

    try:
        return commit_with_retry(
            table,
            _attempt,
            op_name="rewrite_manifests",
            on_conflict=_on_conflict,
        )
    except CommitRetryExhausted as exc:
        raise RewriteManifestsFailedException(
            "rewrite_manifests: commit could not land within the retry budget"
        ) from exc


@dataclass
class _Plan:
    rewrite_id: str
    matching_manifests: list[ManifestFile]
    untouched_manifests: list[ManifestFile]
    bytes_rewritten: int
    total_live_entries: int
    no_op_reason: str | None = None


def _plan(
    *,
    table: PyIcebergTable,
    spec_id: int,
    target_branch: str,
    target_size_bytes: int,
    min_count_to_merge: int,
) -> _Plan:
    snapshot = table.metadata.snapshot_by_name(target_branch)
    if snapshot is None:
        return _Plan(
            rewrite_id="",
            matching_manifests=[],
            untouched_manifests=[],
            bytes_rewritten=0,
            total_live_entries=0,
            no_op_reason=f"branch {target_branch!r} has no current snapshot",
        )

    manifests = list(snapshot.manifests(table.io))
    matching: list[ManifestFile] = []
    untouched: list[ManifestFile] = []
    for m in manifests:
        if int(m.partition_spec_id) == int(spec_id):
            matching.append(m)
        else:
            untouched.append(m)

    bytes_rewritten = sum(int(m.manifest_length) for m in matching)
    total_live_entries = sum(
        int(m.added_files_count or 0) + int(m.existing_files_count or 0)
        for m in matching
    )

    rewrite_id = _resolve_rewrite_id(
        table=table,
        target_branch=target_branch,
        spec_id=spec_id,
        target_size_bytes=target_size_bytes,
        matching_manifest_paths=[m.manifest_path for m in matching],
    )

    no_op_reason: str | None = None
    if not matching:
        no_op_reason = "no manifests for spec_id"
    else:
        roll = int(target_size_bytes * _ROLL_FACTOR)
        max_size = max(int(m.manifest_length) for m in matching)
        expected_count = max(1, (bytes_rewritten + target_size_bytes - 1) // target_size_bytes)
        already_balanced = (
            abs(len(matching) - expected_count) <= 1 and max_size <= roll
        )
        below_min_count = len(matching) < min_count_to_merge and already_balanced
        if already_balanced and (below_min_count or len(matching) == expected_count):
            no_op_reason = (
                f"manifest layout already balanced "
                f"(count={len(matching)}, expected={expected_count}, max_size={max_size}, target={target_size_bytes})"
            )

    return _Plan(
        rewrite_id=rewrite_id,
        matching_manifests=matching,
        untouched_manifests=untouched,
        bytes_rewritten=bytes_rewritten,
        total_live_entries=total_live_entries,
        no_op_reason=no_op_reason,
    )


def _commit_attempt(
    *,
    table: PyIcebergTable,
    plan: _Plan,
    target_branch: str,
    operation: Any,
    target_size_bytes: int,
) -> RewriteManifestsResult:
    snapshot_props = {
        SNAPSHOT_PROP_MAINTENANCE_OP: SNAPSHOT_PROP_MAINTENANCE_OP_VALUE,
        SNAPSHOT_PROP_REWRITE_ID: plan.rewrite_id,
        SNAPSHOT_PROP_REWRITE_STRATEGY: SNAPSHOT_PROP_REWRITE_STRATEGY_VALUE,
        SNAPSHOT_PROP_INPUT_MANIFESTS: str(len(plan.matching_manifests)),
        SNAPSHOT_PROP_INPUT_BYTES: str(plan.bytes_rewritten),
        SNAPSHOT_PROP_SPEC_ID: str(_spec_id_of_plan(plan)),
    }

    producer_cls = _producer_class()
    with table.transaction() as txn:
        producer = producer_cls(
            operation=operation,
            transaction=txn,
            io=table.io,
            branch=target_branch,
            snapshot_properties=snapshot_props,
            commit_uuid=_uuid.uuid4(),
            plan=plan,
            target_size_bytes=target_size_bytes,
        )
        producer.build_new_manifests()
        producer.snapshot_properties[SNAPSHOT_PROP_OUTPUT_MANIFESTS] = str(
            len(producer.new_manifests)
        )
        producer.snapshot_properties[SNAPSHOT_PROP_OUTPUT_BYTES] = str(
            producer.bytes_added
        )
        producer.commit()
        committed_snapshot_id = int(producer.snapshot_id)
        added = len(producer.new_manifests)
        bytes_added = producer.bytes_added

    return RewriteManifestsResult(
        rewritten_manifests_count=len(plan.matching_manifests),
        added_manifests_count=added,
        bytes_rewritten=plan.bytes_rewritten,
        bytes_added=bytes_added,
        rewrite_id=plan.rewrite_id,
        snapshot_id=committed_snapshot_id,
    )


def _spec_id_of_plan(plan: _Plan) -> int:
    return int(plan.matching_manifests[0].partition_spec_id)


_PRODUCER_CLASS: type | None = None


def _producer_class() -> type:
    """Lazily build and cache the snapshot producer used for manifest rewrite.

    The producer overrides ``_existing_manifests`` to return our untouched +
    newly-written manifests and ``_deleted_entries`` to no-op, which is what a
    manifest reshuffle requires.
    """
    global _PRODUCER_CLASS
    if _PRODUCER_CLASS is not None:
        return _PRODUCER_CLASS

    from pyiceberg.table.update.snapshot import _SnapshotProducer

    class _RewriteManifestsProducer(_SnapshotProducer):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *,
            operation: Any,
            transaction: Any,
            io: Any,
            branch: str,
            snapshot_properties: dict[str, str],
            commit_uuid: _uuid.UUID,
            plan: _Plan,
            target_size_bytes: int,
        ) -> None:
            super().__init__(
                operation=operation,
                transaction=transaction,
                io=io,
                commit_uuid=commit_uuid,
                snapshot_properties=dict(snapshot_properties),
                branch=branch,
            )
            self._plan = plan
            self._target_size_bytes = target_size_bytes
            self._untouched_manifests: list[Any] = list(plan.untouched_manifests)
            self._new_manifests: list[Any] = []
            self._bytes_added = 0

        @property
        def new_manifests(self) -> list[Any]:
            return self._new_manifests

        @property
        def bytes_added(self) -> int:
            return self._bytes_added

        def _existing_manifests(self) -> list[Any]:
            return list(self._untouched_manifests) + list(self._new_manifests)

        def _deleted_entries(self) -> list[Any]:
            return []

        def _summary(self, snapshot_properties: dict[str, str]) -> Any:
            # Manifest reshuffle leaves data totals unchanged, so carry them
            # forward verbatim from the parent snapshot.
            from pyiceberg.table.snapshots import (
                TOTAL_DATA_FILES,
                TOTAL_DELETE_FILES,
                TOTAL_EQUALITY_DELETES,
                TOTAL_FILE_SIZE,
                TOTAL_POSITION_DELETES,
                TOTAL_RECORDS,
                Operation,
                Summary,
            )

            previous_snapshot = (
                self._transaction.table_metadata.snapshot_by_id(
                    self._parent_snapshot_id
                )
                if self._parent_snapshot_id is not None
                else None
            )
            prev_props: dict[str, str] = {}
            if previous_snapshot is not None and previous_snapshot.summary is not None:
                prev_props = dict(
                    getattr(previous_snapshot.summary, "additional_properties", {})
                    or {}
                )
            carry = {
                k: prev_props[k]
                for k in (
                    TOTAL_DATA_FILES,
                    TOTAL_DELETE_FILES,
                    TOTAL_RECORDS,
                    TOTAL_FILE_SIZE,
                    TOTAL_POSITION_DELETES,
                    TOTAL_EQUALITY_DELETES,
                )
                if k in prev_props
            }
            return Summary(operation=Operation.REPLACE, **carry, **snapshot_properties)

        def build_new_manifests(self) -> None:
            """Read live entries from each matching manifest and write target-sized output.

            Entries cluster by ``(partition_spec_id, partition_key_tuple)`` so a
            partition-scoped query reads one manifest per partition instead of
            scanning across the rewritten set.
            """
            from pyiceberg.manifest import ManifestEntry, ManifestEntryStatus

            if not self._plan.matching_manifests:
                return

            avg_bytes = max(
                1,
                self._plan.bytes_rewritten
                // max(1, self._plan.total_live_entries),
            )
            roll_target_entries = max(
                1, int(self._target_size_bytes * _ROLL_FACTOR / avg_bytes)
            )

            rollers: dict[tuple[int, tuple[Any, ...]], _RollingManifestWriter] = {}

            for manifest in self._plan.matching_manifests:
                spec_id = manifest.partition_spec_id
                spec = self._transaction.table_metadata.specs()[spec_id]

                for entry in manifest.fetch_manifest_entry(
                    self._io, discard_deleted=True
                ):
                    partition_key = _partition_key_tuple(entry.data_file)
                    key = (spec_id, partition_key)
                    roller = rollers.get(key)
                    if roller is None:
                        roller = _RollingManifestWriter(
                            producer=self,
                            spec=spec,
                            roll_at_entries=roll_target_entries,
                        )
                        rollers[key] = roller
                    roller.add(
                        ManifestEntry.from_args(
                            status=ManifestEntryStatus.EXISTING,
                            snapshot_id=entry.snapshot_id,
                            sequence_number=entry.sequence_number,
                            file_sequence_number=entry.file_sequence_number,
                            data_file=entry.data_file,
                        )
                    )

            for roller in rollers.values():
                for mf in roller.finish():
                    self._new_manifests.append(mf)
                    self._bytes_added += int(mf.manifest_length)

    _PRODUCER_CLASS = _RewriteManifestsProducer
    return _PRODUCER_CLASS


class _RollingManifestWriter:
    """Open a new manifest every ``roll_at_entries`` entries."""

    def __init__(self, *, producer: _RewriteManifests, spec: Any, roll_at_entries: int):
        self._producer = producer
        self._spec = spec
        self._roll_at = max(1, roll_at_entries)
        self._writer: Any = None
        self._count = 0
        self._finished: list[ManifestFile] = []

    def add(self, entry: Any) -> None:
        if self._writer is None or self._count >= self._roll_at:
            self._close_writer()
            self._writer = self._producer.new_manifest_writer(self._spec).__enter__()  # type: ignore[attr-defined]
            self._count = 0
        self._writer.add_entry(entry)
        self._count += 1

    def _close_writer(self) -> None:
        if self._writer is None:
            return
        self._writer.__exit__(None, None, None)
        self._finished.append(self._writer.to_manifest_file())
        self._writer = None

    def finish(self) -> list[ManifestFile]:
        self._close_writer()
        return self._finished


def _partition_key_tuple(data_file: Any) -> tuple[Any, ...]:
    """Stable hashable representation of a data file's partition tuple."""
    partition = getattr(data_file, "partition", None)
    if partition is None:
        return ()
    items = getattr(partition, "__dict__", None) or {}
    if items:
        return tuple(sorted(items.items()))
    fields = getattr(partition, "_fields", None)
    if fields:
        return tuple(getattr(partition, f) for f in fields)
    try:
        return tuple(partition)
    except TypeError:
        return (partition,)


def _resolve_spec_id(table: PyIcebergTable, spec_id: int | None) -> int:
    if spec_id is None:
        return int(table.spec().spec_id)
    if int(spec_id) not in {int(s) for s in table.specs().keys()}:
        raise ValueError(
            f"spec_id={spec_id!r} is not present in table.specs() "
            f"({sorted(int(s) for s in table.specs().keys())})"
        )
    return int(spec_id)


def _resolve_branch(table: PyIcebergTable, branch: str | None) -> str:
    from pyiceberg.table.refs import MAIN_BRANCH, SnapshotRefType

    if branch is None:
        return MAIN_BRANCH
    ref = table.metadata.refs.get(branch)
    if ref is None:
        raise ValueError(f"branch {branch!r} does not exist on this table")
    if ref.snapshot_ref_type != SnapshotRefType.BRANCH:
        raise ValueError(f"{branch!r} is a tag, not a branch")
    return branch


def _resolve_rewrite_id(
    *,
    table: PyIcebergTable,
    target_branch: str,
    spec_id: int,
    target_size_bytes: int,
    matching_manifest_paths: list[str],
) -> str:
    payload = {
        "table_uuid": str(table.metadata.table_uuid),
        "branch": target_branch,
        "spec_id": int(spec_id),
        "target_size_bytes": int(target_size_bytes),
        "manifests": sorted(matching_manifest_paths),
    }
    import json

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _lookup_idempotent_result(
    table: PyIcebergTable, rewrite_id: str
) -> RewriteManifestsResult | None:
    if not rewrite_id:
        return None
    for snap in list(table.metadata.snapshots or [])[-50:]:
        summary = _summary_as_dict(snap.summary)
        if (
            summary.get(SNAPSHOT_PROP_REWRITE_ID) == rewrite_id
            and summary.get(SNAPSHOT_PROP_REWRITE_STRATEGY)
            == SNAPSHOT_PROP_REWRITE_STRATEGY_VALUE
        ):
            return RewriteManifestsResult(
                rewritten_manifests_count=int(
                    summary.get(SNAPSHOT_PROP_INPUT_MANIFESTS, 0)
                ),
                added_manifests_count=int(
                    summary.get(SNAPSHOT_PROP_OUTPUT_MANIFESTS, 0)
                ),
                bytes_rewritten=int(summary.get(SNAPSHOT_PROP_INPUT_BYTES, 0)),
                bytes_added=int(summary.get(SNAPSHOT_PROP_OUTPUT_BYTES, 0)),
                rewrite_id=rewrite_id,
                snapshot_id=int(snap.snapshot_id),
            )
    return None


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
