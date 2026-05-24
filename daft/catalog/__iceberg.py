"""WARNING! These APIs are internal; please use Catalog.from_iceberg() and Table.from_iceberg()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyiceberg.catalog import Catalog as InnerCatalog
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NoSuchNamespaceError, NoSuchTableError
from pyiceberg.io.pyarrow import _pyarrow_to_schema_without_ids
from pyiceberg.partitioning import PartitionField as PyIcebergPartitionField
from pyiceberg.partitioning import PartitionSpec as PyIcebergPartitionSpec
from pyiceberg.partitioning import _PartitionNameGenerator
from pyiceberg.schema import Schema as PyIcebergSchema
from pyiceberg.schema import assign_fresh_schema_ids
from pyiceberg.table import Table as InnerTable
from pyiceberg.transforms import (
    BucketTransform,
    DayTransform,
    HourTransform,
    IdentityTransform,
    MonthTransform,
    TruncateTransform,
    YearTransform,
)

from daft.catalog import (
    Catalog,
    Function,
    Identifier,
    NotFoundError,
    Properties,
    Schema,
    Table,
)
from daft.io.iceberg._iceberg import read_iceberg

if TYPE_CHECKING:
    from collections.abc import Callable

    from daft.dataframe import DataFrame
    from daft.io.partitioning import PartitionField


class IcebergCatalog(Catalog):
    _inner: InnerCatalog

    def __init__(self) -> None:
        raise RuntimeError(
            "IcebergCatalog.__init__ is not supported, please use `Catalog.from_iceberg` instead."
        )

    @staticmethod
    def _from_obj(obj: object) -> IcebergCatalog:
        """Returns an IcebergCatalog instance if the given object can be adapted so."""
        if isinstance(obj, InnerCatalog):
            c = IcebergCatalog.__new__(IcebergCatalog)
            c._inner = obj
            return c
        raise ValueError(f"Unsupported iceberg catalog type: {type(obj)}")

    @staticmethod
    def _load_catalog(name: str, **options: str | None) -> IcebergCatalog:
        c = IcebergCatalog.__new__(IcebergCatalog)
        c._inner = load_catalog(name, **options)
        return c

    @property
    def name(self) -> str:
        return self._inner.name

    @staticmethod
    def _partition_fields_to_pyiceberg_spec(
        iceberg_schema: PyIcebergSchema, partition_fields: list[PartitionField] | None
    ) -> PyIcebergPartitionSpec | None:
        """Converts Daft partition fields to a PyIceberg PartitionSpec."""
        if not partition_fields:
            return None

        # Convert Daft schema → PyArrow schema → PyIceberg schema (with IDs)
        iceberg_partition_fields = []
        for idx, pf in enumerate(partition_fields):
            source_field = iceberg_schema.find_field(pf.field.name)
            source_id = source_field.field_id
            source_name = source_field.name
            field_id = 1000 + idx
            if pf.transform is None or pf.transform.is_identity():
                transform = IdentityTransform()
                pf_name = _PartitionNameGenerator().identity(
                    field_id=field_id,
                    source_name=source_name,
                    source_id=source_id,
                )
            elif pf.transform.is_year():
                transform = YearTransform()
                pf_name = _PartitionNameGenerator().year(
                    field_id=field_id,
                    source_name=source_name,
                    source_id=source_id,
                )
            elif pf.transform.is_month():
                transform = MonthTransform()
                pf_name = _PartitionNameGenerator().month(
                    field_id=field_id,
                    source_name=source_name,
                    source_id=source_id,
                )
            elif pf.transform.is_day():
                transform = DayTransform()
                pf_name = _PartitionNameGenerator().day(
                    field_id=field_id,
                    source_name=source_name,
                    source_id=source_id,
                )
            elif pf.transform.is_hour():
                transform = HourTransform()
                pf_name = _PartitionNameGenerator().hour(
                    field_id=field_id,
                    source_name=source_name,
                    source_id=source_id,
                )
            elif pf.transform.is_iceberg_bucket():
                transform = BucketTransform(num_buckets=pf.transform.num_buckets)
                pf_name = _PartitionNameGenerator().bucket(
                    field_id=field_id,
                    source_name=source_name,
                    source_id=source_id,
                    num_buckets=pf.transform.num_buckets,
                )
            elif pf.transform.is_iceberg_truncate():
                transform = TruncateTransform(width=pf.transform.width)
                pf_name = _PartitionNameGenerator().truncate(
                    field_id=field_id,
                    source_name=source_name,
                    source_id=source_id,
                    width=pf.transform.width,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported partition transform: {pf.transform}"
                )

            iceberg_partition_fields.append(
                PyIcebergPartitionField(
                    source_id=source_id,
                    field_id=field_id,
                    transform=transform,
                    name=pf_name,
                )
            )
        return PyIcebergPartitionSpec(*iceberg_partition_fields)

    ###
    # create_*
    ###

    def _create_function(
        self, ident: Identifier, function: Function | Callable[..., Any]
    ) -> None:
        raise NotImplementedError("Iceberg does not support function registration.")

    def _get_function(self, ident: Identifier) -> Function:
        raise NotFoundError(f"Function '{ident}' not found in catalog '{self.name}'")

    def _create_namespace(self, identifier: Identifier) -> None:
        ident = _to_pyiceberg_ident(identifier)
        self._inner.create_namespace(ident)

    def _create_table(
        self,
        identifier: Identifier,
        schema: Schema,
        properties: Properties | None = None,
        partition_fields: list[PartitionField] | None = None,
    ) -> Table:
        i = _to_pyiceberg_ident(identifier)
        pa_schema = schema.to_pyarrow_schema()
        iceberg_schema = assign_fresh_schema_ids(
            _pyarrow_to_schema_without_ids(pa_schema)
        )
        partition_spec = self._partition_fields_to_pyiceberg_spec(
            iceberg_schema, partition_fields
        )
        t = IcebergTable.__new__(IcebergTable)
        if partition_spec is not None:
            t._inner = self._inner.create_table(
                i,
                schema=iceberg_schema,
                partition_spec=partition_spec,
            )
        else:
            t._inner = self._inner.create_table(
                i,
                schema=iceberg_schema,
            )
        return t

    ###
    # drop_*
    ###

    def _drop_namespace(self, identifier: Identifier) -> None:
        ident = _to_pyiceberg_ident(identifier)
        self._inner.drop_namespace(ident)

    def _drop_table(self, identifier: Identifier) -> None:
        ident = _to_pyiceberg_ident(identifier)
        self._inner.drop_table(ident)

    ###
    # has_*
    ###

    def _has_namespace(self, identifier: Identifier) -> bool:
        ident = _to_pyiceberg_ident(identifier)
        try:
            _ = self._inner.list_namespaces(ident)
            return True
        except NoSuchNamespaceError:
            return False

    def _has_table(self, identifier: Identifier) -> bool:
        ident = _to_pyiceberg_ident(identifier)
        try:
            # using load_table instead of table_exists because table_exists does not work with an instance of the `tabulario/iceberg-rest` Docker image
            self._inner.load_table(ident)
            return True
        except NoSuchTableError:
            return False

    ###
    # get_*
    ###

    def _get_table(self, identifier: Identifier) -> IcebergTable:
        ident = _to_pyiceberg_ident(identifier)
        try:
            return IcebergTable._from_obj(self._inner.load_table(ident))
        except NoSuchTableError as ex:
            # convert to not found because we want to (sometimes) ignore it internally
            raise NotFoundError() from ex
        except Exception as ex:
            # wrap original exceptions
            raise Exception(
                "pyiceberg raised an exception while calling get_table"
            ) from ex

    ###
    # list_*
    ###

    def _list_namespaces(self, pattern: str | None = None) -> list[Identifier]:
        prefix = () if pattern is None else _to_pyiceberg_ident(pattern)
        return [Identifier(*tup) for tup in self._inner.list_namespaces(prefix)]

    def _list_tables(self, pattern: str | None = None) -> list[Identifier]:
        if pattern is None:
            tables = []
            for ns in self.list_namespaces():
                tables.extend(self._inner.list_tables(str(ns)))
        else:
            tables = self._inner.list_tables(pattern)
        return [Identifier(*tup) for tup in tables]


class IcebergTable(Table):
    _inner: InnerTable

    _read_options = {"snapshot_id"}
    _write_options: set[str] = set()

    def __init__(self) -> None:
        raise RuntimeError(
            "IcebergTable.__init__ is not supported, please use `Table.from_iceberg` instead."
        )

    @property
    def name(self) -> str:
        return self._inner.name()[-1]

    def schema(self) -> Schema:
        return self.read().schema()

    @staticmethod
    def _from_obj(obj: object) -> IcebergTable:
        """Returns an IcebergTable if the given object can be adapted so."""
        if isinstance(obj, InnerTable):
            t = IcebergTable.__new__(IcebergTable)
            t._inner = obj
            return t
        raise ValueError(f"Unsupported iceberg table type: {type(obj)}")

    def read(self, **options: Any | None) -> DataFrame:
        Table._validate_options("Iceberg read", options, IcebergTable._read_options)
        return read_iceberg(self._inner, snapshot_id=options.get("snapshot_id"))

    def append(self, df: DataFrame, **options: Any) -> None:
        self._validate_options("Iceberg write", options, IcebergTable._write_options)

        df.write_iceberg(self._inner, mode="append")

    def overwrite(self, df: DataFrame, **options: Any) -> None:
        self._validate_options("Iceberg write", options, IcebergTable._write_options)

        df.write_iceberg(self._inner, mode="overwrite")

    def rewrite_data_files(
        self,
        strategy: str = "binpack",
        *,
        sort_order: list[tuple[str, str, str]] | None = None,
        zorder_by: list[str] | None = None,
        where: Any | None = None,
        branch: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Compact or re-cluster the data files of this table.

        Reads matching data files, writes new files sized close to
        ``target-file-size-bytes``, and commits an atomic snapshot that
        replaces the inputs with the outputs. Tolerates concurrent appends to
        partitions outside the rewrite scope; raises ``RewriteConflict`` when
        another writer modifies an affected partition between plan and commit.

        Parameters
        ----------
        strategy : {"binpack", "sort", "zorder"}, default ``"binpack"``
            ``"binpack"`` packs small files into larger ones without ordering.
            ``"sort"`` sorts each output by ``sort_order``. ``"zorder"`` clusters
            each output along an interleaved-bits curve over ``zorder_by``.
        sort_order : list of (str, str, str), optional
            Required when ``strategy="sort"``. Each entry is
            ``(column, "asc"|"desc", "nulls-first"|"nulls-last")``.
        zorder_by : list of str, optional
            Required when ``strategy="zorder"``. Primitive numeric, boolean,
            string, binary, date, time, timestamp, or decimal columns.
        where : str or expression, optional
            Filter narrowing the candidate files. Files outside the predicate
            are not touched.
        branch : str, optional
            Branch to commit to. Defaults to ``main``.
        options : dict, optional
            Tuning knobs: ``target-file-size-bytes``, ``min-input-files``,
            ``min-file-size-bytes``, ``max-file-size-bytes``,
            ``max-file-group-size-bytes``, ``rewrite-all``,
            ``rewrite-job-order``, ``max-concurrent-file-group-rewrites``,
            ``delete-file-threshold``, ``partial-progress.enabled``,
            ``partial-progress.max-commits``,
            ``partial-progress.max-failed-commits``, ``compression-factor``,
            ``remove-dangling-deletes``, ``zorder.max-output-size``,
            ``zorder.var-length-contribution``. Commit retry is tuned via
            table properties ``commit.retry.num-retries``,
            ``commit.retry.min-wait-ms``, ``commit.retry.max-wait-ms``,
            ``commit.retry.total-timeout-ms``.

        Returns
        -------
        RewriteResult
            File and byte counts, commit count, snapshot ids, and a stable
            ``rewrite_id`` used for idempotent replay.

        Raises
        ------
        ValueError
            On unknown ``strategy`` or invalid ``sort_order`` / ``zorder_by``.
        EqualityDeletesPresentError
            When the scanned scope contains equality-delete files; apply them
            before retrying.
        RewriteConflict
            When a concurrent writer modified a partition this call is
            rewriting, or removed one of its input files.

        Examples
        --------
        >>> table.rewrite_data_files("binpack")  # doctest: +SKIP
        >>> table.rewrite_data_files(  # doctest: +SKIP
        ...     "sort",
        ...     sort_order=[("event_ts", "asc", "nulls-last")],
        ... )
        >>> table.rewrite_data_files(  # doctest: +SKIP
        ...     "zorder",
        ...     zorder_by=["lat", "lon"],
        ...     where="region = 'us'",
        ... )
        """
        from daft.io.iceberg._compact import run

        return run(
            self._inner,
            strategy=strategy,
            sort_order=sort_order,
            zorder_by=zorder_by,
            where=where,
            branch=branch,
            options=options,
        )

    def expire_snapshots(
        self,
        *,
        older_than: Any | None = None,
        retain_last: int | None = None,
        snapshot_ids: list[int] | None = None,
        clean_expired_files: bool = True,
        stream_results: bool = False,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Expire old snapshots and reclaim their files.

        Parameters
        ----------
        older_than : datetime.datetime or int, optional
            Expire snapshots with a timestamp strictly older than this value.
            ``int`` is treated as epoch milliseconds. If no retention argument is
            supplied at all, the table property ``history.expire.max-snapshot-age-ms``
            (default 5 days) is used to compute a default ``older_than``.
        retain_last : int, optional
            Always retain the N most-recent snapshots reachable from the current
            ref. The table property ``history.expire.min-snapshots-to-keep`` acts
            as a floor — the effective retention is ``max(retain_last, floor)``.
        snapshot_ids : list of int, optional
            Explicit IDs to expire. Branch and tag heads are always protected and
            raise ``ValueError`` if listed here.
        clean_expired_files : bool, default True
            When ``True``, physically delete files (data, position deletes,
            equality deletes, manifests, manifest lists, statistics) that become
            unreachable. When ``False``, only the snapshot metadata is removed.
        stream_results : bool, default False
            Stream candidate file paths through a generator instead of materializing
            them. Bounds memory at the per-snapshot manifest size; the kept-files
            set remains in memory.
        options : dict, optional
            Tuning knobs:

            - ``max-concurrent-deletes`` (int, default 4)
            - ``max-concurrent-manifest-reads`` (int, default 4)
            - ``delete-num-retries`` (int, default 3)
            - ``delete-backoff-base-seconds`` (float, default 0.1)

            Commit retry is tuned via table properties
            ``commit.retry.num-retries``, ``commit.retry.min-wait-ms``,
            ``commit.retry.max-wait-ms``, ``commit.retry.total-timeout-ms``.

        Returns
        -------
        ExpireResult
            Counts of files removed, broken out by file type.

        Raises
        ------
        ValueError
            If the table property ``gc.enabled`` is false, or if
            ``snapshot_ids`` contains protected or unknown IDs, or if
            ``retain_last < 1``.

        Examples
        --------
        >>> table.expire_snapshots(retain_last=10)  # doctest: +SKIP
        >>> from datetime import datetime, timedelta, timezone
        >>> cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)
        >>> table.expire_snapshots(older_than=cutoff)  # doctest: +SKIP
        """
        from daft.io.iceberg._expire import run as _expire_run

        return _expire_run(
            self._inner,
            older_than=older_than,
            retain_last=retain_last,
            snapshot_ids=snapshot_ids,
            clean_expired_files=clean_expired_files,
            stream_results=stream_results,
            options=options,
        )

    def remove_orphan_files(
        self,
        *,
        older_than: Any | None = None,
        location: str | None = None,
        dry_run: bool = False,
        prefix_mismatch_mode: str = "error",
        stream_results: bool = False,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Delete files under the table location that no snapshot references.

        Lists physical files under ``location`` (defaulting to ``table.location()``),
        subtracts the union of files reachable from every snapshot (data, delete,
        manifest, manifest-list, statistics, metadata.json), and deletes the rest.

        Parameters
        ----------
        older_than : datetime.datetime or int, optional
            Cutoff: only files modified strictly before this point are considered.
            ``int`` is treated as epoch milliseconds. Defaults to 3 days ago.
            Cutoffs newer than 24 hours ago are rejected to avoid racing live
            writers (override only in tests with ``options={'allow-recent': True}``).
        location : str, optional
            Subpath of ``table.location()`` to limit listing to. Useful for
            incremental cleanup. Defaults to the full table location.
        dry_run : bool, default False
            When ``True``, identify orphans without deleting. ``deleted_files_count``
            is ``0`` and ``sample_paths`` contains a sample of what would be deleted.
        prefix_mismatch_mode : str, default ``"error"``
            How to treat listed files whose scheme/authority is absent from the
            reachable set. ``"error"`` raises (default; safest), ``"delete"``
            treats them as orphans, ``"ignore"`` drops them from consideration.
            Scheme aliases (``s3``/``s3a``/``s3n``) are canonicalized before this
            check.
        stream_results : bool, default False
            Stream the listing through the reachability filter rather than
            materializing it. Bounds memory at the per-subdir listing size; the
            reachable set is always in memory.
        options : dict, optional
            Tuning knobs:

            - ``max-concurrent-list`` (int, default 4)
            - ``max-concurrent-deletes`` (int, default 4)
            - ``delete-num-retries`` (int, default 3)
            - ``delete-backoff-base-seconds`` (float, default 0.1)
            - ``sample-limit`` (int, default 1000) — cap on ``sample_paths``.
            - ``allow-recent`` (bool, default False) — disables the 24-hour
              floor; for tests only.

        Returns
        -------
        RemoveOrphanResult
            Counts and a bounded sample of orphan paths.

        Raises
        ------
        ValueError
            If the table property ``gc.enabled`` is false, if ``older_than`` is
            within 24 hours of now (and ``allow-recent`` is not set), if
            ``prefix_mismatch_mode`` is not one of the three allowed values, or
            if ``location`` is not a subpath of ``table.location()``.
        PrefixMismatchError
            With ``prefix_mismatch_mode='error'``, when any listed file uses a
            scheme/authority absent from the reachable set.

        Examples
        --------
        >>> table.remove_orphan_files(dry_run=True)  # doctest: +SKIP
        >>> from datetime import datetime, timedelta, timezone
        >>> cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)
        >>> table.remove_orphan_files(older_than=cutoff)  # doctest: +SKIP
        """
        from daft.io.iceberg._remove_orphan import run as _remove_orphan_run

        return _remove_orphan_run(
            self._inner,
            older_than=older_than,
            location=location,
            dry_run=dry_run,
            prefix_mismatch_mode=prefix_mismatch_mode,
            stream_results=stream_results,
            options=options,
        )

    def rewrite_manifests(
        self,
        *,
        spec_id: int | None = None,
        branch: str | None = None,
        use_caching: bool = False,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Repack live manifest entries into target-sized manifests.

        Reads the manifests of the target branch's current snapshot, keeps
        only live entries, and writes a fresh manifest set sized to
        ``manifest-target-size-bytes``. Output entries cluster by partition
        so a partition-scoped query reads at most one manifest per partition.
        Commits a single REPLACE snapshot in place of the rewritten manifests;
        data and delete files are unchanged.

        Parameters
        ----------
        spec_id : int, optional
            Partition spec to rewrite. Defaults to the table's current spec.
            Only manifests stamped with this spec are touched; manifests for
            other specs pass through unchanged.
        branch : str, optional
            Branch to rewrite. Defaults to ``main``.
        use_caching : bool, default False
            Reserved for signature stability; ignored — manifests are read
            from object storage on demand.
        options : dict, optional
            Tuning knobs (fall back to table properties where noted):

            - ``manifest-target-size-bytes`` (int, default 8 MiB, falls back
              to ``commit.manifest.target-size-bytes``)
            - ``manifest-min-count-to-merge`` (int, default 100, falls back to
              ``commit.manifest.min-count-to-merge``)

            Commit retry is tuned via table properties
            ``commit.retry.num-retries``, ``commit.retry.min-wait-ms``,
            ``commit.retry.max-wait-ms``, ``commit.retry.total-timeout-ms``.

        Returns
        -------
        RewriteManifestsResult
            Counts and byte totals for the rewritten and added manifests, plus
            the new snapshot id (``None`` when nothing needed rewriting).

        Raises
        ------
        ValueError
            If the table property ``gc.enabled`` is false, ``spec_id`` is not a
            valid spec on the table, or ``branch`` does not exist (or is a tag).

        Examples
        --------
        >>> table.rewrite_manifests()  # doctest: +SKIP
        >>> table.rewrite_manifests(spec_id=0)  # doctest: +SKIP
        """
        from daft.io.iceberg._rewrite_manifests import run as _rewrite_manifests_run

        return _rewrite_manifests_run(
            self._inner,
            spec_id=spec_id,
            branch=branch,
            use_caching=use_caching,
            options=options,
        )

    def compact_files(
        self,
        *,
        where: Any | None = None,
        branch: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Compact small files. Alias for ``rewrite_data_files("binpack", ...)``.

        Examples
        --------
        >>> table.compact_files()  # doctest: +SKIP
        >>> table.compact_files(where="region = 'us'")  # doctest: +SKIP
        """
        return self.rewrite_data_files(
            "binpack",
            where=where,
            branch=branch,
            options=options,
        )


def _to_pyiceberg_ident(ident: Identifier | str) -> tuple[str, ...] | str:
    return tuple(ident) if isinstance(ident, Identifier) else ident
