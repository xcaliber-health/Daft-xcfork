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
        """Compact or re-cluster data files in this table.

        `strategy` is one of:
            - ``"binpack"``: consolidate small/skewed files into target-sized files.
            - ``"sort"``: sort by ``sort_order`` (list of ``(column, asc|desc, nulls-first|nulls-last)``).
            - ``"zorder"``: spatially cluster on ``zorder_by`` columns via z-order curve.

        `where` filters candidate files via pyiceberg's expression language. `branch`
        targets a non-default branch. `options` is a dict of tuning knobs; see the
        docs for the full table.

        Returns a ``RewriteResult`` summarizing files removed, files added, and the
        commit's ``rewrite_id`` (used for idempotent replay).

        Equality deletes anywhere in the scanned scope raise
        ``daft.daft.EqualityDeletesPresentError`` — apply them first.

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
