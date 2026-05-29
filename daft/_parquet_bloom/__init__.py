"""Daft's Python-side glue around the libparquet bloom-filter shim.

Public entry point is :func:`prune_row_groups_for_iceberg`: given a parquet
file handle (via pyiceberg's FileIO so remote backends "just work") and a
pyiceberg ``BooleanExpression`` describing the row filter, return the list
of row group indices that still need to be scanned. A return of ``None``
means "could not run bloom pruning, scan everything" — the caller must
treat ``None`` as a soft-fail, not an error.

Side-effect of importing this module: it preloads PyArrow's
libparquet/libarrow/libarrow_python via :mod:`._preload`. The shim's
undefined symbols resolve against those once they're in the process. This
is best-effort: if pyarrow isn't importable, every call below short-circuits
to ``None``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import _preload as _preload_mod
from ._preload import preload

if TYPE_CHECKING:
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.io import FileIO
    from pyiceberg.schema import Schema as IcebergSchema

_logger = logging.getLogger(__name__)

# Trigger lib preload as a side-effect of import. Daft callers do
# `from daft._parquet_bloom import prune_row_groups_for_iceberg` — by the
# time that returns, pyarrow's libraries are in the process.
preload()


def _try_get_reader_cls():
    """Look up the PyO3 reader class by name. Import is delayed so this
    module stays loadable when the preload step failed (e.g. pyarrow not
    importable in the host venv): callers see ``None`` instead of an
    ``ImportError`` at module import time.

    Reads ``_preload.preloaded`` via the module reference because the
    preload function mutates the value AFTER this module imports — a
    plain ``from ._preload import preloaded`` would capture the initial
    ``False`` and never see the update.
    """
    if not _preload_mod.preloaded:
        return None
    try:
        from daft.daft import ParquetBloomReader

        return ParquetBloomReader
    except ImportError:
        return None


def prune_row_groups_for_iceberg(
    file_io: "FileIO",
    file_path: str,
    boolean_expr: "BooleanExpression",
    schema: "IcebergSchema",
) -> list[int] | None:
    """Run bloom-filter pruning for one Iceberg data file.

    Args:
        file_io: pyiceberg FileIO that knows how to open ``file_path`` —
            handles local files, S3, GCS, ABFS, HDFS transparently.
        file_path: full URI of the parquet data file.
        boolean_expr: row filter produced by :func:`convert_row_filter` (or
            any bound pyiceberg BooleanExpression).
        schema: Iceberg schema for the table.

    Returns:
        List of surviving row group indices (sorted, ascending), or ``None``
        when bloom pruning could not be performed and the caller must scan
        the full file. ``[]`` (empty list) means every row group can be
        skipped.
    """
    reader_cls = _try_get_reader_cls()
    if reader_cls is None:
        if _preload_mod.preload_error is not None:
            _logger.debug("bloom pruning skipped: %s", _preload_mod.preload_error)
        return None

    from ._walk import extract_probes

    probes = extract_probes(boolean_expr, schema)
    if not probes:
        # No probable conjuncts — bloom pruning can't help. Caller scans
        # everything as usual.
        return None

    try:
        native_file = file_io.new_input(file_path).open()
    except Exception as e:  # noqa: BLE001
        _logger.debug("bloom pruning skipped: open(%s) failed: %s", file_path, e)
        return None

    try:
        try:
            reader = reader_cls.open_from_native_file(native_file)
        except Exception as e:  # noqa: BLE001
            _logger.debug(
                "bloom pruning skipped: open_from_native_file(%s) failed: %s",
                file_path,
                e,
            )
            return None

        n_rgs = reader.num_row_groups
        surviving = list(range(n_rgs))

        try:
            for probe in probes:
                new_surviving: list[int] = []
                for rg in surviving:
                    # Conjunct is true for this RG if ANY listed literal could
                    # be present (or if the bloom is inconclusive).
                    keep = False
                    for lit_bytes in probe.literals:
                        rc = reader.probe(rg, probe.column, probe.type_id, lit_bytes)
                        if rc != 0:  # 1 = present, -1 = no bloom (inconclusive)
                            keep = True
                            break
                    if keep:
                        new_surviving.append(rg)
                surviving = new_surviving
                if not surviving:
                    break
        except Exception as e:  # noqa: BLE001
            # Any error mid-probe (shim error, transient IO, bad literal
            # encoding) must soft-fail to "scan everything" rather than
            # poison the whole scan plan.
            _logger.debug("bloom pruning soft-fail on %s: %s", file_path, e)
            return None

        return surviving
    finally:
        # Drop the reader BEFORE closing the file. The C++ side wraps the
        # NativeFile via arrow::py::PyReadableFile which holds a borrowed
        # PyObject — releasing the reader first ensures no probe runs
        # against a closed file. See shim.h "LIFETIME CONTRACT".
        try:
            del reader
        except UnboundLocalError:
            pass
        try:
            native_file.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["prune_row_groups_for_iceberg"]
