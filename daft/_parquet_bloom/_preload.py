"""Preload PyArrow's libparquet/libarrow/libarrow_python into the process.

When the Daft extension is loaded, its `daft-parquet-bloom` shim has
unresolved symbols pointing into PyArrow's C++ libraries (which ship inside
the pyarrow wheel). Those libraries are NOT on the system loader's default
search path, so we must explicitly load them before any code in the shim
runs. This module is imported as the very first thing inside
`daft._parquet_bloom.__init__`.

The preload is best-effort: if it fails the rest of Daft still works, only
bloom-filter pruning becomes unavailable.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

# Tracks whether the preload succeeded. Public for the orchestration layer
# to gate calls into the shim.
preloaded: bool = False
preload_error: str | None = None


def _candidate_lib_names(stem: str) -> list[str]:
    """Generate platform-specific shared-library file names for a given stem.

    PyArrow names its libraries with a SOVERSION suffix on Linux (e.g.
    ``libparquet.so.2000``). The bare ``libparquet.so`` is a symlink that
    only ships in dev installs, so we cannot rely on it. We probe a few
    versioned forms — PyArrow's SOVERSION moves with each minor release.
    """
    if sys.platform == "win32":
        # Windows uses plain DLL names; the .lib import file is unversioned.
        return [f"{stem}.dll"]
    if sys.platform == "darwin":
        return [f"lib{stem}.dylib", f"lib{stem}.2000.dylib", f"lib{stem}.1900.dylib"]
    # Linux / other Unix.
    return [
        f"lib{stem}.so",
        f"lib{stem}.so.2000",
        f"lib{stem}.so.1900",
        f"lib{stem}.so.1800",
        f"lib{stem}.so.1700",
        f"lib{stem}.so.1600",
    ]


def _load_first_match(lib_dir: Path, stem: str) -> ctypes.CDLL | None:
    """Try each candidate filename for ``stem`` in ``lib_dir`` and return the
    first one that loads, or ``None`` if none do."""
    flags = 0
    if sys.platform != "win32":
        # RTLD_GLOBAL is critical: when the daft extension loads next, its
        # undefined parquet/arrow symbols must resolve against the libraries
        # we are loading now. RTLD_LAZY avoids unrelated symbol-resolution
        # failures in libraries we don't actually call into.
        flags = ctypes.RTLD_GLOBAL | os.RTLD_LAZY
    for name in _candidate_lib_names(stem):
        path = lib_dir / name
        if not path.exists():
            continue
        try:
            if sys.platform == "win32":
                return ctypes.WinDLL(str(path))
            return ctypes.CDLL(str(path), mode=flags)
        except OSError as e:  # noqa: PERF203
            _logger.debug("Failed to load %s: %s", path, e)
    return None


def preload() -> None:
    """Best-effort preload of PyArrow's C++ libraries. Idempotent."""
    global preloaded, preload_error
    if preloaded:
        return

    try:
        import pyarrow
    except ImportError as e:
        preload_error = f"pyarrow not importable: {e}"
        return

    lib_dirs = [Path(d) for d in pyarrow.get_library_dirs()]
    if not lib_dirs:
        preload_error = "pyarrow.get_library_dirs() returned empty"
        return

    # On Windows we additionally have to extend the DLL search path before
    # the next LoadLibrary so transitive dependencies resolve.
    if sys.platform == "win32":
        for d in lib_dirs:
            if d.exists():
                os.add_dll_directory(str(d))

    # Load order matters: arrow first (parquet links it), then parquet,
    # then arrow_python (depends on both arrow and Python).
    loaded: dict[str, ctypes.CDLL] = {}
    for stem in ("arrow", "parquet", "arrow_python"):
        for lib_dir in lib_dirs:
            handle = _load_first_match(lib_dir, stem)
            if handle is not None:
                loaded[stem] = handle
                break
        if stem not in loaded:
            preload_error = (
                f"could not locate lib{stem} in any of: {[str(d) for d in lib_dirs]}"
            )
            return

    preloaded = True
