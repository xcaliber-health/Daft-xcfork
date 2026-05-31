"""Derive a single ordered key that clusters rows along a space-filling curve.

The key interleaves order-preserving byte encodings of several columns so that
rows close in the multi-column space sort close together. It is produced as a
batch expression, so the key for each batch of rows is computed as that batch
streams through the engine rather than holding every row at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import daft
from daft import DataType

if TYPE_CHECKING:
    import pyarrow as pa

    from daft.expressions import Expression
    from daft.series import Series


@daft.func.batch(return_dtype=DataType.binary())
def _zorder_key_batch(
    *columns: Series,
    var_length_contribution: int,
    max_output_size: int,
) -> "pa.Array":
    """Compute the interleaved key for one batch of the given columns."""
    import pyarrow as pa

    from daft.daft import _iceberg as _rust_iceberg

    arrays = []
    for column in columns:
        arr = column.to_arrow()
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        arrays.append(arr)
    return _rust_iceberg.build_zorder_key_py(arrays, var_length_contribution, max_output_size)


def zorder_key(
    columns: list[Expression],
    var_length_contribution: int,
    max_output_size: int,
) -> Expression:
    """Return an expression yielding one ordered key per row from the columns.

    Parameters
    ----------
    columns
        Column expressions to cluster on, in priority order.
    var_length_contribution
        Number of leading bytes taken from each variable-length value (text or
        binary) when encoding it into the key.
    max_output_size
        Maximum length in bytes of each row's key; the interleaved result is
        truncated to this size.

    Returns
    -------
    Expression
        A binary-valued expression. Sorting by it clusters nearby rows together.
    """
    # Called with column expressions, the batch function returns an expression;
    # the cast narrows the call's value/expression union for callers.
    key = _zorder_key_batch(
        *columns,
        var_length_contribution=var_length_contribution,
        max_output_size=max_output_size,
    )
    return cast("Expression", key)
