"""Walk a pyiceberg ``BooleanExpression`` and extract bloom-probable terms.

A bloom filter only proves *absence*. For pruning to be sound we may only
consult it for predicate shapes that are equivalent to "value must equal
one of {literals}":

  * ``EqualTo(col, lit)`` — must equal exactly one literal.
  * ``In(col, {lit, lit, ...})`` — must equal one of the listed literals.

Anything else (``Or``, ``Not``, ``NotEqualTo``, ranges) is treated as
inconclusive: the row group survives.

Literal byte conversion follows Parquet's PHYSICAL type encoding, because
that is what the writer hashed into the bloom filter:

  * Small ints (int8/16) widen to INT32 little-endian.
  * Unsigned ints reinterpret as their signed counterpart.
  * Booleans aren't supported (Parquet bloom filters on BOOLEAN are spec'd
    but not commonly written and Iceberg doesn't emit them).
  * Strings → raw UTF-8 bytes, no length prefix.
  * UUID → raw 16-byte big-endian.
  * Floats with NaN are skipped (writer hash of NaN is bit-pattern
    dependent and not stable across producers).
"""

from __future__ import annotations

import math
import struct
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.schema import Schema as IcebergSchema


# Parquet physical type IDs (matches the parquet::Type::type C++ enum
# values used in shim.cpp). Kept in sync manually because there is no
# Python source of truth for this enum that pyarrow exposes.
PT_BOOLEAN = 0
PT_INT32 = 1
PT_INT64 = 2
PT_INT96 = 3
PT_FLOAT = 4
PT_DOUBLE = 5
PT_BYTE_ARRAY = 6
PT_FIXED_LEN_BYTE_ARRAY = 7


@dataclass(frozen=True)
class BloomProbe:
    column: str
    type_id: int
    literals: tuple[bytes, ...]

def extract_probes(
    expr: "BooleanExpression",
    schema: "IcebergSchema",
) -> list[BloomProbe]:
    """Return the list of bloom-probable conjuncts inside `expr`.
    Returns an empty list when no probable shape is present. Callers should
    skip bloom pruning when the list is empty.
    """
    conjuncts = list(_split_and(expr))
    probes: list[BloomProbe] = []
    for c in conjuncts:
        p = _conjunct_to_probe(c, schema)
        if p is not None:
            probes.append(p)
    return probes

def _split_and(expr: "BooleanExpression") -> Iterable["BooleanExpression"]:
    """Flatten left/right children of nested ``And`` nodes."""
    from pyiceberg.expressions import And

    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, And):
            stack.append(e.right)
            stack.append(e.left)
        else:
            yield e


def _conjunct_to_probe(
    conjunct: "BooleanExpression",
    schema: "IcebergSchema",
) -> BloomProbe | None:
    """Try to extract a (column, literals, type_id) probe from one conjunct.
    Returns ``None`` for any shape we do not know how to prove false via a
    bloom filter.
    """
    from pyiceberg.expressions import BoundEqualTo, BoundIn, EqualTo, In

    # pyiceberg gives us either Bound* (after binding to a schema) or the
    # unbound textual form. After ``convert_row_filter`` the expression is
    # bound, but we cover both shapes for safety.
    if isinstance(conjunct, (BoundEqualTo, EqualTo)):
        col_name = _term_name(conjunct.term)
        literal = _literal_value(conjunct.literal)
        if col_name is None or literal is None:
            return None
        field = schema.find_field(col_name)
        type_id, value_bytes = _encode_literal(field.field_type, literal)
        if type_id is None:
            return None
        return BloomProbe(column=col_name, type_id=type_id, literals=(value_bytes,))

    if isinstance(conjunct, (BoundIn, In)):
        col_name = _term_name(conjunct.term)
        if col_name is None:
            return None
        field = schema.find_field(col_name)
        out: list[bytes] = []
        type_id_seen: int | None = None
        for lit in conjunct.literals:
            v = _literal_value(lit)
            if v is None:
                # Mixed null-in literals make the IN clause inconclusive
                # for bloom purposes.
                return None
            t, b = _encode_literal(field.field_type, v)
            if t is None:
                return None
            if type_id_seen is None:
                type_id_seen = t
            elif type_id_seen != t:
                return None
            out.append(b)
        if not out or type_id_seen is None:
            return None
        return BloomProbe(column=col_name, type_id=type_id_seen, literals=tuple(out))

    return None


def _term_name(term) -> str | None:
    """Pull the column name out of a pyiceberg term.
    Handles both ``Reference("col")`` (unbound) and ``BoundReference`` (bound)
    shapes by looking for a ``name`` attribute first, then falling back to
    the bound-form's accessor field. Nested paths (struct subfields) are
    rejected — Iceberg bloom filters are leaf-only on flat columns.
    """
    name = getattr(term, "name", None)
    if name is not None and "." not in name:
        return name
    ref = getattr(term, "ref", None)
    if ref is not None:
        n = getattr(ref, "name", None) or getattr(ref, "field_name", None)
        if n is not None and "." not in n:
            return n
    return None


def _literal_value(lit):
    if lit is None:
        return None
    return getattr(lit, "value", lit)


def _encode_literal(field_type, value) -> tuple[int | None, bytes]:
    """Convert a python literal into ``(parquet_type_id, raw_bytes)``.

    Returns ``(None, b"")`` when the type/value combination is not safe to
    bloom-probe. Bytes layout matches what the parquet writer would have
    hashed for that physical type.
    """
    from pyiceberg.types import (
        BinaryType,
        BooleanType,
        DateType,
        DecimalType,
        DoubleType,
        FixedType,
        FloatType,
        IntegerType,
        LongType,
        StringType,
        TimestampType,
        TimestamptzType,
        TimeType,
        UUIDType,
    )

    # Iceberg "int" is a 32-bit signed; "long" is 64-bit signed.
    if isinstance(field_type, IntegerType):
        return PT_INT32, struct.pack("<i", int(value))
    if isinstance(field_type, LongType):
        return PT_INT64, struct.pack("<q", int(value))
    if isinstance(field_type, DateType):
        # Parquet date is days-since-epoch as INT32.
        if hasattr(value, "toordinal"):
            from datetime import date

            value = (value - date(1970, 1, 1)).days
        return PT_INT32, struct.pack("<i", int(value))
    if isinstance(field_type, (TimestampType, TimestamptzType, TimeType)):
        # Iceberg encodes timestamps as INT64 micros; bloom hash uses the
        # writer's INT64 byte layout, so we mirror that.
        return PT_INT64, struct.pack("<q", int(value))
    if isinstance(field_type, FloatType):
        f = float(value)
        if math.isnan(f):
            return None, b""
        return PT_FLOAT, struct.pack("<f", f)
    if isinstance(field_type, DoubleType):
        f = float(value)
        if math.isnan(f):
            return None, b""
        return PT_DOUBLE, struct.pack("<d", f)
    if isinstance(field_type, StringType):
        if not isinstance(value, str):
            return None, b""
        return PT_BYTE_ARRAY, value.encode("utf-8")
    if isinstance(field_type, BinaryType):
        if not isinstance(value, (bytes, bytearray)):
            return None, b""
        return PT_BYTE_ARRAY, bytes(value)
    if isinstance(field_type, FixedType):
        if not isinstance(value, (bytes, bytearray)):
            return None, b""
        return PT_FIXED_LEN_BYTE_ARRAY, bytes(value)
    if isinstance(field_type, UUIDType):
        if isinstance(value, uuid.UUID):
            return PT_FIXED_LEN_BYTE_ARRAY, value.bytes
        if isinstance(value, (bytes, bytearray)) and len(value) == 16:
            return PT_FIXED_LEN_BYTE_ARRAY, bytes(value)
        return None, b""
    if isinstance(field_type, BooleanType):
        # Parquet booleans don't ship with bloom filters in practice.
        return None, b""
    if isinstance(field_type, DecimalType):
        # Iceberg decimals serialize as fixed bytes; mapping requires
        # awareness of (precision, scale, byte width). Skip for now — small
        # potential pruning win against meaningful complexity.
        return None, b""
    return None, b""
