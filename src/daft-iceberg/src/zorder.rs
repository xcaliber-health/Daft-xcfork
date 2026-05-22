//! Z-order curve primitives.
//!
//! Two-step pipeline:
//!   1. `normalize_to_ordered_bytes` — encode each column's values as fixed-width byte
//!      arrays whose lexicographic order matches the value's natural order.
//!   2. `interleave_bits` — bit-interleave the per-column byte arrays into a single
//!      binary key, then sort by that key to produce a z-order (Morton) curve.
//!
//! Nulls normalize to an empty byte slice, which sorts before any concrete value
//! (nulls-first semantics, matching Iceberg's `ZOrderByteUtils`).

use arrow::array::{
    Array, ArrayData, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array,
    Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, LargeBinaryArray,
    LargeStringArray, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow::datatypes::{DataType, TimeUnit};

use crate::errors::IcebergRewriteError;

/// Column name used for the synthetic interleaved key during a z-order rewrite. The
/// column is appended just long enough to sort, then projected away before write.
pub const ZORDER_KEY_COL: &str = "__daft_zorder_key__";

/// Encode each row of `array` as an ordered byte slice. Returns one `Vec<u8>` per row.
///
/// The per-type byte widths match Iceberg's `ZOrderByteUtils` so that values from
/// different Daft installations interleave identically. Nulls produce an empty slice.
pub fn normalize_to_ordered_bytes(
    array: &dyn Array,
    var_length_contribution: u32,
) -> Result<Vec<Vec<u8>>, IcebergRewriteError> {
    let n = array.len();
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(n);
    let var_len = var_length_contribution as usize;

    match array.data_type() {
        DataType::Boolean => fill(&mut out, array, n, |i, nulls| {
            let a = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            if nulls.is_null(i) {
                vec![]
            } else if a.value(i) {
                vec![0xFFu8]
            } else {
                vec![0x00u8]
            }
        }),
        DataType::Int8 => fill_int!(out, array, Int8Array, i8, n, 1),
        DataType::Int16 => fill_int!(out, array, Int16Array, i16, n, 2),
        DataType::Int32 => fill_int!(out, array, Int32Array, i32, n, 4),
        DataType::Int64 => fill_int!(out, array, Int64Array, i64, n, 8),
        DataType::UInt8 => fill_uint!(out, array, UInt8Array, u8, n, 1),
        DataType::UInt16 => fill_uint!(out, array, UInt16Array, u16, n, 2),
        DataType::UInt32 => fill_uint!(out, array, UInt32Array, u32, n, 4),
        DataType::UInt64 => fill_uint!(out, array, UInt64Array, u64, n, 8),
        DataType::Float32 => fill(&mut out, array, n, |i, nulls| {
            let a = array.as_any().downcast_ref::<Float32Array>().unwrap();
            if nulls.is_null(i) {
                vec![]
            } else {
                encode_float32(a.value(i)).to_vec()
            }
        }),
        DataType::Float64 => fill(&mut out, array, n, |i, nulls| {
            let a = array.as_any().downcast_ref::<Float64Array>().unwrap();
            if nulls.is_null(i) {
                vec![]
            } else {
                encode_float64(a.value(i)).to_vec()
            }
        }),
        DataType::Date32 => fill_int!(out, array, Date32Array, i32, n, 4),
        DataType::Timestamp(unit, _) => match unit {
            TimeUnit::Second => fill_int!(out, array, TimestampSecondArray, i64, n, 8),
            TimeUnit::Millisecond => fill_int!(out, array, TimestampMillisecondArray, i64, n, 8),
            TimeUnit::Microsecond => fill_int!(out, array, TimestampMicrosecondArray, i64, n, 8),
            TimeUnit::Nanosecond => fill_int!(out, array, TimestampNanosecondArray, i64, n, 8),
        },
        DataType::Decimal128(_, _) => fill(&mut out, array, n, |i, nulls| {
            let a = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
            if nulls.is_null(i) {
                vec![]
            } else {
                let mut bytes = a.value(i).to_be_bytes();
                bytes[0] ^= 0x80;
                bytes.to_vec()
            }
        }),
        DataType::Utf8 => fill_var!(out, array, StringArray, str_to_bytes, n, var_len),
        DataType::LargeUtf8 => fill_var!(out, array, LargeStringArray, str_to_bytes, n, var_len),
        DataType::Binary => fill_var!(out, array, BinaryArray, bytes_passthrough, n, var_len),
        DataType::LargeBinary => fill_var!(
            out,
            array,
            LargeBinaryArray,
            bytes_passthrough,
            n,
            var_len
        ),
        other => {
            return Err(IcebergRewriteError::UnsupportedZOrderType {
                column: String::new(),
                dtype: format!("{other:?}"),
            });
        }
    }
    Ok(out)
}

fn fill<F: FnMut(usize, &dyn ArrayNullCheck) -> Vec<u8>>(
    out: &mut Vec<Vec<u8>>,
    array: &dyn Array,
    n: usize,
    mut f: F,
) {
    let null_check = NullCheckImpl { array };
    for i in 0..n {
        out.push(f(i, &null_check));
    }
}

trait ArrayNullCheck {
    fn is_null(&self, i: usize) -> bool;
}
struct NullCheckImpl<'a> {
    array: &'a dyn Array,
}
impl<'a> ArrayNullCheck for NullCheckImpl<'a> {
    fn is_null(&self, i: usize) -> bool {
        self.array.is_null(i)
    }
}

macro_rules! fill_int {
    ($out:expr, $array:expr, $arr_ty:ty, $prim_ty:ty, $n:expr, $width:literal) => {{
        let a = $array.as_any().downcast_ref::<$arr_ty>().unwrap();
        for i in 0..$n {
            if a.is_null(i) {
                $out.push(vec![]);
            } else {
                let v = a.value(i) as $prim_ty;
                let mut bytes = v.to_be_bytes();
                // Flip the sign bit so that two's-complement order matches lexicographic order.
                bytes[0] ^= 0x80;
                $out.push(bytes.to_vec());
            }
        }
    }};
}
pub(crate) use fill_int;

macro_rules! fill_uint {
    ($out:expr, $array:expr, $arr_ty:ty, $prim_ty:ty, $n:expr, $width:literal) => {{
        let a = $array.as_any().downcast_ref::<$arr_ty>().unwrap();
        for i in 0..$n {
            if a.is_null(i) {
                $out.push(vec![]);
            } else {
                let v = a.value(i) as $prim_ty;
                $out.push(v.to_be_bytes().to_vec());
            }
        }
    }};
}
pub(crate) use fill_uint;

fn str_to_bytes(s: &str) -> &[u8] {
    s.as_bytes()
}
fn bytes_passthrough(b: &[u8]) -> &[u8] {
    b
}

macro_rules! fill_var {
    ($out:expr, $array:expr, $arr_ty:ty, $extract:expr, $n:expr, $var_len:expr) => {{
        let a = $array.as_any().downcast_ref::<$arr_ty>().unwrap();
        for i in 0..$n {
            if a.is_null(i) {
                $out.push(vec![]);
            } else {
                let raw: &[u8] = $extract(a.value(i));
                let mut buf = vec![0u8; $var_len];
                let take = raw.len().min($var_len);
                buf[..take].copy_from_slice(&raw[..take]);
                $out.push(buf);
            }
        }
    }};
}
pub(crate) use fill_var;

fn encode_float32(v: f32) -> [u8; 4] {
    let bits = v.to_bits();
    // Flip sign bit for non-negatives; flip all bits for negatives. NaN sorts last
    // because its raw bit pattern's top bits are 1.
    let mapped = if bits & 0x8000_0000 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    };
    mapped.to_be_bytes()
}

fn encode_float64(v: f64) -> [u8; 8] {
    let bits = v.to_bits();
    let mapped = if bits & 0x8000_0000_0000_0000 == 0 {
        bits ^ 0x8000_0000_0000_0000
    } else {
        !bits
    };
    mapped.to_be_bytes()
}

/// Interleave the bits of each row's per-column byte arrays into a single binary key.
///
/// `columns[c][r]` is the normalized bytes for column `c` row `r`. Output has `n_rows`
/// entries each of length `output_size`. Shorter columns are zero-padded; rows where
/// every column is empty (all-null) produce an empty key.
pub fn interleave_bits(columns: &[Vec<Vec<u8>>], output_size: u64) -> Vec<Vec<u8>> {
    if columns.is_empty() {
        return Vec::new();
    }
    let n_rows = columns[0].len();
    debug_assert!(columns.iter().all(|c| c.len() == n_rows));
    let n_cols = columns.len();
    let output_bytes = output_size as usize;
    let output_bits = output_bytes * 8;

    let mut out: Vec<Vec<u8>> = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let all_null = columns.iter().all(|c| c[r].is_empty());
        if all_null {
            out.push(Vec::new());
            continue;
        }
        let mut buf = vec![0u8; output_bytes];
        let mut out_bit = 0usize;
        let max_col_bits = columns
            .iter()
            .map(|c| c[r].len() * 8)
            .max()
            .unwrap_or(0);
        let mut src_bit = 0usize;
        while out_bit < output_bits && src_bit < max_col_bits {
            for c in 0..n_cols {
                if out_bit >= output_bits {
                    break;
                }
                let bytes = &columns[c][r];
                let bit = if src_bit < bytes.len() * 8 {
                    let byte = bytes[src_bit / 8];
                    (byte >> (7 - (src_bit % 8))) & 1
                } else {
                    0
                };
                if bit != 0 {
                    buf[out_bit / 8] |= 1 << (7 - (out_bit % 8));
                }
                out_bit += 1;
            }
            src_bit += 1;
        }
        out.push(buf);
    }
    out
}

/// Build the z-order key as a `BinaryArray` for a set of input arrays.
///
/// Each row's key is the bit-interleave of per-column ordered byte encodings, truncated
/// to `max_output_size` bytes. The returned array length equals `arrays[0].len()`.
pub fn build_zorder_key_array(
    arrays: &[ArrayRef],
    var_length_contribution: u32,
    max_output_size: u64,
) -> Result<ArrayData, IcebergRewriteError> {
    if arrays.is_empty() {
        return Err(IcebergRewriteError::InvalidOption {
            name: "zorder_by".into(),
            reason: "expected at least one column".into(),
        });
    }
    let n_rows = arrays[0].len();
    for a in arrays.iter() {
        if a.len() != n_rows {
            return Err(IcebergRewriteError::InvalidOption {
                name: "zorder_by".into(),
                reason: "all z-order columns must share row count".into(),
            });
        }
    }
    let mut per_column: Vec<Vec<Vec<u8>>> = Vec::with_capacity(arrays.len());
    for a in arrays {
        per_column.push(normalize_to_ordered_bytes(a.as_ref(), var_length_contribution)?);
    }
    let keys = interleave_bits(&per_column, max_output_size);
    let arr = BinaryArray::from_iter_values(keys.iter().map(|v| v.as_slice()));
    Ok(arr.into_data())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn assert_lex_order_matches<T, F>(values: Vec<T>, encode: F)
    where
        T: PartialOrd + Copy + std::fmt::Debug,
        F: Fn(T) -> Vec<u8>,
    {
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let encoded: Vec<(T, Vec<u8>)> = sorted.iter().map(|v| (*v, encode(*v))).collect();
        for w in encoded.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "lex order broken: {:?} -> {:?} vs {:?} -> {:?}",
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            );
        }
    }

    #[test]
    fn int32_ordering_matches_lex() {
        let vals = vec![-100i32, -1, 0, 1, 100, i32::MIN, i32::MAX];
        assert_lex_order_matches(vals, |v| {
            let arr = Int32Array::from(vec![v]);
            let bytes = normalize_to_ordered_bytes(&arr, 8).unwrap();
            bytes[0].clone()
        });
    }

    #[test]
    fn int64_ordering_matches_lex() {
        let vals = vec![-1_000_000_000i64, -1, 0, 1, 1_000_000_000, i64::MIN, i64::MAX];
        assert_lex_order_matches(vals, |v| {
            let arr = Int64Array::from(vec![v]);
            let bytes = normalize_to_ordered_bytes(&arr, 8).unwrap();
            bytes[0].clone()
        });
    }

    #[test]
    fn float64_ordering_matches_lex() {
        // NaN deliberately excluded: behavior is left to upstream callers.
        let vals = vec![
            -f64::INFINITY,
            -1e10,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1e10,
            f64::INFINITY,
        ];
        assert_lex_order_matches(vals, |v| {
            let arr = Float64Array::from(vec![v]);
            let bytes = normalize_to_ordered_bytes(&arr, 8).unwrap();
            bytes[0].clone()
        });
    }

    #[test]
    fn null_becomes_empty_bytes() {
        let arr = Int32Array::from(vec![Some(1), None, Some(2)]);
        let bytes = normalize_to_ordered_bytes(&arr, 8).unwrap();
        assert_eq!(bytes[1], Vec::<u8>::new());
        assert!(!bytes[0].is_empty() && !bytes[2].is_empty());
    }

    #[test]
    fn utf8_truncates_and_pads() {
        let arr = StringArray::from(vec![Some("a"), Some("abcdefghijklmnop"), None]);
        let bytes = normalize_to_ordered_bytes(&arr, 4).unwrap();
        assert_eq!(bytes[0], b"a\x00\x00\x00");
        assert_eq!(bytes[1], b"abcd");
        assert_eq!(bytes[2], Vec::<u8>::new());
    }

    #[test]
    fn interleave_two_columns_known_fixture() {
        // Two columns, one byte each: 0xAA and 0xFF.
        // Bits of A = 1010_1010, bits of B = 1111_1111.
        // Interleaved (A bit then B bit) gives: 11 11 11 11 11 11 11 11 starting with A=1,B=1.
        // Sequence: A0=1,B0=1, A1=0,B1=1, A2=1,B2=1, A3=0,B3=1, A4=1,B4=1, A5=0,B5=1, A6=1,B6=1, A7=0,B7=1
        // → 11 01 11 01 11 01 11 01 = 0xDD 0xDD
        let cols = vec![vec![vec![0xAAu8]], vec![vec![0xFFu8]]];
        let out = interleave_bits(&cols, 2);
        assert_eq!(out[0], vec![0xDDu8, 0xDDu8]);
    }

    #[test]
    fn interleave_all_null_yields_empty() {
        let cols = vec![vec![Vec::<u8>::new()], vec![Vec::<u8>::new()]];
        let out = interleave_bits(&cols, 4);
        assert_eq!(out[0], Vec::<u8>::new());
    }

    #[test]
    fn interleave_respects_output_size_truncation() {
        let cols = vec![vec![vec![0xFFu8; 16]], vec![vec![0xFFu8; 16]]];
        let out = interleave_bits(&cols, 4);
        assert_eq!(out[0].len(), 4);
        assert!(out[0].iter().all(|b| *b == 0xFFu8));
    }

    #[test]
    fn build_key_array_round_trip() {
        let a = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let b = Arc::new(StringArray::from(vec!["x", "y", "z"])) as ArrayRef;
        let data = build_zorder_key_array(&[a, b], 8, 16).unwrap();
        let arr = BinaryArray::from(data);
        assert_eq!(arr.len(), 3);
        // Keys must differ pairwise.
        assert_ne!(arr.value(0), arr.value(1));
        assert_ne!(arr.value(1), arr.value(2));
    }
}
