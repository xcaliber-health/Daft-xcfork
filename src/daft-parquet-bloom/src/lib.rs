//! Daft bloom-filter shim: thin Rust FFI to a C++ shim that wraps libparquet.
//!
//! The crate exists for one reason: PyArrow's Cython wrapper does not expose
//! parquet bloom filters to Python. libparquet (which PyArrow ships in its
//! wheel) has full bloom read support. This crate links to that libparquet
//! through a small `extern "C"` shim and re-exports the API to Daft's Python
//! layer via PyO3.
//!
//! Runtime contract: PyArrow's libparquet/libarrow must already be loaded
//! into the process when this crate's PyO3 module is imported. Daft's
//! `daft/_parquet_bloom/__init__.py` is responsible for that preload step
//! before `import daft.daft` resolves these symbols.

#![allow(unsafe_code)] // FFI to a C++ shim is the whole point of this crate.

use std::ffi::{c_char, CStr, CString};

mod ffi {
    use std::ffi::c_char;

    #[repr(C)]
    pub struct DaftBloomReader {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn daft_bloom_open_local(path: *const c_char) -> *mut DaftBloomReader;
        pub fn daft_bloom_open_from_pyarrow_native_file(
            py_native_file: *mut std::ffi::c_void,
        ) -> *mut DaftBloomReader;
        pub fn daft_bloom_close(reader: *mut DaftBloomReader);
        pub fn daft_bloom_num_row_groups(reader: *const DaftBloomReader) -> i32;
        pub fn daft_bloom_probe(
            reader: *mut DaftBloomReader,
            row_group: i32,
            column_path: *const c_char,
            type_id: i32,
            value_bytes: *const u8,
            value_len: usize,
        ) -> i32;
        pub fn daft_bloom_last_error() -> *const c_char;
    }
}

/// Verdict returned from a single bloom probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeResult {
    /// Bloom reports the value is probably present. Caller must keep the row group.
    Present,
    /// Bloom proves the value is definitely absent. Caller may prune.
    Absent,
    /// No bloom filter exists for this (row_group, column). Inconclusive.
    NoBloom,
}

/// Owning handle around the C++ reader. Drop closes the underlying file.
pub struct BloomReader {
    raw: *mut ffi::DaftBloomReader,
}

// SAFETY: The C++ shim guarantees that a single handle is only ever accessed
// from one thread at a time. PyO3 requires Sync for #[pyclass]; access is
// serialised through Python's GIL in practice.
unsafe impl Send for BloomReader {}
unsafe impl Sync for BloomReader {}

impl Drop for BloomReader {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::daft_bloom_close(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

impl BloomReader {
    /// Open a local-filesystem parquet file. Use this only for plain
    /// `file://` paths; for any other protocol prefer
    /// [`Self::open_from_pyarrow_native_file`] so pyarrow's IO layer handles
    /// the transport.
    pub fn open_local(path: &str) -> Result<Self, String> {
        let c_path = CString::new(path)
            .map_err(|_| "path contains interior null byte".to_string())?;
        let raw = unsafe { ffi::daft_bloom_open_local(c_path.as_ptr()) };
        if raw.is_null() {
            return Err(last_error().unwrap_or_else(|| "open_local returned null".to_string()));
        }
        Ok(Self { raw })
    }

    /// Open from a raw pointer to a `pyarrow.NativeFile` PyObject.
    ///
    /// LIFETIME CONTRACT: the C++ shim wraps this via
    /// `arrow::py::PyReadableFile` which holds a BORROWED reference (no
    /// INCREF). The caller MUST keep the PyObject alive for the full
    /// lifetime of the returned `BloomReader`, NOT just for this call.
    /// Dropping the Python NativeFile while the reader still exists will
    /// crash the next probe.
    ///
    /// Safety: `py_native_file` must point to a live `pyarrow.NativeFile`
    /// (or anything `arrow::py::PyReadableFile` accepts) and that
    /// reference must remain valid until `BloomReader::drop` runs.
    pub unsafe fn open_from_pyarrow_native_file(
        py_native_file: *mut std::ffi::c_void,
    ) -> Result<Self, String> {
        let raw = unsafe { ffi::daft_bloom_open_from_pyarrow_native_file(py_native_file) };
        if raw.is_null() {
            return Err(last_error()
                .unwrap_or_else(|| "open_from_pyarrow_native_file returned null".to_string()));
        }
        Ok(Self { raw })
    }

    pub fn num_row_groups(&self) -> i32 {
        unsafe { ffi::daft_bloom_num_row_groups(self.raw) }
    }

    /// Probe a single (row_group, column) with one literal.
    ///
    /// `type_id` follows `parquet::Type::type`:
    /// BOOLEAN=0, INT32=1, INT64=2, INT96=3, FLOAT=4, DOUBLE=5,
    /// BYTE_ARRAY=6, FIXED_LEN_BYTE_ARRAY=7.
    pub fn probe(
        &mut self,
        row_group: i32,
        column_path: &str,
        type_id: i32,
        value: &[u8],
    ) -> Result<ProbeResult, String> {
        let c_col = CString::new(column_path)
            .map_err(|_| "column path contains interior null byte".to_string())?;
        let rc = unsafe {
            ffi::daft_bloom_probe(
                self.raw,
                row_group,
                c_col.as_ptr(),
                type_id,
                value.as_ptr(),
                value.len(),
            )
        };
        match rc {
            1 => Ok(ProbeResult::Present),
            0 => Ok(ProbeResult::Absent),
            -1 => Ok(ProbeResult::NoBloom),
            _ => Err(last_error().unwrap_or_else(|| format!("probe failed: rc={rc}"))),
        }
    }
}

fn last_error() -> Option<String> {
    let p = unsafe { ffi::daft_bloom_last_error() };
    if p.is_null() {
        None
    } else {
        let s = unsafe { CStr::from_ptr(p as *const c_char) };
        Some(s.to_string_lossy().into_owned())
    }
}

#[cfg(feature = "python")]
pub mod python {
    use pyo3::prelude::*;

    use super::{BloomReader, ProbeResult};

    /// Python-facing wrapper. Keeps the underlying C++ reader alive for the
    /// lifetime of the Python object. Calls release the GIL while crossing
    /// into libparquet so concurrent probes don't serialize through Python.
    #[pyclass(module = "daft.daft", name = "ParquetBloomReader")]
    pub struct PyBloomReader {
        inner: BloomReader,
    }

    #[pymethods]
    impl PyBloomReader {
        /// Open a local parquet file. Remote callers should use
        /// [`Self::open_from_native_file`] so pyarrow's IO layer handles
        /// the transport — this entrypoint is for `file://` only.
        #[staticmethod]
        fn open_local(path: &str) -> PyResult<Self> {
            BloomReader::open_local(path)
                .map(|inner| Self { inner })
                .map_err(pyo3::exceptions::PyIOError::new_err)
        }

        /// Open from a `pyarrow.NativeFile` (any input that pyarrow's
        /// `unwrap_random_access_file` accepts). The underlying transport
        /// is whatever the NativeFile was constructed from — local file,
        /// S3, GCS, ABFS, HDFS, etc.
        ///
        /// Typical pyiceberg usage:
        /// ```python
        /// nf = iceberg_table.io.new_input(path).open()  # -> NativeFile
        /// reader = ParquetBloomReader.open_from_native_file(nf)
        /// ```
        #[staticmethod]
        fn open_from_native_file(native_file: Bound<PyAny>) -> PyResult<Self> {
            let raw_ptr = native_file.as_ptr() as *mut std::ffi::c_void;
            // SAFETY: native_file is a live PyObject for the duration of
            // this call (Bound<PyAny> holds a strong reference).
            unsafe { BloomReader::open_from_pyarrow_native_file(raw_ptr) }
                .map(|inner| Self { inner })
                .map_err(pyo3::exceptions::PyIOError::new_err)
        }

        #[getter]
        fn num_row_groups(&self) -> i32 {
            self.inner.num_row_groups()
        }

        /// Probe a single (row_group, column). Returns one of:
        ///   1  bloom says probably present (keep row group)
        ///   0  bloom says definitely absent (prune row group)
        ///  -1  no bloom filter on this (row_group, column) — inconclusive
        ///
        /// `type_id` follows the parquet::Type enum: INT32=1, INT64=2,
        /// FLOAT=4, DOUBLE=5, BYTE_ARRAY=6, FIXED_LEN_BYTE_ARRAY=7.
        fn probe(
            &mut self,
            py: Python,
            row_group: i32,
            column: &str,
            type_id: i32,
            value: &[u8],
        ) -> PyResult<i32> {
            let res = py.detach(|| self.inner.probe(row_group, column, type_id, value));
            match res {
                Ok(ProbeResult::Present) => Ok(1),
                Ok(ProbeResult::Absent) => Ok(0),
                Ok(ProbeResult::NoBloom) => Ok(-1),
                Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
            }
        }
    }

    pub fn register_modules(parent: &Bound<PyModule>) -> PyResult<()> {
        parent.add_class::<PyBloomReader>()?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register_modules;
