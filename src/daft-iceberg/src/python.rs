use arrow::array::{ArrayRef, make_array};
use common_arrow_ffi::{ToPyArrow, array_to_rust};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyModule, PyModuleMethods},
    wrap_pyfunction,
};

use crate::{
    errors::IcebergRewriteError,
    options::{JobOrder, RewriteOptions},
    planner::{CandidateFile, FileGroup, plan_file_groups},
    zorder::build_zorder_key_array,
};

pyo3::create_exception!(
    daft.daft,
    EqualityDeletesPresentError,
    pyo3::exceptions::PyException,
    "Equality deletes prevent rewrite; apply them first."
);

fn err_to_py(e: IcebergRewriteError) -> PyErr {
    match e {
        IcebergRewriteError::EqualityDeletesPresent { ref sample, .. } => {
            EqualityDeletesPresentError::new_err(format!(
                "equality deletes present in files: {sample:?}"
            ))
        }
        IcebergRewriteError::InvalidOption { .. } => PyValueError::new_err(e.to_string()),
        IcebergRewriteError::UnsupportedZOrderType { .. } => PyValueError::new_err(e.to_string()),
        IcebergRewriteError::UnknownOutputSpec { .. } => PyValueError::new_err(e.to_string()),
    }
}

macro_rules! get_opt {
    ($d:expr, $key:literal, $ty:ty) => {{
        match $d.get_item($key)? {
            Some(v) => Some(v.extract::<$ty>()?),
            None => None,
        }
    }};
}

fn parse_options(py_opts: &Bound<'_, PyDict>) -> PyResult<RewriteOptions> {
    let mut o = RewriteOptions::default();
    if let Some(v) = get_opt!(py_opts, "target-file-size-bytes", u64) {
        o.target_file_size_bytes = v;
    }
    if let Some(v) = get_opt!(py_opts, "min-input-files", u32) {
        o.min_input_files = v;
    }
    if let Some(v) = get_opt!(py_opts, "max-file-group-size-bytes", u64) {
        o.max_file_group_size_bytes = v;
    }
    if let Some(v) = get_opt!(py_opts, "delete-file-threshold", u32) {
        o.delete_file_threshold = v;
    }
    if let Some(v) = get_opt!(py_opts, "rewrite-all", bool) {
        o.rewrite_all = v;
    }
    if let Some(v) = get_opt!(py_opts, "partial-progress.enabled", bool) {
        o.partial_progress_enabled = v;
    }
    if let Some(v) = get_opt!(py_opts, "partial-progress.max-commits", u32) {
        o.partial_progress_max_commits = v;
    }
    if let Some(v) = get_opt!(py_opts, "partial-progress.max-failed-commits", u32) {
        o.partial_progress_max_failed_commits = Some(v);
    }
    if let Some(v) = get_opt!(py_opts, "max-concurrent-file-group-rewrites", u32) {
        o.max_concurrent_file_group_rewrites = v;
    }
    if let Some(v) = get_opt!(py_opts, "output-spec-id", i32) {
        o.output_spec_id = Some(v);
    }
    if let Some(v) = get_opt!(py_opts, "use-starting-sequence-number", bool) {
        o.use_starting_sequence_number = v;
    }
    if let Some(v) = get_opt!(py_opts, "remove-dangling-deletes", bool) {
        o.remove_dangling_deletes = v;
    }
    if let Some(v) = get_opt!(py_opts, "max-files-to-rewrite", u32) {
        o.max_files_to_rewrite = Some(v);
    }
    if let Some(v) = get_opt!(py_opts, "min-file-size-bytes", u64) {
        o.min_file_size_bytes = Some(v);
    }
    if let Some(v) = get_opt!(py_opts, "max-file-size-bytes", u64) {
        o.max_file_size_bytes = Some(v);
    }
    if let Some(v) = get_opt!(py_opts, "rewrite-job-order", String) {
        o.job_order = JobOrder::parse(&v).map_err(err_to_py)?;
    }
    if let Some(v) = get_opt!(py_opts, "compression-factor", f64) {
        o.compression_factor = v;
    }
    if let Some(v) = get_opt!(py_opts, "max-output-size", u64) {
        o.zorder_max_output_size = v;
    }
    if let Some(v) = get_opt!(py_opts, "var-length-contribution", u32) {
        o.zorder_var_length_contribution = v;
    }
    o.validate().map_err(err_to_py)?;
    Ok(o)
}

fn candidate_from_dict(d: &Bound<'_, PyDict>) -> PyResult<CandidateFile> {
    let path = get_opt!(d, "path", String)
        .ok_or_else(|| PyValueError::new_err("candidate missing `path`"))?;
    let size_bytes = get_opt!(d, "size_bytes", u64)
        .ok_or_else(|| PyValueError::new_err("candidate missing `size_bytes`"))?;
    let partition_key = get_opt!(d, "partition_key", String).unwrap_or_default();
    let partition_spec_id = get_opt!(d, "partition_spec_id", i32).unwrap_or(0);
    let positional_delete_paths =
        get_opt!(d, "positional_delete_paths", Vec<String>).unwrap_or_default();
    let has_equality_deletes = get_opt!(d, "has_equality_deletes", bool).unwrap_or(false);
    Ok(CandidateFile {
        path,
        size_bytes,
        partition_key,
        partition_spec_id,
        positional_delete_paths,
        has_equality_deletes,
    })
}

fn group_to_dict<'py>(py: Python<'py>, g: &FileGroup) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("partition_key", &g.partition_key)?;
    out.set_item("output_spec_id", g.output_spec_id)?;
    out.set_item("total_bytes", g.total_bytes)?;
    let files = PyList::empty(py);
    for f in &g.files {
        let fd = PyDict::new(py);
        fd.set_item("path", &f.path)?;
        fd.set_item("size_bytes", f.size_bytes)?;
        fd.set_item("partition_key", &f.partition_key)?;
        fd.set_item("partition_spec_id", f.partition_spec_id)?;
        fd.set_item("positional_delete_paths", &f.positional_delete_paths)?;
        fd.set_item("has_equality_deletes", f.has_equality_deletes)?;
        files.append(fd)?;
    }
    out.set_item("files", files)?;
    Ok(out)
}

/// Group candidate files into rewrite units.
///
/// `candidates` is a list of dicts shaped like `CandidateFile`. `options` is a dict of
/// option keys (kebab-case). Returns a list of group dicts ordered by `rewrite-job-order`.
#[pyfunction]
#[pyo3(signature = (candidates, options, current_spec_id))]
fn plan_file_groups_py<'py>(
    py: Python<'py>,
    candidates: &Bound<'py, PyList>,
    options: &Bound<'py, PyDict>,
    current_spec_id: i32,
) -> PyResult<Bound<'py, PyList>> {
    let opts = parse_options(options)?;
    let mut cs = Vec::with_capacity(candidates.len());
    for item in candidates.iter() {
        let d = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("candidate must be a dict"))?;
        cs.push(candidate_from_dict(&d)?);
    }
    let groups = plan_file_groups(cs, &opts, current_spec_id).map_err(err_to_py)?;
    let out = PyList::empty(py);
    for g in &groups {
        out.append(group_to_dict(py, g)?)?;
    }
    Ok(out)
}

/// Validate a raw options dict and return a normalized dict with defaults filled in.
#[pyfunction]
#[pyo3(signature = (options))]
fn validate_options_py<'py>(
    py: Python<'py>,
    options: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = parse_options(options)?;
    let d = PyDict::new(py);
    d.set_item("target-file-size-bytes", o.target_file_size_bytes)?;
    d.set_item("min-input-files", o.min_input_files)?;
    d.set_item("max-file-group-size-bytes", o.max_file_group_size_bytes)?;
    d.set_item("delete-file-threshold", o.delete_file_threshold)?;
    d.set_item("rewrite-all", o.rewrite_all)?;
    d.set_item("partial-progress.enabled", o.partial_progress_enabled)?;
    d.set_item(
        "partial-progress.max-commits",
        o.partial_progress_max_commits,
    )?;
    d.set_item(
        "partial-progress.max-failed-commits",
        o.effective_max_failed_commits(),
    )?;
    d.set_item(
        "max-concurrent-file-group-rewrites",
        o.max_concurrent_file_group_rewrites,
    )?;
    match o.output_spec_id {
        Some(v) => d.set_item("output-spec-id", v)?,
        None => {}
    }
    d.set_item(
        "use-starting-sequence-number",
        o.use_starting_sequence_number,
    )?;
    d.set_item("remove-dangling-deletes", o.remove_dangling_deletes)?;
    if let Some(v) = o.max_files_to_rewrite {
        d.set_item("max-files-to-rewrite", v)?;
    }
    d.set_item("min-file-size-bytes", o.effective_min_file_size_bytes())?;
    d.set_item("max-file-size-bytes", o.effective_max_file_size_bytes())?;
    d.set_item(
        "rewrite-job-order",
        match o.job_order {
            JobOrder::BytesAsc => "bytes-asc",
            JobOrder::BytesDesc => "bytes-desc",
            JobOrder::FilesAsc => "files-asc",
            JobOrder::FilesDesc => "files-desc",
            JobOrder::None => "none",
        },
    )?;
    d.set_item("compression-factor", o.compression_factor)?;
    d.set_item("max-output-size", o.zorder_max_output_size)?;
    d.set_item(
        "var-length-contribution",
        o.zorder_var_length_contribution,
    )?;
    Ok(d)
}

/// Build the synthetic z-order key column as a `pyarrow.Array` of binary.
///
/// `arrays` is a Python list of pyarrow arrays (one per z-order column, equal length).
/// `var_length_contribution` controls how many bytes string/binary columns contribute;
/// `max_output_size` caps the final interleaved key byte length.
#[pyfunction]
#[pyo3(signature = (arrays, var_length_contribution, max_output_size))]
fn build_zorder_key_py<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyList>,
    var_length_contribution: u32,
    max_output_size: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let mut rs_arrays: Vec<ArrayRef> = Vec::with_capacity(arrays.len());
    for a in arrays.iter() {
        let (data, _field) = array_to_rust(&a)?;
        rs_arrays.push(make_array(data));
    }
    let _ = py;
    let data = build_zorder_key_array(&rs_arrays, var_length_contribution, max_output_size)
        .map_err(err_to_py)?;
    data.to_pyarrow(arrays.py())
}

pub fn register_modules(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "_iceberg")?;
    m.add_function(wrap_pyfunction!(plan_file_groups_py, &m)?)?;
    m.add_function(wrap_pyfunction!(validate_options_py, &m)?)?;
    m.add_function(wrap_pyfunction!(build_zorder_key_py, &m)?)?;
    m.add(
        "EqualityDeletesPresentError",
        py.get_type::<EqualityDeletesPresentError>(),
    )?;
    parent.add_submodule(&m)?;
    // Mirror parent module name so `import daft.daft._iceberg` resolves.
    py.import("sys")?
        .getattr("modules")?
        .set_item("daft.daft._iceberg", &m)
        .or_else(|_| -> PyResult<()> { Ok(()) })?;
    Ok(())
}
