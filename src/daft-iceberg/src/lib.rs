//! Iceberg table-maintenance primitives.
//!
//! Provides file-group planning for `rewrite_data_files` and the option/strategy types
//! consumed by both Rust and Python orchestration. The per-group rewrite executor reuses
//! existing Daft read/write crates; this crate owns only the pure planning logic.

pub mod errors;
pub mod options;
pub mod orphan;
pub mod planner;
pub mod zorder;

#[cfg(feature = "python")]
pub mod python;

pub use errors::IcebergRewriteError;
pub use options::{
    JobOrder, NullOrder, RewriteOptions, SortColumn, SortDirection, Strategy, ZOrderKey,
};
pub use orphan::orphan_diff;
pub use planner::{CandidateFile, FileGroup, plan_file_groups};
pub use zorder::{ZORDER_KEY_COL, build_zorder_key_array, interleave_bits, normalize_to_ordered_bytes};

#[cfg(feature = "python")]
pub fn register_modules(parent: &pyo3::Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    python::register_modules(parent)
}
