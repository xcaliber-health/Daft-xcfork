use serde::{Deserialize, Serialize};

use crate::errors::IcebergRewriteError;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

pub const DEFAULT_TARGET_FILE_SIZE_BYTES: u64 = 512 * MIB;
pub const MIN_TARGET_FILE_SIZE_BYTES: u64 = MIB;
pub const MAX_TARGET_FILE_SIZE_BYTES: u64 = 5 * GIB;
pub const DEFAULT_MIN_INPUT_FILES: u32 = 5;
pub const DEFAULT_MAX_GROUP_BYTES: u64 = 100 * GIB;
pub const DEFAULT_DELETE_FILE_THRESHOLD: u32 = u32::MAX;
pub const DEFAULT_MAX_COMMITS: u32 = 10;
pub const DEFAULT_MAX_CONCURRENT: u32 = 5;
pub const DEFAULT_ZORDER_VAR_LEN_CONTRIBUTION: u32 = 8;
// Per-row interleaved key cap. Primitive columns contribute 8 bytes each and
// string columns contribute up to `var_length_contribution`; 4096 is generous
// enough for ~hundreds of columns and keeps the per-row allocation bounded.
pub const DEFAULT_ZORDER_MAX_OUTPUT_SIZE: u64 = 4096;
pub const MAX_ZORDER_MAX_OUTPUT_SIZE: u64 = MIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobOrder {
    BytesAsc,
    BytesDesc,
    FilesAsc,
    FilesDesc,
    None,
}

impl JobOrder {
    pub fn parse(s: &str) -> Result<Self, IcebergRewriteError> {
        match s {
            "bytes-asc" => Ok(Self::BytesAsc),
            "bytes-desc" => Ok(Self::BytesDesc),
            "files-asc" => Ok(Self::FilesAsc),
            "files-desc" => Ok(Self::FilesDesc),
            "none" => Ok(Self::None),
            other => Err(IcebergRewriteError::InvalidOption {
                name: "rewrite-job-order".into(),
                reason: format!(
                    "expected one of bytes-asc|bytes-desc|files-asc|files-desc|none, got `{other}`"
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteOptions {
    pub target_file_size_bytes: u64,
    pub min_input_files: u32,
    pub max_file_group_size_bytes: u64,
    pub delete_file_threshold: u32,
    pub rewrite_all: bool,
    pub partial_progress_enabled: bool,
    pub partial_progress_max_commits: u32,
    pub partial_progress_max_failed_commits: Option<u32>,
    pub max_concurrent_file_group_rewrites: u32,
    pub output_spec_id: Option<i32>,
    pub use_starting_sequence_number: bool,
    pub remove_dangling_deletes: bool,
    pub max_files_to_rewrite: Option<u32>,
    pub min_file_size_bytes: Option<u64>,
    pub max_file_size_bytes: Option<u64>,
    pub job_order: JobOrder,
    pub compression_factor: f64,
    pub zorder_max_output_size: u64,
    pub zorder_var_length_contribution: u32,
}

impl Default for RewriteOptions {
    fn default() -> Self {
        Self {
            target_file_size_bytes: DEFAULT_TARGET_FILE_SIZE_BYTES,
            min_input_files: DEFAULT_MIN_INPUT_FILES,
            max_file_group_size_bytes: DEFAULT_MAX_GROUP_BYTES,
            delete_file_threshold: DEFAULT_DELETE_FILE_THRESHOLD,
            rewrite_all: false,
            partial_progress_enabled: false,
            partial_progress_max_commits: DEFAULT_MAX_COMMITS,
            partial_progress_max_failed_commits: None,
            max_concurrent_file_group_rewrites: DEFAULT_MAX_CONCURRENT,
            output_spec_id: None,
            use_starting_sequence_number: true,
            remove_dangling_deletes: false,
            max_files_to_rewrite: None,
            min_file_size_bytes: None,
            max_file_size_bytes: None,
            job_order: JobOrder::BytesDesc,
            compression_factor: 1.0,
            zorder_max_output_size: DEFAULT_ZORDER_MAX_OUTPUT_SIZE,
            zorder_var_length_contribution: DEFAULT_ZORDER_VAR_LEN_CONTRIBUTION,
        }
    }
}

impl RewriteOptions {
    /// Lower size threshold for rewrite eligibility. Defaults to 75% of target.
    pub fn effective_min_file_size_bytes(&self) -> u64 {
        match self.min_file_size_bytes {
            Some(v) => v,
            None => (self.target_file_size_bytes as f64 * 0.75) as u64,
        }
    }

    /// Upper size threshold for rewrite eligibility. Defaults to 180% of target.
    pub fn effective_max_file_size_bytes(&self) -> u64 {
        match self.max_file_size_bytes {
            Some(v) => v,
            None => (self.target_file_size_bytes as f64 * 1.80) as u64,
        }
    }

    /// Failed-commit budget under partial-progress. Defaults to `partial_progress_max_commits`.
    pub fn effective_max_failed_commits(&self) -> u32 {
        self.partial_progress_max_failed_commits
            .unwrap_or(self.partial_progress_max_commits)
    }
}

impl RewriteOptions {
    pub fn validate(&self) -> Result<(), IcebergRewriteError> {
        let invalid = |name: &str, reason: String| IcebergRewriteError::InvalidOption {
            name: name.into(),
            reason,
        };

        if !(MIN_TARGET_FILE_SIZE_BYTES..=MAX_TARGET_FILE_SIZE_BYTES)
            .contains(&self.target_file_size_bytes)
        {
            return Err(invalid(
                "target-file-size-bytes",
                format!(
                    "must be in [{MIN_TARGET_FILE_SIZE_BYTES}, {MAX_TARGET_FILE_SIZE_BYTES}], got {}",
                    self.target_file_size_bytes
                ),
            ));
        }
        if self.min_input_files < 2 {
            return Err(invalid(
                "min-input-files",
                format!("must be >= 2, got {}", self.min_input_files),
            ));
        }
        if self.max_file_group_size_bytes < self.target_file_size_bytes {
            return Err(invalid(
                "max-file-group-size-bytes",
                "must be >= target-file-size-bytes".into(),
            ));
        }
        if self.partial_progress_enabled && self.partial_progress_max_commits == 0 {
            return Err(invalid(
                "partial-progress.max-commits",
                "must be >= 1 when partial-progress.enabled = true".into(),
            ));
        }
        if self.max_concurrent_file_group_rewrites == 0 {
            return Err(invalid(
                "max-concurrent-file-group-rewrites",
                "must be >= 1".into(),
            ));
        }
        if !(self.compression_factor > 0.0 && self.compression_factor.is_finite()) {
            return Err(invalid(
                "compression-factor",
                format!("must be > 0 and finite, got {}", self.compression_factor),
            ));
        }
        if !(8..=MAX_ZORDER_MAX_OUTPUT_SIZE).contains(&self.zorder_max_output_size) {
            return Err(invalid(
                "max-output-size",
                format!(
                    "must be in [8, {MAX_ZORDER_MAX_OUTPUT_SIZE}] (per-row key cap), got {}",
                    self.zorder_max_output_size
                ),
            ));
        }
        if !(1..=64).contains(&self.zorder_var_length_contribution) {
            return Err(invalid(
                "var-length-contribution",
                format!(
                    "must be in [1, 64], got {}",
                    self.zorder_var_length_contribution
                ),
            ));
        }
        let lower = self.effective_min_file_size_bytes();
        let upper = self.effective_max_file_size_bytes();
        if lower >= self.target_file_size_bytes {
            return Err(invalid(
                "min-file-size-bytes",
                format!(
                    "must be < target-file-size-bytes ({}), got {}",
                    self.target_file_size_bytes, lower
                ),
            ));
        }
        if upper <= self.target_file_size_bytes {
            return Err(invalid(
                "max-file-size-bytes",
                format!(
                    "must be > target-file-size-bytes ({}), got {}",
                    self.target_file_size_bytes, upper
                ),
            ));
        }
        if let Some(cap) = self.max_files_to_rewrite {
            if cap == 0 {
                return Err(invalid(
                    "max-files-to-rewrite",
                    "must be >= 1 when set".into(),
                ));
            }
        }
        if self.partial_progress_enabled {
            if let Some(mfc) = self.partial_progress_max_failed_commits {
                if mfc > self.partial_progress_max_commits {
                    return Err(invalid(
                        "partial-progress.max-failed-commits",
                        format!(
                            "must be <= partial-progress.max-commits ({}), got {}",
                            self.partial_progress_max_commits, mfc
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NullOrder {
    NullsFirst,
    NullsLast,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortColumn {
    pub name: String,
    pub direction: SortDirection,
    pub null_order: NullOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZOrderKey {
    pub columns: Vec<String>,
    pub var_length_contribution: u32,
    pub max_output_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Strategy {
    BinPack,
    Sort { columns: Vec<SortColumn> },
    ZOrder(ZOrderKey),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_validate() {
        RewriteOptions::default().validate().unwrap();
    }

    #[test]
    fn rejects_undersized_target() {
        let mut o = RewriteOptions::default();
        o.target_file_size_bytes = 1024;
        assert!(o.validate().is_err());
    }

    #[test]
    fn rejects_min_input_files_below_2() {
        let mut o = RewriteOptions::default();
        o.min_input_files = 1;
        assert!(o.validate().is_err());
    }

    #[test]
    fn job_order_parse_round_trip() {
        for s in ["bytes-asc", "bytes-desc", "files-asc", "files-desc", "none"] {
            JobOrder::parse(s).unwrap();
        }
        assert!(JobOrder::parse("garbage").is_err());
    }
}
