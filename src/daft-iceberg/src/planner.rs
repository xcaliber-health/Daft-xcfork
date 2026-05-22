use serde::{Deserialize, Serialize};

use crate::{errors::IcebergRewriteError, options::{JobOrder, RewriteOptions}};

/// One Iceberg data file considered for rewrite.
///
/// `partition_key` is a canonical JSON encoding of the partition record; rows with
/// equal `partition_key` and `partition_spec_id` are grouped together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateFile {
    pub path: String,
    pub size_bytes: u64,
    pub partition_key: String,
    pub partition_spec_id: i32,
    pub positional_delete_paths: Vec<String>,
    pub has_equality_deletes: bool,
}

impl CandidateFile {
    fn positional_delete_count(&self) -> u32 {
        self.positional_delete_paths.len() as u32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileGroup {
    pub partition_key: String,
    pub output_spec_id: i32,
    pub files: Vec<CandidateFile>,
    pub total_bytes: u64,
}

impl FileGroup {
    fn empty(partition_key: String, output_spec_id: i32) -> Self {
        Self {
            partition_key,
            output_spec_id,
            files: Vec::new(),
            total_bytes: 0,
        }
    }

    fn push(&mut self, f: CandidateFile) {
        self.total_bytes += f.size_bytes;
        self.files.push(f);
    }
}

/// Group candidate files for rewrite.
///
/// Steps: bucket by `(partition_key, partition_spec_id)`, drop files that don't need
/// rewrite per Iceberg's heuristics, skip buckets below `min_input_files`, bin-pack into
/// groups capped by `max_file_group_size_bytes`, then sort across buckets by `job_order`.
pub fn plan_file_groups(
    candidates: Vec<CandidateFile>,
    opts: &RewriteOptions,
    current_spec_id: i32,
) -> Result<Vec<FileGroup>, IcebergRewriteError> {
    opts.validate()?;
    let output_spec_id = opts.output_spec_id.unwrap_or(current_spec_id);

    let mut buckets: std::collections::BTreeMap<(String, i32), Vec<CandidateFile>> =
        std::collections::BTreeMap::new();
    for c in candidates {
        if c.has_equality_deletes {
            return Err(IcebergRewriteError::EqualityDeletesPresent {
                n: 1,
                sample: vec![c.path],
            });
        }
        buckets
            .entry((c.partition_key.clone(), c.partition_spec_id))
            .or_default()
            .push(c);
    }

    let target = opts.target_file_size_bytes as f64;
    let lower = (target * 0.75) as u64;
    let upper = (target * 1.80) as u64;

    let mut groups: Vec<FileGroup> = Vec::new();
    for ((part_key, spec_id), files) in buckets {
        let needs_spec_change = spec_id != output_spec_id;
        let survivors: Vec<CandidateFile> = files
            .into_iter()
            .filter(|f| {
                opts.rewrite_all
                    || needs_spec_change
                    || f.size_bytes < lower
                    || f.size_bytes > upper
                    || f.positional_delete_count() >= opts.delete_file_threshold
            })
            .collect();

        if !opts.rewrite_all
            && !needs_spec_change
            && (survivors.len() as u32) < opts.min_input_files
        {
            continue;
        }

        groups.extend(pack(survivors, &part_key, output_spec_id, opts.max_file_group_size_bytes));
    }

    sort_groups(&mut groups, opts.job_order);
    Ok(groups)
}

fn pack(
    mut files: Vec<CandidateFile>,
    partition_key: &str,
    output_spec_id: i32,
    cap: u64,
) -> Vec<FileGroup> {
    // First-fit-decreasing: biggest files first so oversized singletons land in their own group
    // and the remainder packs efficiently.
    files.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    let mut out: Vec<FileGroup> = Vec::new();
    let mut current = FileGroup::empty(partition_key.to_string(), output_spec_id);
    for f in files {
        if current.total_bytes + f.size_bytes > cap && !current.files.is_empty() {
            out.push(std::mem::replace(
                &mut current,
                FileGroup::empty(partition_key.to_string(), output_spec_id),
            ));
        }
        current.push(f);
    }
    if !current.files.is_empty() {
        out.push(current);
    }
    out
}

fn sort_groups(groups: &mut [FileGroup], order: JobOrder) {
    match order {
        JobOrder::BytesAsc => groups.sort_by(|a, b| a.total_bytes.cmp(&b.total_bytes)),
        JobOrder::BytesDesc => groups.sort_by(|a, b| b.total_bytes.cmp(&a.total_bytes)),
        JobOrder::FilesAsc => groups.sort_by_key(|g| g.files.len()),
        JobOrder::FilesDesc => groups.sort_by_key(|g| std::cmp::Reverse(g.files.len())),
        JobOrder::None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cf(path: &str, size: u64, part: &str, spec: i32) -> CandidateFile {
        CandidateFile {
            path: path.into(),
            size_bytes: size,
            partition_key: part.into(),
            partition_spec_id: spec,
            positional_delete_paths: vec![],
            has_equality_deletes: false,
        }
    }

    fn opts(target: u64, min_input: u32, cap: u64) -> RewriteOptions {
        RewriteOptions {
            target_file_size_bytes: target,
            min_input_files: min_input,
            max_file_group_size_bytes: cap,
            ..RewriteOptions::default()
        }
    }

    #[test]
    fn empty_input_yields_no_groups() {
        let groups = plan_file_groups(vec![], &RewriteOptions::default(), 0).unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn below_min_input_files_skipped() {
        let target = 64 * 1024 * 1024;
        let o = opts(target, 5, 5 * target);
        let candidates = (0..3)
            .map(|i| cf(&format!("/f{i}.parquet"), 1024, "{}", 0))
            .collect();
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert!(groups.is_empty(), "expected skip when below min-input-files");
    }

    #[test]
    fn rewrite_all_includes_below_min() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            rewrite_all: true,
            ..opts(target, 5, 5 * target)
        };
        let candidates = (0..3)
            .map(|i| cf(&format!("/f{i}.parquet"), 1024, "{}", 0))
            .collect();
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].files.len(), 3);
    }

    #[test]
    fn already_sized_files_excluded() {
        let target = 64 * 1024 * 1024;
        let o = opts(target, 2, 5 * target);
        // Five files near target size — survivor filter drops them all.
        let candidates = (0..5)
            .map(|i| cf(&format!("/f{i}.parquet"), target, "{}", 0))
            .collect();
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert!(groups.is_empty());
    }

    #[test]
    fn oversize_file_is_singleton_group() {
        let target = 64 * 1024 * 1024;
        let cap = 5 * target;
        let o = opts(target, 2, cap);
        let huge = 4 * target;
        let candidates = vec![
            cf("/big.parquet", huge, "{}", 0),
            cf("/tiny0.parquet", 1024, "{}", 0),
            cf("/tiny1.parquet", 1024, "{}", 0),
        ];
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert!(!groups.is_empty());
        // The biggest file must land alone or with a small tail, never exceeding cap.
        for g in &groups {
            assert!(g.total_bytes <= cap);
        }
    }

    #[test]
    fn output_spec_change_disables_min_input_gate() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            output_spec_id: Some(1),
            ..opts(target, 5, 5 * target)
        };
        let candidates = (0..2)
            .map(|i| cf(&format!("/f{i}.parquet"), 1024, "{}", 0))
            .collect();
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].output_spec_id, 1);
    }

    #[test]
    fn equality_delete_short_circuits() {
        let target = 64 * 1024 * 1024;
        let o = opts(target, 2, 5 * target);
        let mut c = cf("/f0.parquet", 1024, "{}", 0);
        c.has_equality_deletes = true;
        let err = plan_file_groups(vec![c], &o, 0).unwrap_err();
        assert!(matches!(err, IcebergRewriteError::EqualityDeletesPresent { .. }));
    }

    #[test]
    fn buckets_per_partition() {
        let target = 64 * 1024 * 1024;
        let o = opts(target, 2, 5 * target);
        let candidates = vec![
            cf("/a0.parquet", 1024, "{\"d\":\"2024-01-01\"}", 0),
            cf("/a1.parquet", 1024, "{\"d\":\"2024-01-01\"}", 0),
            cf("/b0.parquet", 1024, "{\"d\":\"2024-01-02\"}", 0),
            cf("/b1.parquet", 1024, "{\"d\":\"2024-01-02\"}", 0),
        ];
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn bytes_desc_ordering() {
        let target = 64 * 1024 * 1024;
        let o = RewriteOptions {
            job_order: JobOrder::BytesDesc,
            ..opts(target, 2, 5 * target)
        };
        // Two partitions: one large, one small.
        let candidates = vec![
            cf("/big0.parquet", 10_000_000, "p=a", 0),
            cf("/big1.parquet", 10_000_000, "p=a", 0),
            cf("/sm0.parquet", 1024, "p=b", 0),
            cf("/sm1.parquet", 1024, "p=b", 0),
        ];
        let groups = plan_file_groups(candidates, &o, 0).unwrap();
        assert_eq!(groups.len(), 2);
        assert!(groups[0].total_bytes >= groups[1].total_bytes);
    }
}
