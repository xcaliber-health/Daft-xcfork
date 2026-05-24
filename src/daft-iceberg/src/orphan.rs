//! Set-difference primitive for remove_orphan_files.
//!
//! The orchestration layer canonicalizes file paths in Python (applying
//! prefix_mismatch_mode and scheme normalization) and then asks this module
//! to subtract the reachable set from the listed set. Returning a fresh
//! `Vec<String>` keeps the Python boundary simple; the caller decides
//! whether to delete, dry-run, or sample.

use std::collections::HashSet;

/// Return the paths in `listed` that are not present in `reachable`.
///
/// Order of the result matches the order of `listed`. Both inputs are taken
/// by value so the caller can move owned strings in without an extra clone.
pub fn orphan_diff(listed: Vec<String>, reachable: Vec<String>) -> Vec<String> {
    let reachable_set: HashSet<&str> = reachable.iter().map(String::as_str).collect();
    listed
        .into_iter()
        .filter(|p| !reachable_set.contains(p.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_paths_not_in_reachable() {
        let listed = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let reachable = vec!["b".into(), "d".into()];
        let out = orphan_diff(listed, reachable);
        assert_eq!(out, vec!["a".to_string(), "c".to_string()]);
    }

    #[test]
    fn preserves_listed_order() {
        let listed = vec!["z".into(), "y".into(), "x".into()];
        let reachable: Vec<String> = vec![];
        assert_eq!(
            orphan_diff(listed, reachable),
            vec!["z".to_string(), "y".to_string(), "x".to_string()],
        );
    }

    #[test]
    fn empty_listed_empty_output() {
        let out = orphan_diff(Vec::new(), vec!["a".into()]);
        assert!(out.is_empty());
    }

    #[test]
    fn duplicates_in_listed_all_pass_through_when_not_reachable() {
        let listed = vec!["a".into(), "a".into(), "b".into()];
        let reachable = vec!["b".into()];
        assert_eq!(orphan_diff(listed, reachable), vec!["a".to_string(), "a".to_string()]);
    }
}
