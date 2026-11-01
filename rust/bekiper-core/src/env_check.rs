//! Pure-logic environment variable check used by both the Rust and Python
//! CLI startup diagnostics.
//!
//! Living in `bekiper-core` lets the Python bindings in `bekiper-inspect-py`
//! re-expose the same function so the Python CLI gets the exact same
//! check — no more drifting parallel implementations.

use std::path::Path;

/// Result of a single environment check: a list of "hard" problems
/// (missing `ORT_DYLIB_PATH` etc.) and a list of "soft" ROCm hints.
#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct EnvCheckReport {
    /// Problems that will likely break GPU paths at runtime.
    pub hard_issues: Vec<String>,
    /// ROCm-related variables that would be nice to have but won't
    /// break a CPU-only run.
    pub soft_missing: Vec<&'static str>,
}

impl EnvCheckReport {
    pub fn is_clean(&self) -> bool {
        self.hard_issues.is_empty() && self.soft_missing.is_empty()
    }
}

/// Pure inspection: evaluate the environment against our expected vars
/// using the supplied lookup closure. Tests inject a HashMap-backed
/// closure so they never touch the real process environment (which is
/// both `unsafe` and not thread-safe in modern Rust).
pub fn check_env<F, P>(lookup: F, path_exists: P) -> EnvCheckReport
where
    F: Fn(&str) -> Option<String>,
    P: Fn(&Path) -> bool,
{
    let mut report = EnvCheckReport::default();

    // `ORT_DYLIB_PATH` must point at libonnxruntime.so; the `ort` crate
    // reads it implicitly at session-load time.
    match lookup("ORT_DYLIB_PATH") {
        Some(path) if !path.is_empty() => {
            if !path_exists(Path::new(&path)) {
                report
                    .hard_issues
                    .push(format!("ORT_DYLIB_PATH={path} does not exist on disk"));
            }
        }
        _ => report.hard_issues.push("ORT_DYLIB_PATH is unset".into()),
    }

    // `HSA_OVERRIDE_GFX_VERSION` and `PYTORCH_ROCM_ARCH` are only meaningful
    // for ROCm builds. They are harmless to miss on a CPU-only run, so they
    // are tracked as soft hints rather than hard issues.
    for var in ["HSA_OVERRIDE_GFX_VERSION", "PYTORCH_ROCM_ARCH"] {
        let missing = lookup(var).map(|s| s.is_empty()).unwrap_or(true);
        if missing {
            report.soft_missing.push(var);
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup<'a>(
        map: &'a HashMap<&'static str, &'static str>,
    ) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| map.get(k).map(|v| (*v).to_string())
    }

    fn fake_exists(existing: &'static [&'static str]) -> impl Fn(&Path) -> bool {
        move |p| existing.iter().any(|e| Path::new(e) == p)
    }

    #[test]
    fn clean_env_has_no_issues() {
        let env = HashMap::from([
            ("ORT_DYLIB_PATH", "/nix/store/xxx/lib/libonnxruntime.so"),
            ("HSA_OVERRIDE_GFX_VERSION", "11.0.0"),
            ("PYTORCH_ROCM_ARCH", "gfx1100"),
        ]);
        let report = check_env(
            lookup(&env),
            fake_exists(&["/nix/store/xxx/lib/libonnxruntime.so"]),
        );
        assert!(report.is_clean(), "expected clean report, got {report:?}");
    }

    #[test]
    fn missing_ort_dylib_is_hard_issue() {
        let env: HashMap<&'static str, &'static str> = HashMap::new();
        let report = check_env(lookup(&env), fake_exists(&[]));
        assert!(report.hard_issues.iter().any(|s| s.contains("unset")));
    }

    #[test]
    fn empty_ort_dylib_is_hard_issue() {
        let env = HashMap::from([("ORT_DYLIB_PATH", "")]);
        let report = check_env(lookup(&env), fake_exists(&[]));
        assert!(report.hard_issues.iter().any(|s| s.contains("unset")));
    }

    #[test]
    fn nonexistent_ort_dylib_is_hard_issue() {
        let env = HashMap::from([("ORT_DYLIB_PATH", "/does/not/exist.so")]);
        let report = check_env(lookup(&env), fake_exists(&[]));
        assert!(
            report
                .hard_issues
                .iter()
                .any(|s| s.contains("does not exist"))
        );
    }

    #[test]
    fn existing_ort_dylib_yields_no_hard_issue() {
        let env = HashMap::from([("ORT_DYLIB_PATH", "/real/libonnxruntime.so")]);
        let report = check_env(lookup(&env), fake_exists(&["/real/libonnxruntime.so"]));
        assert!(report.hard_issues.is_empty());
    }

    #[test]
    fn missing_rocm_vars_are_soft_hints() {
        let env = HashMap::from([("ORT_DYLIB_PATH", "/ok.so")]);
        let report = check_env(lookup(&env), fake_exists(&["/ok.so"]));
        assert!(report.hard_issues.is_empty());
        assert!(report.soft_missing.contains(&"HSA_OVERRIDE_GFX_VERSION"));
        assert!(report.soft_missing.contains(&"PYTORCH_ROCM_ARCH"));
    }

    #[test]
    fn partially_set_rocm_vars() {
        let env = HashMap::from([
            ("ORT_DYLIB_PATH", "/ok.so"),
            ("HSA_OVERRIDE_GFX_VERSION", "11.0.0"),
        ]);
        let report = check_env(lookup(&env), fake_exists(&["/ok.so"]));
        assert!(report.hard_issues.is_empty());
        assert_eq!(report.soft_missing, vec!["PYTORCH_ROCM_ARCH"]);
    }

    #[test]
    fn empty_string_rocm_var_counts_as_missing() {
        let env = HashMap::from([
            ("ORT_DYLIB_PATH", "/ok.so"),
            ("HSA_OVERRIDE_GFX_VERSION", ""),
            ("PYTORCH_ROCM_ARCH", "gfx1100"),
        ]);
        let report = check_env(lookup(&env), fake_exists(&["/ok.so"]));
        assert_eq!(report.soft_missing, vec!["HSA_OVERRIDE_GFX_VERSION"]);
    }

    #[test]
    fn is_clean_true_only_when_both_empty() {
        let mut r = EnvCheckReport::default();
        assert!(r.is_clean());
        r.hard_issues.push("x".into());
        assert!(!r.is_clean());
        r.hard_issues.clear();
        r.soft_missing.push("y");
        assert!(!r.is_clean());
    }
}
