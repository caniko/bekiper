//! Startup diagnostic for ROCm / ONNX Runtime environment variables.
//!
//! The pure-logic `check_env` lives in `bekiper-core::env_check` so the
//! Python bindings can re-expose the same function. This module only
//! glues that result onto `tracing` and reads from the real process
//! environment.

use bekiper_core::env_check::check_env;

/// Check the runtime environment and emit a single warning line if any
/// expected variable is missing or points to a non-existent path.
/// Diagnostic only — never aborts the run.
pub fn warn_if_env_incomplete() {
    let report = check_env(|k| std::env::var(k).ok(), |p| p.exists());

    if report.is_clean() {
        return;
    }

    if !report.hard_issues.is_empty() {
        tracing::warn!(
            "bekiper runtime environment is incomplete: {}. \
             This usually means you are running outside the nix devshell. \
             Run `nix develop` in the project directory to set up ROCm \
             and ONNX Runtime, then re-invoke `bekiper`.",
            report.hard_issues.join("; ")
        );
    }

    if !report.soft_missing.is_empty() {
        tracing::info!(
            "ROCm env hint: {} not set. \
             If you plan to run GPU pose inference, source the nix devshell.",
            report.soft_missing.join(", ")
        );
    }
}
