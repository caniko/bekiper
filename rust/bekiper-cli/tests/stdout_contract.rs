//! Startup diagnostics must not corrupt machine-readable handoffs.
use std::process::Command;

#[test]
fn readiness_json_remains_parseable_when_gpu_environment_is_missing() {
    let root = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bekiper"))
        .args(["check", "readiness", "--format", "json", "--config"])
        .arg(root.path().join("missing.toml"))
        .env_remove("ORT_DYLIB_PATH")
        .env_remove("HSA_OVERRIDE_GFX_VERSION")
        .env_remove("PYTORCH_ROCM_ARCH")
        .env("RUST_LOG", "info")
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "missing config must fail readiness"
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .expect("stdout must contain only the JSON readiness report");
    assert!(
        report["diagnostics"]
            .as_array()
            .is_some_and(|items| !items.is_empty())
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("ORT_DYLIB_PATH is unset"));
    assert!(!root.path().join("missing.toml").exists());
}
