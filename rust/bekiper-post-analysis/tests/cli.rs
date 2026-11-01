//! Installed-style CLI execution must not depend on a repository or Python for
//! metrics-only analysis. The binary contains its source references and identity.
use serde_json::{Value, json};
use std::process::Command;

#[test]
fn native_binary_runs_outside_checkout_and_retains_unresolved_output_status() {
    let directory = tempfile::tempdir().unwrap();
    let metadata = json!([{"animal_id":"a1","assay":"nort","session_id":"test1",
        "timepoint":"post","phase":"test","five_xfad":"AD","hcar1":"WT","treatment":"lactate",
        "group":"G5","sex":"female","age_treatment_window":"registered_window",
        "excluded":false,"exclusion_reason":null}]);
    let measurements = json!([{"animal_id":"a1","assay":"nort","session_id":"test1",
        "timepoint":"post","phase":"test","TN":12,"TF":8,"analysis_duration_s":360,"synthetic_observation":true,
        "object_configuration":"explicit_counterbalance","protocol_metadata":{"familiarization_s":360,"delay_h":5,"test_s":360}}]);
    let config = json!({"run_mode":"synthetic","synthetic_fixture":true,"inputs":{"metadata_path":"metadata.json","nort_path":"nort.json",
        "units_and_calibration":{"nort":{"time":"s","distance":"m","speed":"m/s"}}},
        "assays":{"nort":{},"ymaze":{}},"outputs":{"requested_outputs":["metrics"]}});
    for (name, value) in [
        ("metadata.json", metadata),
        ("nort.json", measurements),
        ("config.json", config),
    ] {
        std::fs::write(
            directory.path().join(name),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
    }
    let result = Command::new(env!("CARGO_BIN_EXE_bekiper-post-analysis"))
        .current_dir(directory.path())
        .env_remove("BEKIPER_POST_ANALYSIS_OUTPUT")
        .env_remove("BEKIPER_POST_ANALYSIS_REQUESTED_OUTPUTS")
        .env_remove("BEKIPER_POST_ANALYSIS_SENSITIVITY_THRESHOLDS")
        .args([
            "--config",
            "config.json",
            "--output",
            "fresh output",
            "--request",
            "metrics",
            "--no-sensitivity",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let manifest: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(manifest["schema_version"], "2.0");
    assert_eq!(manifest["status"], "partial");
    assert_eq!(manifest["real_data_execution"], false);
    assert_eq!(manifest["code"]["identity_origin"], "embedded_at_build");
    assert!(manifest["source_recipe"]["sha256"].as_str().unwrap().len() == 64);
    let metrics: Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("fresh output/_internal/metrics.json")).unwrap(),
    )
    .unwrap();
    assert!((metrics[0]["DI"].as_f64().unwrap() - 0.2).abs() < 1e-12);
    assert_eq!(
        metrics[0]["qc_flags"]["passes_15s_exploration_threshold"],
        Value::Null
    );
    assert!(
        directory
            .path()
            .join("fresh output/artifact_manifest.json")
            .is_file()
    );
    let again = Command::new(env!("CARGO_BIN_EXE_bekiper-post-analysis"))
        .current_dir(directory.path())
        .args(["--config", "config.json", "--output", "fresh output"])
        .output()
        .unwrap();
    assert!(
        !again.status.success(),
        "a published run must never be overwritten"
    );
}

#[test]
fn cli_rejects_invalid_threshold_and_conflicting_disable_flag() {
    let binary = env!("CARGO_BIN_EXE_bekiper-post-analysis");
    for args in [
        vec!["--config", "missing.json", "--sensitivity-threshold", "44"],
        vec![
            "--config",
            "missing.json",
            "--sensitivity-threshold",
            "25",
            "--no-sensitivity",
        ],
    ] {
        assert!(
            !Command::new(binary)
                .args(args)
                .output()
                .unwrap()
                .status
                .success()
        );
    }
}
