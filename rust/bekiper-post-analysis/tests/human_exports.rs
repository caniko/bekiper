//! Human exports must preserve values, missingness and upstream evidence while
//! making the published package navigable without reading machine payloads.
use std::{collections::BTreeSet, fs, path::Path};

use bekiper_post_analysis::{Record, RunOptions, pipeline};
use serde_json::{Value, json};

fn run(output: &Path) -> Record {
    pipeline::run_records(
        json!({"run_mode":"synthetic","synthetic_fixture":true,"schema_version":"source-plan-v1",
            "inputs":{"units_and_calibration":{
                "oft":{"time":"s","distance":"m","speed":"m/s"},
                "nort":{"time":"s","distance":"m","speed":"m/s"}}},
            "assays":{"oft":{},"ymaze":{},"nort":{"zero_training_total_policy":null}},
            "statistics":{"summary_uncertainty":"SEM"},
            "outputs":{"requested_outputs":["metrics"],"figure_dpi":180}}),
        vec![
            json!({"animal_id":"mouse 2","assay":"oft","session_id":"session 2",
                "timepoint":"post","phase":"session","excluded":false,
                "measurement_available":true,"total_distance_m":0.0,"center_time_s":0.0,
                "mean_speed_m_per_s":0.0,"center_entries":0,"latency_to_first_center_entry_s":null,
                "exclusion_reason":null,"qc_flags":{"tracking_incomplete":false},
                "fps":10.0,"measurement_qc_flags":{"tracking_incomplete":false},
                "recomputation_missing_reasons":{"center_time_s":null},
                "native_source":{"task_type":"open_field"},
                "center_distance_m":92.08333333333333,
                "upstream_payload":[null,0,false,{"quoted/key":null}],
                "unusually_large_integer":18446744073709551615_u64}),
            json!({"animal_id":"mouse 1","assay":"nort","session_id":"test 1",
                "timepoint":"post","phase":"test","excluded":true,
                "exclusion_reason":"registered exclusion","measurement_available":true,
                "TN":0,"TF":0,"analysis_duration_s":24,"analysis_window_s":360,
                "TA":null,"TB":null,
                "synthetic_observation":true}),
        ],
        vec![],
        &output.parent().unwrap().join("fixture.json"),
        &RunOptions {
            output: Some(output.into()),
            ..Default::default()
        },
    )
    .unwrap()
}

fn read_toml(path: &Path) -> Value {
    toml::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn human_metrics_round_trip_every_canonical_field_without_conflating_zero_and_missing() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("run");
    run(&output);
    let document = read_toml(&output.join("metrics.toml"));
    let canonical: Vec<Value> =
        serde_json::from_slice(&fs::read(output.join("_internal/metrics.json")).unwrap()).unwrap();
    let mut seen = BTreeSet::new();
    for assay in ["oft", "ymaze", "nort"] {
        for session in document[assay]["sessions"].as_array().unwrap() {
            let index = session["source_record_index"].as_u64().unwrap() as usize;
            assert!(seen.insert(index));
            let mut row = session.clone();
            for path in session["null_fields"].as_array().unwrap() {
                *row.pointer_mut(path.as_str().unwrap()).unwrap() = Value::Null;
            }
            for path in session["large_integer_fields"].as_array().unwrap() {
                let value = row.pointer_mut(path.as_str().unwrap()).unwrap();
                *value = serde_json::from_str(value.as_str().unwrap()).unwrap();
            }
            let mut reconstructed = json!({});
            for (key, value) in row.as_object().unwrap() {
                if ["source_record_index", "null_fields", "large_integer_fields"]
                    .contains(&key.as_str())
                {
                    continue;
                }
                if bekiper_post_analysis::contracts::KEY.contains(&key.as_str()) {
                    reconstructed[key] = value.clone();
                } else {
                    for (field, value) in value.as_object().unwrap() {
                        assert!(
                            reconstructed.get(field).is_none(),
                            "duplicate field {field}"
                        );
                        reconstructed[field] = value.clone();
                    }
                }
            }
            assert_eq!(reconstructed, canonical[index]);
        }
    }
    assert_eq!(seen.len(), canonical.len());
    let oft = &document["oft"]["sessions"][0];
    assert_eq!(oft["primary"]["total_distance_m"], 0.0);
    assert_eq!(oft["availability"]["excluded"], false);
    assert_eq!(oft["parameters"]["fps"], 10.0);
    assert_eq!(
        oft["qc"]["measurement_qc_flags"]["tracking_incomplete"],
        false
    );
    assert_eq!(
        oft["missingness"]["recomputation_missing_reasons"]["center_time_s"]["state"],
        "unavailable"
    );
    assert_eq!(
        oft["provenance"]["native_source"]["task_type"],
        "open_field"
    );
    assert_eq!(
        oft["primary"]["latency_to_first_center_entry_s"]["state"],
        "unavailable"
    );
    assert_eq!(
        document["nort"]["sessions"][0]["availability"]["excluded"],
        true
    );
    assert_eq!(
        document["nort"]["sessions"][0]["measurements"]["TA"]["state"],
        "unavailable"
    );
}

#[test]
fn parameters_and_package_index_retain_effective_values_and_real_links() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("run");
    let manifest = run(&output);
    let parameters = read_toml(&output.join("parameters.toml"));
    assert_eq!(parameters["outputs"]["figure_dpi"], 180);
    assert_eq!(parameters["statistics"]["summary_uncertainty"], "SEM");
    assert!(
        parameters["representation"]["null_fields"]
            .as_array()
            .unwrap()
            .contains(&json!("/assays/nort/zero_training_total_policy"))
    );
    let parameter_text = fs::read_to_string(output.join("parameters.toml")).unwrap();
    assert!(
        parameter_text.find("[statistics]").unwrap()
            < parameter_text.find("[[upstream_sessions]]").unwrap()
    );
    assert!(
        parameter_text.find("[statistics]").unwrap()
            < parameter_text.find("[representation]").unwrap()
    );
    assert_eq!(
        parameters["assays"]["nort"]["zero_training_total_policy"]["state"],
        "not_specified"
    );
    let mut recovered = parameters.clone();
    for path in parameters["representation"]["null_fields"]
        .as_array()
        .unwrap()
    {
        let path = path.as_str().unwrap();
        if path.starts_with("/code/") {
            assert_eq!(parameters.pointer(path).unwrap()["state"], "not_specified");
            *recovered.pointer_mut(path).unwrap() = Value::Null;
        }
    }
    // Source archives have no Git metadata. Recover explicitly encoded nulls
    // before comparing every code-identity field to the canonical manifest.
    assert_eq!(recovered["code"], manifest["code"]);
    assert_eq!(parameters["upstream_sessions"][0]["fps"], 10.0);
    assert_eq!(
        parameters["reserved_configuration_fields"]["schema_version"],
        "source-plan-v1"
    );
    assert_eq!(manifest["package_layout"]["schema_version"], "1.0");
    for name in [
        "README.md",
        "metrics.toml",
        "parameters.toml",
        "statistics/models.json",
        "audit/inclusion_ledger.json",
        "provenance/BEHAVIORAL_RECIPE.md",
        "_internal/plotting_bundle.json",
    ] {
        assert!(output.join(name).is_file(), "missing {name}");
    }
    assert!(!output.join("metrics.json").exists());
    let index = fs::read_to_string(output.join("README.md")).unwrap();
    assert!(
        index.contains("synthetic")
            && index.contains("metrics.toml")
            && index.contains("parameters.toml")
    );
    for (offset, _) in index.match_indices("](") {
        let target = index[offset + 2..].split(')').next().unwrap();
        assert!(output.join(target).is_file(), "broken index link {target}");
    }
    pipeline::verify_artifacts(&output).unwrap();
    fs::write(output.join("parameters.toml"), "tampered = true").unwrap();
    assert!(pipeline::verify_artifacts(&output).is_err());
}

#[test]
fn one_time_export_preserves_scientific_identity_and_refuses_replacement() {
    let directory = tempfile::tempdir().unwrap();
    let original = directory.path().join("original");
    let original_manifest = run(&original);
    fs::write(
        directory.path().join("unrelated-video.mp4"),
        "not part of post-analysis",
    )
    .unwrap();
    let output = directory.path().join("human-export");
    let manifest = bekiper_post_analysis::export::export_existing(&original, &output).unwrap();
    assert_eq!(manifest["code"], original_manifest["code"]);
    assert_eq!(
        manifest["resolved_config"],
        original_manifest["resolved_config"]
    );
    assert_eq!(
        manifest["presentation_export"]["scientific_results_recomputed"],
        false
    );
    assert!(manifest["presentation_export"]["generator"]["source_sha256"].is_string());
    assert!(output.join("parameters.toml").is_file());
    let human = read_toml(&output.join("metrics.toml"));
    assert_eq!(
        human["oft"]["sessions"][0]["measurements"]["center_distance_m"].as_f64(),
        Some(92.08333333333333)
    );
    assert_eq!(
        manifest["package_layout"]["paths"]["metrics.json"],
        "_internal/metrics.json"
    );
    assert_eq!(
        manifest["package_layout"]["paths"]["models.json"],
        "statistics/models.json"
    );
    assert!(!output.join("unrelated-video.mp4").exists());
    pipeline::verify_artifacts(&output).unwrap();
    let before = fs::read(output.join("artifact_manifest.json")).unwrap();
    assert!(bekiper_post_analysis::export::export_existing(&original, &output).is_err());
    assert_eq!(
        before,
        fs::read(output.join("artifact_manifest.json")).unwrap()
    );
    assert!(
        bekiper_post_analysis::export::export_existing(&original, &original.join("nested"))
            .is_err()
    );
    pipeline::verify_artifacts(&original).unwrap();
}

#[test]
fn presentation_export_respects_project_archive_lease() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("project");
    let source = root.join("data/_post_analysis/source");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    run(&source);
    let output = directory.path().join("presentation");
    let archive = pipeline::ProjectDataLease::archive(&root).unwrap();
    assert!(bekiper_post_analysis::export::export_existing(&source, &output).is_err());
    assert!(!output.exists());
    drop(archive);
    bekiper_post_analysis::export::export_existing(&source, &output).unwrap();
    pipeline::verify_artifacts(&output).unwrap();
    pipeline::verify_artifacts(&source).unwrap();
}

#[test]
fn overlapping_field_names_keep_qc_missingness_and_comparison_precedence() {
    let directory = tempfile::tempdir().unwrap();
    let row = json!({"animal_id":"fixture","assay":"oft","session_id":"session",
        "timepoint":"post","phase":"session",
        "analysis_threshold_qc_flags":{"flag":false},
        "upstream_missing_reasons":{"value":null},
        "upstream_rule_provenance":{"instruction":"historical"},
        "analysis_window_provenance":{"origin":"registered"},
        "passes_15s_exploration_threshold":false,
        "total_distance_m":1.0,
        "unknown_value":{"quoted/key":[null,false,0]}});
    let mut manifest = json!({"run_mode":"synthetic","status":"partial",
        "real_data_execution":false,"resolved_config":{},"figures":[]});
    bekiper_post_analysis::export::finalize(directory.path(), &mut manifest, &[row]).unwrap();
    let document = read_toml(&directory.path().join("metrics.toml"));
    let session = &document["oft"]["sessions"][0];
    assert_eq!(session["qc"]["analysis_threshold_qc_flags"]["flag"], false);
    assert_eq!(session["qc"]["passes_15s_exploration_threshold"], false);
    assert_eq!(
        session["missingness"]["upstream_missing_reasons"]["value"]["state"],
        "unavailable"
    );
    assert_eq!(
        session["comparisons"]["upstream_rule_provenance"]["instruction"],
        "historical"
    );
    assert_eq!(
        session["provenance"]["analysis_window_provenance"]["origin"],
        "registered"
    );
    assert_eq!(session["primary"]["total_distance_m"], 1.0);
    assert!(
        session["null_fields"]
            .as_array()
            .unwrap()
            .contains(&json!("/additional_fields/unknown_value/quoted~1key/0"))
    );
}
