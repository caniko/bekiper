use bekiper_post_analysis::{
    pipeline::publish_descriptive_with_cancel, plotting::build_descriptive_bundle,
};
use serde_json::{Value, json};

fn rows() -> Vec<Value> {
    let mut rows = Vec::new();
    for timepoint in ["pre", "post"] {
        for variant in ["compare_0pct_dwell0s", "compare_0pct_dwell1s", "proximal25"] {
            for (animal, recording, value, excluded) in [
                ("a", "one", 7., false),
                ("a", "two", 9., false),
                ("b", "three", 500., true),
            ] {
                rows.push(json!({"animal_id":animal,"assay":"ymaze","group":"recorded-2",
                    "session_id":recording,"recording_id":recording,"analysis_id":format!("{timepoint}_{recording}__{variant}"),
                    "variant":variant,"analysis_role":format!("ymaze__{variant}"),"timepoint":timepoint,"phase":"session",
                    "threshold_pct":if variant == "proximal25" {25} else {0},
                    "excluded":excluded,"value":value,"scalar_endpoints":["value"],
                    "scoring_provenance":{"minimum_dwell_seconds":if variant.ends_with("1s") {1.} else {0.}},
                    "presentation_partition":{"timepoint":timepoint,"phase":"session","variant":variant},
                    "presentation_label":format!("Y-maze {timepoint} {variant}")}));
            }
        }
    }
    rows
}

#[test]
fn recorded_descriptive_retains_all_partitions_replicates_exclusions_and_values() {
    let bundle = build_descriptive_bundle(
        &rows(),
        &json!({"outputs":{"figure_selection":["figure_ymaze_distributions"]}}),
    )
    .unwrap();
    let figures = bundle["figures"].as_array().unwrap();
    assert_eq!(figures.len(), 6);
    for page in figures {
        for panel in page["panels"].as_array().unwrap() {
            assert_eq!(panel["points"].as_array().unwrap().len(), 2);
            assert_eq!(panel["summaries"][0]["mean"], 8.);
            assert_eq!(panel["summaries"][0]["n_animals"], 1);
            assert_eq!(panel["summaries"][0]["n_observations"], 2);
            assert!(panel["summaries"][0]["sem"].is_null());
            assert!(panel["summaries"][0]["error_half_width"].is_null());
            assert_eq!(panel["status"], "rendered");
            assert_eq!(panel["eligibility_ledger"].as_array().unwrap().len(), 3);
            assert!(
                panel["points"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|p| p["analysis_id"].is_string())
            );
        }
    }
}

#[test]
fn descriptive_publication_exports_recorded_metrics_without_scoring_or_registered_models() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("published");
    let manifest = publish_descriptive_with_cancel(
        rows(),
        vec![],
        vec![],
        json!({"outputs":{"figures_enabled":false}}),
        &output,
        &|| false,
    )
    .unwrap();
    assert_eq!(manifest["analysis_mode"], "recorded_descriptive");
    assert_eq!(manifest["scientific_results_recomputed"], false);
    assert!(output.join("metrics.toml").is_file());
    assert!(!output.join("statistics/models.json").exists());
    let saved: Value = serde_json::from_slice(
        &std::fs::read(bekiper_post_analysis::export::input_path(
            &output,
            "metrics.json",
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(saved, json!(rows()));
    bekiper_post_analysis::pipeline::verify_artifacts(&output).unwrap();
}

#[test]
fn descriptive_primary_contexts_preserve_both_historical_comparisons_and_recorded_values() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("contexts");
    let mut source = rows();
    for threshold in [45, 65] {
        for row in rows().into_iter().filter(|r| r["threshold_pct"] == 25) {
            let mut row = row;
            let variant = format!("proximal{threshold}");
            row["variant"] = json!(variant);
            row["analysis_role"] = json!(format!("y_maze__{variant}"));
            row["analysis_id"] = json!(format!(
                "{}__{}__{variant}",
                row["timepoint"], row["recording_id"]
            ));
            row["threshold_pct"] = json!(threshold);
            source.push(row);
        }
    }
    let manifest = publish_descriptive_with_cancel(
        source,
        vec![],
        vec![],
        json!({"outputs":{"figures_enabled":false},
        "execution":{"ymaze_primary_thresholds":[25,45,65],"primary_workers":2}}),
        &output,
        &|| false,
    )
    .unwrap();
    assert_eq!(manifest["primary_contexts"].as_array().unwrap().len(), 3);
    for threshold in [25, 45, 65] {
        let child = output.join(format!("primary{threshold}"));
        let metrics: Value = serde_json::from_slice(
            &std::fs::read(bekiper_post_analysis::export::input_path(
                &child,
                "metrics.json",
            ))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(metrics.as_array().unwrap().len(), 18);
        assert!(
            metrics
                .as_array()
                .unwrap()
                .iter()
                .all(|r| r["value"] == 7. || r["value"] == 9. || r["value"] == 500.)
        );
        for variant in ["compare_0pct_dwell0s", "compare_0pct_dwell1s"] {
            assert_eq!(
                metrics
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|r| r["variant"] == variant)
                    .count(),
                6
            );
        }
        bekiper_post_analysis::pipeline::verify_artifacts(&child).unwrap();
    }
    bekiper_post_analysis::pipeline::verify_artifacts(&output).unwrap();
}

#[test]
fn descriptive_requests_and_missing_measurements_are_accounted() {
    let mut source = rows();
    source[0]["value"] = Value::Null;
    let bundle =
        build_descriptive_bundle(&source, &json!({"outputs":{"figures_enabled":false}})).unwrap();
    assert!(
        bundle["eligibility_ledger"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["reason"] == "not_measured:value")
    );
    assert!(
        build_descriptive_bundle(
            &source,
            &json!({"outputs":{"figure_selection":["qc_ymaze"]}})
        )
        .unwrap_err()
        .to_string()
        .contains("registered plan")
    );
    let dir = tempfile::tempdir().unwrap();
    let manifest = publish_descriptive_with_cancel(
        source,
        vec![json!({"reason":"recorded_animal_identity_unavailable"})],
        vec![],
        json!({"outputs":{"figures_enabled":false}}),
        &dir.path().join("published"),
        &|| false,
    )
    .unwrap();
    assert_eq!(manifest["status"], "partial");
    assert_eq!(
        manifest["output_statuses"]["figures"]["status"],
        "not_requested"
    );
    assert_eq!(manifest["output_statuses"]["metrics"]["status"], "partial");
}

#[test]
fn descriptive_views_bind_full_recording_provenance_without_endpoint_amplification() {
    let mut row = rows().remove(0);
    let endpoints = (0..20).map(|i| format!("endpoint_{i}")).collect::<Vec<_>>();
    for endpoint in &endpoints {
        row[endpoint] = json!(7.0);
    }
    row["scalar_endpoints"] = json!(endpoints);
    // Native Y-maze evidence includes all dwell bouts and geometry. Its size
    // must not be multiplied by endpoint count, distribution views or exports.
    row["scoring_provenance"]["dwell_bouts"] = json!(
        (0..2048)
            .map(|frame| json!({"start_frame":frame,"duration_s":0.5}))
            .collect::<Vec<_>>()
    );
    let evidence = row["scoring_provenance"].clone();
    let identity = row["analysis_id"].as_str().unwrap().to_owned();
    let bundle = build_descriptive_bundle(&[row.clone()], &json!({})).unwrap();
    assert_eq!(
        bundle["scoring_provenance_by_analysis"][&identity],
        evidence
    );
    for figure in bundle["figures"].as_array().unwrap() {
        for panel in figure["panels"].as_array().unwrap() {
            for field in ["points", "eligibility_ledger"] {
                for record in panel[field].as_array().unwrap() {
                    assert_eq!(record["scoring_provenance_ref"], identity);
                    assert!(record.get("scoring_provenance").is_none());
                }
            }
            assert_eq!(panel["points"][0]["value"], 7.0);
        }
    }
    for field in ["plotting_data", "eligibility_ledger"] {
        for record in bundle[field].as_array().unwrap() {
            assert_eq!(record["scoring_provenance_ref"], identity);
            assert!(record.get("scoring_provenance").is_none());
        }
    }
    assert!(
        serde_json::to_vec(&bundle).unwrap().len()
            < 6 * serde_json::to_vec(&evidence).unwrap().len()
    );
    let output = tempfile::tempdir().unwrap();
    let manifest = publish_descriptive_with_cancel(
        vec![row.clone()],
        vec![],
        vec![],
        json!({"outputs":{"figures_enabled":false}}),
        &output.path().join("published"),
        &|| false,
    )
    .unwrap();
    assert_eq!(
        manifest["scoring_provenance_by_analysis"][&identity],
        evidence
    );
    let saved: Value = serde_json::from_slice(
        &std::fs::read(bekiper_post_analysis::export::input_path(
            &output.path().join("published"),
            "metrics.json",
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(saved, json!([row]));
}

#[test]
fn censored_descriptive_populations_count_animals_and_recordings_separately() {
    let mut source = rows();
    source.truncate(2);
    for row in &mut source {
        row["scalar_endpoints"] = json!(["latency_to_first_center_entry_s"]);
        row["latency_to_first_center_entry_s"] = json!(7.0);
    }
    source[0]["latency_right_censored"] = json!(true);
    let bundle = build_descriptive_bundle(&source, &json!({})).unwrap();
    let summary = &bundle["figures"][0]["panels"][0]["summaries"][0];
    assert_eq!(summary["n_animals"], 1);
    assert_eq!(summary["n_observations"], 2);
    assert_eq!(summary["n_right_censored"], 1);
    assert!(summary["mean"].is_null() && summary["error_half_width"].is_null());
}
