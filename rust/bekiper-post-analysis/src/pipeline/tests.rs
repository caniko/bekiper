use super::*;

fn registration(config: &mut Record, decision: &str, path: &str, value: Record) {
    let mut target = &mut *config;
    for part in path.split('.') {
        target = &mut target[part];
    }
    *target = value.clone();
    if !config["decision_resolutions"][decision].is_object() {
        config["decision_resolutions"][decision] = json!({"origin":"development_fixture","rationale":"synthetic regression only",
            "approval_or_plan_reference":"F4 test","scope":["inputs","statistics","inclusion","assays"],"value":{}});
    }
    config["decision_resolutions"][decision]["value"][path] = value;
}

fn config() -> Record {
    json!({"run_mode":"synthetic","synthetic_fixture":true,"inputs":{},"assays":{"ymaze":{},"nort":{},"oft":{}},
        "statistics":{},"inclusion":{},"decision_resolutions":{},"outputs":{"requested_outputs":["models","within_group_tests"]}})
}

#[test]
fn three_primary_contexts_export_isolated_configs_and_one_primary_per_session() {
    let parent = tempfile::tempdir().unwrap();
    let mut configuration = config();
    configuration["outputs"] =
        json!({"requested_outputs":["metrics"],"figures_enabled":false,"figure_selection":[]});
    registration(
        &mut configuration,
        "D01",
        "assays.ymaze.triad_policy",
        json!("source_table"),
    );
    registration(
        &mut configuration,
        "D02",
        "assays.ymaze.fewer_than_three_entries_policy",
        json!("zero_windows"),
    );
    let rows: Vec<_> = [25, 45, 65]
        .into_iter()
        .map(|t| {
            json!({"animal_id":"a","assay":"ymaze","session_id":"s",
        "timepoint":"post","phase":"session","threshold_pct":t,"measurement_available":false,
        "valid_arm_entry_sequence":["A","B","C"],"excluded":false})
        })
        .collect();
    let output = parent.path().join("primaries");
    let manifest = run_records(
        configuration,
        rows,
        vec![],
        &parent.path().join("absent.json"),
        &RunOptions {
            output: Some(output.clone()),
            primary_thresholds: Some(vec![25, 45, 65]),
            primary_workers: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(manifest["primary_contexts"].as_array().unwrap().len(), 3);
    assert_eq!(manifest["primary_workers"], 2);
    for threshold in [25, 45, 65] {
        let child = output.join(format!("primary{threshold}"));
        let metrics: Record = serde_json::from_slice(
            &std::fs::read(crate::export::path(&child, "metrics.json")).unwrap(),
        )
        .unwrap();
        let primary: Vec<_> = metrics
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["primary_threshold"] == true)
            .collect();
        assert_eq!(primary.len(), 1);
        assert_eq!(primary[0]["threshold_pct"], threshold);
        let manifest: Record =
            serde_json::from_slice(&std::fs::read(child.join("run_manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            manifest["resolved_config"]["assays"]["ymaze"]["primary_threshold_arm_length_pct"],
            threshold
        );
        verify_artifacts(&child).unwrap();
    }
    verify_artifacts(&output).unwrap();
}

#[test]
fn primary_context_cancellation_and_invalid_selection_preserve_fresh_destination() {
    let parent = tempfile::tempdir().unwrap();
    for (label, thresholds, workers, cancelled) in [
        ("duplicate", vec![25, 25], 2, false),
        ("unknown", vec![40], 1, false),
        ("no_workers", vec![25, 45, 65], 0, false),
        ("cancelled", vec![25, 45, 65], 2, true),
    ] {
        let output = parent.path().join(label);
        assert!(
            run_records_with_cancel(
                config(),
                vec![],
                vec![],
                &parent.path().join("absent.json"),
                &RunOptions {
                    output: Some(output.clone()),
                    primary_thresholds: Some(thresholds),
                    primary_workers: Some(workers),
                    ..Default::default()
                },
                &|| cancelled
            )
            .is_err()
        );
        assert!(!output.exists());
    }
}

#[test]
fn primary_context_cancellation_drains_active_workers_before_retaining_failure() {
    use std::sync::{
        Barrier, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("cancelled-workers");
    let caller = std::thread::current().id();
    let workers = std::thread::available_parallelism().unwrap().get().min(2);
    let barrier = Barrier::new(workers);
    let calls = Mutex::new(std::collections::HashMap::new());
    let active = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let cancel = AtomicBool::new(false);
    let cancelled = || {
        if cancel.load(Ordering::SeqCst) {
            return true;
        }
        let thread = std::thread::current().id();
        if thread == caller {
            return false;
        }
        let count = {
            let mut calls = calls.lock().unwrap();
            let count = calls.entry(thread).or_insert(0);
            *count += 1;
            *count
        };
        // The second worker check happens inside execute, after its private
        // primary directory exists. Hold each active worker at that boundary.
        if count == 2 {
            active.fetch_add(1, Ordering::SeqCst);
            barrier.wait();
            cancel.store(true, Ordering::SeqCst);
            completed.fetch_add(1, Ordering::SeqCst);
            active.fetch_sub(1, Ordering::SeqCst);
            return true;
        }
        false
    };
    assert!(
        run_records_with_cancel(
            config(),
            vec![],
            vec![],
            &parent.path().join("absent.json"),
            &RunOptions {
                output: Some(output.clone()),
                primary_thresholds: Some(vec![25, 45, 65]),
                primary_workers: Some(2),
                ..Default::default()
            },
            &cancelled
        )
        .is_err()
    );
    assert_eq!(active.load(Ordering::SeqCst), 0);
    assert_eq!(completed.load(Ordering::SeqCst), workers);
    assert!(!output.exists());
    let evidence = std::fs::read_dir(parent.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.is_dir())
        .unwrap();
    let failure: Record =
        serde_json::from_slice(&std::fs::read(evidence.join("failure_manifest.json")).unwrap())
            .unwrap();
    assert_eq!(failure["status"], "cancelled");
    assert!(!evidence.join("run_manifest.json").exists());
    assert_eq!(
        std::fs::read_dir(evidence)
            .unwrap()
            .filter(|e| e
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("primary"))
            .count(),
        // The warm-up context finishes on the caller thread before the pool starts.
        workers + usize::from(workers > 1)
    );
}

#[test]
fn registered_assays_materialize_hypotheses_with_no_observed_members() {
    let mut config = config();
    registration(
        &mut config,
        "D09",
        "statistics.model_family_by_endpoint",
        json!({
        "oft.total_distance_m":"gaussian_ols","ymaze.spontaneous_alternation_pct":"gaussian_ols","nort.DI":"gaussian_ols"}),
    );
    registration(
        &mut config,
        "D09",
        "statistics.multiplicity",
        json!({"method":"holm","families":{"planned_contrasts":"primary","nort_within_group":"DI_groups"}}),
    );
    config["inputs"]["group_mapping"] = json!({"G1":{},"G2":{},"G5":{},"G6":{}});
    config["statistics"]["nort_within_group_checks"] = json!(["DI", "RI"]);
    let results = run_models(&[], &config).unwrap();
    assert_eq!(results.models.len(), 8);
    assert_eq!(results.within_group_tests.len(), 8);
    assert!(
        results
            .models
            .iter()
            .all(|model| model["status"] == "blocked" && model["contrast"]["family_size"] == 8)
    );
    assert!(
        results
            .within_group_tests
            .iter()
            .all(|test| test["status"] == "blocked" && test["family_size"] == 8)
    );
}

#[test]
fn registered_empty_within_group_family_does_not_depend_on_model_requests() {
    let mut config = config();
    config["outputs"]["requested_outputs"] = json!(["within_group_tests"]);
    config["inputs"]["group_mapping"] = json!({"G1":{},"G2":{},"G5":{},"G6":{}});
    registration(
        &mut config,
        "D09",
        "statistics.nort_within_group_checks",
        json!(["DI", "RI"]),
    );
    registration(
        &mut config,
        "D09",
        "statistics.multiplicity",
        json!({"method":"holm","families":{"nort_within_group":"DI_groups"}}),
    );
    let results = run_models(&[], &config).unwrap();
    assert!(results.models.is_empty());
    assert_eq!(results.within_group_tests.len(), 8);
    assert!(
        results
            .within_group_tests
            .iter()
            .all(|test| test["family_size"] == 8)
    );
}

fn factorial_configuration(assay: &str, endpoint: &str) -> (Record, Record) {
    let oracle: Record =
        serde_json::from_str(include_str!("../../tests/fixtures/statistics_oracle.json")).unwrap();
    let mut config = config();
    config["outputs"]["requested_outputs"] = json!(["models"]);
    registration(
        &mut config,
        "D10",
        "inputs.factor_levels",
        oracle["levels"].clone(),
    );
    registration(
        &mut config,
        "D10",
        "inputs.group_mapping",
        oracle["groups"].clone(),
    );
    for (field, value) in oracle["settings"].as_object().unwrap() {
        if !matches!(field.as_str(), "model_family" | "additional_covariates") {
            registration(
                &mut config,
                "D09",
                &format!("statistics.{field}"),
                value.clone(),
            );
        }
    }
    registration(
        &mut config,
        "D09",
        "statistics.nort_additional_covariates",
        json!([]),
    );
    registration(
        &mut config,
        "D09",
        "statistics.model_family_by_endpoint",
        json!({format!("{assay}.{endpoint}"):"gaussian_ols"}),
    );
    registration(
        &mut config,
        "D09",
        "statistics.model_failure_policy",
        json!("report_without_simplification"),
    );
    registration(
        &mut config,
        "D09",
        "statistics.multiplicity",
        json!({"method":"holm","families":{"planned_contrasts":"primary"}}),
    );
    for (field, value) in [
        ("existing_metadata_exclusions_policy", "honor_metadata"),
        ("qc_flag_exclusion_policy", "flags_only"),
        ("missing_metric_policy", "endpoint_complete_cases"),
        ("upstream_verification_requirement", "allow_unverified"),
    ] {
        registration(
            &mut config,
            "D12",
            &format!("inclusion.{field}"),
            json!(value),
        );
    }
    (oracle, config)
}

#[test]
fn orchestration_preserves_the_independent_factorial_oracle_and_contrast_sign() {
    let (oracle, config) = factorial_configuration("nort", "DI");
    let rows: Vec<_> = oracle["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let mut row = row.clone();
            row["assay"] = "nort".into();
            row["phase"] = "test".into();
            row["timepoint"] = "post".into();
            row
        })
        .collect();
    let results = run_models(&rows, &config).unwrap();
    assert_eq!(results.models.len(), 1);
    let model = &results.models[0];
    assert_eq!(model["status"], "fitted", "{model}");
    assert_eq!(model["n_animals"], oracle["post_only"]["n_animals"]);
    assert_eq!(model["contrast"]["family_size"], 1);
    let estimate = model["contrast"]["estimate"].as_f64().unwrap();
    assert!(
        (estimate
            - oracle["post_only"]["contrast"]["estimate"]
                .as_f64()
                .unwrap())
        .abs()
            < 1e-10
    );
    assert!(estimate > 0.0);
    let p = model["contrast"]["p_adjusted"].as_f64().unwrap();
    let expected_p = oracle["post_only"]["contrast"]["p_raw"].as_f64().unwrap();
    assert!((p / expected_p - 1.0).abs() < 1e-9);
    assert_eq!(
        result_status(&results.models, "fitted")["status"],
        "complete"
    );
}

#[test]
fn three_primary_contexts_fit_independent_models_without_pseudoreplication() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("primaries");
    let (oracle, mut configuration) =
        factorial_configuration("ymaze", "spontaneous_alternation_pct");
    configuration["inputs"]["units_and_calibration"] =
        json!({"ymaze":{"time":"s","distance":"m","speed":"m/s"}});
    registration(
        &mut configuration,
        "D01",
        "assays.ymaze.triad_policy",
        json!("source_table"),
    );
    registration(
        &mut configuration,
        "D02",
        "assays.ymaze.fewer_than_three_entries_policy",
        json!("zero_windows"),
    );
    configuration["outputs"]["figures_enabled"] = json!(false);
    configuration["outputs"]["figure_selection"] = json!([]);
    registration(
        &mut configuration,
        "D09",
        "statistics.ymaze_post_only_formula",
        json!("post_SA ~ 5XFAD_status * HCAR1_status * treatment"),
    );
    let mut rows = Vec::new();
    for (threshold, multiplier) in [(25, 10.0), (45, 20.0), (65, 30.0)] {
        for source in oracle["rows"].as_array().unwrap() {
            let mut row = source.clone();
            row.as_object_mut().unwrap().remove("DI");
            row["assay"] = json!("ymaze");
            row["phase"] = json!("session");
            row["timepoint"] = json!("post");
            row["threshold_pct"] = json!(threshold);
            row["spontaneous_alternation_pct"] =
                json!(50.0 + multiplier * source["DI"].as_f64().unwrap());
            rows.push(row);
        }
    }
    let manifest = run_records(
        configuration,
        rows,
        vec![],
        &root.path().join("synthetic.json"),
        &RunOptions {
            output: Some(output.clone()),
            primary_thresholds: Some(vec![25, 45, 65]),
            primary_workers: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        manifest["shared_trial_records"],
        3 * oracle["rows"].as_array().unwrap().len()
    );
    for (threshold, multiplier) in [(25, 10.0), (45, 20.0), (65, 30.0)] {
        let child = output.join(format!("primary{threshold}"));
        let models: Record = serde_json::from_slice(
            &std::fs::read(crate::export::input_path(&child, "models.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(models.as_array().unwrap().len(), 1);
        let model = &models[0];
        assert_eq!(model["status"], "fitted", "{model}");
        assert_eq!(model["n_animals"], oracle["post_only"]["n_animals"]);
        assert_eq!(model["contrast"]["family_size"], 1);
        let estimate = model["contrast"]["estimate"].as_f64().unwrap();
        assert!(
            (estimate
                - multiplier
                    * oracle["post_only"]["contrast"]["estimate"]
                        .as_f64()
                        .unwrap())
            .abs()
                < 1e-9
        );
        assert!(
            (model["contrast"]["p_raw"].as_f64().unwrap()
                - oracle["post_only"]["contrast"]["p_raw"].as_f64().unwrap())
            .abs()
                < 1e-10
        );
        verify_artifacts(&child).unwrap();
    }
    verify_artifacts(&output).unwrap();
}

#[test]
fn empty_registered_subgroups_keep_attempts_in_their_correction_family() {
    let mut config = config();
    config["outputs"]["requested_outputs"] = json!(["subgroup_inference"]);
    config["inputs"]["group_mapping"] = json!({"G1":{},"G2":{}});
    config["statistics"]["nort_within_group_checks"] = json!(["DI"]);
    registration(
        &mut config,
        "D09",
        "statistics.subgroup_analysis_plan",
        json!({
        "strata":[{"field":"sex","levels":["F","M"],"assays":["nort"]}],
        "outputs":["within_group_tests"],"multiplicity":{"method":"holm","family":"registered_empty_strata"}}),
    );
    let plan = config["statistics"]["subgroup_analysis_plan"].clone();
    registration(
        &mut config,
        "D12",
        "statistics.subgroup_analysis_plan",
        plan,
    );
    let (strata, status) = run_subgroups(&[], &config).unwrap();
    assert_eq!(strata.len(), 2);
    assert_ne!(status["status"], "complete");
    for stratum in &strata {
        let tests = stratum["within_group_tests"].as_array().unwrap();
        assert_eq!(tests.len(), 2);
        assert!(
            tests
                .iter()
                .all(|test| test["family_size"] == 4 && test["p_adjusted"].is_null())
        );
    }
}

#[test]
fn cancelled_execution_retains_only_failure_evidence() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("cancelled");
    let options = RunOptions {
        output: Some(output.clone()),
        ..Default::default()
    };
    assert!(
        run_records_with_cancel(
            config(),
            vec![],
            vec![],
            &parent.path().join("absent.json"),
            &options,
            &|| true
        )
        .is_err()
    );
    assert!(!output.exists());
    let evidence = std::fs::read_dir(parent.path())
        .unwrap()
        .map(|item| item.unwrap().path())
        .find(|path| path.is_dir())
        .unwrap();
    let manifest: Record =
        serde_json::from_slice(&std::fs::read(evidence.join("failure_manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["status"], "cancelled");
    assert!(!evidence.join("run_manifest.json").exists());
}

#[test]
fn figure_selection_removes_unselected_data_and_rejects_unknown_names() {
    let config = json!({"outputs":{"figure_selection":["figure_oft"]}});
    let bundle = crate::plotting::build_bundle(&[], &config).unwrap();
    assert_eq!(bundle["figures"].as_array().unwrap().len(), 1);
    assert_eq!(bundle["figures"][0]["name"], "figure_oft");
    assert!(bundle["plotting_data"].as_array().unwrap().is_empty());
    assert!(bundle["summaries"].as_array().unwrap().is_empty());
    assert!(
        crate::plotting::build_bundle(&[], &json!({"outputs":{"figure_selection":["../escape"]}}))
            .is_err()
    );
}

#[test]
fn output_status_does_not_call_uncorrected_models_complete() {
    let attempts = vec![
        json!({"status":"fitted","contrast":{"p_raw":0.01,"p_adjusted":null,"adjustment_reason":"blocked:D09"}}),
    ];
    assert_eq!(result_status(&attempts, "fitted")["status"], "partial");
    let underflow =
        vec![json!({"status":"fitted","contrast":{"p_adjusted":null,"log_p_adjusted":-900.0}})];
    assert_eq!(result_status(&underflow, "fitted")["status"], "complete");
    let row = json!({"animal_id":"a","assay":"nort","session_id":"s","phase":"test","timepoint":"post",
        "DI":0.2,"excluded":true,"exclusion_reason":"registered","missing_reasons":{"RI":"blocked:D10"},"qc_flags":{"source_warning":true}});
    let ledger = inclusion_ledger(&[row]);
    assert_eq!(
        ledger.iter().find(|row| row["endpoint"] == "DI").unwrap()["eligible_before_model_policy"],
        false
    );
    assert_eq!(
        ledger.iter().find(|row| row["endpoint"] == "RI").unwrap()["metric_status"],
        "blocked"
    );
}

#[test]
fn disabled_figures_keep_their_omission_inventory_without_plot_policy_dependencies() {
    let parent = tempfile::tempdir().unwrap();
    let mut configuration = config();
    configuration["outputs"] = json!({"requested_outputs":["metrics"],"figures_enabled":false});
    let options = RunOptions {
        output: Some(parent.path().join("disabled")),
        ..Default::default()
    };
    let manifest = run_records(
        configuration,
        vec![],
        vec![],
        &parent.path().join("absent.json"),
        &options,
    )
    .unwrap();
    let omissions = manifest["unrendered_figures"].as_array().unwrap();
    assert_eq!(
        omissions
            .iter()
            .map(|item| item["name"].as_str().unwrap())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "figure_oft",
            "figure_ymaze",
            "figure_nort",
            "supplement_ymaze",
            "supplement_nort",
            "qc_ymaze",
            "qc_nort",
            "figure_oft_distributions",
            "figure_ymaze_distributions",
            "figure_nort_distributions",
            "supplement_ymaze_distributions",
            "supplement_nort_distributions",
            "diagnostic_nort_states",
        ])
    );
    assert!(
        omissions
            .iter()
            .all(|figure| figure["status"] == "disabled" && figure["reason"] == "figures_disabled")
    );
}
