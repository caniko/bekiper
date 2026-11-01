//! Numerical and orchestration edge cases: extreme magnitudes stay explicit
//! statuses or errors, never silently finite results or reduced families.
use bekiper_post_analysis::{pipeline, statistics::*};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};

fn oracle() -> Value {
    serde_json::from_str(include_str!("fixtures/statistics_oracle.json")).unwrap()
}

fn fit(rows: &[Value], endpoint: &str, settings: &Value, baseline: bool) -> anyhow::Result<Value> {
    let oracle = oracle();
    fit_factorial(
        rows,
        endpoint,
        &oracle["levels"],
        &oracle["groups"],
        settings,
        baseline,
        Some("allow_unverified"),
    )
}

fn scaled(field: &str, factor: f64) -> Vec<Value> {
    let mut rows = oracle()["rows"].as_array().unwrap().clone();
    for row in &mut rows {
        row[field] = json!(row[field].as_f64().unwrap() * factor);
    }
    rows
}

fn with_covariate(covariate: &str, reference: Value) -> Value {
    let mut settings = oracle()["settings"].clone();
    settings["additional_covariates"] = json!([covariate]);
    settings["contrast_estimand_and_weighting"]["reference_values"][covariate] = reference;
    settings
}

/// Sets the value and its matching registration, so `inputs::choice` accepts it.
fn register(config: &mut Value, decision: &str, path: &str, value: Value) {
    let mut target = &mut *config;
    for field in path.split('.') {
        target = &mut target[field];
    }
    *target = value.clone();
    let record = &mut config["decision_resolutions"][decision];
    if record.is_null() {
        *record = json!({"value":{},"scope":[],"origin":"researcher_decision",
            "rationale":"coverage edge case","approval_or_plan_reference":"test"});
    }
    let section = json!(path.split('.').next().unwrap());
    if !record["scope"].as_array().unwrap().contains(&section) {
        record["scope"].as_array_mut().unwrap().push(section);
    }
    record["value"][path] = value;
}

fn register_inclusion(config: &mut Value) {
    for (path, value) in [
        ("existing_metadata_exclusions_policy", "honor_metadata"),
        ("qc_flag_exclusion_policy", "flags_only"),
        ("missing_metric_policy", "endpoint_complete_cases"),
        ("upstream_verification_requirement", "allow_unverified"),
    ] {
        register(config, "D12", &format!("inclusion.{path}"), json!(value));
    }
}

/// A fully registered NORT model plan over the independent oracle cohort.
fn nort_models() -> (Value, Vec<Value>) {
    let oracle = oracle();
    let mut config = json!({"outputs":{"requested_outputs":["models"]}});
    register_inclusion(&mut config);
    register(
        &mut config,
        "D10",
        "inputs.group_mapping",
        oracle["groups"].clone(),
    );
    register(
        &mut config,
        "D10",
        "inputs.factor_levels",
        oracle["levels"].clone(),
    );
    for (field, value) in oracle["settings"].as_object().unwrap() {
        if !["model_family", "additional_covariates"].contains(&field.as_str()) {
            register(
                &mut config,
                "D09",
                &format!("statistics.{field}"),
                value.clone(),
            );
        }
    }
    register(
        &mut config,
        "D09",
        "statistics.model_family_by_endpoint",
        json!({"nort.DI":"gaussian_ols"}),
    );
    register(
        &mut config,
        "D09",
        "statistics.model_failure_policy",
        json!("report_without_simplification"),
    );
    register(
        &mut config,
        "D09",
        "statistics.nort_additional_covariates",
        json!([]),
    );
    let mut rows = oracle["rows"].as_array().unwrap().clone();
    for row in &mut rows {
        row["assay"] = "nort".into();
        row["phase"] = "test".into();
        row["timepoint"] = "post".into();
    }
    (config, rows)
}

#[test]
fn covariate_lists_latency_and_empty_populations_are_explicit() {
    let rows = oracle()["rows"].as_array().unwrap().clone();
    let mut settings = oracle()["settings"].clone();
    settings
        .as_object_mut()
        .unwrap()
        .remove("additional_covariates");
    assert_eq!(
        fit(&rows, "DI", &settings, false).unwrap()["status"],
        "fitted"
    );
    settings["additional_covariates"] = "age".into();
    let error = fit(&rows, "DI", &settings, false).unwrap_err();
    assert!(error.to_string().contains("require a list"), "{error:#}");
    // Uncensored post-only latency is fitted; only a registered censoring flag blocks it.
    let latency: Vec<_> = rows
        .iter()
        .cloned()
        .map(|mut row| {
            row["latency_to_first_center_entry_s"] = row["DI"].clone();
            row["latency_right_censored"] = false.into();
            row
        })
        .collect();
    let model = fit(
        &latency,
        "latency_to_first_center_entry_s",
        &oracle()["settings"],
        false,
    )
    .unwrap();
    assert_eq!(model["status"], "fitted");
    let excluded: Vec<_> = rows
        .iter()
        .cloned()
        .map(|mut row| {
            row["excluded"] = true.into();
            row
        })
        .collect();
    let model = fit(&excluded, "DI", &oracle()["settings"], false).unwrap();
    assert_eq!(model["status"], "unavailable");
    assert_eq!(model["reason"], "insufficient_data");
    assert_eq!(model["n_animals"], 0);
    assert_eq!(model["subset"].as_array().unwrap().len(), rows.len());
}

#[test]
fn extreme_response_magnitudes_report_numerical_failure_instead_of_inference() {
    let settings = oracle()["settings"].clone();
    // Residual variance squares the residual norm: ~1e200 overflows, ~1e-170 underflows.
    let model = fit(&scaled("DI", 1e200), "DI", &settings, false).unwrap();
    assert_eq!(model["status"], "error");
    assert_eq!(model["reason"], "numerical_nonfinite_model_result");
    assert!(model["contrast"].is_null());
    let model = fit(&scaled("DI", 1e-170), "DI", &settings, false).unwrap();
    assert_eq!(model["reason"], "residual_variance_underflow");
    assert_eq!(model["residual_variance"], 0.0);
    assert!(model["contrast"].is_null());
    // A tiny-unit covariate column is rescaled for the SVD, but its covariance is unrepresentable.
    let model = fit(
        &scaled("age", 1e-200),
        "DI",
        &with_covariate("age", json!("mean")),
        false,
    )
    .unwrap();
    assert_eq!(model["reason"], "numerical_nonfinite_covariance");
    assert_eq!(model["coefficients"], json!([]));
}

#[test]
fn covariate_reference_values_are_registered_not_inferred() {
    let rows = oracle()["rows"].as_array().unwrap().clone();
    let model = fit(&rows, "DI", &with_covariate("sex", json!("unknown")), false).unwrap();
    assert_eq!(model["status"], "blocked");
    assert_eq!(
        model["reason"],
        "blocked:D09:categorical_covariate_reference"
    );
    assert!(model["contrast"].is_null());
    let model = fit(&rows, "DI", &with_covariate("age", json!(7.5)), false).unwrap();
    assert_eq!(model["status"], "fitted");
    assert_eq!(model["contrast"]["reference_values"]["age"], 7.5);
    assert!(fit(&rows, "DI", &with_covariate("age", json!("median")), false).is_err());
    // A finite reference whose cell-mean variance overflows cannot yield a finite interval.
    let error = fit(&rows, "DI", &with_covariate("age", json!(1e200)), false).unwrap_err();
    assert!(error.to_string().contains("standard error"), "{error:#}");
}

#[test]
fn student_t_degenerate_and_overflowing_summaries_are_explained() {
    let zero = t_summary(1.0, 0.0, 3, 0.95).unwrap();
    assert_eq!(zero["reason"], "undefined:zero_standard_error");
    assert!(zero["statistic"].is_null() && zero["ci_lower"].is_null());
    let overflow = t_summary(1e300, 1e-300, 3, 0.95).unwrap();
    assert_eq!(overflow["reason"], "undefined:statistic_overflow");
    assert!(overflow["p_raw"].is_null());
    let interval = t_summary(1e308, 1e307, 1, 0.95).unwrap();
    assert_eq!(interval["reason"], "undefined:confidence_interval_overflow");
    assert!(interval["ci_lower"].is_null() && interval["ci_upper"].is_null());
    assert!(interval["p_raw"].as_f64().unwrap() > 0.0);
}

#[test]
fn unadjusted_family_keeps_log_only_probabilities() {
    let mut tests = vec![
        json!({"p_raw":0.02}),
        json!({"p_raw":null,"log_p_raw":-2000.0}),
        json!({"p_raw":null}),
    ];
    adjust_family(&mut tests, "none", "registered_none").unwrap();
    assert_eq!(tests[0]["p_adjusted"], 0.02);
    assert_eq!(tests[1]["log_p_adjusted"], -2000.0);
    assert!(tests[1]["p_adjusted"].is_null());
    assert_eq!(
        tests[1]["adjustment_reason"],
        "p_underflow:see_log_p_adjusted"
    );
    assert!(tests[2]["log_p_adjusted"].is_null());
    assert!(tests.iter().all(|t| t["family_size"] == 3));
}

#[test]
fn within_group_mean_overflow_fails_instead_of_reporting_a_test() {
    let rows: Vec<_> = [1.7e308, -1.7e308]
        .iter()
        .enumerate()
        .map(|(i, di)| json!({"animal_id":format!("a{i}"),"session_id":format!("s{i}"),"group":"G5","DI":di}))
        .collect();
    let error = within_group(
        &rows,
        &["DI".into()],
        0.95,
        Some("allow_unverified"),
        &["G5".into()],
    )
    .unwrap_err();
    assert!(error.to_string().contains("finite estimate"), "{error:#}");
}

fn spearman_rows(x: &[f64], y: impl Fn(usize) -> f64) -> Vec<Value> {
    x.iter()
        .enumerate()
        .map(|(i, x)| {
            json!({"animal_id":format!("a{i}"),"session_id":format!("s{i}"),
                "spontaneous_alternation_pct":x,"valid_total_arm_entries":y(i),
                "total_distance_m":y(i),"mean_speed_m_per_s":y(i),"neutral_triads_pct":y(i)})
        })
        .collect()
}

#[test]
fn spearman_constant_perfect_and_underflowing_tails_are_not_literal_probabilities() {
    let constant = spearman_pairs(
        &spearman_rows(&[50.0; 4], |i| i as f64),
        Some("allow_unverified"),
    )
    .unwrap();
    assert!(constant.iter().all(|t| t["reason"] == "constant_input"));
    assert!(constant.iter().all(|t| t["rho"].is_null()));
    let perfect = spearman_pairs(
        &spearman_rows(&[1.0, 2.0, 3.0, 4.0], |i| 10.0 * i as f64),
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(perfect[0]["rho"], 1.0);
    assert_eq!(
        perfect[0]["reason"],
        "numerical_p_zero:not_reported_as_literal_zero"
    );
    assert!(perfect[0]["p_raw"].is_null());
    // One adjacent rank swap among 200 animals: rho < 1 but the tail underflows.
    let x: Vec<_> = (0..200).map(f64::from).collect();
    let swapped = spearman_pairs(
        &spearman_rows(&x, |i| match i {
            0 => 1.0,
            1 => 0.0,
            i => i as f64,
        }),
        Some("allow_unverified"),
    )
    .unwrap();
    assert!(swapped[0]["rho"].as_f64().unwrap() < 1.0);
    assert!(swapped[0]["p_raw"].is_null());
    assert!(swapped[0]["log_p_raw"].as_f64().unwrap() < -745.0);
    assert_eq!(swapped[0]["reason"], "p_underflow:see_log_p_raw");
    assert_eq!(swapped[0]["status"], "unavailable");
}

#[test]
fn verification_requirements_are_closed_and_verified_rows_pass() {
    let verified = json!({"input_verification":{"status":"verified_source_rule_definition"}});
    assert_eq!(
        verification_reason(&verified, Some("require_verified")),
        None
    );
    assert_eq!(
        verification_reason(&verified, Some("trust_upstream")).as_deref(),
        Some("blocked:D12:unknown_upstream_verification_requirement")
    );
    let rows = [json!({"animal_id":"a","session_id":"s","DI":1})];
    assert!(analysis_subset(&rows, &["DI"], Some("trust_upstream")).is_err());
}

fn paired(animal: &str, timepoint: &str, sa: Value, extra: Value) -> Value {
    let mut row = json!({"animal_id":animal,"assay":"ymaze","phase":"session","pair_id":"replicate-1",
        "session_id":format!("{timepoint}-{animal}"),"timepoint":timepoint,"spontaneous_alternation_pct":sa,
        "input_verification":{"status":"verified_clarified_entry_definition"}});
    row.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    row
}

#[test]
fn pairing_withholds_excluded_unverified_post_and_missing_endpoint_pairs() {
    let rows = vec![
        paired("excluded", "pre", json!(50), json!({"excluded":true})),
        paired("excluded", "post", json!(60), json!({})),
        paired("unverified", "pre", json!(50), json!({})),
        paired(
            "unverified",
            "post",
            json!(60),
            json!({"input_verification":{"status":"unresolved"}}),
        ),
        paired("missing", "pre", json!(50), json!({})),
        paired("missing", "post", Value::Null, json!({})),
        paired("kept", "pre", json!(50), json!({})),
        paired("kept", "post", json!(65), json!({})),
    ];
    let (pairs, ledger) = pairing(
        &rows,
        "spontaneous_alternation_pct",
        Some("require_verified"),
    )
    .unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0]["DeltaSA"], 15.0);
    let reason = |animal: &str| {
        ledger
            .iter()
            .find(|entry| entry["animal_id"] == animal)
            .unwrap()["reason"]
            .clone()
    };
    assert_eq!(reason("excluded"), "excluded:paired_metadata");
    assert_eq!(reason("unverified"), "upstream_unverified:post:unresolved");
    assert_eq!(reason("missing"), "missing_pair_endpoint");
    assert!(reason("kept").is_null());
    assert!(ledger.iter().all(|entry| entry["pair_id"] == "replicate-1"));
}

#[test]
fn unregistered_inference_is_blocked_per_declared_attempt() {
    let config =
        json!({"outputs":{"requested_outputs":["correlations","within_group_tests","models"]}});
    let rows = [
        json!({"animal_id":"a","session_id":"s","assay":"rotarod","timepoint":"post"}),
        json!({"animal_id":"a","session_id":"n","assay":"nort","phase":"test","timepoint":"post"}),
    ];
    let result = pipeline::run_models(&rows, &config).unwrap();
    // Unknown assays declare no model attempts; registered assays keep blocked attempts.
    let ids: Vec<_> = result
        .models
        .iter()
        .map(|m| m["model_id"].clone())
        .collect();
    assert_eq!(
        ids,
        [json!("nort.DI"), json!("ymaze.spontaneous_alternation_pct")]
    );
    assert!(result.models.iter().all(|m| m["status"] == "blocked"));
    assert_eq!(result.correlations.len(), 4);
    for test in &result.correlations {
        assert_eq!(test["status"], "blocked");
        assert!(
            test["reason"]
                .as_array()
                .unwrap()
                .contains(&json!("D03:primary_threshold"))
        );
    }
    assert_eq!(result.within_group_tests.len(), 1);
    assert_eq!(result.within_group_tests[0]["status"], "blocked");
    assert_eq!(
        result.within_group_tests[0]["adjustment_reason"],
        "blocked:D09:multiplicity_family"
    );
}

#[test]
fn registered_nort_within_group_overflow_and_model_errors_propagate() {
    let (mut config, _) = nort_models();
    config["outputs"]["requested_outputs"] = json!(["within_group_tests"]);
    register(
        &mut config,
        "D09",
        "statistics.nort_within_group_checks",
        json!(["DI"]),
    );
    register(
        &mut config,
        "D09",
        "statistics.nort_within_group_test_method",
        json!("one_sample_t"),
    );
    register(
        &mut config,
        "D09",
        "statistics.multiplicity",
        json!({"method":"holm","families":{"nort_within_group":"nort_checks"}}),
    );
    let rows: Vec<_> = [1.7e308, -1.7e308]
        .iter()
        .enumerate()
        .map(|(i, di)| {
            json!({"animal_id":format!("a{i}"),"session_id":format!("s{i}"),
            "assay":"nort","phase":"test","timepoint":"post","group":"G5","DI":di})
        })
        .collect();
    let error = pipeline::run_models(&rows, &config).err().unwrap();
    assert!(error.to_string().contains("finite estimate"), "{error:#}");
    let (mut config, rows) = nort_models();
    register(
        &mut config,
        "D09",
        "statistics.nort_additional_covariates",
        json!("age"),
    );
    let error = pipeline::run_models(&rows, &config).err().unwrap();
    assert!(error.to_string().contains("require a list"), "{error:#}");
}

#[test]
fn unsupported_multiplicity_methods_fail_for_each_registered_family() {
    for family in ["planned_contrasts", "genotype_simple_effects"] {
        let mut config = json!({"outputs":{"requested_outputs":["models"]}});
        register(
            &mut config,
            "D09",
            "statistics.multiplicity",
            json!({"method":"bonferroni","families":{family:"registered"}}),
        );
        let error = pipeline::run_models(&[], &config).err().unwrap();
        assert!(error.to_string().contains("multiplicity"), "{error:#}");
    }
}

#[test]
fn subgroups_require_both_registrations_and_ledger_missing_strata() {
    let (rows, status) = pipeline::run_subgroups(&[], &json!({})).unwrap();
    assert!(rows.is_empty());
    assert_eq!(status["status"], "blocked");
    assert_eq!(status["decision_ids"], json!(["D09", "D12"]));
    let plan = json!({"strata":[{"field":"sex","levels":["female"],"assays":["nort"]}],
        "outputs":["models"],"multiplicity":{"method":"holm","family":"subgroups"}});
    let mut config = json!({});
    for decision in ["D09", "D12"] {
        register(
            &mut config,
            decision,
            "statistics.subgroup_analysis_plan",
            plan.clone(),
        );
    }
    let members = [
        json!({"animal_id":"a","session_id":"s1","assay":"nort","phase":"test","timepoint":"post","sex":"female"}),
        json!({"animal_id":"b","session_id":"s2","assay":"nort","phase":"test","timepoint":"post","sex":null}),
        json!({"animal_id":"c","session_id":"s3","assay":"nort","phase":"test","timepoint":"post","sex":"male"}),
    ];
    let (results, _) = pipeline::run_subgroups(&members, &config).unwrap();
    let reasons: Vec<_> = results[0]["membership_ledger"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["reason"].clone())
        .collect();
    assert_eq!(
        reasons,
        [
            Value::Null,
            json!("missing_stratum_metadata"),
            json!("outside_registered_stratum")
        ]
    );
}

#[test]
fn policy_comparison_without_estimates_is_unavailable_not_compared() {
    let (mut config, mut rows) = nort_models();
    let plan = json!({"primary":"A","policies":["A","B","C"]});
    register(&mut config, "D10", "statistics.hcar1_policy_plan", plan);
    register(
        &mut config,
        "D09",
        "statistics.model_family_by_endpoint",
        json!({"nort.DI":"unsupported"}),
    );
    for row in &mut rows {
        let groups = &oracle()["groups"];
        row["group"] = groups
            .as_object()
            .unwrap()
            .iter()
            .find(|(_, g)| {
                ["five_xfad", "hcar1", "treatment"]
                    .iter()
                    .all(|f| g[*f] == row[*f])
            })
            .map_or(Value::Null, |(name, _)| json!(name));
    }
    let analyses = pipeline::genotypes::run_policies(&rows, &config).unwrap();
    let comparison = pipeline::genotypes::comparison(&analyses);
    let interpretation = comparison["endpoint_interpretation"].as_array().unwrap();
    assert_eq!(interpretation.len(), 2);
    for row in interpretation {
        assert_eq!(row["status"], "comparison_unavailable");
        assert_eq!(row["primary_status"], "blocked");
    }
}

#[test]
fn archive_checks_accept_absent_exports_and_empty_selections() {
    let root = tempfile::tempdir().unwrap();
    assert!(
        pipeline::verify_project_exports(root.path())
            .unwrap()
            .is_empty()
    );
    pipeline::verify_data_archive(&root.path().join("absent.tar.zst"), &[]).unwrap();
}

#[test]
fn renderer_is_not_started_after_cancellation() {
    let root = tempfile::tempdir().unwrap();
    let stdout = root.path().join("stdout");
    let error = pipeline::run_child(
        std::process::Command::new("sh").args(["-c", "echo started"]),
        &stdout,
        &root.path().join("stderr"),
        &|| true,
    )
    .unwrap_err();
    assert!(error.to_string().contains("cancelled"));
    assert!(!stdout.exists());
}

#[test]
fn cancelled_descriptive_primary_context_retains_failure_evidence() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("contexts");
    let rows = vec![
        json!({"animal_id":"a","assay":"ymaze","session_id":"one","timepoint":"post",
        "phase":"session","threshold_pct":25,"value":7.0,"scalar_endpoints":["value"]}),
    ];
    // Calls: outer publication, primary worker, then the context's own publication.
    let calls = AtomicUsize::new(0);
    let error = pipeline::publish_descriptive_with_cancel(
        rows,
        vec![],
        vec![],
        json!({"outputs":{"figures_enabled":false},
            "execution":{"ymaze_primary_thresholds":[25],"primary_workers":1}}),
        &output,
        &|| calls.fetch_add(1, Ordering::SeqCst) >= 2,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("cancelled"), "{error:#}");
    assert!(!output.exists());
    let retained: Vec<_> = std::fs::read_dir(root.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.join("failure_manifest.json").is_file())
        .collect();
    assert_eq!(retained.len(), 1);
    // The context's own failure evidence is nested inside the retained parent staging.
    let nested: Vec<_> = std::fs::read_dir(&retained[0])
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.join("failure_manifest.json").is_file())
        .collect();
    assert_eq!(nested.len(), 1);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(nested[0].join("failure_manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["status"], "cancelled");
    assert_eq!(manifest["published"], false);
}
