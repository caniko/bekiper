use bekiper_post_analysis::{genotype, statistics};
use serde_json::{Value, json};

#[test]
fn recorded_annotations_and_missing_biology_are_retained_but_other_factor_conflicts_fail() {
    let (config, mut rows) = factorial_fixture();
    for row in &mut rows {
        row["exclusion_reason"] = Value::Null;
        row["age_treatment_window"] = Value::Null;
    }
    let mut missing = rows[0].clone();
    missing["animal_id"] = "missing".into();
    missing["session_id"] = "missing".into();
    for field in ["group", "five_xfad", "hcar1", "treatment"] {
        missing[field] = Value::Null;
    }
    rows.push(missing);
    let indexed = bekiper_post_analysis::inputs::validate_metadata(&rows, &config).unwrap();
    assert_eq!(indexed.len(), 41);
    let (coded, _) = genotype::project(&rows, &config, "C").unwrap();
    let (subset, ledger) = statistics::analysis_subset(
        &coded,
        &["DI", "five_xfad", "hcar1", "treatment"],
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(subset.len(), 40);
    assert_eq!(
        ledger[40]["genotype_eligibility"]["reason"],
        "missing_assigned_group"
    );
    assert_eq!(
        ledger[40]["missing_fields"],
        json!(["five_xfad", "hcar1", "treatment"])
    );
    let tests = statistics::within_group(
        &coded,
        &["DI".into()],
        0.95,
        Some("allow_unverified"),
        &config["inputs"]["group_mapping"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        tests
            .iter()
            .filter_map(|t| t["n_animals"].as_u64())
            .sum::<u64>(),
        40
    );
    rows[0]["treatment"] = "unknown".into();
    assert!(bekiper_post_analysis::inputs::validate_metadata(&rows, &config).is_err());
}

#[test]
fn sensitivity_context_does_not_promote_unregistered_statistics_to_approved_choices() {
    let (mut config, rows) = factorial_fixture();
    config["statistics"]["multiplicity"]["families"]["planned_contrasts"] = "unapproved".into();
    let analyses =
        bekiper_post_analysis::pipeline::genotypes::run_policies(&rows, &config).unwrap();
    for analysis in analyses {
        assert_eq!(
            analysis.inference.models[0]["contrast"]["adjustment_reason"],
            "blocked:D09:multiplicity_family"
        );
    }
}

#[test]
fn failed_models_keep_registered_genotype_effect_attempts_in_the_holm_family() {
    let (mut config, rows) = factorial_fixture();
    config["statistics"]["model_family_by_endpoint"]["nort.DI"] = "unsupported".into();
    config["statistics"]["multiplicity"]["families"]["genotype_simple_effects"] =
        "supplement".into();
    for key in ["model_family_by_endpoint", "multiplicity"] {
        config["decision_resolutions"]["D09"]["value"][format!("statistics.{key}")] =
            config["statistics"][key].clone();
    }
    let analyses =
        bekiper_post_analysis::pipeline::genotypes::run_policies(&rows, &config).unwrap();
    for (analysis, n) in analyses.iter().zip([2, 3, 2]) {
        let effects = analysis.inference.models[0]["genotype_lactate_effects"]
            .as_array()
            .unwrap();
        assert_eq!(effects.len(), n);
        for effect in effects {
            assert_eq!(effect["family_size"], n);
            assert_eq!(effect["status"], "not_evaluated");
            assert!(effect["p_adjusted"].is_null());
        }
    }
}

#[test]
fn policy_comparisons_preserve_log_only_probabilities_and_registered_confidence() {
    let (config, rows) = factorial_fixture();
    let mut analyses =
        bekiper_post_analysis::pipeline::genotypes::run_policies(&rows, &config).unwrap();
    let tail = statistics::t_summary(1e200, 1.0, 29, 0.95).unwrap();
    assert!(tail["p_raw"].is_null());
    for analysis in &mut analyses {
        let contrast = &mut analysis.inference.models[0]["contrast"];
        contrast["p_raw"] = tail["p_raw"].clone();
        contrast["log_p_raw"] = tail["log_p_raw"].clone();
        contrast["p_adjusted"] = Value::Null;
        contrast["log_p_adjusted"] = tail["log_p_raw"].clone();
    }
    let comparison = bekiper_post_analysis::pipeline::genotypes::comparison(&analyses);
    for row in comparison["rows"].as_array().unwrap() {
        assert_eq!(row["log_p_raw"], tail["log_p_raw"]);
        assert_eq!(row["confidence_level"], 0.95);
    }
    assert_eq!(
        comparison["endpoint_interpretation"][0]["primary_log_adjusted_p"],
        tail["log_p_raw"]
    );
}

#[test]
fn native_policy_models_use_different_animals_without_rescoring_or_primary_selection() {
    let (mut config, mut rows) = factorial_fixture();
    let source = rows.clone();
    let analyses =
        bekiper_post_analysis::pipeline::genotypes::run_policies(&rows, &config).unwrap();
    assert_eq!(analyses.len(), 3);
    for (analysis, n) in analyses.iter().zip([37, 39, 40]) {
        assert_eq!(analysis.inference.models[0]["n_animals"], n);
        assert_eq!(analysis.rows.len(), 40);
        assert_eq!(
            analysis
                .audit
                .iter()
                .filter(|r| r["endpoint"] == "DI")
                .count(),
            40
        );
    }
    let comparison = bekiper_post_analysis::pipeline::genotypes::comparison(&analyses);
    assert_eq!(comparison["primary_policy"], "A");
    assert_eq!(comparison["rows"][0]["primary"], true);
    assert_eq!(comparison["rows"][2]["primary"], false);
    let het_mean = analyses[1].inference.models[0]["factor_cell_estimates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["five_xfad"] == "AD" && r["hcar1"] == "Het" && r["treatment"] == "lactate")
        .unwrap();
    assert_eq!(het_mean["status"], "computed");
    let het_effect = analyses[1].inference.models[0]["genotype_lactate_effects"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["hcar1"] == "Het")
        .unwrap();
    assert_eq!(het_effect["status"], "not_estimable");
    assert_eq!(
        analyses[1].inference.models[0]["contrast"]["status"],
        "computed"
    );
    assert_eq!(rows, source);

    config["run_mode"] = "synthetic".into();
    config["synthetic_fixture"] = true.into();
    config["inputs"]["units_and_calibration"] =
        json!({"nort":{"time":"s","distance":"m","speed":"m/s"}});
    config["assays"] = json!({"oft":{},"nort":{},"ymaze":{}});
    config["outputs"] =
        json!({"requested_outputs":["metrics","models","report"],"report":{"enabled":false}});
    for row in &mut rows {
        let di = row["DI"].as_f64().unwrap() / 100.0;
        row["TN"] = json!((1.0 + di) * 20.0);
        row["TF"] = json!((1.0 - di) * 20.0);
        row["synthetic_observation"] = true.into();
        row["exclusion_reason"] = Value::Null;
    }
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("policies");
    let manifest = bekiper_post_analysis::pipeline::run_records(
        config,
        rows.clone(),
        vec![],
        &parent.path().join("plan.json"),
        &bekiper_post_analysis::RunOptions {
            output: Some(output.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(manifest["hcar1_policies"]["primary"], "A");
    let read = |name: &str| -> Value {
        serde_json::from_slice(
            &std::fs::read(bekiper_post_analysis::export::input_path(&output, name)).unwrap(),
        )
        .unwrap()
    };
    let retained = read("metrics.json");
    assert_eq!(retained.as_array().unwrap().len(), 40);
    assert_eq!(
        retained
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["animal_id"] == "37")
            .unwrap()["hcar1"],
        "Het/KO"
    );
    let results = read("genotype_policies.json");
    assert_eq!(results["policies"].as_array().unwrap().len(), 3);
    assert_eq!(read("models.json")[0]["n_animals"], 37);
    let report = std::fs::read_to_string(output.join("report.md")).unwrap();
    assert!(report.contains("primary A and sensitivity B/C"));
    assert!(report.contains("assigned-group"));
    bekiper_post_analysis::pipeline::verify_artifacts(&output).unwrap();
    let verified = bekiper_post_analysis::verification::verify_run(&output).unwrap();
    assert_eq!(
        verified["contexts"][0]["policies"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    std::fs::write(
        output.join("metrics.toml"),
        "schema_version = '1.0'\nrun_mode = 'synthetic'\nstatus = 'partial'\n",
    )
    .unwrap();
    bekiper_post_analysis::pipeline::write_artifact_manifest(&output).unwrap();
    assert!(bekiper_post_analysis::verification::verify_run(&output).is_err());
}

fn factorial_fixture() -> (Value, Vec<Value>) {
    let oracle: Value =
        serde_json::from_str(include_str!("fixtures/statistics_oracle.json")).unwrap();
    let groups = json!({
        "G1":{"five_xfad":"AD","hcar1":"KO","treatment":"lactate"},
        "G2":{"five_xfad":"AD","hcar1":"KO","treatment":"vehicle"},
        "G3":{"five_xfad":"WT","hcar1":"KO","treatment":"lactate"},
        "G4":{"five_xfad":"WT","hcar1":"KO","treatment":"vehicle"},
        "G5":{"five_xfad":"AD","hcar1":"WT","treatment":"lactate"},
        "G6":{"five_xfad":"AD","hcar1":"WT","treatment":"vehicle"},
        "G7":{"five_xfad":"WT","hcar1":"WT","treatment":"lactate"},
        "G8":{"five_xfad":"WT","hcar1":"WT","treatment":"vehicle"}});
    let plan = json!({"primary":"A", "policies":["A", "B", "C"]});
    let mut config = json!({"statistics":{"hcar1_policy_plan":plan},"inputs":{"group_mapping":groups},
        "decision_resolutions":{"D10":{"origin":"researcher_decision","rationale":"Three authorized policies",
            "approval_or_plan_reference":"test researcher plan","scope":["statistics","inputs"],
            "value":{"statistics.hcar1_policy_plan":plan,"inputs.group_mapping":groups}}}});
    config["inputs"]["factor_levels"] =
        json!({"five_xfad":["WT","AD"],"hcar1":["KO","WT"],"treatment":["vehicle","lactate"]});
    config["decision_resolutions"]["D10"]["value"]["inputs.group_mapping"] = groups;
    config["decision_resolutions"]["D10"]["value"]["inputs.factor_levels"] =
        config["inputs"]["factor_levels"].clone();
    config["statistics"]
        .as_object_mut()
        .unwrap()
        .extend(oracle["settings"].as_object().unwrap().clone());
    config["statistics"]["model_family_by_endpoint"] = json!({"nort.DI":"gaussian_ols"});
    config["statistics"]["nort_additional_covariates"] = json!([]);
    config["statistics"]["model_failure_policy"] = "report_without_simplification".into();
    config["statistics"]["multiplicity"] =
        json!({"method":"holm","families":{"planned_contrasts":"test_primary"}});
    config["outputs"] = json!({"requested_outputs":["models"]});
    config["inclusion"] = json!({"existing_metadata_exclusions_policy":"honor_metadata","qc_flag_exclusion_policy":"flags_only",
        "missing_metric_policy":"endpoint_complete_cases","upstream_verification_requirement":"allow_unverified"});
    for (decision, section) in [("D09", "statistics"), ("D12", "inclusion")] {
        let values: serde_json::Map<String, Value> = config[section]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (format!("{section}.{k}"), v.clone()))
            .collect();
        config["decision_resolutions"][decision] = json!({"origin":"researcher_decision","rationale":"test only","approval_or_plan_reference":"test",
            "scope":[section],"value":values});
    }
    let mut rows = oracle["rows"].as_array().unwrap().clone();
    for row in &mut rows {
        row["assay"] = "nort".into();
        row["phase"] = "test".into();
        row["timepoint"] = "post".into();
        let group = match (
            row["five_xfad"].as_str().unwrap(),
            row["hcar1"].as_str().unwrap(),
            row["treatment"].as_str().unwrap(),
        ) {
            ("AD", "KO", "lactate") => "G1",
            ("AD", "KO", _) => "G2",
            ("WT", "KO", "lactate") => "G3",
            ("WT", "KO", _) => "G4",
            ("AD", "WT", "lactate") => "G5",
            ("AD", "WT", _) => "G6",
            ("WT", "WT", "lactate") => "G7",
            _ => "G8",
        };
        row["group"] = group.into();
    }
    for id in ["het1", "het2"] {
        let mut r = rows.iter().find(|r| r["group"] == "G5").unwrap().clone();
        r["animal_id"] = id.into();
        r["session_id"] = id.into();
        r["hcar1"] = "Het".into();
        rows.push(r);
    }
    let mut unresolved = rows.iter().find(|r| r["group"] == "G4").unwrap().clone();
    unresolved["animal_id"] = "37".into();
    unresolved["session_id"] = "37".into();
    unresolved["hcar1"] = "Het/KO".into();
    rows.push(unresolved);
    (config, rows)
}

#[test]
fn paired_assays_and_all_three_thresholds_retain_policy_population_and_labels() {
    let (mut config, source) = factorial_fixture();
    config["statistics"]["oft_baseline_strategy"] = "baseline_adjusted".into();
    config["statistics"]["contrast_estimand_and_weighting"]["reference_values"] =
        json!({"pre_outcome":"mean"});
    config["inclusion"]["unmatched_prepost_policy"] = "report_and_withhold_pair".into();
    config["statistics"]["model_family_by_endpoint"] =
        json!({"nort.DI":"gaussian_ols","ymaze.spontaneous_alternation_pct":"gaussian_ols"});
    for &endpoint in bekiper_post_analysis::contracts::oft_primary_fields() {
        config["statistics"]["model_family_by_endpoint"][format!("oft.{endpoint}")] =
            json!("gaussian_ols");
    }
    config["statistics"]["summary_uncertainty"] = "95% CI".into();
    config["outputs"] = json!({"requested_outputs":["models"],"figure_selection":["figure_ymaze"]});
    for (decision, section) in [("D09", "statistics"), ("D12", "inclusion")] {
        for (key, value) in config[section].as_object().unwrap().clone() {
            config["decision_resolutions"][decision]["value"][format!("{section}.{key}")] = value;
        }
    }
    let mut rows = Vec::new();
    for source in &source {
        rows.push(source.clone());
        for (assay, thresholds) in [
            ("oft", vec![None]),
            ("ymaze", vec![Some(25), Some(45), Some(65)]),
        ] {
            for threshold in thresholds {
                for time in ["pre", "post"] {
                    let mut row = source.clone();
                    row["assay"] = assay.into();
                    row["phase"] = "session".into();
                    row["timepoint"] = time.into();
                    row["session_id"] =
                        format!("{assay}-{time}-{}", row["animal_id"].as_str().unwrap()).into();
                    row["threshold_pct"] = json!(threshold);
                    row["exclusion_reason"] = Value::Null;
                    row["qc_flags"] = json!({"low_quality":true});
                    let field = if time == "pre" { "pre_outcome" } else { "DI" };
                    let value = source[field].as_f64().unwrap();
                    for endpoint in bekiper_post_analysis::contracts::oft_primary_fields()
                        .iter()
                        .copied()
                        .chain(["spontaneous_alternation_pct"])
                    {
                        row[endpoint] = json!(value);
                    }
                    rows.push(row);
                }
            }
        }
    }
    for threshold in [25, 45, 65] {
        config["assays"] = json!({"ymaze":{"primary_threshold_arm_length_pct":threshold}});
        config["decision_resolutions"]["D03"] = json!({"origin":"researcher_decision","scope":["assays.ymaze"],"rationale":"test","approval_or_plan_reference":"test",
            "value":{"assays.ymaze.primary_threshold_arm_length_pct":threshold}});
        for row in &mut rows {
            row["primary_threshold"] = json!(row["threshold_pct"] == threshold);
        }
        let analyses =
            bekiper_post_analysis::pipeline::genotypes::run_policies(&rows, &config).unwrap();
        for (analysis, n) in analyses.iter().zip([37, 39, 40]) {
            assert_eq!(analysis.inference.models.len(), 8);
            for model in &analysis.inference.models {
                assert_eq!(model["n_animals"], n);
                assert_eq!(
                    model["contrast"]["status"], "computed",
                    "{} / {}: {}",
                    analysis.policy, model["model_id"], model["reason"]
                );
                assert_eq!(model["contrast"]["family_size"], 8);
                assert!(
                    model["contrast"]["correction_family"]
                        .as_str()
                        .unwrap()
                        .ends_with(&format!("HCAR1_policy_{}", analysis.policy))
                );
                assert_eq!(
                    model["formula"].as_str().unwrap().contains("pre_outcome"),
                    model["model_id"] != "nort.DI"
                );
            }
            let bundle =
                bekiper_post_analysis::plotting::build_bundle(&analysis.rows, &analysis.config)
                    .unwrap();
            let figure = &bundle["figures"][0];
            for panel in figure["panels"].as_array().unwrap().iter().take(3) {
                assert_eq!(
                    panel["n_animals"], n,
                    "policy {} / {:?}",
                    analysis.policy, panel["label"]
                );
                if analysis.policy == "A" {
                    assert!(
                        panel["points"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|point| point["literal_hcar1"] == "WT"
                                || point["literal_hcar1"] == "KO")
                    );
                }
            }
        }
    }
}
