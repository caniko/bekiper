use bekiper_post_analysis::plotting::build_bundle;

#[test]
fn distribution_selection_preserves_four_nort_partitions_and_native_points() {
    let mut configuration = config();
    configuration["outputs"]["figure_selection"] = json!(["figure_nort_distributions"]);
    let mut rows = Vec::new();
    for timepoint in ["pre", "post"] {
        for phase in ["training", "test"] {
            for (animal, number) in [("a", 1.0), ("b", 3.0)] {
                let mut r = row(animal, "nort", timepoint, &format!("{timepoint}_{phase}"));
                r["phase"] = phase.into();
                for endpoint in [
                    "TA",
                    "TB",
                    "TF",
                    "TN",
                    "DI",
                    "percent_novel",
                    "object_bias",
                    "side_bias",
                ] {
                    r[endpoint] = number.into();
                }
                rows.push(r);
            }
        }
    }
    let bundle = build_bundle(&rows, &configuration).unwrap();
    assert_eq!(bundle["figures"].as_array().unwrap().len(), 17);
    let panels = figure(&bundle, "figure_nort_distributions")["panels"]
        .as_array()
        .unwrap();
    assert_eq!(panels.len(), 48);
    let pages = bundle["figures"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["parent_figure"] == "figure_nort_distributions")
        .collect::<Vec<_>>();
    assert_eq!(pages.len(), 16);
    let mut copies = Vec::new();
    for page in pages {
        let page_panels = page["panels"].as_array().unwrap();
        assert_eq!(page_panels.len(), 3);
        assert!(
            page_panels
                .iter()
                .all(|p| p["endpoint"] == page_panels[0]["endpoint"]
                    && p["session_partition"] == page_panels[0]["session_partition"])
        );
        for panel in page_panels {
            let mut panel = panel.clone();
            panel.as_object_mut().unwrap().remove("panel_id");
            copies.push(panel.to_string());
        }
    }
    let mut originals = panels
        .iter()
        .map(|panel| {
            let mut panel = panel.clone();
            panel.as_object_mut().unwrap().remove("panel_id");
            panel.to_string()
        })
        .collect::<Vec<_>>();
    copies.sort();
    originals.sort();
    assert_eq!(copies, originals);
    for panel in panels {
        assert_eq!(panel["n_animals"], 2);
        assert_eq!(panel["points"].as_array().unwrap().len(), 2);
        let distribution = &panel["summaries"][0]["distribution"];
        assert_eq!(distribution["box"]["med"], 2.0);
        if matches!(panel["endpoint"].as_str(), Some("TA" | "TB" | "TF" | "TN")) {
            assert_eq!(panel["unit"], "s");
            assert!(panel["label"].as_str().unwrap().contains("(s)"));
        } else {
            assert!(panel["formula"].as_str().unwrap().contains('/'));
        }
        assert!(
            panel["points"]
                .as_array()
                .unwrap()
                .iter()
                .all(|p| p["phase"] == panel["session_partition"]["phase"]
                    && p["timepoint"] == panel["session_partition"]["timepoint"])
        );
    }
}
use serde_json::{Value, json};

#[test]
fn cohort_states_use_recorded_fractions_and_keep_session_partitions() {
    let mut configuration = config();
    configuration["outputs"]["figure_selection"] = json!(["diagnostic_nort_states"]);
    let mut rows = Vec::new();
    for timepoint in ["pre", "post"] {
        for phase in ["training", "test"] {
            let mut r = row("a", "nort", timepoint, phase);
            r["native_state_distributions"] = json!([{
                "kind":"state_distribution", "label":"sensory_disagreement_A",
                "category_labels":["P0 G0 F0","Missing"], "frame_counts":[7,3],
                "seconds":[1.0,3.0], "percentages":[25.0,75.0],
                "denominator_seconds":4.0,"denominator_frames":10,
                "source_columns":["near","gaze","final"], "basis":"half-open recorded clock"
            }]);
            rows.push(r);
        }
    }
    let bundle = build_bundle(&rows, &configuration).unwrap();
    let panels = bundle["figures"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|f| f["panels"].as_array().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(panels.len(), 8);
    for panel in panels {
        assert_eq!(panel["points"].as_array().unwrap().len(), 1);
        let point = &panel["points"][0];
        assert_eq!(point["phase"], panel["session_partition"]["phase"]);
        assert_eq!(point["timepoint"], panel["session_partition"]["timepoint"]);
        assert_eq!(
            point["value"],
            if panel["native_state_category"] == "Missing" {
                json!(75.0)
            } else {
                json!(25.0)
            }
        );
        assert_eq!(panel["style"], "boxplot");
        assert_eq!(point["native_state_provenance"]["denominator_seconds"], 4.0);
        assert_eq!(
            point["native_state_provenance"]["basis"],
            "half-open recorded clock"
        );
    }
    // Category ordering and target identities may change between phases. Join
    // the recorded labels; absent categories stay missing rather than zero.
    rows[0]["native_state_distributions"][0]["category_labels"] = json!(["new-target", "Missing"]);
    let bundle = build_bundle(&rows, &configuration).unwrap();
    let panels = bundle["figures"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|f| f["panels"].as_array().unwrap())
        .collect::<Vec<_>>();
    let missing = panels
        .iter()
        .find(|p| {
            p["native_state_category"] == "P0 G0 F0"
                && p["session_partition"]["timepoint"] == "pre"
                && p["session_partition"]["phase"] == "training"
        })
        .unwrap();
    assert!(missing["points"].as_array().unwrap().is_empty());
    assert_eq!(
        missing["eligibility_ledger"][0]["reason"],
        "category_not_recorded_for_session"
    );
    rows[0]["native_state_distributions"][0]["percentages"][0] = 70.0.into();
    assert!(build_bundle(&rows, &configuration).is_err());
}

fn config() -> Value {
    let mut value = json!({"run_mode":"synthetic","synthetic_fixture":true,
        "assays":{"ymaze":{"primary_threshold_arm_length_pct":45,"requested_sensitivity_thresholds_arm_length_pct":[65]}},
        "statistics":{"summary_uncertainty":"SEM"},
        "inclusion":{"upstream_verification_requirement":"allow_unverified"},
        "outputs":{"figure_formats":["svg"]},"decision_resolutions":{}});
    for (decision, path, setting) in [
        (
            "D03",
            "assays.ymaze.primary_threshold_arm_length_pct",
            json!(45),
        ),
        ("D09", "statistics.summary_uncertainty", json!("SEM")),
        (
            "D12",
            "inclusion.upstream_verification_requirement",
            json!("allow_unverified"),
        ),
    ] {
        value["decision_resolutions"][decision] = json!({"value":{path:setting},
            "scope":["assays","statistics","inclusion"],"origin":"development_fixture",
            "rationale":"test-only","approval_or_plan_reference":"F5 regression fixture"});
    }
    value
}

fn row(animal: &str, assay: &str, timepoint: &str, phase: &str) -> Value {
    json!({"animal_id":animal,"assay":assay,"session_id":format!("{animal}_{assay}_{timepoint}_{phase}"),
        "timepoint":timepoint,"phase":phase,"group":"G5","sex":"female","excluded":false,
        "exclusion_reason":null,"input_verification":{"status":"unresolved"},
        "five_xfad":"AD","hcar1":"WT","treatment":"lactate",
        "qc_flags":{"low_quality":true},"total_distance_m":2,"mean_speed_m_per_s":0.02,
        "center_time_pct":20,"center_entries":4,"latency_to_first_center_entry_s":10})
}

fn figure<'a>(bundle: &'a Value, name: &str) -> &'a Value {
    bundle["figures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|figure| figure["name"] == name)
        .unwrap()
}

#[test]
fn native_population_flags_exclusions_and_uncertainty_are_explicit() {
    let mut rows = vec![
        row("a", "oft", "post", "session"),
        row("b", "oft", "post", "session"),
        row("excluded", "oft", "post", "session"),
    ];
    rows[1]["total_distance_m"] = json!(4);
    rows[2]["total_distance_m"] = json!(999);
    rows[2]["excluded"] = json!(true);
    rows[2]["exclusion_reason"] = json!("metadata exclusion");
    let bundle = build_bundle(&rows, &config()).unwrap();
    let panel = &figure(&bundle, "figure_oft")["panels"][0];
    assert_eq!(panel["n_animals"], 2);
    let summary = &panel["summaries"][0];
    assert_eq!(summary["mean"], 3.0);
    assert!((summary["sem"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    assert_eq!(summary["error_half_width"], 1.0);
    assert!(
        panel["points"]
            .as_array()
            .unwrap()
            .iter()
            .all(|point| point["animal_id"] != "excluded")
    );
    assert!(
        panel["eligibility_ledger"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["animal_id"] == "excluded"
                && entry["reason"].as_str().unwrap().starts_with("excluded:"))
    );
    assert_eq!(figure(&bundle, "figure_oft")["status"], "rendered");
    assert_eq!(panel["points"][0]["qc_flags"]["low_quality"], true);
    assert_eq!(
        panel["eligibility_ledger"][0]["qc_flags"]["low_quality"],
        true
    );
    for key in ["animal_id", "assay", "session_id", "timepoint", "phase"] {
        assert_eq!(panel["points"][0][key], rows[0][key]);
        assert_eq!(panel["eligibility_ledger"][0][key], rows[0][key]);
    }
}

#[test]
fn native_ci_matches_independent_student_t_oracle() {
    let mut configuration = config();
    configuration["statistics"]["summary_uncertainty"] = json!("95% CI");
    configuration["decision_resolutions"]["D09"]["value"]["statistics.summary_uncertainty"] =
        json!("95% CI");
    let mut rows = Vec::new();
    for (animal, distance) in [("a", 1.0), ("b", 3.0), ("c", 5.0)] {
        let mut record = row(animal, "oft", "post", "session");
        record["total_distance_m"] = json!(distance);
        rows.push(record);
    }
    let bundle = build_bundle(&rows, &configuration).unwrap();
    let summary = &figure(&bundle, "figure_oft")["panels"][0]["summaries"][0];
    // SciPy t.ppf(.975,2) * sample_std([1,3,5],ddof=1)/sqrt(3).
    assert!((summary["error_half_width"].as_f64().unwrap() - 4.968275423439091).abs() < 1e-9);
    assert_eq!(summary["n_animals"], 3);
}

#[test]
fn unregistered_summary_does_not_default_to_sem() {
    let mut configuration = config();
    configuration["decision_resolutions"]
        .as_object_mut()
        .unwrap()
        .remove("D09");
    let bundle = build_bundle(
        &[
            row("a", "oft", "post", "session"),
            row("b", "oft", "post", "session"),
        ],
        &configuration,
    )
    .unwrap();
    let result = figure(&bundle, "figure_oft");
    assert_eq!(result["status"], "partial");
    assert_eq!(result["panels"][0]["summaries"][0]["mean"], 2.0);
    assert_eq!(
        result["panels"][0]["summaries"][0]["error_half_width"],
        Value::Null
    );
    assert_eq!(
        result["panels"][0]["summaries"][0]["status"],
        "blocked:D09:summary_uncertainty"
    );
}

#[test]
fn threshold_selection_preserves_primary_and_registered_pairing() {
    let mut rows = Vec::new();
    for threshold in [25, 45, 65] {
        for timepoint in ["pre", "post"] {
            let mut record = row("a", "ymaze", timepoint, "session");
            record["session_id"] = json!(format!("a_{timepoint}"));
            record["threshold_pct"] = json!(threshold);
            record["primary_threshold"] = json!(threshold == 45);
            record["spontaneous_alternation_pct"] =
                json!(if timepoint == "pre" { 20.0 } else { 30.0 });
            record["spontaneous_alternation_pct_standard"] = json!(10.0);
            record["valid_total_arm_entries"] = json!(8);
            record["neutral_triads_pct"] = json!(20.0);
            rows.push(record);
        }
    }
    let bundle = build_bundle(&rows, &config()).unwrap();
    let main = figure(&bundle, "figure_ymaze");
    assert_eq!(main["panels"][0]["points"][0]["threshold_pct"], 45);
    assert_eq!(main["panels"][1]["points"][0]["value"], 10.0);
    assert_eq!(main["panels"][1]["y_label"], "DeltaSA (percentage points)");
    assert_eq!(main["panels"][2]["points"].as_array().unwrap().len(), 2);
    assert_eq!(
        main["panels"][2]["points"][0]["plot_pair_id"],
        main["panels"][2]["points"][1]["plot_pair_id"]
    );
    assert_eq!(main["panels"][2]["summaries"][0]["category"], "G5:pre");
    assert_eq!(main["panels"][2]["summaries"][1]["category"], "G5:post");
    let supplement = figure(&bundle, "supplement_ymaze");
    assert_eq!(supplement["panels"].as_array().unwrap().len(), 2);
    assert!(
        supplement["panels"]
            .as_array()
            .unwrap()
            .iter()
            .all(|panel| panel["points"][0]["threshold_pct"] == 65)
    );
    let mut configuration = config();
    configuration["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"] = json!([]);
    let disabled = build_bundle(&rows, &configuration).unwrap();
    assert!(
        !disabled["figures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|figure| figure["name"] == "supplement_ymaze")
    );
}

#[test]
fn nort_raw_object_pairs_and_reference_lines_are_in_native_bundle() {
    let mut training = row("a", "nort", "post", "training");
    training["TA"] = json!(8);
    training["TB"] = json!(12);
    let mut test = row("a", "nort", "post", "test");
    test["TN"] = json!(12);
    test["TF"] = json!(8);
    test["DI"] = json!(0.2);
    test["percent_novel"] = json!(60);
    test["total_test_exploration"] = json!(20);
    let bundle = build_bundle(&[training, test], &config()).unwrap();
    let main = figure(&bundle, "figure_nort");
    assert_eq!(main["panels"][2]["reference_line"], 0.0);
    assert_eq!(main["panels"][3]["reference_line"], 50.0);
    assert_eq!(main["panels"][0]["n_animals"], 1);
    assert_eq!(main["panels"][0]["points"].as_array().unwrap().len(), 2);
    assert_eq!(main["panels"][1]["points"][0]["value"], 8.0);
    assert_eq!(main["panels"][1]["points"][1]["value"], 12.0);
    assert_eq!(
        main["panels"][1]["points"][0]["plot_pair_id"],
        main["panels"][1]["points"][1]["plot_pair_id"]
    );
}

#[test]
fn require_verified_repeated_animals_and_unknown_figure_names_are_guarded() {
    let mut configuration = config();
    configuration["inclusion"]["upstream_verification_requirement"] = json!("require_verified");
    configuration["decision_resolutions"]["D12"]["value"]["inclusion.upstream_verification_requirement"] =
        json!("require_verified");
    let bundle = build_bundle(
        &[row("unverified", "oft", "post", "session")],
        &configuration,
    )
    .unwrap();
    assert_eq!(figure(&bundle, "figure_oft")["panels"][0]["n_animals"], 0);
    assert!(
        build_bundle(
            &[
                row("a", "oft", "post", "session"),
                row("a", "oft", "post", "session")
            ],
            &config()
        )
        .is_err()
    );
    configuration["outputs"]["figure_selection"] = json!(["../unselected"]);
    assert!(build_bundle(&[], &configuration).is_err());
}

#[test]
fn selection_and_pdf_report_use_only_selected_artifacts() {
    let mut configuration = config();
    configuration["outputs"]["figure_selection"] = json!(["figure_oft"]);
    configuration["outputs"]["report"] = json!({"enabled":true});
    let bundle = build_bundle(&[], &configuration).unwrap();
    assert_eq!(bundle["figures"].as_array().unwrap().len(), 1);
    assert_eq!(bundle["figures"][0]["name"], "figure_oft");
    assert_eq!(bundle["figures"][0]["formats"], json!(["svg", "pdf"]));
    assert_eq!(bundle["figures"][0]["status"], "partial");
    configuration["outputs"]["figure_selection"] = json!([]);
    assert_eq!(
        build_bundle(&[], &configuration).unwrap()["figures"],
        json!([])
    );
}

#[test]
fn raster_dpi_and_omitted_figure_reasons_cross_the_native_boundary() {
    let mut configuration = config();
    configuration["outputs"]["figure_selection"] = json!(["figure_oft"]);
    configuration["outputs"]["figure_dpi"] = json!(180);
    let bundle = build_bundle(&[row("a", "oft", "post", "session")], &configuration).unwrap();
    assert_eq!(figure(&bundle, "figure_oft")["dpi"], 180);
    let omitted = bundle["unrendered_figures"].as_array().unwrap();
    assert_eq!(
        omitted
            .iter()
            .map(|item| item["name"].as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from([
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
    assert!(omitted.iter().all(|entry| entry["status"] == "not_selected" && entry["reason"] == "figure_selection"));
    configuration["outputs"]["figure_selection"] = Value::Null;
    configuration["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"] = json!([]);
    let bundle = build_bundle(&[], &configuration).unwrap();
    assert!(
        bundle["unrendered_figures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["name"] == "supplement_ymaze"
                && entry["reason"] == "empty_sensitivity_selection")
    );
    assert!(
        bundle["unrendered_figures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["name"] == "figure_nort"
                && entry["reason"] == "no_assay_measurements")
    );
    for invalid in [
        json!(0),
        json!(-1),
        json!(true),
        json!(1.5),
        json!(u64::MAX),
    ] {
        configuration["outputs"]["figure_dpi"] = invalid;
        assert!(build_bundle(&[], &configuration).is_err());
    }
}

#[test]
fn registered_empty_subgroups_keep_visible_zero_animal_panels() {
    let mut configuration = config();
    let plan = json!({"strata":[{"field":"sex","levels":["female","male"],"assays":["nort"]}],
        "outputs":["models"],"multiplicity":{"method":"holm","family":"fixture_sex_models"}});
    configuration["statistics"]["subgroup_analysis_plan"] = plan.clone();
    for decision in ["D09", "D12"] {
        configuration["decision_resolutions"][decision]["value"]["statistics.subgroup_analysis_plan"] =
            plan.clone();
        configuration["decision_resolutions"][decision]["scope"] =
            json!(["statistics", "inclusion"]);
    }
    let mut test = row("female1", "nort", "post", "test");
    test["DI"] = json!(0.2);
    let bundle = build_bundle(&[test], &configuration).unwrap();
    let subgroup = figure(&bundle, "subgroup_nort_sex_0");
    assert_eq!(subgroup["panels"].as_array().unwrap().len(), 2);
    assert_eq!(subgroup["panels"][1]["stratum"]["level"], "male");
    assert_eq!(subgroup["panels"][1]["n_animals"], 0);
    assert_eq!(subgroup["panels"][1]["status"], "unavailable_or_blocked");
    assert_eq!(subgroup["status"], "partial");
}

#[test]
fn missing_values_are_distinct_from_zero_and_qc_scatter_uses_pairwise_population() {
    let mut zero = row("zero", "nort", "post", "test");
    zero["DI"] = json!(0.0);
    zero["total_test_exploration"] = json!(20.0);
    let mut missing = row("missing", "nort", "post", "test");
    missing["DI"] = Value::Null;
    missing["total_test_exploration"] = json!(20.0);
    missing["missing_reasons"] = json!({"DI":"zero_total_exploration"});
    let mut incomplete = row("incomplete", "nort", "post", "test");
    incomplete["DI"] = json!(0.3);
    incomplete["total_test_exploration"] = Value::Null;
    let bundle = build_bundle(&[zero, missing, incomplete], &config()).unwrap();
    let main = &figure(&bundle, "figure_nort")["panels"][2];
    assert_eq!(main["n_animals"], 2);
    assert!(
        main["points"]
            .as_array()
            .unwrap()
            .iter()
            .any(|point| point["value"] == 0.0)
    );
    assert!(
        main["eligibility_ledger"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["reason"] == "zero_total_exploration")
    );
    let qc = &figure(&bundle, "qc_nort")["panels"][0];
    assert_eq!(qc["n_animals"], 1);
    assert_eq!(qc["points"][0]["animal_id"], "zero");
    assert_eq!(qc["status"], "partial");
    assert_eq!(qc["x_endpoint"], "total_test_exploration");
    assert_eq!(qc["x_label"], "Total test exploration (s)");
}

#[test]
fn unresolved_primary_designation_blocks_only_dependent_panels() {
    let mut configuration = config();
    configuration["decision_resolutions"]
        .as_object_mut()
        .unwrap()
        .remove("D03");
    let mut record = row("a", "ymaze", "post", "session");
    record["threshold_pct"] = json!(65);
    record["primary_threshold"] = json!(true);
    record["spontaneous_alternation_pct"] = json!(30.0);
    record["spontaneous_alternation_pct_standard"] = json!(20.0);
    let bundle = build_bundle(&[record], &configuration).unwrap();
    let main = figure(&bundle, "figure_ymaze");
    assert_eq!(main["panels"][0]["n_animals"], 0);
    assert_eq!(main["panels"][0]["reason"], "blocked:D03:primary_threshold");
    assert_eq!(
        figure(&bundle, "supplement_ymaze")["panels"][0]["n_animals"],
        1
    );
}

#[test]
fn delta_and_pair_panels_retain_unmatched_attempts_in_their_ledgers() {
    let mut record = row("unpaired", "ymaze", "post", "session");
    record["threshold_pct"] = json!(45);
    record["primary_threshold"] = json!(true);
    record["spontaneous_alternation_pct"] = json!(30.0);
    let bundle = build_bundle(&[record], &config()).unwrap();
    let main = figure(&bundle, "figure_ymaze");
    assert_eq!(main["panels"][0]["n_animals"], 1);
    for index in [1, 2] {
        assert_eq!(main["panels"][index]["n_animals"], 0);
        assert_eq!(
            main["panels"][index]["pairing_ledger"][0]["reason"],
            "unmatched_prepost"
        );
    }
}

#[test]
fn censored_latency_bounds_are_visible_without_an_uncensored_mean() {
    let mut censored = row("censored", "oft", "post", "session");
    censored["latency_to_first_center_entry_s"] = json!(300.0);
    censored["latency_right_censored"] = json!(true);
    let observed = row("observed", "oft", "post", "session");
    let bundle = build_bundle(&[censored, observed], &config()).unwrap();
    let main = figure(&bundle, "figure_oft");
    let latency = &main["panels"][4];
    assert_eq!(latency["n_animals"], 2);
    assert_eq!(latency["points"][0]["censored"], true);
    assert_eq!(latency["summaries"][0]["mean"], Value::Null);
    assert_eq!(latency["summaries"][0]["error_half_width"], Value::Null);
    assert_eq!(
        latency["summaries"][0]["status"],
        "blocked:D09:censored_latency_summary"
    );
    assert_eq!(main["panels"][0]["summaries"][0]["mean"], 2.0);
    assert!(
        main["summary_label"]
            .as_str()
            .unwrap()
            .contains("right-censored")
    );
}
