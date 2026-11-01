#[test]
fn competing_object_intervals_cannot_be_consumed_as_merge_gaps() {
    // The old object-wise merge counted 4 s inside a 2.4 s session span.
    let parts: Vec<_> = (0..6)
        .map(|i| {
            segment(
                &format!("s{i}"),
                if i % 2 == 0 { "TN" } else { "TF" },
                f64::from(i) * 0.4,
                f64::from(i + 1) * 0.4,
            )
        })
        .collect();
    for policy in ["include_gaps", "exclude_gaps"] {
        let result = nort(&parts, policy);
        assert_eq!(result["valid_bouts"], 0);
        assert_eq!(result["object_exploration_s"]["TN"], 0.0);
        assert_eq!(result["object_exploration_s"]["TF"], 0.0);
        let merged: Vec<_> = result["bout_debug"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|b| b["stage"] == "merged")
            .collect();
        assert_eq!(merged.len(), 6);
        assert!(
            merged
                .iter()
                .any(|b| !b["merge_barrier_ids"].as_array().unwrap().is_empty())
        );
    }
}

#[test]
fn bouts_merge_before_filtering_and_retain_observed_samples_lineage() {
    let parts = [
        segment("a", "novel", 0.0, 0.6),
        segment("b", "novel", 1.0, 1.6),
    ];
    let result = nort(&parts, "exclude_gaps");
    assert!((result["object_exploration_s"]["novel"].as_f64().unwrap() - 1.2).abs() < 1e-12);
    let merged = result["bout_debug"].as_array().unwrap().last().unwrap();
    assert_eq!(merged["component_ids"], json!(["a", "b"]));
    assert_eq!(merged["median_angle_degrees"], 0.0);
    assert_eq!(merged["gap_median_policy"], "observed_samples_only");
    assert_eq!(
        nort(&parts, "include_gaps")["object_exploration_s"]["novel"],
        1.6
    );
}

#[test]
fn source_bout_boundaries_and_behavioral_exception_are_exact() {
    for (duration, speed, angle, valid) in [
        (0.5, 0.049999, 45.0, true),
        (0.499999, 0.01, 0.0, false),
        (0.500001, 0.5, 0.0, true),
        (0.5, 0.05, 0.0, true),
        (0.5, 0.01, 45.000001, false),
    ] {
        let mut part = segment("a", "novel", 0.0, duration);
        part["angles_degrees"] = json!([angle]);
        part["nose_speeds_m_per_s"] = json!([speed]);
        assert_eq!(
            nort(&[part], "exclude_gaps")["valid_bouts"],
            if valid { 1 } else { 0 }
        );
    }
    let mut part = segment("g", "novel", 0.0, 1.0);
    part["behaviour"] = json!("grooming");
    assert_eq!(nort(&[part.clone()], "exclude_gaps")["valid_bouts"], 0);
    part["object_directed"] = json!(true);
    assert_eq!(nort(&[part], "exclude_gaps")["valid_bouts"], 1);
    assert_eq!(
        nort(
            &[segment("a", "A", 0.0, 0.6), segment("b", "A", 1.1, 1.7)],
            "include_gaps"
        )["valid_bouts"],
        1
    );
}

#[test]
fn nort_one_second_gaps_merge_inclusively_and_speed_is_optional() {
    for (gap, count) in [(0.999999, 1), (1.0, 1), (1.000001, 2)] {
        let mut first = segment("a", "A", 0.0, 0.5);
        first.as_object_mut().unwrap().remove("nose_speeds_m_per_s");
        let mut second = segment("b", "A", 0.5 + gap, 1.0 + gap);
        second["nose_speeds_m_per_s"] = json!([0.8]);
        let result = nort(&[first, second], "include_gaps");
        assert_eq!(result["valid_bouts"], count);
        assert_eq!(result["removed_speed_bouts"], 0);
        let merged: Vec<_> = result["bout_debug"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|bout| bout["stage"] == "merged")
            .collect();
        assert_eq!(merged.len(), count as usize);
        assert_eq!(merged.last().unwrap()["median_nose_speed_m_per_s"], 0.8);
        assert_eq!(merged.last().unwrap()["speed_filter_mode"], "disabled");
    }
}

#[test]
fn rejected_behavior_on_another_object_is_a_barrier_and_overlap_is_invalid() {
    let mut rejected = segment("grooming", "B", 0.65, 0.95);
    rejected["behaviour"] = json!("grooming");
    let result = nort(
        &[
            segment("a1", "A", 0.0, 0.6),
            rejected.clone(),
            segment("a2", "A", 1.0, 1.6),
        ],
        "include_gaps",
    );
    assert_eq!(result["valid_bouts"], 2);
    assert!(
        result["bout_debug"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|bout| bout["stage"] == "merged")
            .all(|bout| bout["component_ids"].as_array().unwrap().len() == 1)
    );
    rejected["start_s"] = json!(0.5);
    assert!(
        events::nort_bouts(
            &[segment("a", "A", 0.0, 2.0), rejected],
            "exclude_gaps",
            "observed_samples_only"
        )
        .is_err()
    );
}

#[test]
fn nort_arithmetic_matches_all_reference_goldens() {
    let cases: Value = serde_json::from_str(include_str!(
        "../../../../behavioral_post_analysis_harness/tests/golden_cases.json"
    ))
    .unwrap();
    for group in ["nort_test_scores", "nort_training_scores"] {
        for case in cases[group].as_array().unwrap() {
            let result = metrics::nort_scores(
                case["TA"].as_f64(),
                case["TB"].as_f64(),
                case["TN"].as_f64(),
                case["TF"].as_f64(),
            )
            .unwrap();
            for (key, value) in case["expected"].as_object().unwrap() {
                if let Some(value) = value.as_f64() {
                    assert!((result[key].as_f64().unwrap() - value).abs() < 1e-12);
                } else {
                    assert_eq!(&result[key], value);
                }
            }
        }
    }
    assert!(metrics::nort_scores(None, None, Some(f64::NAN), Some(1.0)).is_err());
    assert!(metrics::nort_scores(Some(-1.0), Some(1.0), None, None).is_err());
}

#[test]
fn nort_exact_session_role_assignment_overrides_global_mapping() {
    let mut input = row("nort");
    input["object_configuration"] = json!({"objects":{"left":{},"right":{}}});
    let mut assignment = input.clone();
    assignment["roles"] = json!({"TN":"left","TF":"right"});
    let mapping = json!({"test":{"TN":"right","TF":"left"},"sessions":[assignment]});
    assert_eq!(
        events::nort_object_roles(&input, &mapping)
            .unwrap()
            .unwrap(),
        json!({"TN":"left","TF":"right"})
    );
    assert!(events::nort_object_roles(&input, &json!({"test":{"TN":"left","TF":"left"}})).is_err());
}

#[test]
fn nort_percent_novel_stays_finite_for_large_finite_components() {
    let result = metrics::nort_scores(None, None, Some(1e307), Some(1e307)).unwrap();
    assert_eq!(result["percent_novel"], 50.0);
    assert_eq!(result["LR"], 0.0);
}

#[test]
fn nort_training_aliases_and_imported_totals_obey_the_declared_window() {
    let mut training = row("nort");
    training["phase"] = json!("training");
    training["analysis_duration_s"] = json!(100);
    training["object_A_exploration_s"] = json!(60);
    training["object_B_exploration_s"] = json!(60);
    assert!(metrics::score_records(&[training], &config()).is_err());
    let mut input = row("nort");
    input["TN"] = json!(12);
    input["TF"] = json!(8);
    input["total_test_exploration"] = json!(0);
    assert!(metrics::score_records(&[input], &config()).is_err());
}

#[test]
fn nort_quality_flag_uses_registered_phase_quantity_and_exact_operator() {
    let flag = "passes_15s_exploration_threshold";
    let mut input = row("nort");
    input["TN"] = json!(10);
    input["TF"] = json!(5);
    input[flag] = json!(false);
    let (rows, _) = metrics::score_records(&[input.clone()], &config()).unwrap();
    assert!(rows[0][flag].is_null());
    assert_eq!(rows[0]["upstream_passes_15s_exploration_threshold"], false);
    assert_eq!(rows[0]["missing_reasons"][flag], "blocked:D06");
    for (operator, expected) in [(">=", true), (">", false)] {
        let mut config = config();
        register_choice(
            &mut config,
            "D06",
            "assays.nort.exploration_quality_flag",
            json!({"name":flag,"phase":"test","quantity":"total_test_exploration","operator":operator,"threshold_s":15}),
        );
        let (rows, _) = metrics::score_records(&[input.clone()], &config).unwrap();
        assert_eq!(rows[0][flag], expected);
        assert_eq!(rows[0]["qc_flags"][flag], expected);
        assert_eq!(rows[0]["excluded"], false);
        assert!(rows[0]["missing_reasons"][flag].is_null());
    }
    for (phase, quantity, reason, expected) in [
        (
            "training",
            "total_test_exploration",
            "not_applicable:quality_flag_phase",
            Value::Null,
        ),
        (
            "test",
            "unexported_exploration_s",
            "not_measured:quality_quantity",
            json!(false),
        ),
    ] {
        let mut config = config();
        register_choice(
            &mut config,
            "D06",
            "assays.nort.exploration_quality_flag",
            json!({"name":flag,"phase":phase,"quantity":quantity,"operator":">=","threshold_s":15}),
        );
        let (rows, _) = metrics::score_records(&[input.clone()], &config).unwrap();
        assert_eq!(rows[0][flag], expected);
        assert_eq!(rows[0]["qc_flags"][flag], expected);
        if expected.is_null() {
            assert_eq!(rows[0]["missing_reasons"][flag], reason);
        } else {
            assert_eq!(rows[0]["recomputation_missing_reasons"][flag], reason);
            assert!(rows[0]["missing_reasons"][flag].is_null());
        }
        assert_eq!(rows[0]["upstream_passes_15s_exploration_threshold"], false);
        assert_eq!(rows[0]["excluded"], false);
    }
}

#[test]
fn recipe_physical_side_bias_requires_exact_recording_and_configuration() {
    use bekiper_feature::heuristic::combined::exploration::NortBoutPolicy;
    for policy in [NortBoutPolicy::recipe(), NortBoutPolicy::default()] {
        let mut input = row("nort");
        input["phase"] = json!("training");
        input["recording_id"] = json!("recording-a");
        input["object_configuration"] = json!("configuration-a");
        input["TA"] = json!(3.0);
        input["TB"] = json!(1.0);
        let mut plan = config();
        plan["assays"]["nort"]["bout_policy"] = serde_json::to_value(policy).unwrap();
        let (scored, _) = metrics::score_records(&[input.clone()], &plan).unwrap();
        assert!(scored[0]["side_bias"].is_null());
        assert_eq!(
            scored[0]["qc_flags"]["physical_side_mapping_unavailable"],
            true
        );
        assert_eq!(
            scored[0]["missing_reasons"]["side_bias"],
            "blocked:D10:recording_specific_physical_side_mapping"
        );
        let assignment = json!({"animal_id":input["animal_id"],"assay":"nort","session_id":input["session_id"],
        "timepoint":"post","phase":"training","recording_id":input["recording_id"],"object_configuration":input["object_configuration"],
        "roles":{"TA":"a","TB":"b"},"physical_sides":{"TA":"right","TB":"left"}});
        register_choice(
            &mut plan,
            "D10",
            "assays.nort.object_role_and_location_mapping",
            json!({"sessions":[assignment]}),
        );
        let (scored, _) = metrics::score_records(&[input.clone()], &plan).unwrap();
        assert_eq!(scored[0]["object_bias"], 0.5);
        assert_eq!(scored[0]["side_bias"], 0.5);
        for field in ["recording_id", "object_configuration"] {
            let mut wrong = input.clone();
            wrong[field] = json!("another");
            assert!(metrics::score_records(&[wrong], &plan).is_err());
        }
        input["session_id"] = json!("other-session");
        let (scored, _) = metrics::score_records(&[input.clone()], &plan).unwrap();
        assert!(scored[0]["side_bias"].is_null());
        input["TA"] = json!(0.0);
        input["TB"] = json!(0.0);
        let (scored, _) = metrics::score_records(&[input], &plan).unwrap();
        assert!(scored[0]["object_bias"].is_null() && scored[0]["side_bias"].is_null());
        assert_eq!(scored[0]["total_training_exploration"], 0.0);
    }
}

#[test]
fn nort_registered_reconstruction_preserves_counterbalanced_components_and_window_blocks() {
    let mut input = row("nort");
    input["analysis_start_s"] = json!(4);
    input["analysis_duration_s"] = json!(100);
    input["object_configuration"] = json!({"objects":{"left":{},"right":{}}});
    input["candidate_segments"] = json!([
        segment("left", "left", 4.0, 10.0),
        segment("right", "right", 40.0, 44.0)
    ]);
    input["TN"] = json!(6);
    input["TF"] = json!(4);
    let saved = input.clone();
    let mut config = config();
    register_choice(
        &mut config,
        "D07",
        "assays.nort.merged_gap_duration_policy",
        json!("exclude_gaps"),
    );
    register_choice(
        &mut config,
        "D07",
        "assays.nort.merged_gap_median_policy",
        json!("observed_samples_only"),
    );
    register_choice(
        &mut config,
        "D10",
        "assays.nort.object_role_and_location_mapping",
        json!({"test":{"TN":"right","TF":"left"}}),
    );
    register_choice(
        &mut config,
        "D14",
        "assays.nort.window_and_tracking_policy",
        json!({"interval":"half_open","origin":"analysis_start_s","duration":"analysis_duration_s","out_of_window":"reject","protocol_departures":"flag_only"}),
    );
    let (rows, debug) = metrics::score_records(&[input.clone()], &config).unwrap();
    assert_eq!(rows[0]["TN"], 4.0);
    assert_eq!(rows[0]["TF"], 6.0);
    assert_eq!(rows[0]["upstream_TN"], 6);
    assert_eq!(rows[0]["upstream_TF"], 4);
    assert_eq!(rows[0]["DI"], -0.2);
    assert_eq!(rows[0]["metric_sources"]["TN"], "reconstructed");
    assert_eq!(debug.len(), 4);
    assert!(
        debug
            .iter()
            .all(|event| event["animal_id"] == saved["animal_id"])
    );
    assert_eq!(input, saved);
    let mut zero_total = input.clone();
    let mut gap_input = input.clone();
    let mut gap_config = config.clone();
    register_choice(
        &mut gap_config,
        "D07",
        "assays.nort.merged_gap_duration_policy",
        json!("include_gaps"),
    );
    register_choice(
        &mut gap_config,
        "D07",
        "assays.nort.merged_gap_median_policy",
        json!("include_gaps"),
    );
    let mut first = segment("left1", "left", 4.0, 4.2);
    first["angles_degrees"] = json!([30.0, 30.0]);
    first["nose_contacts"] = json!([false, false]);
    let mut second = segment("left2", "left", 5.0, 5.2);
    second["angles_degrees"] = json!([30.0, 30.0]);
    second["nose_contacts"] = json!([false, false]);
    second["gap_before"] = json!({"start_s":4.2,"end_s":5.0,
        "angles_degrees":vec![90.0; 8],"nose_contacts":[true,false,false,false,false,false,false,false]});
    gap_input["candidate_segments"] = json!([first, second, segment("right", "right", 40.0, 44.0)]);
    let (gap_rows, gap_debug) = metrics::score_records(&[gap_input], &gap_config).unwrap();
    assert_eq!(gap_rows[0]["TN"], 4.0);
    assert!((gap_rows[0]["TF"].as_f64().unwrap() - 1.2).abs() < 1e-12);
    assert_eq!(gap_rows[0]["bouts_rescued_by_contact_override"], 1);
    let merged = gap_debug
        .iter()
        .find(|bout| bout["stage"] == "merged" && bout["object_id"] == "left")
        .unwrap();
    assert_eq!(merged["contact_override_applied"], true);
    assert_eq!(merged["gap_median_policy"], "include_gaps");
    assert_eq!(merged["gap_evidence"][0]["start_s"], 4.2);
    let fields = zero_total.as_object_mut().unwrap();
    fields.remove("TN");
    fields.remove("TF");
    fields.insert("total_test_exploration".into(), json!(0));
    let (rows, _) = metrics::score_records(&[zero_total], &config).unwrap();
    assert_eq!(rows[0]["total_test_exploration"], 10.0);
    assert_eq!(rows[0]["upstream_total_test_exploration"], 0);
    assert_eq!(rows[0]["DI"], -0.2);
    assert!(rows[0]["missing_reasons"]["DI"].is_null());
    input["candidate_segments"][0]["end_s"] = json!(105);
    assert!(metrics::score_records(&[input.clone()], &config).is_err());
    config["decision_resolutions"]
        .as_object_mut()
        .unwrap()
        .remove("D14");
    let (rows, debug) = metrics::score_records(&[input], &config).unwrap();
    assert_eq!(rows[0]["TN"].as_f64(), Some(6.0));
    assert_eq!(rows[0]["metric_sources"]["TN"], "upstream");
    assert!(
        rows[0]["blocked_outputs"]
            .as_array()
            .unwrap()
            .contains(&json!("nort_bout_reconstruction:D14:window_policy"))
    );
    assert!(debug.is_empty());
}
