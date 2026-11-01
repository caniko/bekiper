#[test]
fn impossible_imported_window_and_count_values_are_rejected() {
    let mut oft = row("oft");
    oft["analysis_duration_s"] = json!(300);
    oft["latency_to_first_center_entry_s"] = json!(500);
    let mut nor = row("nort");
    nor["analysis_duration_s"] = json!(360);
    nor["TN"] = json!(400);
    nor["TF"] = json!(400);
    let mut maze = row("ymaze");
    maze["threshold_pct"] = json!(45);
    maze["valid_total_arm_entries"] = json!(8);
    maze["usable_triads"] = json!(100);
    maze["low_quality"] = json!(false);
    for input in [oft, nor, maze] {
        assert!(metrics::score_records(&[input], &config()).is_err());
    }
}

#[test]
fn upstream_qc_flags_survive_and_conflicts_are_explicit() {
    let mut input = row("ymaze");
    input["threshold_pct"] = json!(45);
    input["valid_arm_entry_sequence"] = json!("ABC");
    input["qc_flags"] = json!({"manual_tracking_warning":true,"low_quality":false});
    let (rows, _) = metrics::score_records(&[input], &config()).unwrap();
    assert_eq!(rows[0]["qc_flags"]["manual_tracking_warning"], true);
    assert_eq!(rows[0]["qc_flags"]["low_quality"], true);
    assert_eq!(rows[0]["upstream_qc_flags"]["low_quality"], false);
    assert_eq!(
        rows[0]["qc_flag_conflicts"]["low_quality"]["computed"],
        true
    );
    assert_eq!(rows[0]["excluded"], false);
}

#[test]
fn plausible_imported_disagreements_are_retained_without_overwriting_lineage() {
    let mut input = row("nort");
    input["TN"] = json!(12);
    input["TF"] = json!(8);
    input["DI"] = json!(0.1);
    input["analysis_duration_s"] = json!(360);
    let (rows, _) = metrics::score_records(&[input], &config()).unwrap();
    assert_eq!(rows[0]["upstream_DI"], 0.1);
    assert_eq!(rows[0]["recomputed_DI"], 0.2);
    assert_eq!(rows[0]["DI"], 0.2);
    assert!(
        rows[0]["disagreements"]
            .as_array()
            .unwrap()
            .contains(&json!("DI"))
    );
}

#[test]
fn production_never_uses_synthetic_scientific_approvals() {
    let mut input = row("ymaze");
    input["threshold_pct"] = json!(45);
    input["valid_arm_entry_sequence"] = json!("ABCABCAB");
    input["spontaneous_alternation_pct"] = json!(90);
    let mut config = config();
    config["run_mode"] = json!("production");
    let (rows, _) = metrics::score_records(&[input], &config).unwrap();
    assert!(rows[0]["spontaneous_alternation_pct"].is_null());
    assert_eq!(
        rows[0]["missing_reasons"]["spontaneous_alternation_pct"],
        "blocked:D01"
    );
    assert_eq!(rows[0]["upstream_spontaneous_alternation_pct"], 90);
    assert!(rows[0]["primary_threshold"].is_null());
}

#[test]
fn source_only_measurements_survive_unavailable_recomputation_without_becoming_verified() {
    let mut input = row("nort");
    input["DI"] = json!(0.2);
    input["total_test_exploration"] = json!(20);
    let (rows, _) = metrics::score_records(&[input], &config()).unwrap();
    assert_eq!(rows[0]["DI"], 0.2);
    assert_eq!(rows[0]["metric_sources"]["DI"], "upstream");
    assert_eq!(
        rows[0]["recomputation_missing_reasons"]["DI"],
        "not_measured:component_missing"
    );
    assert_eq!(
        rows[0]["input_verification"]["status"],
        "upstream_unverified"
    );
    let mut zero = row("nort");
    zero["DI"] = json!(0.2);
    zero["total_test_exploration"] = json!(0);
    let (rows, _) = metrics::score_records(&[zero], &config()).unwrap();
    assert!(rows[0]["DI"].is_null());
    assert_eq!(
        rows[0]["missing_reasons"]["DI"],
        "undefined:zero_exploration_total"
    );
    assert_eq!(rows[0]["upstream_DI"], 0.2);
}

#[test]
fn low_quality_boundary_is_exact_and_never_an_exclusion() {
    for (entries, usable, expected) in [(8, 5, false), (7, 5, true), (8, 4, true)] {
        let mut input = row("ymaze");
        input["threshold_pct"] = json!(45);
        input["valid_total_arm_entries"] = json!(entries);
        input["usable_triads"] = json!(usable);
        let (rows, _) = metrics::score_records(&[input], &config()).unwrap();
        assert_eq!(rows[0]["qc_flags"]["low_quality"], expected);
        assert_eq!(rows[0]["excluded"], false);
    }
}

#[test]
fn half_open_window_rejects_end_samples_but_allows_interval_end() {
    let mut input = row("ymaze");
    input["first_detection_s"] = json!(4);
    input["analysis_duration_s"] = json!(360);
    let policy = json!({"interval":"half_open","origin":"first_detection_s","duration":"analysis_duration_s", "out_of_window":"reject",
        "missing_pose":"observed_samples_break_on_missing_no_interpolation"});
    let window = events::reconstruction_window(&input, Some(&policy))
        .unwrap()
        .unwrap();
    assert!(events::validate_window(&[json!({"timestamp_s":364})], &window, false).is_err());
    assert!(events::validate_window(&[json!({"start_s":4,"end_s":364})], &window, true).is_ok());
    assert!(events::validate_window(&[json!({"start_s":3,"end_s":5})], &window, true).is_err());
}

#[test]
fn default_dwell_does_not_verify_historical_zero_dwell_events() {
    let result = events::ymaze_provenance(
        &json!({"scoring_provenance":{"apply_dwell_filter":false,"minimum_dwell_seconds":0}}),
        &json!({}),
    )
    .unwrap();
    assert_eq!(
        result["status"],
        "requires_regeneration_or_registered_amendment"
    );
    assert!(
        result["reasons"]
            .as_array()
            .unwrap()
            .contains(&json!("incompatible_dwell_filter"))
    );
}

#[test]
fn imported_count_identities_cannot_be_hidden_by_recomputation() {
    let cases = [
        json!({"valid_arm_entry_sequence":"ABCABCAB","valid_total_arm_entries":7}),
        json!({"valid_total_arm_entries":8,"valid_arm_A_entries":9}),
        json!({"valid_total_arm_entries":8,"valid_arm_A_entries":3,"valid_arm_B_entries":3,"valid_arm_C_entries":1}),
        json!({"valid_total_arm_entries":8,"total_triads":5}),
        json!({"valid_total_arm_entries":8,"total_triads":6,"correct_alternations":2,"incorrect_alternations":2,"neutral_triads":1}),
        json!({"valid_total_arm_entries":8,"usable_triads":5,"correct_alternations":2,"incorrect_alternations":2}),
        json!({"valid_total_arm_entries":8,"usable_triads":1,"correct_alternations":5}),
        json!({"valid_total_arm_entries":8,"correct_alternations":4,"neutral_triads":4}),
        json!({"valid_total_arm_entries":8,"raw_total_arm_entries":9,"raw_to_scored_entry_difference":2}),
    ];
    for fields in cases {
        let mut input = row("ymaze");
        input["threshold_pct"] = json!(45);
        input
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        assert!(
            metrics::score_records(&[input.clone()], &config()).is_err(),
            "accepted {input}"
        );
    }
}

#[test]
fn zero_total_cannot_restore_an_imported_lr_through_missing_pseudocount_components() {
    let mut input = row("nort");
    input["total_test_exploration"] = json!(0);
    input["LR"] = json!(0.5);
    let mut config = config();
    register_choice(
        &mut config,
        "D13",
        "assays.nort.lr_pseudocount",
        json!({"enabled":true,"value":1,"units":"s","formula":"ln((TN+p)/(TF+p))"}),
    );
    let (rows, _) = metrics::score_records(&[input.clone()], &config).unwrap();
    assert!(rows[0]["LR"].is_null());
    assert_eq!(rows[0]["upstream_LR"], 0.5);
    input["TN"] = json!(0);
    input["TF"] = json!(0);
    let (rows, _) = metrics::score_records(&[input], &config).unwrap();
    assert_eq!(rows[0]["LR"], 0.0);
    assert!(rows[0]["DI"].is_null());
}

#[test]
fn historical_lineage_and_verification_are_retained_when_new_values_differ() {
    let mut input = row("nort");
    input["TN"] = json!(12);
    input["TF"] = json!(8);
    input["DI"] = json!(0.1);
    input["upstream_DI"] = json!(0.05);
    input["recomputed_DI"] = json!(0.1);
    input["input_verification"] = json!({"status":"historical_audit","reference":"previous_run"});
    let (rows, _) = metrics::score_records(&[input], &config()).unwrap();
    assert_eq!(rows[0]["upstream_DI"], 0.1);
    assert_eq!(rows[0]["recomputed_DI"], 0.2);
    assert_eq!(rows[0]["prior_lineage"]["upstream_DI"], 0.05);
    assert_eq!(rows[0]["prior_lineage"]["recomputed_DI"], 0.1);
    assert_eq!(
        rows[0]["upstream_input_verification"]["status"],
        "historical_audit"
    );
}

#[test]
fn coordinate_reconstruction_preserves_both_imported_and_recomputed_event_sets() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../bekiper-behaviour/src/radial_arm/tests/golden/principal_entry_parity.json"
    ))
    .unwrap();
    let mut input = row("ymaze");
    input["threshold_pct"] = json!(45);
    input["geometry"] = json!({"version":fixture["geometry_version"],"canonical_vertices":true,"center":fixture["center"],"arms":fixture["arms"]});
    input["valid_arm_entry_sequence"] = json!("ABC");
    input["raw_arm_entry_sequence"] = json!("ABC");
    input["analysis_duration_s"] = json!(360);
    input["first_detection_s"] = json!(0);
    input["coordinate_sample_end_s"] = json!(0.2);
    input["coordinate_samples"] = json!(
        [
            (0.0, 0.0),
            (0.9, 0.0),
            (0.0, 0.0),
            (-0.9, 0.0),
            (0.0, 0.0),
            (0.0, 0.9)
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (x, y))| json!({"timestamp_s":i as f64 / 30.0,
            "head_center_x":x,"head_center_y":y,"tail_base_x":x,"tail_base_y":y}))
        .collect::<Vec<_>>()
    );
    let saved = input.clone();
    let mut config = config();
    register_choice(
        &mut config,
        "D04",
        "assays.ymaze.window_and_tracking_policy",
        json!({"interval":"half_open","origin":"first_detection_s","duration":"analysis_duration_s","out_of_window":"reject",
            "missing_pose":"observed_samples_break_on_missing_no_interpolation"}),
    );
    for zero_dwell in [false, true] {
        let mut effective = config.clone();
        if zero_dwell {
            effective["assays"]["ymaze"]["temporal_policy"] =
                serde_json::to_value(bekiper_feature::tolerance::TemporalPolicy::default())
                    .unwrap();
        }
        let (rows, debug) = metrics::score_records(&[input.clone()], &effective).unwrap();
        assert_eq!(
            rows[0]["valid_arm_entry_sequence"],
            if zero_dwell {
                json!(["A", "C", "B"])
            } else {
                json!([])
            }
        );
        assert_eq!(rows[0]["raw_arm_entry_sequence"], json!(["A", "C", "B"]));
        assert_eq!(rows[0]["upstream_valid_arm_entry_sequence"], "ABC");
        assert_eq!(rows[0]["upstream_raw_arm_entry_sequence"], "ABC");
        assert_eq!(
            rows[0]["removed_short_entries"],
            if zero_dwell { 0 } else { 3 }
        );
        assert_eq!(
            rows[0]["raw_to_scored_entry_difference"],
            if zero_dwell { 0 } else { 3 }
        );
        assert_eq!(
            rows[0]["dwell_rejected_entries"],
            if zero_dwell { 0 } else { 3 }
        );
        assert_eq!(
            rows[0]["input_verification"]["status"],
            "verified_clarified_entry_definition"
        );
        assert_eq!(debug.len(), 18);
        assert!(debug.iter().all(|event| event["threshold_pct"] == 45));
    }
    assert_eq!(input, saved);
}

#[test]
fn synthetic_audit_can_verify_rule_definitions_but_not_event_sets() {
    let rules = json!({"zone_extension_m":0.03,"body_centroid_outside_object_mask":true,"candidate_orientation_or_contact":true,
        "angle_aggregation":"bout_median","angle_max_degrees":45,"merge_gap_seconds":1,"merge_gap_comparison":"<=",
        "merge_before_duration":true,"minimum_dwell_seconds":0.5,"duration_comparison":">=","speed_filter_mode":"disabled",
        "behaviour_exclusions":["climbing","sitting_on_object","grooming","rearing"],"object_directed_exception":true});
    let mut input = row("nort");
    input["scoring_provenance"] = rules.clone();
    input["scoring_provenance"]["code_revision"] = json!("fixture_revision");
    input["_rule_audit"] = json!({"origin":"development_fixture","rules":rules,"code_revision":"fixture_revision",
        "rationale":"structure test only","approval_or_plan_reference":"synthetic_scoring_audit"});
    input["_rule_evidence_files"] = json!(
        [
            "rule_audit",
            "implementation",
            "geometry",
            "landmarks",
            "window",
            "annotations"
        ]
        .into_iter()
        .map(
            |role| json!({"role":role,"name":"synthetic_fixture","sha256":"a".repeat(64),"bytes":1})
        )
        .collect::<Vec<_>>()
    );
    let result = events::assay_provenance(&input, &config()).unwrap();
    assert_eq!(result["status"], "verified_source_rule_definition");
    assert_eq!(result["level"], "documented_rule_audit_not_event_set_audit");
    let mut gap_input = input.clone();
    let mut gap_plan = config();
    for (path, field, policy) in [
        (
            "assays.nort.merged_gap_duration_policy",
            "gap_duration_policy",
            "include_gaps",
        ),
        (
            "assays.nort.merged_gap_median_policy",
            "gap_median_policy",
            "observed_samples_only",
        ),
    ] {
        register_choice(&mut gap_plan, "D07", path, json!(policy));
        gap_input["scoring_provenance"][field] = json!("include_gaps");
        gap_input["_rule_audit"]["rules"][field] = json!("include_gaps");
    }
    let incompatible = events::assay_provenance(&gap_input, &gap_plan).unwrap();
    assert_eq!(incompatible["status"], "upstream_unverified");
    assert!(
        incompatible["reasons"]
            .as_array()
            .unwrap()
            .contains(&json!("gap_median_policy"))
    );
    register_choice(
        &mut gap_plan,
        "D07",
        "assays.nort.merged_gap_median_policy",
        json!("include_gaps"),
    );
    assert_eq!(
        events::assay_provenance(&gap_input, &gap_plan).unwrap()["status"],
        "verified_source_rule_definition"
    );
    let mut historical = input.clone();
    historical["scoring_provenance"]["minimum_dwell_seconds"] = json!(1.0);
    historical["scoring_provenance"]["merge_gap_seconds"] = json!(0.5);
    historical["scoring_provenance"]["speed_filter_mode"] = json!("per_bout_median");
    assert_eq!(
        events::assay_provenance(&historical, &config()).unwrap()["status"],
        "upstream_unverified"
    );
    input["scoring_provenance"]["body_centroid_outside_object_mask"] = json!(1);
    assert_eq!(
        events::assay_provenance(&input, &config()).unwrap()["status"],
        "upstream_unverified"
    );
    input["scoring_provenance"]["body_centroid_outside_object_mask"] = json!(true);
    let mut production = config();
    production["run_mode"] = json!("production");
    assert_eq!(
        events::assay_provenance(&input, &production).unwrap()["status"],
        "upstream_unverified"
    );
}

#[test]
fn rescoring_keeps_previous_qc_lineage_when_current_flags_differ() {
    let mut input = row("ymaze");
    input["threshold_pct"] = json!(45);
    input["valid_arm_entry_sequence"] = json!("ABC");
    input["qc_flags"] = json!({"low_quality":false,"manual_tracking_warning":true});
    input["upstream_qc_flags"] = json!({"historical_warning":true});
    input["qc_flag_conflicts"] = json!({"historical_conflict":{"upstream":true,"computed":false}});
    let (rows, _) = metrics::score_records(&[input], &config()).unwrap();
    assert_eq!(rows[0]["qc_flags"]["manual_tracking_warning"], true);
    assert_eq!(
        rows[0]["prior_lineage"]["upstream_qc_flags"]["historical_warning"],
        true
    );
    assert_eq!(
        rows[0]["prior_lineage"]["qc_flag_conflicts"]["historical_conflict"]["upstream"],
        true
    );
}
