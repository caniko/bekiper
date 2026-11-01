#[test]
fn descriptive_nort_retains_labelled_objects_when_configuration_reverses_order() {
    let mut trial = nort_trial(false);
    trial.animal_metadata["group"] = json!(2);
    trial.session_context["object_a"] = "right".into();
    trial.session_context["object_b"] = "left".into();
    let (rows, _) = descriptive::rows(&[trial]).unwrap();
    assert_eq!(rows[0]["right_exploration_s"], 8.0);
    assert_eq!(rows[0]["left_exploration_s"], 12.0);
    assert_eq!(rows[0]["phase"], "test");
    assert_eq!(rows[0]["object_roles"]["labels"], json!(["right", "left"]));
    assert!(
        rows[0]["scalar_endpoints"]
            .as_array()
            .unwrap()
            .contains(&json!("object_bias"))
    );
}

#[test]
fn native_nort_handoff_uses_actual_window_and_registered_object_identities() {
    for training in [true, false] {
        let phase = if training { "training" } else { "test" };
        let mut meta = metadata("13", phase);
        meta["assay"] = "nort".into();
        meta["object_configuration"] = json!({"objects":{"left":{},"right":{}}});
        let plan = nort_plan(training);
        let rows = attach_native_trials(&[meta], &[nort_trial(training)], &plan).unwrap();
        assert_eq!(rows[0]["analysis_duration_s"], 200.0);
        assert_eq!(rows[0]["analysis_window_s"], 360.0);
        let (left, right) = if training { ("TA", "TB") } else { ("TN", "TF") };
        assert_eq!(rows[0][left], 12.0);
        assert_eq!(rows[0][right], 8.0);
        let (scored, _) = bekiper_post_analysis::metrics::score_records(&rows, &plan).unwrap();
        assert_eq!(scored[0][left], 12.0);
        assert_eq!(scored[0][right], 8.0);
        if !training {
            assert_eq!(scored[0]["DI"], 0.2);
        }
    }
}

#[test]
fn native_nort_arena_metrics_and_scoring_policy_survive_post_analysis() {
    for training in [true, false] {
        let mut meta = metadata("13", if training { "training" } else { "test" });
        meta["assay"] = "nort".into();
        meta["object_configuration"] = json!({"objects":{"left":{},"right":{}}});
        let plan = nort_plan(training);
        let mut trial = nort_trial(training);
        trial.detailed["latency_to_center"] = json!(2.0);
        trial.detailed["zone_metrics"] = json!({"center_summary":{
            "zone_name":"in_center","entries":3,"duration_seconds":12.0,
            "distance_meters":1.5,"mean_speed":0.125,"latency_seconds":2.0},
            "quadrant_summaries":[],"quadrant_motion_summaries":[]});
        let policy = json!({"speed_filter_mode":"disabled","minimum_dwell_seconds":0.5,
            "merge_gap_seconds":1.0,"merge_gap_comparison":"<=","duration_comparison":">="});
        trial.detailed["scoring_provenance"] = policy.clone();
        let rows = attach_native_trials(&[meta], &[trial], &plan).unwrap();
        let (scored, _) = bekiper_post_analysis::metrics::score_records(&rows, &plan).unwrap();
        assert_eq!(scored[0]["center_time_s"], 12.0);
        assert_eq!(scored[0]["center_entries"], 3);
        assert_eq!(scored[0]["center_distance_m"], 1.5);
        assert_eq!(scored[0]["center_mean_speed"], 0.125);
        assert_eq!(scored[0]["latency_to_first_center_entry_s"], 2.0);
        assert_eq!(scored[0]["scoring_provenance"], policy);
    }
}

#[test]
fn native_nort_unregistered_roles_withhold_only_role_dependent_measurements() {
    let mut meta = metadata("13", "training");
    meta["assay"] = "nort".into();
    let mut plan = nort_plan(true);
    plan["decision_resolutions"] = json!({});
    let rows = attach_native_trials(&[meta], &[nort_trial(true)], &plan).unwrap();
    let (scored, _) = bekiper_post_analysis::metrics::score_records(&rows, &plan).unwrap();
    assert!(scored[0]["TA"].is_null() && scored[0]["TB"].is_null());
    assert!(scored[0]["native_unmapped_object_exploration"].is_object());
    assert_eq!(scored[0]["total_distance_m"], 2.0);
}

#[test]
fn native_nort_protocol_must_agree_with_canonical_phase() {
    let mut meta = metadata("13", "test");
    meta["assay"] = "nort".into();
    let mut trial = nort_trial(true);
    trial.session_context["phase"] = "test".into();
    assert!(attach_native_trials(&[meta], &[trial], &nort_plan(false)).is_err());
}

#[test]
fn native_nort_role_join_retains_upstream_index_disagreements_and_window_bounds() {
    let mut meta = metadata("13", "test");
    meta["assay"] = "nort".into();
    meta["object_configuration"] = json!({"objects":{"left":{},"right":{}}});
    let plan = nort_plan(false);
    let mut trial = nort_trial(false);
    trial.detailed["indices"] = serde_json::to_value(
        bekiper_behaviour::nort::NortIndices::compute(8.0, 12.0, 1, 2),
    )
    .unwrap();
    let rows = attach_native_trials(&[meta.clone()], &[trial.clone()], &plan).unwrap();
    let (scored, _) = bekiper_post_analysis::metrics::score_records(&rows, &plan).unwrap();
    assert_eq!(scored[0]["upstream_TN"], 8.0);
    assert_eq!(scored[0]["recomputed_TN"], 12.0);
    assert_eq!(scored[0]["upstream_DI"], -0.2);
    assert_eq!(scored[0]["DI"], 0.2);
    assert!(
        scored[0]["disagreements"]
            .as_array()
            .unwrap()
            .contains(&json!("DI"))
    );
    trial.detailed["exploration_times"] = json!([180.0, 120.0]);
    let rows = attach_native_trials(&[meta], &[trial], &plan).unwrap();
    assert!(bekiper_post_analysis::metrics::score_records(&rows, &plan).is_err());
}
