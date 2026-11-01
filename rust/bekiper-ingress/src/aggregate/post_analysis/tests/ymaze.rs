#[test]
fn native_cohort_applies_zero_dwell_override_and_preserves_original_plan_identity() {
    use bekiper_behaviour::radial_arm::YMaze;
    use bekiper_feature::tolerance::DwellOverrides;
    use bekiper_perimeter::{perimeter::Perimeter, polygon::Polygon, radial_maze::RadialMaze};
    use polars::prelude::*;

    let root =
        std::env::temp_dir().join(format!("bekiper-native-zero-dwell-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let maze = Perimeter::new(
        "maze",
        RadialMaze::new(
            Polygon::new(vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]),
            vec![
                Polygon::new(vec![(0.1, -0.2), (1.1, -0.2), (1.1, 0.2), (0.1, 0.2)]),
                Polygon::new(vec![(-0.2, 0.1), (0.2, 0.1), (0.2, 1.1), (-0.2, 1.1)]),
                Polygon::new(vec![(-1.1, -0.2), (-0.1, -0.2), (-0.1, 0.2), (-1.1, 0.2)]),
            ],
        ),
    );
    let mut task = YMaze::new(maze, 2.0);
    task.temporal_policy = Some(Default::default());
    let x = [0.0, 0.9, 0.0, 0.9, 0.9, 0.0];
    let native = task.full_analyze(
        df! {
            "center_ear_x" => x, "center_ear_y" => [0.0;6],
            "tail_base_x" => x, "tail_base_y" => [0.0;6],
            "mouse_center_x" => x, "mouse_center_y" => [0.0;6]
        }
        .unwrap()
        .lazy(),
    );
    assert_eq!(native.analysis.arm_sequence, ["A", "A"]);
    let mut trial = oft_trial();
    trial.task_type = TaskType::YMaze;
    trial.session_context["timepoint"] = "post".into();
    trial.detailed = serde_json::to_value(native).unwrap();
    let mut meta = metadata("13", "session");
    meta["assay"] = "ymaze".into();
    std::fs::write(root.join("sessions.json"), json!([meta]).to_string()).unwrap();
    let plan_path = root.join("plan.json");
    let plan = json!({"run_mode":"synthetic","synthetic_fixture":true,
        "inputs":{"metadata_path":"sessions.json",
            "units_and_calibration":{"ymaze":{"time":"s","distance":"m","speed":"m/s"}},
            "upstream_rule_provenance":{"ymaze":{"landmark_roles":{
                "head_center":{"x":"center_ear_x","y":"center_ear_y"},
                "tail_base":{"x":"tail_base_x","y":"tail_base_y"}}}}},
        "assays":{"ymaze":{"temporal_policy":{"min_duration_seconds":1.0},"principal_minimum_dwell_s":1.0}},
        "outputs":{"requested_outputs":["metrics"]}});
    std::fs::write(&plan_path, plan.to_string()).unwrap();
    let original = inputs::fingerprint(&plan_path).unwrap();
    let mut config: IngressConfig =
        toml::from_str("[project]\nproject_name = 'cohort-dwell'\ndata_directory = '.'").unwrap();
    config.post_analysis.enabled = true;
    config.post_analysis.plan_path = Some(plan_path.clone());
    config.post_analysis.run_id = Some("zero-dwell".into());
    config.post_analysis.figures.enabled = false;
    config.post_analysis.report.enabled = false;
    config
        .apply_ymaze_dwell_overrides(DwellOverrides {
            seconds: Some(0.0),
            ..Default::default()
        })
        .unwrap();
    let run_root = root.join("run");
    let manifest = export_post_analysis_trials(&config, &run_root, &[trial])
        .unwrap()
        .unwrap();
    let output = post_analysis_output_path(&run_root, "zero-dwell").unwrap();
    let metrics = inputs::load_json(&bekiper_post_analysis::export::path(
        &output,
        "metrics.json",
    ))
    .unwrap();
    assert_eq!(metrics[0]["valid_total_arm_entries"], 2);
    assert_eq!(metrics[0]["dwell_rejected_entries"], 0);
    assert_eq!(
        metrics[0]["input_verification"]["status"],
        "verified_clarified_entry_definition"
    );
    assert_eq!(
        manifest["resolved_config"]["assays"]["ymaze"]["temporal_policy"]["min_duration_seconds"],
        0.0
    );
    let overrides = &manifest["resolved_config"]["execution_overrides"]["ymaze_dwell"];
    assert_eq!(overrides.as_array().unwrap().len(), 1);
    assert_eq!(
        overrides[0]["configured_policy"]["min_duration_seconds"],
        1.0
    );
    assert_eq!(
        overrides[0]["effective_policy"]["min_duration_seconds"],
        0.0
    );
    assert!(
        manifest["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["role"] == "post_analysis_plan"
                && entry["sha256"] == original["sha256"])
    );
    assert_eq!(inputs::fingerprint(&plan_path).unwrap(), original);
    bekiper_post_analysis::pipeline::verify_artifacts(&output).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_ymaze_handoff_is_threshold_scoped_and_rejects_conflicting_thresholds() {
    let mut meta = metadata("13", "session");
    meta["assay"] = "ymaze".into();
    let trials: Vec<_> = [25, 45, 65]
        .into_iter()
        .map(|threshold| {
            let mut trial = oft_trial();
            trial.task_type = TaskType::YMaze;
            trial.session_context["timepoint"] = "post".into();
            trial.session_context["threshold_pct"] = threshold.into();
            trial.detailed = json!({"proximal_threshold":f64::from(threshold)/100.0,
            "analysis_duration_seconds":480.0,"analysis":{"arm_sequence":["A","B","C"]}});
            trial
        })
        .collect();
    let rows = attach_native_trials(&[meta.clone()], &trials, &json!({"inputs":{}})).unwrap();
    assert_eq!(rows.len(), 3);
    assert!(
        rows.iter()
            .all(|row| row["valid_arm_entry_sequence"] == json!(["A", "B", "C"]))
    );
    let mut conflicting = trials[0].clone();
    conflicting.session_context["threshold_pct"] = 65.into();
    assert!(attach_native_trials(&[meta], &[conflicting], &json!({"inputs":{}})).is_err());
}

#[test]
fn native_ymaze_comparison_accepts_emitted_and_historical_roles_without_relabeling() {
    let mut meta = metadata("13", "session");
    meta["assay"] = "ymaze".into();
    for prefix in ["y_maze", "ymaze"] {
        for variant in ["compare_0pct_dwell0s", "compare_0pct_dwell1s"] {
            let mut trial = oft_trial();
            trial.task_type = TaskType::YMaze;
            // Explicit roles must work even with descriptive experiment names.
            trial.experiment = "descriptive_recording".into();
            let role = format!("{prefix}__{variant}");
            trial.analysis_role = Some(role.clone());
            trial.session_context["timepoint"] = "post".into();
            trial.detailed = json!({"proximal_threshold":0.0});
            let sheet = experiment_for_trial(&trial).sheet_key(&trial).unwrap();
            assert_eq!(sheet.variant.unwrap().0, variant);
            let rows =
                attach_native_trials(&[meta.clone()], &[trial.clone()], &json!({"inputs":{}}))
                    .unwrap();
            assert_eq!(rows.len(), 1);
            assert!(!rows[0]["measurement_available"].as_bool().unwrap());
            assert_eq!(rows[0]["native_comparisons"][0]["analysis_role"], role);
            assert_eq!(
                rows[0]["native_comparisons"][0]["routing"],
                "native_comparison_only"
            );
            trial.analysis_role = Some(format!("{prefix}__proximal25"));
            // An explicit principal role cannot be rescued by a comparison filename.
            trial.experiment = format!("recording__{variant}");
            assert!(
                attach_native_trials(&[meta.clone()], &[trial], &json!({"inputs":{}})).is_err()
            );
        }
    }
}

#[test]
fn native_ymaze_comparison_is_retained_outside_principal_scoring() {
    let mut meta = metadata("13", "session");
    meta["assay"] = "ymaze".into();
    let mut trials: Vec<_> = [25, 45, 65, 0].into_iter().map(|threshold| {
        let mut trial = oft_trial();
        trial.task_type = TaskType::YMaze;
        trial.experiment = format!("ymaze_session__{}", if threshold == 0 { "compare_0pct_dwell1s".into() } else { format!("proximal{threshold}") });
        trial.analysis_role = Some(if threshold == 0 { "ymaze__compare_0pct_dwell1s".into() } else { format!("ymaze__proximal{threshold}") });
        trial.session_context["timepoint"] = "post".into();
        trial.detailed = json!({"proximal_threshold":f64::from(threshold)/100.0,"analysis_duration_seconds":480.0,
            "analysis":{"arm_sequence":["A","B","C"]},"scoring_provenance":{"comparison_policy":"retained"}});
        trial
    }).collect();
    let rows = attach_native_trials(&[meta.clone()], &trials, &json!({"inputs":{}})).unwrap();
    assert_eq!(rows.len(), 3);
    assert!(
        rows.iter()
            .all(|row| row["threshold_pct"].as_f64().unwrap() > 0.0)
    );
    let comparisons: Vec<_> = rows
        .iter()
        .flat_map(|row| row["native_comparisons"].as_array().into_iter().flatten())
        .collect();
    assert_eq!(comparisons.len(), 1);
    assert_eq!(comparisons[0]["threshold_pct"], 0.0);
    assert_eq!(
        comparisons[0]["native_result"]["detailed"]["scoring_provenance"]["comparison_policy"],
        "retained"
    );
    let output = std::env::temp_dir().join(format!(
        "bekiper_native_comparison_export_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&output);
    let plan = json!({"run_mode":"synthetic","synthetic_fixture":true,
        "inputs":{"units_and_calibration":{"ymaze":{"time":"s","distance":"m","speed":"m/s"}}},
        "outputs":{"requested_outputs":["metrics"]}});
    let manifest = bekiper_post_analysis::pipeline::run_records(
        plan,
        rows,
        vec![],
        &output.with_extension("json"),
        &RunOptions {
            output: Some(output.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(manifest["native_comparisons"]["count"], 1);
    assert_eq!(manifest["native_comparisons"]["principal_inference"], false);
    let exported =
        inputs::load_json(&output.join(manifest["native_comparisons"]["path"].as_str().unwrap()))
            .unwrap();
    assert_eq!(
        exported[0]["native_result"]["detailed"]["analysis"]["arm_sequence"],
        json!(["A", "B", "C"])
    );
    let metrics = inputs::load_json(&bekiper_post_analysis::export::path(
        &output,
        "metrics.json",
    ))
    .unwrap();
    assert_eq!(metrics.as_array().unwrap().len(), 3);
    assert!(
        metrics
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["threshold_pct"].as_f64().unwrap() > 0.0)
    );
    bekiper_post_analysis::pipeline::verify_artifacts(&output).unwrap();
    std::fs::remove_dir_all(output).unwrap();
    trials.push(trials[3].clone());
    assert!(attach_native_trials(&[meta.clone()], &trials, &json!({"inputs":{}})).is_err());
    trials.truncate(4);
    trials[3].analysis_role = None;
    trials[3].experiment = "unmarked_zero_threshold".into();
    assert!(attach_native_trials(&[meta], &trials, &json!({"inputs":{}})).is_err());
}

#[test]
fn native_ymaze_handoff_preserves_difference_separately_from_dwell_rejections() {
    let mut meta = metadata("13", "session");
    meta["assay"] = "ymaze".into();
    let mut trial = oft_trial();
    trial.task_type = TaskType::YMaze;
    trial.session_context["timepoint"] = "post".into();
    trial.detailed = json!({"proximal_threshold":0.45,
        "analysis":{"raw_arm_sequence":["A","B","C","A"],"arm_sequence":["A","B","C"]},
        "scoring_provenance":{"dwell_bouts":{"in_arm_A":[{"accepted":true}],
            "in_arm_B":[{"accepted":true}],"in_arm_C":[{"accepted":true}],"in_center":[{"accepted":false,"rejection_reason":"minimum_dwell"}]}}});
    let rows = attach_native_trials(&[meta], &[trial], &json!({"inputs":{}})).unwrap();
    assert_eq!(rows[0]["removed_short_entries"], 1);
    assert_eq!(rows[0]["raw_to_scored_entry_difference"], 1);
    assert_eq!(rows[0]["dwell_rejected_entries"], 0);
}
