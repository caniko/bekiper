#[test]
fn native_handoff_retains_descriptive_state_payload_without_recalculating_it() {
    let mut trial = oft_trial();
    let payload = json!([{"kind":"state_distribution","label":"recorded","seconds":[1.0,2.0]}]);
    trial.detailed["native_state_distributions"] = payload.clone();
    let rows = attach_native_trials(
        &[metadata("13", "session")],
        &[trial],
        &json!({"inputs":{"session_key_mapping":{"timepoint":{"after":"post"}}}}),
    )
    .unwrap();
    assert_eq!(rows[0]["native_state_distributions"], payload);
}

#[test]
fn descriptive_populations_use_policy_not_individual_geometry_or_events() {
    let mut first = oft_trial();
    first.experiment = "recording_one__compare_0pct_dwell0s".into();
    first.task_type = TaskType::YMaze;
    first.animal_metadata = json!({"animal":13,"group":2,"age":9});
    first.session_context =
        json!({"animal":"13","timepoint":"pre","phase":"session","session_id":"registered"});
    first.detailed["scoring_provenance"] = json!({"minimum_dwell_seconds":0.,"entry_rule":"configured_comparison_v1","arm_polygons":[1],"entry_frames":[1]});
    let mut second = first.clone();
    second.experiment = "recording_two__compare_0pct_dwell0s".into();
    second.detailed["scoring_provenance"]["arm_polygons"] = json!([2]);
    second.detailed["scoring_provenance"]["entry_frames"] = json!([2, 3]);
    let mut archived = second.clone();
    archived.experiment = "recording_two__compare_0pct_dwell1s".into();
    archived.detailed["scoring_provenance"]["minimum_dwell_seconds"] = 1.0.into();
    let (rows, _) = descriptive::rows(&[first, second, archived]).unwrap();
    assert_eq!(
        rows[0]["presentation_partition"],
        rows[1]["presentation_partition"]
    );
    assert_ne!(
        rows[1]["presentation_partition"],
        rows[2]["presentation_partition"]
    );
    assert_eq!(rows[0]["group"], "2");
    assert_eq!(rows[0]["session_id"], "registered");
    assert!(
        !rows[0]["scalar_endpoints"]
            .as_array()
            .unwrap()
            .contains(&json!("age"))
    );
    assert!(
        rows[0]["scalar_endpoints"]
            .as_array()
            .unwrap()
            .contains(&json!("center_latency_s"))
    );
}

#[test]
fn descriptive_metadata_collisions_retain_recorded_groups_and_all_exclusions() {
    let mut trial = oft_trial();
    trial.session_context = json!({"animal":"13","group":"recorded","timepoint":"pre"});
    trial.animal_metadata = json!({"animal":13,"group":"workbook","excluded":false});
    trial.detailed["recorded_trial_metadata"] = json!({"animal":"13","group":"old-metadata","excluded":true,"exclusion_reason":"recorded exclusion"});
    let (rows, ledger) = descriptive::rows(&[trial.clone()]).unwrap();
    assert_eq!(rows[0]["group"], "recorded");
    assert_eq!(rows[0]["excluded"], true);
    assert_eq!(rows[0]["exclusion_reason"], "recorded exclusion");
    assert_eq!(rows[0]["metadata_collisions"][0]["field"], "group");
    assert!(
        ledger
            .iter()
            .any(|entry| entry["reason"] == "recorded_metadata_collision")
    );
    trial.animal_metadata["animal"] = "14".into();
    assert!(descriptive::rows(&[trial]).is_err());
}

#[test]
fn descriptive_rows_omit_unidentified_trials_and_reject_malformed_nort_objects() {
    let mut cheeseboard = oft_trial();
    cheeseboard.task_type = TaskType::Cheeseboard;
    let mut anonymous = oft_trial();
    anonymous.experiment = "anonymous".into();
    anonymous.animal_metadata = Value::Null;
    let mut context_only = oft_trial();
    context_only.experiment = "context_only".into();
    context_only.animal_metadata = Value::Null;
    context_only.session_context =
        json!({"animal": "13", "phase_part": "2", "profile": "videomap"});
    let mut pre = context_only.clone();
    pre.experiment = "pre".into();
    pre.session_context["phase_part"] = "pre".into();
    let mut post = context_only.clone();
    post.experiment = "post".into();
    post.session_context["phase_part"] = "post".into();
    let mut videomap_pre = context_only.clone();
    videomap_pre.experiment = "videomap_pre".into();
    videomap_pre.session_context["phase_part"] = "1".into();
    let mut habituation = nort_trial(true);
    habituation.experiment = "habituation".into();
    habituation.detailed["protocol"] = "oul_training".into();
    habituation.session_context = json!({"animal": "13", "stage": "0"});
    let (rows, omissions) = descriptive::rows(&[
        cheeseboard,
        anonymous,
        context_only,
        pre,
        post,
        videomap_pre,
        habituation,
    ])
    .unwrap();
    let timepoints: Vec<_> = rows.iter().map(|row| row["timepoint"].clone()).collect();
    assert_eq!(
        timepoints,
        [
            json!("post"),
            json!("pre"),
            json!("post"),
            json!("pre"),
            json!("unrecorded")
        ]
    );
    assert_eq!(
        rows[4]["phase"], "0",
        "NORT phase falls back to the recorded stage"
    );
    assert!(
        omissions
            .iter()
            .any(|o| o["reason"] == "recorded_animal_identity_unavailable")
    );

    for (change, message) in [
        (
            json!({"exploration_times": [1.0]}),
            "NORT object array shape mismatch",
        ),
        (
            json!({"object_labels": ["a", "a"]}),
            "duplicate NORT object labels",
        ),
        (
            json!({"exploration_times": [-1.0, 2.0]}),
            "invalid native exploration duration",
        ),
    ] {
        let mut trial = nort_trial(false);
        for (key, value) in change.as_object().unwrap() {
            trial.detailed[key] = value.clone();
        }
        let error = descriptive::rows(&[trial]).unwrap_err().to_string();
        assert!(error.contains(message), "{message}: {error}");
    }
    let duplicate = oft_trial();
    assert!(
        descriptive::rows(&[duplicate.clone(), duplicate])
            .unwrap_err()
            .to_string()
            .contains("duplicate descriptive analysis identity")
    );
}

#[test]
fn descriptive_cohort_export_publishes_recorded_rows_with_workbook_biology() {
    let root =
        std::env::temp_dir().join(format!("bekiper_descriptive_export_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    crate::write_test_xlsx(
        &root.join("animals.xlsx"),
        &[&["Animal ID", "Sex"], &["13", "F"]],
    );
    let mut config: IngressConfig = toml::from_str(
        "[project]\nproject_name = 'descriptive'\ndata_directory = '.'\n[aggregate.metadata]\npath = 'animals.xlsx'\n",
    )
    .unwrap();
    config.project.data_directory = root.clone();
    config.post_analysis.enabled = true;
    config.post_analysis.run_id = Some("descriptive".into());
    config.post_analysis.figures.enabled = false;
    config.post_analysis.report.enabled = false;
    let run_root = root.join("run");
    let mut nort = nort_trial(true);
    nort.experiment = "nort_recording".into();
    let trials = [oft_trial(), nort];
    export_post_analysis_trials(&config, &run_root, &trials)
        .unwrap()
        .unwrap();
    let output = post_analysis_output_path(&run_root, "descriptive").unwrap();
    bekiper_post_analysis::pipeline::verify_artifacts(&output).unwrap();
    // Without metadata or a run id the export still publishes under a fresh id.
    config.aggregate.metadata = None;
    config.post_analysis.run_id = None;
    export_post_analysis_trials(&config, &run_root, &trials[..1])
        .unwrap()
        .unwrap();
    assert!(
        export_post_analysis_trials_with_cancel(&config, &run_root, &trials, &|| true)
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
    let file_root = root.join("file");
    std::fs::write(&file_root, "").unwrap();
    assert!(export_post_analysis_trials(&config, &file_root, &trials).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn ymaze_recording_without_a_metadata_row_keeps_its_named_timepoint() {
    // Imen pre videos 2, 28 and 30 have no annotations/ymaze/metadata.csv row.
    let mut trial = oft_trial();
    trial.task_type = TaskType::YMaze;
    trial.experiment = "y_maze_videos_y_maze_pre_pre_video_2__proximal45".into();
    trial.animal_metadata = json!({"animal":2});
    trial.session_context = json!(null);
    let mut unnamed = trial.clone();
    unnamed.experiment = "y_maze_videos_recording_3__proximal45".into();
    let (rows, _) = descriptive::rows(&[trial, unnamed]).unwrap();
    assert_eq!(rows[0]["timepoint"], "pre");
    assert_eq!(rows[1]["timepoint"], "unrecorded");
}
