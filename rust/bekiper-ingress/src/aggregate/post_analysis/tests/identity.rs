#[test]
fn native_handoff_preserves_plan_exclusions_missing_sessions_and_qc() {
    let plan = json!({"inputs":{"session_key_mapping":{"timepoint":{"after":"post"}}}});
    let rows = attach_native_trials(
        &[metadata("13", "session"), metadata("14", "session")],
        &[oft_trial()],
        &plan,
    )
    .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["measurement_available"], true);
    assert_eq!(rows[0]["excluded"], true);
    assert_eq!(rows[0]["exclusion_reason"], "registered reason");
    assert_eq!(rows[0]["qc_flags"]["source_warning"], true);
    assert_eq!(rows[1]["measurement_available"], false);
}

#[test]
fn native_oft_handoff_retains_first_detection_and_clipped_observed_window() {
    let mut trial = oft_trial();
    trial.detailed["total_duration_seconds"] = json!(150.0);
    trial.detailed["window"] = json!({"start_seconds":4.0,"end_seconds":154.0,
        "requested_seconds":300.0,"complete":false,"source_start_frame":120,"source_end_frame":4620,
        "policy":"first_finite_centroid_half_open_v1"});
    let rows = attach_native_trials(
        &[metadata("13", "session")],
        &[trial.clone()],
        &json!({"inputs":{"session_key_mapping":{"timepoint":{"after":"post"}}}}),
    )
    .unwrap();
    assert_eq!(rows[0]["analysis_start_s"], 4.0);
    assert_eq!(rows[0]["first_detection_s"], 4.0);
    assert_eq!(rows[0]["analysis_duration_s"], 150.0);
    assert_eq!(rows[0]["window_complete_flag"], false);
    assert_eq!(rows[0]["native_window_policy"], trial.detailed["window"]);
    trial.detailed["total_duration_seconds"] = json!(0.0);
    let rows = attach_native_trials(
        &[metadata("13", "session")],
        &[trial],
        &json!({"inputs":{"session_key_mapping":{"timepoint":{"after":"post"}}}}),
    )
    .unwrap();
    assert!(rows[0]["first_detection_s"].is_null());
    assert_eq!(rows[0]["analysis_duration_s"], 0.0);
}

#[test]
fn native_handoff_keeps_metadata_qc_and_records_conflicting_trial_flags() {
    let plan = json!({"inputs":{"session_key_mapping":{"timepoint":{"after":"post"}}}});
    let mut meta = metadata("13", "session");
    meta["qc_flags"] = json!({"metadata_warning":true,"source_warning":false});
    let rows = attach_native_trials(&[meta], &[oft_trial()], &plan).unwrap();
    assert_eq!(rows[0]["qc_flags"]["metadata_warning"], true);
    assert_eq!(rows[0]["metadata_qc_flags"]["source_warning"], false);
    assert_eq!(rows[0]["measurement_qc_flags"]["source_warning"], true);
    assert_eq!(
        rows[0]["upstream_qc_join_conflicts"]["source_warning"],
        json!({"metadata":false,"measurement":true})
    );
}

#[test]
fn native_handoff_rejects_inferred_or_duplicate_session_identity() {
    let mut trial = oft_trial();
    trial.session_context["timepoint"] = "post".into();
    let meta = vec![metadata("13", "session")];
    assert!(
        attach_native_trials(
            &meta,
            &[trial.clone(), trial.clone()],
            &json!({"inputs":{}})
        )
        .is_err()
    );
    trial
        .session_context
        .as_object_mut()
        .unwrap()
        .remove("session_id");
    assert!(attach_native_trials(&meta, &[trial], &json!({"inputs":{}})).is_err());
}

#[test]
fn native_session_context_cannot_change_the_trial_task_identity() {
    let mut trial = oft_trial();
    trial.session_context["assay"] = "ymaze".into();
    trial.session_context["timepoint"] = "post".into();
    let mut meta = metadata("13", "session");
    meta["assay"] = "ymaze".into();
    assert!(attach_native_trials(&[meta], &[trial], &json!({"inputs":{}})).is_err());
}

#[test]
fn native_handoff_maps_registered_sessions_without_legacy_biology_overrides() {
    let mut trial = oft_trial();
    trial.session_context = json!({"animal":"13","stage":"0","phase":"B","phase_part":"2",
        "group":"8","genotype":"+/+/wt","sex":"F","treatment":"PBS"});
    trial.animal_metadata = json!({"animal":13,"group":2,"hcar1":"Het/KO","five_xfad":"tg","sex":"F","treatment":"vehicle"});
    let mut meta = metadata("13", "session");
    meta["group"] = "G2".into();
    meta["hcar1"] = "Het/KO".into();
    meta["five_xfad"] = "AD".into();
    meta["treatment"] = "vehicle".into();
    let plan = json!({"inputs":{
        "column_mapping":{"oft":{"experiment":"session_id","stage":"phase","phase":"cohort_phase","phase_part":"timepoint"}},
        "session_key_mapping":{"session_id":{"an_uninformative_name":"registered"},
            "phase":{"0":"session"},"timepoint":{"2":"post"},"group":{"2":"G2"},"five_xfad":{"tg":"AD"}}}});
    let rows = attach_native_trials(&[meta.clone()], &[trial.clone()], &plan).unwrap();
    assert_eq!(rows[0]["session_id"], "registered");
    assert_eq!(rows[0]["timepoint"], "post");
    assert_eq!(rows[0]["group"], "G2");
    assert_eq!(rows[0]["hcar1"], "Het/KO");
    assert_eq!(rows[0]["native_session_context"]["group"], "8");
    trial.animal_metadata["hcar1"] = "WT".into();
    assert!(attach_native_trials(&[meta], &[trial], &plan).is_err());
}

#[test]
fn native_handoff_uses_unique_registered_timepoint_for_explicit_recording_mapping() {
    let mut trial = oft_trial();
    trial.session_context = Value::Null;
    trial.experiment = "source_recording__proximal25".into();
    trial.task_type = TaskType::YMaze;
    trial.detailed = json!({"proximal_threshold":0.25});
    let mut meta = metadata("13", "session");
    meta["assay"] = "ymaze".into();
    let plan = json!({"inputs":{"column_mapping":{"ymaze":{"recording_id":"session_id"}},
        "session_key_mapping":{"session_id":{"source_recording":"registered"}}}});
    let rows = attach_native_trials(&[meta.clone()], &[trial.clone()], &plan).unwrap();
    assert_eq!(rows[0]["timepoint"], "post");
    assert_eq!(rows[0]["phase"], "session");
    let mut other = meta.clone();
    other["timepoint"] = "pre".into();
    assert!(attach_native_trials(&[meta, other], &[trial], &plan).is_err());
}

#[test]
fn native_session_context_cannot_replace_fresh_measurements() {
    let mut trial = oft_trial();
    trial.session_context["timepoint"] = "post".into();
    trial.session_context["total_distance_m"] = 99.0.into();
    assert!(
        attach_native_trials(
            &[metadata("13", "session")],
            &[trial],
            &json!({"inputs":{}})
        )
        .is_err()
    );
}

#[test]
fn native_session_context_cannot_reassign_the_trial_animal() {
    let mut trial = oft_trial();
    trial.session_context["timepoint"] = "post".into();
    trial.session_context["animal_id"] = "14".into();
    assert!(
        attach_native_trials(
            &[metadata("14", "session")],
            &[trial],
            &json!({"inputs":{}})
        )
        .is_err()
    );
}

#[test]
fn visual_export_switches_are_authoritative_over_plan_requests() {
    let mut plan = json!({"outputs":{"requested_outputs":["metrics","figures","report_pdf"]}});
    let mut settings = crate::config::PostAnalysisConfig::default();
    settings.figures.enabled = false;
    settings.report.enabled = false;
    apply_project_outputs(&mut plan, &settings).unwrap();
    assert_eq!(plan["outputs"]["requested_outputs"], json!(["metrics"]));
    settings.figures.enabled = true;
    settings.figures.selection = Some(vec!["figure_oft".into()]);
    settings.figures.formats = vec!["svg".into()];
    settings.report.enabled = true;
    apply_project_outputs(&mut plan, &settings).unwrap();
    assert_eq!(
        plan["outputs"]["requested_outputs"],
        json!(["metrics", "figures", "report"])
    );
    assert_eq!(plan["outputs"]["figure_selection"], json!(["figure_oft"]));
    assert_eq!(plan["outputs"]["figure_formats"], json!(["svg"]));
    assert!(post_analysis_output_path(Path::new("output"), "../escape").is_err());
    assert_eq!(
        post_analysis_output_path(Path::new("output"), "cohort-v1").unwrap(),
        Path::new("output/data/_post_analysis/cohort-v1")
    );
}
