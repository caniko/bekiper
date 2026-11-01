fn attach(
    meta: Record,
    trial: TrialResult,
    plan: Record,
) -> std::result::Result<Vec<Record>, String> {
    attach_native_trials(&[meta], &[trial], &plan).map_err(|error| error.to_string())
}

#[test]
fn native_join_skips_unscored_protocols_and_rejects_malformed_trials() {
    let mut cheeseboard = oft_trial();
    cheeseboard.task_type = TaskType::Cheeseboard;
    let unmeasured =
        |rows: Vec<Record>| rows.iter().all(|row| row["measurement_available"] == false);
    assert!(unmeasured(
        attach(metadata("13", "session"), cheeseboard, json!({})).unwrap()
    ));
    let mut oul = nort_trial(true);
    oul.detailed["protocol"] = "oul_training".into();
    let mut training = metadata("13", "training");
    training["assay"] = "nort".into();
    assert!(unmeasured(attach(training, oul, json!({})).unwrap()));
    let mut nort_meta = metadata("13", "test");
    nort_meta["assay"] = "nort".into();
    for (change, message) in [
        (json!({"exploration_times": [1.0]}), "different lengths"),
        (
            json!({"object_labels": ["a", " "]}),
            "distinct and nonempty",
        ),
    ] {
        let mut trial = nort_trial(false);
        for (key, value) in change.as_object().unwrap() {
            trial.detailed[key] = value.clone();
        }
        let error = attach(nort_meta.clone(), trial, nort_plan(false)).unwrap_err();
        assert!(error.contains(message), "{message}: {error}");
    }
    let plan = |inputs: Record| json!({"inputs": inputs});
    let meta = metadata("13", "session");
    assert!(
        attach(
            meta.clone(),
            oft_trial(),
            plan(json!({"column_mapping": {"oft": "bad"}}))
        )
        .is_err()
    );
    assert!(
        attach(
            meta.clone(),
            oft_trial(),
            plan(json!({"session_key_mapping": {"bogus": {}}}))
        )
        .is_err()
    );
    let mut other_session = meta.clone();
    other_session["session_id"] = "other".into();
    let error = attach(
        other_session,
        oft_trial(),
        plan(json!({"session_key_mapping":{"timepoint":{"after":"post"}}})),
    )
    .unwrap_err();
    assert!(
        error.contains("no matching registered metadata session"),
        "{error}"
    );
    // A trial without a recorded animal is joined by its session context.
    let mut contextual = oft_trial();
    contextual.animal_metadata = Value::Null;
    contextual.session_context["animal"] = "13".into();
    let rows = attach(
        meta.clone(),
        contextual,
        plan(json!({"session_key_mapping":{"timepoint":{"after":"post"}}})),
    )
    .unwrap();
    assert_eq!(rows.len(), 1);
}

#[test]
fn native_ymaze_join_requires_a_threshold_and_names_unlabelled_comparisons() {
    let mut meta = metadata("13", "session");
    meta["assay"] = "ymaze".into();
    let mut trial = oft_trial();
    trial.task_type = TaskType::YMaze;
    trial.session_context["timepoint"] = "post".into();
    trial.detailed = json!({});
    let error = attach(meta.clone(), trial.clone(), json!({"inputs":{}})).unwrap_err();
    assert!(error.contains("needs explicit threshold"), "{error}");
    trial.fps = 25.0;
    let mapped = attach(
        meta.clone(),
        trial.clone(),
        json!({"inputs":{"column_mapping":{"ymaze":{"fps":"threshold_pct"}}}}),
    )
    .unwrap();
    assert_eq!(mapped[0]["threshold_pct"], 25.0);
    trial.experiment = "recording__compare_0pct_dwell0s".into();
    trial.detailed = json!({"proximal_threshold": 0.0});
    let rows = attach(meta, trial, json!({"inputs":{}})).unwrap();
    assert_eq!(
        rows[0]["native_comparisons"][0]["analysis_role"],
        "ymaze__compare_0pct_dwell0s"
    );
}

#[test]
fn native_nort_roles_need_labelled_measurements_or_record_blocked_outputs() {
    let mut row = json!({"phase": "test"});
    map_native_nort_roles(&mut row, &json!({})).unwrap();
    assert_eq!(
        row["blocked_outputs"],
        json!(["nort_object_role_mapping:D10"])
    );
    assert!(row["missing_reasons"]["DI"].is_string() || row["DI"].is_null());
    let mut blocked = json!({"phase": "training", "blocked_outputs": "not a list"});
    assert!(map_native_nort_roles(&mut blocked, &json!({})).is_err());

    let plan = nort_plan(false);
    let mut meta = metadata("13", "test");
    meta["assay"] = "nort".into();
    meta["object_configuration"] = json!({"objects":{"left":{},"right":{}}});
    let mut rows = attach_native_trials(&[meta], &[nort_trial(false)], &plan).unwrap();
    let mut unmeasured = rows[0].clone();
    unmeasured["native_object_measurements"] = Value::Null;
    assert!(
        map_native_nort_roles(&mut unmeasured, &plan)
            .unwrap_err()
            .to_string()
            .contains("needs labelled object measurements")
    );
    rows[0]["native_object_measurements"] = json!([{"object_id": "left", "exploration_s": 1.0}]);
    assert!(
        map_native_nort_roles(&mut rows[0], &plan)
            .unwrap_err()
            .to_string()
            .contains("has no native measurement")
    );
}

#[test]
fn project_outputs_default_to_metrics_and_cells_reject_nonfinite_lists() {
    let mut plan = json!({"outputs": 5});
    let mut settings = crate::config::PostAnalysisConfig::default();
    settings.figures.enabled = false;
    settings.report.enabled = false;
    apply_project_outputs(&mut plan, &settings).unwrap();
    assert!(plan["outputs"]["requested_outputs"].is_array());
    let mut figures_only = json!({"outputs": {"requested_outputs": ["figures"]}});
    apply_project_outputs(&mut figures_only, &settings).unwrap();
    assert_eq!(
        figures_only["outputs"]["requested_outputs"],
        json!(["metrics"])
    );
    assert_eq!(
        cell_value(CellValue::FloatList(vec![1.0])).unwrap(),
        json!([1.0])
    );
    assert!(cell_value(CellValue::FloatList(vec![f64::NAN])).is_err());
}

/// OFT cohort with one registered session and a file plan beside `run/`.
fn plan_cohort(name: &str) -> (PathBuf, IngressConfig) {
    let root =
        std::env::temp_dir().join(format!("bekiper_plan_cohort_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("sessions.json"),
        json!([metadata("13", "session")]).to_string(),
    )
    .unwrap();
    let plan = json!({"run_mode":"synthetic","synthetic_fixture":true,
        "inputs":{"metadata_path":"sessions.json",
            "session_key_mapping":{"timepoint":{"after":"post"}},
            "units_and_calibration":{"oft":{"time":"s","distance":"m","speed":"m/s"}}}});
    std::fs::write(root.join("plan.json"), plan.to_string()).unwrap();
    let mut config: IngressConfig =
        toml::from_str("[project]\nproject_name = 'plan'\ndata_directory = '.'").unwrap();
    config.project.data_directory = root.clone();
    config.post_analysis.enabled = true;
    config.post_analysis.plan_path = Some(root.join("plan.json"));
    config.post_analysis.figures.enabled = false;
    config.post_analysis.report.enabled = false;
    (root, config)
}

#[test]
fn plan_cohort_export_derives_outputs_and_a_run_id_from_a_relative_root() {
    let (root, config) = plan_cohort("relative");
    let mut cheeseboard = oft_trial();
    cheeseboard.task_type = TaskType::Cheeseboard;
    let cwd = std::env::current_dir().unwrap();
    let relative: PathBuf = cwd
        .components()
        .skip(1)
        .map(|_| std::path::Component::ParentDir.as_os_str())
        .collect::<PathBuf>()
        .join(root.join("run").strip_prefix("/").unwrap());
    let manifest = export_post_analysis_trials(&config, &relative, &[oft_trial(), cheeseboard])
        .unwrap()
        .unwrap();
    assert!(
        manifest["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["role"] == "native_trial_result")
    );
    assert!(
        root.join("run/data/_post_analysis")
            .read_dir()
            .unwrap()
            .next()
            .is_some()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn plan_cohort_export_rejects_bad_plans_cancellation_and_missing_metadata() {
    let (root, mut config) = plan_cohort("failures");
    let run_root = root.join("run");
    let trials = [oft_trial()];
    let error = |config: &IngressConfig, cancelled: &(dyn Fn() -> bool + Sync)| {
        export_post_analysis_trials_with_cancel(config, &run_root, &trials, cancelled)
            .unwrap_err()
            .to_string()
    };
    assert!(error(&config, &|| true).contains("cancelled before input validation"));
    let mut missing_plan = config.clone();
    missing_plan.post_analysis.plan_path = Some(root.join("absent.json"));
    assert!(error(&missing_plan, &|| false).contains("post-analysis cohort export failed"));
    let mut frozen = config.clone();
    frozen.post_analysis.plan_path = None;
    frozen.post_analysis.frozen_plan = Some(
        serde_json::from_str(&std::fs::read_to_string(root.join("plan.json")).unwrap()).unwrap(),
    );
    assert!(error(&frozen, &|| false).contains("captured plan requires its sealed fingerprint"));

    use crate::config::MissingMetadataPolicy::{AllowBlanks, Fail};
    let workbook = |policy| {
        Some(crate::config::AggregateMetadataConfig {
            path: "animals.xlsx".into(),
            sheet: "Updated_Mice-overview".into(),
            missing_policy: policy,
        })
    };
    config.aggregate.metadata = workbook(Fail);
    assert!(error(&config, &|| false).contains("workbook is missing"));
    crate::write_test_xlsx(&root.join("animals.xlsx"), &[&["Animal ID"], &["99"]]);
    assert!(error(&config, &|| false).contains("missing required Animal ID row(s): 13"));
    std::fs::remove_file(root.join("animals.xlsx")).unwrap();
    config.aggregate.metadata = workbook(AllowBlanks);
    config.post_analysis.run_id = Some("blank-metadata".into());
    crate::traced(|| export_post_analysis_trials(&config, &run_root, &trials))
        .unwrap()
        .unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
