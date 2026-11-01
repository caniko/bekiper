/// Set up an experiment output directory with all artifacts from earlier stages,
/// then exercise the summary and export logic.
fn setup_experiment_artifacts(
    tmp: &Path,
    name: &str,
    task_type: TaskType,
    perimeters: &[PerimeterSpec],
    fps: f64,
    object_protocol: Option<bekiper_core::types::ObjectRecognitionProtocol>,
) -> (ExperimentOutputDir, AnalysisResult) {
    let exp_dir = ExperimentOutputDir::new(tmp, name);
    exp_dir.ensure_dir().unwrap();

    // Write perimeters.toml
    let peri_path = exp_dir.perimeters_toml();
    std::fs::create_dir_all(peri_path.parent().unwrap()).unwrap();
    let peri_file = PerimetersFile {
        perimeters: perimeters.to_vec(),
    };
    let peri_toml = toml::to_string_pretty(&peri_file).unwrap();
    std::fs::write(&peri_path, peri_toml).unwrap();

    // Write poses.parquet (pixel-space)
    let n = 600;
    let mut poses = synthetic_poses(n, LABELS);
    write_parquet(&mut poses, &exp_dir.poses_parquet()).unwrap();

    // Write augmented.parquet (meter-space)
    let mut augmented = meter_space_poses(n, LABELS);
    write_parquet(&mut augmented, &exp_dir.augmented_parquet()).unwrap();

    // Write scale.toml (intermediate — now lives under checkpoint/)
    exp_dir.ensure_checkpoints_dir().unwrap();
    let scale_info = ScaleDerivationInfo {
        meters_per_pixel_y: None,
        method: "manual".into(),
        physical_meters: None,
        pixel_measurement: None,
        arm_pixel_spans: Vec::new(),
        reference_label: None,
        endpoints: None,
        meters_per_pixel: 0.001,
    };
    std::fs::write(
        exp_dir.scale_toml(),
        toml::to_string_pretty(&scale_info).unwrap(),
    )
    .unwrap();

    // Run behaviour analysis
    let lf = bekiper_pose::io::scan_file(&exp_dir.augmented_parquet()).unwrap();
    let params = TaskParams::new(fps);
    let task = bekiper_behaviour::factory::build_task(
        task_type,
        perimeters,
        params,
        object_protocol,
        None,
        None,
    )
    .unwrap();
    let timing = FrameTiming::cfr(fps);
    let (result, detailed) = task.analyze_detailed(lf, &timing);

    // Write behaviour outputs (intermediate TOMLs → checkpoint/)
    if let Some(ref df) = result.evaluated_df {
        let mut df_clone = df.clone();
        write_parquet(&mut df_clone, &exp_dir.behaviour_parquet()).unwrap();
    }
    let summaries_toml = toml::to_string_pretty(&SummariesFile {
        summaries: result.summaries.clone(),
        decompositions: result.decompositions.clone(),
    })
    .unwrap();
    std::fs::write(exp_dir.behaviour_summaries_toml(), summaries_toml).unwrap();
    if let Some(detailed) = &detailed {
        std::fs::write(
            exp_dir.behaviour_result_toml(),
            json_value_to_toml_string(&detailed.as_json_value()).unwrap(),
        )
        .unwrap();
    }

    (exp_dir, result)
}

#[test]
fn summary_and_export_nort() {
    let tmp = test_dir("summary_nort");
    let name = "nort_t1_video10";
    let perimeters = nort_perimeters();
    let (exp_dir, result) = setup_experiment_artifacts(
        &tmp,
        name,
        TaskType::ObjectRecognition,
        &perimeters,
        30.0,
        None,
    );

    // Build inspection manifest
    let labels: Vec<String> = LABELS.iter().map(|l| l.to_string()).collect();
    let coordinate_columns: HashMap<String, CoordinateColumnSpec> = labels
        .iter()
        .map(|l| {
            (
                l.clone(),
                CoordinateColumnSpec {
                    x: format!("{l}_x"),
                    y: format!("{l}_y"),
                },
            )
        })
        .collect();

    let heuristics: Vec<HeuristicMeta> = result
        .summaries
        .iter()
        .map(|s| HeuristicMeta {
            name: s.name.clone(),
            result_column: s.name.clone(),
            true_frames: s.true_frames,
            total_frames: s.total_frames,
            seconds: s.seconds,
            truth_spans: vec![],
        })
        .collect();

    let manifest = InspectionManifest {
        video: VideoMetadata {
            fps: 30.0,
            total_frames: 600,
            resolution: (0, 0),
            meters_per_pixel: MetersPerPixel(0.001),
        },
        video_path: None,
        session_context: None,
        frame_image: None,
        labels: labels.clone(),
        label_colors: HashMap::new(),
        coordinate_columns,
        perimeters: perimeters.clone(),
        heuristics,
        settings: InspectionSettings {
            invert_y: false,
            minimum_seconds_tolerance: 0.5,
            maximum_seconds_distraction: 1.0 / 3.0,
            meters_per_pixel: 0.001,
        },
        scale_derivation: None,
        quadrants: Vec::new(),
        feature_inspections: Vec::new(),
        pixel_perimeters: Vec::new(),
        pixel_quadrants: Vec::new(),
    };

    // Export inspection artifacts
    export_inspection(exp_dir.root(), name, &result, &manifest).unwrap();

    // Verify outputs exist
    let eval_path = exp_dir.root().join(format!("{name}_evaluated.parquet"));
    let manifest_path = exp_dir.root().join(format!("{name}_inspection.toml"));
    assert!(eval_path.exists(), "evaluated parquet should exist");
    assert!(manifest_path.exists(), "inspection TOML should exist");

    // Roundtrip manifest
    let loaded = InspectionManifest::from_toml(&manifest_path).unwrap();
    assert_eq!(loaded.labels.len(), LABELS.len());
    assert!(!loaded.heuristics.is_empty());
    assert_eq!(loaded.perimeters.len(), 3); // arena + 2 objects

    // Export ODS spreadsheet
    let detailed_path = exp_dir.behaviour_result_toml();
    let detailed = load_typed_detail(&detailed_path, TaskType::ObjectRecognition);
    assert!(matches!(
        detailed.as_ref(),
        Some(TaskDetail::ObjectRecognition(_))
    ));
    export_spreadsheet(&exp_dir, name, &result.summaries, detailed.as_ref(), None).unwrap();

    let ods_path = exp_dir.root().join(format!("{name}.ods"));
    assert!(ods_path.exists(), "OpenDocument spreadsheet should exist");

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn object_recognition_aggregate_uses_serialized_metric_bundles() {
    let tmp = test_dir("aggregate_object_recognition_bundles");
    let name = "721_seq0_oultraining";
    let perimeters = oul_perimeters();
    // OUL training is an explicit protocol even when only two objects are used.
    let (exp_dir, _) = setup_experiment_artifacts(
        &tmp,
        name,
        TaskType::ObjectRecognition,
        &perimeters,
        30.0,
        Some(bekiper_core::types::ObjectRecognitionProtocol::OulTraining),
    );

    let detailed_json =
        toml_str_to_json_value(&std::fs::read_to_string(exp_dir.behaviour_result_toml()).unwrap())
            .unwrap();
    assert_eq!(detailed_json["protocol"], "oul_training");
    let trial = TrialResult {
        experiment: name.into(),
        task_type: TaskType::ObjectRecognition,
        analysis_role: None,
        fps: 30.0,
        meters_per_pixel: 0.001,
        animal_metadata: serde_json::json!({
            "animal": 721,
            "cohort": "cohortA",
            "gene": "wt",
            "sex": "m",
            "age": "10",
        }),
        session_context: serde_json::Value::Null,
        summaries: vec![],
        detailed: detailed_json,
    };

    let sheets = build_sheet_dataframes(vec![trial], &AggregateConfig::default()).unwrap();
    assert_eq!(sheets.len(), 1);

    let (_, df, _) = &sheets[0];
    assert_eq!(df.height(), 1);
    assert!(
        df.column("total_observation_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .is_some()
    );
    assert!(
        df.column("pair_object_1_object_2_delta_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .is_some()
    );
    assert!(
        df.column("center_duration_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .is_some()
    );
    assert!(
        df.column("all_motion_median_acceleration")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .is_some()
    );
    assert!(
        df.column("object_1_body_total_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .is_some()
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn summary_and_export_oft() {
    let tmp = test_dir("summary_oft");
    let name = "oft_video42";
    let perimeters = oft_perimeters();
    let (exp_dir, result) =
        setup_experiment_artifacts(&tmp, name, TaskType::OpenField, &perimeters, 30.0, None);

    // Write summary TOML (mimicking run_summary)
    let summary = serde_json::json!({
        "experiment": name,
        "task_type": "open_field",
        "fps": 30.0,
        "meters_per_pixel": 0.001,
        "summaries": result.summaries,
    });
    std::fs::write(
        exp_dir.result_toml(),
        json_value_to_toml_string(&summary).unwrap(),
    )
    .unwrap();

    assert!(exp_dir.result_toml().exists());
    assert!(exp_dir.behaviour_parquet().exists());

    // Verify summary content
    let loaded: serde_json::Value =
        toml_str_to_json_value(&std::fs::read_to_string(exp_dir.result_toml()).unwrap()).unwrap();
    assert_eq!(loaded["experiment"], "oft_video42");
    assert_eq!(loaded["task_type"], "open_field");

    // Export ODS spreadsheet
    export_spreadsheet(&exp_dir, name, &result.summaries, None, None).unwrap();
    assert!(exp_dir.root().join(format!("{name}.ods")).exists());

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn summary_and_export_ymaze() {
    let tmp = test_dir("summary_ymaze");
    let name = "ymaze_video01";
    let perimeters = ymaze_perimeters();
    let (exp_dir, result) =
        setup_experiment_artifacts(&tmp, name, TaskType::YMaze, &perimeters, 30.0, None);

    // Export inspection artifacts
    let labels: Vec<String> = LABELS.iter().map(|l| l.to_string()).collect();
    let coordinate_columns: HashMap<String, CoordinateColumnSpec> = labels
        .iter()
        .map(|l| {
            (
                l.clone(),
                CoordinateColumnSpec {
                    x: format!("{l}_x"),
                    y: format!("{l}_y"),
                },
            )
        })
        .collect();

    let heuristics: Vec<HeuristicMeta> = result
        .summaries
        .iter()
        .map(|s| HeuristicMeta {
            name: s.name.clone(),
            result_column: s.name.clone(),
            true_frames: s.true_frames,
            total_frames: s.total_frames,
            seconds: s.seconds,
            truth_spans: vec![],
        })
        .collect();

    let manifest = InspectionManifest {
        video: VideoMetadata {
            fps: 30.0,
            total_frames: 600,
            resolution: (0, 0),
            meters_per_pixel: MetersPerPixel(0.001),
        },
        video_path: None,
        session_context: None,
        frame_image: None,
        labels: labels.clone(),
        label_colors: HashMap::new(),
        coordinate_columns,
        perimeters: perimeters.clone(),
        heuristics,
        settings: InspectionSettings {
            invert_y: false,
            minimum_seconds_tolerance: 0.5,
            maximum_seconds_distraction: 1.0 / 3.0,
            meters_per_pixel: 0.001,
        },
        scale_derivation: None,
        quadrants: Vec::new(),
        feature_inspections: Vec::new(),
        pixel_perimeters: Vec::new(),
        pixel_quadrants: Vec::new(),
    };

    export_inspection(exp_dir.root(), name, &result, &manifest).unwrap();

    let manifest_path = exp_dir.root().join(format!("{name}_inspection.toml"));
    assert!(manifest_path.exists());
    let loaded = InspectionManifest::from_toml(&manifest_path).unwrap();
    assert_eq!(loaded.perimeters.len(), 1); // single RadialMaze

    // Export ODS spreadsheet with detailed results
    let detailed_path = exp_dir.behaviour_result_toml();
    let detailed = load_typed_detail(&detailed_path, TaskType::YMaze);
    assert!(matches!(detailed.as_ref(), Some(TaskDetail::YMaze(_))));
    export_spreadsheet(&exp_dir, name, &result.summaries, detailed.as_ref(), None).unwrap();
    assert!(exp_dir.root().join(format!("{name}.ods")).exists());

    std::fs::remove_dir_all(&tmp).ok();
}
