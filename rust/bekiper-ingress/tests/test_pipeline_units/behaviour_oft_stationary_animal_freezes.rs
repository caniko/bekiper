/// Stationary animal should trigger freezing heuristic in OFT.
#[test]
fn behaviour_oft_stationary_animal_freezes() {
    let n = 600;
    let fps = 30.0;
    let frames: Vec<i64> = (0..n as i64).collect();
    let mut columns: Vec<Column> = vec![Column::new("frame".into(), &frames)];

    // Completely stationary: all keypoints at fixed positions inside arena
    for (li, label) in LABELS.iter().enumerate() {
        let offset = li as f64 * 0.01;
        let xs: Vec<f64> = vec![0.20 + offset; n];
        let ys: Vec<f64> = vec![0.20 + offset; n];
        columns.push(Column::new(format!("{label}_x").into(), &xs));
        columns.push(Column::new(format!("{label}_y").into(), &ys));
    }
    // Add snout direction columns
    let snout_dx: Vec<f64> = vec![-0.03; n];
    let snout_dy: Vec<f64> = vec![-0.03; n];
    columns.push(Column::new("snout_dx".into(), &snout_dx));
    columns.push(Column::new("snout_dy".into(), &snout_dy));

    let df = DataFrame::new(n, columns).unwrap();
    let lf = df.lazy();

    let perimeters = oft_perimeters();
    let params = TaskParams::new(fps);
    let task = bekiper_behaviour::factory::build_task(
        TaskType::OpenField,
        &perimeters,
        params,
        None,
        None,
        None,
    )
    .unwrap();
    let timing = FrameTiming::cfr(fps);
    let (result, _) = task.analyze_detailed(lf, &timing);

    // A stationary animal should produce a nonzero freezing summary
    let freezing = result.summaries.iter().find(|s| s.name.contains("freez"));
    assert!(freezing.is_some(), "OFT should report freezing heuristic");
    if let Some(f) = freezing {
        assert!(
            f.true_frames > 0,
            "stationary animal should have freezing frames, got 0"
        );
    }
}

/// Animal outside arena bounds — behaviour analysis should still complete.
#[test]
fn behaviour_oft_animal_outside_arena() {
    let n = 300;
    let fps = 30.0;
    let frames: Vec<i64> = (0..n as i64).collect();
    let mut columns: Vec<Column> = vec![Column::new("frame".into(), &frames)];

    // All coordinates well outside the 0.40m×0.40m arena centered at (0.20, 0.20)
    for (li, label) in LABELS.iter().enumerate() {
        let offset = li as f64 * 0.01;
        let xs: Vec<f64> = vec![5.0 + offset; n]; // way outside
        let ys: Vec<f64> = vec![5.0 + offset; n];
        columns.push(Column::new(format!("{label}_x").into(), &xs));
        columns.push(Column::new(format!("{label}_y").into(), &ys));
    }
    let snout_dx: Vec<f64> = vec![-0.03; n];
    let snout_dy: Vec<f64> = vec![-0.03; n];
    columns.push(Column::new("snout_dx".into(), &snout_dx));
    columns.push(Column::new("snout_dy".into(), &snout_dy));

    let df = DataFrame::new(n, columns).unwrap();
    let lf = df.lazy();

    let perimeters = oft_perimeters();
    let params = TaskParams::new(fps);
    let task = bekiper_behaviour::factory::build_task(
        TaskType::OpenField,
        &perimeters,
        params,
        None,
        None,
        None,
    )
    .unwrap();
    let timing = FrameTiming::cfr(fps);
    let (result, _) = task.analyze_detailed(lf, &timing);

    // Should still complete without panicking
    assert!(result.evaluated_df.is_some());
    assert_eq!(result.evaluated_df.as_ref().unwrap().height(), n);

    // Animal should NOT be in center zone
    let center_heuristic = result.summaries.iter().find(|s| s.name.contains("center"));
    if let Some(c) = center_heuristic {
        assert_eq!(
            c.true_frames, 0,
            "animal outside arena should have 0 center frames"
        );
    }
}

/// Y-maze without RadialMaze perimeter should error.
#[test]
fn behaviour_ymaze_missing_perimeter_errors() {
    let fps = 30.0;
    let params = TaskParams::new(fps);
    let result =
        bekiper_behaviour::factory::build_task(TaskType::YMaze, &[], params, None, None, None);
    assert!(
        result.is_err(),
        "Y-maze with no RadialMaze perimeter should fail"
    );
}

/// OFT without arena perimeter should error.
#[test]
fn behaviour_oft_missing_arena_errors() {
    let fps = 30.0;
    let params = TaskParams::new(fps);
    let result =
        bekiper_behaviour::factory::build_task(TaskType::OpenField, &[], params, None, None, None);
    assert!(result.is_err(), "OFT with no arena perimeter should fail");
}

/// Very short recording — below minimum bout/dwell thresholds.
#[test]
fn behaviour_nort_very_short_recording() {
    let n = 10; // only ~0.33 seconds at 30 fps (below all min_bout thresholds)
    let fps = 30.0;
    let df = meter_space_poses(n, LABELS);
    let lf = df.lazy();

    let perimeters = nort_perimeters();
    let params = TaskParams::new(fps);
    let task = bekiper_behaviour::factory::build_task(
        TaskType::ObjectRecognition,
        &perimeters,
        params,
        None,
        None,
        None,
    )
    .unwrap();
    let timing = FrameTiming::cfr(fps);
    let (result, _) = task.analyze_detailed(lf, &timing);

    // Should complete without panicking, even if no exploration is detected
    assert!(result.evaluated_df.is_some());
    assert_eq!(result.evaluated_df.as_ref().unwrap().height(), n);
}

/// Perimeter scaling: pixel → meter conversion must scale all fields correctly.
#[test]
fn perimeter_scale_pixel_to_meters() {
    let mpp = 0.002; // 2 mm per pixel

    // Circle
    let circle = PerimeterSpec::Circle {
        label: "obj".into(),
        center_x: 100.0,
        center_y: 200.0,
        radius: 50.0,
    };
    let scaled = circle.scale(mpp);
    match scaled {
        PerimeterSpec::Circle {
            center_x,
            center_y,
            radius,
            ..
        } => {
            assert!((center_x - 0.2).abs() < 1e-10);
            assert!((center_y - 0.4).abs() < 1e-10);
            assert!((radius - 0.1).abs() < 1e-10);
        }
        _ => panic!("expected Circle"),
    }

    // Rectangle
    let rect = PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 320.0,
        center_y: 240.0,
        width: 400.0,
        height: 300.0,
    };
    let scaled = rect.scale(mpp);
    match scaled {
        PerimeterSpec::Rectangle {
            center_x,
            center_y,
            width,
            height,
            ..
        } => {
            assert!((center_x - 0.64).abs() < 1e-10);
            assert!((center_y - 0.48).abs() < 1e-10);
            assert!((width - 0.8).abs() < 1e-10);
            assert!((height - 0.6).abs() < 1e-10);
        }
        _ => panic!("expected Rectangle"),
    }
}

/// Derived scale with non-square arena uses shorter side.
#[test]
fn scale_derivation_nonsquare_arena() {
    let arena_px = vec![PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 320.0,
        center_y: 240.0,
        width: 600.0,  // wider
        height: 400.0, // shorter side
    }];

    let config = ScaleConfig::Derived(bekiper_core::scale::ScaleMethod::ArenaSide {
        meters: 0.40,
        square: false,
    });
    let (mpp, _) = config.resolve_with_info(&arena_px).unwrap();

    // Should use shorter side: 0.40 / 400 = 0.001
    assert!(
        (mpp.0 - 0.001).abs() < 1e-6,
        "non-square arena should use shorter side, got {}",
        mpp.0
    );
}

/// Derived scale with no arena perimeter should error.
#[test]
fn scale_derivation_no_arena_errors() {
    let config = ScaleConfig::Derived(bekiper_core::scale::ScaleMethod::ArenaSide {
        meters: 0.40,
        square: false,
    });
    let result = config.resolve(&[]);
    assert!(
        result.is_err(),
        "derived scale without perimeters should fail"
    );
}

/// Export with no evaluated DataFrame produces only the JSON manifest.
#[test]
fn export_no_evaluated_df() {
    let tmp = test_dir("export_no_df");

    let result = AnalysisResult {
        task_type: TaskType::OpenField,
        summaries: vec![],
        decompositions: vec![],
        evaluated_df: None,
    };

    let manifest = InspectionManifest {
        video: VideoMetadata {
            fps: 30.0,
            total_frames: 0,
            resolution: (0, 0),
            meters_per_pixel: MetersPerPixel(0.001),
        },
        video_path: None,
        session_context: None,
        frame_image: None,
        labels: vec![],
        label_colors: HashMap::new(),
        coordinate_columns: HashMap::new(),
        perimeters: vec![],
        heuristics: vec![],
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

    export_inspection(&tmp, "empty_test", &result, &manifest).unwrap();

    assert!(
        !tmp.join("empty_test_evaluated.parquet").exists(),
        "no parquet without evaluated_df"
    );
    assert!(
        tmp.join("empty_test_inspection.toml").exists(),
        "TOML manifest should always be written"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

/// Config with backend priority list (the imen config uses ["classical_cv", "sam2_auto"]).
#[test]
fn config_backend_priority_list() {
    let toml = r#"
[project]
project_name = "test"
data_directory = "."

[[experiments]]
name = "multi_backend"
task_type = "object_recognition"
fps = 30.0
meters_per_pixel = 0.001

[experiments.detection]
recipe = "nort_training"
backend = ["classical_cv", "sam2_auto"]
"#;

    let tmp = test_dir("config_backend");
    let config_path = tmp.join("config.toml");
    std::fs::write(&config_path, toml).unwrap();

    let config = IngressConfig::from_file(&config_path).unwrap();
    let det = config.experiments[0].detection.as_ref().unwrap();
    let backends = det.backend.backends();
    assert_eq!(backends.len(), 2);
    assert_eq!(
        backends[0],
        bekiper_core::types::DetectionBackend::ClassicalCv
    );
    assert_eq!(backends[1], bekiper_core::types::DetectionBackend::Sam2Auto);

    std::fs::remove_dir_all(&tmp).ok();
}

/// Config with single string backend (not array).
#[test]
fn config_single_backend_string() {
    let toml = r#"
[project]
project_name = "test"
data_directory = "."

[[experiments]]
name = "single_backend"
task_type = "open_field"
fps = 30.0
meters_per_pixel = 0.001

[experiments.detection]
recipe = "oft"
backend = "classical_cv"
"#;

    let tmp = test_dir("config_single_backend");
    let config_path = tmp.join("config.toml");
    std::fs::write(&config_path, toml).unwrap();

    let config = IngressConfig::from_file(&config_path).unwrap();
    let det = config.experiments[0].detection.as_ref().unwrap();
    assert_eq!(det.backend.backends().len(), 1);

    std::fs::remove_dir_all(&tmp).ok();
}

/// Config with task_params overrides (imen has exploration, ymaze, rearing, grooming).
#[test]
fn config_task_params_overrides() {
    let toml = r#"
[project]
project_name = "test"
data_directory = "."

[[experiments]]
name = "with_overrides"
task_type = "object_recognition"
fps = 30.0
meters_per_pixel = 0.001

[experiments.task_params]
freezing_displacement_threshold = 0.003
peripheral_band_fraction = 0.25

[experiments.task_params.exploration]
nose_max_distance = 0.04
max_angle_deg = 45.0
"#;

    let tmp = test_dir("config_overrides");
    let config_path = tmp.join("config.toml");
    std::fs::write(&config_path, toml).unwrap();

    let config = IngressConfig::from_file(&config_path).unwrap();
    let overrides = config.experiments[0].task_params.as_ref().unwrap();

    assert_eq!(overrides.freezing_displacement_threshold, Some(0.003));
    assert_eq!(overrides.peripheral_band_fraction, Some(0.25));

    let expl = overrides.exploration.as_ref().unwrap();
    assert_eq!(expl.nose_max_distance, Some(0.04));
    assert_eq!(expl.max_angle_deg, Some(45.0));
    // Unset fields should be None
    assert!(expl.contact_distance.is_none());

    // Verify overrides apply to TaskParams
    let base = TaskParams::new(30.0);
    let applied = overrides.apply(base);
    assert_eq!(applied.freezing_displacement_threshold, 0.003);
    assert_eq!(applied.peripheral_band_fraction, 0.25);

    std::fs::remove_dir_all(&tmp).ok();
}
