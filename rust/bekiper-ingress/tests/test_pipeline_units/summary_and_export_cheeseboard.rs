#[test]
fn summary_and_export_cheeseboard() {
    let tmp = test_dir("summary_cheeseboard");
    let name = "cheeseboard_video01";
    let perimeters = cheeseboard_perimeters();
    let (exp_dir, result) =
        setup_experiment_artifacts(&tmp, name, TaskType::Cheeseboard, &perimeters, 30.0, None);

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

    let detailed_path = exp_dir.behaviour_result_toml();
    let detailed = load_typed_detail(&detailed_path, TaskType::Cheeseboard);
    match detailed.as_ref() {
        Some(TaskDetail::Cheeseboard(detail)) => {
            assert_eq!(detail.enclosure_label, "board");
            assert_eq!(
                detail.reward_labels,
                vec!["reward_a".to_string(), "reward_b".to_string()]
            );
        }
        other => panic!("expected Cheeseboard detail, got {other:?}"),
    }

    export_spreadsheet(&exp_dir, name, &result.summaries, detailed.as_ref(), None).unwrap();

    assert!(
        exp_dir
            .root()
            .join(format!("{name}_evaluated.parquet"))
            .exists()
    );
    assert!(
        exp_dir
            .root()
            .join(format!("{name}_inspection.toml"))
            .exists()
    );
    assert!(exp_dir.root().join(format!("{name}.ods")).exists());

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn cheeseboard_typed_detail_roundtrip_preserves_explicit_and_missing_start_fields() {
    let tmp = test_dir("cheeseboard_detail_roundtrip");
    let exp_dir = ExperimentOutputDir::new(&tmp, "cheeseboard_roundtrip");
    exp_dir.ensure_dir().unwrap();
    exp_dir.ensure_checkpoints_dir().unwrap();

    let detail = cheeseboard_detail(Some(12), Some(0.4), Some(27), Some(0.9), Some(0.5));
    let detail_path = exp_dir.behaviour_result_toml();
    std::fs::write(
        &detail_path,
        json_value_to_toml_string(&detail.as_json_value()).unwrap(),
    )
    .unwrap();

    let loaded = load_typed_detail(&detail_path, TaskType::Cheeseboard);
    match loaded.as_ref() {
        Some(TaskDetail::Cheeseboard(detail)) => {
            assert_eq!(detail.start_area_exit_frame, Some(12));
            assert_eq!(detail.start_area_exit_seconds, Some(0.4));
            assert_eq!(detail.reward_arrival_frame, Some(27));
            assert_eq!(detail.reward_arrival_seconds, Some(0.9));
            assert_eq!(detail.seconds_to_find_reward, Some(0.5));
        }
        other => panic!("expected explicit Cheeseboard detail, got {other:?}"),
    }

    export_spreadsheet(
        &exp_dir,
        "cheeseboard_roundtrip",
        &[],
        loaded.as_ref(),
        None,
    )
    .unwrap();
    assert!(exp_dir.root().join("cheeseboard_roundtrip.ods").exists());

    let missing_start = cheeseboard_detail(None, None, Some(8), Some(0.2), None);
    std::fs::write(
        &detail_path,
        json_value_to_toml_string(&missing_start.as_json_value()).unwrap(),
    )
    .unwrap();

    let loaded_missing = load_typed_detail(&detail_path, TaskType::Cheeseboard);
    match loaded_missing.as_ref() {
        Some(TaskDetail::Cheeseboard(detail)) => {
            assert_eq!(detail.start_area_exit_frame, None);
            assert_eq!(detail.start_area_exit_seconds, None);
            assert_eq!(detail.reward_arrival_frame, Some(8));
            assert_eq!(detail.reward_arrival_seconds, Some(0.2));
            assert_eq!(detail.seconds_to_find_reward, None);
        }
        other => panic!("expected missing-start Cheeseboard detail, got {other:?}"),
    }

    std::fs::remove_dir_all(&tmp).ok();
}

// =========================================================================
// 5. Stage output directory — freshness and layout
// =========================================================================

#[test]
fn stage_output_dir_layout() {
    let tmp = test_dir("stage_layout");
    let exp_dir = ExperimentOutputDir::new(&tmp, "test_experiment");

    assert_eq!(
        exp_dir.pixel_perimeters_toml(),
        tmp.join("checkpoint/test_experiment/pixel_perimeters.toml")
    );
    assert_eq!(
        exp_dir.perimeters_toml(),
        tmp.join("checkpoint/test_experiment/perimeters.toml")
    );
    assert_eq!(
        exp_dir.poses_parquet(),
        tmp.join("checkpoint/test_experiment/poses.parquet")
    );
    assert_eq!(
        exp_dir.augmented_parquet(),
        tmp.join("data/test_experiment/augmented.parquet")
    );
    assert_eq!(
        exp_dir.behaviour_parquet(),
        tmp.join("data/test_experiment/behaviour.parquet")
    );
    assert_eq!(
        exp_dir.result_toml(),
        tmp.join("data/test_experiment/result.toml")
    );
    assert_eq!(
        exp_dir.scale_toml(),
        tmp.join("checkpoint/test_experiment/scale.toml")
    );
    assert_eq!(
        exp_dir.behaviour_summaries_toml(),
        tmp.join("checkpoint/test_experiment/behaviour_summaries.toml")
    );
    assert_eq!(
        exp_dir.behaviour_result_toml(),
        tmp.join("checkpoint/test_experiment/behaviour_result.toml")
    );
    assert_eq!(
        exp_dir.figures_dir(),
        tmp.join("data/test_experiment/figures")
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn stage_freshness_detects_missing_outputs() {
    let tmp = test_dir("stage_freshness");
    let exp_dir = ExperimentOutputDir::new(&tmp, "fresh_test");

    // All stages should be stale when nothing exists
    for stage in Stage::all() {
        assert!(
            !exp_dir.is_fresh(*stage, None),
            "{stage} should not be fresh when output doesn't exist"
        );
    }

    std::fs::remove_dir_all(&tmp).ok();
}

// =========================================================================
// 6. Perimeter serialization roundtrip (all types used by imen)
// =========================================================================

#[test]
fn perimeter_serialization_roundtrip() {
    let all_perimeters: Vec<Vec<PerimeterSpec>> =
        vec![nort_perimeters(), oft_perimeters(), ymaze_perimeters()];

    for perimeters in all_perimeters {
        let peri_file = PerimetersFile {
            perimeters: perimeters.clone(),
        };
        let toml_str = toml::to_string_pretty(&peri_file).unwrap();
        let roundtripped_file: PerimetersFile = toml::from_str(&toml_str).unwrap();
        let roundtripped = roundtripped_file.perimeters;
        assert_eq!(
            perimeters.len(),
            roundtripped.len(),
            "perimeter count should survive roundtrip"
        );
        // Verify labels survive
        let orig_labels: Vec<String> = perimeters
            .iter()
            .map(|p| match p {
                PerimeterSpec::Circle { label, .. } => label.clone(),
                PerimeterSpec::Rectangle { label, .. } => label.clone(),
                PerimeterSpec::Polygon { label, .. } => label.clone(),
                PerimeterSpec::RadialMaze { label, .. } => label.clone(),
                PerimeterSpec::Triangle { label, .. } => label.clone(),
            })
            .collect();
        let rt_labels: Vec<String> = roundtripped
            .iter()
            .map(|p| match p {
                PerimeterSpec::Circle { label, .. } => label.clone(),
                PerimeterSpec::Rectangle { label, .. } => label.clone(),
                PerimeterSpec::Polygon { label, .. } => label.clone(),
                PerimeterSpec::RadialMaze { label, .. } => label.clone(),
                PerimeterSpec::Triangle { label, .. } => label.clone(),
            })
            .collect();
        assert_eq!(orig_labels, rt_labels);
    }
}

// =========================================================================
// 7. Scale derivation — arena_side and manual (used by imen)
// =========================================================================

#[test]
fn scale_derivation_arena_side() {
    let arena_px = vec![PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 320.0,
        center_y: 240.0,
        width: 500.0,
        height: 500.0,
    }];

    let config = ScaleConfig::Derived(bekiper_core::scale::ScaleMethod::ArenaSide {
        meters: 0.40,
        square: false,
    });
    let (mpp, info) = config.resolve_with_info(&arena_px).unwrap();

    // 0.40 m / 500 px = 0.0008 m/px
    assert!(
        (mpp.0 - 0.0008).abs() < 1e-6,
        "expected 0.0008, got {}",
        mpp.0
    );
    assert!(info.method.contains("arena"));
}

#[test]
fn scale_derivation_manual() {
    let config = ScaleConfig::Manual(0.001914);
    let (mpp, info) = config.resolve_with_info(&[]).unwrap();
    assert!((mpp.0 - 0.001914).abs() < 1e-10);
    assert_eq!(info.method, "manual");
}

// =========================================================================
// 8. Edge cases
// =========================================================================

/// Augmentation with fewer frames than smoothing window (window=5, data=3).
#[test]
fn augment_fewer_frames_than_smoothing_window() {
    let n = 3;
    let mut df = synthetic_poses(n, LABELS);
    let tmp = test_dir("augment_short");

    let poses_path = tmp.join("poses.parquet");
    write_parquet(&mut df, &poses_path).unwrap();
    let lf = bekiper_pose::io::scan_file(&poses_path).unwrap();

    let pipeline = build_augmentation_pipeline(LABELS, &default_pipeline_config(), 0.001);
    let result = pipeline.build_and_collect(lf).unwrap();

    // Should not panic or lose rows
    assert_eq!(result.height(), n);

    std::fs::remove_dir_all(&tmp).ok();
}

/// Augmentation with every single keypoint below likelihood threshold.
#[test]
fn augment_all_keypoints_below_threshold() {
    let n = 50;
    let frames: Vec<i64> = (0..n as i64).collect();
    let mut columns: Vec<Column> = vec![Column::new("frame".into(), &frames)];
    for label in LABELS {
        let xs: Vec<f64> = (0..n).map(|i| 200.0 + i as f64).collect();
        let ys: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let liks: Vec<f64> = vec![0.05; n]; // all below 0.9 threshold
        columns.push(Column::new(format!("{label}_x").into(), &xs));
        columns.push(Column::new(format!("{label}_y").into(), &ys));
        columns.push(Column::new(format!("{label}_likelihood").into(), &liks));
    }
    let mut df = DataFrame::new(n, columns).unwrap();

    let tmp = test_dir("augment_all_low");
    let poses_path = tmp.join("poses.parquet");
    write_parquet(&mut df, &poses_path).unwrap();
    let lf = bekiper_pose::io::scan_file(&poses_path).unwrap();

    // Apply only the likelihood filter to verify all coords become null
    let (x_cols, y_cols, lik_cols) = label_columns(LABELS);
    let coord_cols: Vec<String> = x_cols.iter().chain(y_cols.iter()).cloned().collect();
    let pipeline =
        AugmentedBuilder::new().add_step(LikelihoodFilter::new(0.9, lik_cols, coord_cols));
    let result = pipeline.build_and_collect(lf).unwrap();

    // Every coordinate column should be entirely null
    let nose_x = result.column("nose_x").unwrap();
    assert_eq!(
        nose_x.null_count(),
        n,
        "all keypoints below threshold should yield all-null column"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

/// Consecutive velocity spikes should all be filtered.
#[test]
fn augment_consecutive_velocity_spikes() {
    let n = 100;
    let frames: Vec<i64> = (0..n as i64).collect();
    let mut columns: Vec<Column> = vec![Column::new("frame".into(), &frames)];
    for (li, label) in LABELS.iter().enumerate() {
        let offset = li as f64 * 20.0;
        let xs: Vec<f64> = (0..n)
            .map(|i| {
                if (40..=45).contains(&i) {
                    9999.0 // 6 consecutive spike frames
                } else {
                    200.0 + offset + (i as f64) * 0.5
                }
            })
            .collect();
        let ys: Vec<f64> = (0..n).map(|i| 150.0 + offset + (i as f64) * 0.3).collect();
        let liks: Vec<f64> = vec![0.95; n];
        columns.push(Column::new(format!("{label}_x").into(), &xs));
        columns.push(Column::new(format!("{label}_y").into(), &ys));
        columns.push(Column::new(format!("{label}_likelihood").into(), &liks));
    }
    let mut df = DataFrame::new(n, columns).unwrap();

    let tmp = test_dir("augment_consec_spike");
    let poses_path = tmp.join("poses.parquet");
    write_parquet(&mut df, &poses_path).unwrap();
    let lf = bekiper_pose::io::scan_file(&poses_path).unwrap();

    let (x_cols, y_cols, lik_cols) = label_columns(LABELS);
    let coord_cols: Vec<String> = x_cols.iter().chain(y_cols.iter()).cloned().collect();
    let pipeline = AugmentedBuilder::new()
        .add_step(LikelihoodFilter::new(0.9, lik_cols, coord_cols))
        .add_step(HighVelocityFilter {
            max_velocity: 50.0,
            x_columns: x_cols,
            y_columns: y_cols,
            max_iterations: 3,
            sigma_threshold: 0.0,
        });
    let result = pipeline.build_and_collect(lf).unwrap();

    // At least the transition frames (40 and 46) should be nullified
    let nose_x = result.column("nose_x").unwrap();
    assert!(
        nose_x.get(40).unwrap() == AnyValue::Null,
        "first spike frame should be nullified"
    );

    std::fs::remove_dir_all(&tmp).ok();
}
