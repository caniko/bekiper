#[test]
fn parse_imen_scale_configs() {
    let toml = r#"
[project]
project_name = "test"
data_directory = "."

[[experiments]]
name = "manual_scale"
task_type = "open_field"
fps = 30.0
meters_per_pixel = 0.001914

[[experiments]]
name = "derived_arena"
task_type = "open_field"
fps = 30.0
meters_per_pixel = { method = "arena_side", meters = 0.40 }

[[experiments]]
name = "derived_arm"
task_type = "y_maze"
fps = 30.0
meters_per_pixel = { method = "arm_length", meters = 0.35 }
"#;

    let tmp = test_dir("config_scale");
    let config_path = tmp.join("config.toml");
    std::fs::write(&config_path, toml).unwrap();

    let config = IngressConfig::from_file(&config_path).unwrap();
    assert_eq!(config.experiments.len(), 3);

    // Manual scale resolves directly
    let manual_mpp = config.experiments[0].meters_per_pixel.resolve(&[]).unwrap();
    assert!((manual_mpp.0 - 0.001914).abs() < 1e-10);

    // Derived scale from arena side: needs an arena perimeter in pixel space
    let pixel_arena = vec![PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 320.0,
        center_y: 240.0,
        width: 400.0,
        height: 400.0,
    }];
    let derived_mpp = config.experiments[1]
        .meters_per_pixel
        .resolve(&pixel_arena)
        .unwrap();
    // 0.40 m / 400 px = 0.001 m/px
    assert!(
        (derived_mpp.0 - 0.001).abs() < 1e-6,
        "expected ~0.001, got {}",
        derived_mpp.0
    );

    std::fs::remove_dir_all(&tmp).ok();
}

// =========================================================================
// 2. Augmentation pipeline — LikelihoodFilter + VelocityFilter + Smoother + Transform
// =========================================================================

#[test]
fn augment_filters_low_confidence_keypoints() {
    let n = 200;
    let mut df = synthetic_poses(n, LABELS);
    let tmp = test_dir("augment_likelihood");

    // Write poses to parquet, then scan back as LazyFrame
    let poses_path = tmp.join("poses.parquet");
    write_parquet(&mut df, &poses_path).unwrap();
    let lf = bekiper_pose::io::scan_file(&poses_path).unwrap();

    // Apply ONLY the likelihood filter (not the full pipeline, because the
    // downstream MedianSmoother interpolates nulls back).
    let (x_cols, y_cols, lik_cols) = label_columns(LABELS);
    let coord_cols: Vec<String> = x_cols.iter().chain(y_cols.iter()).cloned().collect();
    let filter = LikelihoodFilter::new(0.9, lik_cols, coord_cols);

    let pipeline = AugmentedBuilder::new().add_step(filter);
    let result = pipeline.build_and_collect(lf).unwrap();

    assert_eq!(result.height(), n);

    // Frames at indices 50, 100, 150 have likelihood=0.1 (below 0.9 threshold)
    // so their coordinates should be null after filtering
    let nose_x = result.column("nose_x").unwrap();
    assert!(
        nose_x.get(50).unwrap() == AnyValue::Null,
        "low-confidence frame should be nullified"
    );
    // Frame 1 should remain (high confidence)
    assert!(
        nose_x.get(1).unwrap() != AnyValue::Null,
        "high-confidence frame should be preserved"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn augment_filters_velocity_spikes() {
    let n = 200;
    let mut df = synthetic_poses_with_spike(n, LABELS);
    let tmp = test_dir("augment_velocity");

    let poses_path = tmp.join("poses.parquet");
    write_parquet(&mut df, &poses_path).unwrap();
    let lf = bekiper_pose::io::scan_file(&poses_path).unwrap();

    // Apply ONLY likelihood + velocity filters (not the full pipeline, because
    // the downstream MedianSmoother interpolates nulls back).
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

    // The spike at frame n/2 should be nullified by the velocity filter
    let nose_x = result.column("nose_x").unwrap();
    assert!(
        nose_x.get(n / 2).unwrap() == AnyValue::Null,
        "velocity spike should be nullified"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn augment_scales_pixel_to_meters() {
    let n = 100;
    let mut df = synthetic_poses(n, LABELS);
    let tmp = test_dir("augment_scale");

    let poses_path = tmp.join("poses.parquet");
    write_parquet(&mut df, &poses_path).unwrap();
    let lf = bekiper_pose::io::scan_file(&poses_path).unwrap();

    let mpp = 0.001; // 1 mm per pixel
    let pipeline = build_augmentation_pipeline(LABELS, &default_pipeline_config(), mpp);
    let result = pipeline.build_and_collect(lf).unwrap();

    // Original pixel values around 200; after scaling at 0.001 m/px → ~0.2 m
    // (exact value affected by smoothing and filtering, but order of magnitude holds)
    let nose_x = result.column("nose_x").unwrap().f64().unwrap();
    let sample = nose_x.get(10).unwrap();
    assert!(
        sample < 1.0 && sample > 0.1,
        "scaled coordinate should be in meter range, got {sample}"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn augment_full_pipeline_to_parquet() {
    let n = 300;
    let mut df = synthetic_poses(n, LABELS);
    let tmp = test_dir("augment_parquet");

    let poses_path = tmp.join("poses.parquet");
    write_parquet(&mut df, &poses_path).unwrap();
    let lf = bekiper_pose::io::scan_file(&poses_path).unwrap();

    let output_path = tmp.join("augmented.parquet");
    let pipeline = build_augmentation_pipeline(LABELS, &default_pipeline_config(), 0.001);
    pipeline.build_and_sink_parquet(lf, &output_path).unwrap();

    // Read back and verify
    let result = LazyFrame::scan_parquet(output_path.to_str().unwrap().into(), Default::default())
        .unwrap()
        .collect()
        .unwrap();

    assert_eq!(result.height(), n);
    for label in LABELS {
        assert!(result.column(&format!("{label}_x")).is_ok());
        assert!(result.column(&format!("{label}_y")).is_ok());
    }

    std::fs::remove_dir_all(&tmp).ok();
}

// =========================================================================
// 3. Behaviour analysis — NORT, OFT, Y-maze
// =========================================================================

/// Generate augmented poses in meter-space for behaviour analysis.
/// Includes `snout_dx`/`snout_dy` direction columns required by NORT exploration heuristic.
fn meter_space_poses(n: usize, labels: &[&str]) -> DataFrame {
    let frames: Vec<i64> = (0..n as i64).collect();
    let mut columns: Vec<Column> = vec![Column::new("frame".into(), &frames)];

    for (li, label) in labels.iter().enumerate() {
        let offset = li as f64 * 0.01;
        // Move within a small arena (~0.40 m side)
        let xs: Vec<f64> = (0..n)
            .map(|i| 0.10 + offset + (i as f64) * 0.0002)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|i| 0.10 + offset + (i as f64) * 0.0001)
            .collect();
        columns.push(Column::new(format!("{label}_x").into(), &xs));
        columns.push(Column::new(format!("{label}_y").into(), &ys));
    }

    // Compute snout direction (center_ear → nose) required by exploration heuristic
    let nose_idx = labels.iter().position(|l| *l == "nose").unwrap_or(0);
    let ear_idx = labels.iter().position(|l| *l == "center_ear").unwrap_or(3);
    let snout_dx: Vec<f64> = (0..n)
        .map(|i| {
            let nose_x = 0.10 + nose_idx as f64 * 0.01 + (i as f64) * 0.0002;
            let ear_x = 0.10 + ear_idx as f64 * 0.01 + (i as f64) * 0.0002;
            nose_x - ear_x
        })
        .collect();
    let snout_dy: Vec<f64> = (0..n)
        .map(|i| {
            let nose_y = 0.10 + nose_idx as f64 * 0.01 + (i as f64) * 0.0001;
            let ear_y = 0.10 + ear_idx as f64 * 0.01 + (i as f64) * 0.0001;
            nose_y - ear_y
        })
        .collect();
    columns.push(Column::new("snout_dx".into(), &snout_dx));
    columns.push(Column::new("snout_dy".into(), &snout_dy));

    // Ear direction vectors (center_ear → left_ear, center_ear → right_ear)
    // needed by WhiskerHeuristic for qualia decomposition.
    let left_idx = labels.iter().position(|l| *l == "left_ear").unwrap_or(1);
    let right_idx = labels.iter().position(|l| *l == "right_ear").unwrap_or(2);
    let left_ear_dx: Vec<f64> = (0..n)
        .map(|i| {
            let left_x = 0.10 + left_idx as f64 * 0.01 + (i as f64) * 0.0002;
            let ear_x = 0.10 + ear_idx as f64 * 0.01 + (i as f64) * 0.0002;
            left_x - ear_x
        })
        .collect();
    let left_ear_dy: Vec<f64> = (0..n)
        .map(|i| {
            let left_y = 0.10 + left_idx as f64 * 0.01 + (i as f64) * 0.0001;
            let ear_y = 0.10 + ear_idx as f64 * 0.01 + (i as f64) * 0.0001;
            left_y - ear_y
        })
        .collect();
    let right_ear_dx: Vec<f64> = (0..n)
        .map(|i| {
            let right_x = 0.10 + right_idx as f64 * 0.01 + (i as f64) * 0.0002;
            let ear_x = 0.10 + ear_idx as f64 * 0.01 + (i as f64) * 0.0002;
            right_x - ear_x
        })
        .collect();
    let right_ear_dy: Vec<f64> = (0..n)
        .map(|i| {
            let right_y = 0.10 + right_idx as f64 * 0.01 + (i as f64) * 0.0001;
            let ear_y = 0.10 + ear_idx as f64 * 0.01 + (i as f64) * 0.0001;
            right_y - ear_y
        })
        .collect();
    columns.push(Column::new("left_ear_dx".into(), &left_ear_dx));
    columns.push(Column::new("left_ear_dy".into(), &left_ear_dy));
    columns.push(Column::new("right_ear_dx".into(), &right_ear_dx));
    columns.push(Column::new("right_ear_dy".into(), &right_ear_dy));

    DataFrame::new(n, columns).unwrap()
}

#[test]
fn behaviour_nort_produces_heuristics() {
    let n = 600; // 20 seconds at 30 fps
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
    let (result, detailed) = task.analyze_detailed(lf, &timing);

    assert_eq!(result.task_type, TaskType::ObjectRecognition);
    assert!(
        !result.summaries.is_empty(),
        "NORT should produce at least one heuristic summary"
    );
    assert!(
        result.evaluated_df.is_some(),
        "NORT should produce an evaluated DataFrame"
    );

    // Detailed result should contain object exploration metrics
    assert!(detailed.is_some(), "NORT should produce detailed results");

    let eval_df = result.evaluated_df.as_ref().unwrap();
    assert_eq!(eval_df.height(), n);
}

#[test]
fn behaviour_oft_produces_heuristics() {
    let n = 900; // 30 seconds at 30 fps
    let fps = 30.0;
    let df = meter_space_poses(n, LABELS);
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
    let (result, _detailed) = task.analyze_detailed(lf, &timing);

    assert_eq!(result.task_type, TaskType::OpenField);
    assert!(
        !result.summaries.is_empty(),
        "OFT should produce heuristic summaries (freezing, zone, etc.)"
    );
    assert!(result.evaluated_df.is_some());

    let eval_df = result.evaluated_df.as_ref().unwrap();
    assert_eq!(eval_df.height(), n);
}

#[test]
fn behaviour_ymaze_produces_heuristics() {
    let n = 900;
    let fps = 30.0;
    let df = meter_space_poses(n, LABELS);
    let lf = df.lazy();

    let perimeters = ymaze_perimeters();
    let params = TaskParams::new(fps);
    let task = bekiper_behaviour::factory::build_task(
        TaskType::YMaze,
        &perimeters,
        params,
        None,
        None,
        None,
    )
    .unwrap();

    let timing = FrameTiming::cfr(fps);
    let (result, detailed) = task.analyze_detailed(lf, &timing);

    assert_eq!(result.task_type, TaskType::YMaze);
    assert!(
        !result.summaries.is_empty(),
        "Y-maze should produce heuristic summaries"
    );
    assert!(result.evaluated_df.is_some());
    assert!(
        detailed.is_some(),
        "Y-maze should produce detailed results (arm entries, alternations)"
    );
}

// =========================================================================
// 4. Summary + Export — the full summary→export→readback cycle
// =========================================================================
