/// Stage freshness becomes true once output is written.
#[test]
fn stage_freshness_becomes_true_after_write() {
    let tmp = test_dir("stage_fresh_write");
    let exp_dir = ExperimentOutputDir::new(&tmp, "fresh_write");
    exp_dir.ensure_dir().unwrap();

    // Detect: pixel_perimeters.toml in checkpoints dir
    assert!(!exp_dir.is_fresh(Stage::Detect, None));
    let peri_path = exp_dir.pixel_perimeters_toml();
    std::fs::create_dir_all(peri_path.parent().unwrap()).unwrap();
    std::fs::write(&peri_path, "perimeters = []\n").unwrap();
    assert!(exp_dir.is_fresh(Stage::Detect, None));

    // Pose: poses.parquet
    assert!(!exp_dir.is_fresh(Stage::Pose, None));
    std::fs::write(exp_dir.poses_parquet(), b"").unwrap();
    assert!(exp_dir.is_fresh(Stage::Pose, None));

    // Augment: augmented.parquet
    assert!(!exp_dir.is_fresh(Stage::Augment, None));
    std::fs::write(exp_dir.augmented_parquet(), b"").unwrap();
    assert!(exp_dir.is_fresh(Stage::Augment, None));

    // Behaviour: behaviour.parquet
    assert!(!exp_dir.is_fresh(Stage::Behaviour, None));
    std::fs::write(exp_dir.behaviour_parquet(), b"").unwrap();
    assert!(exp_dir.is_fresh(Stage::Behaviour, None));

    // Summary: result.toml
    assert!(!exp_dir.is_fresh(Stage::Summary, None));
    std::fs::write(exp_dir.result_toml(), b"").unwrap();
    assert!(exp_dir.is_fresh(Stage::Summary, None));

    std::fs::remove_dir_all(&tmp).ok();
}

// =========================================================================
// 9. End-to-end: Augment → Behaviour → Summary → Export for each task type
// =========================================================================

#[test]
fn end_to_end_nort_pipeline() {
    let tmp = test_dir("e2e_nort");
    let n = 600;
    let fps = 30.0;
    let mpp = 0.001;

    // Augment
    let mut poses = synthetic_poses(n, LABELS);
    let poses_path = tmp.join("poses.parquet");
    write_parquet(&mut poses, &poses_path).unwrap();
    let lf = bekiper_pose::io::scan_file(&poses_path).unwrap();

    let pipeline = build_augmentation_pipeline(LABELS, &default_pipeline_config(), mpp);
    let mut aug_df = pipeline.build_and_collect(lf).unwrap();

    // Add snout direction columns (center_ear → nose) required by NORT
    let nose_x = aug_df.column("nose_x").unwrap().f64().unwrap().clone();
    let ear_x = aug_df
        .column("center_ear_x")
        .unwrap()
        .f64()
        .unwrap()
        .clone();
    let nose_y = aug_df.column("nose_y").unwrap().f64().unwrap().clone();
    let ear_y = aug_df
        .column("center_ear_y")
        .unwrap()
        .f64()
        .unwrap()
        .clone();
    let _ = aug_df.with_column(
        (&nose_x - &ear_x)
            .into_column()
            .with_name("snout_dx".into()),
    );
    let _ = aug_df.with_column(
        (&nose_y - &ear_y)
            .into_column()
            .with_name("snout_dy".into()),
    );

    // Ear direction columns (center_ear → left_ear, center_ear → right_ear) for WhiskerHeuristic
    let left_ear_x = aug_df.column("left_ear_x").unwrap().f64().unwrap().clone();
    let left_ear_y = aug_df.column("left_ear_y").unwrap().f64().unwrap().clone();
    let right_ear_x = aug_df.column("right_ear_x").unwrap().f64().unwrap().clone();
    let right_ear_y = aug_df.column("right_ear_y").unwrap().f64().unwrap().clone();
    let _ = aug_df.with_column(
        (&left_ear_x - &ear_x)
            .into_column()
            .with_name("left_ear_dx".into()),
    );
    let _ = aug_df.with_column(
        (&left_ear_y - &ear_y)
            .into_column()
            .with_name("left_ear_dy".into()),
    );
    let _ = aug_df.with_column(
        (&right_ear_x - &ear_x)
            .into_column()
            .with_name("right_ear_dx".into()),
    );
    let _ = aug_df.with_column(
        (&right_ear_y - &ear_y)
            .into_column()
            .with_name("right_ear_dy".into()),
    );

    let aug_path = tmp.join("augmented.parquet");
    write_parquet(&mut aug_df, &aug_path).unwrap();
    assert!(aug_path.exists());

    // Behaviour
    let aug_lf = bekiper_pose::io::scan_file(&aug_path).unwrap();
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
    let (result, detailed) = task.analyze_detailed(aug_lf, &timing);

    assert!(!result.summaries.is_empty());
    assert!(result.evaluated_df.is_some());

    // Summary TOML
    let mut summary = serde_json::json!({
        "experiment": "nort_test",
        "task_type": "object_recognition",
        "fps": fps,
        "meters_per_pixel": mpp,
        "summaries": result.summaries,
    });
    if let Some(ref d) = detailed {
        summary
            .as_object_mut()
            .unwrap()
            .insert("detailed".into(), d.as_json_value());
    }
    let summary_path = tmp.join("summary.toml");
    std::fs::write(&summary_path, json_value_to_toml_string(&summary).unwrap()).unwrap();
    assert!(summary_path.exists());

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn end_to_end_oft_pipeline() {
    let tmp = test_dir("e2e_oft");
    let n = 900;
    let fps = 30.0;
    let mpp = 0.001;

    // Augment
    let mut poses = synthetic_poses(n, LABELS);
    let poses_path = tmp.join("poses.parquet");
    write_parquet(&mut poses, &poses_path).unwrap();
    let lf = bekiper_pose::io::scan_file(&poses_path).unwrap();

    let aug_path = tmp.join("augmented.parquet");
    let pipeline = build_augmentation_pipeline(LABELS, &default_pipeline_config(), mpp);
    pipeline.build_and_sink_parquet(lf, &aug_path).unwrap();

    // Behaviour
    let aug_lf = bekiper_pose::io::scan_file(&aug_path).unwrap();
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
    let (result, _) = task.analyze_detailed(aug_lf, &timing);

    assert!(!result.summaries.is_empty());
    assert!(result.evaluated_df.is_some());

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn end_to_end_ymaze_pipeline() {
    let tmp = test_dir("e2e_ymaze");
    let n = 900;
    let fps = 30.0;
    let mpp = 0.001;

    // Augment
    let mut poses = synthetic_poses(n, LABELS);
    let poses_path = tmp.join("poses.parquet");
    write_parquet(&mut poses, &poses_path).unwrap();
    let lf = bekiper_pose::io::scan_file(&poses_path).unwrap();

    let aug_path = tmp.join("augmented.parquet");
    let pipeline = build_augmentation_pipeline(LABELS, &default_pipeline_config(), mpp);
    pipeline.build_and_sink_parquet(lf, &aug_path).unwrap();

    // Behaviour
    let aug_lf = bekiper_pose::io::scan_file(&aug_path).unwrap();
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
    let (result, detailed) = task.analyze_detailed(aug_lf, &timing);

    assert!(!result.summaries.is_empty());
    assert!(result.evaluated_df.is_some());
    assert!(detailed.is_some());

    std::fs::remove_dir_all(&tmp).ok();
}
