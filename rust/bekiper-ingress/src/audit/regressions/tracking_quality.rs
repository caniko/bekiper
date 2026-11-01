#[test]
fn pose_quality_metrics_count_gaps_interpolation_and_jumps() {
    let raw_df = df! {
        "frame" => &[0i64, 1, 2, 3, 4, 5],
        "nose_x" => &[Some(0.0), Some(1.0), None, None, Some(4.0), Some(5.0)],
        "nose_y" => &[Some(0.0), Some(0.0), None, None, Some(0.0), Some(0.0)],
        "nose_likelihood" => &[Some(0.95), Some(0.2), Some(0.1), Some(0.1), Some(0.93), Some(0.94)],
    }
    .unwrap();
    let augmented_df = df! {
        "frame" => &[0i64, 1, 2, 3, 4, 5],
        "nose_x" => &[0.0, 1.0, 2.0, 3.0, 100.0, 101.0],
        "nose_y" => &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();

    let metrics = compute_pose_quality_metrics(
        &raw_df,
        Some(&augmented_df),
        &["nose".to_string()],
        0.9,
        10.0,
    )
    .unwrap();

    let nose = &metrics[0];
    assert_eq!(nose.low_likelihood_count, Some(3));
    assert_eq!(nose.raw_missing_count, 2);
    assert_eq!(nose.augmented_missing_count, Some(0));
    assert_eq!(nose.longest_gap_frames, 3);
    assert_eq!(nose.short_gap_count, 1);
    assert_eq!(nose.interpolation_burden_count, Some(3));
    assert_eq!(nose.augmented_jump_count, Some(1));
    assert_eq!(nose.worst_augmented_displacement, Some(97.0));
}

#[test]
fn pose_quality_reports_missing_poses_without_writing_outputs() {
    let root = reset_fixture_root("bekiper_pose_quality_missing_poses");
    std::fs::create_dir_all(&root).unwrap();
    let config_path = write_config(
        &root,
        r#"
        [project]
        project_name = "pose_quality"
        data_directory = "."
        output_dir = "output"

        [[experiments]]
        name = "trial1"
        data_files = ["trial1.parquet"]
        task_type = "open_field"
        fps = 30.0
        meters_per_pixel = 0.001
        "#,
    );

    let output_root = root.join("output");
    assert!(!output_root.exists());
    let report = check_pose_quality(&config_path);

    assert!(report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.severity == AuditSeverity::Error
            && diagnostic.message.contains("poses.parquet is missing")
            && diagnostic
                .validation_command
                .as_deref()
                .is_some_and(|cmd| cmd.contains("bekiper check pose-quality"))
    }));
    assert!(
        !output_root.exists(),
        "pose-quality check must be read-only"
    );

    std::fs::remove_dir_all(&root).ok();
}
