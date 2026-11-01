#[test]
fn likelihood_pair_row_scores_high_and_low_confidence() {
    let df = df! {
        "frame" => &[0i64, 1, 2, 3],
        "nose_x" => &[0.0, 1.0, 2.0, 3.0],
        "nose_y" => &[0.0, 0.0, 0.0, 0.0],
        "nose_likelihood" => &[Some(0.95), Some(0.2), None, Some(0.91)],
    }
    .unwrap();

    let row = likelihood_pair_row(&df, "trial", "nose", 0.9).unwrap();

    assert_eq!(row.frames, 4);
    assert_eq!(row.low_likelihood_frames, 2);
    assert!((row.low_likelihood_fraction - 0.5).abs() < 1e-12);
    assert_eq!(row.min_likelihood, Some(0.2));
    assert_eq!(row.max_likelihood, Some(0.95));
}

#[test]
fn likelihood_pair_row_skips_missing_likelihood_column() {
    let df = df! {
        "frame" => &[0i64, 1],
        "nose_x" => &[0.0, 1.0],
        "nose_y" => &[0.0, 0.0],
    }
    .unwrap();

    assert!(likelihood_pair_row(&df, "trial", "nose", 0.9).is_none());
}

#[test]
fn likelihood_report_aggregates_rankings_and_is_read_only() {
    let root = reset_fixture_root("bekiper_likelihood_report");
    std::fs::create_dir_all(root.join("checkpoint/oft_videos_oft_pre_pre_video_1")).unwrap();
    std::fs::create_dir_all(root.join("checkpoint/nort_videos_nort_post_post_video_2")).unwrap();

    let config_path = write_config(
        &root,
        r#"
        [project]
        project_name = "likelihood"
        data_directory = "."
        output_dir = "output"

        [pipeline]
        likelihood_threshold = 0.9

        [[experiments]]
        name = "oft_videos_oft_pre_pre_video_1"
        data_files = ["trial1.parquet"]
        task_type = "open_field"
        fps = 30.0
        meters_per_pixel = 0.001

        [[experiments]]
        name = "nort_videos_nort_post_post_video_2"
        data_files = ["trial2.parquet"]
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001
        "#,
    );

    let mut first = df! {
        "frame" => &[0i64, 1, 2],
        "nose_x" => &[0.0, 1.0, 2.0],
        "nose_y" => &[0.0, 0.0, 0.0],
        "nose_likelihood" => &[0.95, 0.92, 0.91],
        "tail_base_x" => &[0.0, 1.0, 2.0],
        "tail_base_y" => &[0.0, 0.0, 0.0],
    }
    .unwrap();
    bekiper_pose::io::write_parquet(
        &mut first,
        &root.join("checkpoint/oft_videos_oft_pre_pre_video_1/poses.parquet"),
    )
    .unwrap();

    let mut second = df! {
        "frame" => &[0i64, 1, 2],
        "nose_x" => &[0.0, 1.0, 2.0],
        "nose_y" => &[0.0, 0.0, 0.0],
        "nose_likelihood" => &[0.1, 0.2, 0.3],
    }
    .unwrap();
    bekiper_pose::io::write_parquet(
        &mut second,
        &root.join("checkpoint/nort_videos_nort_post_post_video_2/poses.parquet"),
    )
    .unwrap();

    let report = check_likelihood(&config_path, false);

    assert!(!report.has_errors());
    assert_eq!(report.summary.experiments_expected, 2);
    assert_eq!(report.summary.experiments_inspected, 2);
    assert_eq!(report.summary.labels_scored, 2);
    assert_eq!(report.summary.labels_skipped_missing_likelihood, 1);
    assert!(!report.summary.all_labels);
    assert_eq!(
        report.summary.pose_label_model,
        PoseLabelModel::SuperanimalTopviewmouse
    );
    assert_eq!(
        report.summary.focused_labels,
        vec![
            "center_ear",
            "head_midpoint",
            "mouse_center",
            "nose",
            "tail_base"
        ]
    );
    assert_eq!(report.summary.low_likelihood_frames, 3);
    assert_eq!(
        report.worst_pairs[0].experiment,
        "nort_videos_nort_post_post_video_2"
    );
    assert_eq!(report.worst_labels[0].key, "nose");
    assert!(
        report
            .task_phase_summary
            .iter()
            .any(|row| row.key == "oft/pre")
    );
    assert!(
        !root
            .join("checkpoint/oft_videos_oft_pre_pre_video_1/augmented.parquet")
            .exists(),
        "likelihood report must not create augmented outputs"
    );

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn likelihood_report_errors_on_missing_poses() {
    let root = reset_fixture_root("bekiper_likelihood_missing_poses");
    std::fs::create_dir_all(&root).unwrap();
    let config_path = root.join("bekiper_config.toml");
    std::fs::write(&config_path, LIKELIHOOD_CONFIG).unwrap();

    let output_root = root.join("output");
    let report = check_likelihood(&config_path, false);

    assert!(report.has_errors());
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.severity == AuditSeverity::Error
            && diagnostic.message.contains("poses.parquet is missing")
            && diagnostic
                .validation_command
                .as_deref()
                .is_some_and(|cmd| cmd.contains("bekiper check likelihood"))
    }));
    assert!(!output_root.exists(), "likelihood check must be read-only");

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn likelihood_report_all_labels_flag_includes_non_analysis_labels() {
    let root = reset_fixture_root("bekiper_likelihood_all_labels");
    std::fs::create_dir_all(root.join("checkpoint/trial1")).unwrap();

    let config_path = root.join("bekiper_config.toml");
    std::fs::write(&config_path, LIKELIHOOD_CONFIG).unwrap();

    let mut df = df! {
        "frame" => &[0i64, 1],
        "nose_x" => &[0.0, 1.0],
        "nose_y" => &[0.0, 0.0],
        "nose_likelihood" => &[0.95, 0.95],
        "tail4_x" => &[0.0, 1.0],
        "tail4_y" => &[0.0, 0.0],
        "tail4_likelihood" => &[0.1, 0.1],
    }
    .unwrap();
    bekiper_pose::io::write_parquet(&mut df, &root.join("checkpoint/trial1/poses.parquet"))
        .unwrap();

    let focused = check_likelihood(&config_path, false);
    let general = check_likelihood(&config_path, true);

    assert_eq!(focused.summary.labels_scored, 1);
    assert!(!focused.worst_labels.iter().any(|row| row.key == "tail4"));
    assert!(general.summary.all_labels);
    assert_eq!(general.summary.labels_scored, 2);
    assert!(general.worst_labels.iter().any(|row| row.key == "tail4"));

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn likelihood_report_uses_configured_imen_label_model() {
    let root = reset_fixture_root("bekiper_likelihood_imen_model");
    std::fs::create_dir_all(root.join("checkpoint/trial1")).unwrap();

    let config_path = write_config(
        &root,
        r#"
        [project]
        project_name = "likelihood"
        data_directory = "."
        output_dir = "output"

        [pipeline]
        pose_label_model = "imen_dlc"

        [[experiments]]
        name = "trial1"
        data_files = ["trial1.parquet"]
        task_type = "open_field"
        fps = 30.0
        meters_per_pixel = 0.001
        "#,
    );

    let mut df = df! {
        "frame" => &[0i64, 1],
        "nose_x" => &[0.0, 1.0],
        "nose_y" => &[0.0, 0.0],
        "nose_likelihood" => &[0.95, 0.95],
        "left_ear_x" => &[0.0, 1.0],
        "left_ear_y" => &[0.0, 0.0],
        "left_ear_likelihood" => &[0.8, 0.85],
        "tail4_x" => &[0.0, 1.0],
        "tail4_y" => &[0.0, 0.0],
        "tail4_likelihood" => &[0.1, 0.1],
    }
    .unwrap();
    bekiper_pose::io::write_parquet(&mut df, &root.join("checkpoint/trial1/poses.parquet"))
        .unwrap();

    let report = check_likelihood(&config_path, false);

    assert_eq!(report.summary.pose_label_model, PoseLabelModel::ImenDlc);
    assert_eq!(
        report.summary.focused_labels,
        vec!["left_ear", "nose", "right_ear", "tail"]
    );
    assert_eq!(report.summary.labels_scored, 2);
    assert!(report.worst_labels.iter().any(|row| row.key == "left_ear"));
    assert!(!report.worst_labels.iter().any(|row| row.key == "tail4"));

    std::fs::remove_dir_all(&root).ok();
}
