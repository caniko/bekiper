#[test]
fn ymaze_alignment_diagnostic_flags_zero_region_overlap() {
    let dir = reset_fixture_root("bekiper_ymaze_alignment_diagnostic");
    let exp_dir = ExperimentOutputDir::new(&dir, "trial");
    exp_dir.ensure_checkpoints_dir().unwrap();
    std::fs::write(
        exp_dir.perimeters_toml(),
        r#"
        [[perimeters]]
        shape = "radial_maze"
        label = "maze"
        center_vertices = [[-0.2, -0.2], [0.2, -0.2], [0.0, 0.2]]
        arms = [
          [[0.2, -0.1], [0.2, 0.1], [1.0, 0.1], [1.0, -0.1]],
          [[-0.1, 0.2], [0.1, 0.2], [0.1, 1.0], [-0.1, 1.0]],
          [[-1.0, -0.1], [-1.0, 0.1], [-0.2, 0.1], [-0.2, -0.1]],
        ]
        "#,
    )
    .unwrap();

    let df = df! {
        "center_ear_x" => [0.6, 0.7, 0.8],
        "center_ear_y" => [0.0, 0.0, 0.0],
    }
    .unwrap();
    let experiment = ExperimentConfig {
        recording_id: None,
        name: "trial".into(),
        task_type: TaskType::YMaze,
        fps: Default::default(),
        meters_per_pixel: bekiper_core::scale::ScaleConfig::Manual(0.001),
        data_files: Vec::new(),
        video: None,
        pose_model: None,
        detection: None,
        task_params: None,
        trial_start_perimeter: None,
        annotation_dir: None,
        frame_timing: None,
        video_resolution: None,
        frame_image: None,
        timestamps_file: None,
        animal_id: None,
        session_context: None,
    };

    let diagnostics =
        ymaze_alignment_diagnostics(&experiment, Some(&df), &exp_dir, "bekiper check").unwrap();

    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].severity, AuditSeverity::Error);
    assert!(diagnostics[0].message.contains("center"));
    assert!(diagnostics[0].message.contains("arm_B"));
    assert!(diagnostics[0].message.contains("arm_C"));

    std::fs::remove_dir_all(&dir).ok();
}
