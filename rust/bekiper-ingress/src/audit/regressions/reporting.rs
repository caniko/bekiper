#[test]
fn readiness_aggregates_multiple_detection_path_failures() {
    let dir = reset_fixture_root("bekiper_readiness_multiple_detection_failures");
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = write_config(
        &dir,
        r#"
        [project]
        project_name = "readiness"
        data_directory = "."

        [[experiments]]
        name = "missing_a"
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001
        video = "missing-a.mp4"

        [experiments.detection]
        recipe = "nort_training"
        backend = ["classical_cv"]

        [[experiments]]
        name = "missing_b"
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001
        video = "missing-b.mp4"

        [experiments.detection]
        recipe = "nort_training"
        backend = ["classical_cv"]
        "#,
    );

    let report = check_readiness(&config_path);
    let path_errors = report
        .diagnostics
        .iter()
        .filter(|diagnostic| {
            diagnostic.category == "path/video" && diagnostic.severity == AuditSeverity::Error
        })
        .count();
    assert_eq!(path_errors, 2);
    assert!(report.has_errors());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn audit_report_serializes_stable_json_fields() {
    let mut report = AuditReport::new("test", None);
    report.push(
        AuditDiagnostic::new(AuditSeverity::Error, "path/video", "missing video")
            .experiment("trial")
            .artifact_path("missing.mp4")
            .cause("input video is required")
            .fix_hint("restore the video")
            .validation_command("bekiper check readiness"),
    );

    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("\"severity\":\"error\""));
    assert!(json.contains("\"category\":\"path/video\""));
    assert!(json.contains("\"validation_command\":\"bekiper check readiness\""));
}
