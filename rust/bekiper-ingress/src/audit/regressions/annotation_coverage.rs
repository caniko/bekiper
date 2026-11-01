#[test]
fn manual_annotation_coverage_accepts_resolved_sessions() {
    let root = reset_fixture_root("bekiper_manual_annotation_coverage_ok");
    let video_dir = root.join("videos/T1");
    let annotation_dir = root.join("annotations/nort");
    let plugin_dir = annotation_dir.join("plugin_files");
    let models_dir = root.join("models");
    std::fs::create_dir_all(&video_dir).unwrap();
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::write(video_dir.join("pre.video-10.mp4"), "").unwrap();
    std::fs::write(
        plugin_dir.join("perimeter-rectangle-A1_100.csv"),
        "familiar,48,49,25,38,img.png,224,224\nnovel,152,150,28,27,img.png,224,224\n",
    )
    .unwrap();
    std::fs::write(
        annotation_dir.join("video_map.csv"),
        "video_number,perimeter_id\n10,A1_100\n",
    )
    .unwrap();
    std::fs::write(models_dir.join("model.onnx"), "").unwrap();
    std::fs::write(models_dir.join("model.json"), test_model_spec_json()).unwrap();

    let config_path = root.join("bekiper_config.toml");
    std::fs::write(&config_path, ANNOTATION_COVERAGE_CONFIG).unwrap();

    let summary = check_manual_annotation_coverage(&config_path).unwrap();
    assert_eq!(summary.total_manual_sessions, 1);
    assert_eq!(summary.assigned_sessions, 1);
    assert!(summary.is_complete());

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn manual_annotation_coverage_reports_unresolved_sessions() {
    let root = reset_fixture_root("bekiper_manual_annotation_coverage_missing");
    let video_dir = root.join("videos/T1");
    let annotation_dir = root.join("annotations/nort");
    let plugin_dir = annotation_dir.join("plugin_files");
    let models_dir = root.join("models");
    std::fs::create_dir_all(&video_dir).unwrap();
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::write(video_dir.join("pre.video-10.mp4"), "").unwrap();
    std::fs::write(
        annotation_dir.join("video_map.csv"),
        "video_number,perimeter_id\n11,A1_100\n",
    )
    .unwrap();
    std::fs::write(models_dir.join("model.onnx"), "").unwrap();
    std::fs::write(models_dir.join("model.json"), test_model_spec_json()).unwrap();

    let config_path = root.join("bekiper_config.toml");
    std::fs::write(&config_path, ANNOTATION_COVERAGE_CONFIG).unwrap();

    let summary = check_manual_annotation_coverage(&config_path).unwrap();
    assert_eq!(summary.total_manual_sessions, 1);
    assert_eq!(summary.assigned_sessions, 0);
    assert_eq!(summary.missing_sessions.len(), 1);
    assert_eq!(
        summary.missing_sessions[0].experiment,
        "videos_t1_pre_video_10"
    );
    assert!(summary.missing_sessions[0].error.contains("video"));

    std::fs::remove_dir_all(&root).ok();
}
