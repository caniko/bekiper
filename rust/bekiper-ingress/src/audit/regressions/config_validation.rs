#[test]
fn validation_rejects_missing_data_file() {
    let dir = reset_fixture_root("bekiper_config_audit_missing_data");
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = write_config(
        &dir,
        r#"
        [project]
        project_name = "audit"
        data_directory = "."

        [[experiments]]
        name = "trial"
        data_files = ["missing.csv"]
        task_type = "open_field"
        fps = 30.0
        meters_per_pixel = 0.001
        "#,
    );

    let err = validate_config_file(&config_path).unwrap_err();
    assert!(err.to_string().contains("missing data file"));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn validation_accepts_pose_model_paths_rebased_from_nested_config_dir() {
    let root = reset_fixture_root("bekiper_config_audit_pose_model_nested_ok");
    let config_dir = root.join("examples/neo/imen");
    let models_dir = root.join("models");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::write(config_dir.join("trial.csv"), "frame,nose_x\n0,0.0\n").unwrap();
    std::fs::write(models_dir.join("model.onnx"), "").unwrap();
    std::fs::write(models_dir.join("model.json"), test_model_spec_json()).unwrap();

    let config_path = write_config(
        &config_dir,
        r#"
        [project]
        project_name = "audit"
        data_directory = "."

        [[experiments]]
        name = "trial"
        data_files = ["trial.csv"]
        task_type = "open_field"
        fps = 30.0
        meters_per_pixel = 0.001

        [experiments.pose_model]
        path = "../../../models/model.onnx"
        spec = "../../../models/model.json"
        "#,
    );

    let summary = validate_config_file(&config_path).unwrap();
    assert_eq!(summary.experiments, 1);

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn validation_reports_resolved_missing_pose_model_path() {
    let root = reset_fixture_root("bekiper_config_audit_pose_model_missing");
    let config_dir = root.join("examples/neo/imen");
    let models_dir = root.join("models");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::write(config_dir.join("trial.csv"), "frame,nose_x\n0,0.0\n").unwrap();
    std::fs::write(models_dir.join("model.json"), test_model_spec_json()).unwrap();

    let config_path = write_config(
        &config_dir,
        r#"
        [project]
        project_name = "audit"
        data_directory = "."

        [[experiments]]
        name = "trial"
        data_files = ["trial.csv"]
        task_type = "open_field"
        fps = 30.0
        meters_per_pixel = 0.001

        [experiments.pose_model]
        path = "../../../models/missing.onnx"
        spec = "../../../models/model.json"
        "#,
    );

    let err = validate_config_file(&config_path).unwrap_err();
    let expected_path = config_dir.join("../../../models/missing.onnx");
    assert!(
        err.to_string()
            .contains(&expected_path.display().to_string())
    );

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn validation_accepts_imen_pose_label_model_with_matching_custom_spec() {
    let root = reset_fixture_root("bekiper_config_audit_imen_model_ok");
    let models_dir = root.join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::write(models_dir.join("model.onnx"), "").unwrap();
    std::fs::write(models_dir.join("model.json"), imen_model_spec_json()).unwrap();
    std::fs::write(root.join("video.mp4"), "").unwrap();

    let config_path = root.join("bekiper_config.toml");
    std::fs::write(&config_path, IMEN_NORT_MODEL_CONFIG).unwrap();

    let summary = validate_config_file(&config_path).unwrap();
    assert_eq!(summary.experiments, 1);

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn validation_rejects_imen_pose_label_model_with_superanimal_spec() {
    let root = reset_fixture_root("bekiper_config_audit_imen_model_conflict");
    let models_dir = root.join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::write(models_dir.join("model.onnx"), "").unwrap();
    std::fs::write(models_dir.join("model.json"), test_model_spec_json()).unwrap();
    std::fs::write(root.join("video.mp4"), "").unwrap();

    let config_path = root.join("bekiper_config.toml");
    std::fs::write(&config_path, IMEN_NORT_MODEL_CONFIG).unwrap();

    let err = validate_config_file(&config_path).unwrap_err();
    assert!(err.to_string().contains("pose_label_model 'imen_dlc'"));
    assert!(err.to_string().contains("superanimal_topviewmouse"));

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn validation_accepts_resolved_scan_annotation_dir_without_double_joining() {
    let root = reset_fixture_root("bekiper_config_audit_scan_annotation_dir");
    let config_dir = root.join("examples/neo/imen");
    let video_dir = config_dir.join("videos/T1");
    let annotation_dir = config_dir.join("annotations/nort");
    let plugin_dir = annotation_dir.join("plugin_files");
    let models_dir = root.join("models");
    std::fs::create_dir_all(&video_dir).unwrap();
    std::fs::create_dir_all(&plugin_dir).unwrap();
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::write(video_dir.join("trial.mp4"), "").unwrap();
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

    let config_path = write_config(
        &config_dir,
        r#"
        [project]
        project_name = "audit"
        data_directory = "."

        [pose_model]
        path = "../../../models/model.onnx"
        spec = "../../../models/model.json"

        [[scan]]
        video_dir = "videos"
        task_type = "object_recognition"
        meters_per_pixel = 0.001
        backend = ["manual_annotation"]
        annotations = { dir = "annotations/nort" }

        [scan.recipe_rules]
        "**/T1/**" = "nort_training"
        "#,
    );

    let summary = validate_config_file(&config_path).unwrap();
    assert_eq!(summary.experiments, 1);

    std::fs::remove_dir_all(&root).ok();
}
