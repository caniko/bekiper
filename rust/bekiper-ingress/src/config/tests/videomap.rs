#[test]
fn videomap_oft_videos_expand_with_habituation_markers() {
    let dir = std::env::temp_dir().join(format!(
        "bekiper_videomap_oft_habituation_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let video_path = dir.join("OFT videos/OFT (pre)/pre.video-1.mp4");
    std::fs::create_dir_all(video_path.parent().unwrap()).unwrap();
    std::fs::write(&video_path, b"video").unwrap();

    let mut config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "imen"
        data_directory = "."

        [[ingress_profile]]
        profile = "videomap"

        [[ingress_profile.tasks]]
        video_dir = "OFT videos"
        task_type = "open_field"
        meters_per_pixel = 0.001
        recipe = "oft"
        "#,
    )
    .unwrap();

    config
        .expand_for_validation(&dir, crate::stage::Stage::all())
        .unwrap();

    assert_eq!(config.experiments.len(), 1);
    let experiment = &config.experiments[0];
    assert_eq!(experiment.task_type, TaskType::OpenField);
    assert!(experiment.name.starts_with("oft_videos_"));
    assert!(
        experiment
            .video
            .as_ref()
            .is_some_and(|path| path.components().any(|component| {
                matches!(component, std::path::Component::Normal(name) if name == "OFT videos")
            }))
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn videomap_oft_binds_swapped_stage_zero_trials_and_keeps_calibration() {
    let dir =
        std::env::temp_dir().join(format!("bekiper_oft_stage0_binding_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("OFT videos")).unwrap();
    std::fs::create_dir_all(dir.join("annotations/nort")).unwrap();
    for name in ["pre.video-1.mp4", "pre.video-2.mp4", "post.video-1.mp4"] {
        std::fs::write(dir.join("OFT videos").join(name), b"video").unwrap();
    }
    std::fs::write(
        dir.join("annotations/nort/metadata.csv"),
        concat!(
            "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,FecalBoli\n",
            "A,1,2,0,1,3,4\n",
            ",,1,0,2,3,\n",
            ",,99,1,1,3,\n",
            "A,2,7,0,1,3,0\n",
        ),
    )
    .unwrap();
    let mut config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "imen"
        data_directory = "."
        [[ingress_profile]]
        profile = "videomap"
        [[ingress_profile.tasks]]
        video_dir = "OFT videos"
        task_type = "open_field"
        meters_per_pixel = { method = "arena_side", meters = 0.40 }
        recipe = "oft"
        backend = "arena_calibration_cv"
        annotations = { dir = "annotations/nort" }
        [ingress_profile.tasks.arena_calibration]
        center_zone_meters = [0.24, 0.24]
    "#,
    )
    .unwrap();
    let mut missing_binding = config.clone();
    config.expand_for_validation(&dir, &[]).unwrap();
    assert_eq!(config.experiments.len(), 3);
    for experiment in &config.experiments {
        let context = experiment.session_context.as_ref().unwrap();
        let (animal, trial, part) = if experiment.name.contains("post") {
            ("1", "7", "2")
        } else if experiment.name.ends_with("video_1") {
            ("1", "2", "1")
        } else {
            ("2", "1", "1")
        };
        assert_eq!(context.animal.as_deref(), Some(animal));
        assert_eq!(context.trial.as_deref(), Some(trial));
        assert_eq!(context.phase_part.as_deref(), Some(part));
        assert_eq!(context.stage.as_deref(), Some("0"));
        assert!(!crate::analysis_exclusion::is_analysis_excluded_experiment(
            experiment
        ));
        assert!(!crate::analysis_exclusion::is_analysis_excluded_result(
            &experiment.name,
            experiment.task_type,
            &serde_json::to_value(context).unwrap(),
            false
        ));
        let detection = experiment.detection.as_ref().unwrap();
        assert_eq!(detection.recipe, Some(DetectionRecipe::Oft));
        assert!(detection.arena_calibration.is_some());
    }
    std::fs::write(
        dir.join("annotations/nort/metadata.csv"),
        "Phase,PhasePart,TrialNumber,Stage,Animal\nA,1,2,0,1\n",
    )
    .unwrap();
    assert!(
        missing_binding
            .expand_for_validation(&dir, &[])
            .unwrap_err()
            .to_string()
            .contains("(Animal, 0, PhasePart)")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parses_videomap_ingress_profile_config() {
    let config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "imen"
        data_directory = "."

        [[ingress_profile]]
        profile = "videomap"

        [[ingress_profile.tasks]]
        video_dir = "NORT videos"
        pose_data_dir = "deeplabcut/poses"
        task_type = "object_recognition"
        meters_per_pixel = { method = "arena_side", meters = 0.40 }
        backend = ["manual_annotation"]
        annotations = { dir = "annotations/nort" }

        [ingress_profile.tasks.recipe_rules]
        "**/T1/**" = "nort_training"
        "**/T2/**" = "nort_novelty"
        "#,
    )
    .unwrap();

    assert_eq!(config.ingress_profile.len(), 1);
    let profile = &config.ingress_profile[0];
    assert_eq!(profile.profile, IngressProfileId::VideoMap);
    assert_eq!(profile.tasks.len(), 1);
    assert_eq!(
        profile.tasks[0].pose_data_dir.as_deref(),
        Some(Path::new("deeplabcut/poses"))
    );
    assert_eq!(
        profile.tasks[0].recipe_rules.get("**/T2/**"),
        Some(&DetectionRecipe::NortNovelty)
    );
}

#[test]
fn videomap_pose_data_dir_sets_data_file_and_disables_pose_model() {
    let dir =
        std::env::temp_dir().join(format!("bekiper_videomap_pose_data_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("videos")).unwrap();
    std::fs::create_dir_all(dir.join("deeplabcut/poses/videos")).unwrap();
    std::fs::write(dir.join("videos/trial.mp4"), b"video").unwrap();
    std::fs::write(dir.join("deeplabcut/poses/videos/trial.h5"), b"h5").unwrap();
    std::fs::create_dir_all(dir.join("models")).unwrap();
    std::fs::write(dir.join("models/pose.onnx"), b"model").unwrap();
    std::fs::write(dir.join("models/spec.json"), b"{}").unwrap();

    let mut config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "pose_data"
        data_directory = "."

        [pose_model]
        path = "models/pose.onnx"
        spec = "models/spec.json"

        [[ingress_profile]]
        profile = "videomap"

        [[ingress_profile.tasks]]
        video_dir = "videos"
        pose_data_dir = "deeplabcut/poses"
        task_type = "open_field"
        meters_per_pixel = 0.001
        recipe = "oft"
        "#,
    )
    .unwrap();

    config.expand_for_validation(&dir, &[]).unwrap();
    assert_eq!(config.experiments.len(), 1);
    let experiment = &config.experiments[0];
    assert_eq!(
        experiment.data_files,
        vec![PathBuf::from("deeplabcut/poses/videos/trial.h5")]
    );
    assert!(experiment.pose_model.is_none());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn videomap_recipe_rule_survives_imen_stage_zero_video_number_collision() {
    let dir = std::env::temp_dir().join(format!(
        "bekiper_videomap_stage0_collision_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("NORT videos/NORT (pre)/T1/Config A")).unwrap();
    std::fs::create_dir_all(dir.join("deeplabcut/poses/NORT videos/NORT (pre)/T1/Config A"))
        .unwrap();
    std::fs::create_dir_all(dir.join("annotations/nort/bekiper/sessions")).unwrap();
    std::fs::create_dir_all(dir.join("annotations/nort/bekiper/perimeters")).unwrap();
    std::fs::create_dir_all(dir.join("annotations/nort/bekiper/references")).unwrap();
    std::fs::write(
        dir.join("NORT videos/NORT (pre)/T1/Config A/pre.T1.configA.video-10.mp4"),
        b"video",
    )
    .unwrap();
    std::fs::write(
        dir.join("deeplabcut/poses/NORT videos/NORT (pre)/T1/Config A/pre.T1.configA.video-10.h5"),
        b"h5",
    )
    .unwrap();
    std::fs::write(dir.join("annotations/nort/bekiper/manifest.toml"), "").unwrap();
    std::fs::write(
        dir.join("annotations/nort/bekiper/sessions/metadata_row_55_animal_10_stage_1_part_1.toml"),
        concat!(
            "key = \"metadata_row_55_animal_10_stage_1_part_1\"\n",
            "video_number = 10\n",
            "animal = 10\n",
            "stage = 1\n",
            "phase_part = 1\n",
            "perimeter_id = \"training_2__group_target__A1_training_2\"\n",
            "source_perimeter_id = \"training_2\"\n",
            "coordinate_space = \"target\"\n",
            "target_image = \"A1_training_2\"\n",
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join(
            "annotations/nort/bekiper/perimeters/training_2__group_target__A1_training_2.toml",
        ),
        concat!(
            "[[perimeters]]\n",
            "label = \"familiar\"\n",
            "shape = \"polygon\"\n",
            "vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]\n",
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("annotations/nort/bekiper/references/training_2.toml"),
        concat!(
            "[source]\n",
            "x = 154.0\n",
            "y = 50.0\n",
            "image_name = \"A1_training_2.png\"\n",
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("annotations/nort/metadata.csv"),
        concat!(
            "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Video_file_name,Perimeter,ChangeReference,ChangeReferenceImageName\n",
            "A,1,10,0,10,2,Test 10.szv,,,\n",
            ",,57,1,10,2,Test 57.szv,training_2,,A1_training_2\n",
        ),
    )
    .unwrap();

    let mut config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "imen"
        data_directory = "."

        [[ingress_profile]]
        profile = "videomap"

        [[ingress_profile.tasks]]
        video_dir = "NORT videos"
        pose_data_dir = "deeplabcut/poses"
        task_type = "object_recognition"
        meters_per_pixel = { method = "arena_side", meters = 0.40 }
        backend = ["manual_annotation"]
        annotations = { dir = "annotations/nort" }

        [ingress_profile.tasks.recipe_rules]
        "**/T1/**" = "nort_training"
        "#,
    )
    .unwrap();

    config.expand_for_validation(&dir, &[]).unwrap();
    let experiment = &config.experiments[0];
    assert_eq!(
        experiment.detection.as_ref().and_then(|d| d.recipe),
        Some(DetectionRecipe::NortTraining)
    );
    assert_eq!(
        experiment
            .session_context
            .as_ref()
            .and_then(|context| context.source_row),
        Some(3)
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn videomap_nort_t2_recipe_uses_stage_two_context_despite_metadata_collision() {
    let dir = std::env::temp_dir().join(format!(
        "bekiper_videomap_t2_stage_collision_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let video = "NORT videos/NORT (post)/T2/Config B/post.T2.configB.video-59.mp4";
    std::fs::create_dir_all(dir.join("NORT videos/NORT (post)/T2/Config B")).unwrap();
    std::fs::create_dir_all(dir.join("deeplabcut/poses/NORT videos/NORT (post)/T2/Config B"))
        .unwrap();
    std::fs::write(dir.join(video), b"video").unwrap();
    std::fs::write(
        dir.join("deeplabcut/poses")
            .join(Path::new(video).with_extension("h5")),
        b"h5",
    )
    .unwrap();
    write_canonical_test_session(&dir, 73, 1, 2, "training_4", &["familiar", "variable"]);
    write_canonical_test_session(&dir, 73, 2, 2, "novel_4", &["familiar", "novel"]);
    std::fs::write(
        dir.join("annotations/nort/metadata.csv"),
        concat!(
            "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Video_file_name,Perimeter,ChangeReference,ChangeReferenceImageName\n",
            "B,2,59,1,73,2,Test 59.szv,training_4,,B2_training_4\n",
            ",,59,2,73,2,Test 440.szv,novel_4,,B2_novel_4\n",
        ),
    )
    .unwrap();

    let mut config = imen_nort_config_with_rule("**/T2/**", DetectionRecipe::NortNovelty);
    config.expand_for_validation(&dir, &[]).unwrap();
    let experiment = &config.experiments[0];

    assert_eq!(
        experiment.detection.as_ref().and_then(|d| d.recipe),
        Some(DetectionRecipe::NortNovelty)
    );
    assert_eq!(
        experiment
            .session_context
            .as_ref()
            .and_then(|context| context.animal.as_deref()),
        Some("73")
    );
    assert_eq!(
        experiment
            .session_context
            .as_ref()
            .and_then(|context| context.stage.as_deref()),
        Some("2")
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn videomap_nort_t1_recipe_uses_stage_one_context_despite_metadata_collision() {
    let dir = std::env::temp_dir().join(format!(
        "bekiper_videomap_t1_stage_collision_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let video = "NORT videos/NORT (pre)/T1/Config B/pre.T1.configB.video-83.mp4";
    std::fs::create_dir_all(dir.join("NORT videos/NORT (pre)/T1/Config B")).unwrap();
    std::fs::create_dir_all(dir.join("deeplabcut/poses/NORT videos/NORT (pre)/T1/Config B"))
        .unwrap();
    std::fs::write(dir.join(video), b"video").unwrap();
    std::fs::write(
        dir.join("deeplabcut/poses")
            .join(Path::new(video).with_extension("h5")),
        b"h5",
    )
    .unwrap();
    write_canonical_test_session(&dir, 56, 2, 1, "novel_2", &["familiar", "novel"]);
    write_canonical_test_session(&dir, 56, 1, 1, "training_2", &["familiar", "variable"]);
    std::fs::write(
        dir.join("annotations/nort/metadata.csv"),
        concat!(
            "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Video_file_name,Perimeter,ChangeReference,ChangeReferenceImageName\n",
            "B,1,83,2,56,2,Test 83.szv,novel_2,,B1_novel_2\n",
            ",,83,1,56,2,Test 441.szv,training_2,,B1_training_2\n",
        ),
    )
    .unwrap();

    let mut config = imen_nort_config_with_rule("**/T1/**", DetectionRecipe::NortTraining);
    config.expand_for_validation(&dir, &[]).unwrap();
    let experiment = &config.experiments[0];

    assert_eq!(
        experiment.detection.as_ref().and_then(|d| d.recipe),
        Some(DetectionRecipe::NortTraining)
    );
    assert_eq!(
        experiment
            .session_context
            .as_ref()
            .and_then(|context| context.animal.as_deref()),
        Some("56")
    );
    assert_eq!(
        experiment
            .session_context
            .as_ref()
            .and_then(|context| context.stage.as_deref()),
        Some("1")
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn videomap_pose_data_dir_errors_when_mirrored_h5_is_missing() {
    let dir = std::env::temp_dir().join(format!(
        "bekiper_videomap_pose_missing_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("videos")).unwrap();
    std::fs::write(dir.join("videos/trial.mp4"), b"video").unwrap();

    let mut config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "pose_data"
        data_directory = "."

        [[ingress_profile]]
        profile = "videomap"

        [[ingress_profile.tasks]]
        video_dir = "videos"
        pose_data_dir = "deeplabcut/poses"
        task_type = "open_field"
        meters_per_pixel = 0.001
        recipe = "oft"
        "#,
    )
    .unwrap();

    let err = config.expand_for_validation(&dir, &[]).unwrap_err();
    assert!(
        err.to_string().contains("missing mirrored pose file"),
        "{err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
