#[test]
fn variants_round_trip() {
    let config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "variants"
        data_directory = "."

        [[scan]]
        video_dir = "Y-maze videos"
        task_type = "y_maze"
        meters_per_pixel = { method = "arm_length", meters = 0.35 }

        [[scan.variants]]
        name = "proximal65"
        [scan.variants.ymaze]
        proximal_threshold = 0.65

        [[scan.variants]]
        name = "proximal45"
        [scan.variants.ymaze]
        proximal_threshold = 0.45
        "#,
    )
    .unwrap();

    assert_eq!(config.scan.len(), 1);
    let variants = &config.scan[0].variants;
    assert_eq!(variants.len(), 2);
    assert_eq!(variants[0].name, "proximal65");
    assert_eq!(
        variants[0]
            .task_params
            .ymaze
            .as_ref()
            .and_then(|params| params.proximal_threshold),
        Some(0.65)
    );
    assert_eq!(variants[1].name, "proximal45");
    assert_eq!(
        variants[1]
            .task_params
            .ymaze
            .as_ref()
            .and_then(|params| params.proximal_threshold),
        Some(0.45)
    );
}

#[test]
fn merged_overrides_variant_ymaze_field_wins() {
    let task = TaskParamsOverrides {
        ymaze: Some(YMazeOverrides {
            proximal_threshold: Some(0.25),
            arm_membership_hysteresis: Some(true),
            zone_max_gap_seconds: Some(0.1),
            pose_gap_fill_seconds: Some(0.2),
            ..Default::default()
        }),
        ..Default::default()
    };
    let variant = TaskParamsOverrides {
        ymaze: Some(YMazeOverrides {
            proximal_threshold: Some(0.45),
            arm_membership_hysteresis: Some(false),
            zone_max_gap_seconds: Some(0.2),
            pose_gap_fill_seconds: Some(0.3),
            ..Default::default()
        }),
        ..Default::default()
    };

    let merged = task.merged(&variant);
    let ymaze = merged.ymaze.unwrap();
    assert_eq!(ymaze.proximal_threshold, Some(0.45));
    assert_eq!(ymaze.arm_membership_hysteresis, Some(false));
    assert_eq!(ymaze.zone_max_gap_seconds, Some(0.2));
    assert_eq!(ymaze.pose_gap_fill_seconds, Some(0.3));
}

#[test]
fn merged_overrides_variant_shared_field_wins() {
    let task = TaskParamsOverrides {
        freezing_displacement_threshold: Some(0.003),
        min_dwell_seconds: Some(0.5),
        ..Default::default()
    };
    let variant = TaskParamsOverrides {
        freezing_displacement_threshold: Some(0.007),
        ..Default::default()
    };

    let merged = task.merged(&variant);
    assert_eq!(merged.freezing_displacement_threshold, Some(0.007));
    assert_eq!(merged.min_dwell_seconds, Some(0.5));
}

#[test]
fn expand_scans_fans_out_variants_and_merges_task_params() {
    let dir = std::env::temp_dir().join(format!(
        "bekiper_expand_scan_variants_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("Y-maze videos")).unwrap();
    std::fs::write(dir.join("Y-maze videos/video-a.mp4"), b"video").unwrap();
    std::fs::write(dir.join("Y-maze videos/video-b.mp4"), b"video").unwrap();

    let mut config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "variants"
        data_directory = "."

        [[scan]]
        video_dir = "Y-maze videos"
        task_type = "y_maze"
        meters_per_pixel = 0.001
        recipe = "ymaze"

        [scan.task_params.ymaze]
        start_mode = "video_start"
        arm_membership_hysteresis = true
        zone_max_gap_seconds = 0.1
        pose_gap_fill_seconds = 0.2

        [[scan.variants]]
        name = "proximal65"
        [scan.variants.ymaze]
        proximal_threshold = 0.65

        [[scan.variants]]
        name = "proximal45"
        [scan.variants.ymaze]
        proximal_threshold = 0.45
        arm_membership_hysteresis = false
        zone_max_gap_seconds = 0.2
        pose_gap_fill_seconds = 0.3
        "#,
    )
    .unwrap();

    config
        .expand_for_validation(&dir, crate::stage::Stage::all())
        .unwrap();

    assert_eq!(config.experiments.len(), 4);
    let names: Vec<_> = config
        .experiments
        .iter()
        .map(|experiment| experiment.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec![
            "y_maze_videos_video_a__proximal65",
            "y_maze_videos_video_a__proximal45",
            "y_maze_videos_video_b__proximal65",
            "y_maze_videos_video_b__proximal45",
        ]
    );

    let proximal65 = config
        .experiments
        .iter()
        .find(|experiment| experiment.name == "y_maze_videos_video_a__proximal65")
        .unwrap()
        .task_params
        .as_ref()
        .and_then(|params| params.ymaze.as_ref())
        .unwrap();
    assert_eq!(proximal65.proximal_threshold, Some(0.65));
    assert_eq!(proximal65.arm_membership_hysteresis, Some(true));
    assert_eq!(proximal65.zone_max_gap_seconds, Some(0.1));
    assert_eq!(proximal65.pose_gap_fill_seconds, Some(0.2));
    assert_eq!(
        proximal65.start_mode,
        Some(bekiper_behaviour::radial_arm::YMazeStartMode::VideoStart)
    );

    let proximal45 = config
        .experiments
        .iter()
        .find(|experiment| experiment.name == "y_maze_videos_video_a__proximal45")
        .unwrap()
        .task_params
        .as_ref()
        .and_then(|params| params.ymaze.as_ref())
        .unwrap();
    assert_eq!(proximal45.proximal_threshold, Some(0.45));
    assert_eq!(proximal45.arm_membership_hysteresis, Some(false));
    assert_eq!(proximal45.zone_max_gap_seconds, Some(0.2));
    assert_eq!(proximal45.pose_gap_fill_seconds, Some(0.3));
    assert_eq!(
        proximal45.start_mode,
        Some(bekiper_behaviour::radial_arm::YMazeStartMode::VideoStart)
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn expand_videomap_profile_fans_out_variants_and_merges_task_params() {
    let dir = std::env::temp_dir().join(format!(
        "bekiper_expand_videomap_variants_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("videos")).unwrap();
    std::fs::write(dir.join("videos/trial.mp4"), b"video").unwrap();

    let mut config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "variants"
        data_directory = "."

        [[ingress_profile]]
        profile = "videomap"

        [[ingress_profile.tasks]]
        video_dir = "videos"
        task_type = "y_maze"
        meters_per_pixel = 0.001
        recipe = "ymaze"

        [ingress_profile.tasks.task_params.ymaze]
        arm_membership_hysteresis = true
        zone_max_gap_seconds = 0.1
        pose_gap_fill_seconds = 0.2

        [[ingress_profile.tasks.variants]]
        name = "proximal65"
        [ingress_profile.tasks.variants.ymaze]
        proximal_threshold = 0.65

        [[ingress_profile.tasks.variants]]
        name = "proximal45"
        [ingress_profile.tasks.variants.ymaze]
        proximal_threshold = 0.45
        arm_membership_hysteresis = false
        zone_max_gap_seconds = 0.2
        pose_gap_fill_seconds = 0.3
        "#,
    )
    .unwrap();

    config
        .expand_for_validation(&dir, crate::stage::Stage::all())
        .unwrap();

    assert_eq!(config.experiments.len(), 2);
    assert_eq!(config.experiments[0].name, "videos_trial__proximal65");
    assert_eq!(config.experiments[1].name, "videos_trial__proximal45");
    for experiment in &config.experiments {
        let ymaze = experiment
            .task_params
            .as_ref()
            .and_then(|params| params.ymaze.as_ref())
            .unwrap();
        let expected_gap = if experiment.name.ends_with("__proximal65") {
            Some(0.1)
        } else {
            Some(0.2)
        };
        let expected_pose_gap_fill = if experiment.name.ends_with("__proximal65") {
            Some(0.2)
        } else {
            Some(0.3)
        };
        let expected_hysteresis = if experiment.name.ends_with("__proximal65") {
            Some(true)
        } else {
            Some(false)
        };
        assert_eq!(ymaze.arm_membership_hysteresis, expected_hysteresis);
        assert_eq!(ymaze.zone_max_gap_seconds, expected_gap);
        assert_eq!(ymaze.pose_gap_fill_seconds, expected_pose_gap_fill);
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn validation_rejects_empty_variant_name() {
    let config = config_with_scan_variants(
        r#"
        [[scan.variants]]
        name = ""
        "#,
    );

    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("empty variant name"));
}

#[test]
fn validation_rejects_duplicate_variant_name() {
    let config = config_with_scan_variants(
        r#"
        [[scan.variants]]
        name = "proximal65"

        [[scan.variants]]
        name = "proximal65"
        "#,
    );

    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("duplicate variant name"));
    assert!(err.contains("proximal65"));
}

#[test]
fn validation_rejects_invalid_variant_slug() {
    let config = config_with_scan_variants(
        r#"
        [[scan.variants]]
        name = "Proximal 65"
        "#,
    );

    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("invalid variant name"));
    assert!(err.contains("Proximal 65"));
    assert!(err.contains("proximal65"));
}
