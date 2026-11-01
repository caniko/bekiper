use bekiper_detect::config::ArenaCenterRule;
use bekiper_ingress::config::IngressConfig;

#[test]
fn video_relative_rules_load_from_experiment_and_profile_toml_without_center_dimensions() {
    let config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "video-centers"
        data_directory = "."
        [[experiments]]
        name = "nort"
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001
        video = "nort.mp4"
        [experiments.detection]
        recipe = "nort_training"
        backend = ["manual_annotation"]
        [experiments.detection.arena_calibration]
        center_zone_rule = { method = "video_relative" }

        [[ingress_profile]]
        profile = "videomap"
        [[ingress_profile.tasks]]
        video_dir = "OFT videos"
        task_type = "open_field"
        meters_per_pixel = { method = "arena_side", meters = 0.40 }
        recipe = "oft"
        backend = "arena_calibration_cv"
        [ingress_profile.tasks.arena_calibration]
        center_zone_rule = { method = "video_relative", width_fraction = 0.6 }
        "#,
    )
    .unwrap();
    let nort = config.experiments[0]
        .detection
        .as_ref()
        .unwrap()
        .arena_calibration
        .as_ref()
        .unwrap();
    let oft = config.ingress_profile[0].tasks[0]
        .arena_calibration
        .as_ref()
        .unwrap();
    for (calibration, width) in [(nort, 0.5), (oft, 0.6)] {
        calibration.validate().unwrap();
        assert!(calibration.center_zone_meters.is_none());
        assert_eq!(
            calibration.effective_center_rule(),
            ArenaCenterRule::VideoRelative {
                width_fraction: Some(width),
                height_fraction: Some(0.5),
            }
        );
    }
}
