#[test]
fn pipeline_skeleton_config_parses() {
    let cfg: PipelineConfig = toml::from_str(
        r#"
        [skeleton]
        preset = "mouse_top_view"
        repair = "redetect_then_interpolate"
        relative_tolerance = 0.25
        mad_multiplier = 3.5
        min_baseline_frames = 12
        "#,
    )
    .unwrap();

    let skeleton = cfg.skeleton.expect("skeleton config should parse");
    assert_eq!(
        skeleton.preset,
        bekiper_pose::skeleton::SkeletonPreset::MouseTopView
    );
    assert_eq!(skeleton.repair, SkeletonRepairMode::RedetectThenInterpolate);
    assert!((skeleton.relative_tolerance - 0.25).abs() < 1e-12);
    assert!((skeleton.mad_multiplier - 3.5).abs() < 1e-12);
    assert_eq!(skeleton.min_baseline_frames, 12);
}

#[test]
fn pipeline_skeleton_config_requires_preset() {
    let result: std::result::Result<PipelineConfig, toml::de::Error> = toml::from_str(
        r#"
        [skeleton]
        repair = "redetect_then_interpolate"
        "#,
    );

    assert!(result.is_err(), "missing preset should be rejected");
}

#[test]
fn pipeline_quality_config_parses() {
    let cfg: PipelineConfig = toml::from_str(
        r#"
        [quality]
        max_low_likelihood_fraction = 0.12
        max_null_gap_seconds = 1.5
        max_augmented_jump_count = 2
        "#,
    )
    .unwrap();

    assert!((cfg.quality.max_low_likelihood_fraction - 0.12).abs() < 1e-12);
    assert!((cfg.quality.max_null_gap_seconds - 1.5).abs() < 1e-12);
    assert_eq!(cfg.quality.max_augmented_jump_count, 2);
}

#[test]
fn pipeline_pose_label_model_config_parses() {
    let cfg: PipelineConfig = toml::from_str(
        r#"
        pose_label_model = "imen_dlc"
        "#,
    )
    .unwrap();

    assert_eq!(cfg.pose_label_model, PoseLabelModel::ImenDlc);
}

#[test]
fn pipeline_pose_label_model_defaults_to_superanimal() {
    let cfg = PipelineConfig::default();
    assert_eq!(
        cfg.pose_label_model,
        PoseLabelModel::SuperanimalTopviewmouse
    );
}

#[test]
fn pipeline_pose_label_model_rejects_unknown_value() {
    let result: std::result::Result<PipelineConfig, toml::de::Error> = toml::from_str(
        r#"
        pose_label_model = "unknown_model"
        "#,
    );

    assert!(result.is_err());
}

#[test]
fn pipeline_kalman_smoothing_config_parses() {
    let cfg: PipelineConfig = toml::from_str(
        r#"
        smoothing_method = "kalman"
        kalman_process_noise = 0.75
        kalman_measurement_noise = 3.5
        kalman_min_likelihood = 0.2
        kalman_max_gap_frames = 12
        "#,
    )
    .unwrap();

    assert_eq!(cfg.smoothing_method, SmoothingMethod::Kalman);
    assert!((cfg.kalman_process_noise - 0.75).abs() < 1e-12);
    assert!((cfg.kalman_measurement_noise - 3.5).abs() < 1e-12);
    assert!((cfg.kalman_min_likelihood - 0.2).abs() < 1e-12);
    assert_eq!(cfg.kalman_max_gap_frames, 12);
}

#[test]
fn pipeline_quality_defaults_are_conservative() {
    let cfg = PipelineConfig::default();
    assert_eq!(cfg.smoothing_method, SmoothingMethod::Median);
    assert!((cfg.kalman_process_noise - 1.0).abs() < 1e-12);
    assert!((cfg.kalman_measurement_noise - 4.0).abs() < 1e-12);
    assert!((cfg.kalman_min_likelihood - 0.05).abs() < 1e-12);
    assert_eq!(cfg.kalman_max_gap_frames, 30);
    assert!((cfg.quality.max_low_likelihood_fraction - 0.05).abs() < 1e-12);
    assert!((cfg.quality.max_null_gap_seconds - 0.5).abs() < 1e-12);
    assert_eq!(cfg.quality.max_augmented_jump_count, 0);
}

#[test]
fn aggregate_env_overlay_applies_output_and_disable_flag() {
    let mut cfg = AggregateConfig::default();
    let env = HashMap::from([
        ("BEKIPER_AGGREGATE_DISABLED", "true"),
        ("BEKIPER_AGGREGATE_OUTPUT", "renamed.xlsx"),
    ]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!(!cfg.enabled);
    assert_eq!(cfg.output_filename, "renamed.ods");
}

#[test]
fn pipeline_env_overlay_applies_invert_y() {
    let mut cfg = PipelineConfig::default();
    let env = HashMap::from([("BEKIPER_PIPELINE_INVERT_Y", "yes")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!(cfg.invert_y);
}

#[test]
fn pipeline_env_overlay_applies_pose_label_model() {
    let mut cfg = PipelineConfig::default();
    let env = HashMap::from([("BEKIPER_PIPELINE_POSE_LABEL_MODEL", "imen_dlc")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert_eq!(cfg.pose_label_model, PoseLabelModel::ImenDlc);
}

#[test]
fn pipeline_env_overlay_applies_kalman_smoothing() {
    let mut cfg = PipelineConfig::default();
    let env = HashMap::from([
        ("BEKIPER_PIPELINE_SMOOTHING_METHOD", "kalman"),
        ("BEKIPER_PIPELINE_KALMAN_PROCESS_NOISE", "0.5"),
        ("BEKIPER_PIPELINE_KALMAN_MEASUREMENT_NOISE", "3.25"),
        ("BEKIPER_PIPELINE_KALMAN_MIN_LIKELIHOOD", "0.15"),
        ("BEKIPER_PIPELINE_KALMAN_MAX_GAP_FRAMES", "17"),
    ]);

    cfg.apply_env_overrides_from(env_lookup(&env));

    assert_eq!(cfg.smoothing_method, SmoothingMethod::Kalman);
    assert!((cfg.kalman_process_noise - 0.5).abs() < 1e-12);
    assert!((cfg.kalman_measurement_noise - 3.25).abs() < 1e-12);
    assert!((cfg.kalman_min_likelihood - 0.15).abs() < 1e-12);
    assert_eq!(cfg.kalman_max_gap_frames, 17);
}

#[test]
fn pipeline_env_overlay_ignores_invalid_kalman_values() {
    let mut cfg = PipelineConfig::default();
    let env = HashMap::from([
        ("BEKIPER_PIPELINE_SMOOTHING_METHOD", "spline"),
        ("BEKIPER_PIPELINE_KALMAN_PROCESS_NOISE", "0"),
        ("BEKIPER_PIPELINE_KALMAN_MEASUREMENT_NOISE", "-1"),
        ("BEKIPER_PIPELINE_KALMAN_MIN_LIKELIHOOD", "2.0"),
        ("BEKIPER_PIPELINE_KALMAN_MAX_GAP_FRAMES", "many"),
    ]);

    cfg.apply_env_overrides_from(env_lookup(&env));

    assert_eq!(cfg.smoothing_method, SmoothingMethod::Median);
    assert!((cfg.kalman_process_noise - 1.0).abs() < 1e-12);
    assert!((cfg.kalman_measurement_noise - 4.0).abs() < 1e-12);
    assert!((cfg.kalman_min_likelihood - 0.05).abs() < 1e-12);
    assert_eq!(cfg.kalman_max_gap_frames, 30);
}

#[test]
fn pipeline_env_overlay_ignores_invalid_boolean() {
    let mut cfg = PipelineConfig::default();
    let env = HashMap::from([("BEKIPER_PIPELINE_INVERT_Y", "sideways")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!(!cfg.invert_y);
}
