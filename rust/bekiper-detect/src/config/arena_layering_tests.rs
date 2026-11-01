use super::*;

#[test]
fn center_rule_environment_shadows_malformed_legacy_dimensions() {
    let lookup = |key: &str| match key {
        "BEKIPER_ARENA_CENTER_RULE" => Some(r#"{"method":"manual"}"#.into()),
        "BEKIPER_ARENA_CENTER_ZONE_METERS" => Some("obsolete-invalid-value".into()),
        _ => None,
    };
    assert_eq!(
        ArenaCalibrationConfig::resolve_from(None, None, lookup)
            .unwrap()
            .effective_center_rule(),
        ArenaCenterRule::Manual {}
    );
    // Without the overriding rule, the malformed active legacy value fails.
    assert!(
        ArenaCalibrationConfig::resolve_from(None, None, |key| {
            (key == "BEKIPER_ARENA_CENTER_ZONE_METERS").then(|| "obsolete-invalid-value".into())
        })
        .is_err()
    );
}

#[test]
fn center_rule_layers_preserve_legacy_dimensions_and_allow_explicit_manual() {
    let file: ArenaCalibrationConfig = serde_json::from_value(serde_json::json!({
        "center_zone_meters":[0.20,0.20],
        "center_zone_rule":{"method":"wall_inset","inset_meters":0.03}
    }))
    .unwrap();
    let env = |key: &str| {
        (key == "BEKIPER_ARENA_CENTER_RULE")
            .then(|| r#"{"method":"dimensions","width_meters":0.12,"height_meters":0.10}"#.into())
    };
    assert_eq!(
        ArenaCalibrationConfig::resolve_from(Some(file.clone()), None, env)
            .unwrap()
            .effective_center_rule(),
        ArenaCenterRule::Dimensions {
            width_meters: 0.12,
            height_meters: 0.10
        }
    );
    assert_eq!(
        ArenaCalibrationConfig::resolve_from(Some(file.clone()), None, |_| None)
            .unwrap()
            .effective_center_rule(),
        ArenaCenterRule::WallInset { inset_meters: 0.03 }
    );
    let mut explicit = file.clone();
    explicit.set_center_rule(ArenaCenterRule::default());
    assert_eq!(
        ArenaCalibrationConfig::resolve_from(Some(file), Some(explicit), env)
            .unwrap()
            .effective_center_rule(),
        ArenaCenterRule::default()
    );
    let legacy: ArenaCalibrationConfig =
        serde_json::from_value(serde_json::json!({"center_zone_meters":[0.2,0.1]})).unwrap();
    assert_eq!(
        legacy.effective_center_rule(),
        ArenaCenterRule::Dimensions {
            width_meters: 0.2,
            height_meters: 0.1
        }
    );
    assert_eq!(
        ArenaCalibrationConfig::default().effective_center_rule(),
        ArenaCenterRule::default()
    );
    for invalid in [
        r#"{"method":"unknown"}"#,
        r#"{"method":"manual","inset_meters":0.1}"#,
        r#"{"method":"wall_inset","inset_meters":0}"#,
    ] {
        assert!(
            ArenaCalibrationConfig::resolve_from(None, None, |key| (key
                == "BEKIPER_ARENA_CENTER_RULE")
                .then(|| invalid.into()))
            .is_err()
        );
    }
}

#[test]
fn request_env_file_default_precedence_uses_injected_environment() {
    let file = ArenaCalibrationConfig {
        min_edge_score: 2.0,
        ..Default::default()
    };
    let lookup = |key: &str| (key == "BEKIPER_ARENA_MIN_EDGE_SCORE").then(|| "3.0".into());
    let resolved = ArenaCalibrationConfig::resolve_from(Some(file.clone()), None, lookup).unwrap();
    assert_eq!(resolved.min_edge_score, 3.0);
    assert_eq!(resolved.center_zone_meters, None);
    let explicit = ArenaCalibrationConfig {
        min_edge_score: 4.0,
        ..file.clone()
    };
    assert_eq!(
        ArenaCalibrationConfig::resolve_from(Some(file.clone()), Some(explicit), lookup)
            .unwrap()
            .min_edge_score,
        4.0
    );
    assert_eq!(
        ArenaCalibrationConfig::resolve_from(Some(file), None, |_| None)
            .unwrap()
            .min_edge_score,
        2.0
    );
    assert_eq!(
        ArenaCalibrationConfig::resolve_from(None, None, |_| None)
            .unwrap()
            .min_edge_score,
        DEFAULT_ARENA_MIN_EDGE_SCORE
    );
    assert!(
        ArenaCalibrationConfig::resolve_from(None, None, |key| (key
            == "BEKIPER_ARENA_MIN_EDGE_SCORE")
            .then(|| "NaN".into()))
        .is_err()
    );
}

#[test]
fn pipeline_sections_default_to_the_canonical_fractions() {
    assert_eq!(
        MouseRemovalConfig::default(),
        MouseRemovalConfig {
            min_area_frac: DEFAULT_MOUSE_MIN_AREA_FRAC,
            dilate_frac: DEFAULT_MOUSE_DILATE_FRAC,
        }
    );
    assert_eq!(
        AreaFilterConfig::default(),
        AreaFilterConfig {
            min_frac: DEFAULT_AREA_MIN_FRAC,
            max_frac: DEFAULT_AREA_MAX_FRAC,
        }
    );
    let config: DetectionPipelineConfig = serde_json::from_value(serde_json::json!({
        "expected_count": 2,
        "label_strategy": {"method": "largest", "label": "arena"}
    }))
    .unwrap();
    assert_eq!(config.mouse_removal, MouseRemovalConfig::default());
    assert_eq!(config.area_filter, AreaFilterConfig::default());
    assert_eq!(config.threshold, ThresholdConfig::default());
}

#[test]
fn arena_validation_rejects_out_of_range_settings() {
    let ok = ArenaCalibrationConfig::default();
    assert!(ok.validate().is_ok());
    let cases = [
        ArenaCalibrationConfig {
            sample_positions: vec![0.5; 17],
            ..ok.clone()
        },
        ArenaCalibrationConfig {
            sample_positions: vec![1.5],
            ..ok.clone()
        },
        ArenaCalibrationConfig {
            sample_positions: vec![f64::NAN],
            ..ok.clone()
        },
        ArenaCalibrationConfig {
            search_margin_fraction: 0.34,
            ..ok.clone()
        },
        ArenaCalibrationConfig {
            smoothing_fraction: -0.01,
            ..ok.clone()
        },
        ArenaCalibrationConfig {
            max_aspect_error: -0.1,
            ..ok.clone()
        },
        ArenaCalibrationConfig {
            min_edge_score: 0.0,
            ..ok.clone()
        },
        ArenaCalibrationConfig {
            center_zone_meters: Some((0.2, 0.0)),
            ..ok.clone()
        },
        ArenaCalibrationConfig {
            center_zone_meters: Some((f64::INFINITY, 0.2)),
            ..ok.clone()
        },
    ];
    for config in cases {
        assert!(config.validate().is_err(), "{config:?}");
    }
    // Each limit is inclusive.
    let edges = ArenaCalibrationConfig {
        sample_positions: vec![0.0, 1.0],
        search_margin_fraction: 0.33,
        smoothing_fraction: 0.25,
        min_side_fraction: 1.0,
        max_aspect_error: 0.0,
        ..ok
    };
    assert!(edges.validate().is_ok());
}

#[test]
fn arena_environment_overlays_scalars_lists_and_legacy_dimensions() {
    let env = |pairs: &'static [(&'static str, &'static str)]| {
        move |key: &str| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.to_string())
        }
    };
    let config = ArenaCalibrationConfig::resolve_from(
        None,
        None,
        env(&[
            ("BEKIPER_ARENA_MIN_EDGE_SCORE", "2.5"),
            ("BEKIPER_ARENA_SAMPLE_POSITIONS", "0.2, 0.8"),
            ("BEKIPER_ARENA_CENTER_ZONE_METERS", "0.24,0.24"),
        ]),
    )
    .unwrap();
    assert_eq!(config.min_edge_score, 2.5);
    assert_eq!(config.sample_positions, vec![0.2, 0.8]);
    assert_eq!(
        config.effective_center_rule(),
        ArenaCenterRule::Dimensions {
            width_meters: 0.24,
            height_meters: 0.24
        }
    );
    let err = ArenaCalibrationConfig::resolve_from(
        None,
        None,
        env(&[("BEKIPER_ARENA_MIN_EDGE_SCORE", "high")]),
    )
    .unwrap_err();
    assert_eq!(err, "invalid BEKIPER_ARENA_MIN_EDGE_SCORE");
    for (key, value) in [
        ("BEKIPER_ARENA_SAMPLE_POSITIONS", "0.2,x"),
        ("BEKIPER_ARENA_CENTER_ZONE_METERS", "0.24"),
        ("BEKIPER_ARENA_CENTER_ZONE_METERS", "0.24,0.24,0.1"),
        ("BEKIPER_ARENA_CENTER_RULE", "{"),
        ("BEKIPER_ARENA_SEARCH_MARGIN_FRACTION", "0.5"),
    ] {
        let lookup = move |k: &str| (k == key).then(|| value.to_string());
        assert!(
            ArenaCalibrationConfig::resolve_from(None, None, lookup).is_err(),
            "{key}={value}"
        );
    }
    // Explicit values are validated too, and bypass the environment.
    let bad = ArenaCalibrationConfig {
        min_edge_score: -1.0,
        ..ArenaCalibrationConfig::default()
    };
    assert!(ArenaCalibrationConfig::resolve_from(None, Some(bad), |_| None).is_err());
}
