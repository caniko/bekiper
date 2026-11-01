use bekiper_core::inspection::PerimeterSpec;
use bekiper_detect::{
    arena::{calibrate_arena, center_zone_for_arena},
    config::{ArenaCalibrationConfig, ArenaCenterRule},
};
use image::{GrayImage, Luma};

fn rule(value: serde_json::Value) -> ArenaCenterRule {
    serde_json::from_value(value).unwrap()
}

fn arena() -> PerimeterSpec {
    PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 180.0,
        center_y: 120.0,
        width: 200.0,
        height: 160.0,
    }
}

#[test]
fn video_relative_center_needs_no_physical_scale_and_defaults_to_central_half() {
    let center = center_zone_for_arena(
        &arena(),
        None,
        &rule(serde_json::json!({"method":"video_relative"})),
    )
    .unwrap()
    .unwrap();
    assert!(matches!(
        center,
        PerimeterSpec::Rectangle {
            center_x: 180.0,
            center_y: 120.0,
            width: 100.0,
            height: 80.0,
            ..
        }
    ));
}

#[test]
fn video_relative_center_uses_independent_arena_fractions_and_ignores_scale() {
    let rule = rule(serde_json::json!({
        "method":"video_relative", "width_fraction":0.6, "height_fraction":0.25
    }));
    for scale in [f64::NAN, 0.0, 0.001, 0.003] {
        let center = center_zone_for_arena(&arena(), scale, &rule)
            .unwrap()
            .unwrap();
        assert!(matches!(
            center,
            PerimeterSpec::Rectangle {
                center_x: 180.0,
                center_y: 120.0,
                width: 120.0,
                height: 40.0,
                ..
            }
        ));
    }
}

#[test]
fn video_relative_center_tracks_detected_arena_rather_than_frame_midpoint() {
    let mut frame = GrayImage::from_pixel(240, 240, Luma([190]));
    for x in 60..=220 {
        frame.put_pixel(x, 40, Luma([20]));
        frame.put_pixel(x, 200, Luma([20]));
    }
    for y in 40..=200 {
        frame.put_pixel(60, y, Luma([20]));
        frame.put_pixel(220, y, Luma([20]));
    }
    let config = serde_json::from_value(serde_json::json!({
        "center_zone_rule":{"method":"video_relative"}
    }))
    .unwrap();
    let result = calibrate_arena(&[frame], 0.40, &config).unwrap();
    let [
        PerimeterSpec::Rectangle {
            center_x: ax,
            center_y: ay,
            width: aw,
            height: ah,
            ..
        },
        PerimeterSpec::Rectangle {
            center_x: cx,
            center_y: cy,
            width: cw,
            height: ch,
            ..
        },
    ] = result.perimeters.as_slice()
    else {
        panic!("expected arena and center rectangles");
    };
    assert!(*ax > 130.0 && *ax < 150.0, "detected arena center: {ax}");
    assert_eq!((cx, cy), (ax, ay));
    assert_eq!((*cw, *ch), (*aw / 2.0, *ah / 2.0));
}

#[test]
fn video_relative_fractions_must_leave_a_nonempty_center_and_periphery() {
    for value in [-0.1, 0.0, 1.0, 1.1] {
        for field in ["width_fraction", "height_fraction"] {
            let center_rule = rule(serde_json::json!({"method":"video_relative", (field):value}));
            let config = ArenaCalibrationConfig {
                center_zone_rule: Some(center_rule.clone()),
                ..Default::default()
            };
            assert!(config.validate().is_err());
            assert!(center_zone_for_arena(&arena(), 0.001, &center_rule).is_err());
        }
    }
    for value in [f64::NAN, f64::INFINITY] {
        let center_rule = ArenaCenterRule::VideoRelative {
            width_fraction: Some(value),
            height_fraction: None,
        };
        let config = ArenaCalibrationConfig {
            center_zone_rule: Some(center_rule.clone()),
            ..Default::default()
        };
        assert!(config.validate().is_err());
        assert!(center_zone_for_arena(&arena(), None, &center_rule).is_err());
    }
}

#[test]
fn physical_center_rules_still_require_scale() {
    let physical = rule(serde_json::json!({"method":"wall_inset","inset_meters":0.02}));
    assert!(center_zone_for_arena(&arena(), None, &physical).is_err());
}

#[test]
fn video_relative_rule_obeys_request_environment_file_precedence() {
    let file: ArenaCalibrationConfig = serde_json::from_value(serde_json::json!({
        "center_zone_rule":{"method":"video_relative", "width_fraction":0.4}
    }))
    .unwrap();
    let lookup = |key: &str| {
        (key == "BEKIPER_ARENA_CENTER_RULE")
            .then(|| r#"{"method":"video_relative","width_fraction":0.7}"#.into())
    };
    let explicit: ArenaCalibrationConfig = serde_json::from_value(serde_json::json!({
        "center_zone_rule":{"method":"video_relative", "width_fraction":0.6}
    }))
    .unwrap();
    for (explicit, use_env, expected_width) in [
        (None, false, 80.0),
        (None, true, 140.0),
        (Some(explicit), true, 120.0),
    ] {
        let config = ArenaCalibrationConfig::resolve_from(Some(file.clone()), explicit, |key| {
            if use_env { lookup(key) } else { None }
        })
        .unwrap();
        let center = center_zone_for_arena(&arena(), f64::NAN, &config.effective_center_rule())
            .unwrap()
            .unwrap();
        assert!(
            matches!(center, PerimeterSpec::Rectangle { width, height:80.0, .. } if width == expected_width)
        );
    }
}

#[test]
fn frame_centered_geometry_uses_the_frame_center_and_physical_sizes() {
    use bekiper_detect::arena::frame_centered_geometry;
    let rule = ArenaCenterRule::FrameCentered {
        arena_width_meters: 0.40,
        arena_height_meters: 0.40,
        center_width_meters: 0.20,
        center_height_meters: 0.20,
        detect_box: false,
    };
    let geometry = frame_centered_geometry(&rule, (300, 200), 0.002).unwrap();
    let rect = |index: usize| match &geometry[index] {
        PerimeterSpec::Rectangle {
            label,
            center_x,
            center_y,
            width,
            height,
        } => (label.as_str(), *center_x, *center_y, *width, *height),
        other => panic!("unexpected {other:?}"),
    };
    assert_eq!(rect(0), ("arena", 150.0, 100.0, 200.0, 200.0));
    assert_eq!(rect(1), ("center_zone", 150.0, 100.0, 100.0, 100.0));
    // An arena larger than the frame means the scale is wrong.
    assert!(frame_centered_geometry(&rule, (150, 200), 0.002).is_err());
    // Other rules generate nothing.
    assert!(
        frame_centered_geometry(&ArenaCenterRule::Manual {}, (300, 200), 0.002)
            .unwrap()
            .is_empty()
    );
    // Validation rejects a center that is not smaller than the arena.
    let mut config = bekiper_detect::config::ArenaCalibrationConfig::default();
    config.set_center_rule(ArenaCenterRule::FrameCentered {
        arena_width_meters: 0.40,
        arena_height_meters: 0.40,
        center_width_meters: 0.40,
        center_height_meters: 0.20,
        detect_box: false,
    });
    assert!(config.validate().is_err());
}

#[test]
fn detected_box_defines_arena_scale_and_center_with_frame_center_fallback() {
    use bekiper_detect::arena::generate_arena;
    let rule = ArenaCenterRule::FrameCentered {
        arena_width_meters: 0.40,
        arena_height_meters: 0.40,
        center_width_meters: 0.20,
        center_height_meters: 0.20,
        detect_box: true,
    };
    let mut calibration = ArenaCalibrationConfig::nort_default();
    calibration.set_center_rule(rule.clone());
    // 232x216 frame, dark walls around a lighter floor that is off-center.
    let mut frame = GrayImage::from_pixel(232, 216, Luma([20]));
    for x in 20..220 {
        for y in 10..210 {
            frame.put_pixel(x, y, Luma([190]));
        }
    }
    let detected = generate_arena(&rule, &calibration, (232, 216), &[frame], None)
        .unwrap()
        .unwrap();
    assert!(detected.fallback_reason.is_none());
    let [
        PerimeterSpec::Rectangle {
            center_x: ax,
            center_y: ay,
            width: aw,
            height: ah,
            ..
        },
        PerimeterSpec::Rectangle {
            center_x: cx,
            center_y: cy,
            width: cw,
            ..
        },
    ] = detected.perimeters.as_slice()
    else {
        panic!("expected arena and center rectangles");
    };
    // Floor center is (119.5, 109.5); the edge-projection detector is ~1% off.
    assert!(
        (ax - 119.5).abs() <= 3.0 && (ay - 109.5).abs() <= 3.0,
        "box center {ax},{ay}"
    );
    assert!(*ax > 116.0 && *ay > 108.0, "must not be the frame center");
    // The zone sits on the box center, not the frame center (116, 108).
    assert_eq!((cx, cy), (ax, ay));
    // Each axis gets its own scale; the 20 cm center spans half of each side.
    assert!((detected.meters_per_pixel - 0.40 / aw).abs() < 1e-12);
    assert!((detected.meters_per_pixel_y - 0.40 / ah).abs() < 1e-12);
    assert!((cw - aw / 2.0).abs() < 1e-9);
    let PerimeterSpec::Rectangle {
        height: center_h, ..
    } = &detected.perimeters[1]
    else {
        panic!("expected center rectangle");
    };
    assert!((center_h - ah / 2.0).abs() < 1e-9);

    // No box: centered on the frame with the recording's scale, and the
    // reason is retained; without any scale the fallback fails loudly.
    let blank = GrayImage::from_pixel(232, 216, Luma([120]));
    let fallback = generate_arena(
        &rule,
        &calibration,
        (232, 216),
        std::slice::from_ref(&blank),
        Some(0.002),
    )
    .unwrap()
    .unwrap();
    assert!(
        fallback
            .fallback_reason
            .unwrap()
            .contains("no box detected")
    );
    let PerimeterSpec::Rectangle {
        center_x,
        center_y,
        width,
        ..
    } = &fallback.perimeters[0]
    else {
        panic!("expected arena rectangle");
    };
    assert_eq!((*center_x, *center_y, *width), (116.0, 108.0, 200.0));
    assert!(generate_arena(&rule, &calibration, (232, 216), &[blank], None).is_err());
}
