use super::*;

#[test]
fn wall_inset_and_dimension_rules_use_physical_scale_and_leave_a_periphery() {
    let frames = vec![arena_frame(0, 0)];
    for rule in [
        serde_json::json!({"method":"wall_inset","inset_meters":0.05}),
        serde_json::json!({"method":"dimensions","width_meters":0.30,"height_meters":0.30}),
    ] {
        let settings =
            serde_json::from_value(serde_json::json!({"center_zone_rule":rule})).unwrap();
        let result = calibrate_arena(&frames, 0.40, &settings).unwrap();
        assert_eq!(result.perimeters.len(), 2);
        let PerimeterSpec::Rectangle {
            width: arena_width, ..
        } = result.perimeters[0]
        else {
            panic!()
        };
        let PerimeterSpec::Rectangle {
            width,
            height,
            center_x,
            center_y,
            ..
        } = result.perimeters[1]
        else {
            panic!()
        };
        assert!((width - arena_width * 0.75).abs() < 1e-9);
        assert!((height - width).abs() < 1e-9);
        assert!((center_x - 60.0).abs() <= 2.0 && (center_y - 62.0).abs() <= 2.0);
    }
    for inset in [0.0, -0.01, 0.20, 0.21] {
        let settings = serde_json::from_value(
            serde_json::json!({"center_zone_rule": {"method":"wall_inset", "inset_meters":inset}}),
        )
        .unwrap();
        assert!(calibrate_arena(&frames, 0.40, &settings).is_err());
    }
}
use image::Luma;

fn arena_frame(shift_x: i32, shift_y: i32) -> GrayImage {
    let mut image = GrayImage::from_pixel(120, 120, Luma([190]));
    let left = (20 + shift_x) as u32;
    let right = (100 + shift_x) as u32;
    let top = (22 + shift_y) as u32;
    let bottom = (102 + shift_y) as u32;
    for x in left..=right {
        image.put_pixel(x, top, Luma([20]));
        image.put_pixel(x, bottom, Luma([20]));
    }
    for y in top..=bottom {
        image.put_pixel(left, y, Luma([20]));
        image.put_pixel(right, y, Luma([20]));
    }
    image
}

#[test]
fn clean_square_arena_succeeds() {
    let config = ArenaCalibrationConfig {
        center_zone_meters: Some((0.20, 0.20)),
        ..ArenaCalibrationConfig::default()
    };
    let result = calibrate_arena(&[arena_frame(0, 0)], 0.40, &config).unwrap();
    assert_eq!(result.perimeters.len(), 2);
    match &result.perimeters[0] {
        PerimeterSpec::Rectangle {
            label,
            width,
            height,
            ..
        } => {
            assert_eq!(label, "arena");
            assert!((*width - 80.0).abs() <= 2.0, "{width}");
            assert!((*height - 80.0).abs() <= 2.0, "{height}");
        }
        _ => panic!("expected arena rectangle"),
    }
}

#[test]
fn shifted_frames_converge_when_within_tolerance() {
    let result = calibrate_arena(
        &[arena_frame(0, 0), arena_frame(1, -1), arena_frame(-1, 1)],
        0.40,
        &ArenaCalibrationConfig::default(),
    )
    .unwrap();
    assert_eq!(result.perimeters.len(), 1);
}

#[test]
fn inconsistent_frames_fail() {
    let config = ArenaCalibrationConfig {
        max_center_shift_fraction: 0.01,
        ..ArenaCalibrationConfig::default()
    };
    let err = calibrate_arena(&[arena_frame(0, 0), arena_frame(5, 0)], 0.40, &config).unwrap_err();
    assert!(err.to_string().contains("disagree"));
}

#[test]
fn low_contrast_frame_fails() {
    let image = GrayImage::from_pixel(120, 120, Luma([190]));
    let err = calibrate_arena(&[image], 0.40, &ArenaCalibrationConfig::default()).unwrap_err();
    assert!(err.to_string().contains("edge") || err.to_string().contains("too small"));
}

#[test]
fn boundary_detection_needs_no_physical_scale_or_center_convention() {
    let config = ArenaCalibrationConfig {
        center_zone_meters: Some((0.24, 0.24)),
        ..Default::default()
    };
    let result = detect_arena_boundary(&[arena_frame(0, 0)], &config).unwrap();
    assert_eq!(result.perimeters.len(), 1);
    assert_eq!(result.perimeters[0].label(), "arena");
}

#[test]
fn mismatched_frame_dimensions_and_nonfinite_parameters_are_rejected() {
    let frames = [arena_frame(0, 0), GrayImage::new(240, 240)];
    assert!(detect_arena_boundary(&frames, &Default::default()).is_err());
    assert!(calibrate_arena(&[arena_frame(0, 0)], f64::NAN, &Default::default()).is_err());
    let config = ArenaCalibrationConfig {
        min_edge_score: f64::NAN,
        ..Default::default()
    };
    assert!(detect_arena_boundary(&[arena_frame(0, 0)], &config).is_err());
}

#[test]
fn center_equal_to_arena_does_not_leave_a_periphery() {
    let config = ArenaCalibrationConfig {
        center_zone_meters: Some((0.40, 0.40)),
        ..Default::default()
    };
    assert!(calibrate_arena(&[arena_frame(0, 0)], 0.40, &config).is_err());
}

/// Dark outline of the box `left..=right` x `top..=bottom` on a 120 px floor.
fn box_frame(left: u32, top: u32, right: u32, bottom: u32) -> GrayImage {
    let mut image = GrayImage::from_pixel(120, 120, Luma([190]));
    for x in left..=right {
        image.put_pixel(x, top, Luma([20]));
        image.put_pixel(x, bottom, Luma([20]));
    }
    for y in top..=bottom {
        image.put_pixel(left, y, Luma([20]));
        image.put_pixel(right, y, Luma([20]));
    }
    image
}

fn rect(width: f64, height: f64) -> PerimeterSpec {
    PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 100.0,
        center_y: 100.0,
        width,
        height,
    }
}

fn frame_centered(center_meters: f64, detect_box: bool) -> ArenaCenterRule {
    ArenaCenterRule::FrameCentered {
        arena_width_meters: 0.40,
        arena_height_meters: 0.40,
        center_width_meters: center_meters,
        center_height_meters: center_meters,
        detect_box,
    }
}

#[test]
fn center_rules_need_a_rectangular_arena_and_valid_inputs() {
    let polygon = PerimeterSpec::Polygon {
        label: "arena".into(),
        vertices: vec![(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)],
    };
    let video = ArenaCenterRule::VideoRelative {
        width_fraction: None,
        height_fraction: None,
    };
    let err = center_zone_for_arena(&polygon, None, &video).unwrap_err();
    assert!(err.to_string().contains("axis-aligned"), "{err}");
    // Manual never generates, whatever the arena.
    assert!(
        center_zone_for_arena(&polygon, None, &ArenaCenterRule::Manual {})
            .unwrap()
            .is_none()
    );
    for inset in [0.0, -0.01, f64::NAN] {
        let rule = ArenaCenterRule::WallInset {
            inset_meters: inset,
        };
        assert!(center_zone_for_arena(&rect(200.0, 200.0), 0.002, &rule).is_err());
    }
    // An 8 cm inset in a 40 cm (200 px) arena leaves the 24 cm OFT center.
    let oft = ArenaCenterRule::WallInset { inset_meters: 0.08 };
    let center = center_zone_for_arena(&rect(200.0, 200.0), 0.002, &oft)
        .unwrap()
        .unwrap();
    assert!(
        matches!(center, PerimeterSpec::Rectangle { width, height, center_x, .. }
        if (width - 120.0).abs() < 1e-9 && (height - 120.0).abs() < 1e-9 && center_x == 100.0)
    );
    // Anisotropic pixels scale each axis on its own.
    let dims = ArenaCenterRule::Dimensions {
        width_meters: 0.2,
        height_meters: 0.2,
    };
    let center = center_zone_for_arena_xy(&rect(200.0, 100.0), Some((0.002, 0.004)), &dims)
        .unwrap()
        .unwrap();
    assert!(
        matches!(center, PerimeterSpec::Rectangle { width, height, .. } if width == 100.0 && height == 50.0)
    );
    for scale in [None, Some((0.0, 0.002)), Some((0.002, f64::INFINITY))] {
        assert!(center_zone_for_arena_xy(&rect(200.0, 200.0), scale, &dims).is_err());
    }
}

#[test]
fn frame_centered_geometry_rejects_missing_scale_or_frame() {
    let rule = frame_centered(0.20, false);
    for (frame, mpp) in [
        ((300, 200), 0.0),
        ((300, 200), f64::NAN),
        ((0, 200), 0.002),
        ((300, 0), 0.002),
    ] {
        let err = frame_centered_geometry(&rule, frame, mpp).unwrap_err();
        assert!(err.to_string().contains("frame size"), "{err}");
    }
}

#[test]
fn generate_arena_detects_the_box_or_falls_back_to_the_frame_center() {
    let config = ArenaCalibrationConfig::default();
    let frames = [arena_frame(0, 0)];
    assert!(
        generate_arena(
            &ArenaCenterRule::Manual {},
            &config,
            (120, 120),
            &frames,
            None
        )
        .unwrap()
        .is_none()
    );

    let detected = generate_arena(
        &frame_centered(0.20, true),
        &config,
        (120, 120),
        &frames,
        None,
    )
    .unwrap()
    .unwrap();
    assert!(detected.fallback_reason.is_none());
    assert_eq!(detected.perimeters.len(), 2);
    // A center as large as the arena cannot leave a periphery in the detected box.
    assert!(
        generate_arena(
            &frame_centered(0.40, true),
            &config,
            (120, 120),
            &frames,
            None
        )
        .is_err()
    );

    let disabled = generate_arena(
        &frame_centered(0.20, false),
        &config,
        (120, 120),
        &frames,
        Some(0.004),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        disabled.fallback_reason.as_deref(),
        Some("box detection disabled")
    );
    assert_eq!(
        (disabled.meters_per_pixel, disabled.meters_per_pixel_y),
        (0.004, 0.004)
    );
    assert!(
        matches!(disabled.perimeters[0], PerimeterSpec::Rectangle { center_x, width, .. }
        if center_x == 60.0 && width == 100.0)
    );

    // No sampled frames: the fallback records why, and needs a scale.
    let fallback = generate_arena(
        &frame_centered(0.20, true),
        &config,
        (120, 120),
        &[],
        Some(0.004),
    )
    .unwrap()
    .unwrap();
    let reason = fallback.fallback_reason.unwrap();
    assert!(
        reason.contains("no box detected") && reason.contains("at least one sampled frame"),
        "{reason}"
    );
    let err =
        generate_arena(&frame_centered(0.20, true), &config, (120, 120), &[], None).unwrap_err();
    assert!(
        err.to_string().contains("needs the recording's scale"),
        "{err}"
    );
}

#[test]
fn box_detection_rejects_small_frames_small_boxes_and_wrong_aspect() {
    let config = ArenaCalibrationConfig::default();
    let err = detect_arena_boundary(&[GrayImage::new(15, 64)], &config).unwrap_err();
    assert!(
        err.to_string().contains("too small for arena calibration"),
        "{err}"
    );
    let strict = ArenaCalibrationConfig {
        min_side_fraction: 0.9,
        ..ArenaCalibrationConfig::default()
    };
    let err = detect_arena_boundary(&[arena_frame(0, 0)], &strict).unwrap_err();
    assert!(
        err.to_string().contains("detected arena too small"),
        "{err}"
    );
    // 80 x 56 px: aspect error 0.43 > 0.20.
    let err = detect_arena_boundary(&[box_frame(20, 30, 100, 86)], &config).unwrap_err();
    assert!(err.to_string().contains("aspect ratio rejected"), "{err}");
    let loose = ArenaCalibrationConfig {
        max_aspect_error: 0.5,
        ..ArenaCalibrationConfig::default()
    };
    assert!(detect_arena_boundary(&[box_frame(20, 30, 100, 86)], &loose).is_ok());
}

#[test]
fn box_detection_rejects_frames_that_disagree_on_size() {
    let config = ArenaCalibrationConfig {
        max_side_shift_fraction: 0.01,
        ..ArenaCalibrationConfig::default()
    };
    // Same center, 80 px vs 88 px sides.
    let frames = [box_frame(20, 22, 100, 102), box_frame(16, 18, 104, 106)];
    let err = detect_arena_boundary(&frames, &config).unwrap_err();
    assert!(err.to_string().contains("disagree on arena side"), "{err}");
    let tolerant = detect_arena_boundary(&frames, &ArenaCalibrationConfig::default()).unwrap();
    assert_eq!(tolerant.debug.candidates.len(), 2);
    assert!(tolerant.debug.messages.is_empty());
}
