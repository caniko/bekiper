use super::*;

#[test]
fn manual_resolves_directly() {
    let cfg = ScaleConfig::Manual(0.001914);
    let mpp = cfg.resolve(&[]).unwrap();
    assert!((mpp.0 - 0.001914).abs() < 1e-9);
}

#[test]
fn arena_side_from_rectangle() {
    let cfg = ScaleConfig::Derived(ScaleMethod::ArenaSide {
        meters: 0.40,
        square: false,
    });
    let perimeters = vec![PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 320.0,
        center_y: 240.0,
        width: 209.0,
        height: 209.0,
    }];

    let mpp = cfg.resolve(&perimeters).unwrap();
    assert!((mpp.0 - 0.40 / 209.0).abs() < 1e-9);
}

#[test]
fn arena_side_from_polygon() {
    let cfg = ScaleConfig::Derived(ScaleMethod::ArenaSide {
        meters: 0.40,
        square: false,
    });
    let perimeters = vec![PerimeterSpec::Polygon {
        label: "arena".into(),
        vertices: vec![(10.0, 10.0), (210.0, 10.0), (210.0, 210.0), (10.0, 210.0)],
    }];

    let mpp = cfg.resolve(&perimeters).unwrap();
    assert!((mpp.0 - 0.002).abs() < 1e-9);
}

#[test]
fn arena_side_missing_label() {
    let cfg = ScaleConfig::Derived(ScaleMethod::ArenaSide {
        meters: 0.40,
        square: false,
    });
    let perimeters = vec![PerimeterSpec::Rectangle {
        label: "wrong".into(),
        center_x: 0.0,
        center_y: 0.0,
        width: 100.0,
        height: 100.0,
    }];

    assert!(cfg.resolve(&perimeters).is_err());
}

#[test]
fn arm_length_from_radial_maze() {
    let cfg = ScaleConfig::Derived(ScaleMethod::ArmLength { meters: 0.35 });
    let arm_poly = |dx: f64, dy: f64| -> Vec<(f64, f64)> {
        vec![
            (100.0 + dx, 100.0 + dy),
            (100.0 + dx + 10.0, 100.0 + dy),
            (100.0 + dx + 10.0, 100.0 + dy + 175.0),
            (100.0 + dx, 100.0 + dy + 175.0),
        ]
    };
    let perimeters = vec![PerimeterSpec::RadialMaze {
        label: "ymaze".into(),
        center_vertices: vec![(95.0, 95.0), (115.0, 95.0), (115.0, 115.0), (95.0, 115.0)],
        arms: vec![
            arm_poly(0.0, 0.0),
            arm_poly(50.0, 0.0),
            arm_poly(-50.0, 0.0),
        ],
    }];

    let mpp = cfg.resolve(&perimeters).unwrap();
    assert!((mpp.0 - 0.35 / 175.0).abs() < 1e-6);
}

#[test]
fn arm_length_no_radial_maze() {
    let cfg = ScaleConfig::Derived(ScaleMethod::ArmLength { meters: 0.35 });
    assert!(cfg.resolve(&[]).is_err());
}

/// Helper: deserialize `ScaleConfig` from a `key = value` TOML string.
fn parse_scale(toml_str: &str) -> ScaleConfig {
    #[derive(Deserialize)]
    struct Wrapper {
        scale: ScaleConfig,
    }
    let w: Wrapper = toml::from_str(toml_str).unwrap();
    w.scale
}

#[test]
fn serde_manual_roundtrip() {
    let cfg = parse_scale("scale = 0.001914");
    assert!(matches!(cfg, ScaleConfig::Manual(v) if (v - 0.001914).abs() < 1e-9));
}

#[test]
fn serde_arena_side_roundtrip() {
    let cfg = parse_scale(r#"scale = { method = "arena_side", meters = 0.40 }"#);
    assert!(
        matches!(cfg, ScaleConfig::Derived(ScaleMethod::ArenaSide { meters, square: false }) if (meters - 0.40).abs() < 1e-9)
    );
}

#[test]
fn serde_arm_length_roundtrip() {
    let cfg = parse_scale(r#"scale = { method = "arm_length", meters = 0.35 }"#);
    assert!(
        matches!(cfg, ScaleConfig::Derived(ScaleMethod::ArmLength { meters }) if (meters - 0.35).abs() < 1e-9)
    );
}

fn arena_circle(radius: f64) -> PerimeterSpec {
    PerimeterSpec::Circle {
        label: "arena".into(),
        center_x: 0.0,
        center_y: 0.0,
        radius,
    }
}

fn maze(arms: Vec<Vec<(f64, f64)>>) -> Vec<PerimeterSpec> {
    vec![
        arena_circle(1.0),
        PerimeterSpec::RadialMaze {
            label: "ymaze".into(),
            center_vertices: vec![],
            arms,
        },
    ]
}

#[test]
fn manual_info_has_no_derivation_inputs() {
    let (mpp, info) = ScaleConfig::Manual(0.002).resolve_with_info(&[]).unwrap();
    assert_eq!(mpp.0, 0.002);
    assert_eq!(info.method, "manual");
    assert!(info.physical_meters.is_none() && info.pixel_measurement.is_none());
    assert!(info.reference_label.is_none() && info.arm_pixel_spans.is_empty());
    assert_eq!(info.meters_per_pixel_y(), 0.002);
}

#[test]
fn arena_side_uses_circle_diameter_and_shorter_rectangle_side() {
    let cfg = ScaleConfig::Derived(ScaleMethod::ArenaSide {
        meters: 0.40,
        square: false,
    });
    let (mpp, info) = cfg.resolve_with_info(&[arena_circle(100.0)]).unwrap();
    assert_eq!(mpp.0, 0.002);
    assert_eq!(info.method, "arena_side");
    assert_eq!(info.physical_meters, Some(0.40));
    assert_eq!(info.pixel_measurement, Some(200.0));
    assert_eq!(info.reference_label.as_deref(), Some("arena"));
    let rect = PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 0.0,
        center_y: 0.0,
        width: 204.0,
        height: 200.0,
    };
    assert_eq!(
        cfg.resolve_with_info(&[rect]).unwrap().1.pixel_measurement,
        Some(200.0)
    );
}

#[test]
fn arena_side_rejects_degenerate_arenas() {
    let cfg = ScaleConfig::Derived(ScaleMethod::ArenaSide {
        meters: 0.40,
        square: false,
    });
    let err = cfg.resolve(&[arena_circle(0.0)]).unwrap_err().to_string();
    assert!(err.contains("zero or negative"), "{err}");
    assert!(cfg.resolve_with_info(&[arena_circle(-1.0)]).is_err());
    // A flat polygon has zero extent; an empty one has none and is skipped.
    let flat = PerimeterSpec::Polygon {
        label: "arena".into(),
        vertices: vec![(0.0, 5.0), (10.0, 5.0), (20.0, 5.0)],
    };
    assert!(cfg.resolve(&[flat]).is_err());
    let empty = PerimeterSpec::Polygon {
        label: "arena".into(),
        vertices: vec![],
    };
    let err = cfg.resolve(&[empty]).unwrap_err().to_string();
    assert!(err.contains("labeled 'arena'"), "{err}");
}

#[test]
fn arm_length_info_reports_each_arm_span() {
    let cfg = ScaleConfig::Derived(ScaleMethod::ArmLength { meters: 0.35 });
    // A legacy two-point arm line, a quadrilateral and a pentagon.
    let perimeters = maze(vec![
        vec![(0.0, 0.0), (0.0, 175.0)],
        vec![(0.0, 0.0), (10.0, 0.0), (10.0, 175.0), (0.0, 175.0)],
        vec![
            (0.0, 0.0),
            (3.0, 4.0),
            (0.0, 175.0),
            (-3.0, 4.0),
            (-1.0, 1.0),
        ],
    ]);
    let (mpp, info) = cfg.resolve_with_info(&perimeters).unwrap();
    assert_eq!(info.method, "arm_length");
    assert_eq!(info.arm_pixel_spans, vec![175.0, 175.0, 175.0]);
    assert_eq!(info.pixel_measurement, Some(175.0));
    assert_eq!(info.reference_label.as_deref(), Some("ymaze"));
    assert_eq!(mpp.0, 0.35 / 175.0);
    assert_eq!(cfg.resolve(&perimeters).unwrap().0, mpp.0);
}

#[test]
fn arm_length_rejects_missing_or_degenerate_arms() {
    let cfg = ScaleConfig::Derived(ScaleMethod::ArmLength { meters: 0.35 });
    let err = cfg.resolve(&maze(vec![])).unwrap_err().to_string();
    assert!(err.contains("no arms"), "{err}");
    let err = cfg
        .resolve_with_info(&maze(vec![vec![], vec![(1.0, 1.0)]]))
        .unwrap_err()
        .to_string();
    assert!(err.contains("zero pixel extent"), "{err}");
    assert!(cfg.resolve_with_info(&[arena_circle(1.0)]).is_err());
}

#[test]
fn square_arena_side_scales_each_axis_and_defaults_to_the_short_side() {
    let rect = |width: f64, height: f64| {
        vec![PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 112.0,
            center_y: 112.0,
            width,
            height,
        }]
    };
    let square = parse_scale(r#"scale = { method = "arena_side", meters = 0.40, square = true }"#);
    // A square 40 cm floor recorded as 204 x 210 px: one scale per axis.
    let (mpp, info) = square.resolve_with_info(&rect(204.0, 210.0)).unwrap();
    assert_eq!(mpp.0, 0.40 / 204.0);
    assert_eq!(info.meters_per_pixel_y, Some(0.40 / 210.0));
    assert_eq!(info.pixel_measurement, Some(204.0));
    assert_eq!(square.resolve(&rect(204.0, 210.0)).unwrap().0, 0.40 / 204.0);
    // Square pixels stay isotropic (byte-identical files).
    let (_, info) = square.resolve_with_info(&rect(204.0, 204.0)).unwrap();
    assert_eq!(info.meters_per_pixel_y, None);
    // Circles stay isotropic.
    let circle = vec![PerimeterSpec::Circle {
        label: "arena".into(),
        center_x: 0.0,
        center_y: 0.0,
        radius: 100.0,
    }];
    let (mpp, info) = square.resolve_with_info(&circle).unwrap();
    assert_eq!((mpp.0, info.meters_per_pixel_y), (0.40 / 200.0, None));
    // Without `square` a rectangular arena keeps the short side.
    let plain = parse_scale(r#"scale = { method = "arena_side", meters = 0.40 }"#);
    let (mpp, info) = plain.resolve_with_info(&rect(204.0, 300.0)).unwrap();
    assert_eq!((mpp.0, info.meters_per_pixel_y), (0.40 / 204.0, None));
    // The flag is omitted when false so existing configs serialise unchanged.
    let ScaleConfig::Derived(method) = plain else {
        unreachable!()
    };
    assert!(!toml::to_string(&method).unwrap().contains("square"));
}

#[test]
fn square_oft_box_gives_a_24cm_center_and_8cm_band_on_both_axes() {
    // OFT: detected 204 x 210 px floor, center = 0.6 of each side (pixels).
    let arena = PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 112.0,
        center_y: 112.0,
        width: 204.0,
        height: 210.0,
    };
    let center = PerimeterSpec::Rectangle {
        label: "center_zone".into(),
        center_x: 112.0,
        center_y: 112.0,
        width: 0.6 * 204.0,
        height: 0.6 * 210.0,
    };
    let square = parse_scale(r#"scale = { method = "arena_side", meters = 0.40, square = true }"#);
    let (_, info) = square
        .resolve_with_info(std::slice::from_ref(&arena))
        .unwrap();
    let metric =
        |p: &PerimeterSpec| match p.scale_xy(info.meters_per_pixel, info.meters_per_pixel_y()) {
            PerimeterSpec::Rectangle { width, height, .. } => (width, height),
            _ => unreachable!(),
        };
    let ((aw, ah), (cw, ch)) = (metric(&arena), metric(&center));
    for (value, expected) in [(aw, 0.40), (ah, 0.40), (cw, 0.24), (ch, 0.24)] {
        assert!((value - expected).abs() < 1e-12, "{value} vs {expected}");
    }
    // Periphery band (arena − center) / 2 is 8 cm wide on both axes.
    assert!(((aw - cw) / 2.0 - 0.08).abs() < 1e-12 && ((ah - ch) / 2.0 - 0.08).abs() < 1e-12);
}
