#[test]
fn spatial_heuristic_with_matching_perimeter_becomes_zone() {
    let perimeters = vec![PerimeterSpec::Circle {
        label: "center".into(),
        center_x: 0.0,
        center_y: 0.0,
        radius: 0.1,
    }];
    let coords = coord_cols(&["snout", "tail_base"]);
    let heuristics = vec![h("in_center")];
    let features = collect_feature_inspections(
        &perimeters,
        &[],
        &heuristics,
        &[],
        &coords,
        None,
        None,
        &[],
        &[],
        &[],
    );

    // perimeter + zone + angle (snout/tail_base); baseline trajectory
    // is suppressed because Zone already draws the snout trajectory.
    assert_eq!(features.len(), 3);
    assert!(matches!(features[0], FeatureInspection::Perimeter { .. }));
    match &features[1] {
        FeatureInspection::Zone {
            perimeter_label,
            result_column,
            ..
        } => {
            assert_eq!(perimeter_label, "center");
            assert_eq!(result_column, "in_center");
        }
        other => panic!("expected Zone, got {other:?}"),
    }
    assert!(matches!(features[2], FeatureInspection::Angle { .. }));
    assert!(
        !features
            .iter()
            .any(|f| matches!(f, FeatureInspection::Trajectory { .. })),
        "trajectory should be suppressed when a Zone already covers the primary label"
    );
}

#[test]
fn freezing_emits_cloud_heuristic_feature() {
    let coords = coord_cols(&["snout"]);
    let heuristics = vec![h("freezing")];
    let features = collect_feature_inspections(
        &[],
        &[],
        &heuristics,
        &[],
        &coords,
        None,
        None,
        &[],
        &[],
        &[],
    );
    // Only the heuristic: baseline trajectory is suppressed because
    // the HeuristicCloud variant already covers the primary label.
    assert_eq!(features.len(), 1);
    match &features[0] {
        FeatureInspection::HeuristicCloud {
            perimeter_label,
            result_column,
            coordinate_labels,
            ..
        } => {
            assert_eq!(perimeter_label, &None);
            assert_eq!(result_column, "freezing");
            assert_eq!(coordinate_labels, &["snout"]);
        }
        other => panic!("expected HeuristicCloud, got {other:?}"),
    }
}

#[test]
fn heuristic_decomposition_is_emitted_after_parent_heuristic() {
    let coords = coord_cols(&["snout"]);
    let heuristics = vec![h("body_proximity")];
    let decompositions = vec![HeuristicDecompositionSpec {
        heuristic_name: "body_proximity".into(),
        result_column: "__bk_diag__body_proximity__final".into(),
        coordinate_labels: vec!["snout".into()],
        perimeter_label: Some("arena".into()),
        components: vec![bekiper_core::inspection::HeuristicDecompositionComponent {
            label: "raw_combined".into(),
            result_column: "__bk_diag__body_proximity__raw_combined".into(),
            coordinate_labels: vec!["snout".into()],
            perimeter_label: Some("arena".into()),
            ..Default::default()
        }],
        show_final_result: true,
    }];
    let features = collect_feature_inspections(
        &[],
        &[],
        &heuristics,
        &decompositions,
        &coords,
        None,
        None,
        &[],
        &[],
        &[],
    );

    assert!(matches!(features[0], FeatureInspection::Heuristic { .. }));
    match &features[1] {
        FeatureInspection::HeuristicDecomposition {
            heuristic_name,
            components,
            show_final_result,
            ..
        } => {
            assert_eq!(heuristic_name, "body_proximity");
            assert_eq!(components.len(), 1);
            assert!(*show_final_result);
        }
        other => panic!("expected HeuristicDecomposition, got {other:?}"),
    }
}

#[test]
fn trajectory_not_suppressed_when_heuristic_uses_different_label() {
    let mut coords = coord_cols(&["snout"]);
    coords.insert(
        "tail_base".to_string(),
        CoordinateColumnSpec {
            x: "tail_base_x".into(),
            y: "tail_base_y".into(),
        },
    );
    let primary = "snout".to_string();
    let heuristics = vec![h("freezing")];
    let features = collect_feature_inspections(
        &[],
        &[],
        &heuristics,
        &[],
        &coords,
        Some(&primary),
        None,
        &[],
        &[],
        &[],
    );
    // heuristic draws snout, so trajectory is suppressed.
    assert!(
        !features
            .iter()
            .any(|f| matches!(f, FeatureInspection::Trajectory { .. })),
        "baseline trajectory should be suppressed when the heuristic covers the same label"
    );
}

#[test]
fn quadrant_emits_quadrant_feature() {
    let quadrants = vec![QuadrantSpec {
        label: "open_field_grid".into(),
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        rows: 4,
        cols: 4,
    }];
    let coords = coord_cols(&["snout"]);
    let features = collect_feature_inspections(
        &[],
        &quadrants,
        &[],
        &[],
        &coords,
        None,
        None,
        &[],
        &[],
        &[],
    );
    assert_eq!(features.len(), 2);
    assert!(matches!(features[0], FeatureInspection::Quadrant { .. }));
    assert!(matches!(features[1], FeatureInspection::Trajectory { .. }));
}

#[test]
fn body_axis_pair_emits_angle_feature() {
    let coords = coord_cols(&["snout", "tail_base"]);
    let features =
        collect_feature_inspections(&[], &[], &[], &[], &coords, None, None, &[], &[], &[]);
    assert_eq!(features.len(), 2);
    match &features[0] {
        FeatureInspection::Angle {
            from_coordinate_label,
            to_coordinate_label,
            ..
        } => {
            assert_eq!(from_coordinate_label, "snout");
            assert_eq!(to_coordinate_label, "tail_base");
        }
        other => panic!("expected Angle, got {other:?}"),
    }
}

#[test]
fn explicit_primary_label_overrides_default() {
    let coords = coord_cols(&["snout", "centroid"]);
    let primary = "centroid".to_string();
    let features = collect_feature_inspections(
        &[],
        &[],
        &[],
        &[],
        &coords,
        Some(&primary),
        None,
        &[],
        &[],
        &[],
    );
    // angle + trajectory (centroid)
    assert!(features.iter().any(|f| matches!(
            f,
            FeatureInspection::Trajectory { coordinate_labels, .. } if coordinate_labels == &["centroid"]
        )));
}

#[test]
fn open_field_like_pipeline_yields_expected_features() {
    let perimeters = vec![
        PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 0.5,
            center_y: 0.5,
            width: 1.0,
            height: 1.0,
        },
        PerimeterSpec::Rectangle {
            label: "center".into(),
            center_x: 0.5,
            center_y: 0.5,
            width: 0.5,
            height: 0.5,
        },
    ];
    let coords = coord_cols(&["snout", "tail_base"]);
    let heuristics = vec![
        h("in_center"),
        h("in_periphery"),
        h("freezing"),
        h("rearing"),
        h("grooming"),
    ];
    let features = collect_feature_inspections(
        &perimeters,
        &[],
        &heuristics,
        &[],
        &coords,
        None,
        None,
        &[],
        &[],
        &[],
    );

    // 2 perimeters + 5 heuristics + 1 angle = 8. Baseline trajectory
    // is suppressed because every heuristic already draws the snout
    // trajectory (primary label).
    assert_eq!(features.len(), 8);

    // First two are perimeters
    assert!(matches!(features[0], FeatureInspection::Perimeter { .. }));
    assert!(matches!(features[1], FeatureInspection::Perimeter { .. }));

    // in_center should resolve to the "center" perimeter as a Zone
    assert!(matches!(
        &features[2],
        FeatureInspection::Zone { perimeter_label, .. } if perimeter_label == "center"
    ));
    assert!(matches!(
        &features[3],
        FeatureInspection::Heuristic {
            perimeter_label: None,
            ..
        }
    ));
    assert!(matches!(
        &features[4],
        FeatureInspection::HeuristicCloud {
            perimeter_label: None,
            ..
        }
    ));
    for feature in features.iter().take(7).skip(5) {
        assert!(matches!(
            feature,
            FeatureInspection::Heuristic {
                perimeter_label: None,
                ..
            }
        ));
    }

    assert!(matches!(features[7], FeatureInspection::Angle { .. }));
    assert!(
        !features
            .iter()
            .any(|f| matches!(f, FeatureInspection::Trajectory { .. })),
        "trajectory suppressed by heuristic coverage"
    );
}

#[test]
fn scale_derivation_emitted_for_non_manual_method() {
    let coords = coord_cols(&["snout"]);
    let sd = ScaleDerivationInfo {
        meters_per_pixel_y: None,
        method: "arena_side".into(),
        physical_meters: Some(0.4),
        pixel_measurement: Some(400.0),
        arm_pixel_spans: vec![],
        reference_label: Some("arena".into()),
        endpoints: None,
        meters_per_pixel: 0.001,
    };
    let features =
        collect_feature_inspections(&[], &[], &[], &[], &coords, None, Some(&sd), &[], &[], &[]);
    assert!(
        features
            .iter()
            .any(|f| matches!(f, FeatureInspection::ScaleDerivation { .. }))
    );
    let sd_feat = features
        .iter()
        .find(|f| matches!(f, FeatureInspection::ScaleDerivation { .. }))
        .unwrap();
    assert_eq!(sd_feat.representative_frame_index(), Some(0));
}

#[test]
fn scale_derivation_skipped_for_manual_method() {
    let coords = coord_cols(&["snout"]);
    let sd = ScaleDerivationInfo {
        meters_per_pixel_y: None,
        method: "manual".into(),
        physical_meters: None,
        pixel_measurement: None,
        arm_pixel_spans: vec![],
        reference_label: None,
        endpoints: None,
        meters_per_pixel: 0.001,
    };
    let features =
        collect_feature_inspections(&[], &[], &[], &[], &coords, None, Some(&sd), &[], &[], &[]);
    assert!(
        !features
            .iter()
            .any(|f| matches!(f, FeatureInspection::ScaleDerivation { .. })),
        "manual scale derivation should not emit a ScaleDerivation feature"
    );
}

#[test]
fn scale_derivation_skipped_when_absent() {
    let coords = coord_cols(&["snout"]);
    let features =
        collect_feature_inspections(&[], &[], &[], &[], &coords, None, None, &[], &[], &[]);
    assert!(
        !features
            .iter()
            .any(|f| matches!(f, FeatureInspection::ScaleDerivation { .. })),
    );
}

#[test]
fn ray_cast_spec_emits_ray_cast_feature() {
    let coords = coord_cols(&["nose"]);
    let specs: Vec<RayCastSpec> = vec![("nose".into(), "heading_rad".into(), 0.05, 60.0)];
    let features =
        collect_feature_inspections(&[], &[], &[], &[], &coords, None, None, &specs, &[], &[]);
    // ray_cast + trajectory (nose primary)
    assert_eq!(features.len(), 2);
    match &features[0] {
        FeatureInspection::RayCast {
            label,
            origin_label,
            angle_column,
            length_meters,
            cone_width_deg,
            ..
        } => {
            assert_eq!(label, "raycast_nose");
            assert_eq!(origin_label, "nose");
            assert_eq!(angle_column, "heading_rad");
            assert!((length_meters - 0.05).abs() < f64::EPSILON);
            assert!((cone_width_deg - 60.0).abs() < f64::EPSILON);
        }
        other => panic!("expected RayCast, got {other:?}"),
    }
    assert!(matches!(features[1], FeatureInspection::Trajectory { .. }));
}
