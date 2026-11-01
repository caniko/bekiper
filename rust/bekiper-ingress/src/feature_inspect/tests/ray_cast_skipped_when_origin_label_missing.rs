#[test]
fn ray_cast_skipped_when_origin_label_missing() {
    let coords = coord_cols(&["snout"]);
    let specs: Vec<RayCastSpec> = vec![(
        "nose".into(), // "nose" not in coords
        "heading_rad".into(),
        0.05,
        60.0,
    )];
    let features =
        collect_feature_inspections(&[], &[], &[], &[], &coords, None, None, &specs, &[], &[]);
    assert!(
        !features
            .iter()
            .any(|f| matches!(f, FeatureInspection::RayCast { .. })),
        "ray_cast should be skipped when origin_label is not in coordinate_columns"
    );
}

#[test]
fn motion_columns_emit_time_series_features() {
    let coords = coord_cols(&["snout"]);
    let motion = &[("speed_mps", "m/s"), ("displacement_m", "m")];
    let features =
        collect_feature_inspections(&[], &[], &[], &[], &coords, None, None, &[], motion, &[]);
    // trajectory + 2 time series
    assert_eq!(features.len(), 3);
    assert!(matches!(features[0], FeatureInspection::Trajectory { .. }));
    match &features[1] {
        FeatureInspection::TimeSeries {
            label,
            value_column,
            unit,
            description,
            representative_frame_index,
            ..
        } => {
            assert_eq!(label, "timeseries_speed_mps");
            assert_eq!(value_column, "speed_mps");
            assert_eq!(unit, "m/s");
            assert_eq!(description, "speed_mps over time (m/s)");
            assert_eq!(*representative_frame_index, None);
        }
        other => panic!("expected TimeSeries, got {other:?}"),
    }
    match &features[2] {
        FeatureInspection::TimeSeries {
            value_column, unit, ..
        } => {
            assert_eq!(value_column, "displacement_m");
            assert_eq!(unit, "m");
        }
        other => panic!("expected TimeSeries, got {other:?}"),
    }
}

#[test]
fn empty_motion_columns_add_nothing() {
    let coords = coord_cols(&["snout"]);
    let features =
        collect_feature_inspections(&[], &[], &[], &[], &coords, None, None, &[], &[], &[]);
    assert!(
        !features
            .iter()
            .any(|f| matches!(f, FeatureInspection::TimeSeries { .. })),
        "no motion columns means no TimeSeries features"
    );
}

#[test]
fn proximity_spec_emits_proximity_feature() {
    let coords = coord_cols(&["nose"]);
    let proximity_specs: Vec<ProximitySpec> =
        vec![("arena".into(), "near_arena".into(), 0.02, "nose".into())];
    let features = collect_feature_inspections(
        &[],
        &[],
        &[],
        &[],
        &coords,
        None,
        None,
        &[],
        &[],
        &proximity_specs,
    );
    // proximity only — trajectory is suppressed because Proximity
    // already draws the nose trajectory.
    assert_eq!(features.len(), 1);
    match &features[0] {
        FeatureInspection::Proximity {
            label,
            perimeter_label,
            buffer_meters,
            ..
        } => {
            assert_eq!(label, "proximity_arena");
            assert_eq!(perimeter_label, "arena");
            assert!((buffer_meters - 0.02).abs() < 1e-9);
        }
        other => panic!("expected Proximity, got {other:?}"),
    }
}

// ── Representative frame computation tests ───────────────────────

#[test]
fn longest_true_bout_midpoint_basic() {
    use polars::prelude::*;
    // [F, T, T, T, F, T, T, F] → longest bout at indices 1..4, midpoint = 2
    let s = Series::new(
        "b".into(),
        &[false, true, true, true, false, true, true, false],
    );
    assert_eq!(super::longest_true_bout_midpoint(&s), Some(2));
}

#[test]
fn longest_true_bout_midpoint_all_false() {
    use polars::prelude::*;
    let s = Series::new("b".into(), &[false, false, false]);
    assert_eq!(super::longest_true_bout_midpoint(&s), None);
}

#[test]
fn longest_true_bout_midpoint_all_true() {
    use polars::prelude::*;
    // [T, T, T, T, T] → bout 0..5, midpoint = 2
    let s = Series::new("b".into(), &[true, true, true, true, true]);
    assert_eq!(super::longest_true_bout_midpoint(&s), Some(2));
}

#[test]
fn longest_true_bout_midpoint_single_true() {
    use polars::prelude::*;
    let s = Series::new("b".into(), &[false, false, true, false]);
    assert_eq!(super::longest_true_bout_midpoint(&s), Some(2));
}

#[test]
fn longest_true_bout_midpoint_empty() {
    use polars::prelude::*;
    let s: Series = BooleanChunked::from_slice("b".into(), &[]).into_series();
    assert_eq!(super::longest_true_bout_midpoint(&s), None);
}

#[test]
fn median_non_null_index_basic() {
    use polars::prelude::*;
    // [null, 1.0, 2.0, null, 4.0] → non-null at [1, 2, 4], median index = 2
    let s = Series::new(
        "x".into(),
        &[None, Some(1.0_f64), Some(2.0), None, Some(4.0)],
    );
    assert_eq!(super::median_non_null_index(&s), Some(2));
}

#[test]
fn median_non_null_index_all_null() {
    use polars::prelude::*;
    let s = Series::new("x".into(), &[None::<f64>, None, None]);
    assert_eq!(super::median_non_null_index(&s), None);
}

fn test_df(columns: Vec<polars::prelude::Column>) -> polars::prelude::DataFrame {
    polars::prelude::DataFrame::new_infer_height(columns).unwrap()
}

#[test]
fn populate_representative_frames_fills_from_boolean_column() {
    use polars::prelude::*;

    let coords = coord_cols(&["snout"]);
    let heuristics = vec![h("freezing")];
    let mut features = collect_feature_inspections(
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

    // All features start with representative_frame_index = None.
    assert!(
        features
            .iter()
            .all(|f| f.representative_frame_index().is_none())
    );

    // Build a DataFrame with the boolean column and coordinate columns.
    let df = test_df(vec![
        Series::new(
            "freezing".into(),
            &[
                false, false, true, true, true, false, false, true, false, false,
            ],
        )
        .into(),
        Series::new("snout_x".into(), &[1.0_f64; 10]).into(),
        Series::new("snout_y".into(), &[2.0_f64; 10]).into(),
    ]);

    super::populate_representative_frames(&mut features, &df, &coords, 10);

    // The heuristic feature should pick the midpoint of the longest True bout (indices 2..5 → midpoint 3).
    let heuristic_feat = features
        .iter()
        .find(|f| matches!(f, FeatureInspection::HeuristicCloud { .. }))
        .unwrap();
    assert_eq!(heuristic_feat.representative_frame_index(), Some(3));
}

#[test]
fn populate_representative_frames_fills_from_coordinates() {
    use polars::prelude::*;

    let coords = coord_cols(&["snout"]);
    let mut features = vec![FeatureInspection::Trajectory {
        label: "trajectory_snout".into(),
        coordinate_labels: vec!["snout".into()],
        description: "test".into(),
        representative_frame_index: None,
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    }];

    let df = test_df(vec![
        Series::new(
            "snout_x".into(),
            &[None, Some(1.0_f64), Some(2.0), Some(3.0), None],
        )
        .into(),
        Series::new(
            "snout_y".into(),
            &[None, Some(1.0_f64), Some(2.0), Some(3.0), None],
        )
        .into(),
    ]);

    super::populate_representative_frames(&mut features, &df, &coords, 5);

    // Non-null indices: [1, 2, 3], median = 2
    assert_eq!(features[0].representative_frame_index(), Some(2));
}

#[test]
fn populate_representative_frames_skips_already_set() {
    use polars::prelude::*;

    let coords = coord_cols(&["snout"]);
    let mut features = vec![FeatureInspection::ScaleDerivation {
        label: "scale_derivation".into(),
        description: "test".into(),
        representative_frame_index: Some(42),
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    }];

    let df = test_df(vec![
        Series::new("snout_x".into(), &[1.0_f64; 5]).into(),
        Series::new("snout_y".into(), &[2.0_f64; 5]).into(),
    ]);

    super::populate_representative_frames(&mut features, &df, &coords, 100);

    // Should remain at 42, not be overwritten.
    assert_eq!(features[0].representative_frame_index(), Some(42));
}

#[test]
fn populate_representative_frames_clamps_to_total() {
    use polars::prelude::*;

    let coords = coord_cols(&["snout"]);
    let mut features = vec![FeatureInspection::Trajectory {
        label: "trajectory_snout".into(),
        coordinate_labels: vec!["snout".into()],
        description: "test".into(),
        representative_frame_index: None,
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    }];

    // Non-null at indices [0..100], median = 50, but total_frames = 10.
    let xs: Vec<Option<f64>> = (0..100).map(|i| Some(i as f64)).collect();
    let df = test_df(vec![
        Series::new("snout_x".into(), &xs).into(),
        Series::new("snout_y".into(), &xs).into(),
    ]);

    super::populate_representative_frames(&mut features, &df, &coords, 10);

    // Clamped to total_frames - 1 = 9.
    assert_eq!(features[0].representative_frame_index(), Some(9));
}

// ── Story key tests ─────────────────────────────────────────────

#[test]
fn story_keys_for_open_field_pipeline() {
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
    let heuristics = vec![h("in_center"), h("freezing")];
    let mut features = collect_feature_inspections(
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

    super::populate_story_keys(&mut features, &heuristics, &perimeters);

    // Perimeters → "geometry"
    assert_eq!(features[0].story_key(), Some("geometry"));
    assert_eq!(features[1].story_key(), Some("geometry"));
    // in_center (Zone) → "objects" (perimeter-anchored)
    assert_eq!(features[2].story_key(), Some("objects"));
    // freezing (HeuristicCloud, no perimeter) → "freezing" (standalone)
    assert_eq!(features[3].story_key(), Some("freezing"));
    // Angle → "geometry"
    assert_eq!(features[4].story_key(), Some("geometry"));
}
