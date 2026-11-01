//! Golden snapshot of the inspection manifest TOML serialized by
//! `bekiper-core`. The Python test suite in `bekiper-inspect` consumes
//! this golden file as its `inspection_dir` fixture, so any drift in
//! the Rust serialization (renamed field, changed variant tag, added
//! required field, …) is caught here first with a clear regeneration
//! message.
//!
//! To regenerate after an intentional schema change:
//!
//! ```text
//! UPDATE_GOLDEN=1 cargo test -p bekiper-ingress \
//!     --test test_golden_manifest golden_inspection_toml_matches_snapshot
//! ```

use std::collections::HashMap;
use std::path::Path;

use bekiper_core::inspection::{
    CoordinateColumnSpec, FeatureInspection, HeuristicDecompositionComponent, HeuristicMeta,
    InspectionManifest, InspectionSettings, PerimeterSpec, QuadrantSpec, QualiaChannelSpec,
};
use bekiper_core::scale::ScaleDerivationInfo;
use bekiper_core::types::MetersPerPixel;
use bekiper_core::video::VideoMetadata;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../python/bekiper-inspect/tests/fixtures/golden_inspection.toml",
);

#[test]
fn decomposition_bindings_and_exact_geometry_survive_toml_roundtrip() {
    let component = HeuristicDecompositionComponent {
        label: "left direction".into(),
        result_column: "accepted_left".into(),
        coordinate_labels: vec!["left_ear".into()],
        perimeter_label: Some("A".into()),
        buffer_meters: Some(0.037),
        ray_origin_label: Some("left_ear".into()),
        ray_direction_x_column: Some("left_dx".into()),
        ray_direction_y_column: Some("left_dy".into()),
        ray_target_x_column: Some("native_tx".into()),
        ray_target_y_column: Some("native_ty".into()),
        value_column: Some("native_angle".into()),
        value_unit: Some("degrees".into()),
        evaluated_perimeter: Some(PerimeterSpec::Circle {
            label: "eroded_A".into(),
            center_x: 0.15,
            center_y: 0.12,
            radius: 0.019,
        }),
        buffer_perimeter: Some(PerimeterSpec::Polygon {
            label: "expanded_A".into(),
            vertices: vec![(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)],
        }),
    };
    let decoded: HeuristicDecompositionComponent =
        toml::from_str(&toml::to_string_pretty(&component).unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(&decoded).unwrap(),
        serde_json::to_value(&component).unwrap()
    );
    let historical: HeuristicDecompositionComponent = toml::from_str(
        "label = 'head'\nresult_column = 'head_near'\ncoordinate_labels = ['nose']\n",
    )
    .unwrap();
    assert!(historical.ray_origin_label.is_none());
    assert!(historical.evaluated_perimeter.is_none());
}

/// Canonical inspection manifest used as the Python test fixture.
///
/// Values are chosen so every existing assertion in
/// `python/bekiper-inspect/tests/test_manifest.py` continues to pass.
fn golden_inspection_manifest() -> InspectionManifest {
    let mut label_colors: HashMap<String, String> = HashMap::new();
    label_colors.insert("nose".into(), "#1f77b4".into());
    label_colors.insert("tail_base".into(), "#ff7f0e".into());

    let mut coordinate_columns: HashMap<String, CoordinateColumnSpec> = HashMap::new();
    coordinate_columns.insert(
        "nose".into(),
        CoordinateColumnSpec {
            x: "nose_x".into(),
            y: "nose_y".into(),
        },
    );
    coordinate_columns.insert(
        "tail_base".into(),
        CoordinateColumnSpec {
            x: "tail_x".into(),
            y: "tail_y".into(),
        },
    );

    let perimeters = vec![
        PerimeterSpec::Circle {
            label: "arena".into(),
            center_x: 0.15,
            center_y: 0.12,
            radius: 0.14,
        },
        PerimeterSpec::Rectangle {
            label: "zone_a".into(),
            center_x: 0.1,
            center_y: 0.1,
            width: 0.05,
            height: 0.04,
        },
        PerimeterSpec::Polygon {
            label: "pen".into(),
            vertices: vec![(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)],
        },
        PerimeterSpec::RadialMaze {
            label: "ymaze".into(),
            center_vertices: vec![(0.14, 0.14), (0.16, 0.14), (0.15, 0.16)],
            arms: vec![
                vec![(0.14, 0.16), (0.16, 0.16), (0.16, 0.28), (0.14, 0.28)],
                vec![(0.14, 0.14), (0.04, 0.08), (0.05, 0.06), (0.15, 0.12)],
                vec![(0.16, 0.14), (0.26, 0.08), (0.25, 0.06), (0.15, 0.12)],
            ],
        },
    ];

    let heuristics = vec![
        HeuristicMeta {
            name: "body_proximity".into(),
            result_column: "body_prox".into(),
            true_frames: 40,
            total_frames: 100,
            seconds: 1.33,
            truth_spans: vec![(10, 30), (50, 70)],
        },
        HeuristicMeta {
            name: "olfaction".into(),
            result_column: "olfaction".into(),
            true_frames: 20,
            total_frames: 100,
            seconds: 0.67,
            truth_spans: vec![(5, 25)],
        },
        HeuristicMeta {
            name: "freezing".into(),
            result_column: "freezing".into(),
            true_frames: 30,
            total_frames: 100,
            seconds: 1.0,
            truth_spans: vec![(40, 70)],
        },
    ];

    // Golden covers ONE of every FeatureInspection variant we currently
    // serialize so the Python parser and Rust serializer cannot drift.
    // The Zone entry also carries an explicit representative_frame_index
    // to exercise the optional field's serialization path.
    let quadrants: Vec<QuadrantSpec> = vec![QuadrantSpec {
        label: "open_field_grid".into(),
        x_min: 0.0,
        y_min: 0.0,
        x_max: 0.30,
        y_max: 0.24,
        rows: 4,
        cols: 4,
    }];

    let feature_inspections: Vec<FeatureInspection> = vec![
        FeatureInspection::StateDistribution {
            label: "session_composition_fixture".into(),
            category_labels: vec!["object_1".into(), "Unassigned".into(), "Missing".into()],
            frame_counts: vec![50, 40, 10],
            seconds: vec![5.0, 4.0, 1.0],
            percentages: vec![50.0, 40.0, 10.0],
            denominator_frames: 100,
            denominator_seconds: 10.0,
            source_columns: vec!["olfaction".into()],
            basis: "half-open CFR 1/fps; evaluated masks, no extra tolerance".into(),
            style: "pie".into(),
            description: "Whole-session composition fixture".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Perimeter {
            label: "perimeter_arena".into(),
            perimeter_label: "arena".into(),
            description: "Perimeter 'arena' on experiment frame".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Zone {
            label: "zone_in_zone_a".into(),
            perimeter_label: "zone_a".into(),
            result_column: "in_zone_a".into(),
            coordinate_labels: vec!["nose".into()],
            description: "Zone confinement 'in_zone_a' inside perimeter 'zone_a'".into(),
            representative_frame_index: Some(42),
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Quadrant {
            label: "quadrant_open_field_grid".into(),
            quadrant_label: "open_field_grid".into(),
            coordinate_labels: vec!["nose".into()],
            description: "4x4 quadrant grid 'open_field_grid' on experiment frame".into(),
            occupancy_grid: None,
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Heuristic {
            label: "heuristic_body_proximity".into(),
            result_column: "body_prox".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: None,
            description: "Heuristic 'body_proximity' over experiment frame".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_body_proximity".into(),
            heuristic_name: "body_proximity".into(),
            result_column: "body_prox_final".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("arena".into()),
            components: vec![
                HeuristicDecompositionComponent {
                    label: "center_ear_near".into(),
                    result_column: "near_arena".into(),
                    coordinate_labels: vec!["nose".into()],
                    perimeter_label: Some("arena".into()),
                    ..Default::default()
                },
                HeuristicDecompositionComponent {
                    label: "raw_combined".into(),
                    result_column: "body_prox".into(),
                    coordinate_labels: vec!["nose".into()],
                    perimeter_label: Some("arena".into()),
                    ..Default::default()
                },
            ],
            show_final_result: true,
            description: "Heuristic decomposition for 'body_proximity'".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::HeuristicCloud {
            label: "heuristic_freezing".into(),
            result_column: "freezing".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: None,
            description: "Heuristic 'freezing' over experiment frame".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Angle {
            label: "angle_nose_tail_base".into(),
            from_coordinate_label: "nose".into(),
            to_coordinate_label: "tail_base".into(),
            description: "Body axis vector (nose → tail_base) on experiment frame".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Trajectory {
            label: "trajectory_tail_base".into(),
            coordinate_labels: vec!["tail_base".into()],
            description: "Trajectory of 'tail_base' on experiment frame".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::RayCast {
            label: "raycast_nose".into(),
            origin_label: "nose".into(),
            angle_column: "heading_rad".into(),
            length_meters: 0.05,
            cone_width_deg: 60.0,
            description: "Olfactory sensing cone at nose".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::DetectionFields {
            label: "detection_fields".into(),
            nose_label: "nose".into(),
            head_label: "tail_base".into(),
            cone_half_angle_deg: 45.0,
            nose_reach_m: 0.03,
            contact_distance_m: 0.005,
            whisker_apex_fraction: Some(0.8),
            whisker_half_angle_deg: Some(30.0),
            whisker_length_m: Some(0.025),
            object_labels: vec![],
            nose_near_columns: vec![],
            nose_gate_columns: vec![],
            whisker_hit_columns: vec![],
            description: "Detection fields on the animal".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::ScaleDerivation {
            label: "scale_derivation".into(),
            description: "How meters-per-pixel was derived".into(),
            representative_frame_index: Some(0),
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::TimeSeries {
            label: "timeseries_speed_mps".into(),
            value_column: "speed_mps".into(),
            unit: "m/s".into(),
            description: "speed_mps over time (m/s)".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Proximity {
            label: "proximity_arena".into(),
            perimeter_label: "arena".into(),
            buffer_meters: 0.02,
            result_column: "near_arena".into(),
            coordinate_labels: vec!["nose".into()],
            description: "Proximity to arena within 0.02m buffer".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::NortDiscrimination {
            label: "object_discrimination".into(),
            protocol: Some("nort_novelty".into()),
            object_labels: vec!["novel".into(), "familiar".into()],
            exploration_times: vec![12.5, 7.5],
            discrimination_index: Some(0.25),
            recognition_index: Some(0.625),
            qualia_channels: Some(vec![
                QualiaChannelSpec {
                    heuristic_type: "body_proximity".into(),
                    per_object_seconds: vec![8.0, 5.0],
                    per_object_bouts: vec![3, 2],
                    relative_bias: vec![61.5, 38.5],
                },
                QualiaChannelSpec {
                    heuristic_type: "olfaction".into(),
                    per_object_seconds: vec![10.0, 6.0],
                    per_object_bouts: vec![4, 3],
                    relative_bias: vec![62.5, 37.5],
                },
                QualiaChannelSpec {
                    heuristic_type: "whisker_interaction".into(),
                    per_object_seconds: vec![6.0, 3.5],
                    per_object_bouts: vec![2, 2],
                    relative_bias: vec![63.2, 36.8],
                },
            ]),
            alternation_count: Some(5),
            merged_channels: None,
            coordinate_labels: vec!["nose".into()],
            description: "Object discrimination (nort_novelty)".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: Some("Object Discrimination: NORT Novelty".into()),
            story_key: None,
        },
        FeatureInspection::ArmSequence {
            label: "arm_sequence".into(),
            arm_sequence: vec!["A".into(), "B".into(), "C".into(), "A".into(), "C".into()],
            coordinate_labels: vec!["nose".into()],
            alternation_pct: Some(66.7),
            description: "Arm entry sequence".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::YMazeArmSections {
            label: "ymaze_arm_sections".into(),
            perimeter_label: "ymaze".into(),
            proximal_threshold: 0.25,
            threshold_lines: vec![bekiper_core::inspection::ArmThresholdLine {
                arm: "A".into(),
                p1: (0.35, -0.15),
                p2: (0.325, 0.15),
            }],
            coordinate_labels: vec!["nose".into()],
            description: "Y-maze arm section thresholds".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::CenterPeriphery {
            label: "center_periphery".into(),
            center_perimeter_label: "zone_a".into(),
            arena_perimeter_label: "arena".into(),
            coordinate_labels: vec!["nose".into()],
            center_time_pct: 35.2,
            periphery_time_pct: 64.8,
            description: "Center vs. periphery occupancy".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
    ];

    let mut manifest = InspectionManifest {
        video: VideoMetadata {
            fps: 30.0,
            total_frames: 100,
            resolution: (640, 480),
            meters_per_pixel: MetersPerPixel(0.001),
        },
        video_path: None,
        session_context: None,
        frame_image: None,
        labels: vec!["nose".into(), "tail_base".into()],
        label_colors,
        coordinate_columns,
        perimeters,
        heuristics,
        settings: InspectionSettings {
            invert_y: false,
            minimum_seconds_tolerance: 0.5,
            maximum_seconds_distraction: 0.333,
            meters_per_pixel: 0.001,
        },
        scale_derivation: Some(ScaleDerivationInfo {
            meters_per_pixel_y: None,
            method: "arena_side".into(),
            physical_meters: Some(0.4),
            pixel_measurement: Some(400.0),
            arm_pixel_spans: vec![],
            reference_label: Some("arena".into()),
            endpoints: None,
            meters_per_pixel: 0.001,
        }),
        quadrants,
        feature_inspections,
        pixel_perimeters: vec![],
        pixel_quadrants: vec![],
    };

    // Populate pixel-space geometry via the ToVideoPixels trait,
    // exactly as summary.rs does for real manifests.
    manifest.compute_pixel_geometry();

    manifest
}

/// Serialize the manifest in a fully deterministic form.
///
/// The real `InspectionManifest` uses `HashMap` for `label_colors` and
/// `coordinate_columns`, which gives non-deterministic iteration order.
/// Funnelling through `toml::Value` first normalizes the key order
/// (the default `toml::value::Table` is a `BTreeMap<String, Value>`),
/// so the snapshot is stable across runs.
fn serialize_sorted(manifest: &InspectionManifest) -> String {
    let mut value = toml::Value::try_from(manifest)
        .expect("failed to convert golden InspectionManifest to toml::Value");
    sort_toml_value(&mut value);
    toml::to_string_pretty(&value).expect("failed to serialize normalized toml::Value")
}

fn sort_toml_value(value: &mut toml::Value) {
    match value {
        toml::Value::Table(table) => {
            for (_, child) in table.iter_mut() {
                sort_toml_value(child);
            }
            let mut entries: Vec<_> = table.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            table.clear();
            for (key, child) in entries {
                table.insert(key, child);
            }
        }
        toml::Value::Array(items) => {
            for item in items {
                sort_toml_value(item);
            }
        }
        _ => {}
    }
}

#[test]
fn golden_inspection_toml_matches_snapshot() {
    let manifest = golden_inspection_manifest();
    let actual = serialize_sorted(&manifest);

    if std::env::var("UPDATE_GOLDEN").is_ok() {
        let golden_path = Path::new(GOLDEN);
        std::fs::create_dir_all(golden_path.parent().unwrap())
            .expect("failed to create golden fixture directory");
        std::fs::write(golden_path, &actual).expect("failed to write golden fixture");
        eprintln!("wrote golden fixture to {}", golden_path.display());
        return;
    }

    let expected = std::fs::read_to_string(GOLDEN).unwrap_or_else(|e| {
        panic!(
            "golden fixture missing at {GOLDEN} ({e}); \
             run with UPDATE_GOLDEN=1 to create it"
        )
    });

    // Formatters may rewrap arrays in the shared Python fixture. Compare the
    // complete snapshot in the same canonical representation as the builder;
    // schema, values and sequence order must still match exactly.
    let mut expected: toml::Value =
        toml::from_str(&expected).expect("golden inspection fixture must be valid TOML");
    sort_toml_value(&mut expected);
    let expected = toml::to_string_pretty(&expected)
        .expect("failed to serialize normalized golden inspection fixture");

    assert_eq!(
        actual, expected,
        "golden inspection TOML drifted; rerun with UPDATE_GOLDEN=1 \
         to regenerate python/bekiper-inspect/tests/fixtures/golden_inspection.toml"
    );
}

/// Sanity-check that the golden builder is actually round-trippable
/// through the Rust serializer — guards against a future PerimeterSpec
/// variant that silently fails to deserialize.
#[test]
fn golden_manifest_roundtrips_through_toml() {
    let manifest = golden_inspection_manifest();
    let serialized = toml::to_string_pretty(&manifest).unwrap();
    let parsed: InspectionManifest =
        toml::from_str(&serialized).expect("golden manifest must round-trip through TOML");
    assert_eq!(parsed.labels.len(), 2);
    assert_eq!(parsed.perimeters.len(), 4);
    assert_eq!(parsed.heuristics.len(), 3);
}
