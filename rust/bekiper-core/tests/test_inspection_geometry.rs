use bekiper_core::inspection::*;
use bekiper_core::scale::ScaleDerivationInfo;
use bekiper_core::types::TaskType;

const GOLDEN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../python/bekiper-inspect/tests/fixtures/golden_inspection.toml"
));

fn golden() -> InspectionManifest {
    toml::from_str(GOLDEN).expect("golden inspection fixture parses")
}

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("bekiper_{}_{name}", std::process::id()))
}

fn every_shape() -> Vec<PerimeterSpec> {
    vec![
        PerimeterSpec::Circle {
            label: "c".into(),
            center_x: 1.0,
            center_y: 2.0,
            radius: 1.0,
        },
        PerimeterSpec::Rectangle {
            label: "r".into(),
            center_x: 1.0,
            center_y: 2.0,
            width: 2.0,
            height: 4.0,
        },
        PerimeterSpec::Polygon {
            label: "p".into(),
            vertices: vec![(0.0, 1.0), (2.0, 3.0)],
        },
        PerimeterSpec::RadialMaze {
            label: "m".into(),
            center_vertices: vec![(0.0, 1.0)],
            arms: vec![vec![(1.0, 2.0)], vec![(3.0, 4.0)]],
        },
        PerimeterSpec::Triangle {
            label: "t".into(),
            vertices: [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)],
        },
    ]
}

#[test]
fn reflect_y_mirrors_every_shape_about_the_height() {
    let reflected: Vec<_> = every_shape().iter().map(|s| s.reflect_y(10.0)).collect();
    let labels: Vec<_> = reflected.iter().map(|s| s.label().as_str()).collect();
    assert_eq!(labels, vec!["c", "r", "p", "m", "t"]);
    assert!(
        matches!(reflected[0], PerimeterSpec::Circle { center_y, radius, .. } if center_y == 8.0 && radius == 1.0)
    );
    assert!(
        matches!(reflected[1], PerimeterSpec::Rectangle { center_y, height, .. } if center_y == 8.0 && height == 4.0)
    );
    assert!(
        matches!(&reflected[2], PerimeterSpec::Polygon { vertices, .. } if vertices == &[(0.0, 9.0), (2.0, 7.0)])
    );
    assert!(
        matches!(&reflected[3], PerimeterSpec::RadialMaze { center_vertices, arms, .. }
        if center_vertices == &[(0.0, 9.0)] && arms == &[vec![(1.0, 8.0)], vec![(3.0, 6.0)]])
    );
    assert!(
        matches!(reflected[4], PerimeterSpec::Triangle { vertices, .. } if vertices == [(0.0, 9.0), (1.0, 8.0), (2.0, 6.0)])
    );
    // Reflecting twice is the identity.
    for (shape, back) in every_shape().iter().zip(&reflected) {
        assert_eq!(&back.reflect_y(10.0), shape);
    }
}

#[test]
fn scale_xy_scales_axes_independently_and_circles_by_geometric_mean() {
    let scaled: Vec<_> = every_shape().iter().map(|s| s.scale_xy(2.0, 8.0)).collect();
    assert!(
        matches!(scaled[0], PerimeterSpec::Circle { center_x, center_y, radius, .. }
        if center_x == 2.0 && center_y == 16.0 && radius == 4.0)
    );
    assert!(
        matches!(scaled[1], PerimeterSpec::Rectangle { width, height, .. } if width == 4.0 && height == 32.0)
    );
    assert!(
        matches!(&scaled[2], PerimeterSpec::Polygon { vertices, .. } if vertices == &[(0.0, 8.0), (4.0, 24.0)])
    );
    assert!(
        matches!(&scaled[3], PerimeterSpec::RadialMaze { center_vertices, arms, .. }
        if center_vertices == &[(0.0, 8.0)] && arms[1] == vec![(6.0, 32.0)])
    );
    assert!(
        matches!(scaled[4], PerimeterSpec::Triangle { vertices, .. } if vertices[2] == (4.0, 32.0))
    );
    assert_eq!(
        every_shape()[1].scale(2.0),
        every_shape()[1].scale_xy(2.0, 2.0)
    );
}

#[test]
fn video_pixel_conversion_divides_by_positive_scale_only() {
    let shapes = every_shape();
    let pixels = to_video_pixels_vec(&shapes, 0.5);
    assert_eq!(pixels[1], shapes[1].scale(2.0));
    // A missing (zero) or invalid scale leaves geometry untouched.
    assert_eq!(to_video_pixels_vec(&shapes, 0.0), shapes);
    assert_eq!(to_video_pixels_vec(&shapes, -1.0), shapes);

    let grid = QuadrantSpec {
        label: "grid".into(),
        x_min: 0.0,
        y_min: 0.5,
        x_max: 1.0,
        y_max: 2.0,
        rows: 2,
        cols: 3,
    };
    let px = grid.to_video_pixels(0.5);
    assert_eq!(
        (px.x_min, px.y_min, px.x_max, px.y_max),
        (0.0, 1.0, 2.0, 4.0)
    );
    assert_eq!((px.rows, px.cols, px.label.as_str()), (2, 3, "grid"));
    let same = grid.to_video_pixels(0.0);
    assert_eq!((same.x_max, same.y_max), (1.0, 2.0));
}

#[test]
fn polygon_rectangle_conversion_rejects_non_finite_and_degenerate_shapes() {
    let rect = |w: f64, h: f64, x: f64| PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: x,
        center_y: 0.0,
        width: w,
        height: h,
    };
    let poly = |vertices: Vec<(f64, f64)>| PerimeterSpec::Polygon {
        label: "arena".into(),
        vertices,
    };
    let square = poly(vec![(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)]);
    assert_eq!(
        rect(2.0, 1.0, 0.0).convert_polygon_rectangle(),
        Some(square.clone())
    );
    assert_eq!(
        square.convert_polygon_rectangle(),
        Some(rect(2.0, 1.0, 0.0))
    );
    assert!(rect(0.0, 1.0, 0.0).convert_polygon_rectangle().is_none());
    assert!(
        rect(2.0, f64::NAN, 0.0)
            .convert_polygon_rectangle()
            .is_none()
    );
    // Finite inputs whose corners overflow are rejected too.
    assert!(
        rect(f64::MAX, 1.0, f64::MAX)
            .convert_polygon_rectangle()
            .is_none()
    );
    assert!(
        poly(vec![(0.0, 0.0), (1.0, 0.0)])
            .convert_polygon_rectangle()
            .is_none()
    );
    assert!(
        poly(vec![(0.0, 0.0), (1.0, f64::INFINITY), (1.0, 1.0)])
            .convert_polygon_rectangle()
            .is_none()
    );
    // Collinear vertices have no area.
    assert!(
        poly(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
            .convert_polygon_rectangle()
            .is_none()
    );
    assert!(every_shape()[0].convert_polygon_rectangle().is_none());
    assert!(every_shape()[4].convert_polygon_rectangle().is_none());
}

#[test]
fn decomposition_labels_accept_a_single_string_or_a_list() {
    let single: HeuristicDecompositionComponent =
        toml::from_str("label = \"near\"\nresult_column = \"c\"\ncoordinate_labels = \"nose\"")
            .unwrap();
    assert_eq!(single.coordinate_labels, vec!["nose"]);
    let many: HeuristicDecompositionSpec = toml::from_str(
        "heuristic_name = \"h\"\nresult_column = \"h\"\ncoordinate_labels = [\"nose\", \"tail_base\"]",
    )
    .unwrap();
    assert_eq!(many.coordinate_labels, vec!["nose", "tail_base"]);
    assert!(many.components.is_empty() && !many.show_final_result);
    assert!(
        !toml::to_string(&many)
            .unwrap()
            .contains("show_final_result")
    );
    assert!(
        toml::from_str::<HeuristicDecompositionSpec>(
            "heuristic_name = \"h\"\nresult_column = \"h\"\ncoordinate_labels = 3"
        )
        .is_err()
    );
}

#[test]
fn every_feature_kind_exposes_its_common_fields() {
    let mut manifest = golden();
    let kinds: Vec<_> = manifest
        .feature_inspections
        .iter()
        .map(|f| f.kind())
        .collect();
    assert_eq!(
        kinds,
        vec![
            "state_distribution",
            "perimeter",
            "zone",
            "quadrant",
            "heuristic",
            "heuristic_decomposition",
            "heuristic_cloud",
            "angle",
            "trajectory",
            "ray_cast",
            "detection_fields",
            "scale_derivation",
            "time_series",
            "proximity",
            "object_discrimination",
            "arm_sequence",
            "ymaze_arm_sections",
            "center_periphery",
        ]
    );
    for (index, feature) in manifest.feature_inspections.iter_mut().enumerate() {
        assert!(!feature.label().is_empty());
        let _ = (
            feature.color(),
            feature.linewidth(),
            feature.title_override(),
        );
        feature.set_story_key(format!("story_{index}"));
        feature.set_representative_frame_index(index as u64);
        assert_eq!(feature.story_key(), Some(format!("story_{index}").as_str()));
        assert_eq!(feature.representative_frame_index(), Some(index as u64));
        let has_result = matches!(
            feature.kind(),
            "zone" | "heuristic" | "heuristic_cloud" | "heuristic_decomposition" | "proximity"
        );
        assert_eq!(
            feature.result_column().is_some(),
            has_result,
            "{}",
            feature.kind()
        );
        let has_coordinates = !matches!(
            feature.kind(),
            "state_distribution"
                | "perimeter"
                | "angle"
                | "ray_cast"
                | "detection_fields"
                | "scale_derivation"
                | "time_series"
        );
        assert_eq!(
            feature.coordinate_labels().is_some(),
            has_coordinates,
            "{}",
            feature.kind()
        );
    }
    let zone = &golden().feature_inspections[2];
    assert_eq!(zone.representative_frame_index(), Some(42));
    assert_eq!(zone.result_column(), Some("in_zone_a"));
}

#[test]
fn identity_colors_cover_named_grouped_and_hashed_labels() {
    assert_eq!(inspection_identity_color("P0 G0 F0"), "#0072B2");
    assert_eq!(inspection_identity_color("P1 G0 O1"), "#56B4E9");
    assert_eq!(inspection_identity_color("P1 G1 F1"), "#332288");
    assert_eq!(
        inspection_identity_color("nose"),
        inspection_identity_color("familiar")
    );
    assert_eq!(inspection_identity_color("tail_base"), "#D55E00");
    assert_eq!(inspection_identity_color("body_center"), "#56B4E9");
    assert_eq!(inspection_identity_color("Missing"), "#777777");
    assert_eq!(inspection_identity_color("Unassigned"), "#dddddd");
    assert_eq!(inspection_identity_color("Overlapping targets"), "#332288");
    // object_0 is not a 1-based identity, so it hashes like any free label.
    let hashed = inspection_identity_color("object_0");
    assert_eq!(hashed, inspection_identity_color("object_0"));
    assert_eq!(inspection_identity_color("object_1"), "#0072B2");
}

#[test]
fn source_overlay_geometry_inverts_y_for_display_only() {
    let mut manifest = golden();
    manifest.settings.invert_y = false;
    let unchanged = manifest.source_overlay_geometry();
    assert_eq!(unchanged.perimeters, manifest.perimeters);

    manifest.settings.invert_y = true;
    for feature in &mut manifest.feature_inspections {
        if let FeatureInspection::Quadrant { occupancy_grid, .. } = feature {
            *occupancy_grid = Some(vec![vec![1.0], vec![2.0]]);
        }
    }
    let display = manifest.source_overlay_geometry();
    let height = manifest.metric_source_height();
    let pixel_height = manifest.video.resolution.1 as f64;
    for (shown, stored) in display.perimeters.iter().zip(&manifest.perimeters) {
        assert_eq!(shown, &stored.reflect_y(height));
    }
    for (shown, stored) in display
        .pixel_perimeters
        .iter()
        .zip(&manifest.pixel_perimeters)
    {
        assert_eq!(shown, &stored.reflect_y(pixel_height));
    }
    let (q, stored) = (&display.quadrants[0], &manifest.quadrants[0]);
    assert_eq!(
        (q.y_min, q.y_max),
        (height - stored.y_max, height - stored.y_min)
    );
    let grid = display
        .feature_inspections
        .iter()
        .find_map(|f| match f {
            FeatureInspection::Quadrant { occupancy_grid, .. } => occupancy_grid.clone(),
            _ => None,
        })
        .unwrap();
    assert_eq!(grid, vec![vec![2.0], vec![1.0]]);
    let (positions, vectors) = manifest.overlay_y_bindings();
    assert!(positions.contains(&"nose_y".to_string()));
    assert!(!vectors.is_empty());
}

#[test]
fn y_aspect_follows_the_scale_derivation() {
    let mut manifest = golden();
    manifest.scale_derivation = None;
    assert_eq!(manifest.y_aspect(), 1.0);
    manifest.scale_derivation = Some(ScaleDerivationInfo {
        method: "arena_box_detected".into(),
        physical_meters: Some(0.4),
        pixel_measurement: Some(200.0),
        arm_pixel_spans: vec![],
        reference_label: Some("arena".into()),
        endpoints: None,
        meters_per_pixel: 0.002,
        meters_per_pixel_y: Some(0.004),
    });
    assert_eq!(manifest.y_aspect(), 2.0);
    manifest.settings.meters_per_pixel = 0.002;
    let rows = manifest.video.resolution.1 as f64;
    assert!((manifest.metric_source_height() - rows * 0.004).abs() < 1e-12);
    manifest.compute_pixel_geometry();
    assert_eq!(manifest.pixel_perimeters.len(), manifest.perimeters.len());
    assert_eq!(manifest.pixel_quadrants.len(), manifest.quadrants.len());
}

#[test]
fn experiment_result_round_trips_through_toml() {
    let manifest = golden();
    let result = ExperimentResult {
        experiment: ExperimentInfo {
            name: "imen_t2".into(),
            task_type: TaskType::ObjectRecognition,
            fps: manifest.video.fps,
            meters_per_pixel: manifest.settings.meters_per_pixel,
        },
        video: manifest.video,
        video_path: None,
        frame_image: None,
        labels: manifest.labels.clone(),
        label_colors: manifest.label_colors.clone(),
        coordinate_columns: manifest.coordinate_columns.clone(),
        perimeters: manifest.perimeters.clone(),
        heuristics: manifest.heuristics.clone(),
        settings: manifest.settings,
        scale_derivation: None,
        detailed: Some(toml::Value::Table(toml::map::Map::new())),
        animal_metadata: Some(AnimalMetadata {
            animal: 7,
            cohort: "c1".into(),
            gene: "wt".into(),
            sex: "f".into(),
            age: "p60".into(),
        }),
        session_context: Some(IngressSessionContext {
            profile: "neo/imen".into(),
            trial: Some("T2".into()),
            ..Default::default()
        }),
        quadrants: manifest.quadrants.clone(),
        feature_inspections: manifest.feature_inspections.clone(),
    };
    let path = temp_path("result.toml");
    result.write_toml(&path).unwrap();
    let loaded = ExperimentResult::from_toml(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(loaded.experiment.name, "imen_t2");
    assert_eq!(loaded.experiment.task_type, TaskType::ObjectRecognition);
    assert_eq!(loaded.animal_metadata.unwrap().animal, 7);
    assert_eq!(loaded.session_context, result.session_context);
    assert_eq!(loaded.feature_inspections.len(), 18);
    assert!(ExperimentResult::from_toml(&temp_path("missing_result.toml")).is_err());
    let bad = temp_path("bad_result.toml");
    std::fs::write(&bad, "experiment = 3").unwrap();
    assert!(ExperimentResult::from_toml(&bad).is_err());
    std::fs::remove_file(&bad).ok();
}

#[test]
fn json_toml_helpers_drop_nulls_and_keep_types() {
    let value = serde_json::json!({
        "a": 1,
        "b": 2.5,
        "big": u64::MAX,
        "c": "text",
        "d": [true, null],
        "e": null,
        "f": {"g": false}
    });
    let text = json_value_to_toml_string(&value).unwrap();
    assert!(!text.contains("e ="));
    let back = toml_str_to_json_value(&text).unwrap();
    assert_eq!(back["a"], 1);
    assert_eq!(back["b"], 2.5);
    assert_eq!(back["big"], u64::MAX as f64);
    assert_eq!(back["c"], "text");
    // Array nulls cannot be dropped without shifting positions: empty string.
    assert_eq!(back["d"], serde_json::json!([true, ""]));
    assert_eq!(back["f"]["g"], false);
    let dated = toml_str_to_json_value("when = 1979-05-27T07:32:00Z").unwrap();
    assert_eq!(dated["when"], "1979-05-27T07:32:00Z");
    assert!(toml_str_to_json_value("not = valid = toml").is_err());
}
