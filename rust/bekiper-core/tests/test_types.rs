use bekiper_core::types::*;

#[test]
fn nort_annotation_roles_are_phase_specific_and_unique() {
    assert!(
        DetectionRecipe::NortTraining
            .annotation_label_errors(["arena", "variable", "familiar"])
            .is_empty()
    );
    assert!(
        DetectionRecipe::NortNovelty
            .annotation_label_errors(["familiar", "novel"])
            .is_empty()
    );
    assert_eq!(
        DetectionRecipe::NortTraining
            .annotation_label_errors(["familiar", "Familiar", "novel"])
            .len(),
        3
    );
    assert_eq!(
        DetectionRecipe::NortNovelty
            .annotation_label_errors(["familiar", "novel", "variable"])
            .len(),
        1
    );
}

#[test]
fn polygon_rectangle_conversion_preserves_label_and_rejects_degenerate_geometry() {
    use bekiper_core::inspection::PerimeterSpec;
    let rectangle = PerimeterSpec::Rectangle {
        label: "familiar".into(),
        center_x: 20.0,
        center_y: 30.0,
        width: 10.0,
        height: 8.0,
    };
    let polygon = rectangle.convert_polygon_rectangle().unwrap();
    assert!(matches!(polygon, PerimeterSpec::Polygon { .. }));
    assert_eq!(polygon.convert_polygon_rectangle(), Some(rectangle));
    let line = PerimeterSpec::Polygon {
        label: "variable".into(),
        vertices: vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
    };
    assert!(line.convert_polygon_rectangle().is_none());
}

#[test]
fn bounding_box_contains_inside() {
    let bb = BoundingBox {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 10.0,
        max_y: 10.0,
    };
    assert!(bb.contains(5.0, 5.0));
    assert!(bb.contains(0.0, 0.0));
    assert!(bb.contains(10.0, 10.0));
}

#[test]
fn bounding_box_contains_outside() {
    let bb = BoundingBox {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 10.0,
        max_y: 10.0,
    };
    assert!(!bb.contains(-1.0, 5.0));
    assert!(!bb.contains(5.0, 11.0));
    assert!(!bb.contains(11.0, 5.0));
    assert!(!bb.contains(5.0, -1.0));
}

#[test]
fn bounding_box_expand() {
    let bb = BoundingBox {
        min_x: 1.0,
        min_y: 2.0,
        max_x: 3.0,
        max_y: 4.0,
    };
    let expanded = bb.expand(0.5);
    assert!((expanded.min_x - 0.5).abs() < f64::EPSILON);
    assert!((expanded.min_y - 1.5).abs() < f64::EPSILON);
    assert!((expanded.max_x - 3.5).abs() < f64::EPSILON);
    assert!((expanded.max_y - 4.5).abs() < f64::EPSILON);
    // Expanded box should contain points just outside original
    assert!(expanded.contains(0.6, 1.6));
    assert!(!bb.contains(0.6, 1.6));
}

#[test]
fn bounding_box_expand_negative() {
    let bb = BoundingBox {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 10.0,
        max_y: 10.0,
    };
    let contracted = bb.expand(-1.0);
    assert!((contracted.min_x - 1.0).abs() < f64::EPSILON);
    assert!((contracted.max_x - 9.0).abs() < f64::EPSILON);
    assert!(!contracted.contains(0.5, 5.0));
}

#[test]
fn coordinate_columns_new() {
    let cc = CoordinateColumns::new("nose_x", "nose_y");
    assert_eq!(cc.x, "nose_x");
    assert_eq!(cc.y, "nose_y");
}

#[test]
fn coordinate_columns_from_string() {
    let cc = CoordinateColumns::new(String::from("a"), String::from("b"));
    assert_eq!(cc.x, "a");
    assert_eq!(cc.y, "b");
}

#[test]
fn coordinate_enum_as_str() {
    assert_eq!(Coordinate::X.as_str(), "x");
    assert_eq!(Coordinate::Y.as_str(), "y");
    assert_eq!(Coordinate::Likelihood.as_str(), "likelihood");
}

#[test]
fn coordinate_enum_equality() {
    assert_eq!(Coordinate::X, Coordinate::X);
    assert_ne!(Coordinate::X, Coordinate::Y);
}

#[test]
fn meters_per_pixel_serde_roundtrip() {
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Wrapper {
        mpp: MetersPerPixel,
    }
    let w = Wrapper {
        mpp: MetersPerPixel(0.001),
    };
    let toml_str = toml::to_string(&w).unwrap();
    let deserialized: Wrapper = toml::from_str(&toml_str).unwrap();
    assert_eq!(w.mpp, deserialized.mpp);
}

#[test]
fn midpoint_group_serde() {
    let mg = MidpointGroup {
        output_label: "center_ear".into(),
        source_labels: vec!["left_ear".into(), "right_ear".into()],
    };
    let toml_str = toml::to_string(&mg).unwrap();
    let deserialized: MidpointGroup = toml::from_str(&toml_str).unwrap();
    assert_eq!(deserialized.output_label, "center_ear");
    assert_eq!(deserialized.source_labels.len(), 2);
}

/// Parse every alias, check the canonical spelling round-trips through
/// `Display`, `FromStr` and serde, and that unknown names are rejected.
fn check_names<T>(cases: &[(&str, T)], canonical: impl Fn(&T) -> String)
where
    T: std::str::FromStr<Err = String>
        + std::fmt::Display
        + std::fmt::Debug
        + PartialEq
        + serde::Serialize
        + serde::de::DeserializeOwned,
{
    for (alias, expected) in cases {
        assert_eq!(&alias.parse::<T>().unwrap(), expected, "{alias}");
        let name = canonical(expected);
        assert_eq!(expected.to_string(), name);
        let json = serde_json::to_string(expected).unwrap();
        assert_eq!(json, format!("\"{name}\""));
        assert_eq!(&serde_json::from_str::<T>(&json).unwrap(), expected);
    }
    let err = "definitely-not-a-name".parse::<T>().unwrap_err();
    assert!(err.contains("definitely-not-a-name"), "{err}");
    assert!(serde_json::from_str::<T>("\"definitely-not-a-name\"").is_err());
    assert!(serde_json::from_str::<T>("3").is_err());
}

#[test]
fn task_type_names_and_aliases() {
    use TaskType::*;
    check_names(
        &[
            ("object_recognition", ObjectRecognition),
            ("novel_object_recognition", ObjectRecognition),
            ("objects_in_updating_locations", ObjectRecognition),
            ("nort", ObjectRecognition),
            ("oul", ObjectRecognition),
            ("y_maze", YMaze),
            ("ymaze", YMaze),
            ("radial_arm", YMaze),
            ("open_field", OpenField),
            ("open_field_test", OpenField),
            ("oft", OpenField),
            ("cheeseboard", Cheeseboard),
            ("reward_tracing", Cheeseboard),
        ],
        |t| t.as_str().to_string(),
    );
}

#[test]
fn detection_recipe_names_aliases_and_protocols() {
    use DetectionRecipe::*;
    check_names(
        &[
            ("nort_training", NortTraining),
            ("nort", NortTraining),
            ("nort_novelty", NortNovelty),
            ("oul_training", OulTraining),
            ("oul_update", OulUpdate),
            ("oul_test", OulTest),
            ("oft", Oft),
            ("open_field", Oft),
            ("ymaze", YMaze),
            ("y_maze", YMaze),
        ],
        |r| r.as_str().to_string(),
    );
    for recipe in [NortTraining, NortNovelty, OulTraining, OulUpdate, OulTest] {
        let protocol = recipe.object_recognition_protocol().unwrap();
        assert_eq!(protocol.as_str(), recipe.as_str());
    }
    assert!(Oft.object_recognition_protocol().is_none());
    assert!(YMaze.object_recognition_protocol().is_none());
    // Only the NORT phases carry annotation-role contracts.
    for recipe in [OulTraining, OulUpdate, OulTest, Oft, YMaze] {
        assert!(recipe.required_annotation_labels().is_empty());
        assert!(recipe.forbidden_annotation_labels().is_empty());
        assert!(
            recipe
                .annotation_label_errors(["novel", "variable"])
                .is_empty()
        );
    }
    assert_eq!(
        NortNovelty.annotation_label_errors([" Novel "]),
        vec!["Missing annotation: familiar".to_string()]
    );
}

#[test]
fn object_recognition_protocol_names_and_semantics() {
    use ObjectRecognitionProtocol::*;
    check_names(
        &[
            ("nort_training", NortTraining),
            ("nort_t1", NortTraining),
            ("nort_novelty", NortNovelty),
            ("nort_test", NortNovelty),
            ("nort_t2", NortNovelty),
            ("oul_training", OulTraining),
            ("oul_update", OulUpdate),
            ("oul_test", OulTest),
        ],
        |p| p.as_str().to_string(),
    );
    let all = [NortTraining, NortNovelty, OulTraining, OulUpdate, OulTest];
    let novelty: Vec<_> = all.iter().map(|p| p.computes_novelty_indices()).collect();
    assert_eq!(novelty, vec![false, true, false, false, false]);
    let training: Vec<_> = all.iter().map(|p| p.is_training_like()).collect();
    assert_eq!(training, vec![true, false, true, true, false]);
}

#[test]
fn detection_backend_names_aliases_and_gpu_needs() {
    use DetectionBackend::*;
    check_names(
        &[
            ("pre_computed", PreComputed),
            ("precomputed", PreComputed),
            ("manual_annotation", ManualAnnotation),
            ("manual", ManualAnnotation),
            ("annotation", ManualAnnotation),
            ("classical_cv", ClassicalCv),
            ("cv", ClassicalCv),
            ("arena_calibration_cv", ArenaCalibrationCv),
            ("arena_cv", ArenaCalibrationCv),
            ("arena_calibration", ArenaCalibrationCv),
            ("sam2_auto", Sam2Auto),
            ("sam2", Sam2Auto),
            ("grounding_dino_sam2", GroundingDinoSam2),
            ("grounding_dino", GroundingDinoSam2),
            ("gdino", GroundingDinoSam2),
        ],
        |b| b.as_str().to_string(),
    );
    let gpu: Vec<_> = [
        PreComputed,
        ManualAnnotation,
        ClassicalCv,
        ArenaCalibrationCv,
        Sam2Auto,
        GroundingDinoSam2,
    ]
    .iter()
    .map(|b| b.requires_gpu())
    .collect();
    assert_eq!(gpu, vec![false, false, false, false, true, true]);
}

#[test]
fn pose_model_presets_round_trip_through_strings() {
    use PoseModelPreset::*;
    for (alias, preset, name) in [
        ("dlc-super-animal", DlcSuperAnimal, "dlc-super-animal"),
        ("deeplabcut", DlcSuperAnimal, "dlc-super-animal"),
        (
            "sleap-single-instance",
            SleapSingleInstance,
            "sleap-single-instance",
        ),
        ("sleap", SleapSingleInstance, "sleap-single-instance"),
        ("lightning-pose", LightningPose, "lightning-pose"),
        ("lightning", LightningPose, "lightning-pose"),
        (
            "models/mouse.json",
            Custom("models/mouse.json".into()),
            "models/mouse.json",
        ),
    ] {
        let parsed = PoseModelPreset::from(alias.to_string());
        assert_eq!(parsed, preset);
        assert_eq!(parsed.to_string(), name);
        assert_eq!(String::from(parsed.clone()), name);
        let json = serde_json::to_string(&parsed).unwrap();
        assert_eq!(
            serde_json::from_str::<PoseModelPreset>(&json).unwrap(),
            parsed
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let raw = std::ffi::OsStr::from_bytes(&[0x66, 0xff]);
        assert_eq!(Custom(raw.into()).as_str(), "custom");
    }
}

#[test]
fn ymaze_arms_have_stable_names_columns_and_indices() {
    let names: Vec<_> = YMazeArm::ALL.iter().map(|arm| arm.to_string()).collect();
    assert_eq!(names, vec!["A", "B", "C"]);
    let columns: Vec<_> = YMazeArm::ALL.iter().map(|arm| arm.column_name()).collect();
    assert_eq!(columns, vec!["in_arm_A", "in_arm_B", "in_arm_C"]);
    let indices: Vec<_> = YMazeArm::ALL.iter().map(|arm| arm.index()).collect();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn coordinate_display_and_meters_per_pixel_access() {
    assert_eq!(Coordinate::Likelihood.to_string(), "likelihood");
    assert_eq!(format!("{}_{}", "nose", Coordinate::X), "nose_x");
    let mpp = MetersPerPixel::from(0.002);
    assert_eq!(*mpp, 0.002);
    assert_eq!(mpp.0 * 500.0, 1.0);
}
