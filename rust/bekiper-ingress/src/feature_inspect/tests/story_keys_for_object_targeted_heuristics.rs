#[test]
fn diagnostic_geometry_names_do_not_become_unresolvable_zone_targets() {
    let spec = HeuristicDecompositionSpec {
        heuristic_name: "in_center".into(),
        result_column: "in_center".into(),
        coordinate_labels: vec!["nose".into()],
        perimeter_label: Some("in_center".into()),
        components: vec![],
        show_final_result: true,
    };
    let perimeters = [PerimeterSpec::Circle {
        label: "center_zone".into(),
        center_x: 0.2,
        center_y: 0.2,
        radius: 0.1,
    }];
    let features = super::collect_feature_inspections(
        &perimeters,
        &[],
        &[h("in_center")],
        &[spec],
        &coord_cols(&["nose"]),
        None,
        None,
        &[],
        &[],
        &[],
    );
    assert!(features.iter().all(
        |feature| !matches!(feature, FeatureInspection::Zone { perimeter_label, .. }
        if !perimeters.iter().any(|perimeter| perimeter.label() == perimeter_label))
    ));
}

#[test]
fn arbitrary_object_labels_bind_and_route_to_their_actual_target() {
    let perimeters = ["arena", "A", "B"].map(|label| PerimeterSpec::Circle {
        label: label.into(),
        center_x: 0.2,
        center_y: 0.2,
        radius: 0.1,
    });
    let heuristics = [h("olfaction_A"), h("olfaction_B")];
    let mut features = super::collect_feature_inspections(
        &perimeters,
        &[],
        &heuristics,
        &[],
        &coord_cols(&["nose"]),
        None,
        None,
        &[],
        &[],
        &[],
    );
    super::populate_story_keys(&mut features, &heuristics, &perimeters);
    for target in ["A", "B"] {
        let feature = features
            .iter()
            .find(|feature| feature.label() == format!("heuristic_olfaction_{target}"))
            .unwrap();
        let FeatureInspection::Heuristic {
            perimeter_label, ..
        } = feature
        else {
            panic!("expected parent heuristic")
        };
        assert_eq!(perimeter_label.as_deref(), Some(target));
        assert_eq!(feature.story_key(), Some(target));
    }
}

#[test]
fn story_keys_for_object_targeted_heuristics() {
    let perimeters = vec![
        PerimeterSpec::Circle {
            label: "object_1".into(),
            center_x: 0.2,
            center_y: 0.2,
            radius: 0.1,
        },
        PerimeterSpec::Circle {
            label: "object_2".into(),
            center_x: 0.4,
            center_y: 0.2,
            radius: 0.1,
        },
        PerimeterSpec::Circle {
            label: "novel".into(),
            center_x: 0.6,
            center_y: 0.2,
            radius: 0.1,
        },
        PerimeterSpec::Circle {
            label: "arena".into(),
            center_x: 0.5,
            center_y: 0.5,
            radius: 0.4,
        },
        PerimeterSpec::Circle {
            label: "center".into(),
            center_x: 0.5,
            center_y: 0.5,
            radius: 0.2,
        },
    ];
    let heuristics = vec![
        h("body_proximity_object_1"),
        h("olfaction_object_2"),
        h("exploring_novel"),
        h("exploring_arena"),
        h("in_center"),
    ];
    let mut features = vec![
        FeatureInspection::Heuristic {
            label: "heuristic_body_proximity_object_1".into(),
            result_column: "body_proximity_object_1".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("object_1".into()),
            description: "object_1".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Heuristic {
            label: "heuristic_olfaction_object_2".into(),
            result_column: "olfaction_object_2".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("object_2".into()),
            description: "object_2".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Heuristic {
            label: "heuristic_exploring_novel".into(),
            result_column: "exploring_novel".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("novel".into()),
            description: "novel".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Heuristic {
            label: "heuristic_exploring_arena".into(),
            result_column: "exploring_arena".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("arena".into()),
            description: "arena".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::Zone {
            label: "zone_in_center".into(),
            perimeter_label: "center".into(),
            result_column: "in_center".into(),
            coordinate_labels: vec!["nose".into()],
            description: "center".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
    ];

    super::populate_story_keys(&mut features, &heuristics, &perimeters);

    assert_eq!(features[0].story_key(), Some("object_1"));
    assert_eq!(features[1].story_key(), Some("object_2"));
    assert_eq!(features[2].story_key(), Some("novel"));
    assert_eq!(features[3].story_key(), Some("objects"));
    assert_eq!(features[4].story_key(), Some("objects"));
}

#[test]
fn story_keys_for_object_decomposition_targets() {
    let perimeters = vec![
        PerimeterSpec::Circle {
            label: "object_1".into(),
            center_x: 0.2,
            center_y: 0.2,
            radius: 0.1,
        },
        PerimeterSpec::Circle {
            label: "object_2".into(),
            center_x: 0.4,
            center_y: 0.2,
            radius: 0.1,
        },
        PerimeterSpec::Circle {
            label: "novel".into(),
            center_x: 0.6,
            center_y: 0.2,
            radius: 0.1,
        },
        PerimeterSpec::Circle {
            label: "arena".into(),
            center_x: 0.5,
            center_y: 0.5,
            radius: 0.4,
        },
        PerimeterSpec::Circle {
            label: "center_zone".into(),
            center_x: 0.5,
            center_y: 0.5,
            radius: 0.2,
        },
    ];
    let mut features = vec![
        FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_body_proximity_object_1".into(),
            heuristic_name: "body_proximity_object_1".into(),
            result_column: "body_proximity_object_1".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("object_1".into()),
            components: vec![],
            show_final_result: true,
            description: "object_1".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_body_proximity_object_2".into(),
            heuristic_name: "body_proximity_object_2".into(),
            result_column: "body_proximity_object_2".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("object_2".into()),
            components: vec![],
            show_final_result: true,
            description: "object_2".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_body_proximity_novel".into(),
            heuristic_name: "body_proximity_novel".into(),
            result_column: "body_proximity_novel".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("novel".into()),
            components: vec![],
            show_final_result: true,
            description: "novel".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_olfaction_object_1".into(),
            heuristic_name: "olfaction_object_1".into(),
            result_column: "olfaction_object_1".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("object_1".into()),
            components: vec![],
            show_final_result: true,
            description: "object_1".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_olfaction_object_2".into(),
            heuristic_name: "olfaction_object_2".into(),
            result_column: "olfaction_object_2".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("object_2".into()),
            components: vec![],
            show_final_result: true,
            description: "object_2".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_olfaction_novel".into(),
            heuristic_name: "olfaction_novel".into(),
            result_column: "olfaction_novel".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("novel".into()),
            components: vec![],
            show_final_result: true,
            description: "novel".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_olfaction_arena".into(),
            heuristic_name: "olfaction_arena".into(),
            result_column: "olfaction_arena".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("arena".into()),
            components: vec![],
            show_final_result: true,
            description: "arena".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_body_proximity_arena".into(),
            heuristic_name: "body_proximity_arena".into(),
            result_column: "body_proximity_arena".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("arena".into()),
            components: vec![],
            show_final_result: true,
            description: "arena".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
        FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_body_proximity_center_zone".into(),
            heuristic_name: "body_proximity_center_zone".into(),
            result_column: "body_proximity_center_zone".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: Some("center_zone".into()),
            components: vec![],
            show_final_result: true,
            description: "center_zone".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        },
    ];

    super::populate_story_keys(&mut features, &[], &perimeters);

    assert_eq!(
        features[0].story_key(),
        Some("object_decomposition_body_proximity")
    );
    assert_eq!(
        features[1].story_key(),
        Some("object_decomposition_body_proximity")
    );
    assert_eq!(
        features[2].story_key(),
        Some("object_decomposition_body_proximity")
    );
    assert_eq!(
        features[3].story_key(),
        Some("object_decomposition_olfaction_object_1")
    );
    assert_eq!(
        features[4].story_key(),
        Some("object_decomposition_olfaction_object_2")
    );
    assert_eq!(
        features[5].story_key(),
        Some("object_decomposition_olfaction_novel")
    );
    assert_eq!(features[6].story_key(), None);
    assert_eq!(features[7].story_key(), None);
    assert_eq!(features[8].story_key(), None);
}

#[test]
fn story_keys_for_task_specific_features() {
    let primary = "snout".to_string();

    let nort_detail = nort_detail();
    let mut features = collect_task_detailed_inspections(
        bekiper_core::types::TaskType::ObjectRecognition,
        Some(&nort_detail),
        Some(&primary),
    );
    super::populate_story_keys(&mut features, &[], &[]);
    assert_eq!(features[0].story_key(), Some("discrimination"));

    let mut ymaze_detail = ymaze_detail();
    if let TaskDetail::YMaze(ref mut detail) = ymaze_detail {
        detail.analysis.arm_sequence = vec!["A".into(), "B".into(), "C".into()];
    }
    let mut features = collect_task_detailed_inspections(
        bekiper_core::types::TaskType::YMaze,
        Some(&ymaze_detail),
        Some(&primary),
    );
    super::populate_story_keys(&mut features, &[], &[]);
    assert_eq!(features[0].story_key(), Some("arm_sequence"));

    let oft_detail = oft_detail(35.2, 64.8);
    let mut features = collect_task_detailed_inspections(
        bekiper_core::types::TaskType::OpenField,
        Some(&oft_detail),
        Some(&primary),
    );
    super::populate_story_keys(&mut features, &[], &[]);
    assert_eq!(features[0].story_key(), Some("center_periphery"));
}

#[test]
fn ungrouped_features_have_no_story_key() {
    let coords = coord_cols(&["snout"]);
    let motion = &[("speed_mps", "m/s")];
    let mut features =
        collect_feature_inspections(&[], &[], &[], &[], &coords, None, None, &[], motion, &[]);
    super::populate_story_keys(&mut features, &[], &[]);

    // Trajectory → None (ungrouped)
    let traj = features
        .iter()
        .find(|f| matches!(f, FeatureInspection::Trajectory { .. }))
        .unwrap();
    assert_eq!(traj.story_key(), None);

    // TimeSeries → None (ungrouped)
    let ts = features
        .iter()
        .find(|f| matches!(f, FeatureInspection::TimeSeries { .. }))
        .unwrap();
    assert_eq!(ts.story_key(), None);
}
