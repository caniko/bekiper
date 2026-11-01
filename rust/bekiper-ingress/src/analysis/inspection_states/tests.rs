use super::*;
use bekiper_core::inspection::HeuristicDecompositionComponent;

fn spec(
    name: &str,
    result: &str,
    components: Vec<HeuristicDecompositionComponent>,
) -> HeuristicDecompositionSpec {
    HeuristicDecompositionSpec {
        heuristic_name: name.into(),
        result_column: result.into(),
        coordinate_labels: vec!["nose".into()],
        perimeter_label: Some("native_target_expanded_-0.01".into()),
        components,
        show_final_result: true,
    }
}

#[test]
fn observation_and_modality_states_bind_distinct_native_masks_and_canonical_targets() {
    let df = df!(
        "elapsed_s" => [0.0, 0.2, 0.7, 1.0],
        "near" => [Some(true), Some(true), Some(false), None],
        "gaze" => [false, true, true, true],
        "modality" => [false, true, false, true],
        "observed" => [true, false, true, true],
    )
    .unwrap();
    let components = [("nose_near", "near"), ("ray_hits_object", "gaze")].map(|(label, column)| {
        HeuristicDecompositionComponent {
            label: label.into(),
            result_column: column.into(),
            ..Default::default()
        }
    });
    let specs = [
        spec("olfaction_native_target", "modality", components.into()),
        spec("observation_native_target", "observed", vec![]),
    ];
    let perimeters = ["native_target", "native_target_expanded_-0.01", "arena"].map(|label| {
        PerimeterSpec::Circle {
            label: label.into(),
            center_x: 0.1,
            center_y: 0.2,
            radius: 0.03,
        }
    });
    let features = state_inspections(&df, &specs, &perimeters, 10.0);
    for (label, final_column, expected_categories) in [
        (
            "sensory_disagreement_olfaction_native_target_nose",
            "modality",
            [1, 7, 2, 8],
        ),
        (
            "attention_state_distribution_olfaction_native_target_nose",
            "observed",
            [5, 3, 6, 8],
        ),
    ] {
        let feature = features.iter().find(|f| f.label() == label).unwrap();
        let FeatureInspection::StateDistribution {
            source_columns,
            frame_counts,
            seconds,
            denominator_seconds,
            story_key,
            ..
        } = feature
        else {
            unreachable!()
        };
        assert_eq!(source_columns, &["near", "gaze", final_column]);
        assert!((denominator_seconds - 1.1).abs() < 1e-12);
        for (category, duration) in expected_categories.into_iter().zip([0.2, 0.5, 0.3, 0.1]) {
            assert_eq!(frame_counts[category], 1);
            assert!((seconds[category] - duration).abs() < 1e-12);
        }
        assert!(!story_key.as_ref().unwrap().contains("expanded"));
    }
    let composition = features
        .iter()
        .find(|f| f.label() == "session_composition_observation")
        .unwrap();
    let FeatureInspection::StateDistribution {
        category_labels,
        source_columns,
        ..
    } = composition
    else {
        unreachable!()
    };
    assert_eq!(
        category_labels,
        &[
            "native_target",
            "Overlapping targets",
            "Unassigned",
            "Missing"
        ]
    );
    assert_eq!(source_columns, &["observed"]);
}

#[test]
fn vector_examples_span_valid_rows_and_preserve_native_binding() {
    let component = HeuristicDecompositionComponent {
        label: "ray".into(),
        result_column: "hit".into(),
        ray_origin_label: Some("nose".into()),
        ray_direction_x_column: Some("dx".into()),
        ray_direction_y_column: Some("dy".into()),
        ray_target_x_column: Some("tx".into()),
        ray_target_y_column: Some("ty".into()),
        value_column: Some("angle".into()),
        value_unit: Some("degrees".into()),
        ..Default::default()
    };
    let mut direction = vec![1.0; 11];
    direction[0] = 0.0;
    direction[2] = f64::NAN;
    let mut target = vec![0.2; 11];
    target[9] = f64::NAN;
    let df = df!("dx" => direction, "dy" => vec![0.0; 11], "tx" => target,
                     "ty" => vec![0.1; 11], "angle" => vec![123.456; 11])
    .unwrap();
    let features = vector_examples(
        &df,
        &[spec(
            "olfaction_native_target",
            "hit",
            vec![component.clone()],
        )],
    );
    assert_eq!(
        features
            .iter()
            .map(|f| f.representative_frame_index().unwrap())
            .collect::<Vec<_>>(),
        [1, 3, 4, 6, 7, 10]
    );
    for feature in features {
        let FeatureInspection::HeuristicDecomposition {
            components,
            show_final_result,
            result_column,
            ..
        } = feature
        else {
            unreachable!()
        };
        assert!(!show_final_result);
        assert_eq!(result_column, "hit");
        assert_eq!(
            serde_json::to_value(&components[0]).unwrap(),
            serde_json::to_value(&component).unwrap()
        );
    }
}

#[test]
fn native_partitions_preserve_overlap_nulls_and_half_open_time() {
    let df = df!("elapsed_s" => [0.0, 0.2, 0.7, 1.0], "a" => [Some(true), Some(true), Some(false), None], "b" => [false, true, false, true]).unwrap();
    let (weights, basis) = durations(&df, 10.0).unwrap();
    let feature = partition(
        &df,
        &weights,
        &basis,
        vec!["a".into(), "b".into()],
        vec![
            "A".into(),
            "B".into(),
            "overlap".into(),
            "unassigned".into(),
            "missing".into(),
        ],
        |s| match (s[0], s[1]) {
            (None, _) | (_, None) => 4,
            (Some(true), Some(true)) => 2,
            (Some(true), _) => 0,
            (_, Some(true)) => 1,
            _ => 3,
        },
        StatePresentation {
            label: "fixture".into(),
            style: "pie",
            description: "fixture".into(),
            story: "fixture".into(),
        },
    )
    .unwrap();
    let FeatureInspection::StateDistribution {
        frame_counts,
        seconds,
        percentages,
        denominator_seconds,
        ..
    } = feature
    else {
        unreachable!()
    };
    assert_eq!(frame_counts, [1, 0, 1, 1, 1]);
    for (got, expected) in seconds.iter().zip([0.2, 0.0, 0.5, 0.3, 0.1]) {
        assert!((got - expected).abs() < 1e-12);
    }
    assert!((denominator_seconds - 1.1).abs() < 1e-12);
    assert!((percentages.iter().sum::<f64>() - 100.0).abs() < 1e-12);
    let invalid = df!("elapsed_s" => [0.0, 0.0]).unwrap();
    assert!(durations(&invalid, 10.0).is_none());
    let serialized = toml::to_string(&FeatureInspection::StateDistribution {
        label: "test".into(),
        category_labels: vec!["none".into()],
        frame_counts: vec![0],
        seconds: vec![0.0],
        percentages: vec![0.0],
        denominator_frames: 0,
        denominator_seconds: 0.0,
        source_columns: vec![],
        basis: basis.clone(),
        style: "pie".into(),
        description: "empty".into(),
        representative_frame_index: None,
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    })
    .unwrap();
    assert!(toml::from_str::<FeatureInspection>(&serialized).is_ok());
}
