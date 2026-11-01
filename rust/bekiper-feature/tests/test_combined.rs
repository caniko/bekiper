use polars::prelude::*;

use bekiper_core::handoff;
use bekiper_core::inspection::{HeuristicDecompositionComponent, HeuristicDecompositionSpec};
use bekiper_core::types::CoordinateColumns;
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::combined::CombinedHeuristic;
use bekiper_feature::heuristic::combined::merge::{MergeRule, MergedHeuristic};
use bekiper_feature::heuristic::helper::outside_perimeter::OutsidePerimeterHeuristic;
use bekiper_feature::heuristic::solo::olfaction::OlfactionHeuristic;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;

fn flags(df: &DataFrame, name: &str) -> Vec<Option<bool>> {
    df.column(name)
        .unwrap()
        .bool()
        .unwrap()
        .into_iter()
        .collect()
}

fn frame() -> DataFrame {
    df! {
        "a" => [true, true, false, true],
        "b" => [true, false, false, true],
        "c" => [false, true, false, false],
        "x" => [true, false, false, true],
    }
    .unwrap()
}

fn evaluate(rule: MergeRule) -> Vec<Option<bool>> {
    let h = MergedHeuristic {
        name: "merged".into(),
        rule,
        decomposition: None,
    };
    flags(&h.evaluate(frame().lazy()).collect().unwrap(), "merged")
}

#[test]
fn merge_rule_combines_require_any_and_exclude() {
    let rule = MergeRule {
        require: vec!["a".into()],
        any_of: vec!["b".into(), "c".into()],
        exclude: vec!["x".into()],
    };
    assert!(!rule.is_empty());
    assert_eq!(
        evaluate(rule),
        vec![Some(false), Some(true), Some(false), Some(false)]
    );
    assert_eq!(
        evaluate(MergeRule {
            any_of: vec!["b".into(), "c".into()],
            ..Default::default()
        }),
        vec![Some(true), Some(true), Some(false), Some(true)]
    );
    assert_eq!(
        evaluate(MergeRule {
            exclude: vec!["a".into()],
            ..Default::default()
        }),
        vec![Some(false), Some(false), Some(true), Some(false)]
    );
}

#[test]
fn empty_merge_rule_leaves_the_frame_alone() {
    let rule = MergeRule::default();
    assert!(rule.is_empty());
    assert!(rule.to_expr().is_none());
    let spec = HeuristicDecompositionSpec {
        heuristic_name: "merged".into(),
        result_column: "merged".into(),
        coordinate_labels: vec![],
        perimeter_label: None,
        components: vec![],
        show_final_result: true,
    };
    let h = MergedHeuristic {
        name: "merged".into(),
        rule,
        decomposition: Some(spec),
    };
    assert_eq!(h.name(), "merged");
    assert_eq!(h.decomposition().unwrap().heuristic_name, "merged");
    let out = h.evaluate(frame().lazy()).collect().unwrap();
    assert!(out.column("merged").is_err());
    // Missing list keys deserialize to empty lists.
    let parsed: MergeRule = handoff::from_str("require = [\"a\"]").unwrap();
    assert_eq!(parsed.require, vec!["a"]);
    assert!(parsed.any_of.is_empty() && parsed.exclude.is_empty());
}

fn olfaction() -> OlfactionHeuristic<Circle> {
    OlfactionHeuristic {
        perimeter: Perimeter::new("obj", Circle::new(0.0, 0.0, 0.1)),
        max_distance: 0.05,
        max_angle: 30.0,
        contact_distance: 0.01,
        name: "olfaction".into(),
        result_column: None,
    }
}

fn outside() -> OutsidePerimeterHeuristic<Circle> {
    OutsidePerimeterHeuristic {
        perimeter: Perimeter::new("obj", Circle::new(0.0, 0.0, 0.1)),
        coords: CoordinateColumns::new("nose_x", "nose_y"),
        name: "outside_obj".into(),
    }
}

#[test]
fn combined_heuristic_ands_primary_and_helpers() {
    let h = CombinedHeuristic::new("exploring", olfaction(), vec![Box::new(outside())]);
    assert_eq!(h.name(), "exploring");
    assert!(h.decomposition().is_none());
    let df = df! {
        // near and heading in, inside the object, far away
        "nose_x" => [0.12, 0.05, 1.0],
        "nose_y" => [0.0, 0.0, 0.0],
        "snout_dx" => [-1.0, -1.0, -1.0],
        "snout_dy" => [0.0, 0.0, 0.0],
    }
    .unwrap();
    let out = h.evaluate(df.lazy()).collect().unwrap();
    assert_eq!(
        flags(&out, "olfaction"),
        vec![Some(true), Some(true), Some(false)]
    );
    assert_eq!(
        flags(&out, "exploring"),
        vec![Some(true), Some(false), Some(false)]
    );
}

#[test]
fn combined_decomposition_inherits_primary_component_metadata() {
    let component = |label: &str, column: &str| HeuristicDecompositionComponent {
        label: label.into(),
        result_column: column.into(),
        ..Default::default()
    };
    let spec = HeuristicDecompositionSpec {
        heuristic_name: "exploring".into(),
        result_column: "exploring".into(),
        coordinate_labels: vec!["nose".into()],
        perimeter_label: Some("obj".into()),
        components: vec![
            component("near", "__bk_diag__olfaction__nose_near"),
            component("ray", "__bk_diag__olfaction__ray_hits_object"),
            component("outside", "outside_obj"),
        ],
        show_final_result: true,
    };
    let h =
        CombinedHeuristic::new("exploring", olfaction(), vec![]).with_decomposition(spec.clone());
    let merged = h.decomposition().unwrap();
    assert_eq!(merged.components[0].buffer_meters, Some(0.05));
    assert!(merged.components[0].buffer_perimeter.is_some());
    assert_eq!(
        merged.components[1].ray_origin_label.as_deref(),
        Some("nose")
    );
    assert_eq!(
        merged.components[1].ray_direction_x_column.as_deref(),
        Some("snout_dx")
    );
    assert_eq!(merged.components[1].value_unit.as_deref(), Some("degrees"));
    assert!(merged.components[1].evaluated_perimeter.is_some());
    // Components without a primary counterpart keep their own metadata.
    assert!(merged.components[2].buffer_meters.is_none());
    assert!(merged.components[2].evaluated_perimeter.is_none());

    // A primary without decomposition metadata leaves the spec untouched.
    struct Bare;
    impl Heuristic for Bare {
        fn name(&self) -> &str {
            "bare"
        }
        fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
            lf
        }
    }
    let bare = CombinedHeuristic::new("exploring", Bare, vec![]).with_decomposition(spec);
    assert!(
        bare.decomposition().unwrap().components[0]
            .buffer_meters
            .is_none()
    );
}
