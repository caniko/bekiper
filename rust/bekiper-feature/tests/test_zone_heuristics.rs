use polars::prelude::*;

use bekiper_core::inspection::PerimeterSpec;
use bekiper_core::types::CoordinateColumns;
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::helper::outside_perimeter::OutsidePerimeterHeuristic;
use bekiper_feature::heuristic::solo::freezing::FreezingHeuristic;
use bekiper_feature::heuristic::solo::periphery::PeripheryHeuristic;
use bekiper_feature::heuristic::solo::zone_confinement::ZoneConfinementHeuristic;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_perimeter::rectangle::Rectangle;

fn flags(df: &DataFrame, name: &str) -> Vec<bool> {
    df.column(name)
        .unwrap()
        .bool()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap())
        .collect()
}

/// 40 x 40 cm OFT arena with the 24 x 24 cm centre (8 cm peripheral band).
fn oft() -> PeripheryHeuristic<Rectangle> {
    PeripheryHeuristic {
        arena: Perimeter::new("arena", Rectangle::new(0.2, 0.2, 0.4, 0.4)),
        center: Perimeter::new("center_zone", Rectangle::new(0.2, 0.2, 0.24, 0.24)),
        coords: CoordinateColumns::new("body_x", "body_y"),
    }
}

#[test]
fn periphery_is_inside_arena_but_outside_center() {
    let h = oft();
    assert_eq!(h.name(), "in_periphery");
    let df = df! {
        // centre, band, outside the arena, missing, band at the centre's 0.08 m edge
        "body_x" => [Some(0.2), Some(0.04), Some(0.5), None, Some(0.0799)],
        "body_y" => [Some(0.2), Some(0.2), Some(0.2), Some(0.2), Some(0.2)],
    }
    .unwrap();
    let out = h.evaluate(df.lazy()).collect().unwrap();
    assert_eq!(
        flags(&out, "in_periphery"),
        vec![false, true, false, false, true]
    );
    assert_eq!(
        flags(&out, "__bk_diag__in_periphery__in_arena"),
        vec![true, true, false, false, true]
    );
    assert_eq!(
        flags(&out, "__bk_diag__in_periphery__in_center"),
        vec![true, false, false, false, false]
    );

    let spec = h.decomposition().unwrap();
    assert_eq!(spec.perimeter_label.as_deref(), Some("arena"));
    assert_eq!(spec.coordinate_labels, vec!["body"]);
    assert!(spec.show_final_result);
    let labels: Vec<_> = spec.components.iter().map(|c| c.label.as_str()).collect();
    assert_eq!(labels, vec!["in_arena", "in_center"]);
    assert!(matches!(
        spec.components[1].evaluated_perimeter,
        Some(PerimeterSpec::Rectangle { width, .. }) if width == 0.24
    ));
}

#[test]
fn zone_confinement_decomposition_carries_the_zone_geometry() {
    let h = ZoneConfinementHeuristic {
        zone_name: "in_arm_a".into(),
        perimeter: Perimeter::new("arm_a", Rectangle::new(0.0, 0.0, 1.0, 1.0)),
        coords: CoordinateColumns::new("nose_x", "nose_y"),
    };
    let spec = h.decomposition().unwrap();
    assert_eq!(spec.heuristic_name, "in_arm_a");
    assert_eq!(spec.perimeter_label.as_deref(), Some("arm_a"));
    assert_eq!(spec.components.len(), 1);
    assert_eq!(
        spec.components[0].result_column,
        "__bk_diag__in_arm_a__in_zone"
    );
    assert!(spec.components[0].evaluated_perimeter.is_some());
    assert!(!spec.show_final_result);
}

#[test]
fn outside_perimeter_treats_missing_as_not_outside() {
    let h = OutsidePerimeterHeuristic {
        perimeter: Perimeter::new("obj", Rectangle::new(0.0, 0.0, 1.0, 1.0)),
        coords: CoordinateColumns::new("x", "y"),
        name: "outside_obj".into(),
    };
    assert_eq!(h.name(), "outside_obj");
    let df = df! {
        "x" => [Some(0.0), Some(2.0), None],
        "y" => [Some(0.0), Some(0.0), Some(0.0)],
    }
    .unwrap();
    let out = h.evaluate(df.lazy()).collect().unwrap();
    // The shared confinement mask maps a missing point to "not inside".
    assert_eq!(flags(&out, "outside_obj"), vec![false, true, true]);
}

#[test]
fn freezing_threshold_is_inclusive_and_decomposed() {
    // 3-frame window at 10 fps; binary-exact steps keep the boundary exact.
    let h = FreezingHeuristic::new("x", "y", 0.3, 0.75, 10.0);
    assert_eq!(h.window_frames, 3);
    let df = df! {
        "x" => [0.0, 0.25, 0.5, 0.75, 1.125],
        "y" => [0.0; 5],
    }
    .unwrap();
    let out = h.evaluate(df.lazy()).collect().unwrap();
    // Window sums: (null), (null), 0.5, 0.75 (on the threshold), 0.875.
    assert_eq!(
        flags(&out, "freezing"),
        vec![false, false, true, true, false]
    );
    let spec = h.decomposition().unwrap();
    assert_eq!(spec.coordinate_labels, vec!["x"]);
    assert_eq!(
        spec.components[0].result_column,
        "__bk_diag__freezing__thresholded_rolling_displacement"
    );
}
