use polars::prelude::*;

use bekiper_core::types::CoordinateColumns;
use bekiper_feature::axiom::proximity::ComputeProximity;
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::helper::outside_perimeter::OutsidePerimeterHeuristic;
use bekiper_feature::heuristic::solo::body_proximity::BodyProximityHeuristic;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;

fn make_test_df() -> DataFrame {
    // Object at (5, 5). Animal body parts at various distances.
    df! {
        "center_ear_x" => &[5.0, 0.0, 5.05],
        "center_ear_y" => &[5.0, 0.0, 5.0],
        "tail_base_x" => &[4.0, -1.0, 4.0],
        "tail_base_y" => &[4.0, -1.0, 4.0],
    }
    .unwrap()
}

#[test]
fn body_proximity_heuristic_evaluates() {
    let perimeter = Perimeter::new("obj", Circle::new(5.0, 5.0, 0.5));
    let h = BodyProximityHeuristic {
        torso_coords: None,
        perimeter,
        max_distance: 0.1,
        head_coords: CoordinateColumns::new("center_ear_x", "center_ear_y"),
        tail_coords: CoordinateColumns::new("tail_base_x", "tail_base_y"),
        name: "body_proximity".into(),
        result_column: None,
    };

    assert_eq!(h.name(), "body_proximity");

    let result = h.evaluate(make_test_df().lazy()).collect().unwrap();
    assert!(result.column("body_proximity").is_ok());
    let bp = result.column("body_proximity").unwrap().bool().unwrap();
    assert_eq!(bp.len(), 3);
}

#[test]
fn heuristic_summary_default() {
    let perimeter = Perimeter::new("obj", Circle::new(5.0, 5.0, 0.5));
    let h = BodyProximityHeuristic {
        torso_coords: None,
        perimeter,
        max_distance: 2.0, // large distance so some points are inside
        head_coords: CoordinateColumns::new("center_ear_x", "center_ear_y"),
        tail_coords: CoordinateColumns::new("tail_base_x", "tail_base_y"),
        name: "body_proximity".into(),
        result_column: None,
    };

    let df = h.evaluate(make_test_df().lazy()).collect().unwrap();
    let timing = bekiper_core::video::FrameTiming::cfr(30.0);
    let summary = h.summary(&df, &timing);
    assert_eq!(summary.name, "body_proximity");
    assert_eq!(summary.total_frames, 3);
    assert!(summary.seconds >= 0.0);
}

#[test]
fn body_proximity_exposes_decomposition_metadata() {
    let perimeter = Perimeter::new("obj", Circle::new(5.0, 5.0, 0.5));
    let h = BodyProximityHeuristic {
        torso_coords: None,
        perimeter,
        max_distance: 2.0,
        head_coords: CoordinateColumns::new("center_ear_x", "center_ear_y"),
        tail_coords: CoordinateColumns::new("tail_base_x", "tail_base_y"),
        name: "body_proximity_obj".into(),
        result_column: Some("__bk_diag__body_proximity_obj__raw_combined".into()),
    };

    let decomposition = h.decomposition().expect("body_proximity decomposition");
    assert_eq!(decomposition.heuristic_name, "body_proximity_obj");
    assert_eq!(
        decomposition.result_column,
        "__bk_diag__body_proximity_obj__raw_combined"
    );
    assert!(
        decomposition
            .components
            .iter()
            .any(|component| component.label == "center_ear_near")
    );
    assert!(
        decomposition
            .components
            .iter()
            .any(|component| component.label == "tail_base_near")
    );
    assert!(!decomposition.show_final_result);
}

#[test]
fn outside_perimeter_exposes_decomposition_metadata() {
    let perimeter = Perimeter::new("obj", Circle::new(5.0, 5.0, 0.5));
    let h = OutsidePerimeterHeuristic {
        perimeter,
        coords: CoordinateColumns::new("center_ear_x", "center_ear_y"),
        name: "outside_obj".into(),
    };

    let decomposition = h.decomposition().expect("outside_perimeter decomposition");
    assert_eq!(decomposition.heuristic_name, "outside_obj");
    assert_eq!(decomposition.result_column, "outside_obj");
    assert_eq!(decomposition.coordinate_labels, vec!["center_ear"]);
    assert_eq!(decomposition.components.len(), 1);
    assert_eq!(decomposition.components[0].label, "outside_perimeter");
    assert!(!decomposition.show_final_result);
}

#[test]
fn compute_proximity_axiom() {
    let perimeter = Perimeter::new("obj", Circle::new(5.0, 5.0, 0.5));
    let proximity = ComputeProximity::new(perimeter, 1.0);
    let coords = CoordinateColumns::new("center_ear_x", "center_ear_y");

    let result = proximity
        .apply(make_test_df().lazy(), &coords, "near_obj")
        .collect()
        .unwrap();

    let near = result.column("near_obj").unwrap().bool().unwrap();
    assert_eq!(near.len(), 3);
    // Point at (5,5) should be near the object at (5,5) with radius 0.5 + distance 1.0
    assert!(near.get(0).unwrap());
    // Point at (0,0) should not be near
    assert!(!near.get(1).unwrap());
}
