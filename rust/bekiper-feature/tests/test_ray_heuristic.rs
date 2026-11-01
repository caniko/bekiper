use polars::prelude::*;

use bekiper_core::shape::RayIntersectable;
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::solo::olfaction::OlfactionHeuristic;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;

fn near_miss_df() -> DataFrame {
    df! {
        "nose_x" => &[0.0],
        "nose_y" => &[0.0],
        "snout_dx" => &[1.0],
        "snout_dy" => &[0.0],
    }
    .unwrap()
}

fn near_miss_circle() -> Circle {
    Circle::new(0.4, 0.12, 0.05)
}

#[test]
fn olfaction_cone_accepts_near_miss_inside_angle_tolerance() {
    let circle = near_miss_circle();
    assert!(
        !circle.ray_intersects((0.0, 0.0), (1.0, 0.0)),
        "setup should be a true ray miss so the cone logic is doing the work"
    );

    let heuristic = OlfactionHeuristic {
        perimeter: Perimeter::new("obj", circle),
        max_distance: 0.4,
        max_angle: 10.0,
        contact_distance: 0.01,
        name: "olfaction".into(),
        result_column: None,
    };

    let result = heuristic.evaluate(near_miss_df().lazy()).collect().unwrap();
    let ray_hits = result
        .column("__bk_diag__olfaction__ray_hits_object")
        .unwrap()
        .bool()
        .unwrap();
    let min_angle = result
        .column("__bk_diag__olfaction__ray_min_angle_deg")
        .unwrap()
        .f64()
        .unwrap();
    let margin = result
        .column("__bk_diag__olfaction__ray_angle_margin_deg")
        .unwrap()
        .f64()
        .unwrap();
    let target_x = result
        .column("__bk_diag__olfaction__ray_target_x")
        .unwrap()
        .f64()
        .unwrap();
    let target_y = result
        .column("__bk_diag__olfaction__ray_target_y")
        .unwrap()
        .f64()
        .unwrap();
    let combined = result.column("olfaction").unwrap().bool().unwrap();

    assert_eq!(ray_hits.get(0), Some(true));
    assert!(min_angle.get(0).unwrap() < 10.0);
    assert!(margin.get(0).unwrap() > 0.0);
    assert!(target_x.get(0).unwrap().is_finite());
    assert!(target_y.get(0).unwrap().is_finite());
    assert_eq!(combined.get(0), Some(true));
}

#[test]
fn olfaction_cone_rejects_near_miss_outside_angle_tolerance() {
    let heuristic = OlfactionHeuristic {
        perimeter: Perimeter::new("obj", near_miss_circle()),
        max_distance: 0.4,
        max_angle: 5.0,
        contact_distance: 0.01,
        name: "olfaction".into(),
        result_column: None,
    };

    let result = heuristic.evaluate(near_miss_df().lazy()).collect().unwrap();
    let ray_hits = result
        .column("__bk_diag__olfaction__ray_hits_object")
        .unwrap()
        .bool()
        .unwrap();
    let min_angle = result
        .column("__bk_diag__olfaction__ray_min_angle_deg")
        .unwrap()
        .f64()
        .unwrap();
    let margin = result
        .column("__bk_diag__olfaction__ray_angle_margin_deg")
        .unwrap()
        .f64()
        .unwrap();
    let combined = result.column("olfaction").unwrap().bool().unwrap();

    assert_eq!(ray_hits.get(0), Some(false));
    assert!(min_angle.get(0).unwrap() > 5.0);
    assert!(margin.get(0).unwrap() < 0.0);
    assert_eq!(combined.get(0), Some(false));
}
