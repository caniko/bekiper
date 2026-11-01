use polars::prelude::*;

use bekiper_core::handoff;
use bekiper_core::types::{CoordinateColumns, RaySpec};
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::proximity::ProximityHeuristic;
use bekiper_feature::heuristic::ray::{RayHeuristic, ray_min_angle_col, ray_target_x_col};
use bekiper_feature::heuristic::solo::body_proximity::BodyProximityHeuristic;
use bekiper_feature::heuristic::solo::olfaction::OlfactionHeuristic;
use bekiper_feature::heuristic::solo::whiskers::WhiskerHeuristic;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;

const CIRCLE: &str = r#"
max_distance = 0.05
max_angle = 30.0

[perimeter]
label = "obj"

[perimeter.shape]
center_x = 0.0
center_y = 0.0
radius = 0.1
"#;

fn xs(columns: Vec<CoordinateColumns>) -> Vec<String> {
    columns.into_iter().map(|c| c.x).collect()
}

fn flags(df: &DataFrame, expr: Expr) -> Vec<bool> {
    df.clone()
        .lazy()
        .select([expr.alias("out")])
        .collect()
        .unwrap()
        .column("out")
        .unwrap()
        .bool()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap_or(false))
        .collect()
}

/// Points at 0.14 (inside), 0.15 (on) and 0.16 (outside) of the expanded
/// 0.15 m radius, plus a missing coordinate.
fn ring(x: &str, y: &str) -> DataFrame {
    df! {
        x => [Some(0.14), Some(0.15), Some(0.16), None],
        y => [Some(0.0), Some(0.0), Some(0.0), Some(0.0)],
    }
    .unwrap()
}

#[test]
fn body_proximity_deserializes_with_default_columns() {
    let h: BodyProximityHeuristic<Circle> = handoff::from_str(CIRCLE).unwrap();
    assert_eq!(h.name, "body_proximity");
    assert_eq!(
        (h.head_coords.x.as_str(), h.head_coords.y.as_str()),
        ("center_ear_x", "center_ear_y")
    );
    assert_eq!(
        (h.tail_coords.x.as_str(), h.tail_coords.y.as_str()),
        ("tail_base_x", "tail_base_y")
    );
    assert!(h.torso_coords.is_none() && h.result_column.is_none());
    assert_eq!(h.max_distance_meters(), 0.05);
    assert_eq!(ProximityHeuristic::perimeter(&h).label, "obj");
    assert_eq!(
        xs(h.coordinate_columns()),
        vec!["center_ear_x", "tail_base_x"]
    );

    // Proximity is inclusive at exactly max_distance from the boundary.
    let mut df = ring("center_ear_x", "center_ear_y");
    df.with_column(Series::new("tail_base_x".into(), [9.0, 9.0, 9.0, 0.0]).into_column())
        .unwrap();
    df.with_column(Series::new("tail_base_y".into(), [9.0, 9.0, 9.0, 0.0]).into_column())
        .unwrap();
    assert_eq!(
        flags(&df, h.proximity_expr()),
        vec![true, true, false, true]
    );
}

#[test]
fn body_proximity_torso_is_a_descriptive_channel_only() {
    let h = BodyProximityHeuristic {
        torso_coords: Some(CoordinateColumns::new("c_x", "c_y")),
        result_column: Some("raw".into()),
        ..handoff::from_str::<BodyProximityHeuristic<Circle>>(CIRCLE).unwrap()
    };
    assert_eq!(h.result_column(), "raw");
    let df = df! {
        "center_ear_x" => [1.0, 0.0],
        "center_ear_y" => [1.0, 0.0],
        "tail_base_x" => [0.0, 1.0],
        "tail_base_y" => [0.0, 1.0],
        "c_x" => [0.0, 1.0],
        "c_y" => [0.0, 1.0],
    }
    .unwrap();
    let out = h.evaluate(df.lazy()).collect().unwrap();
    let col = |name: &str| -> Vec<bool> {
        out.column(name)
            .unwrap()
            .bool()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap())
            .collect()
    };
    // Backing into the object (tail/torso near, head far) is not proximity.
    assert_eq!(col("raw"), vec![false, true]);
    assert_eq!(
        col("__bk_diag__body_proximity__tail_base_near"),
        vec![true, false]
    );
    assert_eq!(
        col("__bk_diag__body_proximity__torso_near"),
        vec![true, false]
    );
}

#[test]
fn olfaction_deserializes_with_defaults_and_checks_nose_only() {
    let h: OlfactionHeuristic<Circle> = handoff::from_str(CIRCLE).unwrap();
    assert_eq!(h.name(), "olfaction");
    assert_eq!(h.contact_distance, 0.01);
    assert_eq!(h.max_distance_meters(), 0.05);
    assert_eq!(ProximityHeuristic::perimeter(&h).label, "obj");
    assert_eq!(xs(h.coordinate_columns()), vec!["nose_x"]);
    let df = ring("nose_x", "nose_y");
    assert_eq!(
        flags(&df, h.proximity_expr()),
        vec![true, true, false, false]
    );
}

#[test]
fn whisker_deserializes_with_defaults_and_checks_either_ear() {
    let h: WhiskerHeuristic<Circle> = handoff::from_str(CIRCLE).unwrap();
    assert_eq!(h.name(), "whisker_interaction");
    assert_eq!(h.max_distance_meters(), 0.05);
    assert_eq!(ProximityHeuristic::perimeter(&h).label, "obj");
    assert_eq!(
        xs(h.coordinate_columns()),
        vec!["left_ear_x", "right_ear_x"]
    );
    let mut df = ring("left_ear_x", "left_ear_y");
    df.with_column(
        Series::new(
            "right_ear_x".into(),
            [Some(1.0), Some(1.0), Some(0.0), Some(0.0)],
        )
        .into_column(),
    )
    .unwrap();
    df.with_column(Series::new("right_ear_y".into(), [0.0; 4]).into_column())
        .unwrap();
    assert_eq!(flags(&df, h.proximity_expr()), vec![true, true, true, true]);
}

/// Trait defaults: 5 cm proximity and a 45 degree field of view.
struct Defaults {
    perimeter: Perimeter<Circle>,
    label: &'static str,
}

impl ProximityHeuristic<Circle> for Defaults {
    fn perimeter(&self) -> &Perimeter<Circle> {
        &self.perimeter
    }
    fn coordinate_columns(&self) -> Vec<CoordinateColumns> {
        vec![CoordinateColumns::new("x", "y")]
    }
}

impl RayHeuristic<Circle> for Defaults {
    fn diagnostic_name(&self) -> &str {
        "defaults"
    }
    fn perimeter(&self) -> &Perimeter<Circle> {
        &self.perimeter
    }
    fn ray_specs(&self) -> Vec<RaySpec> {
        vec![RaySpec {
            label: self.label.into(),
            origin: CoordinateColumns::new("x", "y"),
            direction: CoordinateColumns::new("dx", "dy"),
        }]
    }
}

#[test]
fn proximity_and_ray_trait_defaults() {
    let h = Defaults {
        perimeter: Perimeter::new("obj", Circle::new(1.0, 0.0, 0.1)),
        label: "",
    };
    assert_eq!(h.max_distance_meters(), 0.05);
    assert_eq!(h.max_angle_degrees(), 45.0);
    // Unlabelled rays name their diagnostics after the bare component.
    let spec = &h.ray_specs()[0];
    assert_eq!(
        ray_min_angle_col("defaults", spec),
        "__bk_diag__defaults__min_angle_deg"
    );
    assert_eq!(
        ray_target_x_col("defaults", spec),
        "__bk_diag__defaults__target_x"
    );

    // Headings 0, 90 and 180 degrees from the origin toward the circle.
    let df = df! {
        "x" => [0.0, 0.0, 0.0],
        "y" => [0.0, 0.0, 0.0],
        "dx" => [1.0, 0.0, -1.0],
        "dy" => [0.0, 1.0, 0.0],
    }
    .unwrap();
    assert_eq!(flags(&df, h.ray_expr()), vec![true, false, false]);
    assert_eq!(flags(&df, h.proximity_expr()), vec![false, false, false]);
}

fn column_bools(df: &DataFrame, name: &str) -> Vec<bool> {
    df.column(name)
        .unwrap()
        .bool()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap())
        .collect()
}

#[test]
fn olfaction_contact_overrides_heading_only_when_near() {
    let h = OlfactionHeuristic {
        result_column: Some("olf_raw".into()),
        ..handoff::from_str::<OlfactionHeuristic<Circle>>(CIRCLE).unwrap()
    };
    assert_eq!(h.result_column(), "olf_raw");
    let df = df! {
        // toward + near, away + near, away + in contact, toward + far, missing
        "nose_x" => [Some(0.12), Some(0.12), Some(0.105), Some(0.5), None],
        "nose_y" => [Some(0.0); 5],
        "snout_dx" => [-1.0, 1.0, 1.0, -1.0, -1.0],
        "snout_dy" => [0.0; 5],
    }
    .unwrap();
    let out = h.evaluate(df.lazy()).collect().unwrap();
    assert_eq!(
        column_bools(&out, "olf_raw"),
        vec![true, false, true, false, false]
    );
    assert_eq!(
        column_bools(&out, "__bk_diag__olfaction__nose_contact"),
        vec![false, false, true, false, false]
    );
    assert_eq!(
        column_bools(&out, "__bk_diag__olfaction__ray_hits_object"),
        vec![true, false, false, true, false]
    );

    let spec = h.decomposition().unwrap();
    assert_eq!(spec.result_column, "olf_raw");
    let labels: Vec<_> = spec.components.iter().map(|c| c.label.as_str()).collect();
    assert_eq!(
        labels,
        vec![
            "nose_near",
            "ray_hits_object",
            "nose_contact",
            "raw_combined"
        ]
    );
    assert!(
        spec.components
            .iter()
            .all(|c| c.evaluated_perimeter.is_some())
    );
}

#[test]
fn whisker_selects_the_ear_side_that_explains_the_interaction() {
    let h = WhiskerHeuristic {
        result_column: Some("whisk_raw".into()),
        ..handoff::from_str::<WhiskerHeuristic<Circle>>(CIRCLE).unwrap()
    };
    assert_eq!(h.result_column(), "whisk_raw");
    let df = df! {
        // left only near, right only near, neither near, both near
        "left_ear_x" => [0.12, 0.5, 1.0, 0.12],
        "left_ear_y" => [0.0, 0.5, 1.0, 0.0],
        "right_ear_x" => [0.5, 0.0, 1.0, 0.0],
        "right_ear_y" => [0.0, 0.12, 1.0, 0.12],
        "center_ear_x" => [0.3, 0.0, 1.0, 0.2],
        "center_ear_y" => [0.0, 0.3, 0.0, 0.2],
        "left_ear_dx" => [-1.0, 0.0, -1.0, 0.0],
        "left_ear_dy" => [0.0, 1.0, 0.0, 1.0],
        "right_ear_dx" => [1.0, 0.0, 0.0, -1.0],
        "right_ear_dy" => [0.0, -1.0, 1.0, -1.0],
    }
    .unwrap();
    let out = h.evaluate(df.lazy()).collect().unwrap();
    assert_eq!(
        column_bools(&out, "whisk_raw"),
        vec![true, true, false, true]
    );
    let sides: Vec<_> = out
        .column("__bk_diag__whisker_interaction__selected_ray_side")
        .unwrap()
        .str()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap().to_string())
        .collect();
    // Proximity decides when only one ear is near; otherwise the wider margin wins.
    assert_eq!(sides, vec!["left", "right", "left", "right"]);
    let dx: Vec<_> = out
        .column("__bk_diag__whisker_interaction__selected_ray_dx")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(dx, vec![Some(-1.0), Some(0.0), Some(-1.0), Some(-1.0)]);

    let spec = h.decomposition().unwrap();
    assert_eq!(spec.components.len(), 7);
    assert_eq!(spec.coordinate_labels, vec!["center_ear"]);
    assert_eq!(spec.components[2].label, "left_ray_hits_object");
}
