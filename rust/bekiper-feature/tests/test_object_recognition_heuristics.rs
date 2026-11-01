use polars::prelude::*;

use bekiper_core::types::CoordinateColumns;
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::combined::exploration::ExplorationHeuristic;
use bekiper_feature::heuristic::solo::body_proximity::BodyProximityHeuristic;
use bekiper_feature::heuristic::solo::olfaction::OlfactionHeuristic;
use bekiper_feature::heuristic::solo::whiskers::WhiskerHeuristic;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;

fn exploration_df(direction: (f64, f64)) -> DataFrame {
    df! {
        "nose_x" => &[1.0, 1.0],
        "nose_y" => &[0.0, 0.0],
        "mouse_center_x" => &[1.2, 1.2],
        "mouse_center_y" => &[0.0, 0.0],
        "center_ear_x" => &[1.1, 1.1],
        "center_ear_y" => &[0.0, 0.0],
        "snout_dx" => &[direction.0, direction.0],
        "snout_dy" => &[direction.1, direction.1],
    }
    .unwrap()
}

fn olfaction_df(nose: (f64, f64), direction: (f64, f64)) -> DataFrame {
    df! {
        "nose_x" => &[nose.0, nose.0],
        "nose_y" => &[nose.1, nose.1],
        "snout_dx" => &[direction.0, direction.0],
        "snout_dy" => &[direction.1, direction.1],
    }
    .unwrap()
}

#[test]
fn exploration_requires_heading_toward_object() {
    let perimeter = Perimeter::new("obj", Circle::new(0.0, 0.0, 0.5));
    let mut on_angle = ExplorationHeuristic::new(perimeter.clone(), "obj", (0.0, 0.0), 30.0);
    on_angle.nose_max_distance = 0.6;
    on_angle.max_angle_deg = 20.0;

    let mut off_angle = ExplorationHeuristic::new(perimeter, "obj", (0.0, 0.0), 30.0);
    off_angle.nose_max_distance = 0.6;
    off_angle.max_angle_deg = 20.0;

    let on_df = on_angle
        .evaluate(exploration_df((-1.0, 0.0)).lazy())
        .collect()
        .unwrap();
    let off_df = off_angle
        .evaluate(exploration_df((0.0, 1.0)).lazy())
        .collect()
        .unwrap();

    let on = on_df.column("exploring_obj_raw").unwrap().bool().unwrap();
    let off = off_df.column("exploring_obj_raw").unwrap().bool().unwrap();
    let on_angle_deg = on_df
        .column("__bk_diag__exploring_obj__within_angle_deg")
        .unwrap()
        .f64()
        .unwrap();
    let on_margin_deg = on_df
        .column("__bk_diag__exploring_obj__within_angle_margin_deg")
        .unwrap()
        .f64()
        .unwrap();
    let off_angle_deg = off_df
        .column("__bk_diag__exploring_obj__within_angle_deg")
        .unwrap()
        .f64()
        .unwrap();
    let off_margin_deg = off_df
        .column("__bk_diag__exploring_obj__within_angle_margin_deg")
        .unwrap()
        .f64()
        .unwrap();

    assert!(
        on.get(1).unwrap(),
        "on-angle frame should count as exploration"
    );
    assert!(on_angle_deg.get(1).unwrap() < 20.0);
    assert!(on_margin_deg.get(1).unwrap() > 0.0);
    assert!(
        !off.get(1).unwrap(),
        "off-angle frame should fail exploration"
    );
    assert!(off_angle_deg.get(1).unwrap() > 20.0);
    assert!(off_margin_deg.get(1).unwrap() < 0.0);
}

#[test]
fn olfaction_requires_heading_unless_nose_is_in_contact() {
    let perimeter = Perimeter::new("obj", Circle::new(0.0, 0.0, 0.5));
    let on_angle = OlfactionHeuristic {
        perimeter: perimeter.clone(),
        max_distance: 0.6,
        max_angle: 20.0,
        contact_distance: 0.01,
        name: "olfaction_obj".into(),
        result_column: None,
    };
    let off_angle = OlfactionHeuristic {
        perimeter: perimeter.clone(),
        max_distance: 0.6,
        max_angle: 20.0,
        contact_distance: 0.01,
        name: "olfaction_obj".into(),
        result_column: None,
    };
    let contact_override = OlfactionHeuristic {
        perimeter,
        max_distance: 0.6,
        max_angle: 20.0,
        contact_distance: 0.01,
        name: "olfaction_obj".into(),
        result_column: None,
    };

    let on_df = on_angle
        .evaluate(olfaction_df((1.0, 0.0), (-1.0, 0.0)).lazy())
        .collect()
        .unwrap();
    let off_df = off_angle
        .evaluate(olfaction_df((1.0, 0.0), (0.0, 1.0)).lazy())
        .collect()
        .unwrap();
    let contact_df = contact_override
        .evaluate(olfaction_df((0.53, 0.0), (0.0, 1.0)).lazy())
        .collect()
        .unwrap();

    assert!(
        on_df
            .column("olfaction_obj")
            .unwrap()
            .bool()
            .unwrap()
            .get(1)
            .unwrap()
    );
    assert!(
        !off_df
            .column("olfaction_obj")
            .unwrap()
            .bool()
            .unwrap()
            .get(1)
            .unwrap()
    );
    assert!(
        contact_df
            .column("olfaction_obj")
            .unwrap()
            .bool()
            .unwrap()
            .get(1)
            .unwrap(),
        "direct nose contact should override noisy direction vectors"
    );
}

#[test]
fn olfaction_close_range_override_accepts_zero_heading_frames_inside_10mm_band() {
    let heuristic = OlfactionHeuristic {
        perimeter: Perimeter::new("obj", Circle::new(0.0, 0.0, 0.5)),
        max_distance: 0.6,
        max_angle: 20.0,
        contact_distance: 0.01,
        name: "olfaction_obj".into(),
        result_column: None,
    };

    let result = heuristic
        .evaluate(olfaction_df((0.5075, 0.0), (0.0, 0.0)).lazy())
        .collect()
        .unwrap();

    let nose_contact = result
        .column("__bk_diag__olfaction_obj__nose_contact")
        .unwrap()
        .bool()
        .unwrap();
    let ray_hits = result
        .column("__bk_diag__olfaction_obj__ray_hits_object")
        .unwrap()
        .bool()
        .unwrap();
    let combined = result.column("olfaction_obj").unwrap().bool().unwrap();

    assert_eq!(nose_contact.get(1), Some(true));
    assert_eq!(ray_hits.get(1), Some(false));
    assert_eq!(combined.get(1), Some(true));
}

#[test]
fn olfaction_close_range_override_rejects_zero_heading_frames_outside_10mm_band() {
    let heuristic = OlfactionHeuristic {
        perimeter: Perimeter::new("obj", Circle::new(0.0, 0.0, 0.5)),
        max_distance: 0.6,
        max_angle: 20.0,
        contact_distance: 0.01,
        name: "olfaction_obj".into(),
        result_column: None,
    };

    let result = heuristic
        .evaluate(olfaction_df((0.511, 0.0), (0.0, 0.0)).lazy())
        .collect()
        .unwrap();

    let nose_contact = result
        .column("__bk_diag__olfaction_obj__nose_contact")
        .unwrap()
        .bool()
        .unwrap();
    let ray_hits = result
        .column("__bk_diag__olfaction_obj__ray_hits_object")
        .unwrap()
        .bool()
        .unwrap();
    let combined = result.column("olfaction_obj").unwrap().bool().unwrap();

    assert_eq!(nose_contact.get(1), Some(false));
    assert_eq!(ray_hits.get(1), Some(false));
    assert_eq!(combined.get(1), Some(false));
}

#[test]
fn whisker_requires_same_side_proximity_and_heading() {
    let heuristic = WhiskerHeuristic {
        perimeter: Perimeter::new("obj", Circle::new(0.0, 0.0, 0.5)),
        max_distance: 0.5,
        max_angle: 20.0,
        name: "whisker_obj".into(),
        result_column: None,
    };

    let on_df = heuristic
        .evaluate(
            df! {
                "center_ear_x" => &[1.5, 1.5],
                "center_ear_y" => &[0.0, 0.0],
                "left_ear_x" => &[0.9, 0.9],
                "left_ear_y" => &[0.0, 0.0],
                "right_ear_x" => &[3.0, 3.0],
                "right_ear_y" => &[0.0, 0.0],
                "left_ear_dx" => &[-1.0, -1.0],
                "left_ear_dy" => &[0.0, 0.0],
                "right_ear_dx" => &[1.0, 1.0],
                "right_ear_dy" => &[0.0, 0.0],
            }
            .unwrap()
            .lazy(),
        )
        .collect()
        .unwrap();

    let cross_side_df = heuristic
        .evaluate(
            df! {
                "center_ear_x" => &[1.5, 1.5],
                "center_ear_y" => &[0.0, 0.0],
                "left_ear_x" => &[0.9, 0.9],
                "left_ear_y" => &[0.0, 0.0],
                "right_ear_x" => &[3.0, 3.0],
                "right_ear_y" => &[0.0, 0.0],
                "left_ear_dx" => &[0.0, 0.0],
                "left_ear_dy" => &[1.0, 1.0],
                "right_ear_dx" => &[-1.0, -1.0],
                "right_ear_dy" => &[0.0, 0.0],
            }
            .unwrap()
            .lazy(),
        )
        .collect()
        .unwrap();

    assert!(
        on_df
            .column("whisker_obj")
            .unwrap()
            .bool()
            .unwrap()
            .get(1)
            .unwrap()
    );
    // The inspection's left/right panels must describe the same-side gates,
    // including the rejected cross-side case, rather than rebuilding them.
    for (df, left, right) in [(&on_df, true, false), (&cross_side_df, false, false)] {
        assert_eq!(
            df.column("__bk_diag__whisker_obj__left_combined")
                .unwrap()
                .bool()
                .unwrap()
                .get(1),
            Some(left)
        );
        assert_eq!(
            df.column("__bk_diag__whisker_obj__right_combined")
                .unwrap()
                .bool()
                .unwrap()
                .get(1),
            Some(right)
        );
    }
    assert!(
        on_df
            .column("__bk_diag__whisker_obj__left_ray_min_angle_deg")
            .unwrap()
            .f64()
            .unwrap()
            .get(1)
            .unwrap()
            < 20.0
    );
    assert_eq!(
        on_df
            .column("__bk_diag__whisker_obj__selected_ray_dx")
            .unwrap()
            .f64()
            .unwrap()
            .get(1),
        Some(-1.0)
    );
    assert!(
        !cross_side_df
            .column("whisker_obj")
            .unwrap()
            .bool()
            .unwrap()
            .get(1)
            .unwrap(),
        "left-near/right-hit combinations must not count"
    );
    assert!(
        cross_side_df
            .column("__bk_diag__whisker_obj__left_ray_angle_margin_deg")
            .unwrap()
            .f64()
            .unwrap()
            .get(1)
            .unwrap()
            < 0.0
    );
    assert_eq!(
        cross_side_df
            .column("__bk_diag__whisker_obj__selected_ray_dx")
            .unwrap()
            .f64()
            .unwrap()
            .get(1),
        Some(0.0)
    );
    assert_eq!(
        cross_side_df
            .column("__bk_diag__whisker_obj__selected_ray_dy")
            .unwrap()
            .f64()
            .unwrap()
            .get(1),
        Some(1.0)
    );
}

#[test]
fn whisker_selected_ray_prefers_nearer_success_path_when_both_sides_are_near() {
    let heuristic = WhiskerHeuristic {
        perimeter: Perimeter::new("obj", Circle::new(0.0, 0.0, 0.5)),
        max_distance: 0.6,
        max_angle: 20.0,
        name: "whisker_obj".into(),
        result_column: None,
    };

    let df = heuristic
        .evaluate(
            df! {
                "center_ear_x" => &[1.2, 1.2],
                "center_ear_y" => &[0.0, 0.0],
                "left_ear_x" => &[0.9, 0.9],
                "left_ear_y" => &[0.0, 0.0],
                "right_ear_x" => &[0.95, 0.95],
                "right_ear_y" => &[0.0, 0.0],
                "left_ear_dx" => &[0.0, 0.0],
                "left_ear_dy" => &[1.0, 1.0],
                "right_ear_dx" => &[-1.0, -1.0],
                "right_ear_dy" => &[0.0, 0.0],
            }
            .unwrap()
            .lazy(),
        )
        .collect()
        .unwrap();

    assert_eq!(
        df.column("__bk_diag__whisker_obj__selected_ray_dx")
            .unwrap()
            .f64()
            .unwrap()
            .get(1),
        Some(-1.0)
    );
    assert_eq!(
        df.column("__bk_diag__whisker_obj__selected_ray_dy")
            .unwrap()
            .f64()
            .unwrap()
            .get(1),
        Some(0.0)
    );
    assert!(
        df.column("__bk_diag__whisker_obj__selected_ray_angle_margin_deg")
            .unwrap()
            .f64()
            .unwrap()
            .get(1)
            .unwrap()
            > 0.0
    );
}

#[test]
fn body_proximity_requires_head_led_contact() {
    let heuristic = BodyProximityHeuristic {
        torso_coords: None,
        perimeter: Perimeter::new("obj", Circle::new(0.0, 0.0, 0.5)),
        max_distance: 0.2,
        head_coords: CoordinateColumns::new("center_ear_x", "center_ear_y"),
        tail_coords: CoordinateColumns::new("tail_base_x", "tail_base_y"),
        name: "body_proximity_obj".into(),
        result_column: None,
    };

    let df = heuristic
        .evaluate(
            df! {
                "center_ear_x" => &[1.0, 1.0],
                "center_ear_y" => &[0.0, 0.0],
                "tail_base_x" => &[0.6, 0.6],
                "tail_base_y" => &[0.0, 0.0],
            }
            .unwrap()
            .lazy(),
        )
        .collect()
        .unwrap();

    assert!(
        !df.column("body_proximity_obj")
            .unwrap()
            .bool()
            .unwrap()
            .get(1)
            .unwrap(),
        "tail-only contact should remain diagnostic and not count as investigation"
    );
    assert!(
        df.column("__bk_diag__body_proximity_obj__tail_base_near")
            .unwrap()
            .bool()
            .unwrap()
            .get(1)
            .unwrap()
    );
}

#[test]
fn torso_is_retained_as_a_diagnostic_without_accepting_tail_led_contact() {
    let h = BodyProximityHeuristic {
        perimeter: Perimeter::new("obj", Circle::new(0.0, 0.0, 0.5)),
        max_distance: 0.2,
        head_coords: CoordinateColumns::new("head_x", "head_y"),
        tail_coords: CoordinateColumns::new("tail_x", "tail_y"),
        torso_coords: Some(CoordinateColumns::new("torso_x", "torso_y")),
        name: "body_obj".into(),
        result_column: None,
    };
    let result = h
        .evaluate(
            df!("head_x" => [1.0], "head_y" => [0.0], "tail_x" => [0.0], "tail_y" => [0.0],
                               "torso_x" => [0.6], "torso_y" => [0.0])
            .unwrap()
            .lazy(),
        )
        .collect()
        .unwrap();
    assert_eq!(
        result
            .column("__bk_diag__body_obj__torso_near")
            .unwrap()
            .bool()
            .unwrap()
            .get(0),
        Some(true)
    );
    assert_eq!(
        result.column("body_obj").unwrap().bool().unwrap().get(0),
        Some(false)
    );
    let component = h
        .decomposition()
        .unwrap()
        .components
        .into_iter()
        .find(|c| c.label == "torso_near")
        .unwrap();
    assert_eq!(component.buffer_meters, Some(0.2));
    assert!(
        matches!(component.buffer_perimeter, Some(bekiper_core::inspection::PerimeterSpec::Circle {radius, ..}) if (radius-0.7).abs()<1e-12)
    );
}

#[test]
fn exploration_zones_offset_the_real_outline_and_erosion_only_shrinks_the_footprint() {
    use bekiper_perimeter::polygon::Polygon;
    // 10 cm square object; criteria: 3 cm interaction zone, 0.5 cm contact,
    // 1 cm footprint erosion for the centroid-outside check only.
    let square = Polygon::new(vec![(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]);
    let mut heuristic =
        ExplorationHeuristic::new(Perimeter::new("obj", square), "obj", (0.05, 0.05), 30.0);
    heuristic.nose_max_distance = 0.03;
    heuristic.contact_distance = 0.005;
    heuristic.footprint_erosion_meters = 0.01;
    let frame = df! {
        // 2.5 cm, 0.4 cm and 3.1 cm from the right edge.
        "nose_x" => &[0.125, 0.104, 0.131],
        "nose_y" => &[0.05, 0.05, 0.05],
        // 0.5 cm inside the real edge, outside the eroded footprint.
        "mouse_center_x" => &[0.095, 0.095, 0.095],
        "mouse_center_y" => &[0.05, 0.05, 0.05],
        "center_ear_x" => &[0.14, 0.14, 0.14],
        "center_ear_y" => &[0.05, 0.05, 0.05],
        "snout_dx" => &[-1.0, -1.0, -1.0],
        "snout_dy" => &[0.0, 0.0, 0.0],
    }
    .unwrap();
    let out = heuristic.evaluate(frame.lazy()).collect().unwrap();
    let mask = |name: &str| -> Vec<bool> {
        out.column(&format!("__bk_diag__exploring_obj__{name}"))
            .unwrap()
            .bool()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap_or(false))
            .collect()
    };
    assert_eq!(mask("outside_object"), vec![true, true, true]);
    assert_eq!(mask("nose_near"), vec![true, true, false]);
    assert_eq!(mask("nose_contact"), vec![false, true, false]);
}
