//! NORT criteria: the head angle is measured to the nose's nearest perimeter
//! point; frames failing angle and contact count as removed by angle; posture
//! frames need the angle because contact no longer rescues them.

use polars::prelude::*;

use bekiper_core::video::FrameTiming;
use bekiper_feature::heuristic::combined::exploration::{BoutFilterMode, ExplorationHeuristic};
use bekiper_feature::heuristic::{Heuristic, diag_column_name};
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;

fn heuristic(posture: bool) -> ExplorationHeuristic<Circle> {
    let mut h = ExplorationHeuristic::new(
        Perimeter::new("obj", Circle::new(0.0, 0.0, 0.02)),
        "novel",
        (0.0, 0.0),
        10.0,
    );
    h.max_angle_deg = 45.0;
    h.speed_filter_mode = BoutFilterMode::Disabled;
    h.posture_col = posture.then(|| "posture".to_string());
    h
}

/// Nose 1 cm (frame 0) or 0.4 cm (frames 1-3, contact) right of a 2 cm-radius
/// object; heading 60° off the nearest point except frame 3 (head-on).
fn frames() -> DataFrame {
    let (c, s) = (60f64.to_radians().cos(), 60f64.to_radians().sin());
    df! {
        "nose_x" => [0.03, 0.024, 0.024, 0.024],
        "nose_y" => [0.0; 4],
        "mouse_center_x" => [0.08; 4],
        "mouse_center_y" => [0.0; 4],
        "center_ear_x" => [0.04; 4],
        "center_ear_y" => [0.0; 4],
        "snout_dx" => [-c, -c, -c, -1.0],
        "snout_dy" => [s, s, s, 0.0],
        "posture" => [false, false, true, true],
    }
    .unwrap()
}

#[test]
fn angle_is_to_the_nearest_perimeter_point_not_any_visible_point() {
    let h = heuristic(false);
    let df = h.evaluate(frames().lazy()).collect().unwrap();
    let angle: Vec<_> = df[diag_column_name("exploring_novel", "within_angle_deg").as_str()]
        .f64()
        .unwrap()
        .into_iter()
        .collect();
    for (got, want) in angle.iter().zip([60.0, 60.0, 60.0, 0.0]) {
        assert!((got.unwrap() - want).abs() < 1e-9, "{angle:?}");
    }
    // At 1 cm the object spans ±41.8° around the nearest point, so a cone of
    // 45° would still see it at 60°; the nearest-point angle rejects it.
    let within: Vec<_> = df[diag_column_name("exploring_novel", "within_angle").as_str()]
        .bool()
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(
        within,
        vec![Some(false), Some(false), Some(false), Some(true)]
    );
    let raw: Vec<_> = df["exploring_novel_raw"]
        .bool()
        .unwrap()
        .into_iter()
        .collect();
    // Contact rescues frames 1 and 2 without a posture exclusion.
    assert_eq!(raw, vec![Some(false), Some(true), Some(true), Some(true)]);
    let a = h.attribution_summary(&df, &FrameTiming::cfr(10.0));
    assert_eq!(a.candidate_frames_removed_by_angle_criterion, 1);
    assert_eq!(a.candidate_frames_removed_by_posture, 0);
}

#[test]
fn posture_frames_need_the_head_angle_contact_alone_does_not_rescue() {
    let h = heuristic(true);
    let df = h.evaluate(frames().lazy()).collect().unwrap();
    let raw: Vec<_> = df["exploring_novel_raw"]
        .bool()
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(raw, vec![Some(false), Some(true), Some(false), Some(true)]);
    let a = h.attribution_summary(&df, &FrameTiming::cfr(10.0));
    assert_eq!(a.candidate_frames_removed_by_angle_criterion, 1);
    assert_eq!(a.candidate_frames_removed_by_posture, 1);
}
