use polars::prelude::*;

use bekiper_core::types::CoordinateColumns;
use bekiper_core::video::FrameTiming;
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::combined::exploration::{
    BoutFilterMode, BoutFilterSpec, ExplorationHeuristic, object_exploration_with_timing,
};
use bekiper_feature::heuristic::diag_column_name;
use bekiper_feature::heuristic::solo::whiskers::{
    WhiskerFanHeuristic, WhiskerFanParams, WhiskerFanSpec,
};
use bekiper_feature::tolerance::BoutTiming;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::object_shape::ObjectShape;
use bekiper_perimeter::perimeter::Perimeter;

const FPS: f64 = 10.0;
const N: usize = 12;

/// Object: 2 cm-radius disc at the origin. The animal heads +x at y = -3 cm, so
/// the left fan (apex 80% to the nose, 11.7 mm) opens toward the object. The
/// nose is within 3 cm but the head points along the object, outside 30 degrees.
fn frames(centroid_x: f64, centroid_y: f64, posture: bool) -> DataFrame {
    let c = |v: f64| vec![v; N];
    df! {
        "nose_x" => c(0.006), "nose_y" => c(-0.03),
        "center_ear_x" => c(-0.004), "center_ear_y" => c(-0.03),
        "mouse_center_x" => c(centroid_x), "mouse_center_y" => c(centroid_y),
        "snout_dx" => c(1.0), "snout_dy" => c(0.0),
        "posture" => vec![posture; N],
    }
    .unwrap()
}

fn heuristic(whisker: bool) -> ExplorationHeuristic<Circle> {
    let mut h = ExplorationHeuristic::new(
        Perimeter::new("obj", Circle::new(0.0, 0.0, 0.02)),
        "novel",
        (0.0, 0.0),
        FPS,
    );
    h.speed_filter_mode = BoutFilterMode::Disabled;
    h.min_bout_seconds = 0.5;
    if whisker {
        h.whisker = Some(WhiskerFanSpec::new(
            WhiskerFanParams::default(),
            &ObjectShape::Circle(Circle::new(0.0, 0.0, 0.02)),
        ));
    }
    h
}

fn raw_count(h: &ExplorationHeuristic<Circle>, df: DataFrame) -> usize {
    let out = h.evaluate(df.lazy()).collect().unwrap();
    out.column("exploring_novel_raw")
        .unwrap()
        .bool()
        .unwrap()
        .sum()
        .unwrap() as usize
}

#[test]
fn whisker_only_frames_count_and_disabled_is_unchanged() {
    assert_eq!(raw_count(&heuristic(false), frames(-0.05, -0.03, false)), 0);
    assert_eq!(raw_count(&heuristic(true), frames(-0.05, -0.03, false)), N);
    let off = heuristic(false)
        .evaluate(frames(-0.05, -0.03, false).lazy())
        .collect()
        .unwrap();
    assert!(
        off.column(&diag_column_name("exploring_novel", "whisker_hit"))
            .is_err()
    );
    let on = heuristic(true)
        .evaluate(frames(-0.05, -0.03, false).lazy())
        .collect()
        .unwrap();
    let side = on
        .column(&diag_column_name("exploring_novel", "whisker_side"))
        .unwrap();
    assert_eq!(side.str().unwrap().get(0), Some("left"));
}

#[test]
fn posture_frames_and_inside_centroid_are_not_rescued_by_whisker() {
    let mut h = heuristic(true);
    h.posture_col = Some("posture".into());
    assert_eq!(raw_count(&h, frames(-0.05, -0.03, true)), 0);
    assert_eq!(raw_count(&h, frames(-0.05, -0.03, false)), N);
    // Body centroid inside the object: climbing, never observation.
    assert_eq!(raw_count(&heuristic(true), frames(0.0, 0.0, false)), 0);
}

#[test]
fn any_whisker_frame_rescues_a_bout_from_the_median_angle_filter() {
    let mut h = heuristic(true);
    h.angle_filter_mode = BoutFilterMode::PerBoutMedian;
    let df = h
        .evaluate(frames(-0.05, -0.03, false).lazy())
        .collect()
        .unwrap();
    // Head angle is ~100 degrees, so only the whisker rescue can accept the bout.
    let summary = h.summary(&df, &FrameTiming::cfr(FPS));
    assert_eq!(summary.true_frames, N as u64);

    let clock = BoutTiming::cfr(FPS, N).unwrap();
    let angle = df
        .column(&diag_column_name("exploring_novel", "within_angle_deg"))
        .unwrap()
        .f64()
        .unwrap();
    let nose_contact = df
        .column(&diag_column_name("exploring_novel", "nose_contact"))
        .unwrap()
        .bool()
        .unwrap();
    let filters = |contact| BoutFilterSpec {
        angle_mode: BoutFilterMode::PerBoutMedian,
        angle_deg: Some(angle),
        nose_contact: contact,
        max_angle_deg: 45.0,
        ..BoutFilterSpec::unfiltered()
    };
    let run = |contact| {
        object_exploration_with_timing(
            &df,
            "exploring_novel_raw",
            &clock,
            0.5,
            1.0,
            0.0,
            filters(contact),
        )
        .1
    };
    assert_eq!(
        run(Some(nose_contact)),
        0,
        "nose contact alone does not rescue"
    );
    let bout_contact = df
        .column(&diag_column_name("exploring_novel", "bout_contact"))
        .unwrap()
        .bool()
        .unwrap();
    assert_eq!(run(Some(bout_contact)), 1);
}

#[test]
fn invalid_params_explain_the_valid_ranges() {
    let bad = WhiskerFanParams {
        apex_fraction: 2.0,
        ..Default::default()
    };
    assert!(
        bad.validate()
            .unwrap_err()
            .contains("apex_fraction in [0,1]")
    );
}

#[test]
fn fan_heuristic_reports_its_name() {
    let shape = ObjectShape::Circle(Circle::new(0.0, 0.0, 0.02));
    let h = WhiskerFanHeuristic {
        spec: WhiskerFanSpec::new(WhiskerFanParams::default(), &shape),
        ear_coords: CoordinateColumns::new("ex", "ey"),
        nose_coords: CoordinateColumns::new("nx", "ny"),
        name: "fan_a".into(),
        result_column: None,
        shared: None,
    };
    assert_eq!(h.name(), "fan_a");
}
