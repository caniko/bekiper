//! Y-maze zone outputs over the configured occupancy, first detection of both
//! landmarks, and freezing seconds from per-frame timing.

use bekiper_behaviour::radial_arm::YMaze;
use bekiper_core::video::FrameTiming;
use bekiper_feature::zone::ZoneOccupancy;
use bekiper_perimeter::{perimeter::Perimeter, polygon::Polygon, radial_maze::RadialMaze};
use polars::prelude::*;

fn maze() -> Perimeter<RadialMaze> {
    let center = vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)];
    let arms = vec![
        vec![(0.1, -0.2), (1.1, -0.2), (1.1, 0.2), (0.1, 0.2)],
        vec![(-0.2, 0.1), (0.2, 0.1), (0.2, 1.1), (-0.2, 1.1)],
        vec![(-1.1, -0.2), (-0.1, -0.2), (-0.1, 0.2), (-1.1, 0.2)],
    ];
    Perimeter::new(
        "maze",
        RadialMaze::new(
            Polygon::new(center),
            arms.into_iter().map(Polygon::new).collect(),
        ),
    )
}

fn track(head: &[Option<f64>], tail: &[Option<f64>]) -> LazyFrame {
    let zeros = vec![0.0; head.len()];
    df! {
        "center_ear_x" => head, "center_ear_y" => &zeros,
        "tail_base_x" => tail, "tail_base_y" => &zeros,
        "mouse_center_x" => head, "mouse_center_y" => &zeros,
    }
    .unwrap()
    .lazy()
}

#[test]
fn raw_occupancy_counts_short_visits_while_entries_keep_the_dwell() {
    // 2 fps: a 0.5 s arm-A visit, then a 1.0 s visit (the only valid entry).
    let x: Vec<_> = [0.0, 0.9, 0.0, 0.9, 0.9, 0.0].map(Some).into();
    let run = |occupancy| {
        let mut task = YMaze::new(maze(), 2.0);
        task.proximal_threshold = 0.45;
        task.zone_occupancy = occupancy;
        task.full_analyze(track(&x, &x))
    };
    let filtered = run(ZoneOccupancy::DwellFiltered);
    let raw = run(ZoneOccupancy::Raw);
    for result in [&filtered, &raw] {
        assert_eq!(result.analysis.arm_sequence, ["A"]);
    }
    let arm_a = |r: &bekiper_behaviour::radial_arm::YMazeResult| {
        (
            r.zone_summaries[0].entries,
            r.zone_summaries[0].duration_seconds,
        )
    };
    assert_eq!(arm_a(&filtered), (1, 1.0));
    assert_eq!(arm_a(&raw), (2, 1.5));
    // Centre frames (0, 2, 5) are all shorter than the dwell.
    assert_eq!(filtered.zone_summaries[3].duration_seconds, 0.0);
    assert_eq!(raw.zone_summaries[3].duration_seconds, 1.5);
    assert_eq!(
        filtered.analysis.arm_preference > 0.0,
        raw.analysis.arm_preference > 0.0
    );
    assert_eq!(
        raw.scoring_provenance.unwrap()["zone_occupancy"],
        serde_json::json!("raw")
    );
}

#[test]
fn analysis_starts_when_head_and_tail_are_both_detected() {
    let head = [Some(0.0), Some(0.0), Some(0.0), Some(0.0)];
    let tail = [None, None, Some(0.0), Some(0.0)];
    let result = YMaze::new(maze(), 2.0).full_analyze(track(&head, &tail));
    assert_eq!(
        result.scoring_provenance.unwrap()["analysis_start_frame"],
        serde_json::json!(2)
    );
    assert_eq!(result.analysis_duration_seconds, 1.0);
}

#[test]
fn freezing_seconds_follow_variable_frame_durations() {
    // A motionless mouse: every frame after the 1 s window warm-up is frozen.
    let x = [Some(0.0); 5];
    let timing = FrameTiming::from_timestamps(vec![0.0, 0.5, 1.5, 2.0, 3.0]);
    let task = YMaze::new(maze(), 2.0);
    let result = task.full_analyze_timed(track(&x, &x), &timing);
    // Frames 1-4 are frozen: 1.0 + 0.5 + 1.0 s, plus the final frame at the
    // nominal 4 intervals / 3 s rate (0.75 s), not 4 frames / 2 fps = 2 s.
    assert_eq!(result.freezing_seconds, 3.25);
}

#[test]
fn zero_percent_threshold_sits_at_each_arms_nearest_vertex() {
    let task = YMaze::new(maze(), 2.0);
    // Arm A: nearest vertex (0.1, ±0.2) → r² = 0.05; farthest (1.1, ±0.2) → 1.25.
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(close(task.arm_threshold_distances_sq(0.0)[0], 0.05));
    assert!(close(task.arm_threshold_distances_sq(1.0)[0], 1.25));
}
