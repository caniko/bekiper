use std::collections::HashMap;

use polars::prelude::*;

use bekiper_behaviour::object_recognition::{NortWindowMode, ObjectRecognition};
use bekiper_behaviour::{BehaviouralTask, TaskDetail};
use bekiper_core::types::{CoordinateColumns, ObjectRecognitionProtocol};
use bekiper_feature::heuristic::combined::merge::MergeRule;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::object_shape::ObjectShape;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_perimeter::rectangle::Rectangle;

/// Build a synthetic DataFrame with columns the exploration + qualia heuristics need.
fn synthetic_tracking_df(n: usize) -> LazyFrame {
    let nose_x: Vec<f64> = (0..n).map(|i| (i as f64) * 5.0 / n as f64).collect();
    let nose_y: Vec<f64> = (0..n).map(|i| (i as f64) * 5.0 / n as f64).collect();

    let center_ear_x: Vec<f64> = nose_x.iter().map(|x| x - 0.03).collect();
    let center_ear_y: Vec<f64> = nose_y.clone();
    let tail_base_x: Vec<f64> = nose_x.iter().map(|x| x - 0.1).collect();
    let tail_base_y: Vec<f64> = nose_y.clone();
    // Body centroid slightly behind the head
    let mouse_center_x: Vec<f64> = nose_x.iter().map(|x| x - 0.06).collect();
    let mouse_center_y: Vec<f64> = nose_y.clone();

    // Ears: offset laterally from center_ear
    let left_ear_x: Vec<f64> = center_ear_x.iter().map(|x| x - 0.01).collect();
    let left_ear_y: Vec<f64> = center_ear_y.iter().map(|y| y + 0.01).collect();
    let right_ear_x: Vec<f64> = center_ear_x.iter().map(|x| x - 0.01).collect();
    let right_ear_y: Vec<f64> = center_ear_y.iter().map(|y| y - 0.01).collect();

    let snout_dx: Vec<f64> = center_ear_x
        .iter()
        .zip(nose_x.iter())
        .map(|(ce, n)| n - ce)
        .collect();
    let snout_dy: Vec<f64> = center_ear_y
        .iter()
        .zip(nose_y.iter())
        .map(|(ce, n)| n - ce)
        .collect();
    // Ear direction vectors: center_ear → left_ear, center_ear → right_ear
    let left_ear_dx: Vec<f64> = left_ear_x
        .iter()
        .zip(center_ear_x.iter())
        .map(|(l, c)| l - c)
        .collect();
    let left_ear_dy: Vec<f64> = left_ear_y
        .iter()
        .zip(center_ear_y.iter())
        .map(|(l, c)| l - c)
        .collect();
    let right_ear_dx: Vec<f64> = right_ear_x
        .iter()
        .zip(center_ear_x.iter())
        .map(|(r, c)| r - c)
        .collect();
    let right_ear_dy: Vec<f64> = right_ear_y
        .iter()
        .zip(center_ear_y.iter())
        .map(|(r, c)| r - c)
        .collect();

    df! {
        "nose_x" => &nose_x,
        "nose_y" => &nose_y,
        "center_ear_x" => &center_ear_x,
        "center_ear_y" => &center_ear_y,
        "mouse_center_x" => &mouse_center_x,
        "mouse_center_y" => &mouse_center_y,
        "tail_base_x" => &tail_base_x,
        "tail_base_y" => &tail_base_y,
        "left_ear_x" => &left_ear_x,
        "left_ear_y" => &left_ear_y,
        "right_ear_x" => &right_ear_x,
        "right_ear_y" => &right_ear_y,
        "snout_dx" => &snout_dx,
        "snout_dy" => &snout_dy,
        "left_ear_dx" => &left_ear_dx,
        "left_ear_dy" => &left_ear_dy,
        "right_ear_dx" => &right_ear_dx,
        "right_ear_dy" => &right_ear_dy,
    }
    .unwrap()
    .lazy()
}

fn imen_tracking_df(n: usize) -> LazyFrame {
    let nose_x: Vec<f64> = (0..n).map(|i| (i as f64) * 2.0 / n as f64).collect();
    let nose_y: Vec<f64> = vec![0.0; n];
    let center_ear_x: Vec<f64> = nose_x.iter().map(|x| x - 0.03).collect();
    let center_ear_y: Vec<f64> = nose_y.clone();
    let tail_x: Vec<f64> = center_ear_x.iter().map(|x| x - 0.08).collect();
    let tail_y: Vec<f64> = nose_y.clone();
    let left_ear_x: Vec<f64> = center_ear_x.iter().map(|x| x - 0.01).collect();
    let left_ear_y: Vec<f64> = vec![0.01; n];
    let right_ear_x: Vec<f64> = center_ear_x.iter().map(|x| x - 0.01).collect();
    let right_ear_y: Vec<f64> = vec![-0.01; n];
    let snout_dx: Vec<f64> = nose_x
        .iter()
        .zip(center_ear_x.iter())
        .map(|(nose, head)| nose - head)
        .collect();
    let snout_dy: Vec<f64> = vec![0.0; n];
    let left_ear_dx: Vec<f64> = left_ear_x
        .iter()
        .zip(center_ear_x.iter())
        .map(|(ear, head)| ear - head)
        .collect();
    let left_ear_dy: Vec<f64> = left_ear_y
        .iter()
        .zip(center_ear_y.iter())
        .map(|(ear, head)| ear - head)
        .collect();
    let right_ear_dx: Vec<f64> = right_ear_x
        .iter()
        .zip(center_ear_x.iter())
        .map(|(ear, head)| ear - head)
        .collect();
    let right_ear_dy: Vec<f64> = right_ear_y
        .iter()
        .zip(center_ear_y.iter())
        .map(|(ear, head)| ear - head)
        .collect();

    df! {
        "nose_x" => &nose_x,
        "nose_y" => &nose_y,
        "center_ear_x" => &center_ear_x,
        "center_ear_y" => &center_ear_y,
        "tail_x" => &tail_x,
        "tail_y" => &tail_y,
        "left_ear_x" => &left_ear_x,
        "left_ear_y" => &left_ear_y,
        "right_ear_x" => &right_ear_x,
        "right_ear_y" => &right_ear_y,
        "snout_dx" => &snout_dx,
        "snout_dy" => &snout_dy,
        "left_ear_dx" => &left_ear_dx,
        "left_ear_dy" => &left_ear_dy,
        "right_ear_dx" => &right_ear_dx,
        "right_ear_dy" => &right_ear_dy,
    }
    .unwrap()
    .lazy()
}

fn circle_perimeter(label: &str, cx: f64, cy: f64, r: f64) -> Perimeter<ObjectShape> {
    Perimeter::new(label, ObjectShape::Circle(Circle::new(cx, cy, r)))
}

fn rectangle_perimeter(label: &str, cx: f64, cy: f64, w: f64, h: f64) -> Perimeter<ObjectShape> {
    Perimeter::new(label, Rectangle::new(cx, cy, w, h).into())
}

fn delayed_object_tracking_df(n: usize, approach_frame: usize) -> LazyFrame {
    let nose_x: Vec<f64> = (0..n)
        .map(|idx| if idx < approach_frame { 2.0 } else { 1.0 })
        .collect();
    let nose_y: Vec<f64> = vec![0.0; n];
    let center_ear_x: Vec<f64> = nose_x.iter().map(|x| x + 0.1).collect();
    let center_ear_y: Vec<f64> = vec![0.0; n];
    let tail_base_x: Vec<f64> = center_ear_x.iter().map(|x| x + 0.1).collect();
    let tail_base_y: Vec<f64> = vec![0.0; n];
    let mouse_center_x: Vec<f64> = center_ear_x.iter().map(|x| x + 0.05).collect();
    let mouse_center_y: Vec<f64> = vec![0.0; n];
    let left_ear_x: Vec<f64> = center_ear_x.iter().map(|x| x + 0.01).collect();
    let left_ear_y: Vec<f64> = vec![0.01; n];
    let right_ear_x: Vec<f64> = center_ear_x.iter().map(|x| x + 0.01).collect();
    let right_ear_y: Vec<f64> = vec![-0.01; n];

    df! {
        "nose_x" => &nose_x,
        "nose_y" => &nose_y,
        "center_ear_x" => &center_ear_x,
        "center_ear_y" => &center_ear_y,
        "mouse_center_x" => &mouse_center_x,
        "mouse_center_y" => &mouse_center_y,
        "tail_base_x" => &tail_base_x,
        "tail_base_y" => &tail_base_y,
        "left_ear_x" => &left_ear_x,
        "left_ear_y" => &left_ear_y,
        "right_ear_x" => &right_ear_x,
        "right_ear_y" => &right_ear_y,
        "snout_dx" => &vec![-1.0; n],
        "snout_dy" => &vec![0.0; n],
        "left_ear_dx" => &vec![-1.0; n],
        "left_ear_dy" => &vec![0.0; n],
        "right_ear_dx" => &vec![-1.0; n],
        "right_ear_dy" => &vec![0.0; n],
    }
    .unwrap()
    .lazy()
}

fn center_periphery_df(fps: usize) -> LazyFrame {
    let n = fps * 2;
    let center_frames = fps / 2;
    let centroid_x: Vec<f64> = (0..n)
        .map(|idx| if idx < center_frames { 0.0 } else { 2.0 })
        .collect();
    let centroid_y: Vec<f64> = vec![0.0; n];

    df! {
        "mouse_center_x" => &centroid_x,
        "mouse_center_y" => &centroid_y,
        "center_ear_x" => &centroid_x,
        "center_ear_y" => &centroid_y,
    }
    .unwrap()
    .lazy()
}

fn short_freezing_df() -> LazyFrame {
    let n = 20;
    let head_x: Vec<f64> = (0..n)
        .map(|idx| if idx < 15 { 0.0 } else { idx as f64 })
        .collect();
    let head_y: Vec<f64> = vec![0.0; n];

    df! {
        "center_ear_x" => &head_x,
        "center_ear_y" => &head_y,
        "mouse_center_x" => &head_x,
        "mouse_center_y" => &head_y,
    }
    .unwrap()
    .lazy()
}

#[test]
fn object_recognition_analyze_produces_summaries() {
    let task = ObjectRecognition::new(vec![circle_perimeter("obj1", 5.0, 5.0, 0.5)], 30.0);

    let lf = synthetic_tracking_df(100);
    let result = task.analyze(lf, &bekiper_core::video::FrameTiming::cfr(30.0));

    assert_eq!(
        result.task_type,
        bekiper_core::types::TaskType::ObjectRecognition
    );
    // 1 exploration + 3 qualia (body_prox, olfaction, whiskers) + 1 freezing = 5
    assert_eq!(result.summaries.len(), 5);

    for s in &result.summaries {
        assert_eq!(s.total_frames, 100);
        assert!(s.true_frames <= 100);
        assert!(s.seconds >= 0.0);
    }

    let df = result.evaluated_df.as_ref().unwrap();
    assert_eq!(df.height(), 100);
}

#[test]
fn object_recognition_accepts_imen_coordinate_overrides_without_superanimal_columns() {
    let mut task = ObjectRecognition::new(vec![circle_perimeter("obj1", 1.0, 0.0, 0.3)], 30.0);
    task.params.centroid_coords = CoordinateColumns::new("center_ear_x", "center_ear_y");
    task.params.head_coords = CoordinateColumns::new("center_ear_x", "center_ear_y");
    task.params.tail_coords = CoordinateColumns::new("tail_x", "tail_y");

    let result = task.analyze(
        imen_tracking_df(60),
        &bekiper_core::video::FrameTiming::cfr(30.0),
    );

    assert_eq!(
        result.task_type,
        bekiper_core::types::TaskType::ObjectRecognition
    );
    assert!(result.evaluated_df.is_some());
}

#[test]
fn object_recognition_with_two_objects() {
    let task = ObjectRecognition::new(
        vec![
            circle_perimeter("obj1", 5.0, 5.0, 0.5),
            circle_perimeter("obj2", 1.0, 1.0, 0.3),
        ],
        30.0,
    );

    let lf = synthetic_tracking_df(50);
    let result = task.analyze(lf, &bekiper_core::video::FrameTiming::cfr(30.0));

    // 2 exploration + 6 qualia (3 per object) + 1 freezing = 9
    assert_eq!(result.summaries.len(), 9);
}

#[test]
fn object_recognition_metric_units_populate_shared_bundles() {
    let mut task = ObjectRecognition::new(
        vec![
            circle_perimeter("object_1", 1.0, 1.0, 0.3),
            circle_perimeter("object_2", 4.2, 4.2, 0.3),
            circle_perimeter("arena", 2.5, 2.5, 3.0),
        ],
        30.0,
    );
    task.protocol = ObjectRecognitionProtocol::OulTraining;
    task.center_perimeter = Some(rectangle_perimeter("center_zone", 2.5, 2.5, 1.5, 1.5));

    let nort = task.full_analyze(synthetic_tracking_df(180));

    let totals = nort
        .observation_totals
        .as_ref()
        .expect("observation totals should be populated");
    assert_eq!(totals.object_labels, vec!["object_1", "object_2"]);
    assert_eq!(totals.observation_seconds.len(), 2);
    assert_eq!(totals.observation_bouts.len(), 2);
    assert!(totals.total_seconds >= 0.0);
    assert_eq!(
        totals.total_seconds,
        totals.observation_seconds.iter().sum::<f64>()
    );

    assert_eq!(nort.pairwise_observations.len(), 1);
    let pair = &nort.pairwise_observations[0];
    assert_eq!(pair.lhs_label, "object_1");
    assert_eq!(pair.rhs_label, "object_2");
    assert!(
        (pair.absolute_delta_seconds
            - (totals.observation_seconds[0] - totals.observation_seconds[1]).abs())
        .abs()
            < 1e-9
    );

    assert_eq!(nort.qualia_rollups.len(), 2);
    for rollup in &nort.qualia_rollups {
        assert!(matches!(rollup.label.as_str(), "object_1" | "object_2"));
        assert!(rollup.body_total_seconds >= 0.0);
        assert!(rollup.olfaction_total_seconds >= 0.0);
        assert!(rollup.whisker_total_seconds >= 0.0);
    }

    let zone_metrics = nort
        .zone_metrics
        .as_ref()
        .expect("zone metrics should be populated when arena/center exist");
    assert!(zone_metrics.all_motion_summary.is_some());
    assert!(zone_metrics.center_summary.is_some());
    assert!(zone_metrics.periphery_summary.is_some());
    assert!(zone_metrics.center_motion_summary.is_some());
    assert!(zone_metrics.periphery_motion_summary.is_some());
    assert_eq!(zone_metrics.quadrant_summaries.len(), 4);
    assert_eq!(zone_metrics.quadrant_motion_summaries.len(), 4);
}

#[test]
fn object_recognition_outputs_only_actual_objects() {
    let mut task = ObjectRecognition::new(
        vec![
            circle_perimeter("object_1", 1.0, 1.0, 0.3),
            circle_perimeter("object_2", 4.2, 4.2, 0.3),
            circle_perimeter("arena", 2.5, 2.5, 3.0),
        ],
        30.0,
    );
    task.protocol = ObjectRecognitionProtocol::OulTraining;
    task.center_perimeter = Some(rectangle_perimeter("center_zone", 2.5, 2.5, 1.5, 1.5));

    let analysis = task.analyze(
        synthetic_tracking_df(180),
        &bekiper_core::video::FrameTiming::cfr(30.0),
    );
    let summary_names: Vec<String> = analysis
        .summaries
        .iter()
        .map(|summary| summary.name.clone())
        .collect();
    assert!(
        summary_names
            .iter()
            .any(|name| name == "exploring_object_1")
    );
    assert!(
        summary_names
            .iter()
            .any(|name| name == "exploring_object_2")
    );
    assert!(summary_names.iter().any(|name| name == "in_center"));
    assert!(summary_names.iter().any(|name| name == "in_periphery"));
    assert!(summary_names.iter().any(|name| name == "freezing"));
    assert!(
        !summary_names
            .iter()
            .any(|name| name.contains("arena") || name.contains("center_zone")),
        "context perimeters must not emit primary heuristics: {summary_names:?}"
    );

    let nort = task.full_analyze(synthetic_tracking_df(180));
    assert_eq!(nort.object_labels, vec!["object_1", "object_2"]);
    assert_eq!(nort.exploration_times.len(), 2);
    assert_eq!(nort.bout_counts.len(), 2);
    let qualia = nort
        .qualia
        .as_ref()
        .expect("qualia should remain populated");
    for channel in &qualia.channels {
        assert_eq!(channel.per_object_seconds.len(), 2);
        assert_eq!(channel.per_object_bouts.len(), 2);
    }
}

#[test]
fn object_recognition_first_approach_uses_actual_objects_only() {
    let mut task = ObjectRecognition::new(
        vec![
            circle_perimeter("object_1", 0.0, 0.0, 0.5),
            circle_perimeter("arena", 0.0, 0.0, 5.0),
        ],
        30.0,
    );
    task.protocol = ObjectRecognitionProtocol::OulTraining;
    task.nose_max_distance = 0.6;
    task.center_perimeter = Some(rectangle_perimeter("center_zone", 0.0, 0.0, 1.0, 1.0));

    let nort = task.full_analyze(delayed_object_tracking_df(90, 40));
    assert_eq!(nort.object_labels, vec!["object_1"]);
    assert_eq!(nort.first_approach_frame, Some(41));
}

#[test]
fn object_recognition_window_anchors_to_first_approach_when_full_window_fits() {
    let mut task = ObjectRecognition::new(vec![circle_perimeter("object_1", 0.0, 0.0, 0.5)], 30.0);
    task.protocol = ObjectRecognitionProtocol::OulTraining;
    task.nose_max_distance = 0.6;
    task.analysis_window_seconds = 1.0;

    let nort = task.full_analyze(delayed_object_tracking_df(90, 40));
    assert_eq!(nort.first_approach_frame, Some(41));
    assert_eq!(nort.window_mode, NortWindowMode::AnchoredFirstApproach);
    assert_eq!(nort.window_start_frame, Some(41));
    assert_eq!(nort.window_end_frame_exclusive, Some(71));
    assert!((nort.total_duration_seconds - 1.0).abs() < 1e-9);
}

#[test]
fn object_recognition_window_leads_from_trial_start_when_no_approach_exists() {
    let mut task = ObjectRecognition::new(
        vec![
            circle_perimeter("object_1", 100.0, 100.0, 0.1),
            circle_perimeter("object_2", 110.0, 110.0, 0.1),
        ],
        30.0,
    );
    task.protocol = ObjectRecognitionProtocol::OulTraining;
    task.analysis_window_seconds = 1.0;

    let nort = task.full_analyze(synthetic_tracking_df(90));
    assert_eq!(nort.first_approach_frame, None);
    assert_eq!(nort.window_mode, NortWindowMode::LeadingNoApproach);
    assert_eq!(nort.window_start_frame, Some(0));
    assert_eq!(nort.window_end_frame_exclusive, Some(30));
    assert!((nort.total_duration_seconds - 1.0).abs() < 1e-9);
}

#[test]
fn object_recognition_window_backshifts_when_approach_is_too_late() {
    let mut task = ObjectRecognition::new(vec![circle_perimeter("object_1", 0.0, 0.0, 0.5)], 30.0);
    task.protocol = ObjectRecognitionProtocol::OulTraining;
    task.nose_max_distance = 0.6;
    task.analysis_window_seconds = 2.0;

    let nort = task.full_analyze(delayed_object_tracking_df(90, 80));
    assert_eq!(nort.first_approach_frame, Some(81));
    assert_eq!(nort.window_mode, NortWindowMode::BackshiftedLateApproach);
    assert_eq!(nort.window_start_frame, Some(30));
    assert_eq!(nort.window_end_frame_exclusive, Some(90));
    assert!((nort.total_duration_seconds - 2.0).abs() < 1e-9);
}

#[test]
fn object_recognition_window_uses_full_trial_when_trial_is_shorter_than_window() {
    let mut task = ObjectRecognition::new(vec![circle_perimeter("object_1", 0.0, 0.0, 0.5)], 30.0);
    task.protocol = ObjectRecognitionProtocol::OulTraining;
    task.nose_max_distance = 0.6;
    task.analysis_window_seconds = 2.0;

    let nort = task.full_analyze(delayed_object_tracking_df(30, 10));
    assert_eq!(nort.window_mode, NortWindowMode::FullTrialShortTrial);
    assert_eq!(nort.window_start_frame, Some(0));
    assert_eq!(nort.window_end_frame_exclusive, Some(30));
    assert!((nort.total_duration_seconds - 1.0).abs() < 1e-9);
}

#[test]
fn object_recognition_full_analysis_mode_ignores_internal_window_but_reports_first_approach() {
    let mut task = ObjectRecognition::new(vec![circle_perimeter("object_1", 0.0, 0.0, 0.5)], 30.0);
    task.protocol = ObjectRecognitionProtocol::OulTraining;
    task.nose_max_distance = 0.6;
    task.analysis_window_seconds = 1.0;
    task.window_mode = NortWindowMode::FullAnalysis;

    let nort = task.full_analyze(delayed_object_tracking_df(90, 40));
    assert_eq!(nort.first_approach_frame, Some(41));
    assert_eq!(nort.first_object_approach_time_s, Some(41.0 / 30.0));
    assert_eq!(nort.window_mode, NortWindowMode::FullAnalysis);
    assert_eq!(nort.window_start_frame, Some(0));
    assert_eq!(nort.window_end_frame_exclusive, Some(90));
    assert!((nort.total_duration_seconds - 3.0).abs() < 1e-9);
    assert!(!nort.excluded);
    assert_eq!(
        nort.passes_15s_exploration_threshold,
        nort.observation_totals
            .as_ref()
            .is_some_and(|totals| totals.total_seconds >= task.min_total_exploration)
    );
}

fn two_object_task_with_arena() -> ObjectRecognition {
    let mut task = ObjectRecognition::new(
        vec![
            circle_perimeter("object_1", 1.0, 1.0, 0.3),
            circle_perimeter("object_2", 4.2, 4.2, 0.3),
            rectangle_perimeter("arena", 2.5, 2.5, 6.0, 6.0),
        ],
        30.0,
    );
    task.center_perimeter = Some(rectangle_perimeter("center_zone", 2.5, 2.5, 1.5, 1.5));
    task.window_mode = NortWindowMode::FullAnalysis;
    task
}

#[test]
fn object_recognition_detailed_analysis_reports_merge_rule_comparisons() {
    let mut task = two_object_task_with_arena();
    task.perimeter_erosion_meters = 0.0;
    // Brief passes count, so every qualia channel observes some exploration.
    task.min_bout_seconds = 0.0;
    task.merge_rules = HashMap::from([
        (
            "close".to_string(),
            MergeRule {
                require: vec!["olfaction".into()],
                any_of: vec!["whisker_interaction".into()],
                exclude: vec!["body_proximity".into()],
            },
        ),
        ("empty".to_string(), MergeRule::default()),
    ]);
    let timing = bekiper_core::video::FrameTiming::cfr(30.0);
    let (analysis, detail) = task.analyze_detailed(synthetic_tracking_df(180), &timing);
    assert_eq!(
        analysis.task_type,
        bekiper_core::types::TaskType::ObjectRecognition
    );
    let detail = detail.unwrap();
    assert_eq!(
        detail.task_type(),
        bekiper_core::types::TaskType::ObjectRecognition
    );
    let names: Vec<_> = analysis.summaries.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"close_object_1"), "{names:?}");
    assert!(names.contains(&"in_periphery"), "{names:?}");
    let decompositions: Vec<_> = analysis
        .decompositions
        .iter()
        .map(|d| d.heuristic_name.as_str())
        .collect();
    assert!(decompositions.contains(&"fusion_comparison_close_object_2"));
    // Empty rules produce no fusion comparison.
    assert!(!decompositions.contains(&"fusion_comparison_empty_object_1"));
    let close = analysis
        .decompositions
        .iter()
        .find(|d| d.heuristic_name == "close_object_1")
        .unwrap();
    let components: Vec<_> = close.components.iter().map(|c| c.label.as_str()).collect();
    assert!(
        components.contains(&"exclude: body_proximity_object_1"),
        "{components:?}"
    );
    let df = analysis.evaluated_df.unwrap();
    assert!(
        df.get_column_names()
            .iter()
            .any(|n| n.contains("fusion_comparison_close_object_1"))
    );

    let TaskDetail::ObjectRecognition(nort) = &detail else {
        unreachable!()
    };
    let qualia = nort.qualia.as_ref().unwrap();
    let relative = &qualia.relative_bias["olfaction"];
    assert!(
        (relative.iter().sum::<f64>() - 100.0).abs() < 1e-9,
        "{relative:?}"
    );

    let quadrants = task.quadrants();
    assert_eq!(quadrants.len(), 1);
    assert_eq!(quadrants[0].label, "object_recognition_grid");
    assert_eq!((quadrants[0].x_min, quadrants[0].x_max), (-0.5, 5.5));

    // Without an arena there is no occupancy grid, and with qualia disabled
    // no fusion comparisons are emitted.
    let mut no_arena = ObjectRecognition::new(vec![circle_perimeter("o", 1.0, 1.0, 0.3)], 30.0);
    assert!(no_arena.quadrants().is_empty());
    no_arena.enable_qualia = false;
    no_arena.merge_rules = task.merge_rules.clone();
    let (analysis, _) = no_arena.analyze_detailed(synthetic_tracking_df(30), &timing);
    assert!(
        analysis
            .decompositions
            .iter()
            .all(|d| !d.heuristic_name.starts_with("fusion_comparison"))
    );
}

#[test]
fn object_recognition_empty_recording_reports_zero_rates() {
    let task = two_object_task_with_arena();
    let result = task.full_analyze(synthetic_tracking_df(0));
    assert_eq!(result.total_duration_seconds, 0.0);
    assert_eq!(result.total_mean_speed, 0.0);
    assert_eq!(result.exploration_times, [0.0, 0.0]);
    let qualia = result.qualia.as_ref().unwrap();
    assert_eq!(qualia.absolute_bias.len(), 3);
    for bias in qualia
        .absolute_bias
        .values()
        .chain(qualia.relative_bias.values())
    {
        assert_eq!(bias, &[0.0, 0.0]);
    }
}

#[test]
fn nort_window_modes_have_stable_names() {
    let names: Vec<_> = [
        NortWindowMode::Unknown,
        NortWindowMode::FullAnalysis,
        NortWindowMode::AnchoredFirstApproach,
        NortWindowMode::LeadingNoApproach,
        NortWindowMode::BackshiftedLateApproach,
        NortWindowMode::FullTrialShortTrial,
        NortWindowMode::ConfiguredFullTrial,
    ]
    .into_iter()
    .map(NortWindowMode::as_str)
    .collect();
    assert_eq!(
        names,
        [
            "unknown",
            "full_analysis",
            "anchored_first_approach",
            "leading_no_approach",
            "backshifted_late_approach",
            "full_trial_short_trial",
            "configured_full_trial",
        ]
    );
}
