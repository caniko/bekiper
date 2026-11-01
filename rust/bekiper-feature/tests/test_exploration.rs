use polars::prelude::*;

use bekiper_core::handoff;
use bekiper_core::video::FrameTiming;
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::combined::exploration::{
    BoutFilterMode, BoutFilterSpec, ExplorationHeuristic, IMEN_MAX_ANGLE_DEG, IMEN_MAX_GAP_SECONDS,
    IMEN_MIN_BOUT_SECONDS, NortBoutPolicy, accepts_bout_angle, object_exploration,
    object_exploration_with_timing,
};
use bekiper_feature::tolerance::BoutTiming;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;

const FPS: f64 = 10.0;

/// A 2 cm object at the origin. The nose rests 1 cm from it, heading in, for
/// frames 0..11, then the animal leaves; the body centroid stays outside.
fn heuristic() -> ExplorationHeuristic<Circle> {
    ExplorationHeuristic::new(
        Perimeter::new("obj", Circle::new(0.0, 0.0, 0.02)),
        "novel",
        (0.0, 0.0),
        FPS,
    )
}

fn visit(frames_near: usize, total: usize) -> DataFrame {
    let near = |near: f64, far: f64| -> Vec<f64> {
        (0..total)
            .map(|i| if i < frames_near { near } else { far })
            .collect()
    };
    df! {
        "nose_x" => near(0.03, 1.0),
        "nose_y" => vec![0.0; total],
        "mouse_center_x" => near(0.08, 1.05),
        "mouse_center_y" => vec![0.0; total],
        "center_ear_x" => near(0.04, 1.01),
        "center_ear_y" => vec![0.0; total],
        "snout_dx" => vec![-1.0; total],
        "snout_dy" => vec![0.0; total],
    }
    .unwrap()
}

fn evaluated(h: &ExplorationHeuristic<Circle>, df: DataFrame) -> DataFrame {
    h.evaluate(df.lazy()).collect().unwrap()
}

#[test]
fn bout_exactly_at_minimum_duration_is_scored() {
    let h = heuristic();
    assert_eq!(h.name(), "novel");
    let df = evaluated(&h, visit(11, 20));
    // Frame 0 has no previous nose position, so no speed: the bout is 1..11.
    let raw: Vec<_> = df
        .column("exploring_novel_raw")
        .unwrap()
        .bool()
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(raw.iter().filter(|v| **v == Some(true)).count(), 10);
    assert_eq!(raw[0], Some(false));
    let summary = h.summary(&df, &FrameTiming::cfr(FPS));
    assert_eq!(summary.name, "exploring_novel");
    assert_eq!((summary.true_frames, summary.total_frames), (10, 20));
    assert_eq!(summary.seconds, 1.0);

    let longer = ExplorationHeuristic {
        min_bout_seconds: 1.1,
        ..heuristic()
    };
    let rejected = longer.summary(&df, &FrameTiming::cfr(FPS));
    assert_eq!((rejected.true_frames, rejected.seconds), (0, 0.0));

    let missing = h.summary(&visit(11, 20), &FrameTiming::cfr(FPS));
    assert_eq!((missing.true_frames, missing.total_frames), (0, 0));
}

#[test]
fn attribution_counts_each_gate_and_the_final_bout() {
    let h = heuristic();
    let mut df = evaluated(&h, visit(11, 20));
    let without_final = h.attribution_summary(&df, &FrameTiming::cfr(FPS));
    assert_eq!(without_final.frames_all_gates_raw, 10);
    assert_eq!(without_final.final_valid_bouts, 0);
    assert!(without_final.median_head_angle_per_bout.is_empty());

    let clock = BoutTiming::cfr(FPS, df.height()).unwrap();
    let (seconds, bouts, mut final_mask) = object_exploration_with_timing(
        &df,
        "exploring_novel_raw",
        &clock,
        h.min_bout_seconds,
        h.max_gap_seconds,
        0.0,
        BoutFilterSpec::unfiltered(),
    );
    assert_eq!((seconds, bouts), (1.0, 1));
    final_mask.rename(h.final_toleranced_col().into());
    df.with_column(final_mask.into_column()).unwrap();

    let a = h.attribution_summary_with_timing(&df, &clock);
    assert_eq!(a.frames_in_window, 20);
    assert_eq!(a.frames_pose_ok, 20);
    assert_eq!(a.frames_outside_object, 20);
    assert_eq!(a.frames_nose_near, 11);
    assert_eq!(a.frames_nose_contact, 0);
    assert_eq!(a.frames_within_angle, 20);
    assert_eq!(a.frames_contact_or_angle, 20);
    // Frame 0 (no speed) and frame 11 (the 0.97 m jump) are not slow.
    assert_eq!(a.frames_slow_nose, 18);
    assert_eq!((a.frames_all_gates_raw, a.frames_after_tolerance), (10, 10));
    assert_eq!((a.bouts_pre_filter, a.bouts_post_filter), (1, 1));
    assert_eq!(a.bout_lengths_ms_pre, vec![1000]);
    assert_eq!(a.bout_lengths_ms_post, vec![1000]);
    assert_eq!((a.candidate_bouts_pre_merge, a.bouts_post_merge), (1, 1));
    assert_eq!(a.bouts_removed_by_duration_filter, 0);
    assert_eq!(a.final_valid_bouts, 1);
    assert_eq!(a.median_nose_speed_per_bout, vec![0.0]);
    assert_eq!(a.median_head_angle_per_bout, vec![0.0]);
}

#[test]
fn attribution_without_pose_columns_counts_nothing() {
    let h = heuristic();
    let no_x = df! { "nose_y" => [0.0] }.unwrap();
    let no_y = df! { "nose_x" => [0.0] }.unwrap();
    for df in [no_x, no_y] {
        let a = h.attribution_summary(&df, &FrameTiming::cfr(FPS));
        assert_eq!((a.frames_in_window, a.frames_pose_ok), (1, 0));
        assert_eq!((a.bouts_pre_filter, a.final_valid_bouts), (0, 0));
    }
}

#[test]
fn strict_boundaries_reject_nose_speed_on_the_limit() {
    // 0.0625 m/s at 8 fps is exactly 2^-7 m per frame.
    let base = ExplorationHeuristic {
        fps: 8.0,
        max_nose_speed: 0.0625,
        ..heuristic()
    };
    let strict = ExplorationHeuristic {
        strict_boundaries: true,
        ..base.clone()
    };
    let mut df = visit(3, 3);
    df.with_column(Series::new("nose_x".into(), [0.03125, 0.0390625, 0.046875]).into_column())
        .unwrap();
    let slow = |h: &ExplorationHeuristic<Circle>| -> Vec<Option<bool>> {
        evaluated(h, df.clone())
            .column("__bk_diag__exploring_novel__slow_nose")
            .unwrap()
            .bool()
            .unwrap()
            .into_iter()
            .collect()
    };
    assert_eq!(slow(&base), vec![Some(false), Some(true), Some(true)]);
    assert_eq!(slow(&strict), vec![Some(false), Some(false), Some(false)]);
}

#[test]
fn decomposition_reflects_erosion_and_disabled_speed() {
    let plain = heuristic().decomposition().unwrap();
    assert_eq!(
        plain.result_column,
        "__bk_diag__exploring_novel__final_toleranced"
    );
    assert!(plain.components[0].buffer_meters.is_none());
    assert!(plain.components.iter().any(|c| c.label == "slow_nose"));

    let custom = ExplorationHeuristic {
        footprint_erosion_meters: 0.005,
        speed_filter_mode: BoutFilterMode::Disabled,
        ..heuristic()
    }
    .decomposition()
    .unwrap();
    assert_eq!(custom.components[0].label, "outside_object");
    assert_eq!(custom.components[0].buffer_meters, Some(-0.005));
    assert!(custom.components.iter().all(|c| c.label != "slow_nose"));
}

#[test]
fn object_exploration_cfr_adapter_and_missing_column() {
    let df = df! { "raw" => [true, true, false, true, true, false] }.unwrap();
    // Two 0.2 s bouts bridged by a 0.1 s gap.
    let (seconds, bouts, mask) = object_exploration(&df, "raw", 10.0, 0.5, 0.1, 0.0);
    assert_eq!(bouts, 1);
    assert!((seconds - 0.5).abs() < 1e-12);
    assert_eq!(mask.sum(), Some(5));
    let (seconds, bouts, mask) = object_exploration(&df, "missing", 10.0, 0.5, 0.1, 0.0);
    assert_eq!((seconds, bouts), (0.0, 0));
    assert_eq!(mask.len(), 6);
    assert_eq!(mask.sum(), Some(0));
}

#[test]
fn nort_bout_policies_and_validation() {
    let imen = NortBoutPolicy::default();
    assert_eq!(imen.min_bout_seconds, IMEN_MIN_BOUT_SECONDS);
    assert_eq!(imen.max_gap_seconds, IMEN_MAX_GAP_SECONDS);
    assert_eq!(imen.max_angle_deg, IMEN_MAX_ANGLE_DEG);
    assert_eq!(imen.speed_filter_mode, BoutFilterMode::Disabled);
    assert!(!imen.strict_boundaries);
    let recipe = NortBoutPolicy::recipe();
    assert_eq!(recipe.min_bout_seconds, 1.0);
    assert_eq!(recipe.max_gap_seconds, 0.5);
    assert_eq!(recipe.speed_filter_mode, BoutFilterMode::PerBoutMedian);
    assert!(recipe.strict_boundaries);
    assert!(imen.validate().is_ok() && recipe.validate().is_ok());
    let at_180 = NortBoutPolicy {
        max_angle_deg: 180.0,
        ..imen
    };
    assert!(at_180.validate().is_ok());
    for bad in [
        NortBoutPolicy {
            max_angle_deg: 180.5,
            ..imen
        },
        NortBoutPolicy {
            min_bout_seconds: -0.1,
            ..imen
        },
        NortBoutPolicy {
            max_gap_seconds: f64::NAN,
            ..imen
        },
        NortBoutPolicy {
            max_nose_speed: f64::INFINITY,
            ..imen
        },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }

    let parsed: NortBoutPolicy =
        handoff::from_str("speed_filter_mode = \"per_bout_median\"\nmax_gap_seconds = 0.25")
            .unwrap();
    assert_eq!(parsed.speed_filter_mode, BoutFilterMode::PerBoutMedian);
    assert_eq!(parsed.max_gap_seconds, 0.25);
    assert_eq!(parsed.min_bout_seconds, IMEN_MIN_BOUT_SECONDS);
    assert!(handoff::from_str::<NortBoutPolicy>("min_bout = 1.0").is_err());
}

#[test]
fn contact_rescues_any_bout_angle() {
    assert!(accepts_bout_angle(Some(45.0), 45.0, false));
    assert!(!accepts_bout_angle(Some(45.1), 45.0, false));
    assert!(accepts_bout_angle(Some(90.0), 45.0, true));
    assert!(!accepts_bout_angle(None, 45.0, false));
    assert!(accepts_bout_angle(None, 45.0, true));
}
