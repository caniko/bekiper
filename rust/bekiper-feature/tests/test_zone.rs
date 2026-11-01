use polars::prelude::*;

use bekiper_core::types::CoordinateColumns;
use bekiper_core::video::FrameTiming;
use bekiper_feature::quadrant::{
    QuadrantParams, add_grid_zone_columns, generate_grid_zones, quadrant_analysis,
};
use bekiper_feature::tolerance::BoutTiming;
use bekiper_feature::zone::{
    all_zone_summaries, dwell_filtered_mask, zone_summary, zone_summary_from_mask,
    zone_summary_timed,
};

#[test]
fn zone_summary_all_in_zone() {
    let bools: Vec<bool> = vec![true; 90];
    let xs: Vec<f64> = (0..90).map(|i| i as f64).collect();
    let ys: Vec<f64> = vec![0.0; 90];

    let df = df! {
        "zone_a" => &bools,
        "x" => &xs,
        "y" => &ys,
    }
    .unwrap();

    let s = zone_summary(&df, "zone_a", "x", "y", 30.0, 1);
    assert!((s.duration_seconds - 3.0).abs() < 1e-10);
    assert!(s.entries > 0);
    assert!(s.distance_meters > 0.0);
}

#[test]
fn zone_summary_never_in_zone() {
    let df = df! {
        "zone_a" => &[false, false, false, false, false],
        "x" => &[0.0, 1.0, 2.0, 3.0, 4.0],
        "y" => &[0.0, 0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();

    let s = zone_summary(&df, "zone_a", "x", "y", 30.0, 1);
    assert_eq!(s.entries, 0);
    assert!((s.duration_seconds - 0.0).abs() < 1e-10);
    assert!(s.latency_seconds.is_none());
}

#[test]
fn zone_summary_one_bout() {
    let df = df! {
        "zone_a" => &[true, true, true, true, true, false, false, false],
        "x" => &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "y" => &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();

    let s = zone_summary(&df, "zone_a", "x", "y", 30.0, 1);
    assert_eq!(s.entries, 1);
    assert!((s.duration_seconds - 5.0 / 30.0).abs() < 1e-10);
}

#[test]
fn zone_summary_short_bout_filtered() {
    // Bout of 3 frames, but min_dwell=5 → filtered out
    let df = df! {
        "zone_a" => &[false, true, true, true, false, false],
        "x" => &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y" => &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();

    let s = zone_summary(&df, "zone_a", "x", "y", 30.0, 5);
    assert_eq!(s.entries, 0);
    assert!((s.duration_seconds - 0.0).abs() < 1e-10);
}

#[test]
fn zone_summary_latency() {
    let df = df! {
        "zone_a" => &[false, false, false, true, true],
        "x" => &[0.0, 1.0, 2.0, 3.0, 4.0],
        "y" => &[0.0, 0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();

    let s = zone_summary(&df, "zone_a", "x", "y", 30.0, 1);
    assert!(s.latency_seconds.is_some());
    assert!((s.latency_seconds.unwrap() - 3.0 / 30.0).abs() < 1e-10);
}

#[test]
fn zone_summary_distance() {
    let df = df! {
        "zone_a" => &[true, true, true, false, false],
        "x" => &[0.0, 1.0, 2.0, 3.0, 4.0],
        "y" => &[0.0, 0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();

    let s = zone_summary(&df, "zone_a", "x", "y", 30.0, 1);
    // Distance while in zone: |1-0| + |2-1| = 2.0
    assert!((s.distance_meters - 2.0).abs() < 1e-10);
}

#[test]
fn zone_summary_missing_column() {
    let df = df! {
        "other" => &[1.0, 2.0, 3.0],
        "x" => &[0.0, 1.0, 2.0],
        "y" => &[0.0, 0.0, 0.0],
    }
    .unwrap();

    let s = zone_summary(&df, "nonexistent", "x", "y", 30.0, 1);
    assert_eq!(s.entries, 0);
    assert!((s.duration_seconds - 0.0).abs() < 1e-10);
}

#[test]
fn all_zone_summaries_multiple() {
    let df = df! {
        "zone_a" => &[true, true, false, false],
        "zone_b" => &[false, false, true, true],
        "x" => &[0.0, 1.0, 2.0, 3.0],
        "y" => &[0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();

    let summaries = all_zone_summaries(&df, &["zone_a", "zone_b"], "x", "y", 30.0, 1);
    assert_eq!(summaries.len(), 2);
    assert_eq!(summaries[0].zone_name, "zone_a");
    assert_eq!(summaries[1].zone_name, "zone_b");
    assert!(summaries[0].entries > 0);
    assert!(summaries[1].entries > 0);
}

#[test]
fn dwell_filter_is_inclusive_at_the_minimum() {
    assert!(dwell_filtered_mask(&[], 3).is_empty());
    assert_eq!(
        dwell_filtered_mask(&[true, true, true, false, true, true], 3),
        vec![true, true, true, false, false, false]
    );
    // Bouts touching the recording edges are kept when long enough.
    assert_eq!(
        dwell_filtered_mask(&[true, true, false, true, true], 2),
        vec![true, true, false, true, true]
    );
    assert_eq!(dwell_filtered_mask(&[true, false], 0), vec![true, false]);
}

#[test]
fn zone_summary_from_mask_uses_half_open_timestamps() {
    let df = df! {
        "x" => [0.0, 1.0, 1.0, 4.0],
        "y" => [0.0, 0.0, 4.0, 4.0],
    }
    .unwrap();
    let mask = BooleanChunked::from_iter_options(
        "m".into(),
        [Some(false), Some(true), None, Some(true)].into_iter(),
    );
    let timing = BoutTiming::timestamps(&[10.0, 10.5, 12.0, 12.5], 14.0).unwrap();
    let s = zone_summary_from_mask(&df, "zone", "x", "y", &mask, &timing);
    assert_eq!(s.zone_name, "zone");
    assert_eq!(s.entries, 2);
    // Samples 1 ([10.5, 12)) and 3 ([12.5, 14)).
    assert_eq!(s.duration_seconds, 3.0);
    assert_eq!(s.latency_seconds, Some(0.5));
    // Distance accrues on steps into masked frames: (0,0)->(1,0) and (1,4)->(4,4).
    assert_eq!(s.distance_meters, 4.0);
    assert_eq!(s.mean_speed, 4.0 / 3.0);

    let none = BooleanChunked::from_slice("m".into(), &[false; 4]);
    let empty = zone_summary_from_mask(&df, "zone", "x", "y", &none, &timing);
    assert_eq!(
        (empty.entries, empty.duration_seconds, empty.mean_speed),
        (0, 0.0, 0.0)
    );
    assert_eq!(empty.latency_seconds, None);
}

#[test]
fn zone_summary_timed_vfr_latency_and_duration() {
    let df = df! {
        "zone" => [false, true, true, true],
        "x" => [0.0, 0.0, 0.0, 0.0],
        "y" => [0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();
    let timing = FrameTiming {
        fps: 2.0,
        timestamps: vec![0.0, 1.0, 1.5, 3.0],
        is_vfr: true,
    };
    let s = zone_summary_timed(&df, "zone", "x", "y", &timing, 3);
    assert_eq!(s.entries, 1);
    assert_eq!(s.latency_seconds, Some(1.0));
    assert_eq!(s.duration_seconds, 3.0);
    let rejected = zone_summary_timed(&df, "zone", "x", "y", &timing, 4);
    assert_eq!(rejected.entries, 0);
    assert_eq!(rejected.latency_seconds, None);
    let not_bool = zone_summary_timed(&df, "x", "x", "y", &timing, 1);
    assert_eq!(not_bool.zone_name, "x");
    assert_eq!(not_bool.entries, 0);
}

#[test]
fn grid_zones_tile_the_arena_row_major() {
    let zones = generate_grid_zones(1.0, 1.0, 2.0, 2.0, 2, 2);
    let labels: Vec<_> = zones.iter().map(|(label, _)| label.as_str()).collect();
    assert_eq!(labels, vec!["q_0_0", "q_0_1", "q_1_0", "q_1_1"]);
    assert_eq!(
        (zones[1].1.shape.center_x, zones[1].1.shape.center_y),
        (1.5, 0.5)
    );
    assert_eq!(zones[1].1.shape.width, 1.0);
    assert!(generate_grid_zones(0.0, 0.0, 1.0, 1.0, 0, 3).is_empty());

    let coords = CoordinateColumns::new("x", "y");
    let df = df! { "x" => [0.5, 1.0], "y" => [0.5, 1.5] }.unwrap();
    // Half-open crossing rule: a shared edge belongs to exactly one cell.
    let out = add_grid_zone_columns(df.clone().lazy(), &zones, &coords)
        .collect()
        .unwrap();
    let flags = |name: &str| -> Vec<bool> {
        out.column(name)
            .unwrap()
            .bool()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap())
            .collect()
    };
    assert_eq!(flags("q_0_0"), vec![true, false]);
    assert_eq!(flags("q_1_0"), vec![false, false]);
    assert_eq!(flags("q_1_1"), vec![false, true]);
    let unchanged = add_grid_zone_columns(df.clone().lazy(), &[], &coords)
        .collect()
        .unwrap();
    assert_eq!(unchanged.width(), 2);
}

#[test]
fn quadrant_analysis_summarizes_each_cell() {
    let coords = CoordinateColumns::new("x", "y");
    let df = df! {
        "x" => [0.5, 0.6, 0.7, 1.5, 1.5],
        "y" => [0.5, 0.5, 0.5, 1.5, 1.6],
    }
    .unwrap();
    let params = QuadrantParams {
        arena_cx: 1.0,
        arena_cy: 1.0,
        arena_w: 2.0,
        arena_h: 2.0,
        rows: 2,
        cols: 2,
        coords: &coords,
        fps: 10.0,
        min_dwell_frames: 3,
    };
    let summaries = quadrant_analysis(&df, &params);
    assert_eq!(summaries.len(), 4);
    assert_eq!(summaries[0].zone_name, "q_0_0");
    assert_eq!(summaries[0].entries, 1);
    assert!((summaries[0].duration_seconds - 0.3).abs() < 1e-12);
    assert_eq!(summaries[0].latency_seconds, Some(0.0));
    // Two frames in q_1_1 fall short of the three-frame dwell.
    assert_eq!(summaries[3].entries, 0);
    assert_eq!(summaries[3].duration_seconds, 0.0);
}

#[test]
fn trial_duration_flag_accepts_360_plus_or_minus_10_seconds_inclusive() {
    use bekiper_feature::zone::duration_within_expected;
    for seconds in [350.0, 360.0, 365.1, 370.0] {
        assert!(duration_within_expected(seconds), "{seconds}");
    }
    for seconds in [349.9, 370.1, 0.0, f64::NAN] {
        assert!(!duration_within_expected(seconds), "{seconds}");
    }
}
