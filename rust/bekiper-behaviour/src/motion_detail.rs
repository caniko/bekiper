use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_math::discrete::{apply_tolerance_series, bool_column_seconds};

/// Shared area-level motion summary used by multiple behavioural tasks.
///
/// This captures the legacy "motion in area" concept without reintroducing
/// the old Python class hierarchy. Zone/area occupancy stays in `ZoneSummary`,
/// while this struct stores the richer motion-specific metrics for the same
/// spatial partition.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZoneMotionSummary {
    pub zone_name: String,
    pub total_displacement_meters: f64,
    pub median_speed_mps: f64,
    pub median_acceleration_mps2: f64,
    pub freezing_seconds: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct WeightedMotionSummary {
    total_displacement_meters: f64,
    median_speed_mps: f64,
    median_acceleration_mps2: f64,
    freezing_seconds: f64,
    weight_frames: usize,
}

/// Compute an area-level motion summary from a boolean zone column.
pub fn motion_summary_for_zone(
    df: &DataFrame,
    zone_col: &str,
    x_col: &str,
    y_col: &str,
    fps: f64,
    min_dwell_frames: u32,
    freezing_col: &str,
) -> ZoneMotionSummary {
    let mask = match df.column(zone_col).and_then(|s| Ok(s.bool()?.clone())) {
        Ok(mask) => apply_tolerance_series(&mask, min_dwell_frames, 0),
        Err(_) => {
            return ZoneMotionSummary {
                zone_name: zone_col.to_string(),
                ..ZoneMotionSummary::default()
            };
        }
    };

    motion_summary_from_mask(df, zone_col, &mask, x_col, y_col, fps, freezing_col)
}

/// Compute an area-level motion summary from a precomputed boolean mask.
pub fn motion_summary_from_mask(
    df: &DataFrame,
    zone_name: &str,
    mask: &BooleanChunked,
    x_col: &str,
    y_col: &str,
    fps: f64,
    freezing_col: &str,
) -> ZoneMotionSummary {
    let islands = truth_islands(mask);
    if islands.is_empty() {
        return ZoneMotionSummary {
            zone_name: zone_name.to_string(),
            ..ZoneMotionSummary::default()
        };
    }

    let mut total_displacement_meters = 0.0;
    let mut freezing_seconds = 0.0;
    let mut speed_weighted_sum = 0.0;
    let mut acceleration_weighted_sum = 0.0;
    let mut total_weight = 0usize;

    for (start, end) in islands {
        let weight_frames = end - start;
        let segment = df.slice(start as i64, weight_frames);
        let island = motion_summary_for_segment(&segment, fps, x_col, y_col, freezing_col);

        total_displacement_meters += island.total_displacement_meters;
        freezing_seconds += island.freezing_seconds;
        speed_weighted_sum += island.median_speed_mps * island.weight_frames as f64;
        acceleration_weighted_sum += island.median_acceleration_mps2 * island.weight_frames as f64;
        total_weight += island.weight_frames;
    }

    let weight = total_weight as f64;

    ZoneMotionSummary {
        zone_name: zone_name.to_string(),
        total_displacement_meters,
        median_speed_mps: if total_weight > 0 {
            speed_weighted_sum / weight
        } else {
            0.0
        },
        median_acceleration_mps2: if total_weight > 0 {
            acceleration_weighted_sum / weight
        } else {
            0.0
        },
        freezing_seconds,
    }
}

fn motion_summary_for_segment(
    segment: &DataFrame,
    fps: f64,
    x_col: &str,
    y_col: &str,
    freezing_col: &str,
) -> WeightedMotionSummary {
    let weight_frames = segment.height();
    if weight_frames == 0 {
        return WeightedMotionSummary::default();
    }

    let positions = coordinate_pairs(segment, x_col, y_col);
    let displacements = frame_displacements(&positions);
    let total_displacement_meters = displacements.iter().sum::<f64>();
    let median_speed_mps = median_per_second_speed(&displacements, fps);
    let median_acceleration_mps2 = median_frame_acceleration(&displacements, fps);
    let freezing_seconds = bool_column_seconds(segment, freezing_col, fps);

    WeightedMotionSummary {
        total_displacement_meters,
        median_speed_mps,
        median_acceleration_mps2,
        freezing_seconds,
        weight_frames,
    }
}

fn coordinate_pairs(df: &DataFrame, x_col: &str, y_col: &str) -> Vec<(f64, f64)> {
    let xs = match df.column(x_col).and_then(|s| Ok(s.f64()?.clone())) {
        Ok(xs) => xs,
        Err(_) => return Vec::new(),
    };
    let ys = match df.column(y_col).and_then(|s| Ok(s.f64()?.clone())) {
        Ok(ys) => ys,
        Err(_) => return Vec::new(),
    };

    (0..df.height())
        .filter_map(|idx| match (xs.get(idx), ys.get(idx)) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        })
        .collect()
}

fn frame_displacements(positions: &[(f64, f64)]) -> Vec<f64> {
    positions
        .windows(2)
        .map(|window| {
            let dx = window[1].0 - window[0].0;
            let dy = window[1].1 - window[0].1;
            (dx * dx + dy * dy).sqrt()
        })
        .collect()
}

fn median_per_second_speed(displacements: &[f64], fps: f64) -> f64 {
    if displacements.is_empty() {
        return 0.0;
    }

    let frames_per_second = fps.round().max(1.0) as usize;
    let per_second_speeds: Vec<f64> = displacements
        .chunks(frames_per_second)
        .map(|chunk| chunk.iter().sum::<f64>())
        .collect();

    median(&per_second_speeds)
}

fn median_frame_acceleration(displacements: &[f64], fps: f64) -> f64 {
    if displacements.len() < 2 {
        return 0.0;
    }

    let frame_speeds: Vec<f64> = displacements.iter().map(|d| d * fps).collect();
    let accelerations: Vec<f64> = frame_speeds
        .windows(2)
        .map(|window| (window[1] - window[0]).abs() * fps)
        .collect();

    median(&accelerations)
}

fn median(values: &[f64]) -> f64 {
    Float64Chunked::from_slice("median".into(), values)
        .median()
        .unwrap_or(0.0)
}

fn truth_islands(mask: &BooleanChunked) -> Vec<(usize, usize)> {
    bekiper_math::discrete::true_series_spans(mask)
        .iter()
        .map(|span| (span.start, span.end))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_df() -> DataFrame {
        DataFrame::new(
            6,
            vec![
                Float64Chunked::from_slice("x".into(), &[0.0, 1.0, 3.0, 0.0, 10.0, 13.0])
                    .into_column(),
                Float64Chunked::from_slice("y".into(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    .into_column(),
                BooleanChunked::from_slice("zone".into(), &[true, true, true, false, true, true])
                    .into_column(),
                BooleanChunked::from_slice(
                    "freezing".into(),
                    &[false, true, false, false, true, false],
                )
                .into_column(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn motion_summary_merges_islands_without_bridge_jumps() {
        let df = test_df();
        let summary = motion_summary_for_zone(&df, "zone", "x", "y", 1.0, 1, "freezing");

        assert_eq!(summary.zone_name, "zone");
        assert!((summary.total_displacement_meters - 6.0).abs() < 1e-10);
        assert!((summary.median_speed_mps - 2.1).abs() < 1e-10);
        assert!((summary.median_acceleration_mps2 - 0.6).abs() < 1e-10);
        assert!((summary.freezing_seconds - 2.0).abs() < 1e-10);
    }

    #[test]
    fn motion_summary_missing_zone_column_returns_zeroes() {
        let df = test_df();
        let summary = motion_summary_for_zone(&df, "missing", "x", "y", 1.0, 1, "freezing");

        assert_eq!(summary.zone_name, "missing");
        assert_eq!(summary.total_displacement_meters, 0.0);
        assert_eq!(summary.median_speed_mps, 0.0);
        assert_eq!(summary.median_acceleration_mps2, 0.0);
        assert_eq!(summary.freezing_seconds, 0.0);
    }

    #[test]
    fn motion_summary_without_coordinates_has_no_motion() {
        let df = test_df();
        for (x, y) in [("missing", "y"), ("x", "missing")] {
            let summary = motion_summary_for_zone(&df, "zone", x, y, 1.0, 1, "freezing");
            assert_eq!(summary.total_displacement_meters, 0.0);
            assert_eq!(summary.median_speed_mps, 0.0);
            // Freezing is still read from its own column.
            assert!((summary.freezing_seconds - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn motion_summary_skips_null_coordinates() {
        let df = df! {
            "x" => [Some(0.0), None, Some(1.0)],
            "y" => [Some(0.0), Some(0.0), Some(0.0)],
            "zone" => [true, true, true],
        }
        .unwrap();
        let summary = motion_summary_for_zone(&df, "zone", "x", "y", 1.0, 1, "freezing");
        // The null frame is dropped, joining the remaining two positions.
        assert!((summary.total_displacement_meters - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mask_beyond_the_frame_has_zero_weighted_rates() {
        let df = test_df().slice(0, 2);
        let mask = BooleanChunked::from_slice("zone".into(), &[false, false, false, true]);
        let summary = motion_summary_from_mask(&df, "zone", &mask, "x", "y", 1.0, "freezing");
        assert_eq!(summary.median_speed_mps, 0.0);
        assert_eq!(summary.median_acceleration_mps2, 0.0);
        assert_eq!(summary.total_displacement_meters, 0.0);
    }
}
