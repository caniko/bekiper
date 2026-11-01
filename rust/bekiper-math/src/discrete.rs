use crate::temporal::{DurationComparison, TemporalSelection, select_bouts, state_spans};
use polars::prelude::*;

/// Historical frame-count compatibility. Explicit seconds-based policies use
/// measured duration instead; do not use this quantization for new policies.
pub fn legacy_frame_threshold(seconds: f64, fps: f64) -> u32 {
    (seconds * fps) as u32
}

/// Apply temporal tolerance to a boolean Series.
///
/// 1. Bridge short false-gaps (< `max_gap_frames`): if the animal briefly
///    left a state, treat it as continuous.
/// 2. Remove short true-runs (< `min_bout_frames`): too brief to count.
///
/// The optional `gap_density_threshold` (0.0–1.0) controls gap bridging:
/// a gap is only bridged if the fraction of true frames in the *original*
/// input within that gap exceeds this threshold. This captures the useful
/// property of the legacy hysteresis model (scattered true frames within a
/// gap indicate the animal is still partially engaged) without the stateful
/// complexity. Set to 0.0 (default) to bridge any short gap regardless.
pub fn apply_tolerance_series(
    series: &BooleanChunked,
    min_bout_frames: u32,
    max_gap_frames: u32,
) -> BooleanChunked {
    apply_tolerance_series_with_density(series, min_bout_frames, max_gap_frames, 0.0)
}

/// Like [`apply_tolerance_series`] but with an explicit gap density threshold.
pub fn apply_tolerance_series_with_density(
    series: &BooleanChunked,
    min_bout_frames: u32,
    max_gap_frames: u32,
    gap_density_threshold: f64,
) -> BooleanChunked {
    let selected = select_series_bouts(
        series,
        min_bout_frames,
        max_gap_frames,
        gap_density_threshold,
    );
    BooleanChunked::from_slice(series.name().clone(), &selected.mask)
}

/// Frame-count compatibility adapter to the common temporal kernel.
pub fn select_series_bouts(
    series: &BooleanChunked,
    minimum: u32,
    gap: u32,
    density: f64,
) -> TemporalSelection {
    let raw: Vec<_> = series.into_iter().map(|v| v.unwrap_or(false)).collect();
    select_bouts(
        &raw,
        |a, b| (b - a) as f64,
        f64::from(minimum),
        DurationComparison::AtLeast,
        f64::from(gap),
        density,
    )
}

pub fn true_series_spans(series: &BooleanChunked) -> Vec<std::ops::Range<usize>> {
    let raw: Vec<_> = series.into_iter().map(|v| v.unwrap_or(false)).collect();
    state_spans(&raw, true)
}

/// Count the number of distinct true-bouts in a boolean Series.
/// A bout is a contiguous run of `true` values.
pub fn count_bouts(series: &BooleanChunked) -> u32 {
    true_series_spans(series).len() as u32
}

/// Return the length, in frames, of each contiguous true-bout.
pub fn bout_lengths_frames(series: &BooleanChunked) -> Vec<u32> {
    true_series_spans(series)
        .iter()
        .map(|span| span.len() as u32)
        .collect()
}

/// Find the frame index of the first `true` value in a boolean Series.
pub fn first_true_frame(series: &BooleanChunked) -> Option<usize> {
    series.into_iter().position(|v| v.unwrap_or(false))
}

/// Extract the zone label for each frame from multiple zone boolean columns.
///
/// Given a DataFrame with boolean columns named by zone, returns a String
/// column where each frame is labeled with the zone it belongs to.
/// If a frame is in no zone, it gets "none". If in multiple, first match wins.
pub fn zone_label_series(df: &DataFrame, zone_columns: &[&str]) -> StringChunked {
    let n = df.height();
    let mut labels: Vec<String> = vec!["none".to_string(); n];

    for &zone_name in zone_columns {
        if let Ok(col) = df.column(zone_name)
            && let Ok(bools) = col.bool()
        {
            for (i, val) in bools.into_iter().enumerate() {
                if val.unwrap_or(false) && labels[i] == "none" {
                    labels[i] = zone_name.to_string();
                }
            }
        }
    }

    StringChunked::from_iter_values("zone".into(), labels.iter().map(|s| s.as_str()))
}

/// Sum a boolean column as seconds: `true_count / fps`.
///
/// Returns 0.0 if the column is missing or not boolean.
pub fn bool_column_seconds(df: &DataFrame, col_name: &str, fps: f64) -> f64 {
    df.column(col_name)
        .ok()
        .and_then(|s| s.bool().ok())
        .map(|b| b.sum().unwrap_or(0) as f64 / fps)
        .unwrap_or(0.0)
}

/// Sum boolean frames as seconds, excluding true-bouts below a minimum duration.
///
/// Bout length conversion follows the same convention as zone dwell filtering:
/// `(fps * min_bout_seconds) as u32`, i.e. fractional frame counts are
/// truncated before applying the inclusive `>=` threshold.
///
/// Returns 0.0 if the column is missing or not boolean. Null values are treated
/// as false, matching [`apply_tolerance_series`].
pub fn bool_column_seconds_min_bout(
    df: &DataFrame,
    col_name: &str,
    fps: f64,
    min_bout_seconds: f64,
) -> f64 {
    let min_bout_frames = legacy_frame_threshold(min_bout_seconds, fps);
    df.column(col_name)
        .ok()
        .and_then(|s| s.bool().ok())
        .map(|b| bool_series_seconds_min_bout(b, fps, min_bout_frames))
        .unwrap_or(0.0)
}

fn bool_series_seconds_min_bout(series: &BooleanChunked, fps: f64, min_bout_frames: u32) -> f64 {
    let selected = select_series_bouts(series, min_bout_frames, 0, 0.0);
    selected.mask.iter().filter(|&&v| v).count() as f64 / fps
}

/// Count the number of true values in a boolean column.
pub fn count_true_expr(bool_col: &str) -> Expr {
    col(bool_col).sum()
}

/// Logical OR across multiple boolean columns (any_horizontal).
pub fn any_of(exprs: Vec<Expr>) -> Expr {
    polars::prelude::any_horizontal(exprs).expect("any_horizontal requires at least one expr")
}

/// Logical AND across multiple boolean columns (all_horizontal).
pub fn all_of(exprs: Vec<Expr>) -> Expr {
    polars::prelude::all_horizontal(exprs).expect("all_horizontal requires at least one expr")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bool_df(values: &[Option<bool>]) -> DataFrame {
        DataFrame::new(values.len(), vec![Column::new("flag".into(), values)]).unwrap()
    }

    #[test]
    fn bool_column_seconds_min_bout_empty_column() {
        let df = bool_df(&[]);
        assert_eq!(bool_column_seconds_min_bout(&df, "flag", 10.0, 1.0), 0.0);
    }

    #[test]
    fn bool_column_seconds_min_bout_all_true_column() {
        let df = bool_df(&[Some(true); 12]);
        assert_eq!(bool_column_seconds_min_bout(&df, "flag", 10.0, 1.0), 1.2);
    }

    #[test]
    fn bool_column_seconds_min_bout_excludes_runs_below_threshold() {
        let df = bool_df(&[
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(true),
        ]);
        assert_eq!(bool_column_seconds_min_bout(&df, "flag", 10.0, 1.0), 0.0);
    }

    #[test]
    fn bool_column_seconds_min_bout_includes_runs_above_threshold() {
        let mut values = Vec::new();
        values.extend(vec![Some(true); 3]);
        values.push(Some(false));
        values.extend(vec![Some(true); 12]);
        values.push(Some(false));
        values.push(Some(true));
        values.push(None);
        values.extend(vec![Some(true); 25]);

        let df = bool_df(&values);
        assert_eq!(bool_column_seconds_min_bout(&df, "flag", 10.0, 1.0), 3.7);
    }

    #[test]
    fn bool_column_seconds_min_bout_includes_exact_threshold_boundary() {
        let df = bool_df(&[Some(true); 10]);
        assert_eq!(bool_column_seconds_min_bout(&df, "flag", 10.0, 1.0), 1.0);
    }

    #[test]
    fn bool_column_seconds_min_bout_uses_zone_dwell_frame_truncation() {
        let df = bool_df(&[Some(true); 37]);
        assert_eq!(bool_column_seconds_min_bout(&df, "flag", 25.0, 1.5), 1.48);
    }

    #[test]
    fn bool_column_seconds_min_bout_missing_column_is_zero() {
        let df = bool_df(&[Some(true); 12]);
        assert_eq!(bool_column_seconds_min_bout(&df, "missing", 10.0, 1.0), 0.0);
    }

    #[test]
    fn bool_column_seconds_min_bout_zero_threshold_includes_all_true_frames() {
        // min_bout_seconds == 0 → min_bout_frames == 0 → trivial sum / fps.
        let df = bool_df(&[Some(true), Some(false), Some(true)]);
        assert!((bool_column_seconds_min_bout(&df, "flag", 10.0, 0.0) - 0.2).abs() < 1e-9,);
    }

    #[test]
    fn bool_column_seconds_min_bout_one_frame_threshold_includes_singletons() {
        // min_bout_seconds * fps == 1 → min_bout_frames == 1 → fast path
        // keeps all true frames including singletons.
        let df = bool_df(&[Some(true), Some(false), Some(true), Some(true)]);
        assert!((bool_column_seconds_min_bout(&df, "flag", 10.0, 0.1) - 0.3).abs() < 1e-9,);
    }

    #[test]
    fn bool_column_seconds_min_bout_zero_threshold_with_only_false_returns_zero() {
        let df = bool_df(&[Some(false); 12]);
        assert_eq!(bool_column_seconds_min_bout(&df, "flag", 10.0, 0.0), 0.0);
    }

    #[test]
    fn bool_column_seconds_min_bout_treats_nulls_as_false() {
        // Run interrupted by a null counts as broken — same as a false frame.
        let df = bool_df(&[
            Some(true),
            Some(true),
            Some(true),
            None,
            Some(true),
            Some(true),
        ]);
        // Both runs are 3 and 2 frames, both below threshold of 10 → dropped.
        assert_eq!(bool_column_seconds_min_bout(&df, "flag", 10.0, 1.0), 0.0);
    }

    #[test]
    fn bool_column_seconds_min_bout_just_below_threshold_is_excluded() {
        // 9 frames at 10 fps → 0.9 s, threshold 1.0 s → excluded.
        let df = bool_df(&[Some(true); 9]);
        assert_eq!(bool_column_seconds_min_bout(&df, "flag", 10.0, 1.0), 0.0);
    }

    #[test]
    fn bool_column_seconds_min_bout_rearing_half_second_threshold() {
        // Rearing spec uses 0.5 s. At 30 fps that's 15 frames.
        // Three runs of lengths 10, 15, 20 → only the latter two count
        // → 35/30 ≈ 1.1667 s.
        let mut values: Vec<Option<bool>> = Vec::new();
        values.extend(vec![Some(true); 10]);
        values.push(Some(false));
        values.extend(vec![Some(true); 15]);
        values.push(Some(false));
        values.extend(vec![Some(true); 20]);

        let df = bool_df(&values);
        let result = bool_column_seconds_min_bout(&df, "flag", 30.0, 0.5);
        assert!(
            (result - (35.0 / 30.0)).abs() < 1e-9,
            "rearing-style filter = {result}",
        );
    }

    #[test]
    fn bool_column_seconds_min_bout_trailing_run_captured() {
        // Final true run at end of column (no trailing false) must be
        // counted — exercises the post-loop accumulation branch.
        let mut values: Vec<Option<bool>> = vec![Some(false); 5];
        values.extend(vec![Some(true); 15]);
        let df = bool_df(&values);
        assert!((bool_column_seconds_min_bout(&df, "flag", 10.0, 1.0) - 1.5).abs() < 1e-9,);
    }
}
