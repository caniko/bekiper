use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::tolerance::BoutTiming;
use bekiper_math::discrete::{apply_tolerance_series, count_bouts, first_true_frame};
use bekiper_math::geometry::path_distance;

/// Width/height fraction for the opt-in video-relative center convention.
/// Half of each arena dimension gives one quarter of a rectangular floor's area.
pub const DEFAULT_VIDEO_CENTER_FRACTION: f64 = 0.5;

/// Standard OFT window, starting at the first finite body-centroid detection.
pub const DEFAULT_OFT_WINDOW_SECONDS: f64 = 300.0;

/// Expected Y-maze and NORT trial length; recordings are flagged, not capped.
pub const DEFAULT_EXPECTED_TRIAL_SECONDS: f64 = 360.0;

/// Allowed distance (s) from [`DEFAULT_EXPECTED_TRIAL_SECONDS`], inclusive.
pub const DEFAULT_TRIAL_DURATION_TOLERANCE_SECONDS: f64 = 10.0;

/// True when an analysed duration is within the tolerance of the expected
/// trial length ("approximately 360 s").
pub fn duration_within_expected(seconds: f64) -> bool {
    (seconds - DEFAULT_EXPECTED_TRIAL_SECONDS).abs()
        <= DEFAULT_TRIAL_DURATION_TOLERANCE_SECONDS + 1e-9
}

/// Which frames Y-maze zone outputs (time, entries, distance, speed, arm
/// preference) count. Valid entry sequences and SA always use the dwell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZoneOccupancy {
    /// Only visits that pass the entry dwell (default).
    #[default]
    DwellFiltered,
    /// Every frame inside the zone predicate.
    Raw,
}

/// Summarize an already selected mask, sharing one dwell decision with motion
/// summaries, entry sequences and provenance.
pub fn zone_summary_from_mask(
    df: &DataFrame,
    zone_col: &str,
    x_col: &str,
    y_col: &str,
    mask: &BooleanChunked,
    timing: &BoutTiming,
) -> ZoneSummary {
    let values: Vec<_> = mask.into_iter().map(|v| v.unwrap_or(false)).collect();
    let duration_seconds = timing.mask_seconds(&values);
    let latency_seconds = first_true_frame(mask).map(|i| timing.boundary(i) - timing.boundary(0));
    summarize_selected_mask(
        df,
        zone_col,
        x_col,
        y_col,
        mask,
        (duration_seconds, latency_seconds),
    )
}

fn summarize_selected_mask(
    df: &DataFrame,
    zone_col: &str,
    x_col: &str,
    y_col: &str,
    mask: &BooleanChunked,
    temporal: (f64, Option<f64>),
) -> ZoneSummary {
    let (duration_seconds, latency_seconds) = temporal;
    let distance_meters = path_distance(df, x_col, y_col, Some(mask));
    ZoneSummary {
        zone_name: zone_col.into(),
        entries: count_bouts(mask),
        duration_seconds,
        distance_meters,
        mean_speed: if duration_seconds > 0.0 {
            distance_meters / duration_seconds
        } else {
            0.0
        },
        latency_seconds,
    }
}

/// Per-zone summary statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZoneSummary {
    pub zone_name: String,
    /// Number of entries into this zone (after dwell threshold).
    pub entries: u32,
    /// Total time spent in zone (seconds).
    pub duration_seconds: f64,
    /// Total distance traveled while in zone (meters).
    pub distance_meters: f64,
    /// Mean speed while in zone (m/s).
    pub mean_speed: f64,
    /// Latency: time from trial start to first entry (seconds). None if never entered.
    pub latency_seconds: Option<f64>,
}

/// Return a boolean mask with short true-runs removed by the same dwell rule
/// used by [`zone_summary`].
///
/// This is intentionally vector-based so callers that already own framewise
/// booleans can reuse the exact zone-summary filter before doing their own
/// bout or transition logic.
pub fn dwell_filtered_mask(raw: &[bool], min_dwell_frames: u32) -> Vec<bool> {
    if raw.is_empty() {
        return Vec::new();
    }

    let raw = BooleanChunked::from_slice("zone".into(), raw);
    let filtered = apply_tolerance_series(&raw, min_dwell_frames, 0);
    filtered.into_iter().map(|v| v.unwrap_or(false)).collect()
}

/// Compute per-zone summary from a collected DataFrame.
///
/// Requires:
/// - A boolean column `zone_col` marking frames inside the zone
/// - Coordinate columns for distance/speed computation
///
/// `min_dwell_frames`: minimum consecutive frames in zone to count as an entry.
/// Used for the ≥1.0s dwell requirement (pass `fps * 1.0` as u32).
pub fn zone_summary(
    df: &DataFrame,
    zone_col: &str,
    x_col: &str,
    y_col: &str,
    fps: f64,
    min_dwell_frames: u32,
) -> ZoneSummary {
    zone_summary_timed(
        df,
        zone_col,
        x_col,
        y_col,
        &bekiper_core::video::FrameTiming::cfr(fps),
        min_dwell_frames,
    )
}

/// Compute zone summaries for all zones at once.
pub fn all_zone_summaries(
    df: &DataFrame,
    zone_columns: &[&str],
    x_col: &str,
    y_col: &str,
    fps: f64,
    min_dwell_frames: u32,
) -> Vec<ZoneSummary> {
    zone_columns
        .iter()
        .map(|&zone_col| zone_summary(df, zone_col, x_col, y_col, fps, min_dwell_frames))
        .collect()
}

/// VFR-aware zone summary using actual frame timestamps.
///
/// Like `zone_summary`, but uses `FrameTiming` for accurate duration and latency
/// when the video has variable frame rate.
pub fn zone_summary_timed(
    df: &DataFrame,
    zone_col: &str,
    x_col: &str,
    y_col: &str,
    timing: &bekiper_core::video::FrameTiming,
    min_dwell_frames: u32,
) -> ZoneSummary {
    let raw_bools = match df.column(zone_col).and_then(|s| Ok(s.bool()?.clone())) {
        Ok(b) => b,
        Err(_) => {
            return ZoneSummary {
                zone_name: zone_col.to_string(),
                ..Default::default()
            };
        }
    };

    // Apply dwell threshold: remove entries shorter than min_dwell_frames
    let toleranced = apply_tolerance_series(&raw_bools, min_dwell_frames, 0);

    let latency_seconds = first_true_frame(&toleranced).map(|f| timing.frame_to_elapsed(f));

    // Use FrameTiming for accurate duration
    let mask: Vec<bool> = toleranced.iter().map(|v| v.unwrap_or(false)).collect();
    let duration_seconds = timing.bool_frames_to_seconds(&mask);

    summarize_selected_mask(
        df,
        zone_col,
        x_col,
        y_col,
        &toleranced,
        (duration_seconds, latency_seconds),
    )
}
