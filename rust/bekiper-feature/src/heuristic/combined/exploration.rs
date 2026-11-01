use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_core::shape::{ClosestPoint, Expandable};
use bekiper_core::types::{CoordinateColumns, Label};
use bekiper_math::discrete::count_bouts;
use bekiper_math::velocity::velocity_expr;
use bekiper_perimeter::perimeter::Perimeter;

use crate::heuristic::Heuristic;
use crate::heuristic::solo::whiskers::WhiskerFanSpec;
use crate::heuristic::{coordinate_label_from_xy, decomposition_component, diag_column_name};
use crate::tolerance::BoutTiming;

/// Floating-point slack so a head angle of exactly `max_angle_deg` qualifies.
const ANGLE_EPSILON_DEG: f64 = 1e-9 * (180.0 / std::f64::consts::PI);

// ── NORT exploration defaults ────────────────────────────────────────────
// Canonical defaults used by both this heuristic and the
// `bekiper-behaviour::ObjectRecognition` task layer that re-exports them.
// Overrideable at runtime via `TaskParamsOverrides::exploration` in
// `bekiper-ingress`.

/// Maximum nose-to-object distance (m) for exploration. Points outside
/// this radius are never considered exploration regardless of angle.
pub const DEFAULT_NOSE_MAX_DISTANCE: f64 = 0.03;

/// Distance (m) below which the head-angle check is skipped — a nose in
/// direct contact with the object is always exploration.
pub const DEFAULT_CONTACT_DISTANCE: f64 = 0.005;

/// Maximum head-to-object angle (degrees) for exploration.
pub const DEFAULT_MAX_ANGLE_DEG: f64 = 30.0;

/// Maximum nose speed (m/s) during exploration. Fast nose motion is
/// treated as locomotion, not investigation.
pub const DEFAULT_MAX_NOSE_SPEED: f64 = 0.05;

/// Minimum bout duration (s). Shorter contacts are dropped as noise.
pub const DEFAULT_MIN_BOUT_SECONDS: f64 = 1.0;

/// Maximum gap (s) between bouts that may be bridged into one bout.
pub const DEFAULT_MAX_GAP_SECONDS: f64 = 0.5;

/// Neo/Imen's 2026-10-13 researcher amendment (reaffirming 2026-10-07); new
/// Imen runs set 0.25 s explicitly (2026-10-28). Generic defaults above
/// remain available for reproducing other projects' recorded policies.
pub const IMEN_MIN_BOUT_SECONDS: f64 = 0.5;
pub const IMEN_MAX_GAP_SECONDS: f64 = 1.0;
pub const IMEN_MAX_ANGLE_DEG: f64 = 45.0;

/// Neo/Imen whisker fan (owner decisions 2026-10-28): apex 90% of the way from
/// the head midline (inter-ear midpoint, `center_ear`) to the nose tip, on the
/// mystacial pad; +-30 degrees around each lateral perpendicular.
pub const IMEN_WHISKER_APEX_FRACTION: f64 = 0.9;
pub const IMEN_WHISKER_HALF_ANGLE_DEG: f64 = 30.0;

/// Default whisker length (m): the array-centre arclength of adult C57BL/6J
/// mice, Bresee, Belli, Luo & Hartmann 2023, J Exp Biol 226:jeb245597. No
/// adult age effect is documented, so the default is flat across ages; the
/// reported full range is 2-30 mm.
pub const LITERATURE_WHISKER_LENGTH_M: f64 = 0.0117;

/// Shared standalone/native policy contract. `Default` is the Neo/Imen
/// amendment (see [`NortBoutPolicy::imen`]) because plans and archived results
/// omit the keys they never set; generic consumers (`ExplorationHeuristic::new`,
/// `ObjectRecognition::new`) use the `DEFAULT_*` constants instead and are
/// deliberately not this policy. Select the generic-limit recipe via [`recipe`](Self::recipe).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct NortBoutPolicy {
    pub min_bout_seconds: f64,
    pub max_gap_seconds: f64,
    pub max_nose_speed: f64,
    pub max_angle_deg: f64,
    pub speed_filter_mode: BoutFilterMode,
    pub strict_boundaries: bool,
    /// Candidates carry whisker-fan hits; any hit frame rescues a bout like
    /// nose contact. Geometry is not recomputed here.
    pub whisker_enabled: bool,
}

impl Default for NortBoutPolicy {
    fn default() -> Self {
        Self::imen()
    }
}

impl NortBoutPolicy {
    /// Neo/Imen 2026-10-13 researcher amendment (reaffirming 2026-10-07; limits per
    /// the `IMEN_*` constants); new Imen runs set 0.25 s explicitly (2026-10-28).
    pub fn imen() -> Self {
        Self {
            min_bout_seconds: IMEN_MIN_BOUT_SECONDS,
            max_gap_seconds: IMEN_MAX_GAP_SECONDS,
            max_nose_speed: DEFAULT_MAX_NOSE_SPEED,
            max_angle_deg: IMEN_MAX_ANGLE_DEG,
            speed_filter_mode: BoutFilterMode::Disabled,
            strict_boundaries: false,
            whisker_enabled: false,
        }
    }

    pub fn recipe() -> Self {
        Self {
            min_bout_seconds: DEFAULT_MIN_BOUT_SECONDS,
            max_gap_seconds: DEFAULT_MAX_GAP_SECONDS,
            speed_filter_mode: BoutFilterMode::PerBoutMedian,
            strict_boundaries: true,
            ..Default::default()
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if [
            self.min_bout_seconds,
            self.max_gap_seconds,
            self.max_nose_speed,
            self.max_angle_deg,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
            || self.max_angle_deg > 180.0
        {
            return Err(
                "NORT bout policy requires finite nonnegative limits and angle <=180".into(),
            );
        }
        Ok(())
    }
}

/// Recorded native bout-level contact override, shared with reconstruction.
pub const BOUT_CONTACT_POLICY: &str = "any_observed_nose_contact";

/// Recorded bout-level override when the whisker fan is enabled: a whisker-hit
/// frame rescues a bout exactly like nose contact.
pub const BOUT_CONTACT_POLICY_WHISKER: &str = "any_observed_nose_or_whisker_contact";

/// Contact rescues an otherwise failing bout-median angle, including missing
/// angle observations. Call only after candidate and temporal eligibility.
/// With the whisker fan enabled, callers pass `nose_contact || whisker_hit`.
pub fn accepts_bout_angle(median: Option<f64>, maximum: f64, nose_contact: bool) -> bool {
    median.is_some_and(|angle| angle <= maximum) || nose_contact
}

/// Gap-density threshold (0.0–1.0). A gap is bridged only if the fraction
/// of true frames in the original input within it exceeds this value.
/// 0.0 = bridge any short gap.
pub const DEFAULT_GAP_DENSITY_THRESHOLD: f64 = 0.0;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BoutFilterMode {
    /// Retain the diagnostic metric without using it to reject frames or bouts.
    Disabled,
    #[default]
    PerFrame,
    /// Drop the filter from the per-frame AND mask; instead, accept a bout iff
    /// its post-tolerance median of the underlying metric is within threshold.
    /// Brief frame-level excursions above threshold do not kill the bout.
    /// Bridged-gap frames are included in the median, and even-length bouts use
    /// `(lower + upper) / 2.0` as the tie-breaker.
    PerBoutMedian,
}

/// Combined object exploration heuristic for NORT.
///
/// Exploration is defined as:
/// - Body centroid is OUTSIDE the object footprint
/// - Nose is within `nose_max_distance` of the object perimeter
/// - Head orientation is within `max_angle_deg` of the nearest object boundary
///   OR nose is in direct contact (distance < `contact_distance`)
/// - When enabled, nose speed is below `max_nose_speed`
///
/// After frame-level detection, bouts are:
/// - Merged if gap ≤ `max_gap_seconds`
/// - Removed if shorter than `min_bout_seconds`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationHeuristic<S: Expandable + ClosestPoint> {
    pub object_perimeter: Perimeter<S>,
    pub object_label: Label,
    /// Pre-computed object centroid (x, y) in meters. Retained for serde
    /// compatibility and debugging even though heading checks now use the
    /// nearest boundary point.
    pub object_centroid: (f64, f64),
    /// Body centroid columns (must be OUTSIDE the object).
    pub centroid_coords: CoordinateColumns,
    /// Nose coordinate columns.
    pub nose_coords: CoordinateColumns,
    /// Head base coordinate columns (for computing head direction).
    pub head_base_coords: CoordinateColumns,
    /// Erosion (meters) of the object footprint used only for the
    /// centroid-outside check; nose distances use the real outline.
    #[serde(default)]
    pub footprint_erosion_meters: f64,
    /// Maximum nose-to-object distance (meters). Default: 0.03 (3cm).
    pub nose_max_distance: f64,
    /// Distance below which nose is in "contact" (angle check skipped). Default: 0.005.
    pub contact_distance: f64,
    /// Maximum head-to-object angle (degrees). Default: 30.
    pub max_angle_deg: f64,
    /// Maximum nose speed (m/s) during exploration. Default: 0.05.
    pub max_nose_speed: f64,
    /// Speed filter application mode. Default: per-frame.
    pub speed_filter_mode: BoutFilterMode,
    /// Angle filter application mode. Default: per-frame.
    pub angle_filter_mode: BoutFilterMode,
    /// Minimum bout duration (seconds). Default: 1.0.
    pub min_bout_seconds: f64,
    /// Maximum gap to bridge (seconds). Default: 0.5.
    pub max_gap_seconds: f64,
    /// Strict recipe bounds: gap < maximum and speed < maximum. False retains
    /// historical inclusive comparisons, including the 2026-10-07 amendment.
    #[serde(default)]
    pub strict_boundaries: bool,
    /// Boolean column of grooming/rearing posture frames. When set, nose
    /// contact no longer rescues a frame in that posture: it needs the head
    /// angle (object-directed) like any other frame.
    #[serde(default)]
    pub posture_col: Option<String>,
    /// Lateral whisker fan unioned into the per-frame candidate (`None` = off).
    #[serde(default)]
    pub whisker: Option<WhiskerFanSpec>,
    /// Gap density threshold (0.0–1.0): a gap is only bridged if the fraction
    /// of true frames in the original input within the gap meets this threshold.
    /// 0.0 = bridge any short gap (default). Higher values require more evidence.
    pub gap_density_threshold: f64,
    /// FPS for converting seconds to frames.
    pub fps: f64,
    /// Per-recording geometry shared by a variant group (`None` = polars geometry).
    #[serde(skip)]
    pub shared: Option<std::sync::Arc<SharedObjectGeometry>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationAttribution {
    pub frames_in_window: u64,
    pub frames_pose_ok: u64,
    pub frames_outside_object: u64,
    pub frames_nose_near: u64,
    pub frames_nose_contact: u64,
    pub frames_within_angle: u64,
    pub frames_contact_or_angle: u64,
    pub frames_slow_nose: u64,
    pub frames_all_gates_raw: u64,
    pub frames_after_tolerance: u64,
    pub bouts_pre_filter: u32,
    pub bouts_post_filter: u32,
    pub bout_lengths_ms_pre: Vec<u64>,
    pub bout_lengths_ms_post: Vec<u64>,
    #[serde(default)]
    pub candidate_bouts_pre_merge: u32,
    #[serde(default)]
    pub bouts_post_merge: u32,
    #[serde(default)]
    pub bouts_removed_by_duration_filter: u32,
    #[serde(default)]
    pub bouts_removed_by_speed_filter: u32,
    #[serde(default)]
    pub final_valid_bouts: u32,
    #[serde(default)]
    pub bouts_removed_by_angle_filter: u32,
    /// Frames outside the object with the nose near it that fail both the
    /// head-angle and the contact criteria.
    #[serde(default)]
    pub candidate_frames_removed_by_angle_criterion: u32,
    /// Frames rescued only by nose contact that the posture exclusion drops.
    #[serde(default)]
    pub candidate_frames_removed_by_posture: u32,
    /// Finite medians, or NaN for an unavailable diagnostic at that bout position.
    #[serde(default, deserialize_with = "deserialize_bout_medians")]
    pub median_nose_speed_per_bout: Vec<f64>,
    /// Finite medians, or NaN for an unavailable diagnostic at that bout position.
    #[serde(default, deserialize_with = "deserialize_bout_medians")]
    pub median_head_angle_per_bout: Vec<f64>,
    #[serde(default)]
    pub bouts_rescued_by_contact_override: u32,
}

// Numeric diagnostic arrays use NaN in memory to preserve bout alignment when
// no finite observations exist. Serde JSON writes that sentinel as null; the
// retained JSON-to-TOML adapter writes an array null as an empty string. Recover
// only those two missing forms here, without coercing other malformed values.
fn deserialize_bout_medians<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Median {
        Number(f64),
        Missing(()),
        Text(String),
    }
    Vec::<Median>::deserialize(deserializer)?
        .into_iter()
        .map(|value| match value {
            Median::Number(value) if value.is_finite() || value.is_nan() => Ok(value),
            Median::Missing(()) => Ok(f64::NAN),
            Median::Text(value) if value.is_empty() => Ok(f64::NAN),
            _ => Err(serde::de::Error::custom(
                "bout median must be finite or missing",
            )),
        })
        .collect()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct BoutFilterDiagnostics {
    pub candidate_bouts_pre_merge: u32,
    pub bouts_post_merge: u32,
    pub removed_by_duration: u32,
    /// Bouts rejected by the speed criterion in per-bout median mode.
    /// This is always 0 when speed filtering runs in per-frame mode.
    pub removed_by_speed: u32,
    pub removed_by_angle: u32,
    pub final_valid_bouts: u32,
    pub per_bout_median_nose_speed: Vec<f64>,
    pub per_bout_median_head_angle_deg: Vec<f64>,
    /// Bouts saved from the per-bout-median angle filter because at
    /// least one of their frames had nose contact (≤ contact_distance).
    /// Counted only in `BoutFilterMode::PerBoutMedian` for angle.
    #[serde(default)]
    pub bouts_rescued_by_contact_override: u32,
}

include!("exploration/shared.rs");

include!("exploration/heuristic_state.rs");

include!("exploration/attribution.rs");

include!("exploration/heuristic_evaluation.rs");

include!("exploration/bout_api.rs");

include!("exploration/bout_selection.rs");

include!("exploration/bout_filters.rs");

include!("exploration/bout_diagnostics.rs");

#[cfg(test)]
#[path = "exploration/tests.rs"]
mod tests;
