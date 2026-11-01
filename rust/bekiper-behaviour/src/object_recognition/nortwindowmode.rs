use std::collections::HashMap;

use polars::prelude::*;

use bekiper_core::inspection::{
    HeuristicDecompositionComponent, HeuristicDecompositionSpec, QuadrantSpec,
};
use bekiper_core::shape::{ClosestPoint, Shape};
use bekiper_core::types::{CoordinateColumns, ObjectRecognitionProtocol, TaskType};
use bekiper_core::video::FrameTiming;
use bekiper_feature::heuristic::HeuristicSummary;
use bekiper_feature::heuristic::combined::CombinedHeuristic;
use bekiper_feature::heuristic::combined::exploration::{
    BoutFilterMode, BoutFilterSpec, ExplorationAttribution, ExplorationHeuristic,
    object_exploration_with_timing,
};
use bekiper_feature::heuristic::combined::merge::{MergeRule, MergedHeuristic};
use bekiper_feature::heuristic::helper::outside_perimeter::OutsidePerimeterHeuristic;
use bekiper_feature::heuristic::solo::body_proximity::BodyProximityHeuristic;
use bekiper_feature::heuristic::solo::freezing::FreezingHeuristic;
use bekiper_feature::heuristic::solo::olfaction::OlfactionHeuristic;
use bekiper_feature::heuristic::solo::periphery::PeripheryHeuristic;
use bekiper_feature::heuristic::solo::whiskers::{
    WhiskerFanHeuristic, WhiskerFanParams, WhiskerFanSpec, WhiskerHeuristic,
};
use bekiper_feature::heuristic::solo::zone_confinement::ZoneConfinementHeuristic;
use bekiper_feature::heuristic::{
    Heuristic, coordinate_label_from_xy, decomposition_component, diag_column_name,
};
use bekiper_feature::quadrant::{add_grid_zone_columns, generate_grid_zones};
use bekiper_feature::tolerance::BoutTiming;
use bekiper_feature::zone::{ZoneSummary, zone_summary};
use bekiper_math::discrete::{apply_tolerance_series, apply_tolerance_series_with_density};
use bekiper_math::geometry::path_distance;
use bekiper_perimeter::object_shape::ObjectShape;
use bekiper_perimeter::perimeter::Perimeter;

use crate::BehaviouralTask;
use crate::TaskDetail;
use crate::motion_detail::{ZoneMotionSummary, motion_summary_for_zone, motion_summary_from_mask};
use crate::nort::{
    NortIndices, QualiaChannel, QualiaDecomposition, build_observation_sequence,
    count_alternations, object_bias, side_bias,
};
use crate::params::{DEFAULT_OFT_GROOMING_MIN_BOUT_S, DEFAULT_OFT_REARING_MIN_BOUT_S, TaskParams};
use bekiper_feature::heuristic::helper::posture_bouts::PostureBoutHeuristic;
use bekiper_feature::heuristic::solo::grooming::GroomingHeuristic;
use bekiper_feature::heuristic::solo::rearing::RearingHeuristic;

/// Grooming/rearing posture-proxy bouts used by `exclude_posture`.
pub const POSTURE_COLUMN: &str = "nort_posture_excluded";

// ── NORT exploration defaults ────────────────────────────────────────────
// The per-frame detection knobs live in `bekiper-feature` next to the
// algorithm; they are re-exported here for doc discoverability. The
// task-layer-only knobs (analysis window, exclusion criterion, perimeter
// erosion) are defined in-place because they shape the task output and
// have no meaning inside a single-frame heuristic.
//
// All values are overridable at runtime via `TaskParamsOverrides::exploration`
// in `bekiper-ingress/src/config.rs`.

pub use bekiper_feature::heuristic::combined::exploration::{
    DEFAULT_CONTACT_DISTANCE, DEFAULT_GAP_DENSITY_THRESHOLD, DEFAULT_MAX_ANGLE_DEG,
    DEFAULT_MAX_GAP_SECONDS, DEFAULT_MAX_NOSE_SPEED, DEFAULT_MIN_BOUT_SECONDS,
    DEFAULT_NOSE_MAX_DISTANCE,
};

/// Minimum total exploration (s) for an animal to be included in analysis.
/// Falls below → animal is marked excluded. Task-level concern only.
pub const DEFAULT_MIN_TOTAL_EXPLORATION: f64 = 15.0;

/// Analysis window duration (s) starting from the first object approach.
/// Default 120 s (2 min) matches standard NORT protocols. Set to infinity
/// via config to analyze the full trial. Task-level concern only.
pub const DEFAULT_ANALYSIS_WINDOW_SECONDS: f64 = 120.0;

/// Object-perimeter erosion (m). Perimeters are shrunk by this amount
/// before proximity checks to compensate for noisy detection masks.
/// Task-level concern only.
pub const DEFAULT_PERIMETER_EROSION_METERS: f64 = 0.01;

const OBJECT_RECOGNITION_QUADRANT_ROWS: usize = 2;
const OBJECT_RECOGNITION_QUADRANT_COLS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NortWindowMode {
    #[default]
    Unknown,
    FullAnalysis,
    AnchoredFirstApproach,
    LeadingNoApproach,
    BackshiftedLateApproach,
    FullTrialShortTrial,
    ConfiguredFullTrial,
}

impl NortWindowMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::FullAnalysis => "full_analysis",
            Self::AnchoredFirstApproach => "anchored_first_approach",
            Self::LeadingNoApproach => "leading_no_approach",
            Self::BackshiftedLateApproach => "backshifted_late_approach",
            Self::FullTrialShortTrial => "full_trial_short_trial",
            Self::ConfiguredFullTrial => "configured_full_trial",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WindowSelection {
    mode: NortWindowMode,
    start_frame: usize,
    end_frame_exclusive: usize,
}

trait ObjectRecognitionMetricUnit {
    fn compute(
        &self,
        ctx: &ObjectRecognitionMetricContext<'_>,
        metrics: &mut ObjectRecognitionMetrics,
    );
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ObservationTotals {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub object_labels: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observation_seconds: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observation_bouts: Vec<u32>,
    pub total_seconds: f64,
    pub total_bouts: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PairwiseObservationMetric {
    pub lhs_label: String,
    pub rhs_label: String,
    pub absolute_delta_seconds: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualiaDiagnosticRollup {
    pub label: String,
    pub body_center_ear_seconds: f64,
    pub body_tail_base_seconds: f64,
    pub body_total_seconds: f64,
    pub olfaction_nose_seconds: f64,
    pub olfaction_ray_seconds: f64,
    pub olfaction_total_seconds: f64,
    pub whisker_left_proximity_seconds: f64,
    pub whisker_left_fov_seconds: f64,
    pub whisker_right_proximity_seconds: f64,
    pub whisker_right_fov_seconds: f64,
    pub whisker_total_seconds: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ObjectRecognitionZoneMetrics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all_motion_summary: Option<ZoneMotionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center_summary: Option<ZoneSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub periphery_summary: Option<ZoneSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quadrant_summaries: Vec<ZoneSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center_motion_summary: Option<ZoneMotionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub periphery_motion_summary: Option<ZoneMotionSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quadrant_motion_summaries: Vec<ZoneMotionSummary>,
}

#[derive(Default)]
struct ObjectRecognitionMetrics {
    observation_totals: Option<ObservationTotals>,
    pairwise_observations: Vec<PairwiseObservationMetric>,
    novelty_indices: Option<NortIndices>,
    qualia_rollups: Vec<QualiaDiagnosticRollup>,
    zone_metrics: Option<ObjectRecognitionZoneMetrics>,
}

struct ObjectRecognitionMetricContext<'a> {
    task: &'a ObjectRecognition,
    df: &'a DataFrame,
    clock: &'a BoutTiming,
    object_labels: Vec<String>,
    observation_seconds: Vec<f64>,
    observation_bouts: Vec<u32>,
    arena: Option<Perimeter<ObjectShape>>,
}

struct ObservationTotalsUnit;
struct PairwiseObservationUnit;
struct NoveltyIndexUnit;
struct QualiaBreakdownUnit;
struct ZoneMotionUnit;

fn persist_boolean_mask(df: &mut DataFrame, col_name: &str, mask: &BooleanChunked) {
    let values: Vec<bool> = mask
        .into_iter()
        .map(|value| value.unwrap_or(false))
        .collect();
    let new_col = BooleanChunked::from_slice(col_name.into(), &values);
    let _ = df.with_column(new_col.into_column());
}

fn merge_rule_components(
    rule: &MergeRule,
    coordinate_label: &str,
    perimeter_label: Option<String>,
) -> Vec<HeuristicDecompositionComponent> {
    let mut components = Vec::new();
    for col_name in &rule.require {
        components.push(decomposition_component(
            format!("require: {col_name}"),
            col_name.clone(),
            vec![coordinate_label.to_string()],
            perimeter_label.clone(),
        ));
    }
    for col_name in &rule.any_of {
        components.push(decomposition_component(
            format!("any_of: {col_name}"),
            col_name.clone(),
            vec![coordinate_label.to_string()],
            perimeter_label.clone(),
        ));
    }
    for col_name in &rule.exclude {
        components.push(decomposition_component(
            format!("exclude: {col_name}"),
            col_name.clone(),
            vec![coordinate_label.to_string()],
            perimeter_label.clone(),
        ));
    }
    components
}

/// Novel Object Recognition task.
///
/// Animals explore objects; the combined exploration heuristic detects
/// investigation via nose proximity + head orientation + speed filter.
/// Accepts any perimeter shape (Circle, Rectangle, Polygon, Triangle)
/// via the `ObjectShape` enum.
#[derive(Clone)]
pub struct ObjectRecognition {
    pub params: TaskParams,
    /// Stage-aware object-recognition semantics.
    pub protocol: ObjectRecognitionProtocol,
    pub object_perimeters: Vec<Perimeter<ObjectShape>>,
    /// Optional center zone (inner rectangle) for latency computation.
    /// Same definition as OFT: arena minus peripheral band.
    pub center_perimeter: Option<Perimeter<ObjectShape>>,
    /// Maximum nose-to-object distance (meters). Default: 0.03 (3cm).
    pub nose_max_distance: f64,
    /// Distance below which angle check is skipped (meters). Default: 0.005.
    pub contact_distance: f64,
    /// Maximum head-to-object angle (degrees). Default: 30.
    pub max_angle_deg: f64,
    /// Maximum nose speed during exploration (m/s). Default: 0.05.
    pub max_nose_speed: f64,
    /// Speed filter application mode. Default: per-frame.
    pub speed_filter_mode: BoutFilterMode,
    pub strict_boundaries: bool,
    /// Angle filter application mode. Default: per-frame.
    pub angle_filter_mode: BoutFilterMode,
    /// Minimum bout duration (seconds). Default: 1.0.
    pub min_bout_seconds: f64,
    /// Maximum gap to bridge (seconds). Default: 0.5.
    pub max_gap_seconds: f64,
    /// Minimum total exploration to not be excluded (seconds). Default: 15.0.
    pub min_total_exploration: f64,
    /// Analysis window duration (seconds) starting from first object approach.
    /// Default: 120.0 (2 minutes). Set to f64::INFINITY to analyze the full trial.
    pub analysis_window_seconds: f64,
    /// How to select the NORT analysis window.
    pub window_mode: NortWindowMode,
    /// Gap density threshold (0.0–1.0) for tolerance gap bridging.
    /// A gap is only bridged if the fraction of true frames within it exceeds
    /// this threshold. Default: 0.0 (bridge any short gap).
    pub gap_density_threshold: f64,
    /// Erode object perimeters by this distance (meters) before use,
    /// to limit perimeter artifacts from imperfect detection masks.
    /// Default: 0.01 (1 cm). Set to 0.0 to disable.
    pub perimeter_erosion_meters: f64,
    /// Drop nose-contact rescue in grooming (≥1 s) or rearing (≥0.5 s)
    /// posture-proxy frames: they count only with the head angle met.
    /// Default: false.
    pub exclude_posture: bool,
    /// Enable qualia decomposition: run body_proximity, olfaction, and
    /// whisker_interaction heuristics per object alongside the combined
    /// exploration heuristic. Default: true.
    pub enable_qualia: bool,
    /// Body-proximity max distance (meters) for qualia. Default: 0.05 (5cm).
    pub qualia_body_proximity_distance: f64,
    /// Whisker max distance (meters) for qualia. Default: 0.035 (3.5cm).
    pub qualia_whisker_distance: f64,
    /// Whisker max angle (degrees) for qualia. Default: 45.0.
    pub qualia_whisker_angle: f64,
    /// Olfaction max distance (meters) for qualia. Default: 0.05 (5cm).
    pub qualia_olfaction_distance: f64,
    /// Olfaction max angle (degrees) for qualia. Default: 60.0.
    pub qualia_olfaction_angle: f64,
    /// Named merge rules for combining qualia channels.
    /// Each rule produces a merged boolean column per object, and its
    /// exploration seconds are reported in `QualiaDecomposition.merged`.
    ///
    /// ```toml
    /// [animal_scan.task_params.exploration.merge_rules.close_investigation]
    /// require = ["olfaction", "whisker_interaction"]
    /// ```
    pub merge_rules: HashMap<String, MergeRule>,
    /// Lateral whisker fan unioned into object observation (`None` = nose
    /// rule only). Neo/Imen owner decision 2026-10-28.
    pub whisker_fan: Option<WhiskerFanParams>,
    /// Provenance of an age-table length: animal age (days) and its source.
    pub whisker_age_days: Option<f64>,
    pub whisker_length_source: Option<String>,
    /// Per-object geometry shared by a variant group, keyed by object label
    /// (empty = per-variant polars geometry). Set by `analyze_variants`.
    pub shared_geometry: HashMap<
        String,
        std::sync::Arc<bekiper_feature::heuristic::combined::exploration::SharedObjectGeometry>,
    >,
}
