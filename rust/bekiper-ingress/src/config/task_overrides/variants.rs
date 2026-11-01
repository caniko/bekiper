pub(crate) fn expand_experiment_variants(
    base: ExperimentConfig,
    variants: &[TaskVariant],
) -> Vec<ExperimentConfig> {
    if variants.is_empty() {
        return vec![base];
    }

    let base_name = base.name.clone();
    variants
        .iter()
        .map(|variant| {
            let mut experiment = base.clone();
            experiment.name = format!("{base_name}__{}", variant.name);
            experiment.recording_id = Some(base_name.clone());
            experiment.task_params = Some(
                base.task_params
                    .clone()
                    .unwrap_or_default()
                    .merged(&variant.task_params),
            );
            experiment
        })
        .collect()
}

/// Overrides for object exploration parameters (NORT).
///
/// ```toml
/// [scan.task_params.exploration]
/// nose_max_distance = 0.03
/// contact_distance = 0.005
/// max_angle_deg = 30.0
/// max_nose_speed = 0.05
/// speed_filter_mode = "per_frame"
/// angle_filter_mode = "per_frame"
/// min_bout_seconds = 1.0
/// max_gap_seconds = 0.5
/// min_total_exploration = 15.0
/// analysis_window_seconds = 120.0
/// perimeter_erosion_meters = 0.01
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExplorationOverrides {
    /// Max nose-to-object distance (meters). Default: 0.03 (3 cm).
    pub nose_max_distance: Option<f64>,
    /// Distance below which angle check is skipped (meters). Default: 0.005.
    pub contact_distance: Option<f64>,
    /// Max head-to-object angle (degrees). Default: 30.0.
    pub max_angle_deg: Option<f64>,
    /// Max nose speed during exploration (m/s). Default: 0.05.
    pub max_nose_speed: Option<f64>,
    /// Speed filter application mode: disabled, per_frame (default), or per_bout_median.
    pub speed_filter_mode: Option<BoutFilterMode>,
    /// Angle filter application mode. Default: per_frame.
    pub angle_filter_mode: Option<BoutFilterMode>,
    /// Min exploration bout duration (seconds). Default: 1.0.
    pub min_bout_seconds: Option<f64>,
    /// Max gap to bridge between bouts (seconds). Default: 0.5.
    pub max_gap_seconds: Option<f64>,
    pub strict_boundaries: Option<bool>,
    /// Min total exploration to not exclude animal (seconds). Default: 15.0.
    pub min_total_exploration: Option<f64>,
    /// Analysis window from first approach (seconds). Default: 120.0.
    pub analysis_window_seconds: Option<f64>,
    /// Analysis window selection mode. Default preserves legacy first-approach anchoring.
    pub window_mode: Option<NortWindowMode>,
    /// Gap density threshold (0.0-1.0) for tolerance gap bridging. Default: 0.0.
    pub gap_density_threshold: Option<f64>,
    /// Erode object perimeters by this distance (meters). Default: 0.01.
    pub perimeter_erosion_meters: Option<f64>,
    /// Void nose-contact rescue in grooming/rearing posture-proxy frames. Default: false.
    pub exclude_posture: Option<bool>,
    /// Enable qualia decomposition (body_proximity, olfaction, whiskers). Default: true.
    pub enable_qualia: Option<bool>,
    /// Body-proximity max distance (meters) for qualia. Default: 0.05.
    pub qualia_body_proximity_distance: Option<f64>,
    /// Whisker max distance (meters) for qualia. Default: 0.035.
    pub qualia_whisker_distance: Option<f64>,
    /// Whisker max angle (degrees) for qualia. Default: 45.0.
    pub qualia_whisker_angle: Option<f64>,
    /// Olfaction max distance (meters) for qualia. Default: 0.05.
    pub qualia_olfaction_distance: Option<f64>,
    /// Olfaction max angle (degrees) for qualia. Default: 60.0.
    pub qualia_olfaction_angle: Option<f64>,
    /// Named merge rules for combining qualia channels per object.
    /// ```toml
    /// [animal_scan.task_params.exploration.merge_rules.close_investigation]
    /// require = ["olfaction", "whisker_interaction"]
    /// ```
    #[serde(default)]
    pub merge_rules:
        Option<HashMap<String, bekiper_feature::heuristic::combined::merge::MergeRule>>,
    /// Lateral whisker fan unioned into object observation (disabled unless
    /// `enabled = true`). Neo/Imen owner decision 2026-10-28.
    ///
    /// ```toml
    /// [ingress_profile.tasks.task_params.exploration.whisker]
    /// enabled = true
    /// length = { meters = 0.0117 }  # or { method = "age_table", source = "mice_overview.xlsx", table = [[0, 0.0117]] }
    /// ```
    #[serde(default)]
    pub whisker: Option<bekiper_feature::heuristic::solo::whisker_config::WhiskerOverrides>,
}

/// Overrides for Y-maze / radial arm maze parameters.
///
/// ```toml
/// [scan.task_params.ymaze]
/// proximal_threshold = 0.25
/// threshold_geometry = "imen_radial_vertex_span_v1"
/// start_mode = "first_detection"  # or "video_start"
/// # Geometry: imen_radial_vertex_span_v1 (fixed; see Y-maze threshold docs)
/// # Principal defaults: >= 1 s, uninterrupted dual-landmark membership.
/// # Override with [scan.task_params.ymaze.temporal_policy]:
/// # min_duration_seconds = 0.0 disables dwell for historical reproduction.
/// arm_membership_hysteresis = false
/// zone_max_gap_seconds = 0.0
/// pose_gap_fill_seconds = 0.0
/// zone_occupancy = "dwell_filtered"  # or "raw"
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct YMazeOverrides {
    /// Explicit generic seconds-based policy; overrides legacy dwell fields.
    pub temporal_policy: Option<bekiper_feature::tolerance::TemporalPolicy>,
    /// Optional identity assertion for the fixed Imen radial geometry.
    /// This is not a tunable geometry policy; alternate names fail parsing.
    pub threshold_geometry: Option<bekiper_behaviour::radial_arm::YMazeThresholdGeometry>,
    /// Fraction of radial vertex span: r_min + p * (r_max - r_min),
    /// measured from the arithmetic mean of canonical center vertices.
    pub proximal_threshold: Option<f64>,
    /// Fractions scored as principal analyses (radial geometry only). Default:
    /// `bekiper_behaviour::radial_arm::DEFAULT_PRINCIPAL_THRESHOLDS`.
    pub principal_thresholds: Option<Vec<f64>>,
    /// Where the recording window begins. Default: `first_detection`.
    pub start_mode: Option<bekiper_behaviour::radial_arm::YMazeStartMode>,
    /// Whether the ≥`min_dwell_seconds` filter is applied on top of the
    /// arm-threshold crossing when scoring arm entries. Default: `true`
    /// for historical/comparison thresholds. Set to `false` when the project considers the
    /// arm-threshold crossing (head + tail past the proximal line) a
    /// sufficient definition of a valid arm entry. Principal 25/45/65% scoring
    /// bypasses this historical option; use `temporal_policy` for explicit dwell.
    pub apply_dwell_filter: Option<bool>,
    /// Whether `in_arm_X` should latch until both head and tail retreat behind
    /// the proximal threshold or either body part leaves the arm polygon.
    /// Default: `true`.
    pub arm_membership_hysteresis: Option<bool>,
    /// Maximum length of a contiguous run of `"none"` frames that should be
    /// treated as part of the previous bout instead of resetting the
    /// zone-sequence state. Default 0.1 s ≈ 3 frames @ 30 fps. Set to 0.0 to
    /// reproduce the legacy reset-on-every-none behaviour.
    pub zone_max_gap_seconds: Option<f64>,
    /// Forward-fill short runs of null in head/tail coordinate columns before
    /// Y-maze zone heuristics run. Default 0.2 s ≈ 6 frames @ 30 fps. Set to
    /// 0.0 to disable.
    pub pose_gap_fill_seconds: Option<f64>,
    /// Frames counted by zone time/entries/distance/speed and arm preference:
    /// `dwell_filtered` (default, ≥ dwell visits only) or `raw` (every frame).
    pub zone_occupancy: Option<bekiper_feature::zone::ZoneOccupancy>,
}

/// Overrides for the rearing heuristic.
///
/// ```toml
/// [scan.task_params.rearing]
/// max_speed = 0.01
/// max_body_length_ratio = 0.7
/// max_body_length = 0.05
/// forepaw_proximity = 0.03
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RearingOverrides {
    /// Max centroid speed during rearing (m/s). Default: 0.01.
    pub max_speed: Option<f64>,
    /// Max nose-to-tail ratio during rearing (fraction of baseline). Default: 0.7.
    pub max_body_length_ratio: Option<f64>,
    /// Absolute max nose-to-tail distance (meters). Default: 0.05.
    pub max_body_length: Option<f64>,
    /// Forepaw-to-head proximity threshold (meters). Default: 0.03.
    pub forepaw_proximity: Option<f64>,
}

/// Overrides for the grooming heuristic.
///
/// ```toml
/// [scan.task_params.grooming]
/// max_speed = 0.005
/// max_body_length = 0.08
/// min_body_length = 0.04
/// forepaw_proximity = 0.02
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroomingOverrides {
    /// Max speed during grooming (m/s). Default: 0.005.
    pub max_speed: Option<f64>,
    /// Max nose-to-tail distance (meters). Default: 0.08.
    pub max_body_length: Option<f64>,
    /// Min nose-to-tail distance (meters). Default: 0.04.
    pub min_body_length: Option<f64>,
    /// Forepaw-to-nose proximity threshold (meters). Default: 0.02.
    pub forepaw_proximity: Option<f64>,
}
