use serde::{Deserialize, Serialize};

use bekiper_core::types::{MidpointGroup, TaskType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SkeletonRepairMode {
    #[default]
    RedetectThenInterpolate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SmoothingMethod {
    #[default]
    Median,
    Kalman,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PoseLabelModel {
    #[default]
    SuperanimalTopviewmouse,
    ImenDlc,
}

impl PoseLabelModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SuperanimalTopviewmouse => "superanimal_topviewmouse",
            Self::ImenDlc => "imen_dlc",
        }
    }

    /// Task-agnostic required labels. The Imen DLC model trained the Y-maze
    /// videos with a `base_tail` keypoint and the OFT/NORT videos with a
    /// `tail` keypoint, so use [`PoseLabelModel::required_source_labels_for_task`] when
    /// validating against an actual experiment.
    pub fn required_source_labels(self) -> &'static [&'static str] {
        match self {
            Self::SuperanimalTopviewmouse => &["mouse_center", "nose", "center_ear", "tail_base"],
            Self::ImenDlc => &["nose", "left_ear", "right_ear", "tail"],
        }
    }

    /// Required source labels for this model when scoring a specific task.
    ///
    /// Falls back to [`PoseLabelModel::required_source_labels`] when no task-specific
    /// override is needed.
    pub fn required_source_labels_for_task(
        self,
        task_type: bekiper_core::types::TaskType,
    ) -> &'static [&'static str] {
        use bekiper_core::types::TaskType;
        match (self, task_type) {
            (Self::ImenDlc, TaskType::YMaze) => &["nose", "left_ear", "right_ear", "base_tail"],
            (Self::ImenDlc, TaskType::Cheeseboard) => &["nose", "left_ear", "right_ear"],
            _ => self.required_source_labels(),
        }
    }

    pub fn derived_labels(self) -> &'static [&'static str] {
        match self {
            Self::SuperanimalTopviewmouse => &[],
            Self::ImenDlc => &["center_ear", "body_center"],
        }
    }

    pub fn midpoint_groups_for_task(self, task_type: TaskType) -> Vec<MidpointGroup> {
        match (self, task_type) {
            (Self::ImenDlc, TaskType::OpenField | TaskType::ObjectRecognition) => vec![
                midpoint_group("center_ear", &["left_ear", "right_ear"]),
                midpoint_group("body_center", &["nose", "tail"]),
            ],
            (Self::ImenDlc, TaskType::YMaze) => {
                vec![midpoint_group("center_ear", &["left_ear", "right_ear"])]
            }
            (Self::ImenDlc, TaskType::Cheeseboard) => {
                vec![midpoint_group("center_ear", &["left_ear", "right_ear"])]
            }
            (Self::SuperanimalTopviewmouse, _) => Vec::new(),
        }
    }

    pub fn skeleton_preset_for_task(
        self,
        task_type: bekiper_core::types::TaskType,
    ) -> bekiper_pose::skeleton::SkeletonPreset {
        use bekiper_core::types::TaskType;
        match (self, task_type) {
            (Self::SuperanimalTopviewmouse, _) => {
                bekiper_pose::skeleton::SkeletonPreset::MouseTopView
            }
            (Self::ImenDlc, TaskType::YMaze) => {
                bekiper_pose::skeleton::SkeletonPreset::ImenDlcBaseTail
            }
            (Self::ImenDlc, _) => bekiper_pose::skeleton::SkeletonPreset::ImenDlcTail,
        }
    }

    pub fn infer_from_labels(labels: &[String]) -> Option<Self> {
        let has_all = |required: &[&str]| {
            required
                .iter()
                .all(|label| labels.iter().any(|v| v == label))
        };
        if has_all(Self::SuperanimalTopviewmouse.required_source_labels()) {
            Some(Self::SuperanimalTopviewmouse)
        } else if has_all(Self::ImenDlc.required_source_labels()) {
            Some(Self::ImenDlc)
        } else {
            None
        }
    }

    pub fn missing_source_labels(self, labels: &[String]) -> Vec<&'static str> {
        self.required_source_labels()
            .iter()
            .copied()
            .filter(|required| !labels.iter().any(|label| label == required))
            .collect()
    }

    pub fn missing_source_labels_for_task(
        self,
        task_type: bekiper_core::types::TaskType,
        labels: &[String],
    ) -> Vec<&'static str> {
        self.required_source_labels_for_task(task_type)
            .iter()
            .copied()
            .filter(|required| !labels.iter().any(|label| label == required))
            .collect()
    }
}

fn midpoint_group(output_label: &str, source_labels: &[&str]) -> MidpointGroup {
    MidpointGroup {
        output_label: output_label.to_string(),
        source_labels: source_labels
            .iter()
            .map(|label| label.to_string())
            .collect(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonRepairConfig {
    pub preset: bekiper_pose::skeleton::SkeletonPreset,
    #[serde(default)]
    pub repair: SkeletonRepairMode,
    #[serde(default = "default_skeleton_relative_tolerance")]
    pub relative_tolerance: f64,
    #[serde(default = "default_skeleton_mad_multiplier")]
    pub mad_multiplier: f64,
    #[serde(default = "default_skeleton_min_baseline_frames")]
    pub min_baseline_frames: usize,
}

/// Pipeline parameters shared across all experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pose label model used by read-only audit/report commands to decide
    /// which task-relevant labels should be scored by default.
    #[serde(default)]
    pub pose_label_model: PoseLabelModel,
    #[serde(default = "default_likelihood_threshold")]
    pub likelihood_threshold: f64,
    #[serde(default = "default_smoothing_window")]
    pub smoothing_window: usize,
    /// Coordinate smoother applied after outlier nulling and before metric conversion.
    #[serde(default)]
    pub smoothing_method: SmoothingMethod,
    /// Constant-velocity Kalman/RTS process noise in pixel-space units.
    #[serde(default = "default_kalman_process_noise")]
    pub kalman_process_noise: f64,
    /// Baseline Kalman observation noise in pixel-space units.
    #[serde(default = "default_kalman_measurement_noise")]
    pub kalman_measurement_noise: f64,
    /// Lower confidence bound for Kalman likelihood-weighted measurement noise.
    #[serde(default = "default_kalman_min_likelihood")]
    pub kalman_min_likelihood: f64,
    /// Null gaps longer than this remain null after Kalman smoothing.
    #[serde(default = "default_kalman_max_gap_frames")]
    pub kalman_max_gap_frames: usize,
    #[serde(default = "default_max_velocity")]
    pub max_velocity: f64,
    /// Maximum iterations for high-velocity cascade resolution. Default: 3.
    #[serde(default = "default_high_velocity_iterations")]
    pub high_velocity_iterations: u32,
    /// Null frames where displacement > N * std(displacement). 0 = disabled. Default: 0.
    #[serde(default)]
    pub displacement_sigma_threshold: f64,
    /// Trim trial boundaries where ALL keypoint likelihoods are below the threshold.
    /// When true, frames at the start/end of the recording where pose estimation
    /// was unreliable are dropped. Default: false.
    #[serde(default)]
    pub likelihood_trim: bool,
    /// Seconds to crop from the start of each trial recording. Default: 0.
    #[serde(default)]
    pub crop_start_seconds: f64,
    /// Seconds to crop from the end of each trial recording. Default: 0.
    #[serde(default)]
    pub crop_end_seconds: f64,
    /// Target trial length in seconds after cropping. 0 = no target. Default: 0.
    #[serde(default)]
    pub target_trial_length_seconds: f64,
    /// Invert Y coordinates using the source frame height before converting
    /// to meters. Default: false.
    #[serde(default)]
    pub invert_y: bool,
    /// Midpoint groups: compute virtual body-part midpoints during augmentation.
    ///
    /// ```toml
    /// [[pipeline.midpoint_groups]]
    /// output_label = "center_ear"
    /// source_labels = ["left_ear", "right_ear"]
    ///
    /// [[pipeline.midpoint_groups]]
    /// output_label = "mouse_center"
    /// source_labels = ["center_ear", "tail_base"]
    /// ```
    #[serde(default)]
    pub midpoint_groups: Vec<bekiper_core::types::MidpointGroup>,
    /// Optional topology-aware pose repair using a configured skeleton preset.
    #[serde(default)]
    pub skeleton: Option<SkeletonRepairConfig>,
    /// Read-only pose quality reporting thresholds.
    #[serde(default)]
    pub quality: PoseQualityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseQualityConfig {
    #[serde(default = "default_quality_max_low_likelihood_fraction")]
    pub max_low_likelihood_fraction: f64,
    #[serde(default = "default_quality_max_null_gap_seconds")]
    pub max_null_gap_seconds: f64,
    #[serde(default)]
    pub max_augmented_jump_count: u32,
}

fn default_likelihood_threshold() -> f64 {
    0.9
}

fn default_smoothing_window() -> usize {
    5
}

fn default_max_velocity() -> f64 {
    50.0
}

fn default_kalman_process_noise() -> f64 {
    1.0
}

fn default_kalman_measurement_noise() -> f64 {
    4.0
}

fn default_kalman_min_likelihood() -> f64 {
    0.05
}

fn default_kalman_max_gap_frames() -> usize {
    30
}

fn default_high_velocity_iterations() -> u32 {
    3
}

fn default_skeleton_relative_tolerance() -> f64 {
    0.35
}

fn default_skeleton_mad_multiplier() -> f64 {
    4.0
}

fn default_skeleton_min_baseline_frames() -> usize {
    30
}

fn default_quality_max_low_likelihood_fraction() -> f64 {
    0.05
}

fn default_quality_max_null_gap_seconds() -> f64 {
    0.5
}

impl Default for PoseQualityConfig {
    fn default() -> Self {
        Self {
            max_low_likelihood_fraction: default_quality_max_low_likelihood_fraction(),
            max_null_gap_seconds: default_quality_max_null_gap_seconds(),
            max_augmented_jump_count: 0,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            pose_label_model: PoseLabelModel::default(),
            likelihood_threshold: default_likelihood_threshold(),
            smoothing_window: default_smoothing_window(),
            smoothing_method: SmoothingMethod::default(),
            kalman_process_noise: default_kalman_process_noise(),
            kalman_measurement_noise: default_kalman_measurement_noise(),
            kalman_min_likelihood: default_kalman_min_likelihood(),
            kalman_max_gap_frames: default_kalman_max_gap_frames(),
            max_velocity: default_max_velocity(),
            high_velocity_iterations: default_high_velocity_iterations(),
            displacement_sigma_threshold: 0.0,
            likelihood_trim: false,
            crop_start_seconds: 0.0,
            crop_end_seconds: 0.0,
            target_trial_length_seconds: 0.0,
            invert_y: false,
            midpoint_groups: Vec::new(),
            skeleton: None,
            quality: PoseQualityConfig::default(),
        }
    }
}

impl PipelineConfig {
    pub fn model_skeleton_repair_config(
        &self,
        task_type: bekiper_core::types::TaskType,
    ) -> SkeletonRepairConfig {
        SkeletonRepairConfig {
            preset: self.pose_label_model.skeleton_preset_for_task(task_type),
            repair: SkeletonRepairMode::default(),
            relative_tolerance: default_skeleton_relative_tolerance(),
            mad_multiplier: default_skeleton_mad_multiplier(),
            min_baseline_frames: default_skeleton_min_baseline_frames(),
        }
    }

    pub fn has_model_skeleton_repair(&self) -> bool {
        // Exhaustive so that a new model must decide.
        match self.pose_label_model {
            PoseLabelModel::SuperanimalTopviewmouse | PoseLabelModel::ImenDlc => true,
        }
    }

    /// Overlay environment-variable settings on top of this config.
    ///
    /// Supported keys:
    /// - `BEKIPER_PIPELINE_POSE_LABEL_MODEL` — `superanimal_topviewmouse` or
    ///   `imen_dlc`.
    /// - `BEKIPER_PIPELINE_SMOOTHING_METHOD` — `median` or `kalman`.
    /// - `BEKIPER_PIPELINE_KALMAN_PROCESS_NOISE` — positive float.
    /// - `BEKIPER_PIPELINE_KALMAN_MEASUREMENT_NOISE` — positive float.
    /// - `BEKIPER_PIPELINE_KALMAN_MIN_LIKELIHOOD` — float in `[0, 1]`.
    /// - `BEKIPER_PIPELINE_KALMAN_MAX_GAP_FRAMES` — non-negative integer.
    /// - `BEKIPER_PIPELINE_INVERT_Y` — boolean flag (`1`/`0`, `true`/`false`,
    ///   `yes`/`no`, `on`/`off`; case-insensitive).
    pub fn apply_env_overrides(&mut self) {
        self.apply_env_overrides_from(|k| std::env::var(k).ok());
    }

    /// Testable env overlay variant using an injected lookup closure.
    pub fn apply_env_overrides_from<F>(&mut self, lookup: F)
    where
        F: Fn(&str) -> Option<String>,
    {
        if let Some(raw) = lookup("BEKIPER_PIPELINE_POSE_LABEL_MODEL") {
            match parse_pose_label_model(&raw) {
                Some(model) => {
                    tracing::info!(model = ?model, "BEKIPER_PIPELINE_POSE_LABEL_MODEL override applied");
                    self.pose_label_model = model;
                }
                None => tracing::warn!(
                    value = %raw,
                    "BEKIPER_PIPELINE_POSE_LABEL_MODEL must be 'superanimal_topviewmouse' or 'imen_dlc', ignoring"
                ),
            }
        }
        if let Some(raw) = lookup("BEKIPER_PIPELINE_SMOOTHING_METHOD") {
            match parse_smoothing_method(&raw) {
                Some(method) => {
                    tracing::info!(method = ?method, "BEKIPER_PIPELINE_SMOOTHING_METHOD override applied");
                    self.smoothing_method = method;
                }
                None => tracing::warn!(
                    value = %raw,
                    "BEKIPER_PIPELINE_SMOOTHING_METHOD must be 'median' or 'kalman', ignoring"
                ),
            }
        }
        if let Some(value) = parse_positive_f64_env(
            &lookup,
            "BEKIPER_PIPELINE_KALMAN_PROCESS_NOISE",
            "Kalman process noise",
        ) {
            self.kalman_process_noise = value;
        }
        if let Some(value) = parse_positive_f64_env(
            &lookup,
            "BEKIPER_PIPELINE_KALMAN_MEASUREMENT_NOISE",
            "Kalman measurement noise",
        ) {
            self.kalman_measurement_noise = value;
        }
        if let Some(raw) = lookup("BEKIPER_PIPELINE_KALMAN_MIN_LIKELIHOOD") {
            match raw.trim().parse::<f64>() {
                Ok(value) if (0.0..=1.0).contains(&value) => {
                    tracing::info!("BEKIPER_PIPELINE_KALMAN_MIN_LIKELIHOOD override applied");
                    self.kalman_min_likelihood = value;
                }
                _ => tracing::warn!(
                    value = %raw,
                    "BEKIPER_PIPELINE_KALMAN_MIN_LIKELIHOOD must be in [0, 1], ignoring"
                ),
            }
        }
        if let Some(raw) = lookup("BEKIPER_PIPELINE_KALMAN_MAX_GAP_FRAMES") {
            match raw.trim().parse::<usize>() {
                Ok(value) => {
                    tracing::info!("BEKIPER_PIPELINE_KALMAN_MAX_GAP_FRAMES override applied");
                    self.kalman_max_gap_frames = value;
                }
                _ => tracing::warn!(
                    value = %raw,
                    "BEKIPER_PIPELINE_KALMAN_MAX_GAP_FRAMES must be a non-negative integer, ignoring"
                ),
            }
        }
        if let Some(raw) = lookup("BEKIPER_PIPELINE_INVERT_Y") {
            let trimmed = raw.trim().to_ascii_lowercase();
            match trimmed.as_str() {
                "1" | "true" | "yes" | "on" => {
                    tracing::info!("BEKIPER_PIPELINE_INVERT_Y=true override applied");
                    self.invert_y = true;
                }
                "0" | "false" | "no" | "off" => {
                    tracing::info!("BEKIPER_PIPELINE_INVERT_Y=false override applied");
                    self.invert_y = false;
                }
                "" => tracing::warn!("BEKIPER_PIPELINE_INVERT_Y is empty, ignoring"),
                _ => tracing::warn!(
                    value = %raw,
                    "BEKIPER_PIPELINE_INVERT_Y is not a boolean flag, ignoring"
                ),
            }
        }
    }
}

fn parse_pose_label_model(raw: &str) -> Option<PoseLabelModel> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "superanimal_topviewmouse" => Some(PoseLabelModel::SuperanimalTopviewmouse),
        "imen_dlc" => Some(PoseLabelModel::ImenDlc),
        _ => None,
    }
}

fn parse_smoothing_method(raw: &str) -> Option<SmoothingMethod> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "median" => Some(SmoothingMethod::Median),
        "kalman" => Some(SmoothingMethod::Kalman),
        _ => None,
    }
}

fn parse_positive_f64_env<F>(lookup: &F, key: &str, label: &str) -> Option<f64>
where
    F: Fn(&str) -> Option<String>,
{
    let raw = lookup(key)?;
    match raw.trim().parse::<f64>() {
        Ok(value) if value > 0.0 => {
            tracing::info!(key, label, "pipeline numeric override applied");
            Some(value)
        }
        _ => {
            tracing::warn!(
                key,
                label,
                value = %raw,
                "pipeline numeric override must be positive, ignoring"
            );
            None
        }
    }
}

#[cfg(test)]
#[path = "pipeline/pose_label_model_tests.rs"]
mod pose_label_model_tests;
