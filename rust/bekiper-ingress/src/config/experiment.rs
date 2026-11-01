use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use bekiper_core::inspection::IngressSessionContext;
use bekiper_core::scale::ScaleConfig;
use bekiper_core::types::{ObjectRecognitionProtocol, PoseModelPreset, TaskType};
use bekiper_core::video::FpsConfig;

use super::detection::DetectionConfig;
use super::task_overrides::TaskParamsOverrides;

/// Configuration for a single experiment/session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub name: String,
    pub task_type: TaskType,
    /// FPS: a number for manual override, or omit to auto-detect from video.
    #[serde(default)]
    pub fps: FpsConfig,
    /// Pixel-to-meter scale: either a manual value or a derivation method.
    pub meters_per_pixel: ScaleConfig,

    /// Pre-computed pose CSV/Parquet files (skip pose inference).
    #[serde(default)]
    pub data_files: Vec<PathBuf>,

    /// Video file for pose inference and/or detection.
    pub video: Option<PathBuf>,

    /// Pose model configuration (required if `video` is set and no `data_files`).
    pub pose_model: Option<PoseModelConfig>,

    /// Detection configuration.
    pub detection: Option<DetectionConfig>,

    /// Task-specific parameter overrides.
    ///
    /// ```toml
    /// [experiments.task_params]
    /// freezing_displacement_threshold = 0.005
    /// peripheral_band_fraction = 0.25
    /// centroid_coords = { x = "body_x", y = "body_y" }
    /// ```
    #[serde(default)]
    pub task_params: Option<TaskParamsOverrides>,

    /// Optional perimeter label that defines the trial start area.
    ///
    /// This is currently used by Cheeseboard/reward-tracing to replace the
    /// legacy enclosure-entry proxy with an explicit start-area exit event.
    #[serde(default)]
    pub trial_start_perimeter: Option<String>,

    /// Resolved annotation directory for this experiment's perimeters.
    #[serde(skip)]
    pub annotation_dir: Option<PathBuf>,

    /// Frame timing (populated at runtime from video, not from config file).
    #[serde(skip)]
    pub frame_timing: Option<bekiper_core::video::FrameTiming>,

    /// Video resolution (width, height) parsed from metadata YAML during
    /// `animal_scan` expansion. Used to populate the inspection manifest
    /// when the resolution can't be probed from the video directly.
    #[serde(skip)]
    pub video_resolution: Option<(u32, u32)>,

    /// Path to a single arena screenshot (`{seq}.frame-{animal}.jpeg`).
    /// Used as the background frame for inspection figures when no video
    /// file is available.
    #[serde(skip)]
    pub frame_image: Option<PathBuf>,

    /// Per-frame timestamps CSV (`{seq}.timestamps-{animal}_*.csv`).
    /// When present, an `elapsed_s` column is added to the augmented
    /// dataframe with seconds since the first captured frame.
    #[serde(skip)]
    pub timestamps_file: Option<PathBuf>,

    /// Animal identifier from `[[animal_scan]]`.  When set, the output
    /// directory nests under `{output}/{animal_id}/{sequence_name}/`
    /// instead of the flat `{output}/{experiment_name}/`.
    #[serde(skip)]
    pub animal_id: Option<u32>,

    /// Source-study context populated by ingress profiles.
    #[serde(skip)]
    pub session_context: Option<IngressSessionContext>,

    /// Shared recording identity of parameter variants expanded from one
    /// scan entry (the experiment name without its `__variant` suffix).
    /// Variants of one recording share detect/pose/augment work; `None`
    /// means the experiment is its own recording.
    #[serde(skip)]
    pub recording_id: Option<String>,
}

/// Pose model specification for ONNX inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseModelConfig {
    pub path: PathBuf,
    /// Preset model spec or path to a custom JSON spec file.
    pub spec: PoseModelPreset,
    /// Batch size for inference. Set to 0 for auto-probe.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

fn default_batch_size() -> usize {
    0
}

fn resolve_config_relative_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

impl PoseModelConfig {
    /// Resolve relative pose-model asset paths against the config directory.
    pub fn resolved_against(&self, base_dir: &Path) -> Self {
        Self {
            path: resolve_config_relative_path(base_dir, &self.path),
            spec: match &self.spec {
                PoseModelPreset::Custom(path) => {
                    PoseModelPreset::Custom(resolve_config_relative_path(base_dir, path))
                }
                preset => preset.clone(),
            },
            batch_size: self.batch_size,
        }
    }
}

impl ExperimentConfig {
    /// Whether this experiment needs pose inference (has video but no pre-computed data).
    pub fn needs_pose_inference(&self) -> bool {
        self.data_files.is_empty() && self.video.is_some() && self.pose_model.is_some()
    }

    /// Whether this experiment needs detection (including pre-computed).
    pub fn needs_detection(&self) -> bool {
        self.detection.as_ref().is_some_and(|d| d.recipe.is_some())
    }

    /// Stage-aware object-recognition protocol derived from the detection
    /// recipe when this experiment uses one.
    pub fn object_recognition_protocol(&self) -> Option<ObjectRecognitionProtocol> {
        self.detection
            .as_ref()
            .and_then(|d| d.recipe)
            .and_then(|recipe| recipe.object_recognition_protocol())
    }

    /// Effective FPS: from manual config or auto-detected frame timing.
    pub fn effective_fps(&self) -> bekiper_core::error::Result<f64> {
        match self.fps {
            FpsConfig::Manual(fps) => Ok(fps),
            FpsConfig::FromVideo => {
                if let Some(ref timing) = self.frame_timing {
                    Ok(timing.fps)
                } else {
                    Err(bekiper_core::error::BekiperError::Config(
                        "fps = FromVideo but no video has been probed yet".into(),
                    ))
                }
            }
        }
    }
}
