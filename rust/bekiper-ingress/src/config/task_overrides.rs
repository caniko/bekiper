use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::experiment::ExperimentConfig;
use super::pipeline::PoseLabelModel;
use bekiper_behaviour::object_recognition::NortWindowMode;
use bekiper_feature::heuristic::combined::exploration::BoutFilterMode;

/// Optional overrides for behavioural task parameters.
/// Any field set here overrides the `TaskParams` defaults.
///
/// ```toml
/// [scan.task_params]
/// freezing_displacement_threshold = 0.003
/// min_dwell_seconds = 0.5
/// peripheral_band_fraction = 0.25
/// centroid_coords = { x = "body_x", y = "body_y" }
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskParamsOverrides {
    pub freezing_window_seconds: Option<f64>,
    pub freezing_displacement_threshold: Option<f64>,
    pub min_dwell_seconds: Option<f64>,
    /// OFT seconds from first valid centroid; zero selects historical full-video scoring.
    pub oft_window_seconds: Option<f64>,
    pub peripheral_band_fraction: Option<f64>,
    pub centroid_coords: Option<bekiper_core::types::CoordinateColumns>,
    pub nose_coords: Option<bekiper_core::types::CoordinateColumns>,
    pub head_coords: Option<bekiper_core::types::CoordinateColumns>,
    pub tail_coords: Option<bekiper_core::types::CoordinateColumns>,

    /// Object recognition (NORT) specific overrides.
    #[serde(default)]
    pub exploration: Option<ExplorationOverrides>,

    /// Y-maze / radial arm maze specific overrides.
    #[serde(default)]
    pub ymaze: Option<YMazeOverrides>,

    /// Rearing heuristic overrides (OFT).
    #[serde(default)]
    pub rearing: Option<RearingOverrides>,

    /// Grooming heuristic overrides (OFT).
    #[serde(default)]
    pub grooming: Option<GroomingOverrides>,
}

/// Named parameter variant for scan-style task expansion.
///
/// The flattened `task_params` keeps each `[[scan.variants]]` table shaped like
/// an ordinary task-params table plus a required `name` slug. When maps such as
/// `exploration.merge_rules` are merged, a variant-level map replaces the
/// inherited map wholesale rather than extending it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskVariant {
    pub name: String,
    #[serde(flatten)]
    pub task_params: TaskParamsOverrides,
}

include!("task_overrides/variants.rs");

include!("task_overrides/application.rs");

include!("task_overrides/merging.rs");

include!("task_overrides/behaviour_adapter.rs");

include!("task_overrides/geometry_tests.rs");
