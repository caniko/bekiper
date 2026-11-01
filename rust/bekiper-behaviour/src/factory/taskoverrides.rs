use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::PerimeterSpec;
use bekiper_core::types::{ObjectRecognitionProtocol, TaskType};
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::object_shape::ObjectShape;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_perimeter::polygon::Polygon;
use bekiper_perimeter::radial_maze::RadialMaze;
use bekiper_perimeter::rectangle::Rectangle;
use bekiper_perimeter::triangle::Triangle;

use crate::BehaviouralTask;
use crate::object_recognition::{NortWindowMode, ObjectRecognition};
use crate::open_field::OpenFieldTest;
use crate::params::TaskParams;
use crate::radial_arm::YMaze;
use crate::reward_tracing::Cheeseboard;

/// Opaque trait for task-specific parameter overrides, implemented by
/// `TaskParamsOverrides` in the ingress crate.
///
/// This avoids a circular dependency: the behaviour crate doesn't depend
/// on ingress, but can still receive override values through this trait.
pub trait TaskOverrides {
    // -- Object Recognition (NORT) --
    fn exploration_nose_max_distance(&self) -> Option<f64> {
        None
    }
    fn exploration_contact_distance(&self) -> Option<f64> {
        None
    }
    fn exploration_max_angle_deg(&self) -> Option<f64> {
        None
    }
    fn exploration_max_nose_speed(&self) -> Option<f64> {
        None
    }
    fn exploration_speed_filter_mode(
        &self,
    ) -> Option<bekiper_feature::heuristic::combined::exploration::BoutFilterMode> {
        None
    }
    fn exploration_angle_filter_mode(
        &self,
    ) -> Option<bekiper_feature::heuristic::combined::exploration::BoutFilterMode> {
        None
    }
    fn exploration_min_bout_seconds(&self) -> Option<f64> {
        None
    }
    fn exploration_max_gap_seconds(&self) -> Option<f64> {
        None
    }
    fn exploration_strict_boundaries(&self) -> Option<bool> {
        None
    }
    fn exploration_min_total_exploration(&self) -> Option<f64> {
        None
    }
    fn exploration_analysis_window_seconds(&self) -> Option<f64> {
        None
    }
    fn exploration_window_mode(&self) -> Option<NortWindowMode> {
        None
    }
    fn exploration_perimeter_erosion_meters(&self) -> Option<f64> {
        None
    }
    fn exploration_exclude_posture(&self) -> Option<bool> {
        None
    }
    fn exploration_gap_density_threshold(&self) -> Option<f64> {
        None
    }
    /// Whisker-fan block; `age_table` lengths must already be resolved to meters.
    fn exploration_whisker(
        &self,
    ) -> Option<bekiper_feature::heuristic::solo::whisker_config::WhiskerOverrides> {
        None
    }

    // -- Qualia decomposition --
    fn qualia_enable(&self) -> Option<bool> {
        None
    }
    fn qualia_body_proximity_distance(&self) -> Option<f64> {
        None
    }
    fn qualia_whisker_distance(&self) -> Option<f64> {
        None
    }
    fn qualia_whisker_angle(&self) -> Option<f64> {
        None
    }
    fn qualia_olfaction_distance(&self) -> Option<f64> {
        None
    }
    fn qualia_olfaction_angle(&self) -> Option<f64> {
        None
    }
    fn qualia_merge_rules(
        &self,
    ) -> Option<
        std::collections::HashMap<String, bekiper_feature::heuristic::combined::merge::MergeRule>,
    > {
        None
    }

    // -- Y-maze --
    fn ymaze_temporal_policy(&self) -> Option<bekiper_feature::tolerance::TemporalPolicy> {
        None
    }
    fn ymaze_threshold_geometry(&self) -> Option<crate::radial_arm::YMazeThresholdGeometry> {
        None
    }
    fn ymaze_proximal_threshold(&self) -> Option<f64> {
        None
    }
    fn ymaze_principal_thresholds(&self) -> Option<Vec<f64>> {
        None
    }
    fn ymaze_start_mode(&self) -> Option<crate::radial_arm::YMazeStartMode> {
        None
    }
    fn ymaze_apply_dwell_filter(&self) -> Option<bool> {
        None
    }
    fn ymaze_arm_membership_hysteresis(&self) -> Option<bool> {
        None
    }
    fn ymaze_zone_max_gap_seconds(&self) -> Option<f64> {
        None
    }
    fn ymaze_pose_gap_fill_seconds(&self) -> Option<f64> {
        None
    }
    fn ymaze_zone_occupancy(&self) -> Option<bekiper_feature::zone::ZoneOccupancy> {
        None
    }

    // -- Rearing --
    fn rearing_max_speed(&self) -> Option<f64> {
        None
    }
    fn rearing_max_body_length_ratio(&self) -> Option<f64> {
        None
    }
    fn rearing_max_body_length(&self) -> Option<f64> {
        None
    }
    fn rearing_forepaw_proximity(&self) -> Option<f64> {
        None
    }

    // -- Grooming --
    fn grooming_max_speed(&self) -> Option<f64> {
        None
    }
    fn grooming_max_body_length(&self) -> Option<f64> {
        None
    }
    fn grooming_min_body_length(&self) -> Option<f64> {
        None
    }
    fn grooming_forepaw_proximity(&self) -> Option<f64> {
        None
    }
}

/// Build a `BehaviouralTask` from a task type, perimeters, and shared params.
///
/// Centralises all task-instantiation logic in the behaviour crate so
/// that the orchestration layer (ingress/workflow) doesn't need to know
/// the internals of each task.
pub fn build_task(
    task_type: TaskType,
    perimeters: &[PerimeterSpec],
    params: TaskParams,
    object_protocol: Option<ObjectRecognitionProtocol>,
    trial_start_perimeter: Option<&str>,
    overrides: Option<&dyn TaskOverrides>,
) -> Result<Box<dyn BehaviouralTask>> {
    match task_type {
        TaskType::ObjectRecognition => {
            build_object_recognition(perimeters, params, object_protocol, overrides)
        }
        TaskType::YMaze => build_ymaze(perimeters, params, overrides),
        TaskType::OpenField => build_open_field(perimeters, params, overrides),
        TaskType::Cheeseboard => build_cheeseboard(perimeters, params, trial_start_perimeter)
            .map(|task| Box::new(task) as Box<dyn BehaviouralTask>),
    }
}

include!("taskoverrides/arena_geometry.rs");

include!("taskoverrides/object_recognition.rs");

include!("taskoverrides/ymaze.rs");

include!("taskoverrides/open_field.rs");

include!("taskoverrides/nort_tests.rs");

include!("taskoverrides/ymaze_tests.rs");

include!("taskoverrides/oft_tests.rs");
