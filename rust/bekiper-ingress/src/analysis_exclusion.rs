use std::path::Component;

use bekiper_core::types::TaskType;
use serde_json::Value;

use crate::config::ExperimentConfig;

pub(crate) const NORT_HABITUATION_ROLE: &str = "nort_habituation";

pub(crate) fn is_analysis_excluded_experiment(experiment: &ExperimentConfig) -> bool {
    experiment.task_type == TaskType::ObjectRecognition
        && crate::config::profile::is_imen_habituation_context(experiment.session_context.as_ref())
}

pub(crate) fn is_analysis_excluded_result(
    _experiment: &str,
    task_type: TaskType,
    session_context: &Value,
    known_excluded_experiment: bool,
) -> bool {
    known_excluded_experiment
        || (task_type == TaskType::ObjectRecognition
            && is_imen_habituation_context_value(session_context))
}

pub(crate) fn aggregate_role_for_experiment(experiment: &ExperimentConfig) -> Option<&'static str> {
    is_imen_oft_habituation_experiment(experiment).then_some(NORT_HABITUATION_ROLE)
}

pub(crate) fn aggregate_role_for_result(
    experiment: &str,
    task_type: TaskType,
) -> Option<&'static str> {
    is_imen_oft_habituation_name(experiment, task_type).then_some(NORT_HABITUATION_ROLE)
}

pub(crate) fn is_imen_oft_habituation_experiment(experiment: &ExperimentConfig) -> bool {
    experiment.task_type == TaskType::OpenField
        && (is_imen_oft_habituation_name(&experiment.name, experiment.task_type)
            || experiment.video.as_ref().is_some_and(|path| {
                path.components().any(|component| {
                    matches!(component, Component::Normal(name) if name == "OFT videos")
                })
            }))
}

fn is_imen_oft_habituation_name(name: &str, task_type: TaskType) -> bool {
    task_type == TaskType::OpenField && name.starts_with("oft_videos_")
}

fn is_imen_habituation_context_value(context: &Value) -> bool {
    let Some(object) = context.as_object() else {
        return false;
    };
    object.get("profile").and_then(Value::as_str) == Some("videomap")
        && object.get("stage").and_then(Value::as_str) == Some("0")
        && matches!(object.get("phase").and_then(Value::as_str), Some("A" | "B"))
}
