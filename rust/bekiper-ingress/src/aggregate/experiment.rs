use bekiper_core::types::TaskType;
use indexmap::IndexMap;

use super::cell::CellValue;
use super::cheeseboard::CheeseboardExperiment;
use super::nort::NortExperiment;
use super::oft::OftExperiment;
use super::trial::{TrialResult, parse_recipe_from_experiment};
use super::ymaze::YMazeExperiment;
use crate::analysis_exclusion::aggregate_role_for_result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExperimentId(pub &'static str);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantTag(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SheetKey {
    pub experiment: ExperimentId,
    pub variant: Option<VariantTag>,
}

impl SheetKey {
    /// Render to the canonical aggregate routing key, e.g. `ymaze` or
    /// `ymaze__proximal65`. This is what `recipe_to_sheet` is keyed on.
    pub fn routing_key(&self) -> String {
        if self.experiment.0 == "nort"
            && let Some(VariantTag(variant)) = &self.variant
        {
            return variant.clone();
        }
        match &self.variant {
            Some(VariantTag(variant)) => format!("{}__{variant}", self.experiment.0),
            None => self.experiment.0.to_string(),
        }
    }
}

pub trait Experiment: Send + Sync {
    fn id(&self) -> ExperimentId;

    fn sheet_key(&self, trial: &TrialResult) -> Option<SheetKey>;

    fn static_columns(&self) -> Vec<String>;

    fn static_columns_for_trials(&self, _trials: &[&TrialResult]) -> Vec<String> {
        self.static_columns()
    }

    fn dynamic_columns(&self, trial: &TrialResult) -> Vec<String>;

    fn extract_row(&self, trial: &TrialResult) -> IndexMap<String, CellValue>;

    /// Columns the cross-experiment animal-overview sheet should surface
    /// from this experiment's per-trial rows. Default is empty so an
    /// experiment can opt out.
    fn animal_overview_columns(&self) -> &'static [&'static str] {
        &[]
    }
}

pub fn experiment_for_trial(trial: &TrialResult) -> Box<dyn Experiment> {
    match trial.task_type {
        TaskType::ObjectRecognition => Box::new(NortExperiment::from_trial(trial)),
        TaskType::OpenField => Box::new(OftExperiment),
        TaskType::YMaze => Box::new(YMazeExperiment),
        TaskType::Cheeseboard => Box::new(CheeseboardExperiment),
    }
}

pub(crate) fn dynamic_columns_from_row(
    static_columns: &[&str],
    row: &IndexMap<String, CellValue>,
) -> Vec<String> {
    row.keys()
        .filter(|column| {
            !static_columns
                .iter()
                .any(|static_column| static_column == *column)
        })
        .cloned()
        .collect()
}

pub(crate) fn legacy_routing_key_for_trial(trial: &TrialResult) -> Option<String> {
    if let Some(role) = trial
        .analysis_role
        .as_deref()
        .map(str::trim)
        .filter(|role| !role.is_empty())
    {
        return Some(role.to_string());
    }

    if let Some(recipe) = parse_recipe_from_experiment(&trial.experiment) {
        return Some(recipe);
    }

    if let Some(role) = aggregate_role_for_result(&trial.experiment, trial.task_type) {
        return Some(role.to_string());
    }

    None
}

pub(crate) fn sheet_key_from_routing_key(routing_key: &str) -> SheetKey {
    match routing_key.split_once("__") {
        Some((experiment, variant)) => SheetKey {
            experiment: experiment_id_from_legacy_key(experiment),
            variant: Some(VariantTag(variant.to_string())),
        },
        None => SheetKey {
            experiment: experiment_id_from_legacy_key(routing_key),
            variant: None,
        },
    }
}

fn experiment_id_from_legacy_key(key: &str) -> ExperimentId {
    match key {
        "cheeseboard" => ExperimentId("cheeseboard"),
        "nort_novelty" => ExperimentId("nort_novelty"),
        "nort_training" => ExperimentId("nort_training"),
        "oft" => ExperimentId("oft"),
        "open_field" => ExperimentId("open_field"),
        "radial_arm" => ExperimentId("radial_arm"),
        "y_maze" => ExperimentId("y_maze"),
        "ymaze" => ExperimentId("ymaze"),
        other => ExperimentId(Box::leak(other.to_string().into_boxed_str())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sheet_key_routing_key_renders_bare_experiment() {
        let key = SheetKey {
            experiment: ExperimentId("ymaze"),
            variant: None,
        };

        assert_eq!(key.routing_key(), "ymaze");
    }

    #[test]
    fn sheet_key_routing_key_renders_variant_suffix() {
        let key = SheetKey {
            experiment: ExperimentId("ymaze"),
            variant: Some(VariantTag("proximal65".to_string())),
        };

        assert_eq!(key.routing_key(), "ymaze__proximal65");
    }

    #[test]
    fn sheet_key_routing_key_preserves_underscore_bearing_experiment_id() {
        let key = SheetKey {
            experiment: ExperimentId("nort_training"),
            variant: Some(VariantTag("day2".to_string())),
        };

        assert_eq!(key.routing_key(), "nort_training__day2");
    }
}
