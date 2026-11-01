use indexmap::IndexMap;
use serde_json::Value;

use super::cell::{CellValue, pluck, pluck_trial_metadata};
use super::experiment::{Experiment, ExperimentId, SheetKey, dynamic_columns_from_row};
use super::trial::TrialResult;

/// Column schema for [`bekiper_core::types::TaskType::Cheeseboard`].
///
/// Cheeseboard now has a typed row extractor rather than the generic JSON
/// flattener. The aggregate workbook keeps the legacy reward-tracing fields
/// in explicit columns so callers do not need to depend on nested TOML
/// shapes or dot-pathed keys.
pub struct CheeseboardExperiment;

const CHEESEBOARD_STATIC_COLUMNS: &[&str] = &[
    // Animal metadata
    "animal",
    "cohort",
    "gene",
    "sex",
    "age",
    // Trial identifiers
    "fps",
    // Enclosure summary
    "enclosure_label",
    "enclosure_entries",
    "enclosure_duration_s",
    "enclosure_distance_m",
    "enclosure_mean_speed_mps",
    "enclosure_latency_s",
    // Reward-trace metadata
    "reward_labels",
    "reward_count",
    "start_area_exit_frame",
    "start_area_exit_seconds",
    "reward_arrival_frame",
    "reward_arrival_seconds",
    "seconds_to_find_reward",
    "seconds_in_reward_area",
    "reward_entries",
    "reward_trace_seconds",
    "reward_area_seconds",
    "start_to_reward_displacement_meters",
    // Search-path metrics
    "search_path_distance_meters",
    "search_direct_distance_meters",
    "search_mean_speed_mps",
    "search_tortuosity",
    // Trial totals
    "total_distance_meters",
    "total_mean_speed_mps",
    "total_duration_seconds",
    // Trailing meta
    "experiment",
    "excluded",
];

fn reward_labels_value(detailed: &Value) -> CellValue {
    let Some(labels) = detailed.get("reward_labels").and_then(Value::as_array) else {
        return CellValue::Null;
    };
    let labels: Vec<&str> = labels.iter().filter_map(Value::as_str).collect();
    if labels.is_empty() {
        CellValue::Null
    } else {
        CellValue::String(labels.join("|"))
    }
}

const CHEESEBOARD_ANIMAL_OVERVIEW_COLUMNS: &[&str] = &[
    "total_distance_meters",
    "total_duration_seconds",
    "seconds_to_find_reward",
    "reward_entries",
    "search_path_distance_meters",
];

impl Experiment for CheeseboardExperiment {
    fn id(&self) -> ExperimentId {
        ExperimentId("cheeseboard")
    }

    fn sheet_key(&self, _trial: &TrialResult) -> Option<SheetKey> {
        Some(SheetKey {
            experiment: self.id(),
            variant: None,
        })
    }

    fn static_columns(&self) -> Vec<String> {
        CHEESEBOARD_STATIC_COLUMNS
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn dynamic_columns(&self, trial: &TrialResult) -> Vec<String> {
        dynamic_columns_from_row(CHEESEBOARD_STATIC_COLUMNS, &self.extract_row(trial))
    }

    fn extract_row(&self, trial: &TrialResult) -> IndexMap<String, CellValue> {
        let mut row: IndexMap<String, CellValue> =
            IndexMap::with_capacity(CHEESEBOARD_STATIC_COLUMNS.len());

        // -- Animal metadata -------------------------------------------------
        row.insert("animal".into(), pluck_trial_metadata(trial, "animal"));
        row.insert("cohort".into(), pluck_trial_metadata(trial, "cohort"));
        row.insert("gene".into(), pluck_trial_metadata(trial, "gene"));
        row.insert("sex".into(), pluck_trial_metadata(trial, "sex"));
        row.insert("age".into(), pluck_trial_metadata(trial, "age"));

        // -- Trial-level scalars --------------------------------------------
        row.insert("fps".into(), CellValue::Float(trial.fps));

        // -- Enclosure summary ----------------------------------------------
        let d = &trial.detailed;
        row.insert("enclosure_label".into(), pluck(d, "enclosure_label"));
        let enclosure = d.get("enclosure_summary").unwrap_or(&Value::Null);
        row.insert("enclosure_entries".into(), pluck(enclosure, "entries"));
        row.insert(
            "enclosure_duration_s".into(),
            pluck(enclosure, "duration_seconds"),
        );
        row.insert(
            "enclosure_distance_m".into(),
            pluck(enclosure, "distance_meters"),
        );
        row.insert(
            "enclosure_mean_speed_mps".into(),
            pluck(enclosure, "mean_speed"),
        );
        row.insert(
            "enclosure_latency_s".into(),
            pluck(enclosure, "latency_seconds"),
        );

        // -- Reward-trace metadata ------------------------------------------
        row.insert("reward_labels".into(), reward_labels_value(d));
        let reward_count = d
            .get("reward_labels")
            .and_then(Value::as_array)
            .map(|labels| labels.len() as i64)
            .unwrap_or(0);
        row.insert("reward_count".into(), CellValue::Int(reward_count));
        row.insert(
            "start_area_exit_frame".into(),
            pluck(d, "start_area_exit_frame"),
        );
        row.insert(
            "start_area_exit_seconds".into(),
            pluck(d, "start_area_exit_seconds"),
        );
        row.insert(
            "reward_arrival_frame".into(),
            pluck(d, "reward_arrival_frame"),
        );
        row.insert(
            "reward_arrival_seconds".into(),
            pluck(d, "reward_arrival_seconds"),
        );
        row.insert(
            "seconds_to_find_reward".into(),
            pluck(d, "seconds_to_find_reward"),
        );
        row.insert(
            "seconds_in_reward_area".into(),
            pluck(d, "seconds_in_reward_area"),
        );
        row.insert("reward_entries".into(), pluck(d, "reward_entries"));
        row.insert(
            "reward_trace_seconds".into(),
            pluck(d, "seconds_to_find_reward"),
        );
        row.insert(
            "reward_area_seconds".into(),
            pluck(d, "seconds_in_reward_area"),
        );
        row.insert(
            "start_to_reward_displacement_meters".into(),
            pluck(d, "start_to_reward_displacement_meters"),
        );

        // -- Search path metrics --------------------------------------------
        row.insert(
            "search_path_distance_meters".into(),
            pluck(d, "search_path_distance_meters"),
        );
        row.insert(
            "search_direct_distance_meters".into(),
            pluck(d, "search_direct_distance_meters"),
        );
        row.insert(
            "search_mean_speed_mps".into(),
            pluck(d, "search_mean_speed_mps"),
        );
        row.insert("search_tortuosity".into(), pluck(d, "search_tortuosity"));

        // -- Trial totals ---------------------------------------------------
        row.insert(
            "total_distance_meters".into(),
            pluck(d, "total_distance_meters"),
        );
        row.insert(
            "total_mean_speed_mps".into(),
            pluck(d, "total_mean_speed_mps"),
        );
        row.insert(
            "total_duration_seconds".into(),
            pluck(d, "total_duration_seconds"),
        );

        // -- Trailing identifiers -------------------------------------------
        row.insert(
            "experiment".into(),
            CellValue::String(trial.experiment.clone()),
        );
        row.insert("excluded".into(), pluck(d, "excluded"));

        row
    }

    fn animal_overview_columns(&self) -> &'static [&'static str] {
        CHEESEBOARD_ANIMAL_OVERVIEW_COLUMNS
    }
}
