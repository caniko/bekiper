use indexmap::IndexMap;

use super::cell::{CellValue, pluck, pluck_trial_metadata};
use super::experiment::{
    Experiment, ExperimentId, SheetKey, dynamic_columns_from_row, legacy_routing_key_for_trial,
    sheet_key_from_routing_key,
};
use super::trial::TrialResult;
use crate::export::for_each_json_leaf;

/// Generic fallback: emits the canonical metadata prefix, then recursively
/// flattens `trial.detailed` into dot-pathed columns using
/// `for_each_json_leaf`, then appends the trailing `experiment` / `excluded`
/// columns.
///
/// Intended for tasks that don't have a bespoke extractor yet (Cheeseboard,
/// future experimental protocols). Column set is determined per-trial by the
/// actual shape of `detailed` — downstream `build_sheet_dataframes` unions
/// keys across rows when building the final `DataFrame`, so heterogeneous
/// rows are safe.
pub struct GenericExperiment;

const GENERIC_STATIC_COLUMNS: &[&str] = &[
    "animal",
    "cohort",
    "gene",
    "sex",
    "age",
    "fps",
    "experiment",
    "excluded",
];

impl Experiment for GenericExperiment {
    fn id(&self) -> ExperimentId {
        ExperimentId("generic")
    }

    fn sheet_key(&self, trial: &TrialResult) -> Option<SheetKey> {
        legacy_routing_key_for_trial(trial)
            .map(|key| sheet_key_from_routing_key(&key))
            .or({
                Some(SheetKey {
                    experiment: ExperimentId(trial.task_type.as_str()),
                    variant: None,
                })
            })
    }

    fn static_columns(&self) -> Vec<String> {
        GENERIC_STATIC_COLUMNS
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn dynamic_columns(&self, trial: &TrialResult) -> Vec<String> {
        dynamic_columns_from_row(GENERIC_STATIC_COLUMNS, &self.extract_row(trial))
    }

    fn extract_row(&self, trial: &TrialResult) -> IndexMap<String, CellValue> {
        let mut row: IndexMap<String, CellValue> = IndexMap::new();

        // -- Animal metadata -------------------------------------------------
        row.insert("animal".into(), pluck_trial_metadata(trial, "animal"));
        row.insert("cohort".into(), pluck_trial_metadata(trial, "cohort"));
        row.insert("gene".into(), pluck_trial_metadata(trial, "gene"));
        row.insert("sex".into(), pluck_trial_metadata(trial, "sex"));
        row.insert("age".into(), pluck_trial_metadata(trial, "age"));

        // -- Trial-level scalars --------------------------------------------
        row.insert("fps".into(), CellValue::Float(trial.fps));

        // -- Flattened detailed block ---------------------------------------
        // Empty prefix matches the behaviour of `write_detailed_to_sheet`,
        // so the aggregate row key set mirrors the per-trial detailed
        // spreadsheet sheet exactly.
        //
        // If `detailed` happens to contain a top-level key that collides with
        // a metadata column (`animal`, `fps`, etc.), the flatten pass
        // overwrites the metadata value via `IndexMap::insert`. Real
        // `bekiper-behaviour` outputs never emit those keys in `detailed`, so
        // the collision is theoretical.
        for_each_json_leaf("", &trial.detailed, &mut |key, value| {
            row.insert(key.to_string(), CellValue::from_json(value));
        });

        // -- Trailing identifiers -------------------------------------------
        row.insert(
            "experiment".into(),
            CellValue::String(trial.experiment.clone()),
        );
        row.insert("excluded".into(), pluck(&trial.detailed, "excluded"));

        row
    }
}
