use indexmap::IndexMap;
use serde_json::Value;

use super::cell::{CellValue, pluck, pluck_trial_metadata};
use super::experiment::{
    Experiment, ExperimentId, SheetKey, dynamic_columns_from_row, legacy_routing_key_for_trial,
    sheet_key_from_routing_key,
};
use super::trial::TrialResult;

/// Column schema for [`bekiper_core::types::TaskType::OpenField`].
///
/// Mirrors the public fields of `OftResult`
/// ([rust/bekiper-behaviour/src/open_field.rs](../../bekiper-behaviour/src/open_field.rs))
/// with the canonical short-form naming (`*_s` / `*_m`). Zone-derived columns (`center_*`, `periphery_*`)
/// are plucked from the per-zone `ZoneSummary` sub-objects under
/// `detailed.center_summary` / `detailed.periphery_summary`.
pub struct OftExperiment;

const OFT_STATIC_COLUMNS: &[&str] = &[
    // Animal metadata
    "animal",
    "cohort",
    "gene",
    "sex",
    "age",
    // Trial identifiers
    "fps",
    // Motion / duration
    "total_duration_s",
    "analysis_duration_s",
    "window_start_s",
    "window_end_s",
    "window_complete_flag",
    "total_distance_m",
    "total_mean_speed",
    // Freezing / immobility / posture
    "freezing_s",
    "immobility_s",
    "immobility_pct",
    "rearing_s",
    "grooming_s",
    "fecal_boli_count",
    // Center zone
    "center_duration_s",
    "center_entries",
    "latency_to_center_s",
    "center_distance_m",
    "center_mean_speed",
    "center_time_pct",
    "center_motion_distance_m",
    "center_motion_median_speed",
    "center_motion_median_acceleration",
    "center_motion_freezing_s",
    // Periphery zone
    "periphery_duration_s",
    "periphery_entries",
    "periphery_distance_m",
    "periphery_mean_speed",
    "periphery_time_pct",
    "periphery_motion_distance_m",
    "periphery_motion_median_speed",
    "periphery_motion_median_acceleration",
    "periphery_motion_freezing_s",
    // Trailing meta
    "experiment",
    "excluded",
];

const OFT_ANIMAL_OVERVIEW_COLUMNS: &[&str] = &[
    "total_distance_m",
    "total_duration_s",
    "center_time_pct",
    "periphery_time_pct",
    "freezing_s",
    "rearing_s",
    "grooming_s",
    "fecal_boli_count",
];

impl Experiment for OftExperiment {
    fn id(&self) -> ExperimentId {
        ExperimentId("oft")
    }

    fn sheet_key(&self, trial: &TrialResult) -> Option<SheetKey> {
        legacy_routing_key_for_trial(trial)
            .map(|key| sheet_key_from_routing_key(&key))
            .or({
                Some(SheetKey {
                    experiment: ExperimentId("open_field"),
                    variant: None,
                })
            })
    }

    fn static_columns(&self) -> Vec<String> {
        OFT_STATIC_COLUMNS.iter().map(|s| s.to_string()).collect()
    }

    fn dynamic_columns(&self, trial: &TrialResult) -> Vec<String> {
        dynamic_columns_from_row(OFT_STATIC_COLUMNS, &self.extract_row(trial))
    }

    fn extract_row(&self, trial: &TrialResult) -> IndexMap<String, CellValue> {
        let mut row: IndexMap<String, CellValue> =
            IndexMap::with_capacity(OFT_STATIC_COLUMNS.len());

        // -- Animal metadata -------------------------------------------------
        row.insert("animal".into(), pluck_trial_metadata(trial, "animal"));
        row.insert("cohort".into(), pluck_trial_metadata(trial, "cohort"));
        row.insert("gene".into(), pluck_trial_metadata(trial, "gene"));
        row.insert("sex".into(), pluck_trial_metadata(trial, "sex"));
        row.insert("age".into(), pluck_trial_metadata(trial, "age"));

        // -- Trial-level scalars --------------------------------------------
        row.insert("fps".into(), CellValue::Float(trial.fps));

        // -- Motion / duration (remap long -> short field names) ------------
        let d = &trial.detailed;
        row.insert(
            "total_duration_s".into(),
            pluck(d, "total_duration_seconds"),
        );
        row.insert("total_distance_m".into(), pluck(d, "total_distance_meters"));
        row.insert("total_mean_speed".into(), pluck(d, "total_mean_speed"));
        row.insert(
            "analysis_duration_s".into(),
            pluck(d, "total_duration_seconds"),
        );
        let window = d.get("window").unwrap_or(&Value::Null);
        row.insert("window_start_s".into(), pluck(window, "start_seconds"));
        row.insert("window_end_s".into(), pluck(window, "end_seconds"));
        row.insert("window_complete_flag".into(), pluck(window, "complete"));

        // -- Freezing / immobility / posture --------------------------------
        row.insert("freezing_s".into(), pluck(d, "freezing_seconds"));
        row.insert("immobility_s".into(), pluck(d, "immobility_seconds"));
        row.insert("immobility_pct".into(), pluck(d, "immobility_pct"));
        row.insert("rearing_s".into(), pluck(d, "rearing_seconds"));
        row.insert("grooming_s".into(), pluck(d, "grooming_seconds"));
        row.insert(
            "fecal_boli_count".into(),
            trial
                .session_context
                .get("fecal_boli_count")
                .map(CellValue::from_json)
                .unwrap_or(CellValue::Null),
        );

        // -- Zone summaries (center / periphery) ----------------------------
        let null = Value::Null;
        let center = d.get("center_summary").unwrap_or(&null);
        row.insert(
            "center_duration_s".into(),
            pluck(center, "duration_seconds"),
        );
        row.insert("center_entries".into(), pluck(center, "entries"));
        row.insert(
            "latency_to_center_s".into(),
            pluck(center, "latency_seconds"),
        );
        row.insert("center_distance_m".into(), pluck(center, "distance_meters"));
        row.insert("center_mean_speed".into(), pluck(center, "mean_speed"));
        row.insert("center_time_pct".into(), pluck(d, "center_time_pct"));
        let center_motion = d.get("center_motion_summary").unwrap_or(&null);
        row.insert(
            "center_motion_distance_m".into(),
            pluck(center_motion, "total_displacement_meters"),
        );
        row.insert(
            "center_motion_median_speed".into(),
            pluck(center_motion, "median_speed_mps"),
        );
        row.insert(
            "center_motion_median_acceleration".into(),
            pluck(center_motion, "median_acceleration_mps2"),
        );
        row.insert(
            "center_motion_freezing_s".into(),
            pluck(center_motion, "freezing_seconds"),
        );

        let periphery = d.get("periphery_summary").unwrap_or(&null);
        row.insert(
            "periphery_duration_s".into(),
            pluck(periphery, "duration_seconds"),
        );
        row.insert("periphery_entries".into(), pluck(periphery, "entries"));
        row.insert(
            "periphery_distance_m".into(),
            pluck(periphery, "distance_meters"),
        );
        row.insert(
            "periphery_mean_speed".into(),
            pluck(periphery, "mean_speed"),
        );
        row.insert("periphery_time_pct".into(), pluck(d, "periphery_time_pct"));
        let periphery_motion = d.get("periphery_motion_summary").unwrap_or(&null);
        row.insert(
            "periphery_motion_distance_m".into(),
            pluck(periphery_motion, "total_displacement_meters"),
        );
        row.insert(
            "periphery_motion_median_speed".into(),
            pluck(periphery_motion, "median_speed_mps"),
        );
        row.insert(
            "periphery_motion_median_acceleration".into(),
            pluck(periphery_motion, "median_acceleration_mps2"),
        );
        row.insert(
            "periphery_motion_freezing_s".into(),
            pluck(periphery_motion, "freezing_seconds"),
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
        OFT_ANIMAL_OVERVIEW_COLUMNS
    }
}
