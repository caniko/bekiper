use bekiper_behaviour::object_recognition::{
    NortResult, NortWindowMode, ObjectRecognitionZoneMetrics, ObservationTotals,
    PairwiseObservationMetric, QualiaDiagnosticRollup,
};
use bekiper_core::types::ObjectRecognitionProtocol;
use indexmap::IndexMap;

use super::cell::{CellValue, pluck_trial_metadata};
use super::experiment::{
    Experiment, ExperimentId, SheetKey, VariantTag, legacy_routing_key_for_trial,
};
use super::trial::TrialResult;

const COMMON_COLUMNS: &[&str] = &[
    "animal",
    "cohort",
    "gene",
    "sex",
    "age",
    "fps",
    "protocol",
    "total_duration_s",
    "duration_within_expected",
    "total_distance_m",
    "total_mean_speed",
    "freezing_s",
    "latency_to_center",
    "median_speed_observing",
    "median_speed_not_observing",
    "first_approach_frame",
    "first_object_approach_time_s",
    "analysis_window_s",
    "window_mode",
    "window_start_frame",
    "window_end_frame_exclusive",
    "side_bias",
    "object_bias",
    "object_A_label",
    "object_B_label",
    "object_A_placement",
    "object_B_placement",
    "object_A_exploration_s",
    "object_B_exploration_s",
    "object_A_bouts",
    "object_B_bouts",
    "experiment",
    "excluded",
];

const NOVELTY_COLUMNS: &[&str] = &[
    "DI",
    "RI",
    "pct_novel",
    "log_ratio",
    "novel_s",
    "familiar_s",
    "total_exploration_s",
    "passes_15s_exploration_threshold",
    "novel_bouts",
    "familiar_bouts",
];

const PAIRWISE_COLUMNS: &[&str] = &[
    "total_observation_s",
    "passes_15s_exploration_threshold",
    "total_observation_bouts",
    "all_motion_distance_m",
    "all_motion_median_speed",
    "all_motion_median_acceleration",
    "all_motion_freezing_s",
    "center_duration_s",
    "center_entries",
    "center_distance_m",
    "center_mean_speed",
    "center_motion_distance_m",
    "center_motion_median_speed",
    "center_motion_median_acceleration",
    "center_motion_freezing_s",
    "periphery_duration_s",
    "periphery_entries",
    "periphery_distance_m",
    "periphery_mean_speed",
    "periphery_motion_distance_m",
    "periphery_motion_median_speed",
    "periphery_motion_median_acceleration",
    "periphery_motion_freezing_s",
];

trait ObjectRecognitionSheetSpec: Send + Sync {
    fn columns(&self) -> Vec<String>;
    fn extract(&self, trial: &TrialResult, detail: &NortResult) -> IndexMap<String, CellValue>;
}

struct NoveltySheetSpec;
struct PairwiseSheetSpec;

pub struct NortExperiment {
    spec: Box<dyn ObjectRecognitionSheetSpec>,
}

const NORT_ANIMAL_OVERVIEW_COLUMNS: &[&str] = &[
    "pct_novel",
    "total_distance_m",
    "total_duration_s",
    "freezing_s",
    "novel_s",
    "familiar_s",
    "total_exploration_s",
    "object_bias",
];

impl NortExperiment {
    pub fn from_trial(trial: &TrialResult) -> Self {
        let protocol = detail_for_trial(trial)
            .map(|detail| detail.protocol)
            .unwrap_or_else(|| protocol_from_trial(trial));
        let spec: Box<dyn ObjectRecognitionSheetSpec> = if protocol.computes_novelty_indices() {
            Box::new(NoveltySheetSpec)
        } else {
            Box::new(PairwiseSheetSpec)
        };
        Self { spec }
    }
}

impl Experiment for NortExperiment {
    fn id(&self) -> ExperimentId {
        ExperimentId("nort")
    }

    fn sheet_key(&self, trial: &TrialResult) -> Option<SheetKey> {
        if let Some(key) = legacy_routing_key_for_trial(trial) {
            return Some(SheetKey {
                experiment: self.id(),
                variant: Some(VariantTag(key)),
            });
        }

        if let Some(protocol) = trial
            .detailed
            .get("protocol")
            .and_then(serde_json::Value::as_str)
        {
            return Some(SheetKey {
                experiment: self.id(),
                variant: Some(VariantTag(protocol.to_string())),
            });
        }

        if let Some(key) = object_recognition_key_from_imen_name(&trial.experiment) {
            return Some(SheetKey {
                experiment: self.id(),
                variant: Some(VariantTag(key)),
            });
        }

        None
    }

    fn static_columns(&self) -> Vec<String> {
        self.spec.columns()
    }

    fn dynamic_columns(&self, trial: &TrialResult) -> Vec<String> {
        let static_columns = self.static_columns();
        self.extract_row(trial)
            .keys()
            .filter(|column| {
                !static_columns
                    .iter()
                    .any(|static_column| static_column == *column)
            })
            .cloned()
            .collect()
    }

    fn extract_row(&self, trial: &TrialResult) -> IndexMap<String, CellValue> {
        let detail = detail_for_trial(trial).expect("object-recognition detail always parses");
        self.spec.extract(trial, &detail)
    }

    fn animal_overview_columns(&self) -> &'static [&'static str] {
        NORT_ANIMAL_OVERVIEW_COLUMNS
    }
}

impl ObjectRecognitionSheetSpec for NoveltySheetSpec {
    fn columns(&self) -> Vec<String> {
        COMMON_COLUMNS
            .iter()
            .chain(NOVELTY_COLUMNS.iter())
            .chain(
                PAIRWISE_COLUMNS
                    .iter()
                    .filter(|column| !NOVELTY_COLUMNS.contains(column)),
            )
            .map(|column| (*column).to_string())
            .collect()
    }

    fn extract(&self, trial: &TrialResult, detail: &NortResult) -> IndexMap<String, CellValue> {
        let mut row = common_row(trial, detail);
        let observation_totals = detail
            .observation_totals
            .clone()
            .or_else(|| derive_observation_totals(detail));

        if let Some(indices) = &detail.indices {
            let na = |value: Option<f64>| value.map(CellValue::Float).unwrap_or(CellValue::Null);
            row.insert("DI".into(), na(indices.discrimination_index));
            row.insert("RI".into(), na(indices.recognition_index));
            row.insert("pct_novel".into(), na(indices.pct_novel));
            row.insert(
                "log_ratio".into(),
                indices
                    .log_ratio
                    .map(CellValue::Float)
                    .unwrap_or(CellValue::Null),
            );
            row.insert("novel_s".into(), CellValue::Float(indices.novel_seconds));
            row.insert(
                "familiar_s".into(),
                CellValue::Float(indices.familiar_seconds),
            );
            row.insert(
                "total_exploration_s".into(),
                CellValue::Float(indices.total_seconds),
            );
            row.insert(
                "novel_bouts".into(),
                CellValue::Int(indices.novel_bouts as i64),
            );
            row.insert(
                "familiar_bouts".into(),
                CellValue::Int(indices.familiar_bouts as i64),
            );
        }

        insert_observation_totals(&mut row, observation_totals.as_ref());
        insert_zone_metrics(&mut row, detail.zone_metrics.as_ref());
        ensure_static_columns(&mut row, &self.columns());
        row
    }
}

impl ObjectRecognitionSheetSpec for PairwiseSheetSpec {
    fn columns(&self) -> Vec<String> {
        COMMON_COLUMNS
            .iter()
            .chain(PAIRWISE_COLUMNS.iter())
            .map(|column| (*column).to_string())
            .collect()
    }

    fn extract(&self, trial: &TrialResult, detail: &NortResult) -> IndexMap<String, CellValue> {
        let mut row = common_row(trial, detail);
        let observation_totals = detail
            .observation_totals
            .clone()
            .or_else(|| derive_observation_totals(detail));
        let pairwise_metrics = if detail.pairwise_observations.is_empty() {
            observation_totals
                .as_ref()
                .map(derive_pairwise_metrics)
                .unwrap_or_default()
        } else {
            detail.pairwise_observations.clone()
        };

        insert_observation_totals(&mut row, observation_totals.as_ref());
        insert_pairwise_metrics(&mut row, &pairwise_metrics);
        insert_qualia_rollups(&mut row, &detail.qualia_rollups);
        insert_zone_metrics(&mut row, detail.zone_metrics.as_ref());
        ensure_static_columns(&mut row, &self.columns());

        row
    }
}

fn detail_for_trial(trial: &TrialResult) -> Option<NortResult> {
    serde_json::from_value(trial.detailed.clone())
        .ok()
        .or_else(|| fallback_detail_for_trial(trial))
}

fn protocol_from_trial(trial: &TrialResult) -> ObjectRecognitionProtocol {
    legacy_routing_key_for_trial(trial)
        .as_deref()
        .and_then(protocol_from_recipe)
        .or_else(|| {
            trial
                .detailed
                .get("protocol")
                .and_then(serde_json::Value::as_str)
                .and_then(|value| value.parse().ok())
        })
        .or_else(|| {
            object_recognition_key_from_imen_name(&trial.experiment)
                .as_deref()
                .and_then(protocol_from_recipe)
        })
        .unwrap_or(ObjectRecognitionProtocol::NortNovelty)
}

fn protocol_from_recipe(recipe: &str) -> Option<ObjectRecognitionProtocol> {
    match recipe {
        "norttraining" | "nort_training" => Some(ObjectRecognitionProtocol::NortTraining),
        "nortnovelty" | "nort_novelty" | "norttest" | "nort_test" => {
            Some(ObjectRecognitionProtocol::NortNovelty)
        }
        "oultraining" | "oul_training" => Some(ObjectRecognitionProtocol::OulTraining),
        "oulupdate" | "oul_update" => Some(ObjectRecognitionProtocol::OulUpdate),
        "oultest" | "oul_test" => Some(ObjectRecognitionProtocol::OulTest),
        _ => None,
    }
}

fn object_recognition_key_from_imen_name(name: &str) -> Option<String> {
    // `<recording>__<variant>`: the protocol is the recording's.
    let name = name
        .split_once("__")
        .map_or(name, |(recording, _)| recording);
    if name.contains("_t1_") {
        Some(ObjectRecognitionProtocol::NortTraining.as_str().to_string())
    } else if name.contains("_t2_") {
        Some(ObjectRecognitionProtocol::NortNovelty.as_str().to_string())
    } else {
        None
    }
}

fn fallback_detail_for_trial(trial: &TrialResult) -> Option<NortResult> {
    let detailed = &trial.detailed;
    Some(NortResult {
        protocol: protocol_from_trial(trial),
        scoring_provenance: detailed
            .get("scoring_provenance")
            .filter(|value| !value.is_null())
            .cloned(),
        exploration_times: detailed
            .get("exploration_times")
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(serde_json::Value::as_f64)
                    .collect()
            })
            .unwrap_or_default(),
        bout_counts: detailed
            .get("bout_counts")
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|value| value as u32)
                    .collect()
            })
            .unwrap_or_default(),
        object_labels: detailed
            .get("object_labels")
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        indices: parse_indices(detailed.get("indices")),
        observation_totals: detailed
            .get("observation_totals")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok()),
        pairwise_observations: detailed
            .get("pairwise_observations")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default(),
        qualia_rollups: detailed
            .get("qualia_rollups")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default(),
        zone_metrics: detailed
            .get("zone_metrics")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok()),
        exploration_attributions: detailed
            .get("exploration_attributions")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default(),
        side_bias: detailed
            .get("side_bias")
            .and_then(serde_json::Value::as_f64),
        object_bias: detailed
            .get("object_bias")
            .and_then(serde_json::Value::as_f64),
        latency_to_center: detailed
            .get("latency_to_center")
            .and_then(serde_json::Value::as_f64),
        freezing_seconds: detailed
            .get("freezing_seconds")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        total_distance_meters: detailed
            .get("total_distance_meters")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        total_mean_speed: detailed
            .get("total_mean_speed")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        median_speed_observing: detailed
            .get("median_speed_observing")
            .and_then(serde_json::Value::as_f64),
        median_speed_not_observing: detailed
            .get("median_speed_not_observing")
            .and_then(serde_json::Value::as_f64),
        total_duration_seconds: detailed
            .get("total_duration_seconds")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        first_approach_frame: detailed
            .get("first_approach_frame")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize),
        first_object_approach_time_s: detailed
            .get("first_object_approach_time_s")
            .and_then(serde_json::Value::as_f64),
        analysis_window_seconds: detailed
            .get("analysis_window_seconds")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        window_mode: detailed
            .get("window_mode")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default(),
        window_start_frame: detailed
            .get("window_start_frame")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize),
        window_end_frame_exclusive: detailed
            .get("window_end_frame_exclusive")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize),
        excluded: detailed
            .get("excluded")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        passes_15s_exploration_threshold: detailed
            .get("passes_15s_exploration_threshold")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        qualia: detailed
            .get("qualia")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok()),
        evaluated_df: None,
    })
}

fn parse_indices(
    indices: Option<&serde_json::Value>,
) -> Option<bekiper_behaviour::nort::NortIndices> {
    let indices = indices?;
    serde_json::from_value(indices.clone()).ok().or_else(|| {
        let novel_seconds = indices.get("novel_seconds")?.as_f64()?;
        let familiar_seconds = indices.get("familiar_seconds")?.as_f64()?;
        let novel_bouts = indices
            .get("novel_bouts")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32;
        let familiar_bouts = indices
            .get("familiar_bouts")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32;
        Some(bekiper_behaviour::nort::NortIndices::compute(
            novel_seconds,
            familiar_seconds,
            novel_bouts,
            familiar_bouts,
        ))
    })
}
