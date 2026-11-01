use indexmap::IndexMap;

use super::cell::CellValue;
use super::metadata::{animal_id_from_experiment, is_imen_video_experiment};
use super::trial::TrialResult;

pub(super) const IDENTITY_COLUMNS: &[&str] = &[
    "animal_trial_id",
    "animal_id",
    "trial_id",
    "recording_id",
    "analysis_id",
    "assay",
    "timepoint",
    "stage",
    "variant",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AggregateIdentity {
    pub animal_id: Option<u32>,
    pub trial_id: Option<u32>,
    pub requires_complete_identity: bool,
    pub override_animal_column: bool,
}

impl AggregateIdentity {
    pub fn columns(self, trial: &TrialResult) -> IndexMap<String, CellValue> {
        let mut columns = IndexMap::with_capacity(IDENTITY_COLUMNS.len());
        columns.insert(
            "animal_trial_id".into(),
            self.animal_trial_id()
                .map(CellValue::String)
                .unwrap_or(CellValue::Null),
        );
        columns.insert(
            "animal_id".into(),
            self.animal_id
                .map(|value| CellValue::Int(value as i64))
                .unwrap_or(CellValue::Null),
        );
        columns.insert(
            "trial_id".into(),
            self.trial_id
                .map(|value| CellValue::Int(value as i64))
                .unwrap_or(CellValue::Null),
        );
        let (recording, variant) = trial
            .experiment
            .split_once("__")
            .map_or((trial.experiment.as_str(), None), |(recording, variant)| {
                (recording, Some(variant))
            });
        columns.insert("recording_id".into(), CellValue::String(recording.into()));
        columns.insert(
            "analysis_id".into(),
            CellValue::String(trial.experiment.clone()),
        );
        columns.insert(
            "assay".into(),
            CellValue::String(
                match trial.task_type {
                    bekiper_core::types::TaskType::OpenField => "oft",
                    bekiper_core::types::TaskType::YMaze => "ymaze",
                    bekiper_core::types::TaskType::ObjectRecognition => "nort",
                    bekiper_core::types::TaskType::Cheeseboard => "cheeseboard",
                }
                .into(),
            ),
        );
        let timepoint = session_timepoint(&trial.session_context);
        columns.insert(
            "timepoint".into(),
            timepoint
                .map(|value| CellValue::String(value.into()))
                .unwrap_or(CellValue::Null),
        );
        columns.insert(
            "stage".into(),
            trial
                .session_context
                .get("stage")
                .map(CellValue::from_json)
                .unwrap_or(CellValue::Null),
        );
        columns.insert(
            "variant".into(),
            variant
                .map(|value| CellValue::String(value.into()))
                .unwrap_or(CellValue::Null),
        );
        columns
    }

    pub fn animal_cell(self) -> Option<CellValue> {
        self.animal_id.map(|value| CellValue::Int(value as i64))
    }

    fn animal_trial_id(self) -> Option<String> {
        self.animal_id
            .zip(self.trial_id)
            .map(|(animal_id, trial_id)| format!("A{animal_id}_T{trial_id}"))
    }

    fn validate(self, trial: &TrialResult) -> bekiper_core::error::Result<Self> {
        if !self.requires_complete_identity {
            return Ok(self);
        }
        if self.animal_id.is_none() {
            return Err(bekiper_core::error::BekiperError::Data(format!(
                "aggregate identity is missing animal_id for experiment {}",
                trial.experiment
            )));
        }
        // `resolve_trial_id` falls back to the animal, so a resolved animal
        // always has a trial.
        Ok(self)
    }
}

trait AggregateIdentityResolver {
    fn resolve(
        &self,
        trial: &TrialResult,
        row: &IndexMap<String, CellValue>,
    ) -> Option<AggregateIdentity>;
}

struct ImenVideoAnimalTrialResolver;

impl AggregateIdentityResolver for ImenVideoAnimalTrialResolver {
    fn resolve(
        &self,
        trial: &TrialResult,
        row: &IndexMap<String, CellValue>,
    ) -> Option<AggregateIdentity> {
        if !is_imen_video_experiment(&trial.experiment) {
            return None;
        }

        let animal_id = animal_id_from_experiment(&trial.experiment);
        Some(AggregateIdentity {
            animal_id,
            trial_id: resolve_trial_id(trial, row, animal_id),
            requires_complete_identity: true,
            override_animal_column: true,
        })
    }
}

struct AnimalTrialFallbackResolver;

impl AggregateIdentityResolver for AnimalTrialFallbackResolver {
    fn resolve(
        &self,
        trial: &TrialResult,
        row: &IndexMap<String, CellValue>,
    ) -> Option<AggregateIdentity> {
        let animal_id = json_u32(&trial.animal_metadata, "animal")
            .or_else(|| json_u32(&trial.session_context, "animal"))
            .or_else(|| cell_u32(row.get("animal")))
            .or_else(|| animal_id_from_experiment(&trial.experiment));

        Some(AggregateIdentity {
            animal_id,
            trial_id: resolve_trial_id(trial, row, animal_id),
            requires_complete_identity: false,
            override_animal_column: false,
        })
    }
}

pub(super) fn resolve_aggregate_identity(
    trial: &TrialResult,
    row: &IndexMap<String, CellValue>,
) -> bekiper_core::error::Result<AggregateIdentity> {
    const IMEN_VIDEO_ANIMAL_TRIAL: ImenVideoAnimalTrialResolver = ImenVideoAnimalTrialResolver;
    const ANIMAL_TRIAL_FALLBACK: AnimalTrialFallbackResolver = AnimalTrialFallbackResolver;
    const RESOLVERS: [&dyn AggregateIdentityResolver; 2] =
        [&IMEN_VIDEO_ANIMAL_TRIAL, &ANIMAL_TRIAL_FALLBACK];

    RESOLVERS
        .iter()
        .find_map(|resolver| resolver.resolve(trial, row))
        .expect("fallback aggregate identity resolver must always resolve")
        .validate(trial)
}

fn resolve_trial_id(
    trial: &TrialResult,
    row: &IndexMap<String, CellValue>,
    animal_id: Option<u32>,
) -> Option<u32> {
    json_u32(&trial.session_context, "trial")
        .or_else(|| json_u32(&trial.session_context, "trial_number"))
        .or_else(|| json_u32(&trial.session_context, "number"))
        .or_else(|| cell_u32(row.get("trial")))
        .or_else(|| cell_u32(row.get("trial_id")))
        .or_else(|| animal_id_from_experiment(&trial.experiment))
        .or(animal_id)
}

/// Pre/post timepoint of a session context: `timepoint`, else `phase_part` 1 or 2.
pub(crate) fn session_timepoint(context: &serde_json::Value) -> Option<&str> {
    context
        .get("timepoint")
        .and_then(serde_json::Value::as_str)
        .filter(|value| matches!(*value, "pre" | "post"))
        .or_else(|| match json_u32(context, "phase_part") {
            Some(1) => Some("pre"),
            Some(2) => Some("post"),
            _ => None,
        })
}

fn json_u32(value: &serde_json::Value, key: &str) -> Option<u32> {
    value.get(key).and_then(|value| {
        value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .or_else(|| value.as_str().and_then(parse_u32ish))
    })
}

fn cell_u32(value: Option<&CellValue>) -> Option<u32> {
    match value {
        Some(CellValue::Int(value)) => u32::try_from(*value).ok(),
        Some(CellValue::Float(value)) if value.is_finite() && value.fract() == 0.0 => {
            u32::try_from(*value as i64).ok()
        }
        Some(CellValue::String(value)) => parse_u32ish(value),
        _ => None,
    }
}

fn parse_u32ish(value: &str) -> Option<u32> {
    value
        .trim()
        .parse::<u32>()
        .ok()
        .or_else(|| value.trim().parse::<f64>().ok().map(|value| value as u32))
}
