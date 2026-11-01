pub(super) fn enrich_trials_with_metadata(
    trials: &mut Vec<TrialResult>,
    metadata: &AggregateMetadata,
    policy: MissingMetadataPolicy,
) -> Result<Vec<MetadataExclusion>> {
    let mut exclusions = Vec::new();
    let mut missing = Vec::new();
    let mut seen_missing = HashSet::new();

    trials.retain_mut(|trial| {
        let Some(animal) = trial_animal_id(trial) else {
            return true;
        };
        let Some(row) = metadata.rows.get(&animal) else {
            if seen_missing.insert(animal) {
                missing.push(animal);
                if policy == MissingMetadataPolicy::ExcludeMissing {
                    exclusions.push(MetadataExclusion {
                        animal,
                        reason: "missing_animal_metadata".into(),
                    });
                }
            }
            return policy != MissingMetadataPolicy::ExcludeMissing;
        };
        merge_trial_metadata(trial, row);
        true
    });

    if policy == MissingMetadataPolicy::Fail && !missing.is_empty() {
        missing.sort_unstable();
        return Err(BekiperError::Data(format!(
            "{}: aggregate metadata missing required Animal ID row(s): {}",
            metadata.path.display(),
            missing
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }

    exclusions.sort_by_key(|row| row.animal);
    Ok(exclusions)
}

pub(super) fn trial_animal_id(trial: &TrialResult) -> Option<u32> {
    if is_imen_video_experiment(&trial.experiment)
        && let Some(animal) = animal_id_from_experiment(&trial.experiment)
    {
        return Some(animal);
    }

    trial
        .animal_metadata
        .get("animal")
        .and_then(json_to_u32)
        .or_else(|| trial.session_context.get("animal").and_then(json_to_u32))
        .or_else(|| animal_id_from_experiment(&trial.experiment))
}

fn json_to_u32(value: &Value) -> Option<u32> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .or_else(|| value.as_str().and_then(parse_animal_id))
}

pub(crate) fn animal_id_from_experiment(experiment: &str) -> Option<u32> {
    // Strip any trailing `__<variant>` suffix attached by the task-variant
    // expansion path; the animal id lives on the base experiment name.
    let base = experiment.split("__").next().unwrap_or(experiment);
    if let Some(first) = base.split('_').next()
        && let Ok(value) = first.parse::<u32>()
    {
        return Some(value);
    }
    let marker = "_video_";
    let index = base.rfind(marker)?;
    base[index + marker.len()..].parse::<u32>().ok()
}

pub(crate) fn is_imen_video_experiment(experiment: &str) -> bool {
    matches!(
        experiment,
        name if name.starts_with("nort_videos_")
            || name.starts_with("oft_videos_")
            || name.starts_with("y_maze_videos_")
    )
}

fn merge_trial_metadata(trial: &mut TrialResult, row: &IndexMap<String, CellValue>) {
    let mut object = match trial.animal_metadata.take() {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    for (key, value) in row {
        object.insert(key.clone(), cell_to_json(value));
    }
    trial.animal_metadata = Value::Object(object);
}

pub(super) fn cell_to_json(value: &CellValue) -> Value {
    match value {
        CellValue::Null => Value::Null,
        CellValue::Bool(value) => Value::Bool(*value),
        CellValue::Int(value) => Value::Number((*value).into()),
        CellValue::Float(value) => serde_json::Number::from_f64(*value)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        CellValue::FloatList(values) => Value::Array(
            values
                .iter()
                .filter_map(|value| serde_json::Number::from_f64(*value).map(Value::Number))
                .collect(),
        ),
        CellValue::String(value) => Value::String(value.clone()),
    }
}
