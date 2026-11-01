fn warn_missing_oft_fecal_boli_source(trials: &[TrialResult]) {
    let mut missing = Vec::new();
    for trial in trials
        .iter()
        .filter(|trial| trial.task_type == TaskType::OpenField)
    {
        let has_count = trial
            .session_context
            .get("fecal_boli_count")
            .and_then(|value| value.as_u64())
            .is_some();
        if has_count {
            continue;
        }

        let source_file = trial
            .session_context
            .get("source_file")
            .and_then(|value| value.as_str())
            .unwrap_or("examples/neo/imen/annotations/nort/metadata.csv");
        let source_row = trial
            .session_context
            .get("source_row")
            .and_then(|value| value.as_u64())
            .map(|row| row.to_string())
            .unwrap_or_else(|| "unknown".into());
        missing.push(format!(
            "{}: missing/invalid optional FecalBoli value in {source_file} row {source_row}",
            trial.experiment
        ));
    }

    if missing.is_empty() {
        return;
    }

    let examples = missing.iter().take(10).cloned().collect::<Vec<_>>();
    tracing::warn!(
        count = missing.len(),
        examples = %examples.join("\n"),
        "aggregate OFT rows are missing optional fecal_boli_count metadata; cells will be blank"
    );
}

fn validate_nort_aggregate_consistency(run_root: &Path, trials: &[TrialResult]) -> Result<()> {
    let mut mismatches = Vec::new();
    for trial in trials {
        let Some(expected) = expected_imen_nort_sheet_from_name(&trial.experiment) else {
            continue;
        };
        if trial.task_type != TaskType::ObjectRecognition {
            continue;
        }

        if let Some(protocol) = trial
            .detailed
            .get("protocol")
            .and_then(|value| value.as_str())
            && protocol != expected
        {
            mismatches.push(format!(
                "{}: result protocol is {protocol}, expected {expected} from experiment name",
                trial.experiment
            ));
        }

        if let Some(sheet_key) = experiment_for_trial(trial)
            .sheet_key(trial)
            .map(|sheet_key| sheet_key.routing_key())
            && without_variant(&sheet_key) != expected
        {
            mismatches.push(format!(
                "{}: aggregate sheet key is {sheet_key}, expected {expected} from experiment name",
                trial.experiment
            ));
        }

        let pixel_perimeters = checkpoint_root(run_root)
            .join(&trial.experiment)
            .join("pixel_perimeters.toml");
        if !pixel_perimeters.exists() {
            continue;
        }
        let contents = std::fs::read_to_string(&pixel_perimeters).map_err(|e| {
            BekiperError::Other(format!(
                "failed to read {}: {e}",
                pixel_perimeters.display()
            ))
        })?;
        let file: PerimetersFile = toml::from_str(&contents).map_err(|e| {
            BekiperError::Config(format!("invalid {}: {e}", pixel_perimeters.display()))
        })?;
        let labels: HashSet<String> = file
            .perimeters
            .iter()
            .map(|perimeter| perimeter.label().to_ascii_lowercase())
            .collect();
        match expected {
            "nort_training"
                if !labels.contains("familiar")
                    || !labels.contains("variable")
                    || labels.contains("novel") =>
            {
                mismatches.push(format!(
                    "{}: perimeter labels {:?}, expected familiar + variable for nort_training",
                    trial.experiment, labels
                ));
            }
            "nort_novelty"
                if !labels.contains("familiar")
                    || !labels.contains("novel")
                    || labels.contains("variable") =>
            {
                mismatches.push(format!(
                    "{}: perimeter labels {:?}, expected familiar + novel for nort_novelty",
                    trial.experiment, labels
                ));
            }
            _ => {}
        }
    }

    if mismatches.is_empty() {
        return Ok(());
    }
    Err(BekiperError::Config(format!(
        "aggregate NORT consistency check failed with {} mismatch(es); stale or misclassified result.toml files must be regenerated before aggregation:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    )))
}
