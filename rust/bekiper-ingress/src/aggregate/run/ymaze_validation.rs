fn validate_ymaze_aggregate_source_summaries(
    run_root: &Path,
    trials: &[TrialResult],
) -> Result<()> {
    let ymaze_trials = trials
        .iter()
        .filter(|trial| trial.task_type == TaskType::YMaze)
        .collect::<Vec<_>>();
    if ymaze_trials.is_empty() {
        return Ok(());
    }

    let mut arm_entries = [0_u64; 3];
    let mut arm_durations = [0.0_f64; 3];
    let mut missing_zone_details = Vec::new();

    for trial in &ymaze_trials {
        let Some(stats) = ymaze_arm_zone_stats(&trial.detailed) else {
            missing_zone_details.push(trial.experiment.clone());
            continue;
        };
        for idx in 0..3 {
            arm_entries[idx] += stats[idx].0;
            arm_durations[idx] += stats[idx].1;
        }
    }

    let zero_arms = ["A", "B", "C"]
        .iter()
        .enumerate()
        .filter_map(|(idx, label)| {
            (arm_entries[idx] == 0 && arm_durations[idx] == 0.0).then_some(*label)
        })
        .collect::<Vec<_>>();

    if zero_arms.len() < 2 && missing_zone_details.is_empty() {
        return Ok(());
    }

    let examples = ymaze_trials
        .iter()
        .take(20)
        .map(|trial| {
            format!(
                "{}: {}",
                trial.experiment,
                run_root
                    .join(&trial.experiment)
                    .join("result.toml")
                    .display()
            )
        })
        .collect::<Vec<_>>();

    let mut details = Vec::new();
    if !zero_arms.is_empty() {
        details.push(format!(
            "zero arm(s): {}; total entries A/B/C = {}/{}/{}; total duration A/B/C = {:.3}/{:.3}/{:.3}s",
            zero_arms.join(", "),
            arm_entries[0],
            arm_entries[1],
            arm_entries[2],
            arm_durations[0],
            arm_durations[1],
            arm_durations[2]
        ));
    }
    if !missing_zone_details.is_empty() {
        details.push(format!(
            "missing Y-maze zone_summaries in {} result(s): {}",
            missing_zone_details.len(),
            missing_zone_details.join(", ")
        ));
    }

    Err(BekiperError::Config(format!(
        "aggregate Y-maze source validation failed for {} trial(s); the source result.toml files are degenerate and aggregation will not write a misleading workbook. {}\nFix upstream DLC pose coordinates or Y-maze manual annotation reference mapping, then regenerate detect/behaviour/summary artifacts. Example source artifacts:\n{}",
        ymaze_trials.len(),
        details.join("; "),
        examples.join("\n")
    )))
}

fn ymaze_arm_zone_stats(detailed: &serde_json::Value) -> Option<[(u64, f64); 3]> {
    let zones = detailed.get("zone_summaries")?.as_array()?;
    let mut stats = [(0_u64, 0.0_f64); 3];
    for (idx, label) in ["A", "B", "C"].iter().enumerate() {
        let zone_name = format!("in_arm_{label}");
        let zone = zones.iter().find(|zone| {
            zone.get("zone_name")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|name| name == zone_name)
        })?;
        stats[idx] = (
            zone.get("entries")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            zone.get("duration_seconds")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
        );
    }
    Some(stats)
}

/// The recording part of an analysis name: a parameter variant
/// (`<recording>__<variant>`) scores the same protocol as its recording.
fn without_variant(name: &str) -> &str {
    name.split_once("__")
        .map_or(name, |(recording, _)| recording)
}

fn expected_imen_nort_sheet_from_name(experiment: &str) -> Option<&'static str> {
    let experiment = without_variant(experiment);
    if !experiment.starts_with("nort_videos_") {
        return None;
    }
    if experiment.contains("_t1_") {
        Some("nort_training")
    } else if experiment.contains("_t2_") {
        Some("nort_novelty")
    } else {
        None
    }
}
