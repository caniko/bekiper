/// Extract the animal prefix from an experiment name (everything before `_seq`).
fn animal_prefix(name: &str) -> &str {
    name.split("_seq").next().unwrap_or(name)
}

/// Keep only experiments whose animal matches the Nth unique animal (0-based).
fn filter_to_animal(experiments: &mut Vec<ExperimentConfig>, index: usize) {
    // Collect unique animal IDs in order of first appearance.
    let mut seen = Vec::new();
    for exp in experiments.iter() {
        let prefix = animal_prefix(&exp.name);
        if !seen.iter().any(|s: &String| s == prefix) {
            seen.push(prefix.to_owned());
        }
    }
    if let Some(target) = seen.get(index) {
        let target = target.clone();
        tracing::info!(animal = %target, index, total_animals = seen.len(), "filtering to single animal");
        experiments.retain(|e| animal_prefix(&e.name) == target);
    } else {
        tracing::warn!(
            index,
            total_animals = seen.len(),
            "animal index out of range, running all experiments"
        );
    }
}

fn resolve_stages(opts: &RunOptions) -> Vec<Stage> {
    if let Some(only) = opts.only_stage {
        vec![only]
    } else if let Some(from) = opts.from_stage {
        Stage::from_onward(from)
    } else {
        Stage::all().to_vec()
    }
}
