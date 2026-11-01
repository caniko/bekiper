fn filter_analysis_excluded_experiments(experiments: &mut Vec<ExperimentConfig>, stages: &[Stage]) {
    if !stages_trigger_analysis_exclusion(stages) {
        return;
    }

    let skipped = experiments
        .iter()
        .filter(|experiment| is_analysis_excluded_experiment(experiment))
        .map(|experiment| experiment.name.clone())
        .collect::<Vec<_>>();
    if skipped.is_empty() {
        return;
    }

    experiments.retain(|experiment| !is_analysis_excluded_experiment(experiment));
    tracing::info!(
        count = skipped.len(),
        experiments = %skipped.join(", "),
        "skipping analysis-excluded experiments"
    );
}

fn stages_trigger_analysis_exclusion(stages: &[Stage]) -> bool {
    stages.iter().any(|stage| {
        matches!(
            stage,
            Stage::Detect | Stage::Augment | Stage::Behaviour | Stage::Summary | Stage::Inspect
        )
    })
}

fn is_analysis_excluded_experiment(experiment: &ExperimentConfig) -> bool {
    crate::analysis_exclusion::is_analysis_excluded_experiment(experiment)
}
