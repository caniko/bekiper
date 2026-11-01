fn validate_no_auto_detect_inputs(
    config: &IngressConfig,
    output_dir: &Path,
    stages: &[Stage],
    force: bool,
) -> Result<()> {
    let resolver = PipelineInputResolver::new(output_dir);
    let will_run_detect = stages.contains(&Stage::Detect);
    let needs_detection_artifacts = stages.iter().any(|stage| {
        matches!(
            stage,
            Stage::Augment | Stage::Behaviour | Stage::Summary | Stage::Inspect
        )
    });
    let mut errors = Vec::new();

    for experiment in &config.experiments {
        if !experiment.needs_detection() {
            continue;
        }
        let exp_dir = resolver.exp_dir(experiment);
        let perimeter = resolver.perimeter_source(experiment, &exp_dir);
        let artifact_missing = perimeter.artifact_state == InputArtifactState::Missing;
        let detect_would_run = will_run_detect && (force || artifact_missing);
        let downstream_needs_missing_artifact = needs_detection_artifacts && artifact_missing;
        if !(detect_would_run || downstream_needs_missing_artifact) {
            continue;
        }

        if !perimeter.has_non_auto_source() {
            if let Some(reason) = perimeter.sources.iter().find_map(|source| match source {
                PerimeterInputSource::Missing { reason } => Some(reason.clone()),
                _ => None,
            }) {
                errors.push(format!("{}: {reason}", experiment.name));
                continue;
            }
            errors.push(format!(
                "{}: requires automatic detection ({}) but --no-auto-detect permits only precomputed/manual perimeter sources; provide precomputed/manual perimeters or run without --no-auto-detect",
                experiment.name,
                perimeter.source_summary()
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(BekiperError::Config(format_no_auto_detect_errors(&errors)))
    }
}

fn format_no_auto_detect_errors(errors: &[String]) -> String {
    format_preflight_errors(
        errors,
        format!(
            "--no-auto-detect preflight failed with {} unsupported or missing input(s); automatic detection will not run.",
            errors.len()
        ),
        "input(s)",
    )
}
