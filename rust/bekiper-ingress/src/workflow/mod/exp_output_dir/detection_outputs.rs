fn validate_detect_outputs_before_downstream(
    config: &IngressConfig,
    output_dir: &Path,
    stages: &[Stage],
) -> Result<()> {
    let pose_needs_detection = stages.contains(&Stage::Pose);
    let downstream_needs_detection_artifacts = stages.iter().any(|stage| {
        matches!(
            stage,
            Stage::Augment | Stage::Behaviour | Stage::Summary | Stage::Inspect
        )
    });

    if !pose_needs_detection && !downstream_needs_detection_artifacts {
        return Ok(());
    }

    let resolver = PipelineInputResolver::new(output_dir);
    let mut errors = Vec::new();

    for experiment in &config.experiments {
        if pose_needs_detection
            && !downstream_needs_detection_artifacts
            && !experiment.needs_pose_inference()
        {
            continue;
        }

        let exp_dir = resolver.exp_dir(experiment);
        let perimeter = resolver.perimeter_source(experiment, &exp_dir);
        if perimeter.artifact_state == InputArtifactState::Materialized {
            continue;
        }

        let source_note = if perimeter.has_non_auto_source() {
            format!(
                "; {} but the checkpoint is not materialized; run bekiper run --only detect --no-auto-detect",
                perimeter.source_summary()
            )
        } else {
            format!("; {}", perimeter.source_summary())
        };

        errors.push(format!(
            "{}: missing {}{}",
            experiment.name,
            perimeter.artifact_path.display(),
            source_note
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(BekiperError::Config(format_detect_output_errors(&errors)))
    }
}

fn format_detect_output_errors(errors: &[String]) -> String {
    format_preflight_errors(
        errors,
        format!(
            "detect output preflight failed with {} missing perimeter checkpoint(s); downstream stages will not run.",
            errors.len()
        ),
        "missing checkpoint(s)",
    )
}
