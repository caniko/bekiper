fn validate_no_detect_inputs(
    config: &IngressConfig,
    output_dir: &Path,
    stages: &[Stage],
) -> Result<()> {
    let resolver = PipelineInputResolver::new(output_dir);
    let will_run_pose = stages.contains(&Stage::Pose);
    let needs_pose_checkpoint = stages.iter().any(|stage| {
        matches!(
            stage,
            Stage::Augment | Stage::Behaviour | Stage::Summary | Stage::Inspect
        )
    }) && !will_run_pose;
    let needs_detection_artifacts = stages.iter().any(|stage| {
        matches!(
            stage,
            Stage::Augment | Stage::Behaviour | Stage::Summary | Stage::Inspect
        )
    });

    let mut errors = Vec::new();
    for experiment in &config.experiments {
        let exp_dir = resolver.exp_dir(experiment);

        if will_run_pose {
            let pose = resolver.pose_source(experiment);
            if !pose.is_available() {
                for missing in pose.missing {
                    errors.push(format!("{}: {missing}", experiment.name));
                }
            }
        }

        if needs_pose_checkpoint && !exp_dir.poses_parquet().exists() {
            errors.push(format!(
                "{}: missing {}; run the pose stage or provide precomputed pose data before using --no-detect",
                experiment.name,
                exp_dir.poses_parquet().display()
            ));
        }

        if needs_detection_artifacts && experiment.needs_detection() {
            let perimeter = resolver.perimeter_source(experiment, &exp_dir);
            if perimeter.artifact_state == InputArtifactState::Missing {
                let source_note = if perimeter.has_non_auto_source() {
                    format!(
                        "; {} but {} is not materialized; run bekiper run --only detect --no-auto-detect",
                        perimeter.source_summary(),
                        perimeter.artifact_path.display()
                    )
                } else {
                    format!("; {}", perimeter.source_summary())
                };
                errors.push(format!(
                    "{}: missing {}{}; --no-detect will not run manual, classical, or GPU detection backends",
                    experiment.name,
                    perimeter.artifact_path.display(),
                    source_note
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(BekiperError::Config(format_no_detect_errors(&errors)))
    }
}

fn format_no_detect_errors(errors: &[String]) -> String {
    format_preflight_errors(
        errors,
        format!(
            "--no-detect preflight failed with {} missing input(s); detection will not run.",
            errors.len()
        ),
        "missing input(s)",
    )
}
