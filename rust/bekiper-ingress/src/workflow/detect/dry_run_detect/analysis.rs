pub fn analyze_detection_dry_run(config_path: &Path) -> Result<DetectionDryRunSummary> {
    let mut config = IngressConfig::from_file(config_path)?;
    let base_dir = config_path.parent().unwrap_or(Path::new("."));
    config.expand_for_validation(base_dir, &[Stage::Detect])?;
    normalize_runtime_paths(&mut config, base_dir);

    let mut summary = DetectionDryRunSummary {
        total_experiments: config.experiments.len(),
        ..DetectionDryRunSummary::default()
    };
    let mut python_smoke_cache = HashSet::new();
    let output_dir = base_dir.join(
        config
            .project
            .output_dir
            .as_deref()
            .unwrap_or(Path::new("output")),
    );
    let resolver = PipelineInputResolver::new(&output_dir);

    for experiment in &config.experiments {
        let exp_dir = resolver.exp_dir(experiment);
        analyze_experiment_dry_run(
            experiment,
            &resolver,
            &exp_dir,
            base_dir,
            config_path,
            &mut summary,
            &mut python_smoke_cache,
        );
    }

    Ok(summary)
}

fn analyze_experiment_dry_run(
    experiment: &ExperimentConfig,
    resolver: &PipelineInputResolver<'_>,
    exp_dir: &ExperimentOutputDir,
    base_dir: &Path,
    config_path: &Path,
    summary: &mut DetectionDryRunSummary,
    python_smoke_cache: &mut HashSet<(String, String)>,
) {
    let Some(detection) = experiment.detection.as_ref() else {
        summary.skipped_experiments += 1;
        summary.push(
            DryRunSeverity::Skipped,
            "recipe/backend",
            &experiment.name,
            "no detection configured; experiment skipped",
        );
        return;
    };

    let Some(recipe) = detection.recipe else {
        summary.skipped_experiments += 1;
        summary.push(
            DryRunSeverity::Skipped,
            "recipe/backend",
            &experiment.name,
            "no detection recipe; detection skipped",
        );
        return;
    };

    let effective = detection.effective_backends();
    let backend_strs = effective
        .backends()
        .iter()
        .map(|backend| backend.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    summary.push(
        DryRunSeverity::Ok,
        "recipe/backend",
        &experiment.name,
        format!("recipe {recipe}; backends [{backend_strs}]"),
    );

    let video_probe = probe_experiment_video(experiment, base_dir, summary);

    for backend in effective.backends() {
        match backend {
            DetectionBackend::PreComputed => {
                analyze_precomputed_backend(experiment, resolver, exp_dir, summary);
            }
            DetectionBackend::ManualAnnotation => {
                analyze_manual_annotation_backend(experiment, resolver, exp_dir, summary);
            }
            DetectionBackend::ArenaCalibrationCv => {
                analyze_arena_calibration_backend(experiment, resolver, exp_dir, summary);
            }
            DetectionBackend::ClassicalCv => {
                analyze_classical_cv_backend(experiment, recipe, video_probe.as_ref(), summary);
            }
            DetectionBackend::Sam2Auto | DetectionBackend::GroundingDinoSam2 => {
                analyze_python_backend(
                    experiment,
                    recipe,
                    *backend,
                    config_path,
                    video_probe.as_ref(),
                    summary,
                    python_smoke_cache,
                );
            }
        }
    }
}
