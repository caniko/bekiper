pub(super) fn run_detect_filtered(
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
    gpu_only: bool,
    no_auto_detect: bool,
) -> Result<()> {
    if !experiment.needs_detection() {
        tracing::debug!("no detection configured, skipping");
        return Ok(());
    }

    let detection = experiment.detection.as_ref().unwrap();
    let recipe = detection
        .recipe
        .ok_or_else(|| BekiperError::Config("detection requires a recipe".into()))?;
    let effective = detection.effective_backends();
    let backends: Vec<_> = effective
        .backends()
        .iter()
        .filter(|b| !no_auto_detect || is_non_auto_backend(**b))
        .filter(|b| b.requires_gpu() == gpu_only)
        .collect();

    if backends.is_empty() {
        return Ok(());
    }

    // Try each backend in priority order
    let mut last_error: Option<String> = None;
    for backend in &backends {
        tracing::info!(backend = %backend, recipe = %recipe, "trying detection backend");

        match run_detect_backend(**backend, experiment, recipe, exp_dir) {
            Ok(()) => return Ok(()),
            Err(e) => {
                tracing::warn!(backend = %backend, error = %e, "detection backend failed");
                last_error = Some(e.to_string());
            }
        }
    }

    let gpu_fallback_available = effective.backends().iter().any(|backend| {
        backend.requires_gpu() && (!no_auto_detect || is_non_auto_backend(*backend))
    });

    // For CPU-only pass with GPU fallbacks, don't error yet — the GPU pass
    // will follow. CPU-only backend sets must fail here.
    if !gpu_only && gpu_fallback_available {
        return Ok(());
    }

    Err(BekiperError::Data(format!(
        "all detection backends failed for recipe {recipe}. Last error: {}",
        last_error.unwrap_or_default()
    )))
}

fn is_non_auto_backend(backend: DetectionBackend) -> bool {
    matches!(
        backend,
        DetectionBackend::PreComputed
            | DetectionBackend::ManualAnnotation
            | DetectionBackend::ArenaCalibrationCv
    )
}

fn run_detect_backend(
    backend: DetectionBackend,
    experiment: &ExperimentConfig,
    recipe: bekiper_core::types::DetectionRecipe,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    match backend {
        DetectionBackend::PreComputed => {
            let perimeters_path = experiment
                .detection
                .as_ref()
                .and_then(|d| d.perimeters.as_ref())
                .ok_or_else(|| {
                    BekiperError::Config("pre_computed backend requires a perimeters path".into())
                })?;
            let dest = exp_dir.pixel_perimeters_toml();
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(perimeters_path, &dest)?;
            tracing::info!(src = %perimeters_path.display(), "copied pre-computed perimeters");
            Ok(())
        }
        DetectionBackend::ManualAnnotation => run_detect_manual_annotation(experiment, exp_dir),
        DetectionBackend::ArenaCalibrationCv => run_detect_arena_calibration(experiment, exp_dir),
        DetectionBackend::ClassicalCv => {
            let video = experiment
                .video
                .as_ref()
                .ok_or_else(|| BekiperError::Config("detection requires a video file".into()))?;
            let overrides = experiment
                .detection
                .as_ref()
                .and_then(|d| d.pipeline_overrides.as_ref());
            run_detect_classical_cv(video, recipe, overrides, exp_dir)
        }
        DetectionBackend::Sam2Auto => {
            let video = experiment
                .video
                .as_ref()
                .ok_or_else(|| BekiperError::Config("detection requires a video file".into()))?;
            run_detect_python(video, recipe, exp_dir, "sam2_auto")
        }
        DetectionBackend::GroundingDinoSam2 => {
            let video = experiment
                .video
                .as_ref()
                .ok_or_else(|| BekiperError::Config("detection requires a video file".into()))?;
            run_detect_python(video, recipe, exp_dir, "grounding_dino_sam2")
        }
    }
}
