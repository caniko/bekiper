fn analyze_classical_cv_backend(
    experiment: &ExperimentConfig,
    recipe: bekiper_core::types::DetectionRecipe,
    video_probe: Option<&VideoProbe>,
    summary: &mut DetectionDryRunSummary,
) {
    let Some(video_probe) = video_probe else {
        summary.push(
            DryRunSeverity::Error,
            "classical_cv",
            &experiment.name,
            "skipped smoke check because video probe failed",
        );
        return;
    };
    let _config = bekiper_detect::recipe::recipe_to_config(recipe);
    match VideoDecoder::open(&video_probe.path)
        .and_then(|mut decoder| decoder.frame_at(video_probe.detection_frame))
    {
        Ok(_) => summary.push(
            DryRunSeverity::Ok,
            "classical_cv",
            &experiment.name,
            format!(
                "decoded detection frame {} from {}",
                video_probe.detection_frame,
                video_probe.path.display()
            ),
        ),
        Err(error) => summary.push(
            DryRunSeverity::Error,
            "classical_cv",
            &experiment.name,
            format!(
                "failed to decode detection frame {} from {}: {error}",
                video_probe.detection_frame,
                video_probe.path.display()
            ),
        ),
    }
}

fn analyze_python_backend(
    experiment: &ExperimentConfig,
    recipe: bekiper_core::types::DetectionRecipe,
    backend: DetectionBackend,
    config_path: &Path,
    video_probe: Option<&VideoProbe>,
    summary: &mut DetectionDryRunSummary,
    python_smoke_cache: &mut HashSet<(String, String)>,
) {
    if video_probe.is_none() {
        summary.push(
            DryRunSeverity::Error,
            "python backend",
            &experiment.name,
            format!(
                "{} smoke skipped because video probe failed",
                backend.as_str()
            ),
        );
        return;
    }
    let key = (recipe.as_str().to_string(), backend.as_str().to_string());
    if !python_smoke_cache.insert(key) {
        summary.push(
            DryRunSeverity::Ok,
            "python backend",
            &experiment.name,
            format!(
                "{} smoke already passed for recipe {recipe}",
                backend.as_str()
            ),
        );
        return;
    }
    let python = resolve_python();
    let output = std::process::Command::new(&python)
        .arg("-m")
        .arg("bekiper_auto_detect")
        .arg("--smoke")
        .arg("--recipe")
        .arg(recipe.as_str())
        .arg("--backend")
        .arg(backend.as_str())
        .arg("--config")
        .arg(config_path)
        .output();
    match output {
        Ok(output) if output.status.success() => summary.push(
            DryRunSeverity::Ok,
            "python backend",
            &experiment.name,
            format!("{} smoke passed for recipe {recipe}", backend.as_str()),
        ),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stderr_tail = stderr
                .lines()
                .rev()
                .take(20)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            summary.push(
                DryRunSeverity::Error,
                "python backend",
                &experiment.name,
                format!(
                    "{} smoke failed for recipe {recipe}: {}",
                    backend.as_str(),
                    stderr_tail
                ),
            );
        }
        Err(error) => summary.push(
            DryRunSeverity::Error,
            "python backend",
            &experiment.name,
            format!(
                "failed to run python smoke for {}: {error}",
                backend.as_str()
            ),
        ),
    }
}
