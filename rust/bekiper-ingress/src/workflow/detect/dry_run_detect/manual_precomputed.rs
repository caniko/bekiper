fn analyze_precomputed_backend(
    experiment: &ExperimentConfig,
    resolver: &PipelineInputResolver<'_>,
    exp_dir: &ExperimentOutputDir,
    summary: &mut DetectionDryRunSummary,
) {
    let status = resolver.perimeter_source(experiment, exp_dir);
    match status.sources.iter().find_map(|source| match source {
        PerimeterInputSource::PreComputed { path, count } => Some(Ok((path, count))),
        PerimeterInputSource::Missing { reason }
            if reason.contains("precomputed") || reason.contains("pre_computed") =>
        {
            Some(Err(reason))
        }
        _ => None,
    }) {
        Some(Ok((path, count))) => summary.push(
            DryRunSeverity::Ok,
            "precomputed",
            &experiment.name,
            format!("{} parses ({} perimeter(s))", path.display(), count),
        ),
        Some(Err(reason)) => summary.push(
            DryRunSeverity::Error,
            "precomputed",
            &experiment.name,
            reason.clone(),
        ),
        None => summary.push(
            DryRunSeverity::Error,
            "precomputed",
            &experiment.name,
            "pre_computed backend did not resolve a source",
        ),
    }
}

fn analyze_manual_annotation_backend(
    experiment: &ExperimentConfig,
    resolver: &PipelineInputResolver<'_>,
    exp_dir: &ExperimentOutputDir,
    summary: &mut DetectionDryRunSummary,
) {
    let status = resolver.perimeter_source(experiment, exp_dir);
    match status.sources.iter().find_map(|source| match source {
        PerimeterInputSource::ManualAnnotation(annotation) => Some(Ok(annotation)),
        PerimeterInputSource::Missing { reason }
            if reason.contains("manual_annotation") || reason.contains("annotation") =>
        {
            Some(Err(reason))
        }
        _ => None,
    }) {
        Some(Ok(annotation)) => {
            summary.push(
                DryRunSeverity::Ok,
                "annotation",
                &experiment.name,
                format!(
                    "manual annotations resolve via {} ({} perimeter(s): {})",
                    annotation.detail,
                    annotation.specs.len(),
                    annotation.labels.join(", ")
                ),
            );
        }
        Some(Err(reason)) => summary.push(
            DryRunSeverity::Error,
            "annotation",
            &experiment.name,
            reason.clone(),
        ),
        None => summary.push(
            DryRunSeverity::Error,
            "annotation",
            &experiment.name,
            "manual_annotation backend did not resolve a source",
        ),
    }
}

fn analyze_arena_calibration_backend(
    experiment: &ExperimentConfig,
    resolver: &PipelineInputResolver,
    exp_dir: &ExperimentOutputDir,
    summary: &mut DetectionDryRunSummary,
) {
    let status = resolver.perimeter_source(experiment, exp_dir);
    match status.sources.iter().find(|source| {
        matches!(
            source,
            PerimeterInputSource::ArenaCalibrationCv | PerimeterInputSource::Missing { .. }
        )
    }) {
        Some(PerimeterInputSource::ArenaCalibrationCv) => summary.push(
            DryRunSeverity::Ok,
            "arena_calibration_cv",
            &experiment.name,
            "OFT arena calibration source available",
        ),
        Some(PerimeterInputSource::Missing { reason }) => summary.push(
            DryRunSeverity::Error,
            "arena_calibration_cv",
            &experiment.name,
            reason,
        ),
        _ => summary.push(
            DryRunSeverity::Error,
            "arena_calibration_cv",
            &experiment.name,
            "arena_calibration_cv backend did not resolve a source",
        ),
    }
}
