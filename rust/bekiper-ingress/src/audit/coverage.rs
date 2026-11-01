//! Audit coverage responsibilities.
use super::*;

/// Check that every expanded manual-annotation experiment resolves to an
/// annotation asset. This intentionally mirrors `bekiper-annotate` session
/// indexing: a session is assigned when `AnnotationSession::load_with_context`
/// succeeds.
pub fn check_manual_annotation_coverage(path: &Path) -> Result<ManualAnnotationCoverageSummary> {
    let mut config = IngressConfig::from_file(path)?;
    let base_dir = path.parent().unwrap_or(Path::new("."));
    config.expand_for_validation(base_dir, &[])?;

    let mut total_manual_sessions = 0usize;
    let mut assigned_sessions = 0usize;
    let mut missing_sessions = Vec::new();

    for experiment in &config.experiments {
        let Some(detection) = &experiment.detection else {
            continue;
        };
        if !detection
            .backend
            .backends()
            .contains(&DetectionBackend::ManualAnnotation)
        {
            continue;
        }
        if crate::analysis_exclusion::is_analysis_excluded_experiment(experiment) {
            continue;
        }

        total_manual_sessions += 1;

        let video_path = experiment
            .video
            .as_ref()
            .map(|video| resolve_config_path(base_dir, video));
        let annotation_dir = experiment.annotation_dir.clone();

        let (Some(video_path_ref), Some(annotation_dir_ref)) =
            (video_path.as_ref(), annotation_dir.as_ref())
        else {
            missing_sessions.push(ManualAnnotationCoverageFailure {
                experiment: experiment.name.clone(),
                video_path,
                annotation_dir,
                error: "manual-annotation experiment requires both video and annotation_dir".into(),
            });
            continue;
        };

        match AnnotationSession::load_with_context(
            annotation_dir_ref,
            video_path_ref,
            experiment.session_context.clone(),
        ) {
            Ok(_) => assigned_sessions += 1,
            Err(error) => {
                let error = clearer_manual_annotation_error(
                    annotation_dir_ref,
                    video_path_ref,
                    &error.to_string(),
                );
                missing_sessions.push(ManualAnnotationCoverageFailure {
                    experiment: experiment.name.clone(),
                    video_path,
                    annotation_dir,
                    error,
                });
            }
        }
    }

    Ok(ManualAnnotationCoverageSummary {
        config_path: path.to_path_buf(),
        total_manual_sessions,
        assigned_sessions,
        missing_sessions,
    })
}
