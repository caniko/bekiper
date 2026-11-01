//! Audit validation responsibilities.
use super::*;

/// Parse, expand, and validate a project config without mutating data roots.
pub fn validate_config_file(path: &Path) -> Result<ConfigAuditSummary> {
    let base_dir = path.parent().unwrap_or(Path::new("."));
    validate_config_file_at(path, base_dir)
}

/// Read-only validation using the same relative-path base as a registry data_root.
pub fn validate_config_file_at(path: &Path, base_dir: &Path) -> Result<ConfigAuditSummary> {
    let mut config = IngressConfig::from_file(path)?;
    let scan_entries = config.scan.len();
    let animal_scan_entries = config.animal_scan.len();
    let ingress_profile_entries = config.ingress_profile.len();

    config.expand_for_validation(base_dir, Stage::all())?;

    let mut validated_annotation_dirs = HashSet::new();
    for experiment in &config.experiments {
        validate_experiment(
            base_dir,
            &config,
            experiment,
            &mut validated_annotation_dirs,
        )?;
    }

    Ok(ConfigAuditSummary {
        config_path: path.to_path_buf(),
        experiments: config.experiments.len(),
        scan_entries,
        animal_scan_entries,
        ingress_profile_entries,
    })
}

pub(super) fn validate_experiment(
    base_dir: &Path,
    config: &IngressConfig,
    experiment: &ExperimentConfig,
    validated_annotation_dirs: &mut HashSet<PathBuf>,
) -> Result<()> {
    if experiment.data_files.is_empty() && experiment.video.is_none() {
        return Err(BekiperError::Config(format!(
            "experiment '{}' has neither data_files nor video",
            experiment.name
        )));
    }

    for data_file in &experiment.data_files {
        validate_existing_path(
            "data file",
            &experiment.name,
            &resolve_config_path(base_dir, data_file),
        )?;
    }

    if let Some(video) = &experiment.video {
        validate_existing_path(
            "video",
            &experiment.name,
            &resolve_config_path(base_dir, video),
        )?;
    }

    if let Some(ref pose_model) = experiment.pose_model {
        let resolved_pose_model = pose_model.resolved_against(base_dir);
        validate_existing_path("pose model", &experiment.name, &resolved_pose_model.path)?;

        #[cfg(feature = "pose-gpu")]
        {
            let spec = resolve_model_spec(&resolved_pose_model.spec).map_err(|err| {
                BekiperError::Config(format!(
                    "experiment '{}' has invalid pose model spec '{}': {err}",
                    experiment.name, pose_model.spec
                ))
            })?;
            validate_pose_label_model_labels(
                config.pipeline.pose_label_model,
                experiment.task_type,
                &spec.labels,
                &experiment.name,
                "pose_model.spec",
            )?;
        }
    } else if experiment.video.is_some() && experiment.data_files.is_empty() {
        return Err(BekiperError::Config(format!(
            "experiment '{}' requires a pose_model when video is present and data_files are absent",
            experiment.name
        )));
    }

    if experiment.pose_model.is_none() {
        for data_file in &experiment.data_files {
            let path = resolve_config_path(base_dir, data_file);
            if let Some(labels) = pose_data_labels(&path)? {
                validate_pose_label_model_labels(
                    config.pipeline.pose_label_model,
                    experiment.task_type,
                    &labels,
                    &experiment.name,
                    &format!("data file {}", path.display()),
                )?;
            }
        }
    }

    if let Some(annotation_dir) = &experiment.annotation_dir {
        validate_existing_path("annotation directory", &experiment.name, annotation_dir)?;
        if validated_annotation_dirs.insert(annotation_dir.clone()) {
            AnnotationResolver::from_dir(annotation_dir).map_err(|err| {
                BekiperError::Config(format!(
                    "experiment '{}' has invalid annotation directory '{}': {err}",
                    experiment.name,
                    annotation_dir.display()
                ))
            })?;
        }
    }

    if let Some(detection) = &experiment.detection
        && let Some(perimeters_path) = &detection.perimeters
    {
        let resolved = resolve_config_path(base_dir, perimeters_path);
        let is_virtual_animal_scan_temp = experiment.animal_id.is_some()
            && resolved
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.contains(".perimeters-seq"));
        if !is_virtual_animal_scan_temp {
            validate_existing_path("perimeters TOML", &experiment.name, &resolved)?;
        }
    }

    if config.pipeline.invert_y
        && experiment.video_resolution.is_none()
        && experiment.video.is_none()
    {
        return Err(BekiperError::Config(format!(
            "experiment '{}' cannot use pipeline.invert_y without a video or resolved video metadata",
            experiment.name
        )));
    }

    Ok(())
}

pub(super) fn validate_pose_label_model_labels(
    selected: PoseLabelModel,
    task_type: bekiper_core::types::TaskType,
    labels: &[String],
    experiment_name: &str,
    source: &str,
) -> Result<()> {
    if let Some(inferred) = PoseLabelModel::infer_from_labels(labels)
        && inferred != selected
    {
        return Err(BekiperError::Config(format!(
            "experiment '{experiment_name}' selects pose_label_model '{}' but {source} labels match '{}'",
            selected.as_str(),
            inferred.as_str()
        )));
    }

    let missing = selected.missing_source_labels_for_task(task_type, labels);
    if !missing.is_empty() {
        return Err(BekiperError::Config(format!(
            "experiment '{experiment_name}' selects pose_label_model '{}' but {source} is missing required label(s): {}",
            selected.as_str(),
            missing.join(", ")
        )));
    }

    Ok(())
}

pub(super) fn pose_data_labels(path: &Path) -> Result<Option<Vec<String>>> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("parquet") || ext.eq_ignore_ascii_case("pq") => {
            let schema = bekiper_pose::io::scan_file(path)?.collect_schema()?;
            let mut labels = HashSet::new();
            for name in schema.iter_names() {
                if let Some(label) = name.as_str().strip_suffix("_x") {
                    labels.insert(label.to_string());
                }
            }
            Ok(Some(labels.into_iter().collect()))
        }
        Some(ext)
            if ext.eq_ignore_ascii_case("csv")
                || ext.eq_ignore_ascii_case("h5")
                || ext.eq_ignore_ascii_case("hdf5") =>
        {
            let reader = bekiper_pose::detect::open(path)?;
            Ok(Some(reader.labels().to_vec()))
        }
        _ => Ok(None),
    }
}

pub(super) fn resolve_config_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

pub(super) fn validate_existing_path(kind: &str, experiment_name: &str, path: &Path) -> Result<()> {
    if path.exists() {
        Ok(())
    } else {
        Err(BekiperError::Config(format!(
            "experiment '{}' references missing {} '{}'",
            experiment_name,
            kind,
            path.display()
        )))
    }
}

pub(super) fn clearer_manual_annotation_error(
    annotation_dir: &Path,
    video_path: &Path,
    original: &str,
) -> String {
    if original.contains("could not resolve canonical annotation mapping")
        && let Some(message) = crate::config::profile::missing_imen_metadata_context_message(
            annotation_dir,
            video_path,
        )
    {
        return message;
    }
    original.to_string()
}
