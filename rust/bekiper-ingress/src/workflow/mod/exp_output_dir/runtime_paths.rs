fn format_preflight_errors(errors: &[String], mut message: String, remaining: &str) -> String {
    const MAX_ERRORS: usize = 50;
    for error in errors.iter().take(MAX_ERRORS) {
        message.push('\n');
        message.push_str(error);
    }
    if errors.len() > MAX_ERRORS {
        message.push_str(&format!(
            "\n... and {} more {remaining}",
            errors.len() - MAX_ERRORS
        ));
    }
    message
}

fn apply_run_option_overrides(config: &mut IngressConfig, opts: &RunOptions) {
    if opts.no_inspect {
        config.inspect.enabled = false;
    }
    if let Some(trials) = &opts.inspection_video_trials {
        config.inspect.video_trials = trials.clone();
    }
    if opts.no_aggregate {
        config.aggregate.enabled = false;
    }
    if let Some(output_filename) = &opts.aggregate_output {
        config.aggregate.output_filename = output_filename.clone();
    }
    if let Some(invert_y) = opts.invert_y {
        config.pipeline.invert_y = invert_y;
    }
    if let Some(num_threads) = opts.num_threads {
        config.project.runtime.num_threads = Some(num_threads);
    }
    if let Some(memory_scheduler_enabled) = opts.memory_scheduler_enabled {
        config.project.runtime.memory_scheduler_enabled = memory_scheduler_enabled;
    }
    if let Some(max_ram_usage_fraction) = opts.max_ram_usage_fraction {
        config.project.runtime.max_ram_usage_fraction = max_ram_usage_fraction;
    }
}

pub(super) fn normalize_runtime_paths(config: &mut IngressConfig, base_dir: &Path) {
    if let Some(plan) = config.post_analysis.plan_path.as_mut() {
        *plan = resolve_runtime_path(base_dir, plan);
        if !plan.is_absolute()
            && let Ok(current) = std::env::current_dir()
        {
            *plan = current.join(&*plan);
        }
    }
    if let Some(pose_model) = config.pose_model.as_mut() {
        *pose_model = pose_model.resolved_against(base_dir);
    }

    for experiment in &mut config.experiments {
        if let Some(pose_model) = experiment.pose_model.as_mut() {
            *pose_model = pose_model.resolved_against(base_dir);
        }
        if let Some(video) = experiment.video.as_mut() {
            *video = resolve_runtime_path(base_dir, video);
        }
        for data_file in &mut experiment.data_files {
            *data_file = resolve_runtime_path(base_dir, data_file);
        }
        if let Some(annotation_dir) = experiment.annotation_dir.as_mut() {
            *annotation_dir = resolve_runtime_path(base_dir, annotation_dir);
        }
        if let Some(frame_image) = experiment.frame_image.as_mut() {
            *frame_image = resolve_runtime_path(base_dir, frame_image);
        }
        if let Some(timestamps_file) = experiment.timestamps_file.as_mut() {
            *timestamps_file = resolve_runtime_path(base_dir, timestamps_file);
        }
        if let Some(perimeters) = experiment
            .detection
            .as_mut()
            .and_then(|detection| detection.perimeters.as_mut())
        {
            *perimeters = resolve_runtime_path(base_dir, perimeters);
        }
    }
}

pub(super) fn resolve_runtime_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() || path.starts_with(base_dir) {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}
