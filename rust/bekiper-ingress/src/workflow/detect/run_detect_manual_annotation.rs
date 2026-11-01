/// Resolve perimeters from manual annotations (metadata CSV + plugin files).
///
/// Also writes `annotation_scale.toml` if the plugin directory contains
/// `meters_per_pixel-*.csv` scale calibration files. This is picked up
/// by the augment stage to override the config-level scale.
fn run_detect_manual_annotation(
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    use super::inputs::{
        PerimeterInputSource, PipelineInputResolver, materialize_manual_annotation,
    };

    let resolver = PipelineInputResolver::new(exp_dir.output_root());
    let status = resolver.perimeter_source(experiment, exp_dir);
    let annotation = status.first_manual_annotation().ok_or_else(|| {
        let reason = status
            .sources
            .iter()
            .find_map(|source| match source {
                PerimeterInputSource::Missing { reason } => Some(reason.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "manual_annotation backend did not resolve a source".into());
        BekiperError::Data(reason)
    })?;
    materialize_manual_annotation(experiment, annotation, exp_dir)?;

    tracing::info!(
        experiment = %experiment.name,
        source = %annotation.detail,
        count = annotation.specs.len(),
        "resolved perimeters from manual annotations"
    );

    Ok(())
}

/// Frame index to use for detection: middle of the video.
/// Early frames often have the experimenter's arm placing the animal.
fn detection_frame_index(decoder: &VideoDecoder) -> u64 {
    decoder.info().total_frames / 2
}

fn run_detect_classical_cv(
    video: &Path,
    recipe: bekiper_core::types::DetectionRecipe,
    overrides: Option<&crate::config::DetectionPipelineOverrides>,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    let mut decoder = VideoDecoder::open(video)?;
    let frame_idx = detection_frame_index(&decoder);
    let frame = decoder.frame_at(frame_idx)?;

    // Save detection frame for visual inspection
    let frame_path = exp_dir.detection_frame_png();
    std::fs::create_dir_all(frame_path.parent().unwrap())?;
    frame
        .save(&frame_path)
        .map_err(|e| BekiperError::Other(format!("failed to save detection frame: {e}")))?;
    tracing::info!(frame = frame_idx, path = %frame_path.display(), "saved detection frame");

    let gray = frame.to_luma8();

    // Build config from recipe defaults, applying optional user overrides.
    let config = {
        let base = bekiper_detect::recipe::recipe_to_config(recipe);
        match overrides {
            Some(o) => o.apply(base),
            None => base,
        }
    };

    match bekiper_detect::recipe::run_with_config(config, recipe, &gray) {
        Ok(result) => {
            save_classical_cv_debug(&result.debug, &frame, &exp_dir.debug_dir());
            write_perimeters(&result.perimeters, exp_dir)?;
            tracing::info!(
                backend = "classical_cv",
                count = result.perimeters.len(),
                "detection complete"
            );
            Ok(())
        }
        Err(e) => {
            if let Some(debug) = e.into_debug() {
                save_classical_cv_debug(&debug, &frame, &exp_dir.debug_dir());
            }
            Err(BekiperError::Data(format!(
                "classical_cv detection failed for recipe {recipe}"
            )))
        }
    }
}

fn run_detect_arena_calibration(
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    let video = experiment
        .video
        .as_deref()
        .ok_or_else(|| BekiperError::Config("arena_calibration_cv requires a video file".into()))?;
    if experiment.task_type != bekiper_core::types::TaskType::OpenField {
        return Err(BekiperError::Config(
            "arena_calibration_cv is only valid for open_field tasks".into(),
        ));
    }
    let arena_side_meters = match &experiment.meters_per_pixel {
        bekiper_core::scale::ScaleConfig::Derived(
            bekiper_core::scale::ScaleMethod::ArenaSide { meters, .. },
        ) => *meters,
        _ => {
            return Err(BekiperError::Config(
                "arena_calibration_cv requires meters_per_pixel = { method = \"arena_side\", meters = ... }".into(),
            ));
        }
    };

    let calibration = experiment
        .detection
        .as_ref()
        .and_then(|detection| detection.arena_calibration.clone())
        .unwrap_or_default();

    let mut decoder = VideoDecoder::open(video)?;
    let info = decoder.info().clone();
    let frame_indices = arena_calibration_frame_indices(info.total_frames, &calibration);
    let mut gray_frames = Vec::new();
    let mut first_frame = None;
    for frame_idx in &frame_indices {
        let frame = decoder.frame_at(*frame_idx)?;
        if first_frame.is_none() {
            first_frame = Some(frame.clone());
        }
        gray_frames.push(frame.to_luma8());
    }

    if let Some(frame) = first_frame {
        let frame_path = exp_dir.detection_frame_png();
        std::fs::create_dir_all(frame_path.parent().unwrap())?;
        frame
            .save(&frame_path)
            .map_err(|e| BekiperError::Other(format!("failed to save detection frame: {e}")))?;
    }

    let result =
        bekiper_detect::arena::calibrate_arena(&gray_frames, arena_side_meters, &calibration)
            .map_err(|error| {
                BekiperError::Data(format!(
                    "arena_calibration_cv failed for {}: {error}",
                    experiment.name
                ))
            })?;

    write_perimeters(&result.perimeters, exp_dir)?;
    save_arena_calibration_debug(&result.debug, &frame_indices, &exp_dir.debug_dir());
    tracing::info!(
        backend = "arena_calibration_cv",
        count = result.perimeters.len(),
        "arena calibration complete"
    );
    Ok(())
}

fn arena_calibration_frame_indices(
    total_frames: u64,
    config: &bekiper_detect::config::ArenaCalibrationConfig,
) -> Vec<u64> {
    let last = total_frames.saturating_sub(1);
    let mut positions = if config.sample_positions.is_empty() {
        bekiper_detect::config::default_arena_sample_positions()
    } else {
        config.sample_positions.clone()
    };
    positions.sort_by(f64::total_cmp);
    positions.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON);
    positions
        .into_iter()
        .map(|position| (position.clamp(0.0, 1.0) * last as f64).round() as u64)
        .collect()
}

fn save_arena_calibration_debug(
    debug: &bekiper_detect::arena::ArenaCalibrationDebug,
    frame_indices: &[u64],
    target_dir: &Path,
) {
    if std::fs::create_dir_all(target_dir).is_err() {
        return;
    }
    let meta = serde_json::json!({
        "backend": "arena_calibration_cv",
        "frame_indices": frame_indices,
        "candidates": debug.candidates,
        "messages": debug.messages,
    });
    let _ = std::fs::write(
        target_dir.join("debug.toml"),
        json_value_to_toml_string(&meta).unwrap_or_default(),
    );
}

/// Save classical CV debug artifacts to a target directory.
fn save_classical_cv_debug(
    debug: &bekiper_detect::pipeline::DetectionDebug,
    frame: &image::DynamicImage,
    target_dir: &Path,
) {
    if std::fs::create_dir_all(target_dir).is_err() {
        return;
    }

    // Save threshold mask
    let _ = debug.threshold_image.save(target_dir.join("threshold.png"));

    // Save post-mouse-removal mask
    let _ = debug
        .post_mouse_image
        .save(target_dir.join("post_mouse.png"));

    // Save overlay: detection frame with candidate blobs outlined
    let mut overlay = frame.to_rgb8();
    for (i, blob) in debug.all_candidates.iter().enumerate() {
        let color = match i % 3 {
            0 => image::Rgb([255, 0, 0]),
            1 => image::Rgb([0, 255, 0]),
            _ => image::Rgb([0, 0, 255]),
        };
        // Draw blob boundary pixels
        let (w, h) = blob.mask.dimensions();
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                if blob.mask.get_pixel(x, y).0[0] == 255 {
                    let is_boundary = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                        .iter()
                        .any(|&(nx, ny)| blob.mask.get_pixel(nx, ny).0[0] == 0);
                    if is_boundary {
                        overlay.put_pixel(x, y, color);
                    }
                }
            }
        }
    }
    let _ = overlay.save(target_dir.join("candidates.png"));

    // Save debug metadata as TOML
    let meta = serde_json::json!({
        "backend": "classical_cv",
        "candidate_count": debug.all_candidates.len(),
        "candidates": debug.all_candidates.iter().enumerate().map(|(i, b)| {
            serde_json::json!({
                "index": i,
                "area": b.area,
                "bbox": [b.bbox.0, b.bbox.1, b.bbox.2, b.bbox.3],
                "centroid": [b.centroid.0, b.centroid.1],
            })
        }).collect::<Vec<_>>(),
    });
    let _ = std::fs::write(
        target_dir.join("debug.toml"),
        json_value_to_toml_string(&meta).unwrap_or_default(),
    );

    tracing::info!(path = %target_dir.display(), "saved classical CV debug artifacts");
}

fn run_detect_python(
    video: &Path,
    recipe: bekiper_core::types::DetectionRecipe,
    exp_dir: &ExperimentOutputDir,
    backend_name: &str,
) -> Result<()> {
    let python = resolve_python();
    let output_path = exp_dir.pixel_perimeters_toml();
    let frame_output = exp_dir.detection_frame_png();
    let debug_output = exp_dir.debug_dir();

    // Use middle frame to avoid experimenter's arm at start
    let decoder = VideoDecoder::open(video)?;
    let frame_idx = detection_frame_index(&decoder).to_string();
    drop(decoder);

    let output = std::process::Command::new(&python)
        .args([
            "-m",
            "bekiper_auto_detect",
            "--video",
            &video.to_string_lossy(),
            "--recipe",
            recipe.as_str(),
            "--backend",
            backend_name,
            "--output",
            &output_path.to_string_lossy(),
            "--frame",
            &frame_idx,
            "--frame-output",
            &frame_output.to_string_lossy(),
            "--debug-output",
            &debug_output.to_string_lossy(),
        ])
        .output()
        .map_err(|e| {
            BekiperError::Other(format!(
                "failed to run Python detection backend '{backend_name}': {e}"
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr_tail: String = stderr
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(BekiperError::Data(format!(
            "Python detection backend '{backend_name}' exited with {}: {}",
            output
                .status
                .code()
                .map_or("signal".into(), |c: i32| c.to_string()),
            stderr_tail,
        )));
    }

    tracing::info!(
        backend = backend_name,
        output = %output_path.display(),
        "detection complete"
    );
    Ok(())
}

fn write_perimeters(perimeters: &[PerimeterSpec], exp_dir: &ExperimentOutputDir) -> Result<()> {
    let output_path = exp_dir.pixel_perimeters_toml();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let pf = PerimetersFile {
        perimeters: perimeters.to_vec(),
    };
    let toml_str = toml::to_string_pretty(&pf).map_err(|e| BekiperError::Other(e.to_string()))?;
    std::fs::write(&output_path, toml_str)?;
    Ok(())
}
