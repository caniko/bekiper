//! Direct-source analysis. Kernels share Polars buffers; persistence is an
//! output boundary rather than a prerequisite for the next algorithm.

pub(crate) mod augment;
pub(crate) mod behaviour;
pub(crate) mod effective_parameters;
pub(crate) mod frame_keys;
#[cfg(feature = "pose-gpu")]
pub(crate) mod helpers;
pub mod staged;
pub(crate) mod summary;
#[cfg(all(test, feature = "pose-gpu"))]
mod tests;

/// Column lists are derived once from the source label order.
pub(crate) fn label_columns(labels: &[String]) -> (Vec<String>, Vec<String>, Vec<String>) {
    let x = labels.iter().map(|l| format!("{l}_x")).collect();
    let y = labels.iter().map(|l| format!("{l}_y")).collect();
    let likelihood = labels.iter().map(|l| format!("{l}_likelihood")).collect();
    (x, y, likelihood)
}

pub fn is_analysis_excluded(experiment: &crate::config::ExperimentConfig) -> bool {
    crate::analysis_exclusion::is_analysis_excluded_experiment(experiment)
}

pub fn valid_output_component(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

use crate::aggregate::TrialResult;
use crate::config::{ExperimentConfig, IngressConfig};
use crate::pose_import::{PoseImportOptions, load_normalized_pose};
use crate::stage::ExperimentOutputDir;
use bekiper_behaviour::{AnalysisResult, TaskDetail};
use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::{InspectionManifest, PerimeterSpec};
use bekiper_core::scale::ScaleDerivationInfo;
use bekiper_core::types::MetersPerPixel;
use polars::prelude::{DataFrame, IntoLazy};
use std::path::Path;

/// Source dimensions are shared by analysis transforms and display projection.
/// A retained source image also supports precomputed trials without a video.
pub(crate) fn resolve_source_resolution(
    experiment: &ExperimentConfig,
) -> Result<Option<(u32, u32)>> {
    let resolution = if let Some(resolution) = experiment.video_resolution {
        Some(resolution)
    } else if let Some(path) = experiment.frame_image.as_deref() {
        Some(image::image_dimensions(path).map_err(|error| {
            BekiperError::Data(format!("cannot read source image dimensions: {error}"))
        })?)
    } else {
        #[cfg(feature = "pose-video")]
        if let Some(video) = &experiment.video {
            let info = bekiper_pose::video::probe_video_info(video)?;
            return Ok(Some((info.width, info.height)));
        }
        None
    };
    if resolution.is_some_and(|(width, height)| width == 0 || height == 0) {
        return Err(BekiperError::Config(
            "source resolution must be positive".into(),
        ));
    }
    Ok(resolution)
}

/// Appends the configured `frame_centered` arena and center zone (pixel space)
/// unless reviewed outlines already exist. Returns the scale that the trial
/// must use when it changes: a detected box defines its own video's scale; a
/// frame-center fallback keeps `scale` and records the fallback in its method.
pub fn append_generated_arena(
    experiment: &ExperimentConfig,
    perimeters: &mut Vec<PerimeterSpec>,
    scale: Option<&ScaleDerivationInfo>,
) -> Result<Option<ScaleDerivationInfo>> {
    let calibration = experiment
        .detection
        .as_ref()
        .and_then(|detection| detection.arena_calibration.clone())
        .unwrap_or_default();
    let rule = calibration.effective_center_rule();
    let bekiper_detect::config::ArenaCenterRule::FrameCentered { detect_box, .. } = rule else {
        return Ok(None);
    };
    let present =
        ["arena", "center_zone"].map(|label| perimeters.iter().any(|p| p.label() == label));
    match present {
        [true, true] => return Ok(None),
        [false, false] => {}
        _ => {
            return Err(BekiperError::Config(format!(
                "{}: frame_centered geometry needs both or neither of the reviewed arena/center_zone outlines",
                experiment.name
            )));
        }
    }
    // Each video is resolved in its own pixel space; frame sizes differ.
    let frame = resolve_source_resolution(experiment)?.ok_or_else(|| {
        BekiperError::Config(format!(
            "{}: frame_centered geometry needs the video frame size",
            experiment.name
        ))
    })?;
    let frames = if detect_box {
        sample_box_frames(experiment, &calibration)?
    } else {
        Vec::new()
    };
    let generated = bekiper_detect::arena::generate_arena(
        &rule,
        &calibration,
        frame,
        &frames,
        scale.map(|scale| scale.meters_per_pixel),
    )
    .map_err(|error| BekiperError::Config(format!("{}: {error}", experiment.name)))?
    .expect("frame_centered rule always generates geometry");
    let pixel_width = generated.perimeters.first().map(|arena| match arena {
        PerimeterSpec::Rectangle { width, .. } => *width,
        _ => f64::NAN, // coverage: unreachable — generated arenas are rectangles
    });
    perimeters.extend(generated.perimeters);
    Ok(Some(match (generated.fallback_reason, scale) {
        (None, _) => ScaleDerivationInfo {
            method: "arena_box_detected".into(),
            physical_meters: pixel_width.map(|width| width * generated.meters_per_pixel),
            pixel_measurement: pixel_width,
            arm_pixel_spans: Vec::new(),
            reference_label: Some("arena".into()),
            endpoints: None,
            meters_per_pixel: generated.meters_per_pixel,
            // Square boxes keep the isotropic (byte-identical) form.
            meters_per_pixel_y: (generated.meters_per_pixel_y != generated.meters_per_pixel)
                .then_some(generated.meters_per_pixel_y),
        },
        (Some(reason), Some(scale)) => {
            tracing::warn!(experiment = %experiment.name, %reason, "NORT arena centered on the video frame");
            ScaleDerivationInfo {
                method: format!("{}+frame_center_fallback", scale.method),
                ..scale.clone()
            }
        }
        (Some(_), None) => unreachable!("fallback geometry requires a scale"), // coverage: unreachable — generate_arena errors without a scale
    }))
}

#[cfg(feature = "pose-video")]
fn sample_box_frames(
    experiment: &ExperimentConfig,
    calibration: &bekiper_detect::config::ArenaCalibrationConfig,
) -> Result<Vec<image::GrayImage>> {
    let Some(video) = experiment.video.as_deref() else {
        // No video to inspect: the generator falls back to the frame center.
        return Ok(Vec::new());
    };
    Ok(sample_arena_frames(video, calibration)?.0)
}

#[cfg(not(feature = "pose-video"))]
fn sample_box_frames(
    _experiment: &ExperimentConfig,
    _calibration: &bekiper_detect::config::ArenaCalibrationConfig,
) -> Result<Vec<image::GrayImage>> {
    Ok(Vec::new())
}

/// Frames remain in memory until final export. Cloning a DataFrame does not
/// duplicate its Arrow buffers; transforms allocate their changed columns.
pub struct ExperimentAnalysis {
    pub raw: DataFrame,
    pub augmented: DataFrame,
    pub result: AnalysisResult,
    pub detailed: Option<TaskDetail>,
    pub manifest: InspectionManifest,
    pub trial: TrialResult,
}

impl ExperimentAnalysis {
    /// Label `effective_parameters` keys set through `--set`/`BEKIPER_SET`
    /// (`config.override_record`) as `cli`/`env`.
    pub fn mark_override_sources(&mut self, record: &[crate::config::AppliedOverride]) {
        if let Some(parameters) = self.trial.detailed.get_mut("effective_parameters") {
            effective_parameters::apply_override_sources(
                parameters,
                self.trial.task_type,
                &self.trial.experiment,
                record,
            );
        }
    }
}

/// Analyze a precomputed source directly. Geometry/calibration are supplied by
/// the registry's frozen source snapshot, never resolved from old checkpoints.
pub fn analyze_experiment(
    experiment: &ExperimentConfig,
    config: &IngressConfig,
    pixel_perimeters: &[PerimeterSpec],
    scale: &ScaleDerivationInfo,
) -> Result<ExperimentAnalysis> {
    if !valid_output_component(&experiment.name) {
        return Err(BekiperError::Config("unsafe experiment output name".into()));
    }
    let [source] = experiment.data_files.as_slice() else {
        return Err(BekiperError::Config(
            "direct analysis requires exactly one precomputed pose source per experiment".into(),
        ));
    };
    let (raw, imported) = load_normalized_pose(source, PoseImportOptions::default())?;
    let animal = experiment
        .animal_id
        .or_else(|| experiment.name.split('_').next()?.parse().ok())
        .and_then(|id| config.animal_metadata.get(&id));
    let mut analysis = analyze_loaded_pose(
        experiment,
        &config.pipeline,
        animal,
        pixel_perimeters,
        scale,
        raw.collect()?,
        &imported.labels,
    )?;
    analysis.mark_override_sources(&config.override_record);
    Ok(analysis)
}

/// Analyze an already-decoded trial. A source worker reuses its Arrow buffers
/// for every analysis variant, retaining whole-trial temporal semantics.
pub fn analyze_loaded_pose(
    experiment: &ExperimentConfig,
    pipeline: &crate::config::PipelineConfig,
    animal: Option<&crate::config::AnimalMetadata>,
    pixel_perimeters: &[PerimeterSpec],
    scale: &ScaleDerivationInfo,
    raw: DataFrame,
    source_labels: &[String],
) -> Result<ExperimentAnalysis> {
    if !valid_output_component(&experiment.name) {
        return Err(BekiperError::Config("unsafe experiment output name".into()));
    }
    if !scale.meters_per_pixel.is_finite() || scale.meters_per_pixel <= 0.0 {
        return Err(BekiperError::Config(
            "analysis scale must be finite and positive".into(),
        ));
    }
    if raw.height() == 0 {
        return Err(BekiperError::Data("pose input is empty".into()));
    }
    let augmented = augment_loaded_pose(
        experiment,
        pipeline,
        source_labels,
        pixel_perimeters,
        scale,
        &raw,
    )?;
    analyze_augmented_pose_with_pipeline(
        experiment,
        pipeline,
        animal,
        pixel_perimeters,
        scale,
        raw,
        augmented,
    )
}

/// Whole-trial preparation can be shared by variants with identical timing,
/// geometry, calibration and pipeline inputs.
pub fn augment_loaded_pose(
    experiment: &ExperimentConfig,
    pipeline: &crate::config::PipelineConfig,
    source_labels: &[String],
    pixel_perimeters: &[PerimeterSpec],
    scale: &ScaleDerivationInfo,
    raw: &DataFrame,
) -> Result<DataFrame> {
    if !scale.meters_per_pixel.is_finite() || scale.meters_per_pixel <= 0.0 {
        return Err(BekiperError::Config(
            "analysis scale must be finite and positive".into(),
        ));
    }
    augment::augment_pose(
        raw.clone().lazy(),
        experiment,
        pipeline,
        source_labels,
        pixel_perimeters,
        MetersPerPixel(scale.meters_per_pixel),
        scale.meters_per_pixel_y,
    )?
    .collect()
    .map_err(Into::into)
}

pub fn analyze_augmented_pose(
    experiment: &ExperimentConfig,
    animal: Option<&crate::config::AnimalMetadata>,
    pixel_perimeters: &[PerimeterSpec],
    scale: &ScaleDerivationInfo,
    raw: DataFrame,
    augmented: DataFrame,
) -> Result<ExperimentAnalysis> {
    analyze_augmented_pose_with_pipeline(
        experiment,
        &Default::default(),
        animal,
        pixel_perimeters,
        scale,
        raw,
        augmented,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn analyze_augmented_pose_with_pipeline(
    experiment: &ExperimentConfig,
    pipeline: &crate::config::PipelineConfig,
    animal: Option<&crate::config::AnimalMetadata>,
    pixel_perimeters: &[PerimeterSpec],
    scale: &ScaleDerivationInfo,
    raw: DataFrame,
    augmented: DataFrame,
) -> Result<ExperimentAnalysis> {
    if !valid_output_component(&experiment.name) {
        return Err(BekiperError::Config("unsafe experiment output name".into()));
    }
    if augmented.height() == 0 {
        return Err(BekiperError::Data(
            "augmentation removed every frame".into(),
        ));
    }
    let perimeters = metric_perimeters(experiment, pipeline, pixel_perimeters, scale)?;
    let (mut result, detailed, quadrants) =
        behaviour::analyze_pose(experiment, &perimeters, augmented.clone().lazy())?;
    summary::validate_ymaze_result(
        experiment,
        detailed.as_ref(),
        &perimeters,
        result.evaluated_df.as_ref(),
    )?;
    let (manifest, trial) = summary::assemble_trial(
        experiment,
        animal,
        &mut result,
        detailed.as_ref(),
        &perimeters,
        &quadrants,
        Some(scale.clone()),
        (raw.height() > 0).then_some(&raw),
        Some(&augmented),
        pipeline,
    )?;
    Ok(ExperimentAnalysis {
        raw,
        augmented,
        result,
        detailed,
        manifest,
        trial,
    })
}

/// Preparation-only projection. Task-derived zones belong to scoring and may
/// change independently of these reusable metric coordinates.
pub fn metric_perimeters(
    experiment: &ExperimentConfig,
    pipeline: &crate::config::PipelineConfig,
    pixel_perimeters: &[PerimeterSpec],
    scale: &ScaleDerivationInfo,
) -> Result<Vec<PerimeterSpec>> {
    if !scale.meters_per_pixel.is_finite() || scale.meters_per_pixel <= 0.0 {
        return Err(BekiperError::Config(
            "analysis scale must be finite and positive".into(),
        ));
    }
    let source_height = if pipeline.invert_y {
        resolve_source_resolution(experiment)?
            .map(|(_, height)| height as f64)
            .ok_or_else(|| BekiperError::Config("Y inversion needs source resolution".into()))?
    } else {
        0.0
    };
    Ok(pixel_perimeters
        .iter()
        .map(|p| {
            let metric = p.scale_xy(scale.meters_per_pixel, scale.meters_per_pixel_y());
            if pipeline.invert_y {
                metric.reflect_y(source_height * scale.meters_per_pixel_y())
            } else {
                metric
            }
        })
        .collect())
}

/// Resolve source video timing and configured OFT calibration on a blocking
/// worker, after the registry snapshot lock has been released.
pub fn arena_calibration_sample_count(experiment: &ExperimentConfig) -> usize {
    experiment
        .detection
        .as_ref()
        .and_then(|d| d.arena_calibration.as_ref())
        .map(|c| c.sample_positions.len())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| bekiper_detect::config::default_arena_sample_positions().len())
}

#[cfg(feature = "pose-video")]
pub fn prepare_video_inputs(
    experiment: &mut ExperimentConfig,
    perimeters: &mut Vec<PerimeterSpec>,
    scale: &mut Option<ScaleDerivationInfo>,
) -> Result<()> {
    use bekiper_core::video::FpsConfig;
    let Some(video) = &experiment.video else {
        return Ok(());
    };
    let info = bekiper_pose::video::probe_video_info(video)?;
    experiment.video_resolution = Some((info.width, info.height));
    if matches!(experiment.fps, FpsConfig::FromVideo) && experiment.frame_timing.is_none() {
        experiment.frame_timing = Some(bekiper_pose::video::extract_frame_timestamps(video)?);
    }
    if perimeters.is_empty() {
        let (result, _, _) = calibrate_arena_source(experiment)?;
        *perimeters = result.perimeters;
    }
    if let Some(generated) = append_generated_arena(experiment, perimeters, scale.as_ref())? {
        *scale = Some(generated);
    }
    Ok(())
}

#[cfg(feature = "pose-video")]
pub fn calibrate_arena_source(
    experiment: &ExperimentConfig,
) -> Result<(
    bekiper_detect::arena::ArenaCalibrationResult,
    Vec<u64>,
    Option<image::DynamicImage>,
)> {
    let video = experiment
        .video
        .as_deref()
        .ok_or_else(|| BekiperError::Config("arena calibration requires video".into()))?;
    if experiment.task_type != bekiper_core::types::TaskType::OpenField {
        return Err(BekiperError::Config(
            "arena calibration requires open_field".into(),
        ));
    }
    let meters = match &experiment.meters_per_pixel {
        bekiper_core::scale::ScaleConfig::Derived(
            bekiper_core::scale::ScaleMethod::ArenaSide { meters, .. },
        ) => *meters,
        _ => {
            return Err(BekiperError::Config(
                "arena calibration requires arena_side scale".into(),
            ));
        }
    };
    let calibration = experiment
        .detection
        .as_ref()
        .and_then(|d| d.arena_calibration.clone())
        .unwrap_or_default();
    let (frames, indices, first) = sample_arena_frames(video, &calibration)?;
    let result = bekiper_detect::arena::calibrate_arena(&frames, meters, &calibration)
        .map_err(|e| BekiperError::Data(format!("arena calibration: {e}")))?;
    Ok((result, indices, first))
}

/// Grayscale frames at the calibration's sample positions, plus the indices
/// and the first decoded frame for evidence.
#[cfg(feature = "pose-video")]
fn sample_arena_frames(
    video: &Path,
    calibration: &bekiper_detect::config::ArenaCalibrationConfig,
) -> Result<(Vec<image::GrayImage>, Vec<u64>, Option<image::DynamicImage>)> {
    let mut decoder = bekiper_pose::video::VideoDecoder::open(video)?;
    let indices = arena_calibration_frame_indices(decoder.info().total_frames, calibration);
    let mut frames = Vec::new();
    let mut first = None;
    for index in &indices {
        let frame = decoder.frame_at(*index)?;
        if first.is_none() {
            first = Some(frame.clone());
        }
        frames.push(frame.to_luma8());
    }
    Ok((frames, indices, first))
}

#[cfg(feature = "pose-video")]
pub(crate) fn arena_calibration_frame_indices(
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
        .map(|p| (p.clamp(0.0, 1.0) * last as f64).round() as u64)
        .collect()
}

/// Persist only final output artifacts. No intermediate frame is reread.
pub fn export_analysis(
    output_root: &Path,
    experiment: &ExperimentConfig,
    analysis: ExperimentAnalysis,
) -> Result<TrialResult> {
    let dir = match experiment.animal_id {
        Some(id) => ExperimentOutputDir::for_animal(output_root, id, &experiment.name),
        None => ExperimentOutputDir::new(output_root, &experiment.name),
    };
    dir.ensure_dir()?;
    crate::export::export_inspection(
        dir.root(),
        &experiment.name,
        &analysis.result,
        &analysis.manifest,
    )?;
    let content = bekiper_core::inspection::json_value_to_toml_string(
        &serde_json::to_value(&analysis.trial).map_err(|e| BekiperError::Other(e.to_string()))?,
    )
    .map_err(|e| BekiperError::Other(e.to_string()))?;
    std::fs::write(dir.result_toml(), content)?;
    let mut frames = vec![("augmented", analysis.augmented)];
    if analysis.raw.height() > 0 {
        frames.insert(0, ("poses", analysis.raw));
    }
    if let Some(df) = analysis.result.evaluated_df {
        frames.push(("behaviour", df));
    }
    let animal = serde_json::from_value::<crate::config::AnimalMetadata>(
        analysis.trial.animal_metadata.clone(),
    )
    .ok();
    crate::export::export_spreadsheet_frames_with_parameters(
        &dir,
        &experiment.name,
        &analysis.result.summaries,
        analysis.detailed.as_ref(),
        animal.as_ref(),
        &frames,
        analysis.trial.detailed.get("effective_parameters"),
    )?;
    #[cfg(feature = "pose-video")]
    if let Some(video) = &experiment.video {
        summary::extract_representative_frames(
            video,
            &analysis.manifest.feature_inspections,
            frames
                .iter()
                .find(|(stage, _)| *stage == "behaviour")
                .map(|(_, df)| df),
            &dir,
        )?;
    }
    Ok(analysis.trial)
}

/// Normal source-stream completion includes selected presentation artifacts.
/// The worker holds its native memory lease through rendering and cancellation.
pub fn export_analysis_with_inspection(
    output_root: &Path,
    experiment: &ExperimentConfig,
    analysis: ExperimentAnalysis,
    settings: &crate::config::InspectConfig,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<TrialResult> {
    let dir = match experiment.animal_id {
        Some(id) => ExperimentOutputDir::for_animal(output_root, id, &experiment.name),
        None => ExperimentOutputDir::new(output_root, &experiment.name),
    };
    crate::inspection::capture_context(&dir, experiment, settings)?;
    let trial = export_analysis(output_root, experiment, analysis)?;
    crate::inspection::render(&dir, experiment, settings, cancelled)?;
    Ok(trial)
}
