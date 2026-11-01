use std::path::{Path, PathBuf};

use polars::prelude::IntoColumn;

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::types::PoseModelPreset;
use bekiper_pose::infer::{ModelSpec, OnnxPoseModel};
#[cfg(test)]
use bekiper_pose::io::scan_file;
use bekiper_pose::video::{HwBackend, VideoDecoder};

use crate::config::ExperimentConfig;
use crate::pose_import::{PoseImportOptions, import_pose_file_to_parquet};
use crate::stage::ExperimentOutputDir;

use super::helpers::load_pixel_perimeters;

// ---------------------------------------------------------------------------
// Stage 1b: Pose estimation — read pre-computed or infer from video
// ---------------------------------------------------------------------------

/// Run pose estimation using a pre-loaded shared model.
///
/// If the experiment has pre-computed data files, reads them directly (no model needed).
/// Otherwise, borrows the shared model for GPU inference.
/// Run pose with shared model, preferring the zero-copy GPU pipeline.
pub(super) fn run_pose_with_model_gpu(
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
    shared_model: &mut Option<OnnxPoseModel>,
) -> Result<()> {
    let output_path = exp_dir.poses_parquet();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Pre-computed data: just read and write, no model needed.
    if !experiment.data_files.is_empty() {
        let data_file = &experiment.data_files[0];
        load_precomputed_pose_data(data_file, &output_path)?;
        return Ok(());
    }

    if !experiment.needs_pose_inference() {
        return Err(BekiperError::Config(
            "experiment has no data_files and no video+pose_model configured".into(),
        ));
    }

    let video = experiment
        .video
        .as_ref()
        .ok_or_else(|| BekiperError::Config("pose inference requires a video file".into()))?;

    let pose_cfg = experiment.pose_model.as_ref().ok_or_else(|| {
        BekiperError::Config("pose inference requires a pose_model config".into())
    })?;

    let model = shared_model.as_mut().ok_or_else(|| {
        BekiperError::Config("no shared pose model available for inference".into())
    })?;

    // Try GPU zero-copy pipeline (Vulkan Video decode + MIGraphX inference).
    // Vulkan uses RADV/ACO (no LLVM), so no conflict with MIGraphX's LLVM.
    // Falls back to CPU decode if Vulkan is unavailable.
    run_pose_inference_gpu(model, video, pose_cfg, experiment, exp_dir)
}

/// Run pose estimation, loading its own model (standalone mode).
#[allow(dead_code)]
pub(super) fn run_pose(
    experiment: &ExperimentConfig,
    gpu_cfg: &crate::config::GpuConfig,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    let output_path = exp_dir.poses_parquet();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if !experiment.data_files.is_empty() {
        let data_file = &experiment.data_files[0];
        load_precomputed_pose_data(data_file, &output_path)?;
        return Ok(());
    }

    if !experiment.needs_pose_inference() {
        return Err(BekiperError::Config(
            "experiment has no data_files and no video+pose_model configured".into(),
        ));
    }

    let video = experiment
        .video
        .as_ref()
        .ok_or_else(|| BekiperError::Config("pose inference requires a video file".into()))?;

    let pose_cfg = experiment.pose_model.as_ref().ok_or_else(|| {
        BekiperError::Config("pose inference requires a pose_model config".into())
    })?;

    let spec = resolve_model_spec(&pose_cfg.spec)?;
    let mut model =
        OnnxPoseModel::load_with_options(&pose_cfg.path, spec, gpu_cfg.to_pose_options())?;

    run_pose_inference(&mut model, video, pose_cfg, experiment, exp_dir)
}

/// Core CPU pose inference loop.
fn run_pose_inference(
    model: &mut OnnxPoseModel,
    video: &Path,
    pose_cfg: &crate::config::PoseModelConfig,
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    let output_path = exp_dir.poses_parquet();

    // Check if we can crop to annotation ROI (e.g. Y-maze).
    let crop_region = compute_annotation_crop(experiment, exp_dir, model)?;

    let mut decoder = if crop_region.is_some() {
        VideoDecoder::open(video)?
    } else {
        let model_spec = model.spec();
        VideoDecoder::open_with_resize(video, model_spec.input_width, model_spec.input_height)?
    };

    let batch_size = if pose_cfg.batch_size == 0 {
        let fallback = model.gpu_options().fallback_batch_size;
        model.probe_batch_size(fallback)
    } else {
        pose_cfg.batch_size
    };

    // --- Chunk-based checkpointing: resume from last chunk on crash ---
    let chunks_dir = exp_dir.pose_chunks_dir();
    std::fs::create_dir_all(&chunks_dir)?;

    let (resume_frame, mut chunk_index) = find_resume_point(&chunks_dir)?;

    if resume_frame > 0 {
        tracing::info!(
            resume_frame,
            existing_chunks = chunk_index,
            "resuming pose inference from checkpoint"
        );
        decoder.seek_to(resume_frame)?;
    }

    tracing::info!(
        video = %video.display(),
        model = %pose_cfg.path.display(),
        spec = %pose_cfg.spec,
        batch_size,
        resume_frame,
        crop = ?crop_region.map(|c| format!("{}x{}+{}+{}", c.width, c.height, c.x, c.y)),
        "starting pose inference"
    );

    // 3-stage pipeline: while GPU infers batch N, main thread decodes AND
    // preprocesses batch N+1. This keeps the GPU fed continuously.
    let model_spec = model.spec().clone();

    let video_info = decoder.info();
    let original_size = if let Some(crop) = crop_region {
        (crop.width, crop.height)
    } else {
        (video_info.width, video_info.height)
    };

    // Bootstrap: decode + preprocess first batch
    let mut pending_batch = decoder.next_batch(batch_size)?;
    let preprocess_fn = |batch: &[(u64, image::DynamicImage)]| {
        let images: Vec<_> = batch
            .iter()
            .map(|(_, img)| {
                if let Some(crop) = crop_region {
                    crop.crop(img)
                } else {
                    img.clone()
                }
            })
            .collect();
        let first_frame = batch[0].0 as i64;
        let (tensor, _) = bekiper_pose::infer::preprocess_batch(&images, &model_spec);
        let sizes: Vec<(u32, u32)> = vec![original_size; images.len()];
        (tensor, sizes, first_frame)
    };

    let mut pending_tensor = if !pending_batch.is_empty() {
        Some(preprocess_fn(&pending_batch))
    } else {
        None
    };

    while let Some((tensor, sizes, first_frame)) = pending_tensor.take() {
        let mut infer_result = None;
        std::thread::scope(|s| -> Result<()> {
            // GPU: infer the already-preprocessed tensor
            let infer_handle = s.spawn(|| model.infer_preprocessed(tensor, &sizes));

            // Main thread: decode next batch + preprocess it (overlaps with GPU)
            pending_batch = decoder.next_batch(batch_size)?;
            if !pending_batch.is_empty() {
                pending_tensor = Some(preprocess_fn(&pending_batch));
            }

            // Wait for GPU inference
            let mut batch_df = infer_handle
                .join()
                .map_err(|_| BekiperError::Data("inference thread panicked".into()))??;

            if first_frame > 0 {
                let frame_col = batch_df.column("frame")?.i64()?.clone();
                let adjusted = &frame_col + first_frame;
                let _ = batch_df.with_column(adjusted.into_column());
            }

            // When cropping, offset keypoint coordinates back to full-frame pixel space
            if let Some(crop) = crop_region {
                offset_keypoint_columns(&mut batch_df, crop.x as f64, crop.y as f64)?;
            }

            infer_result = Some(batch_df);
            Ok(())
        })?;

        if let Some(mut df) = infer_result {
            let chunk_path = chunks_dir.join(format!("batch_{chunk_index:04}.parquet"));
            bekiper_pose::io::write_parquet(&mut df, &chunk_path)?;
            chunk_index += 1;
        }
    }

    // Merge all chunks into final poses.parquet
    merge_pose_chunks(&chunks_dir, &output_path)?;

    // Clean up chunk directory
    std::fs::remove_dir_all(&chunks_dir)?;

    tracing::info!(
        output = %output_path.display(),
        n_chunks = chunk_index,
        "pose inference complete"
    );

    Ok(())
}

fn load_precomputed_pose_data(data_file: &Path, output_path: &Path) -> Result<()> {
    let report = import_pose_file_to_parquet(data_file, output_path, PoseImportOptions::default())?;
    tracing::info!(
        src = %data_file.display(),
        output = %output_path.display(),
        format = ?report.format,
        backend = ?report.backend,
        labels = ?report.labels,
        "read pre-computed poses"
    );

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuPosePipeline {
    /// Vulkan Video decode plus GPU preprocessing and inference.
    VulkanDecode,
    /// CPU decode plus GPU preprocessing and inference.
    CpuDecode,
    /// Use the existing full CPU path instead.
    CpuFallback { reason: &'static str },
}

fn select_gpu_pose_pipeline(hw_backend: HwBackend, model_uses_gpu: bool) -> GpuPosePipeline {
    if !model_uses_gpu {
        return GpuPosePipeline::CpuFallback {
            reason: "pose model is using CPU inference",
        };
    }

    match hw_backend {
        HwBackend::Vulkan => GpuPosePipeline::VulkanDecode,
        HwBackend::None => GpuPosePipeline::CpuDecode,
        HwBackend::VaApi => GpuPosePipeline::CpuFallback {
            reason: "VA-API decode plus MIGraphX inference would crash due to LLVM conflict",
        },
    }
}
