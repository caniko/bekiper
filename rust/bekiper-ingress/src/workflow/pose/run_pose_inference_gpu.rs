/// GPU pose inference pipeline: decode → HIP upload/import → fused kernel → ORT.
///
/// Prefers Vulkan Video decode (RADV uses ACO, no LLVM conflict with MIGraphX).
/// For H.264 on the local 7900 XTX path, CPU decode is still allowed to feed
/// GPU preprocessing and MIGraphX inference.
fn run_pose_inference_gpu(
    model: &mut OnnxPoseModel,
    video: &Path,
    pose_cfg: &crate::config::PoseModelConfig,
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    use bekiper_gpu::{GpuFrame, GpuPreprocessor, NormParams};
    use bekiper_pose::infer::Normalization;

    // Open decoder (tries Vulkan first, then VA-API, then CPU).
    let mut decoder = match VideoDecoder::open(video) {
        Ok(d) => {
            let info = d.info();
            tracing::info!(
                hw_accel = info.hw_accel,
                backend = ?d.hw_backend(),
                width = info.width,
                height = info.height,
                "opened video decoder"
            );
            match select_gpu_pose_pipeline(d.hw_backend(), model.uses_gpu()) {
                GpuPosePipeline::VulkanDecode => {
                    tracing::info!("using Vulkan Video decode with GPU preprocessing/inference");
                }
                GpuPosePipeline::CpuDecode => {
                    tracing::info!("using CPU video decode with GPU preprocessing/inference");
                }
                GpuPosePipeline::CpuFallback { reason } => {
                    tracing::warn!(%reason, "falling back to full CPU pose pipeline");
                    return run_pose_inference(model, video, pose_cfg, experiment, exp_dir);
                }
            }
            d
        }
        Err(e) => {
            tracing::warn!(error = %e, "VideoDecoder::open failed, falling back to CPU pipeline");
            return run_pose_inference(model, video, pose_cfg, experiment, exp_dir);
        }
    };

    let hw_backend = decoder.hw_backend();
    tracing::info!(
        decode_backend = ?hw_backend,
        "using GPU pose inference pipeline"
    );

    let output_path = exp_dir.poses_parquet();
    let model_spec = model.spec().clone();

    // Convert model normalization to GPU preprocessor norm params
    let norm = match &model_spec.normalization {
        Normalization::ImageNet { mean, std } => NormParams {
            mean: *mean,
            std: *std,
        },
        Normalization::ZeroOne => NormParams::zero_one(),
    };

    // HIP must be initialized before any GPU API calls.
    // Device selection comes from the resolved pose model's GpuOptions,
    // which the ingress layer populated from `[gpu]` + env overrides.
    let gpu_device_id = model.gpu_options().device_id;
    bekiper_gpu::init(gpu_device_id).map_err(|e| {
        tracing::error!(error = %e, device_id = gpu_device_id, "HIP init failed");
        BekiperError::Inference(format!("HIP init (device {gpu_device_id}): {e}"))
    })?;
    tracing::info!(device_id = gpu_device_id, "HIP runtime initialized");

    let preprocessor = GpuPreprocessor::new(model_spec.input_width, model_spec.input_height, norm)
        .map_err(|e| {
            tracing::error!(error = %e, "GPU preprocessor init failed");
            BekiperError::Inference(format!("GPU preprocessor init: {e}"))
        })?;
    tracing::info!("GPU preprocessor created");

    let batch_size = if pose_cfg.batch_size == 0 {
        let fallback = model.gpu_options().fallback_batch_size;
        model.probe_batch_size(fallback)
    } else {
        pose_cfg.batch_size
    };

    // Chunk-based checkpointing (same as CPU path)
    let chunks_dir = exp_dir.pose_chunks_dir();
    std::fs::create_dir_all(&chunks_dir)?;

    let (resume_frame, mut chunk_index) = find_resume_point(&chunks_dir)?;

    if resume_frame > 0 {
        tracing::info!(
            resume_frame,
            existing_chunks = chunk_index,
            "resuming GPU pose inference from checkpoint"
        );
        decoder.seek_to(resume_frame)?;
    }

    let video_info = decoder.info();
    let original_size = (video_info.width, video_info.height);
    let total_frames = video_info.total_frames;

    tracing::info!(
        video = %video.display(),
        model = %pose_cfg.path.display(),
        spec = %pose_cfg.spec,
        batch_size,
        resume_frame,
        decode_backend = ?hw_backend,
        "starting GPU pose inference"
    );

    // The decoder may start as Vulkan or CPU. CPU decode is intentional for
    // H.264 on the local 7900 XTX path; batches are still uploaded to HIP for
    // preprocessing and MIGraphX inference.

    loop {
        // 1. Decode batch to host NV12, then upload to HIP.
        let mut nv12_frames = decoder.gpu_next_nv12_host_batch(batch_size)?;
        nv12_frames.retain(|frame| decoded_frame_within_total(frame.index, total_frames));
        if nv12_frames.is_empty() {
            break;
        }

        let actual_batch = nv12_frames.len();
        let inference_batch = gpu_inference_batch_size(actual_batch, batch_size);
        let first_frame = nv12_frames[0].index as i64;

        if inference_batch > actual_batch {
            tracing::info!(
                actual_batch,
                inference_batch,
                first_frame,
                "padding final GPU pose batch to fixed inference shape"
            );
        }

        let gpu_frames: Vec<GpuFrame> = (0..inference_batch)
            .map(|i| {
                let f = &nv12_frames[i.min(actual_batch - 1)];
                GpuFrame::from_nv12_host(&f.data, f.width, f.height, f.y_pitch)
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| {
                tracing::error!(error = %e, "NV12->HIP upload failed");
                BekiperError::Inference(format!("NV12->HIP upload: {e}"))
            })?;

        // 3. Fused GPU preprocess: NV12→RGB + resize + normalize → NCHW float32
        let gpu_refs: Vec<&GpuFrame> = gpu_frames.iter().collect();
        let tensor_buf = preprocessor
            .preprocess_nv12_batch(&gpu_refs)
            .map_err(|e| BekiperError::Inference(format!("GPU preprocess: {e}")))?;

        let tensor_ptr = tensor_buf.as_ptr();

        let sizes: Vec<(u32, u32)> = vec![original_size; inference_batch];

        // 5. Run inference via IoBinding (zero-copy input on GPU)
        // Safety: tensor_ptr is a valid HIP device pointer allocated by GpuPreprocessor
        // with the correct shape and size for the model's input.
        let mut batch_df = unsafe { model.infer_gpu_tensor(tensor_ptr, inference_batch, &sizes) }
            .map_err(|e| BekiperError::Inference(format!("GPU inference: {e}")))?;

        if inference_batch > actual_batch {
            batch_df = batch_df.slice(0, actual_batch);
        }

        // Adjust frame numbers
        if first_frame > 0 {
            let frame_col = batch_df.column("frame")?.i64()?.clone();
            let adjusted = &frame_col + first_frame;
            let _ = batch_df.with_column(adjusted.into_column());
        }

        // Write chunk
        let chunk_path = chunks_dir.join(format!("batch_{chunk_index:04}.parquet"));
        bekiper_pose::io::write_parquet(&mut batch_df, &chunk_path)?;
        chunk_index += 1;
    }

    // Merge all chunks into final poses.parquet
    merge_pose_chunks(&chunks_dir, &output_path)?;
    std::fs::remove_dir_all(&chunks_dir)?;

    tracing::info!(
        output = %output_path.display(),
        n_chunks = chunk_index,
        pipeline = "gpu-zero-copy",
        "GPU pose inference complete"
    );

    Ok(())
}

fn gpu_inference_batch_size(actual_batch: usize, configured_batch_size: usize) -> usize {
    if actual_batch == 0 {
        0
    } else {
        configured_batch_size.max(actual_batch)
    }
}

fn decoded_frame_within_total(frame_index: u64, total_frames: u64) -> bool {
    total_frames == 0 || frame_index < total_frames
}

/// Compute a crop region from annotation perimeters (pixel-space).
///
/// When perimeters define a bounded region of interest, cropping to their
/// bounding box focuses the pose model on the relevant area, reducing
/// background noise and wasted computation. Works for any task type with
/// spatial annotations (Y-maze, OFT arena, NORT arena, etc.).
pub(crate) fn compute_annotation_crop(
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
    model: &OnnxPoseModel,
) -> Result<Option<bekiper_pose::infer::CropRegion>> {
    use bekiper_core::inspection::PerimeterSpec;

    let Some(video_path) = experiment.video.as_ref() else {
        return Ok(None);
    };

    let perimeters = load_pixel_perimeters(exp_dir)?;
    if perimeters.is_empty() {
        return Ok(None);
    }

    // Collect all vertices from all perimeters to compute a global bounding box
    let mut all_points: Vec<(f64, f64)> = Vec::new();
    for p in &perimeters {
        match p {
            PerimeterSpec::RadialMaze {
                center_vertices,
                arms,
                ..
            } => {
                all_points.extend(center_vertices);
                for arm in arms {
                    all_points.extend(arm);
                }
            }
            PerimeterSpec::Circle {
                center_x,
                center_y,
                radius,
                ..
            } => {
                all_points.push((center_x - radius, center_y - radius));
                all_points.push((center_x + radius, center_y + radius));
            }
            PerimeterSpec::Rectangle {
                center_x,
                center_y,
                width,
                height,
                ..
            } => {
                all_points.push((center_x - width / 2.0, center_y - height / 2.0));
                all_points.push((center_x + width / 2.0, center_y + height / 2.0));
            }
            PerimeterSpec::Polygon { vertices, .. } => {
                all_points.extend(vertices);
            }
            PerimeterSpec::Triangle { vertices, .. } => {
                all_points.extend(vertices);
            }
        }
    }

    if all_points.is_empty() {
        return Ok(None);
    }

    let min_x = all_points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let min_y = all_points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_x = all_points
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_y = all_points
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    // Add 10% padding around the annotation bounding box
    let pad_x = (max_x - min_x) * 0.10;
    let pad_y = (max_y - min_y) * 0.10;
    let padding = pad_x.max(pad_y);

    // We need the full-frame video dimensions to clamp the crop.
    // Probe the video to get source resolution.
    let info = bekiper_pose::video::probe_video_info(video_path)
        .map_err(|e| BekiperError::Data(format!("probe video info: {e}")))?;

    let crop = bekiper_pose::infer::CropRegion::from_bbox(
        min_x,
        min_y,
        max_x,
        max_y,
        padding,
        info.width,
        info.height,
    );

    let crop_area = crop.width as f64 * crop.height as f64;
    let full_area = info.width as f64 * info.height as f64;
    let reduction_pct = (1.0 - crop_area / full_area) * 100.0;

    tracing::info!(
        crop_x = crop.x,
        crop_y = crop.y,
        crop_w = crop.width,
        crop_h = crop.height,
        frame_w = info.width,
        frame_h = info.height,
        reduction_pct = format!("{reduction_pct:.1}%"),
        "annotation crop: {:.0}% of frame excluded",
        reduction_pct,
    );

    // Only crop if it removes at least 20% of the frame area
    if reduction_pct < 20.0 {
        tracing::info!("crop too small ({reduction_pct:.0}% reduction), using full frame");
        return Ok(None);
    }

    // Suppress unused variable warning — model is available for future use
    let _ = model;

    Ok(Some(crop))
}

/// Add crop offset to all `_x` and `_y` keypoint columns in a DataFrame.
pub(super) fn offset_keypoint_columns(
    df: &mut polars::frame::DataFrame,
    offset_x: f64,
    offset_y: f64,
) -> Result<()> {
    use polars::prelude::*;

    let col_names: Vec<String> = df
        .get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    for name in &col_names {
        if name.ends_with("_x") && name != "frame" {
            let col = df.column(name)?.f64()?.clone();
            let shifted = &col + offset_x;
            let _ = df.with_column(shifted.into_column());
        } else if name.ends_with("_y") {
            let col = df.column(name)?.f64()?.clone();
            let shifted = &col + offset_y;
            let _ = df.with_column(shifted.into_column());
        }
    }
    Ok(())
}

/// Resolve a model spec from a `PoseModelPreset`.
pub(crate) fn resolve_model_spec(preset: &PoseModelPreset) -> Result<ModelSpec> {
    match preset {
        PoseModelPreset::DlcSuperAnimal => Ok(ModelSpec::dlc_super_animal()),
        PoseModelPreset::SleapSingleInstance => Ok(ModelSpec::sleap_single_instance()),
        PoseModelPreset::LightningPose => Ok(ModelSpec::lightning_pose()),
        PoseModelPreset::Custom(path) => {
            let content = std::fs::read_to_string(path).map_err(|e| {
                BekiperError::Config(format!(
                    "cannot read model spec file {}: {e}",
                    path.display()
                ))
            })?;
            let spec: ModelSpec = serde_json::from_str(&content)
                .map_err(|e| BekiperError::Config(format!("invalid model spec JSON: {e}")))?;
            spec.validate().map_err(|e| {
                BekiperError::Config(format!("invalid model spec {}: {e}", path.display()))
            })?;
            Ok(spec)
        }
    }
}

// ---------------------------------------------------------------------------
// Pose chunk helpers: streaming checkpoint + merge
// ---------------------------------------------------------------------------

/// Scan the chunks directory for existing `batch_NNNN.parquet` files.
/// Returns `(next_frame, chunk_count)` — the frame number to resume from
/// and how many chunks already exist.
fn find_resume_point(chunks_dir: &Path) -> Result<(u64, u32)> {
    let mut chunk_paths: Vec<PathBuf> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(chunks_dir) {
        for entry in entries {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "parquet")
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("batch_"))
            {
                chunk_paths.push(path);
            }
        }
    }

    if chunk_paths.is_empty() {
        return Ok((0, 0));
    }

    chunk_paths.sort();
    let chunk_count = chunk_paths.len() as u32;

    // Read the last chunk to find the highest frame number
    let last_chunk = &chunk_paths[chunk_paths.len() - 1];
    let df = polars::prelude::LazyFrame::scan_parquet(
        last_chunk.to_str().unwrap().into(),
        Default::default(),
    )
    .map_err(|e| BekiperError::Data(format!("scan chunk: {e}")))?
    .select([polars::prelude::col("frame")])
    .collect()
    .map_err(|e| BekiperError::Data(format!("collect chunk: {e}")))?;

    let max_frame = df
        .column("frame")
        .ok()
        .and_then(|c| c.i64().ok())
        .and_then(|c| c.into_iter().flatten().max())
        .unwrap_or(0);

    // Resume from the frame after the last one we have
    Ok(((max_frame + 1) as u64, chunk_count))
}
