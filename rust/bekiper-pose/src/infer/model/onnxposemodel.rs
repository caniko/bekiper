use super::*;
use std::path::Path;

use image::DynamicImage;
use ort::session::Session;
use polars::prelude::*;

use bekiper_core::error::{BekiperError, Result};

use super::super::preprocess::{preprocess_batch, preprocess_frame};
use super::super::spec::{Keypoint, ModelSpec};
use super::decode_output::{decode_dataframe, decode_output, read_vram_usage};

impl OnnxPoseModel {
    /// Load an ONNX model with default GPU options. Tests and callers that
    /// have no config can use this; production code should prefer
    /// [`OnnxPoseModel::load_with_options`] and pass a fully resolved
    /// [`GpuOptions`].
    pub fn load(model_path: &Path, spec: ModelSpec) -> Result<Self> {
        Self::load_with_options(model_path, spec, GpuOptions::default())
    }

    /// Load an ONNX model, attempting MIGraphX GPU acceleration on the
    /// device selected by `gpu_options`. Falls back to CPU on failure.
    pub fn load_with_options(
        model_path: &Path,
        spec: ModelSpec,
        gpu_options: GpuOptions,
    ) -> Result<Self> {
        spec.validate()?;
        let (session, uses_gpu) = match Self::load_with_migraphx(model_path, gpu_options.device_id)
        {
            Ok(session) => {
                tracing::info!(
                    device_id = gpu_options.device_id,
                    "ONNX Runtime: using MIGraphX execution provider (GPU)"
                );
                (session, true)
            }
            Err(e) => {
                tracing::info!("MIGraphX EP unavailable ({e}), using CPU");
                let session = Session::builder()
                    .map_err(|e| BekiperError::Inference(format!("ORT session builder: {e}")))?
                    .commit_from_file(model_path)
                    .map_err(|e| BekiperError::Inference(format!("ORT load model: {e}")))?;
                (session, false)
            }
        };

        Ok(Self {
            session,
            spec,
            uses_gpu,
            gpu_options,
        })
    }

    fn load_with_migraphx(
        model_path: &Path,
        device_id: u32,
    ) -> std::result::Result<Session, String> {
        use ort::ep::migraphx::MIGraphX;

        let canonical_model_path = model_path.canonicalize().map_err(|error| {
            format!("canonicalize model path {}: {error}", model_path.display())
        })?;

        // MIGraphX can load compiled .mxr files, but this ONNX Runtime build's
        // save path handling can fail inside session.run even when given a
        // canonical absolute path. Treat the cache as read-only here: use it
        // when it already exists, otherwise run without saving.
        let mut ep = MIGraphX::default().with_device_id(device_id as i32);
        if let Some(cache_path) = prepare_migraphx_cache_path(&canonical_model_path) {
            if let Some(cache_str) = cache_path.to_str() {
                if cache_path.exists() {
                    tracing::info!(cache_path = %cache_path.display(), "loading cached MIGraphX compiled model");
                    ep = ep.with_load_model(cache_str);
                } else {
                    tracing::warn!(
                        cache_path = %cache_path.display(),
                        "MIGraphX compiled model cache missing; running without save_model to avoid runtime cache write failure"
                    );
                }
            } else {
                tracing::warn!(
                    cache_path = %cache_path.display(),
                    "skipping MIGraphX compiled model cache because path is not UTF-8"
                );
            }
        }

        let mut builder = Session::builder()
            .map_err(|e| format!("session builder: {e}"))?
            .with_execution_providers([ep.build()])
            .map_err(|e| format!("MIGraphX EP: {e}"))?;

        builder
            .commit_from_file(&canonical_model_path)
            .map_err(|e| format!("load model: {e}"))
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn labels(&self) -> &[String] {
        &self.spec.labels
    }

    pub fn uses_gpu(&self) -> bool {
        self.uses_gpu
    }

    /// GPU options this model was loaded with.
    pub fn gpu_options(&self) -> &GpuOptions {
        &self.gpu_options
    }

    /// Find the optimal batch size for GPU inference.
    ///
    /// Strategy:
    /// 1. Read available VRAM from ROCm sysfs (`/sys/class/drm/card*/device/mem_info_vram_*`)
    /// 2. Run a single small inference to measure per-element VRAM cost
    /// 3. Calculate max batch size from `free * vram_headroom_fraction`
    ///    and clamp to `[fallback, max_batch_size]`.
    ///
    /// On CPU or if VRAM can't be read, returns `fallback`.
    pub fn probe_batch_size(&mut self, fallback: usize) -> usize {
        if !self.uses_gpu {
            tracing::info!(batch_size = fallback, "CPU mode, skipping batch probe");
            return fallback;
        }

        let (vram_total, vram_used_baseline) = match read_vram_usage() {
            Some(v) => v,
            None => {
                tracing::warn!("cannot read VRAM sysfs, using fallback batch size");
                return fallback;
            }
        };

        let vram_free = vram_total.saturating_sub(vram_used_baseline);
        tracing::info!(
            vram_total_mb = vram_total / (1024 * 1024),
            vram_used_mb = vram_used_baseline / (1024 * 1024),
            vram_free_mb = vram_free / (1024 * 1024),
            "GPU VRAM baseline"
        );

        // Run a small probe batch to measure actual per-element VRAM cost
        // (includes activations, workspace memory, etc. — not just input tensor)
        let probe_size: usize = 4;
        let h = self.spec.input_height as usize;
        let w = self.spec.input_width as usize;
        let c = self.spec.input_channels as usize;

        let tensor = if self.spec.channels_first {
            ndarray::Array4::<f32>::zeros((probe_size, c, h, w))
        } else {
            ndarray::Array4::<f32>::zeros((probe_size, h, w, c))
        };

        let input = match ort::value::Tensor::from_array(tensor) {
            Ok(t) => t,
            Err(_) => return fallback,
        };

        // Run inference so GPU allocates all working memory
        if self.session.run(ort::inputs![input]).is_err() {
            return fallback;
        }

        let vram_after_probe = match read_vram_usage() {
            Some((_, used)) => used,
            None => return fallback,
        };

        // VRAM cost = (post-probe used) - (baseline used)
        // This captures model weights + activations for probe_size elements.
        // Subtract model weight overhead (constant, measured by first element's
        // fixed cost) by computing per-element marginal cost.
        let probe_cost = vram_after_probe.saturating_sub(vram_used_baseline);

        if probe_cost == 0 {
            tracing::warn!("VRAM probe measured zero cost, using fallback");
            return fallback;
        }

        // Per-element cost (rough, includes some fixed overhead amortized over probe_size)
        let per_element = probe_cost / probe_size as u64;

        // Use `vram_headroom_fraction` of free VRAM to leave headroom for
        // runtime spikes. Default 0.75; configurable via GpuOptions.
        let headroom = self.gpu_options.vram_headroom_fraction.clamp(0.0, 1.0);
        let usable_vram = (vram_free as f64 * headroom) as u64;
        let max_batch = (usable_vram / per_element) as usize;

        // Round down to nearest power of 2 (GPU kernels work best with aligned sizes),
        // clamp to [fallback, gpu_options.max_batch_size]
        let cap = self.gpu_options.max_batch_size.max(fallback);
        let batch_size = if max_batch >= 2 {
            (max_batch.next_power_of_two() >> 1).clamp(fallback, cap)
        } else {
            fallback
        };

        tracing::info!(
            per_element_mb = per_element / (1024 * 1024),
            max_batch,
            selected = batch_size,
            "GPU batch size probe complete"
        );

        batch_size
    }

    /// Run inference on a single frame, returning keypoints in original pixel coordinates.
    pub fn infer_frame(&mut self, image: &DynamicImage) -> Result<Vec<Keypoint>> {
        let (tensor, original_size) = preprocess_frame(image, &self.spec);

        let input_tensor = ort::value::Tensor::from_array(tensor)
            .map_err(|e| BekiperError::Inference(format!("input tensor: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .map_err(|e| BekiperError::Inference(format!("inference failed: {e}")))?;

        let output = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| BekiperError::Inference(format!("extract output: {e}")))?;

        decode_output(&self.spec, &output, 0, original_size)
    }

    /// Run inference on a batch of frames, returning a DataFrame with
    /// `frame`, `{label}_x`, `{label}_y`, `{label}_likelihood` columns.
    ///
    /// On GPU, the full batch is sent in one call for maximum throughput.
    /// On CPU, frames are still batched but processed sequentially by ORT.
    pub fn infer_batch(&mut self, images: &[DynamicImage]) -> Result<DataFrame> {
        if images.is_empty() {
            return self.empty_dataframe();
        }

        let (batch_tensor, original_sizes) = preprocess_batch(images, &self.spec);
        self.infer_preprocessed(batch_tensor, &original_sizes)
    }

    /// Run inference on a pre-built tensor, skipping preprocessing.
    ///
    /// Use this with `preprocess_batch` to separate CPU preprocessing
    /// from GPU inference for pipeline overlap.
    pub fn infer_preprocessed(
        &mut self,
        batch_tensor: ndarray::Array4<f32>,
        original_sizes: &[(u32, u32)],
    ) -> Result<DataFrame> {
        let input_tensor = ort::value::Tensor::from_array(batch_tensor)
            .map_err(|e| BekiperError::Inference(format!("input tensor: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .map_err(|e| BekiperError::Inference(format!("inference failed: {e}")))?;

        let output = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| BekiperError::Inference(format!("extract output: {e}")))?;

        decode_dataframe(&self.spec, &output, original_sizes)
    }

    /// Run inference on a GPU-resident tensor using IoBinding (zero-copy).
    ///
    /// The `gpu_tensor_ptr` must be a HIP device pointer to a float32 tensor
    /// in NCHW layout with shape [batch_size, C, H, W]. The output is copied
    /// to CPU for decoding.
    ///
    /// Run inference on a preprocessed tensor that lives in HIP device memory.
    ///
    /// Wraps the HIP device pointer as an ORT tensor via `TensorRefMut::from_raw`
    /// and passes it directly to `session.run()`. No CPU roundtrip.
    /// # Safety
    /// `gpu_tensor_ptr` must be a valid HIP device pointer to a float32 tensor
    /// with the correct shape [batch_size, C, H, W] (or NHWC) and sufficient size.
    #[cfg(feature = "gpu")]
    pub unsafe fn infer_gpu_tensor(
        &mut self,
        gpu_tensor_ptr: *mut std::ffi::c_void,
        batch_size: usize,
        original_sizes: &[(u32, u32)],
    ) -> Result<DataFrame> {
        use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
        use ort::value::{Shape, TensorRefMut};

        let n = original_sizes.len();
        assert_eq!(n, batch_size);

        let h = self.spec.input_height as usize;
        let w = self.spec.input_width as usize;
        let c = self.spec.input_channels as usize;

        let shape = if self.spec.channels_first {
            Shape::new([batch_size as i64, c as i64, h as i64, w as i64])
        } else {
            Shape::new([batch_size as i64, h as i64, w as i64, c as i64])
        };

        // Wrap the HIP device pointer as an ORT tensor
        let mem_info = MemoryInfo::new(
            AllocationDevice::HIP,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| BekiperError::Inference(format!("GPU MemoryInfo: {e}")))?;

        let input_tensor = unsafe {
            // SAFETY: This unsafe function's contract requires `gpu_tensor_ptr`
            // to point to a live HIP allocation containing a float32 tensor
            // with exactly `shape` elements for the duration of `session.run`.
            TensorRefMut::<f32>::from_raw(mem_info, gpu_tensor_ptr, shape)
        }
        .map_err(|e| BekiperError::Inference(format!("GPU tensor from raw: {e}")))?;

        // Run inference directly with the GPU tensor (no IoBinding)
        let input_name = self
            .session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".into());

        let outputs = self
            .session
            .run(ort::inputs![&input_name => input_tensor.into_dyn()])
            .map_err(|e| BekiperError::Inference(format!("session.run: {e}")))?;

        let output = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| BekiperError::Inference(format!("extract output: {e}")))?;

        decode_dataframe(&self.spec, &output, original_sizes)
    }
}

fn prepare_migraphx_cache_path(canonical_model_path: &Path) -> Option<std::path::PathBuf> {
    let cache_path = canonical_model_path.with_extension("mxr");
    let Some(parent) = cache_path.parent() else {
        tracing::warn!(
            cache_path = %cache_path.display(),
            "skipping MIGraphX compiled model cache because path has no parent"
        );
        return None;
    };
    if let Err(error) = std::fs::create_dir_all(parent) {
        tracing::warn!(
            cache_path = %cache_path.display(),
            error = %error,
            "skipping MIGraphX compiled model cache because parent directory is not writable"
        );
        return None;
    }
    Some(cache_path)
}

#[cfg(test)]
mod tests {
    use super::prepare_migraphx_cache_path;

    #[test]
    fn migraphx_cache_path_uses_canonical_absolute_model_path() {
        let root = std::env::temp_dir().join(format!(
            "bekiper_migraphx_cache_path_{}",
            std::process::id()
        ));
        let nested = root.join("models").join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let model_path = root
            .join("models")
            .join("nested")
            .join("..")
            .join("pose.onnx");
        std::fs::write(root.join("models").join("pose.onnx"), b"placeholder").unwrap();

        let canonical_model = model_path.canonicalize().unwrap();
        let cache_path = prepare_migraphx_cache_path(&canonical_model).unwrap();

        assert!(cache_path.is_absolute());
        assert_eq!(
            cache_path.file_name().and_then(|name| name.to_str()),
            Some("pose.mxr")
        );
        assert_ne!(cache_path.to_str(), Some(""));
        assert_eq!(cache_path.parent(), canonical_model.parent());

        std::fs::remove_dir_all(root).ok();
    }
}
