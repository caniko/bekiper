use ort::session::Session;

use super::super::spec::ModelSpec;

/// Default HIP / MIGraphX device index.
pub const DEFAULT_GPU_DEVICE_ID: u32 = 0;

/// Default hard cap on the auto-probed batch size.
pub const DEFAULT_MAX_BATCH_SIZE: usize = 1024;

/// Default fraction of free VRAM usable by the batch probe.
pub const DEFAULT_VRAM_HEADROOM_FRACTION: f64 = 0.75;

/// Default batch size used when the VRAM probe fails or when running
/// on CPU. Conservative enough to fit on machines with little RAM.
pub const DEFAULT_FALLBACK_BATCH_SIZE: usize = 32;

/// Runtime GPU options for ONNX inference.
///
/// The ingress layer constructs this from its TOML `[gpu]` section and
/// environment-variable overrides; this crate consumes it verbatim and
/// stays free of any config-loading concerns.
#[derive(Debug, Clone, Copy)]
pub struct GpuOptions {
    /// HIP / MIGraphX device index.
    pub device_id: u32,
    /// Hard cap applied after the VRAM-based probe.
    pub max_batch_size: usize,
    /// Fraction of free VRAM (0.0–1.0) usable by the probe; values below
    /// 1.0 leave headroom for runtime spikes.
    pub vram_headroom_fraction: f64,
    /// Batch size used when the VRAM probe fails or when running on CPU.
    pub fallback_batch_size: usize,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            device_id: DEFAULT_GPU_DEVICE_ID,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            vram_headroom_fraction: DEFAULT_VRAM_HEADROOM_FRACTION,
            fallback_batch_size: DEFAULT_FALLBACK_BATCH_SIZE,
        }
    }
}

/// ONNX Runtime-backed pose estimation model.
///
/// Attempts MIGraphX GPU acceleration on AMD GPUs, falling back to CPU.
/// Requires `ORT_DYLIB_PATH` pointing to `libonnxruntime.so` at runtime.
pub struct OnnxPoseModel {
    pub(super) session: Session,
    pub(super) spec: ModelSpec,
    pub(super) uses_gpu: bool,
    pub(super) gpu_options: GpuOptions,
}
