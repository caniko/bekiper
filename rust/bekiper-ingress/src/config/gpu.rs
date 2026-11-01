use serde::{Deserialize, Serialize};

/// GPU runtime configuration.
///
/// All fields are optional in TOML (`[gpu]`) and each supports an
/// environment-variable override applied by the workflow loader:
/// `BEKIPER_GPU_DEVICE`, `BEKIPER_GPU_MAX_BATCH_SIZE`,
/// `BEKIPER_GPU_VRAM_HEADROOM`.
///
/// Precedence (highest to lowest):
/// 1. CLI flag (not yet wired)
/// 2. Environment variable
/// 3. TOML `[gpu]` section
/// 4. Built-in default
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// HIP / MIGraphX device index. Defaults to 0.
    #[serde(default = "default_gpu_device_id")]
    pub device_id: u32,
    /// Hard cap on the auto-probed batch size. Defaults to 1024.
    #[serde(default = "default_gpu_max_batch_size")]
    pub max_batch_size: usize,
    /// Fraction of free VRAM (0.0–1.0) usable by the batch probe. Values
    /// below 1.0 leave headroom for runtime spikes. Defaults to 0.75.
    #[serde(default = "default_gpu_vram_headroom")]
    pub vram_headroom_fraction: f64,
    /// Batch size used when the VRAM probe fails or when running on CPU.
    /// Defaults to 32.
    #[serde(default = "default_gpu_fallback_batch_size")]
    pub fallback_batch_size: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: default_gpu_device_id(),
            max_batch_size: default_gpu_max_batch_size(),
            vram_headroom_fraction: default_gpu_vram_headroom(),
            fallback_batch_size: default_gpu_fallback_batch_size(),
        }
    }
}

fn default_gpu_device_id() -> u32 {
    0
}

fn default_gpu_max_batch_size() -> usize {
    1024
}

fn default_gpu_vram_headroom() -> f64 {
    0.75
}

fn default_gpu_fallback_batch_size() -> usize {
    32
}

impl GpuConfig {
    /// Project this config into the narrower `bekiper_pose::infer::GpuOptions`
    /// struct consumed by the inference layer.
    #[cfg(feature = "pose-infer")]
    pub fn to_pose_options(&self) -> bekiper_pose::infer::GpuOptions {
        bekiper_pose::infer::GpuOptions {
            device_id: self.device_id,
            max_batch_size: self.max_batch_size,
            vram_headroom_fraction: self.vram_headroom_fraction,
            fallback_batch_size: self.fallback_batch_size,
        }
    }

    /// Overlay process environment variables on top of this config.
    ///
    /// Thin wrapper around [`GpuConfig::apply_env_overrides_from`] that
    /// reads from the real environment. Tests should prefer the `_from`
    /// variant with an injected lookup closure — mutating the process
    /// environment inside tests is both `unsafe` in modern Rust and not
    /// thread-safe under cargo test's parallel runner.
    pub fn apply_env_overrides(&mut self) {
        self.apply_env_overrides_from(|k| std::env::var(k).ok());
    }

    /// Overlay environment-variable settings on top of this config using
    /// the supplied lookup closure. Invalid / unparseable values are
    /// logged and ignored.
    pub fn apply_env_overrides_from<F>(&mut self, lookup: F)
    where
        F: Fn(&str) -> Option<String>,
    {
        if let Some(raw) = lookup("BEKIPER_GPU_DEVICE") {
            match raw.parse::<u32>() {
                Ok(v) => {
                    tracing::info!(device_id = v, "BEKIPER_GPU_DEVICE override applied");
                    self.device_id = v;
                }
                Err(e) => tracing::warn!(
                    value = %raw,
                    error = %e,
                    "BEKIPER_GPU_DEVICE is not a valid u32, ignoring"
                ),
            }
        }
        if let Some(raw) = lookup("BEKIPER_GPU_MAX_BATCH_SIZE") {
            match raw.parse::<usize>() {
                Ok(v) if v > 0 => {
                    tracing::info!(
                        max_batch_size = v,
                        "BEKIPER_GPU_MAX_BATCH_SIZE override applied"
                    );
                    self.max_batch_size = v;
                }
                Ok(_) => tracing::warn!("BEKIPER_GPU_MAX_BATCH_SIZE must be > 0, ignoring"),
                Err(e) => tracing::warn!(
                    value = %raw,
                    error = %e,
                    "BEKIPER_GPU_MAX_BATCH_SIZE is not a valid usize, ignoring"
                ),
            }
        }
        if let Some(raw) = lookup("BEKIPER_GPU_VRAM_HEADROOM") {
            match raw.parse::<f64>() {
                Ok(v) if (0.0..=1.0).contains(&v) => {
                    tracing::info!(
                        vram_headroom_fraction = v,
                        "BEKIPER_GPU_VRAM_HEADROOM override applied"
                    );
                    self.vram_headroom_fraction = v;
                }
                Ok(_) => {
                    tracing::warn!("BEKIPER_GPU_VRAM_HEADROOM must be in [0.0, 1.0], ignoring")
                }
                Err(e) => tracing::warn!(
                    value = %raw,
                    error = %e,
                    "BEKIPER_GPU_VRAM_HEADROOM is not a valid f64, ignoring"
                ),
            }
        }
        if let Some(raw) = lookup("BEKIPER_GPU_FALLBACK_BATCH_SIZE") {
            match raw.parse::<usize>() {
                Ok(v) if v > 0 => {
                    tracing::info!(
                        fallback_batch_size = v,
                        "BEKIPER_GPU_FALLBACK_BATCH_SIZE override applied"
                    );
                    self.fallback_batch_size = v;
                }
                Ok(_) => tracing::warn!("BEKIPER_GPU_FALLBACK_BATCH_SIZE must be > 0, ignoring"),
                Err(e) => tracing::warn!(
                    value = %raw,
                    error = %e,
                    "BEKIPER_GPU_FALLBACK_BATCH_SIZE is not a valid usize, ignoring"
                ),
            }
        }
    }
}
