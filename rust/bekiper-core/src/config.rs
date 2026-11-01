use serde::{Deserialize, Serialize};

/// Runtime settings, equivalent to Python's BekiperRuntimeSettings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RuntimeSettings {
    /// Minimum duration (seconds) for a behavioral bout to be counted.
    #[serde(default = "default_min_seconds_tolerance")]
    pub minimum_seconds_tolerance: f64,

    /// Maximum gap (seconds) to bridge between two bouts.
    #[serde(default = "default_max_seconds_distraction")]
    pub maximum_seconds_distraction: f64,

    /// Number of Rayon threads for experiment-level parallelism.
    /// `None` means use all available cores.
    #[serde(default)]
    pub num_threads: Option<usize>,

    /// Whether host-RAM-aware admission control should throttle new
    /// experiment work when the machine is already under memory pressure.
    #[serde(default = "default_memory_scheduler_enabled")]
    pub memory_scheduler_enabled: bool,

    /// Maximum whole-machine RAM usage fraction allowed before new parallel
    /// experiment work is throttled. Must be in the open interval `(0, 1]`.
    #[serde(default = "default_max_ram_usage_fraction")]
    pub max_ram_usage_fraction: f64,
}

fn default_min_seconds_tolerance() -> f64 {
    0.5
}

fn default_max_seconds_distraction() -> f64 {
    1.0 / 3.0
}

fn default_memory_scheduler_enabled() -> bool {
    true
}

fn default_max_ram_usage_fraction() -> f64 {
    0.80
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            minimum_seconds_tolerance: default_min_seconds_tolerance(),
            maximum_seconds_distraction: default_max_seconds_distraction(),
            num_threads: None,
            memory_scheduler_enabled: default_memory_scheduler_enabled(),
            max_ram_usage_fraction: default_max_ram_usage_fraction(),
        }
    }
}

impl RuntimeSettings {
    /// Overlay process environment variables on top of this config.
    pub fn apply_env_overrides(&mut self) {
        self.apply_env_overrides_from(|k| std::env::var(k).ok());
    }

    /// Testable env overlay variant using an injected lookup closure.
    pub fn apply_env_overrides_from<F>(&mut self, lookup: F)
    where
        F: Fn(&str) -> Option<String>,
    {
        if let Some(raw) = lookup("BEKIPER_RUNTIME_NUM_THREADS") {
            match raw.parse::<usize>() {
                Ok(v) if v > 0 => {
                    tracing::info!(
                        num_threads = v,
                        "BEKIPER_RUNTIME_NUM_THREADS override applied"
                    );
                    self.num_threads = Some(v);
                }
                Ok(_) => tracing::warn!("BEKIPER_RUNTIME_NUM_THREADS must be > 0, ignoring"),
                Err(e) => tracing::warn!(
                    value = %raw,
                    error = %e,
                    "BEKIPER_RUNTIME_NUM_THREADS is not a valid usize, ignoring"
                ),
            }
        }

        if let Some(raw) = lookup("BEKIPER_RUNTIME_MEMORY_SCHEDULER") {
            match parse_bool_flag(&raw) {
                Some(v) => {
                    tracing::info!(
                        memory_scheduler_enabled = v,
                        "BEKIPER_RUNTIME_MEMORY_SCHEDULER override applied"
                    );
                    self.memory_scheduler_enabled = v;
                }
                None if raw.trim().is_empty() => {
                    tracing::warn!("BEKIPER_RUNTIME_MEMORY_SCHEDULER is empty, ignoring")
                }
                None => tracing::warn!(
                    value = %raw,
                    "BEKIPER_RUNTIME_MEMORY_SCHEDULER is not a boolean flag, ignoring"
                ),
            }
        }

        if let Some(raw) = lookup("BEKIPER_RUNTIME_MAX_RAM_USAGE") {
            match raw.parse::<f64>() {
                Ok(v) if (0.0..=1.0).contains(&v) && v > 0.0 => {
                    tracing::info!(
                        max_ram_usage_fraction = v,
                        "BEKIPER_RUNTIME_MAX_RAM_USAGE override applied"
                    );
                    self.max_ram_usage_fraction = v;
                }
                Ok(_) => tracing::warn!(
                    "BEKIPER_RUNTIME_MAX_RAM_USAGE must be in the interval (0.0, 1.0], ignoring"
                ),
                Err(e) => tracing::warn!(
                    value = %raw,
                    error = %e,
                    "BEKIPER_RUNTIME_MAX_RAM_USAGE is not a valid f64, ignoring"
                ),
            }
        }
    }
}

fn parse_bool_flag(raw: &str) -> Option<bool> {
    let trimmed = raw.trim().to_ascii_lowercase();
    match trimmed.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Project-level configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub project_name: String,
    pub data_directory: std::path::PathBuf,
    /// Output directory for pipeline artifacts. Defaults to "output".
    pub output_dir: Option<std::path::PathBuf>,
    #[serde(default)]
    pub runtime: RuntimeSettings,
}
