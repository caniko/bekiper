mod expand;
mod io_helpers;
mod measurements;
mod population;
mod set_overrides;
mod temporal;
mod whisker;

pub mod aggregate;
pub mod detection;
pub mod experiment;
pub mod gpu;
pub mod inspect;
pub mod pipeline;
pub mod post_analysis;
pub mod profile;
pub mod review;
pub mod scan;
pub mod task_overrides;

#[cfg(test)]
mod tests;

pub use aggregate::*;
pub use detection::*;
pub use experiment::*;
pub use gpu::*;
pub use inspect::*;
pub use measurements::*;
pub use pipeline::*;
pub use population::*;
pub use post_analysis::*;
pub use profile::*;
pub use review::*;
pub use scan::*;
pub use set_overrides::{AppliedOverride, apply_set_overrides, ensure_overrides_retained};
pub use task_overrides::*;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use bekiper_core::config::ProjectConfig;
use bekiper_core::error::{BekiperError, Result};

/// Re-exported so `crate::config::AnimalMetadata` keeps working; the struct
/// itself now lives in `bekiper-core` alongside `ExperimentResult`.
pub use bekiper_core::inspection::AnimalMetadata;
pub use bekiper_feature::heuristic::solo::whisker_config::WhiskerEnv;

/// Full project configuration loaded from a TOML file.
///
/// Supports two modes:
/// - **Explicit**: list individual `[[experiments]]` entries.
/// - **Scan**: list `[[scan]]` entries that walk a directory tree and expand
///   each discovered video into an `ExperimentConfig`.
///
/// Both can coexist; scans are expanded first and prepended to explicit experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressConfig {
    pub project: ProjectConfig,
    #[serde(default)]
    pub pipeline: PipelineConfig,
    /// GPU runtime configuration (device selection, batch-size probing).
    #[serde(default)]
    pub gpu: GpuConfig,
    /// Cross-trial aggregation configuration (`[aggregate]`). Controls the
    /// consolidated `*_results.ods` spreadsheet produced after every per-trial
    /// `Summary` stage has written its `result.toml`.
    #[serde(default)]
    pub aggregate: AggregateConfig,
    /// Post-analysis configuration (`[post_analysis]`). Controls the
    /// behavioural post-analysis pipeline that runs after per-trial
    /// summaries are complete.
    #[serde(default)]
    pub post_analysis: PostAnalysisConfig,
    #[serde(default)]
    pub inspect: InspectConfig,
    /// Reviewer package settings (`[review]`).
    #[serde(default)]
    pub review: ReviewConfig,
    /// Shared pose model applied to every experiment unless overridden.
    pub pose_model: Option<PoseModelConfig>,
    #[serde(default)]
    pub scan: Vec<ScanConfig>,
    /// Animal-structured data directories (pre-computed DLC pose + annotations).
    #[serde(default)]
    pub animal_scan: Vec<AnimalScanConfig>,
    /// In-tree ingress profiles for study-specific data layouts.
    #[serde(default)]
    pub ingress_profile: Vec<IngressProfileConfig>,
    #[serde(default)]
    pub experiments: Vec<ExperimentConfig>,

    /// Animal metadata loaded from `animal_metadata_csv` (not serialized).
    /// Keyed by animal ID. Used for enriching output with cohort/gene/sex/age.
    #[serde(skip)]
    pub animal_metadata: HashMap<u32, AnimalMetadata>,

    /// `BEKIPER_SET` / `--set` overrides applied to the TOML, in application
    /// order (later wins). Not serialized.
    #[serde(skip)]
    pub override_record: Vec<AppliedOverride>,
    /// Leaf key paths (e.g. `ingress_profile[0].tasks[2].task_params.ymaze.proximal_threshold`)
    /// written in the TOML file itself, before overrides. A key is `toml`-sourced if it is
    /// here, `env`/`cli` if the last `override_record` entry targeting it says so, else `default`.
    #[serde(skip)]
    pub toml_paths: Vec<String>,
}

impl IngressConfig {
    /// Parse a TOML config file into an `IngressConfig`.
    ///
    /// This is a pure parse + env-overlay step: no directory walks, no CSV
    /// reads, no file writes. Call [`expand`](Self::expand) before running any
    /// pipeline stage that needs `[[scan]]` / `[[animal_scan]]` entries
    /// turned into concrete `ExperimentConfig`s.
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_file_with_post_analysis_overrides(path, &PostAnalysisOverrides::default())
    }

    /// Resolve constructor > environment > TOML > defaults before validation.
    pub fn from_file_with_post_analysis_overrides(
        path: &Path,
        options: &PostAnalysisOverrides,
    ) -> Result<Self> {
        Self::from_file_with_post_analysis_overrides_from(path, options, |key| {
            std::env::var(key).ok()
        })
    }

    /// Injected environment lookup keeps configuration precedence testable.
    pub fn from_file_with_post_analysis_overrides_from(
        path: &Path,
        options: &PostAnalysisOverrides,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        Self::from_file_with_overrides_from(path, options, Default::default(), lookup)
    }

    pub fn from_file_with_overrides_from(
        path: &Path,
        options: &PostAnalysisOverrides,
        dwell: bekiper_feature::tolerance::DwellOverrides,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        Self::from_file_with_measurement_overrides_from(
            path,
            options,
            dwell,
            Default::default(),
            lookup,
        )
    }

    pub fn from_file_with_measurement_overrides_from(
        path: &Path,
        options: &PostAnalysisOverrides,
        dwell: bekiper_feature::tolerance::DwellOverrides,
        measurements: MeasurementOverrides,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        Self::from_file_with_set_overrides_from(path, options, dwell, measurements, &[], lookup)
    }

    /// As above plus generic `--set KEY=VALUE` pairs (`cli_set`), which beat
    /// `BEKIPER_SET` and the TOML. See [`set_overrides`](self) for key syntax.
    pub fn from_file_with_set_overrides_from(
        path: &Path,
        options: &PostAnalysisOverrides,
        dwell: bekiper_feature::tolerance::DwellOverrides,
        measurements: MeasurementOverrides,
        cli_set: &[String],
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut config = Self::parse_with_set_overrides(&content, cli_set, &lookup)?;
        let set_records = config.override_record.len();

        config.apply_env_overrides_from(&lookup);
        let dwell_resolved = dwell.with_env(&lookup).map_err(BekiperError::Config)?;
        config.apply_ymaze_dwell_overrides(dwell_resolved)?;
        config.post_analysis.apply_overrides(options);
        let measured = measurements.with_env(&lookup)?;
        config.apply_measurement_overrides(measured);
        let whisker = WhiskerEnv::default()
            .with_env(&lookup)
            .map_err(BekiperError::Config)?;
        config.apply_whisker_env(whisker);

        // Dedicated shorthands are recorded as the generic key they set, so
        // source maps show env/cli rather than toml.
        let layer = |cli: bool| if cli { "cli" } else { "env" };
        let mut shorthands: Vec<(&'static str, &str, Option<toml::Value>)> = vec![
            (
                layer(dwell.seconds.is_some()),
                "ymaze.ymaze.temporal_policy.min_duration_seconds",
                dwell_resolved.seconds.map(toml::Value::Float),
            ),
            (
                layer(dwell.comparison.is_some()),
                "ymaze.ymaze.temporal_policy.duration_comparison",
                dwell_resolved
                    .comparison
                    .and_then(|c| toml::Value::try_from(c).ok()),
            ),
            (
                layer(measurements.oft_window_seconds.is_some()),
                "oft.oft_window_seconds",
                measured.oft_window_seconds.map(toml::Value::Float),
            ),
            (
                layer(measurements.nort_strict_boundaries.is_some()),
                "nort.exploration.strict_boundaries",
                measured.nort_strict_boundaries.map(toml::Value::Boolean),
            ),
        ];
        // WhiskerEnv has no constructor/CLI form: always env.
        for (key, value) in [
            ("enabled", whisker.enabled.map(toml::Value::Boolean)),
            (
                "apex_fraction",
                whisker.apex_fraction.map(toml::Value::Float),
            ),
            (
                "half_angle_deg",
                whisker.half_angle_deg.map(toml::Value::Float),
            ),
        ] {
            shorthands.push(("env", key, value));
        }
        for (layer, key, value) in shorthands {
            if let Some(value) = value {
                let key = if layer == "env" && !key.contains('.') {
                    format!("nort.exploration.whisker.{key}")
                } else {
                    key.to_string()
                };
                config.record_shorthand(layer, &key, value);
            }
        }
        if let Some(meters) = whisker.length_m {
            let mut length = toml::Table::new();
            length.insert("meters".into(), toml::Value::Float(meters));
            let length = toml::Value::Table(length);
            config.record_shorthand("env", "nort.exploration.whisker.length", length);
        }
        // CLI > env: `--set` is re-applied over the dedicated env overlays and
        // shorthands (which ran on the typed config after the TOML-level pass).
        if !cli_set.is_empty() {
            let from = set_records - cli_set.len();
            let cli: Vec<_> = config.override_record.drain(from..set_records).collect();
            config.reapply_cli_sets(cli_set)?;
            config.override_record.extend(cli);
        }
        config.validate()?;

        Ok(config)
    }

    /// Validate pure configuration constraints that do not require filesystem
    /// discovery or materializing scan outputs.
    pub fn validate(&self) -> Result<()> {
        self.validate_ymaze_temporal_policies()?;
        self.post_analysis.validate()?;
        for scan in &self.scan {
            let context = format!("scan `{}`", scan.video_dir.display());
            validate_task_variants(&context, &scan.variants)?;
        }

        for profile in &self.ingress_profile {
            for task in &profile.tasks {
                let context = format!(
                    "ingress_profile `{}` task `{}`",
                    profile.profile.as_str(),
                    task.video_dir.display()
                );
                validate_task_variants(&context, &task.variants)?;
            }
        }

        Ok(())
    }

    /// Apply environment-variable overrides on top of the current config.
    ///
    /// Precedence: env > file > default. CLI flags are applied separately by
    /// the command layer on top of the returned config.
    pub fn apply_env_overrides(&mut self) {
        self.apply_env_overrides_from(|k| std::env::var(k).ok());
    }

    /// Testable env overlay variant using an injected lookup closure.
    pub fn apply_env_overrides_from<F>(&mut self, lookup: F)
    where
        F: Fn(&str) -> Option<String>,
    {
        self.project.runtime.apply_env_overrides_from(&lookup);
        self.gpu.apply_env_overrides_from(&lookup);
        self.pipeline.apply_env_overrides_from(&lookup);
        self.aggregate.apply_env_overrides_from(&lookup);
        self.post_analysis.apply_env_overrides_from(&lookup);
        self.inspect.apply_env_overrides_from(&lookup);
    }
}

fn validate_task_variants(context: &str, variants: &[task_overrides::TaskVariant]) -> Result<()> {
    let mut seen = HashSet::new();

    for variant in variants {
        validate_variant_name(context, &variant.name)?;
        if !seen.insert(variant.name.as_str()) {
            return Err(BekiperError::Config(format!(
                "{context} has duplicate variant name {:?}; variant names must be unique within one scan or task entry",
                variant.name
            )));
        }
    }

    Ok(())
}

fn validate_variant_name(context: &str, name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(BekiperError::Config(format!(
            "{context} has an empty variant name; use a non-empty ASCII slug such as \"proximal65\""
        )));
    }

    if !name.bytes().all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
    }) {
        return Err(BekiperError::Config(format!(
            "{context} has invalid variant name {name:?}; use a non-empty ASCII slug containing only lowercase letters, digits, '_' or '-', such as \"proximal65\""
        )));
    }

    Ok(())
}
