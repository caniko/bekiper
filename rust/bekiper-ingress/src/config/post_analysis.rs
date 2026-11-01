use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Post-analysis configuration (`[post_analysis]`).
///
/// Controls the behavioural post-analysis pipeline that runs after per-trial
/// summaries are complete. The pipeline consumes native trial records or
/// per-trial `result.toml` files,
/// produces registered statistical outputs, figures, and a human-readable report,
/// and publishes an atomic artifact manifest.
///
/// All fields are optional in TOML and each supports an environment-variable
/// override applied by the workflow loader:
/// `BEKIPER_POST_ANALYSIS_DISABLED`, `BEKIPER_POST_ANALYSIS_PLAN_PATH`,
/// `BEKIPER_POST_ANALYSIS_RUN_ID`, `BEKIPER_POST_ANALYSIS_FIGURES_ENABLED`,
/// `BEKIPER_POST_ANALYSIS_REPORT_ENABLED`, `BEKIPER_POST_ANALYSIS_REPORT_TITLE`,
/// `BEKIPER_POST_ANALYSIS_REPORT_COMPILER`.
///
/// Precedence (highest to lowest):
/// 1. `PostAnalysisOverrides` passed to the workflow constructor
/// 2. Environment variable
/// 3. TOML `[post_analysis]` section
/// 4. Built-in default
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostAnalysisConfig {
    /// Independent Y-maze primary exports over the same native trial records.
    #[serde(default)]
    pub ymaze_primary_thresholds: Option<Vec<u8>>,
    #[serde(default = "default_primary_workers")]
    pub primary_workers: usize,
    /// One variant of one assay per context (`[[post_analysis.contexts]]`); the
    /// other assays take their principal variant. Replaces the thresholds.
    #[serde(default)]
    pub contexts: Option<Vec<bekiper_post_analysis::contracts::ContextSpec>>,
    /// Total figure-render processes of one cohort publication; split across
    /// concurrently running contexts. Unset keeps the renderer default.
    #[serde(default)]
    pub render_workers: Option<usize>,
    /// Resolved constructor/environment overlay. Included in sealed snapshot
    /// identity so restoration cannot silently restore the plan's old policy.
    #[serde(
        default,
        skip_serializing_if = "bekiper_feature::tolerance::DwellOverrides::is_empty"
    )]
    pub ymaze_dwell_overrides: bekiper_feature::tolerance::DwellOverrides,
    /// When `false`, the post-analysis batch stage is skipped entirely.
    /// Defaults to `false` (disabled by default).
    #[serde(default = "default_post_analysis_enabled")]
    pub enabled: bool,

    /// Optional path to a frozen analysis plan JSON, resolved relative to the
    /// project config file. When present, the pipeline loads the registered
    /// plan instead of inferring one from the discovered data.
    /// The registry's sealed-snapshot engine resolves and captures this plan
    /// before admitted native execution.
    #[serde(default)]
    pub plan_path: Option<PathBuf>,

    /// Optional run identifier for publication and manifest correlation.
    /// When omitted, a timestamp-based ID is generated at pipeline start.
    #[serde(default)]
    pub run_id: Option<String>,

    /// Figure rendering configuration.
    #[serde(default)]
    pub figures: FiguresConfig,

    /// Report generation configuration.
    #[serde(default)]
    pub report: ReportConfig,

    /// Frozen plan embedded as JSON value (serde default), populated by the
    /// registry's sealed-snapshot engine during run preparation.
    /// Takes precedence over `plan_path` when both are present.
    #[serde(default)]
    pub frozen_plan: Option<serde_json::Value>,

    /// Frozen source fingerprints from the registry's sealed-snapshot engine.
    /// Population preserves input provenance across rebased runs.
    #[serde(default)]
    pub frozen_fingerprints: Vec<serde_json::Value>,
}

impl Default for PostAnalysisConfig {
    fn default() -> Self {
        Self {
            ymaze_primary_thresholds: None,
            primary_workers: default_primary_workers(),
            contexts: None,
            render_workers: None,
            ymaze_dwell_overrides: Default::default(),
            enabled: default_post_analysis_enabled(),
            plan_path: None,
            run_id: None,
            figures: FiguresConfig::default(),
            report: ReportConfig::default(),
            frozen_plan: None,
            frozen_fingerprints: Vec::new(),
        }
    }
}

fn default_post_analysis_enabled() -> bool {
    false
}

/// Figure rendering configuration (`[post_analysis.figures]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiguresConfig {
    /// When `false`, figure rendering is skipped entirely.
    /// Defaults to `true` when post-analysis is enabled.
    #[serde(default = "default_figures_enabled")]
    pub enabled: bool,

    /// Optional named figure selection. Omission selects all applicable figures.
    /// An empty selection suppresses all figures.
    #[serde(default)]
    pub selection: Option<Vec<String>>,

    /// DPI for rasterized figure outputs. Defaults to `150`.
    #[serde(default = "default_figures_dpi")]
    pub dpi: u32,

    /// Explicit format list. Defaults to `["pdf", "png"]`.
    #[serde(default = "default_figures_formats")]
    pub formats: Vec<String>,
}

impl Default for FiguresConfig {
    fn default() -> Self {
        Self {
            enabled: default_figures_enabled(),
            selection: None,
            dpi: default_figures_dpi(),
            formats: default_figures_formats(),
        }
    }
}

fn default_figures_enabled() -> bool {
    true
}

fn default_figures_dpi() -> u32 {
    150
}

fn default_figures_formats() -> Vec<String> {
    vec!["pdf".to_string(), "png".to_string()]
}

/// Report generation configuration (`[post_analysis.report]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportConfig {
    /// When `false`, report generation is skipped entirely.
    /// Defaults to `true` when post-analysis is enabled.
    #[serde(default = "default_report_enabled")]
    pub enabled: bool,

    /// Report title. Defaults to `"Behavioural Post-Analysis Report"`.
    #[serde(default = "default_report_title")]
    pub title: String,

    /// Compiler to use for PDF output: `"latexmk"` or `"lualatex"`.
    /// Defaults to `"latexmk"`.
    #[serde(default = "default_report_compiler")]
    pub compiler: String,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            enabled: default_report_enabled(),
            title: default_report_title(),
            compiler: default_report_compiler(),
        }
    }
}

fn default_report_enabled() -> bool {
    true
}

fn default_primary_workers() -> usize {
    bekiper_post_analysis::contracts::DEFAULT_PRIMARY_ANALYSIS_WORKERS
}

fn default_report_title() -> String {
    "Behavioural Post-Analysis Report".to_string()
}

fn default_report_compiler() -> String {
    "latexmk".to_string()
}

/// Constructor/CLI overrides are applied after the environment overlay.
#[derive(Debug, Default, Clone)]
pub struct PostAnalysisOverrides {
    pub ymaze_primary_thresholds: Option<Vec<u8>>,
    pub primary_workers: Option<usize>,
    pub contexts: Option<Vec<bekiper_post_analysis::contracts::ContextSpec>>,
    pub render_workers: Option<usize>,
    pub enabled: Option<bool>,
    pub plan_path: Option<PathBuf>,
    pub run_id: Option<String>,
    pub figures_enabled: Option<bool>,
    pub figure_selection: Option<Vec<String>>,
    pub figure_formats: Option<Vec<String>>,
    pub figure_dpi: Option<u32>,
    pub report_enabled: Option<bool>,
    pub report_title: Option<String>,
    pub report_compiler: Option<String>,
}

include!("post_analysis/memory.rs");

include!("post_analysis/environment.rs");

include!("post_analysis/overrides_validation.rs");

include!("post_analysis/frozen_plan.rs");

pub fn valid_run_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

#[cfg(test)]
#[path = "post_analysis/tests.rs"]
mod tests;
