use std::path::Path;
use std::path::PathBuf;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Per-sheet aggregation rule for the consolidated `*_results.ods` spreadsheet.
///
/// Serialized in snake_case to match how [`bekiper_core::types::DetectionRecipe`] is written in
/// the same config file. Only passthrough aggregation is supported: aggregate
/// output is a consolidated per-trial workbook, not a reductive animal summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationRule {
    /// Emit one row per trial, no collapsing.
    Passthrough,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingMetadataPolicy {
    #[default]
    Fail,
    ExcludeMissing,
    AllowBlanks,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateMetadataConfig {
    /// XLSX workbook path, resolved relative to the project config file.
    pub path: PathBuf,
    /// Worksheet name to load.
    #[serde(default = "default_metadata_sheet")]
    pub sheet: String,
    /// How rows without matching animal metadata are handled.
    #[serde(default)]
    pub missing_policy: MissingMetadataPolicy,
}

fn default_metadata_sheet() -> String {
    "Updated_Mice-overview".to_string()
}

/// Cross-trial aggregation configuration (`[aggregate]`).
///
/// Drives the post-`Summary` batch stage that walks every per-trial
/// `result.toml` under the run root and writes a single consolidated
/// OpenDocument spreadsheet.
///
/// All fields are optional in TOML and each supports an environment-variable
/// override applied by the workflow loader:
/// `BEKIPER_AGGREGATE_DISABLED`, `BEKIPER_AGGREGATE_OUTPUT`.
///
/// Precedence (highest to lowest):
/// 1. CLI flag (`--no-aggregate`, `--aggregate-output`)
/// 2. Environment variable
/// 3. TOML `[aggregate]` section
/// 4. Built-in default
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateConfig {
    /// When `false`, the batch aggregation stage is skipped entirely.
    /// Defaults to `true`.
    #[serde(default = "default_aggregate_enabled")]
    pub enabled: bool,
    /// Output spreadsheet filename, written into the run root. Defaults to
    /// `"results.ods"`.
    #[serde(default = "default_aggregate_output_filename")]
    pub output_filename: String,
    /// Optional recipe -> sheet-name rename map. When empty, each
    /// `DetectionRecipe` becomes its own sheet named after the recipe.
    ///
    /// ```toml
    /// [aggregate.recipe_to_sheet]
    /// oultraining = "training"
    /// oultest     = "test"
    /// ```
    #[serde(default)]
    pub recipe_to_sheet: IndexMap<String, String>,
    /// Per-sheet aggregation rule. Keyed by the *sheet* name (i.e. after
    /// `recipe_to_sheet` rename), not the raw recipe. Sheets not listed
    /// here fall through to [`AggregationRule::Passthrough`]. Reductive
    /// transforms such as `mean_by_animal` are intentionally unsupported and
    /// fail during TOML parsing.
    ///
    /// ```toml
    /// [aggregate.aggregation]
    /// training = "passthrough"
    /// test     = "passthrough"
    /// ```
    #[serde(default)]
    pub aggregation: IndexMap<String, AggregationRule>,
    /// Optional animal-level metadata workbook used to enrich aggregate rows.
    #[serde(default)]
    pub metadata: Option<AggregateMetadataConfig>,
}

impl Default for AggregateConfig {
    fn default() -> Self {
        Self {
            enabled: default_aggregate_enabled(),
            output_filename: default_aggregate_output_filename(),
            recipe_to_sheet: IndexMap::new(),
            aggregation: IndexMap::new(),
            metadata: None,
        }
    }
}

fn default_aggregate_enabled() -> bool {
    true
}

fn default_aggregate_output_filename() -> String {
    "results.ods".to_string()
}

impl AggregateConfig {
    pub fn normalized_output_filename(&self) -> String {
        normalize_spreadsheet_filename(&self.output_filename)
    }

    pub fn normalized(&self) -> Self {
        let mut cfg = self.clone();
        cfg.output_filename = cfg.normalized_output_filename();
        cfg
    }

    /// Overlay process environment variables on top of this config.
    ///
    /// Thin wrapper around [`AggregateConfig::apply_env_overrides_from`]
    /// that reads from the real environment. Tests should prefer the
    /// `_from` variant with an injected lookup closure — mutating the
    /// process environment inside tests is both `unsafe` in modern Rust
    /// and not thread-safe under cargo test's parallel runner.
    pub fn apply_env_overrides(&mut self) {
        self.apply_env_overrides_from(|k| std::env::var(k).ok());
    }

    /// Overlay environment-variable settings on top of this config using
    /// the supplied lookup closure. Invalid / unparseable values are
    /// logged and ignored.
    ///
    /// Supported keys:
    /// - `BEKIPER_AGGREGATE_DISABLED` — when present and truthy
    ///   (`1`/`true`/`yes`/`on`, case-insensitive), sets `enabled = false`.
    /// - `BEKIPER_AGGREGATE_OUTPUT` — sets `output_filename`. An empty
    ///   value is rejected with a warning.
    pub fn apply_env_overrides_from<F>(&mut self, lookup: F)
    where
        F: Fn(&str) -> Option<String>,
    {
        if let Some(raw) = lookup("BEKIPER_AGGREGATE_DISABLED") {
            let trimmed = raw.trim().to_ascii_lowercase();
            match trimmed.as_str() {
                "1" | "true" | "yes" | "on" => {
                    tracing::info!("BEKIPER_AGGREGATE_DISABLED override applied");
                    self.enabled = false;
                }
                "0" | "false" | "no" | "off" => {
                    tracing::info!("BEKIPER_AGGREGATE_DISABLED=false override applied");
                    self.enabled = true;
                }
                "" => tracing::warn!("BEKIPER_AGGREGATE_DISABLED is empty, ignoring"),
                _ => tracing::warn!(
                    value = %raw,
                    "BEKIPER_AGGREGATE_DISABLED is not a boolean flag, ignoring"
                ),
            }
        }
        if let Some(raw) = lookup("BEKIPER_AGGREGATE_OUTPUT") {
            if raw.trim().is_empty() {
                tracing::warn!("BEKIPER_AGGREGATE_OUTPUT is empty, ignoring");
            } else {
                let normalized = normalize_spreadsheet_filename(&raw);
                tracing::info!(
                    output_filename = %normalized,
                    "BEKIPER_AGGREGATE_OUTPUT override applied"
                );
                self.output_filename = normalized;
            }
        }
    }
}

pub fn normalize_spreadsheet_filename(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return default_aggregate_output_filename();
    }

    let mut path = Path::new(trimmed).to_path_buf();
    path.set_extension("ods");
    path.to_string_lossy().into_owned()
}
