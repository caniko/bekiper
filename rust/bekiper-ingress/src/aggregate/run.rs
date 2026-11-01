use std::collections::HashSet;
use std::path::Path;

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::PerimetersFile;
use bekiper_core::types::TaskType;
use serde::{Deserialize, Serialize};

use super::experiment::experiment_for_trial;
use super::metadata::{MetadataExclusion, enrich_trials_with_metadata, load_aggregate_metadata};
use super::object_recognition::{
    NORT_EXPLORATION_DIAGNOSTICS_SHEET, build_nort_exploration_diagnostics_dataframe,
};
use super::pipeline::{
    build_animal_overview_dataframe, build_parameters_dataframe, build_sheet_dataframes,
};
use super::trial::TrialResult;
use crate::config::IngressConfig;
use crate::spreadsheet::{SpreadsheetCell, SpreadsheetSheet, SpreadsheetWorkbook};
use crate::stage::{aggregate_output_path, aggregate_state_path, checkpoint_root};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct AggregateFreshnessState {
    format_version: u32,
    aggregate: crate::config::AggregateConfig,
    metadata_modified_unix_nanos: Option<u128>,
}

impl AggregateFreshnessState {
    const FORMAT_VERSION: u32 = 1;

    fn from_config(config: &IngressConfig, run_root: &Path) -> Self {
        Self {
            format_version: Self::FORMAT_VERSION,
            aggregate: config.aggregate.normalized(),
            metadata_modified_unix_nanos: aggregate_metadata_modified_unix_nanos(config, run_root),
        }
    }
}

pub fn aggregate_output_path_for_config(
    config: &IngressConfig,
    run_root: &Path,
) -> std::path::PathBuf {
    aggregate_output_path(run_root, &config.aggregate.output_filename)
}

pub fn aggregate_state_path_for_config(
    config: &IngressConfig,
    run_root: &Path,
) -> std::path::PathBuf {
    aggregate_state_path(run_root, &config.aggregate.output_filename)
}

fn aggregate_result_scan_root(run_root: &Path) -> std::path::PathBuf {
    let data_root = run_root.join("data");
    if data_root.exists() {
        data_root
    } else {
        run_root.to_path_buf()
    }
}

/// Return whether the aggregate spreadsheet is newer than every discovered
/// per-trial `result.toml` under `run_root`.
pub fn is_aggregate_fresh(config: &IngressConfig, run_root: &Path) -> bool {
    let output_path = aggregate_output_path_for_config(config, run_root);
    if !output_path.exists() {
        return false;
    }
    let state_path = aggregate_state_path_for_config(config, run_root);
    let Ok(state_raw) = std::fs::read_to_string(&state_path) else {
        return false;
    };
    let Ok(saved_state) = toml::from_str::<AggregateFreshnessState>(&state_raw) else {
        return false;
    };
    if saved_state != AggregateFreshnessState::from_config(config, run_root) {
        return false;
    }
    let Ok(output_mtime) = std::fs::metadata(&output_path).and_then(|meta| meta.modified()) else {
        return false; // coverage: unreachable — the output existed a moment ago
    };
    let result_scan_root = aggregate_result_scan_root(run_root);
    let result_paths = collect_result_paths(&result_scan_root, 0, 2);
    if result_paths.is_empty() {
        return false;
    }
    result_paths.into_iter().all(|path| {
        std::fs::metadata(path)
            .and_then(|meta| meta.modified())
            .is_ok_and(|mtime| mtime <= output_mtime)
    })
}

include!("run/export.rs");

include!("run/metadata.rs");

include!("run/nort_validation.rs");

include!("run/ymaze_validation.rs");

include!("run/result_discovery.rs");

#[cfg(test)]
#[path = "run/tests.rs"]
mod tests;
