//! Cohort handoff from native trial results to the shared post-analysis engine.
//! Canonical identity and biological metadata come from the registered plan;
//! no session identifier or scientific approval is inferred from a filename.
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use bekiper_core::{
    error::{BekiperError, Result},
    types::TaskType,
};
use bekiper_post_analysis::{Record, RunOptions, inputs};
use serde_json::{Value, json};

use super::{CellValue, TrialResult, experiment_for_trial};
use crate::config::IngressConfig;
#[path = "post_analysis/descriptive.rs"]
mod descriptive;

pub fn export_post_analysis_trials(
    config: &IngressConfig,
    run_root: &Path,
    trials: &[TrialResult],
) -> Result<Option<Record>> {
    export_post_analysis_trials_with_cancel(config, run_root, trials, &|| false)
}

pub fn export_post_analysis_trials_with_cancel(
    config: &IngressConfig,
    run_root: &Path,
    trials: &[TrialResult],
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Option<Record>> {
    if !config.post_analysis.enabled {
        return Ok(None);
    }
    config.post_analysis.validate()?;
    let _lease = native(bekiper_post_analysis::pipeline::ProjectDataLease::export(
        run_root,
    ))?;
    run(config, run_root, trials, cancelled)
        .map(Some)
        .map_err(|error| {
            BekiperError::Data(format!("post-analysis cohort export failed: {error:#}"))
        })
}

fn native<T>(result: std::result::Result<T, impl std::fmt::Display>) -> Result<T> {
    result.map_err(|error| BekiperError::Data(format!("{error:#}")))
}

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if !condition {
        return Err(BekiperError::Data(message.into()));
    }
    Ok(())
}

include!("post_analysis/run.rs");

include!("post_analysis/output_settings.rs");

include!("post_analysis/trial_join.rs");

include!("post_analysis/session_identity.rs");

include!("post_analysis/nort_roles.rs");

#[cfg(test)]
#[path = "post_analysis/tests.rs"]
mod tests;
