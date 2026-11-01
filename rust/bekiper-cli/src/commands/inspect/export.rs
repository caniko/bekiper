//! Native trial-export orchestration. Rendering stays in the packaged plot backend;
//! discovery, population coverage, process lifetime and evidence belong to Rust.
use anyhow::{Context, Result, ensure};
use bekiper_ingress::{audit::PopulationLedger, config::PopulationDisposition};
use bekiper_post_analysis::{inputs, pipeline};
use clap::Args;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Args)]
pub struct ExportArgs {
    /// Completed analysis run containing `data/<trial>` inspection manifests.
    pub run_root: PathBuf,
    /// Captured/rebased plotting configuration.
    #[arg(long)]
    pub config: PathBuf,
    /// New export directory; occupied destinations are refused.
    #[arg(long)]
    pub output: PathBuf,
    /// Validated pre-filter population TOML/JSON. Requires complete admitted coverage.
    #[arg(long, conflicts_with = "trial")]
    pub population: Option<PathBuf>,
    /// Select a trial path relative to run/data; omitted means all discovered trials.
    #[arg(long)]
    pub trial: Vec<String>,
    /// Also render overview and available sensory videos for this selected trial.
    #[arg(long)]
    pub video_trial: Vec<String>,
}

struct Trial {
    name: String,
    analysis_id: String,
    manifest: PathBuf,
    table: PathBuf,
    manifest_identity: Value,
    table_identity: Value,
}

include!("export/discovery.rs");

include!("export/coverage.rs");

pub fn run(args: ExportArgs) -> Result<()> {
    render(args)?;
    Ok(())
}

include!("export/figure_verification.rs");

include!("export/renderer_process.rs");

include!("export/receipt_verification.rs");

fn render(args: ExportArgs) -> Result<Value> {
    render_with(args, "bekiper-inspect".as_ref())
}

include!("export/preparation.rs");

include!("export/render.rs");

include!("export/final_verification.rs");

#[cfg(test)]
mod tests;
