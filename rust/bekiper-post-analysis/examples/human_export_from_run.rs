//! Make a fresh post-analysis-only presentation package from a verified run.
//! Usage: cargo run -p bekiper-post-analysis --example human_export_from_run --
//!        <verified-post-analysis-run> <fresh-export-directory>
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(args.next().context("verified source run required")?);
    let output = PathBuf::from(args.next().context("fresh export directory required")?);
    ensure!(
        args.next().is_none(),
        "expected source run and fresh export only"
    );
    let manifest = bekiper_post_analysis::export::export_existing(&input, &output)?;
    println!(
        "{}",
        serde_json::json!({"output":output,"status":manifest["status"],
        "run_mode":manifest["run_mode"],"real_data_execution":manifest["real_data_execution"],
        "presentation_export":manifest["presentation_export"]})
    );
    Ok(())
}
