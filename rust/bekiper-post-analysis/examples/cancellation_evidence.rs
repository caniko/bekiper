//! Retain cancellation evidence after native scoring/export, before publication.
//! Usage: cargo run -p bekiper-post-analysis --example cancellation_evidence --
//!        <synthetic-config.json> <fresh-output-directory>
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result, ensure};
use bekiper_post_analysis::{RunOptions, inputs, pipeline};
use serde_json::json;

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let config_path =
        PathBuf::from(arguments.next().context("synthetic config required")?).canonicalize()?;
    let output = PathBuf::from(
        arguments
            .next()
            .context("fresh output directory required")?,
    );
    ensure!(
        arguments.next().is_none(),
        "expected config and output only"
    );
    ensure!(
        output.is_absolute() && !output.exists(),
        "use a fresh absolute output path"
    );
    let parent = output.parent().context("output parent required")?;
    fs::create_dir_all(parent)?;
    let name = output
        .file_name()
        .context("output name required")?
        .to_string_lossy();
    let prefix = format!(".{name}.staging-");
    ensure!(
        !fs::read_dir(parent)?
            .filter_map(Result::ok)
            .any(|entry| { entry.file_name().to_string_lossy().starts_with(&prefix) }),
        "use an output name without prior retained failure evidence"
    );
    let options = RunOptions {
        output: Some(output.clone()),
        requested_outputs: Some(vec!["metrics".into()]),
        ..Default::default()
    };
    let config = inputs::resolved_config(&config_path, &options, &|_| None)?;
    ensure!(
        config["run_mode"] == "synthetic",
        "synthetic demonstration only"
    );
    let (rows, fingerprints) = inputs::load_inputs(&config, config_path.parent().unwrap())?;
    let observed = AtomicBool::new(false);
    let cancelled = || {
        let exported = fs::read_dir(parent).is_ok_and(|entries| {
            entries.filter_map(Result::ok).any(|entry| {
                entry.file_name().to_string_lossy().starts_with(&prefix)
                    && entry.path().join("metrics.json").is_file()
            })
        });
        observed.fetch_or(exported, Ordering::Relaxed) || exported
    };
    let result = pipeline::run_records_with_cancel(
        config,
        rows,
        fingerprints,
        &config_path,
        &options,
        &cancelled,
    );
    ensure!(
        observed.load(Ordering::Relaxed) && result.is_err(),
        "export-stage cancellation was not reached"
    );
    ensure!(!output.exists(), "cancelled run was published");
    ensure!(
        !parent.join(format!(".{name}.claim")).exists(),
        "claim was not released"
    );
    let evidence = fs::read_dir(parent)?
        .filter_map(Result::ok)
        .find(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
        .context("failure evidence was not retained")?
        .path();
    let failure: serde_json::Value =
        serde_json::from_slice(&fs::read(evidence.join("failure_manifest.json"))?)?;
    ensure!(
        failure["status"] == "cancelled" && failure["published"] == false,
        "incorrect failure status"
    );
    ensure!(
        !evidence.join("run_manifest.json").exists(),
        "completed manifest survived"
    );
    let metrics: serde_json::Value =
        serde_json::from_slice(&fs::read(evidence.join("metrics.json"))?)?;
    println!(
        "{}",
        json!({
            "status":"cancelled", "published":false, "evidence":evidence,
            "retained_metric_rows":metrics.as_array().context("metrics array required")?.len(),
            "failure_manifest":failure, "code":bekiper_post_analysis::code_identity(),
        })
    );
    Ok(())
}
