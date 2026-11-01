//! Compile a report from an existing auditable export without running inference.
//! Usage: cargo run -p bekiper-post-analysis --example report_from_export --
//!        <input-export> <fresh-report-directory> [latexmk|lualatex]
use anyhow::{Context, Result, ensure};
use bekiper_post_analysis::{Record, report::write_report};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn array(input: &Path, name: &str) -> Result<Vec<Record>> {
    read(&bekiper_post_analysis::export::input_path(input, name))?
        .as_array()
        .cloned()
        .with_context(|| format!("{name} must be an array"))
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        (2..=3).contains(&args.len()),
        "usage: report_from_export <input-export> <fresh-output> [latexmk|lualatex]"
    );
    let input = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    std::fs::create_dir(&output).context("report output must be fresh with an existing parent")?;
    let manifest = read(&input.join("run_manifest.json"))?;
    let rows = array(&input, "metrics.json")?;
    let models = array(&input, "models.json")?;
    let ledger = array(&input, "inclusion_ledger.json")?;
    let mut figures = manifest["figures"].as_array().cloned().unwrap_or_default();
    for name in [
        "summaries.json",
        "decision_report.json",
        "pairing_ledger.json",
        "within_group_tests.json",
        "correlations.json",
        "subgroup_analyses.json",
    ] {
        let path = bekiper_post_analysis::export::input_path(&input, name);
        if path.is_file() {
            std::fs::copy(path, output.join(name))?;
        }
    }
    for figure in &figures {
        let name = figure["name"].as_str().context("figure name")?;
        ensure!(
            name.bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')),
            "unsafe figure name"
        );
        for format in figure["formats"].as_array().context("figure formats")? {
            let format = format.as_str().context("figure format")?;
            ensure!(
                ["svg", "png", "pdf"].contains(&format),
                "unsupported figure format"
            );
            let filename = format!("{name}.{format}");
            std::fs::copy(
                bekiper_post_analysis::export::input_path(&input, &filename),
                output.join(filename),
            )?;
        }
    }
    bekiper_post_analysis::export::organize_figures(&output, &mut figures)?;
    let mut config = manifest["resolved_config"].clone();
    if !config.is_object() {
        config = json!({});
    }
    if !config["outputs"].is_object() {
        config["outputs"] = json!({});
    }
    config["outputs"]["report"] = json!({"enabled":true,"title":"Behavioural post-analysis",
        "compiler":args.get(2).map(|arg|arg.to_string_lossy().into_owned()).unwrap_or_else(||"latexmk".into())});
    let status = write_report(
        &output, &manifest, &rows, &models, &ledger, &figures, &config,
    )?;
    println!("{status}");
    Ok(())
}
