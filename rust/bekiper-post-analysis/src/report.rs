//! Rust-owned report content and optional, non-shell-escaping LuaLaTeX build.
//! Both human formats share one document model so tables and limitations cannot
//! diverge between the Markdown handoff and the PDF.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Output};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::Record;

#[cfg(test)]
#[path = "report/tests.rs"]
mod tests;

#[derive(Default)]
struct Section {
    heading: String,
    paragraphs: Vec<String>,
    tables: Vec<Table>,
}

struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Section {
    fn new(heading: impl Into<String>) -> Self {
        Self {
            heading: heading.into(),
            ..Self::default()
        }
    }

    fn paragraph(&mut self, text: impl Into<String>) {
        self.paragraphs.push(text.into());
    }

    fn table(&mut self, headers: &[&str], rows: Vec<Vec<String>>) {
        if !rows.is_empty() {
            self.tables.push(Table {
                headers: headers.iter().map(|v| (*v).into()).collect(),
                rows,
            });
        }
    }
}

struct Figure {
    name: String,
    path: String,
    status: String,
    captions: Vec<String>,
    fingerprint: Value,
}

struct ReportOptions<'a> {
    enabled: bool,
    title: &'a str,
    compiler: &'a str,
}

impl<'a> ReportOptions<'a> {
    fn resolve(options: &'a Value) -> Result<Self> {
        let enabled = match options.get("enabled") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(value)) => *value,
            _ => bail!("outputs.report.enabled must be boolean"),
        };
        let title = match options.get("title") {
            None | Some(Value::Null) => "Behavioural post-analysis",
            Some(Value::String(value)) if !value.trim().is_empty() => value,
            _ => bail!("outputs.report.title must be a nonempty string"),
        };
        ensure!(title.len() <= 4096, "report title is too long");
        let compiler = match options.get("compiler") {
            None | Some(Value::Null) => "latexmk",
            Some(Value::String(value)) => value,
            _ => bail!("outputs.report.compiler must be a string"),
        };
        if enabled {
            ensure!(
                matches!(compiler, "latexmk" | "lualatex"),
                "report compiler must be latexmk or lualatex"
            );
        }
        Ok(Self {
            enabled,
            title,
            compiler,
        })
    }
}

/// Always emit the human Markdown handoff. PDF reporting is opt-in via
/// `outputs.report.enabled`; the caller supplies the final anticipated manifest
/// status, then only publishes it if this function succeeds.
pub fn write_report(
    directory: &Path,
    manifest: &Record,
    rows: &[Record],
    models: &[Record],
    ledger: &[Record],
    figures: &[Record],
    config: &Record,
) -> Result<Record> {
    write_report_with_runner(
        directory,
        manifest,
        rows,
        models,
        ledger,
        figures,
        config,
        &mut |mut command| command.output(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn write_report_with_cancel(
    directory: &Path,
    manifest: &Record,
    rows: &[Record],
    models: &[Record],
    ledger: &[Record],
    figures: &[Record],
    config: &Record,
    cancelled: &dyn Fn() -> bool,
) -> Result<Record> {
    let mut sequence = 0;
    write_report_with_runner(
        directory,
        manifest,
        rows,
        models,
        ledger,
        figures,
        config,
        &mut |mut command| {
            sequence += 1;
            let stdout = directory.join(format!("report.child.{sequence}.stdout.log"));
            let stderr = directory.join(format!("report.child.{sequence}.stderr.log"));
            let status = crate::pipeline::run_child(&mut command, &stdout, &stderr, cancelled)
                .map_err(std::io::Error::other)?;
            Ok(Output {
                status,
                stdout: fs::read(stdout)?,
                stderr: fs::read(stderr)?,
            })
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn write_report_with_runner(
    directory: &Path,
    manifest: &Record,
    rows: &[Record],
    models: &[Record],
    ledger: &[Record],
    figures: &[Record],
    config: &Record,
    runner: &mut dyn FnMut(Command) -> std::io::Result<Output>,
) -> Result<Record> {
    ensure!(
        directory.is_dir(),
        "report output directory must already exist"
    );
    let ReportOptions {
        enabled,
        title,
        compiler,
    } = ReportOptions::resolve(&config["outputs"]["report"])?;
    if enabled {
        fs::create_dir_all(directory.join("_internal/report"))?;
        ensure!(
            !directory.join("report.pdf").try_exists()?,
            "report.pdf already exists; use a fresh staged directory"
        );
    }
    let sections = report_sections(directory, manifest, rows, models, ledger, config)?;
    let selected_figures = if enabled {
        report_figures(directory, figures)?
    } else {
        Vec::new()
    };
    write_new(
        &directory.join("report.md"),
        render_markdown(title, &sections, figures).as_bytes(),
    )?;
    if !enabled {
        return Ok(
            json!({"status":"complete","pdf_status":"not_requested","files":["report.md"],
            "inputs":[fingerprint(&directory.join("report.md"))?]}),
        );
    }
    let mode = report_mode(|key| std::env::var(key).ok())?;
    write_new(
        &crate::export::path(directory, "report.tex"),
        render_tex(
            title,
            &sections,
            &selected_figures,
            themekit::Palette::for_mode(mode),
        )
        .as_bytes(),
    )?;
    let mut build = json!({"schema_version":"1.0","status":"building","engine":"LuaLaTeX",
        "color_mode":mode.as_str(),
        "compiler":compiler,"shell_escape":false,"source_date_epoch":0,
        "generator":{"package":env!("CARGO_PKG_NAME"),"version":env!("CARGO_PKG_VERSION"),
            "source_sha256":generator_source_hash()},
        "inputs":[fingerprint(&directory.join("report.md"))?,fingerprint(&crate::export::path(directory,"report.tex"))?],
        "record_inputs":[record_fingerprint("manifest_snapshot",manifest)?,record_fingerprint("metrics",&json!(rows))?,
            record_fingerprint("models",&json!(models))?,record_fingerprint("inclusion_ledger",&json!(ledger))?,
            record_fingerprint("config_snapshot",config)?],
        "figure_inputs":selected_figures.iter().map(|figure| figure.fingerprint.clone()).collect::<Vec<_>>(),
        "versions":{},"commands":[]});
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let compilation = compile(
        directory,
        compiler,
        runner,
        &mut build,
        &mut stdout,
        &mut stderr,
    );
    write_new(&directory.join("report.compiler.stdout.log"), &stdout)?;
    write_new(&directory.join("report.compiler.stderr.log"), &stderr)?;
    let compiled_pdf = directory.join("_internal/report/report.pdf");
    let pdf = directory.join("report.pdf");
    if let Err(error) = compilation.and_then(|_| validate_pdf(&compiled_pdf)) {
        build["status"] = "error".into();
        build["reason"] = error.to_string().into();
        write_new(
            &directory.join("report_build.json"),
            &serde_json::to_vec_pretty(&build)?,
        )?;
        return Err(error);
    }
    fs::rename(compiled_pdf, &pdf)?;
    build["status"] = "complete".into();
    build["output"] = fingerprint(&pdf)?;
    write_new(
        &directory.join("report_build.json"),
        &serde_json::to_vec_pretty(&build)?,
    )?;
    Ok(
        json!({"status":"complete","pdf_status":"complete","files":["report.md","report.tex","report.pdf",
        "report_build.json","report.compiler.stdout.log","report.compiler.stderr.log"],
        "compiler":compiler,"versions":build["versions"],"pdf":build["output"],
        "source":fingerprint(&crate::export::path(directory,"report.tex"))?}),
    )
}

mod compiler;
use compiler::compile;

mod scope;
use scope::{overview_section, rules_section};

mod inclusion;
use inclusion::inclusion_section;

mod supplementary;
use supplementary::{correlation_section, limitations_section, subgroup_sections, summary_section};

mod provenance;
use provenance::{organize_section_paths, provenance_section};

mod genotypes;
mod models;
use models::model_sections;

mod checks;
use checks::check_section;

mod figures;
use figures::report_figures;

mod markdown;
use markdown::render_markdown;

mod tex;
#[cfg(test)]
use tex::tex_escape;
use tex::{markdown_escape, render_tex, report_mode};

mod format;
use format::{descriptive_partition_label, detail, p_value, text};

mod artifacts;
use artifacts::{
    basename, config_input_hash, fingerprint, generator_source_hash, read_rows, record_fingerprint,
    validate_pdf, write_new,
};

fn report_sections(
    directory: &Path,
    manifest: &Record,
    rows: &[Record],
    models: &[Record],
    ledger: &[Record],
    config: &Record,
) -> Result<Vec<Section>> {
    if manifest["analysis_mode"] == "recorded_descriptive" {
        let mut scope = Section::new("Recorded descriptive measurements");
        scope.paragraph("Recorded native endpoints are presented without rescoring, registered models, primary-threshold selection, biological subgroup selection or inferential uncertainty. Pre/post, training/test, variants and scoring policies remain distinct. Repeated recordings remain individual observations; animal counts refer to distinct identities.");
        scope.paragraph("Conditional preference may double-count overlapping targets. Whole-session state partitions retain Missing, Unassigned and overlap, with their native counts, durations, bindings and denominators.");
        let mut partitions: BTreeMap<(String, String), (BTreeSet<String>, usize)> = BTreeMap::new();
        for row in rows {
            let label = format!(
                "{} / variant {} / {}",
                text(&row["presentation_label"]),
                text(&row["variant"]),
                descriptive_partition_label(&row["presentation_partition"])?
            );
            let partition = partitions
                .entry((
                    label,
                    serde_json::to_string(&row["presentation_partition"])?,
                ))
                .or_default();
            if let Some(animal) = row["animal_id"].as_str() {
                partition.0.insert(animal.into());
            }
            partition.1 += 1;
        }
        scope.table(
            &["Recorded population", "Distinct animals", "Observations"],
            partitions
                .into_iter()
                .map(|((label, _), (animals, n))| {
                    vec![label, animals.len().to_string(), n.to_string()]
                })
                .collect(),
        );
        let mut omissions = Section::new("Recorded exclusions and unavailable evidence");
        let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
        for entry in ledger {
            if let Some(reason) = entry["reason"].as_str() {
                *reasons.entry(reason.into()).or_default() += 1;
            }
        }
        omissions.table(
            &["Reason", "Ledger records"],
            reasons
                .into_iter()
                .map(|(reason, n)| vec![reason, n.to_string()])
                .collect(),
        );
        omissions.paragraph("The inclusion ledger records every endpoint observation, including explicit metadata exclusions and missing measurements. Full recorded session metadata, object-label roles and scoring provenance accompany metrics; per-record geometry and events do not create new biological populations.");
        let mut sections = vec![
            overview_section(manifest, rows),
            scope,
            omissions,
            inclusion_section(directory, rows, ledger)?,
            provenance_section(manifest)?,
        ];
        organize_section_paths(&mut sections);
        return Ok(sections);
    }
    let mut sections = vec![
        overview_section(manifest, rows),
        rules_section(directory, config)?,
        inclusion_section(directory, rows, ledger)?,
    ];
    sections.extend(model_sections(
        "Registered models and planned contrast",
        models,
    ));
    sections.extend(genotypes::sections(directory, manifest)?);
    let checks = read_rows(directory, "within_group_tests.json")?;
    let correlations = read_rows(directory, "correlations.json")?;
    sections.push(check_section(
        "Within-group memory / preference checks",
        &checks,
    ));
    sections.push(correlation_section(&correlations));
    sections.extend(subgroup_sections(directory)?);
    sections.push(summary_section(directory)?);
    sections.push(limitations_section(manifest, models, ledger));
    sections.push(provenance_section(manifest)?);
    organize_section_paths(&mut sections);
    Ok(sections)
}
