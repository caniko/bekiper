//! Human-facing exports and the single, portable result-package layout.
//! Computation keeps its canonical records; presentation never rescores them.
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use serde::{Serialize, Serializer, ser::SerializeMap};
use serde_json::{Value, json};

use crate::Record;

#[path = "export/categories.rs"]
mod categories;
use categories::{Category, category};
#[path = "export/encoding.rs"]
mod encoding;
use encoding::{encode, pointer, write_toml};

#[path = "export/reexport.rs"]
mod reexport;
pub use reexport::export_existing;

pub const LAYOUT_VERSION: &str = "1.0";

/// Paths are relative to the published run. Legacy flat inputs remain readable
/// through `input_path`, but new publications have one canonical location.
pub fn relative_path(name: &str) -> String {
    if name.contains('/') {
        return name.into();
    }
    match name {
        "README.md"
        | "metrics.toml"
        | "parameters.toml"
        | "report.pdf"
        | "report.md"
        | "run_manifest.json"
        | "artifact_manifest.json"
        | "failure_manifest.json" => name.into(),
        "models.json"
        | "correlations.json"
        | "within_group_tests.json"
        | "subgroup_analyses.json"
        | "genotype_policies.json"
        | "summaries.json" => format!("statistics/{name}"),
        "inclusion_ledger.json"
        | "pairing_ledger.json"
        | "event_bout_debug.json"
        | "decision_report.json"
        | "native_comparisons.json"
        | "supplementary_statuses.json" => format!("audit/{name}"),
        "BEHAVIORAL_RECIPE.md" | "YMAZE_AMENDMENT.json" | "report_build.json" => {
            format!("provenance/{name}")
        }
        "metrics.json" | "plotting_data.json" | "plotting_bundle.json" => {
            format!("_internal/{name}")
        }
        _ if name.ends_with(".log") && !name.starts_with("report.") => {
            format!("_internal/logs/{name}")
        }
        "report.compiler.stdout.log" | "report.compiler.stderr.log" => {
            format!("_internal/logs/{name}")
        }
        _ if name.starts_with("report.") => format!("_internal/report/{name}"),
        _ if ["figure_", "supplement_", "qc_", "subgroup_"]
            .iter()
            .any(|prefix| name.starts_with(prefix)) =>
        {
            let assay = crate::contracts::assays_in_report_order()
                .iter().copied()
                .find(|assay| name.split(['_', '.']).any(|part| part == *assay))
                .unwrap_or("supplementary");
            format!("figures/{assay}/{name}")
        }
        _ => format!("_internal/other/{name}"),
    }
}

pub fn path(directory: &Path, name: &str) -> PathBuf {
    directory.join(relative_path(name))
}

pub fn input_path(directory: &Path, name: &str) -> PathBuf {
    let organized = path(directory, name);
    if organized.exists() {
        organized
    } else {
        directory.join(name)
    }
}

fn move_artifact(directory: &Path, name: &str) -> Result<()> {
    ensure!(
        crate::pipeline::safe_relative_path(name) && !name.contains('/'),
        "unsafe artifact basename"
    );
    let source = directory.join(name);
    let destination = path(directory, name);
    if source == destination {
        return Ok(());
    }
    ensure!(
        source.symlink_metadata()?.is_file(),
        "artifact must be a regular file"
    );
    ensure!(
        !destination.try_exists()?,
        "organized artifact already exists"
    );
    fs::create_dir_all(destination.parent().context("artifact parent")?)?;
    fs::rename(source, destination)?;
    Ok(())
}

pub fn organize_figures(directory: &Path, figures: &mut [Record]) -> Result<()> {
    for figure in figures {
        let name = figure["name"]
            .as_str()
            .context("figure needs name")?
            .to_owned();
        ensure!(
            !name.is_empty()
                && name.len() <= 128
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')),
            "unsafe figure name"
        );
        let formats = figure["formats"]
            .as_array()
            .context("figure needs formats")?;
        let mut files = Vec::new();
        for format in formats {
            let format = format.as_str().context("figure format")?;
            ensure!(
                ["svg", "png", "pdf"].contains(&format),
                "unsupported figure format"
            );
            let filename = format!("{name}.{format}");
            if directory.join(&filename).try_exists()? {
                move_artifact(directory, &filename)?;
            } else {
                ensure!(
                    path(directory, &filename).symlink_metadata()?.is_file(),
                    "organized figure must be a regular file"
                );
            }
            files.push(relative_path(&filename));
        }
        figure["files"] = json!(files);
    }
    Ok(())
}

#[derive(Default, Serialize)]
struct AssayMetrics {
    sessions: Vec<SessionMetrics>,
}

#[derive(Serialize)]
struct MetricsExport {
    schema_version: &'static str,
    run_mode: Value,
    status: Value,
    encoding: &'static str,
    oft: AssayMetrics,
    ymaze: AssayMetrics,
    nort: AssayMetrics,
}

#[derive(Serialize)]
struct SessionMetrics {
    source_record_index: usize,
    #[serde(flatten)]
    identity: BTreeMap<String, toml::Value>,
    null_fields: Vec<String>,
    large_integer_fields: Vec<String>,
    design: toml::Table,
    parameters: toml::Table,
    primary: toml::Table,
    secondary: toml::Table,
    measurements: toml::Table,
    availability: toml::Table,
    qc: toml::Table,
    missingness: toml::Table,
    verification: toml::Table,
    comparisons: toml::Table,
    provenance: toml::Table,
    additional_fields: toml::Table,
}

fn session(row: &Record, index: usize) -> Result<SessionMetrics> {
    let assay = row["assay"].as_str().context("metric requires assay")?;
    let mut groups: BTreeMap<Category, toml::Table> = BTreeMap::new();
    let mut null_fields = Vec::new();
    let mut large_integer_fields = Vec::new();
    for (field, value) in row.as_object().context("metric must be a record")? {
        let group = category(assay, field, value);
        let location = if group == Category::Identity {
            pointer("", field)
        } else {
            pointer(&pointer("", group.name()), field)
        };
        groups.entry(group).or_default().insert(
            field.clone(),
            encode(
                value,
                &location,
                "unavailable",
                &mut null_fields,
                &mut large_integer_fields,
            )?, // coverage: unreachable — serde_json numbers are finite i64/u64/f64, so encode cannot fail
        );
    }
    Ok(SessionMetrics {
        source_record_index: index,
        identity: groups
            .remove(&Category::Identity)
            .unwrap_or_default()
            .into_iter()
            .collect(),
        null_fields,
        large_integer_fields,
        design: groups.remove(&Category::Design).unwrap_or_default(),
        parameters: groups.remove(&Category::Parameters).unwrap_or_default(),
        primary: groups.remove(&Category::Primary).unwrap_or_default(),
        secondary: groups.remove(&Category::Secondary).unwrap_or_default(),
        measurements: groups.remove(&Category::Measurements).unwrap_or_default(),
        availability: groups.remove(&Category::Availability).unwrap_or_default(),
        qc: groups.remove(&Category::Qc).unwrap_or_default(),
        missingness: groups.remove(&Category::Missingness).unwrap_or_default(),
        verification: groups.remove(&Category::Verification).unwrap_or_default(),
        comparisons: groups.remove(&Category::Comparisons).unwrap_or_default(),
        provenance: groups.remove(&Category::Provenance).unwrap_or_default(),
        additional_fields: groups
            .remove(&Category::AdditionalFields)
            .unwrap_or_default(),
    })
}

fn write_metrics(directory: &Path, manifest: &Record, rows: &[Record]) -> Result<()> {
    let mut document = MetricsExport {
        schema_version: LAYOUT_VERSION,
        run_mode: manifest["run_mode"].clone(),
        status: manifest["status"].clone(),
        encoding: "explicit_null_and_large_integer_paths_v1",
        oft: AssayMetrics::default(),
        ymaze: AssayMetrics::default(),
        nort: AssayMetrics::default(),
    };
    let mut sorted: Vec<_> = rows.iter().enumerate().collect();
    sorted.sort_by_key(|(_, row)| {
        (
            row["animal_id"].to_string(),
            match row["timepoint"].as_str() {
                Some("pre") => 0,
                _ => 1,
            },
            row["session_id"].to_string(),
            row["phase"].to_string(),
            row["threshold_pct"].to_string(),
        )
    });
    for (index, row) in sorted {
        let target = match row["assay"].as_str() {
            Some("oft") => &mut document.oft,
            Some("ymaze") => &mut document.ymaze,
            Some("nort") => &mut document.nort,
            _ => anyhow::bail!("unsupported human metric assay"),
        };
        target.sessions.push(session(row, index)?);
    }
    write_toml(
        &directory.join("metrics.toml"),
        "Categorized session metrics. Field suffixes retain units (s, m, m/s, pct). Primary/secondary refer to recipe endpoint roles, not inclusion eligibility.",
        &document,
    )
}

/// Bind every human session and its exact null/integer encoding to canonical scored records.
pub fn verify_metrics_export(directory: &Path, manifest: &Record, rows: &[Record]) -> Result<()> {
    let human: toml::Value = toml::from_str(&fs::read_to_string(directory.join("metrics.toml"))?)?;
    ensure!(
        human.get("schema_version").and_then(toml::Value::as_str) == Some(LAYOUT_VERSION)
            && human.get("run_mode").and_then(toml::Value::as_str) == manifest["run_mode"].as_str()
            && human.get("status").and_then(toml::Value::as_str) == manifest["status"].as_str(),
        "human metric publication identity differs"
    );
    let mut seen = std::collections::BTreeSet::new();
    for assay in bekiper_core::experiment::post_analysis_descriptors()
        .filter_map(|d| d.post_analysis.as_ref())
        .map(|p| p.assay.as_str())
    {
        for record in human
            .get(assay)
            .and_then(|a| a.get("sessions"))
            .and_then(toml::Value::as_array)
            .context("human session inventory")?
        {
            let index = usize::try_from(
                record
                    .get("source_record_index")
                    .and_then(toml::Value::as_integer)
                    .context("human session source index")?,
            )?;
            let row = rows
                .get(index)
                .context("human session source index exceeds canonical records")?;
            ensure!(
                row["assay"] == assay && seen.insert(index),
                "human session is repeated or assigned to a different assay"
            );
            ensure!(
                *record == toml::Value::try_from(session(row, index)?)?,
                "human metrics differ from canonical session {index}"
            );
        }
    }
    ensure!(
        seen.len() == rows.len(),
        "human metrics omit canonical sessions"
    );
    Ok(())
}

/// Put scientific settings before fingerprints, upstream records and encoding
/// metadata, so the file opens with the parameters a researcher needs.
struct OrderedParameters(toml::Table);

impl Serialize for OrderedParameters {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let preferred = [
            "schema_version",
            "run_mode",
            "synthetic_fixture",
            "assays",
            "statistics",
            "inclusion",
            "decision_resolutions",
            "outputs",
            "inputs",
            "source",
            "source_recipe",
            "ymaze_amendment",
            "code",
            "input_fingerprints",
            "presentation_export",
            "upstream_sessions",
            "reserved_configuration_fields",
            "representation",
        ];
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for name in preferred {
            if let Some(value) = self.0.get(name) {
                map.serialize_entry(name, value)?;
            }
        }
        for (name, value) in &self.0 {
            if !preferred.contains(&name.as_str()) {
                map.serialize_entry(name, value)?;
            }
        }
        map.end()
    }
}

fn write_parameters(directory: &Path, manifest: &Record, rows: &[Record]) -> Result<()> {
    let mut settings = manifest["resolved_config"]
        .as_object()
        .context("resolved configuration required")?
        .clone();
    let mut reserved = serde_json::Map::new();
    for key in [
        "schema_version",
        "code",
        "source_recipe",
        "ymaze_amendment",
        "input_fingerprints",
        "upstream_sessions",
        "presentation_export",
        "null_fields",
        "large_integer_fields",
        "reserved_configuration_fields",
        "representation",
    ] {
        if let Some(value) = settings.remove(key) {
            reserved.insert(key.into(), value);
        }
    }
    if !reserved.is_empty() {
        settings.insert(
            "reserved_configuration_fields".into(),
            Value::Object(reserved),
        );
    }
    settings.insert("schema_version".into(), json!(LAYOUT_VERSION));
    settings.insert("code".into(), manifest["code"].clone());
    settings.insert("source_recipe".into(), manifest["source_recipe"].clone());
    settings.insert(
        "ymaze_amendment".into(),
        manifest["ymaze_amendment"].clone(),
    );
    settings.insert("input_fingerprints".into(), manifest["inputs"].clone());
    if let Some(export) = manifest.get("presentation_export") {
        settings.insert("presentation_export".into(), export.clone());
    }
    settings.insert(
        "upstream_sessions".into(),
        json!(
            rows.iter()
                .enumerate()
                .map(|(index, row)| {
                    let mut value = json!({"source_record_index":index});
                    for (key, field) in row.as_object().into_iter().flatten() {
                        if matches!(
                            category(row["assay"].as_str().unwrap_or(""), key, field),
                            Category::Identity | Category::Parameters | Category::Verification
                        ) || matches!(
                            key.as_str(),
                            "scoring_provenance" | "effective_parameters"
                        ) {
                            value[key] = field.clone();
                        }
                    }
                    value
                })
                .collect::<Vec<_>>()
        ),
    );
    let mut nulls = Vec::new();
    let mut integers = Vec::new();
    let mut value = encode(
        &Value::Object(settings),
        "",
        "not_specified",
        &mut nulls,
        &mut integers,
    )?; // coverage: unreachable — serde_json numbers are finite i64/u64/f64, so encode cannot fail
    let table = value.as_table_mut().context("parameter table")?;
    table.insert(
        "representation".into(),
        toml::Value::Table(toml::Table::from_iter([
            ("null_fields".into(), toml::Value::try_from(nulls)?),
            (
                "large_integer_fields".into(),
                toml::Value::try_from(integers)?,
            ),
        ])),
    );
    write_toml(
        &directory.join("parameters.toml"),
        "Resolved post-analysis parameters and available per-session upstream rule evidence. Unspecified settings are not inferred approvals; source_recipe identifies fixed scientific definitions.",
        &OrderedParameters(table.clone()),
    )
}

/// Called inside the unpublished staging transaction, after native/report work.
pub fn finalize(directory: &Path, manifest: &mut Record, rows: &[Record]) -> Result<()> {
    write_metrics(directory, manifest, rows)?;
    write_parameters(directory, manifest, rows)?;
    let paths = organize_artifacts(directory, manifest)?;
    manifest["package_layout"] = json!({"schema_version":LAYOUT_VERSION,"paths":paths});
    if let Some(files) = manifest["report"]["files"].as_array_mut() {
        for file in files {
            if let Some(name) = file.as_str() {
                *file = json!(relative_path(name));
            }
        }
    }
    write_index(directory, manifest)
}

fn organize_artifacts(
    directory: &Path,
    manifest: &Record,
) -> Result<serde_json::Map<String, Value>> {
    let names = fs::read_dir(directory)?
        .map(|entry| {
            let entry = entry?;
            Ok(if entry.file_type()?.is_dir() {
                None
            } else {
                Some(
                    entry
                        .file_name()
                        .into_string()
                        .map_err(|_| anyhow::anyhow!("artifact name must be UTF-8"))?,
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut paths = serde_json::Map::new();
    for name in names.into_iter().flatten() {
        move_artifact(directory, &name)?;
        paths.insert(name.clone(), json!(relative_path(&name)));
    }
    for figure in manifest["figures"].as_array().into_iter().flatten() {
        for file in figure["files"].as_array().into_iter().flatten() {
            let file = file.as_str().context("figure path")?;
            paths.insert(
                Path::new(file)
                    .file_name()
                    .context("figure basename")?
                    .to_string_lossy()
                    .into_owned(),
                json!(file),
            );
        }
    }
    paths.insert("README.md".into(), json!("README.md"));
    paths.insert("run_manifest.json".into(), json!("run_manifest.json"));
    paths.insert(
        "artifact_manifest.json".into(),
        json!("artifact_manifest.json"),
    );
    // Already-organized artifacts (figures and TeX sources) are discovered too.
    for member in crate::pipeline::artifact_inventory(directory)? {
        let relative = member["path"].as_str().context("artifact path")?;
        paths.entry(relative.to_owned()).or_insert(json!(relative));
        let basename = Path::new(relative)
            .file_name()
            .context("artifact basename")?
            .to_str()
            .context("artifact basename must be UTF-8")?;
        if relative_path(basename) == relative {
            paths.entry(basename.to_owned()).or_insert(json!(relative));
        }
    }
    Ok(paths)
}

fn write_index(directory: &Path, manifest: &Record) -> Result<()> {
    let mut index = format!(
        "# Behavioural post-analysis\n\nRun mode: **{}**. Status: **{}**. Real-data execution: **{}**.\n\n## Start here\n\n- [Categorized session metrics](metrics.toml): OFT, Y-maze and NORT; units remain in field names.\n- [Analysis parameters](parameters.toml): resolved settings, registrations and available upstream evidence.\n- [Run manifest](run_manifest.json): output-specific status, limitations and source identities.\n\n",
        manifest["run_mode"].as_str().unwrap_or("unknown"),
        manifest["status"].as_str().unwrap_or("unknown"),
        manifest["real_data_execution"]
    );
    if directory.join("report.pdf").is_file() {
        index.push_str("- [PDF report](report.pdf)\n");
    }
    if directory.join("report.md").is_file() {
        index.push_str("- [Readable report](report.md)\n");
    }
    if manifest["presentation_export"].is_object() {
        index.push_str("\nThis is a presentation-only export of an existing verified post-analysis run. Scientific measurements, fits and the analysis code identity are preserved. The presentation exporter and source inventory are recorded separately in `run_manifest.json` and `parameters.toml`.\n");
    }
    index.push_str("\n## Browse the evidence\n\n");
    for (title, names) in [
        (
            "Statistical results",
            vec![
                "models.json",
                "summaries.json",
                "correlations.json",
                "within_group_tests.json",
                "subgroup_analyses.json",
                "genotype_policies.json",
            ],
        ),
        (
            "Inclusion, QC and decisions",
            vec![
                "inclusion_ledger.json",
                "pairing_ledger.json",
                "decision_report.json",
                "event_bout_debug.json",
                "native_comparisons.json",
                "supplementary_statuses.json",
            ],
        ),
        (
            "Provenance",
            vec![
                "BEHAVIORAL_RECIPE.md",
                "YMAZE_AMENDMENT.json",
                "report_build.json",
            ],
        ),
    ] {
        index.push_str(&format!("\n### {title}\n\n"));
        for name in names {
            if path(directory, name).is_file() {
                index.push_str(&format!("- [{name}]({})\n", relative_path(name)));
            }
        }
    }
    index.push_str("\n### Figures\n\n");
    for figure in manifest["figures"].as_array().into_iter().flatten() {
        for file in figure["files"].as_array().into_iter().flatten() {
            let name = file.as_str().context("figure path")?;
            index.push_str(&format!("- [{name}]({name})\n"));
        }
    }
    index.push_str("\n## Reading conventions\n\nQC flags, exclusions, missing measurements and unresolved scientific decisions are distinct. `state = \"unavailable\"` is not zero; `state = \"not_specified\"` is not approval. Session `null_fields` (parameters: `representation.null_fields`) records exact JSON-pointer locations of original nulls, including array elements. `large_integer_fields` records exact integers represented as decimal strings. Session `source_record_index` points into [_internal/metrics.json](_internal/metrics.json); every canonical field is retained in its category. Full-precision values are exported. Event ordering remains in the audit records. `_internal/` contains machine payloads, logs and report-build intermediates. [artifact_manifest.json](artifact_manifest.json) hashes every published file.\n");
    fs::write(directory.join("README.md"), index)?;
    Ok(())
}
