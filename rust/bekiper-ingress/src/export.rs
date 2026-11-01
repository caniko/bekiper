use std::path::Path;

use polars::prelude::*;

use bekiper_behaviour::{AnalysisResult, TaskDetail};
use bekiper_core::error::Result;
use bekiper_core::inspection::InspectionManifest;
use bekiper_feature::heuristic::HeuristicSummary;

use crate::config::AnimalMetadata;
use crate::spreadsheet::{SpreadsheetCell, SpreadsheetSheet, SpreadsheetWorkbook};
use crate::stage::ExperimentOutputDir;

/// Export inspection artifacts for a single analysis result.
///
/// Writes:
/// - `{experiment}_evaluated.parquet` — full DataFrame with coordinates + boolean columns
/// - `{experiment}_inspection.toml` — metadata sidecar for the Python inspection package
pub fn export_inspection(
    output_dir: &Path,
    experiment_name: &str,
    result: &AnalysisResult,
    manifest: &InspectionManifest,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    if let Some(ref df) = result.evaluated_df {
        let parquet_path = output_dir.join(format!("{experiment_name}_evaluated.parquet"));
        let mut df_clone = df.clone();
        let mut file = std::fs::File::create(&parquet_path)?;
        ParquetWriter::new(&mut file).finish(&mut df_clone)?;
        tracing::info!(?parquet_path, "wrote evaluation parquet");
    }

    let manifest_path = output_dir.join(format!("{experiment_name}_inspection.toml"));
    manifest.write_toml(&manifest_path)?;
    tracing::info!(?manifest_path, "wrote inspection manifest");

    Ok(())
}

/// Export all pipeline outputs as a multi-sheet OpenDocument spreadsheet (`.ods`).
///
/// Sheets:
/// - **poses** — raw pose estimation output
/// - **augmented** — filtered, smoothed, and coordinate-transformed poses
/// - **behaviour** — augmented poses + behavioural heuristic columns
/// - **summary** — per-heuristic summary statistics
/// - **detailed** — task-specific detailed results (zone summaries, indices, etc.)
pub fn export_spreadsheet(
    exp_dir: &ExperimentOutputDir,
    experiment_name: &str,
    summaries: &[HeuristicSummary],
    detailed: Option<&TaskDetail>,
    animal_metadata: Option<&AnimalMetadata>,
) -> Result<()> {
    export_spreadsheet_with_parameters(
        exp_dir,
        experiment_name,
        summaries,
        detailed,
        animal_metadata,
        None,
    )
}

/// [`export_spreadsheet`] plus the `parameters` sheet from `effective_parameters`.
pub fn export_spreadsheet_with_parameters(
    exp_dir: &ExperimentOutputDir,
    experiment_name: &str,
    summaries: &[HeuristicSummary],
    detailed: Option<&TaskDetail>,
    animal_metadata: Option<&AnimalMetadata>,
    parameters: Option<&serde_json::Value>,
) -> Result<()> {
    let mut frames = Vec::new();
    for (name, path) in [
        ("poses", exp_dir.poses_parquet()),
        ("augmented", exp_dir.augmented_parquet()),
        ("behaviour", exp_dir.behaviour_parquet()),
    ] {
        if path.exists() {
            frames.push((name, bekiper_pose::io::scan_file(&path)?.collect()?));
        }
    }
    export_spreadsheet_frames_with_parameters(
        exp_dir,
        experiment_name,
        summaries,
        detailed,
        animal_metadata,
        &frames,
        parameters,
    )
}

/// Export borrowed in-memory frames. DataFrame clones share column buffers.
pub fn export_spreadsheet_frames(
    exp_dir: &ExperimentOutputDir,
    experiment_name: &str,
    summaries: &[HeuristicSummary],
    detailed: Option<&TaskDetail>,
    animal_metadata: Option<&AnimalMetadata>,
    frames: &[(&str, DataFrame)],
) -> Result<()> {
    export_spreadsheet_frames_with_parameters(
        exp_dir,
        experiment_name,
        summaries,
        detailed,
        animal_metadata,
        frames,
        None,
    )
}

/// Like [`export_spreadsheet_frames`], plus a `parameters` sheet (`key`, `value`,
/// `source`) from a trial's `detailed.effective_parameters`.
pub fn export_spreadsheet_frames_with_parameters(
    exp_dir: &ExperimentOutputDir,
    experiment_name: &str,
    summaries: &[HeuristicSummary],
    detailed: Option<&TaskDetail>,
    animal_metadata: Option<&AnimalMetadata>,
    frames: &[(&str, DataFrame)],
    parameters: Option<&serde_json::Value>,
) -> Result<()> {
    let ods_path = exp_dir.root().join(format!("{experiment_name}.ods"));
    let mut workbook = SpreadsheetWorkbook::new();
    for (name, df) in frames {
        workbook.push_sheet(SpreadsheetSheet::from_dataframe(*name, df.clone()));
    }

    if !summaries.is_empty() {
        workbook.push_sheet(summary_sheet(summaries));
    }

    if let Some(detail_val) = detailed
        && let serde_json::Value::Object(obj) = detail_val.as_json_value()
    {
        workbook.push_sheet(detailed_sheet(&obj));
    }

    if let Some(sheet) = parameters.and_then(parameters_sheet) {
        workbook.push_sheet(sheet);
    }

    if let Some(meta) = animal_metadata {
        workbook.push_sheet(animal_sheet(meta));
    }

    workbook.save_ods(&ods_path)?;
    tracing::info!(output = %ods_path.display(), "exported OpenDocument spreadsheet");
    Ok(())
}

/// Backward-compatible alias retained while downstream callers migrate to
/// the format-neutral name.
pub fn export_excel(
    exp_dir: &ExperimentOutputDir,
    experiment_name: &str,
    summaries: &[HeuristicSummary],
    detailed: Option<&TaskDetail>,
    animal_metadata: Option<&AnimalMetadata>,
) -> Result<()> {
    export_spreadsheet(
        exp_dir,
        experiment_name,
        summaries,
        detailed,
        animal_metadata,
    )
}

fn summary_sheet(summaries: &[HeuristicSummary]) -> SpreadsheetSheet {
    let mut rows = Vec::with_capacity(summaries.len() + 1);
    rows.push(vec![
        SpreadsheetCell::string("heuristic"),
        SpreadsheetCell::string("true_frames"),
        SpreadsheetCell::string("total_frames"),
        SpreadsheetCell::string("fraction"),
        SpreadsheetCell::string("seconds"),
    ]);

    for summary in summaries {
        let fraction = if summary.total_frames > 0 {
            summary.true_frames as f64 / summary.total_frames as f64
        } else {
            0.0
        };
        rows.push(vec![
            SpreadsheetCell::string(summary.name.clone()),
            SpreadsheetCell::Int(summary.true_frames as i64),
            SpreadsheetCell::Int(summary.total_frames as i64),
            SpreadsheetCell::Float(fraction),
            SpreadsheetCell::Float(summary.seconds),
        ]);
    }

    SpreadsheetSheet::from_rows("summary", rows, 1)
}

fn parameters_sheet(parameters: &serde_json::Value) -> Option<SpreadsheetSheet> {
    let values = parameters.get("values")?.as_object()?;
    let mut rows = vec![
        ["key", "value", "source"]
            .map(SpreadsheetCell::string)
            .to_vec(),
    ];
    for (key, value) in values {
        let source = parameters["source"][key].as_str().unwrap_or("default");
        rows.push(vec![
            SpreadsheetCell::string(key.clone()),
            SpreadsheetCell::string(match value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            }),
            SpreadsheetCell::string(source),
        ]);
    }
    Some(SpreadsheetSheet::from_rows("parameters", rows, 1))
}

fn detailed_sheet(obj: &serde_json::Map<String, serde_json::Value>) -> SpreadsheetSheet {
    let mut rows = vec![vec![
        SpreadsheetCell::string("metric"),
        SpreadsheetCell::string("value"),
    ]];
    flatten_json_to_rows(obj, "", &mut rows);
    SpreadsheetSheet::from_rows("detailed", rows, 1)
}

fn animal_sheet(meta: &AnimalMetadata) -> SpreadsheetSheet {
    SpreadsheetSheet::from_rows(
        "animal",
        vec![
            vec![
                SpreadsheetCell::string("field"),
                SpreadsheetCell::string("value"),
            ],
            vec![
                SpreadsheetCell::string("animal"),
                SpreadsheetCell::Int(meta.animal as i64),
            ],
            vec![
                SpreadsheetCell::string("cohort"),
                SpreadsheetCell::string(meta.cohort.clone()),
            ],
            vec![
                SpreadsheetCell::string("gene"),
                SpreadsheetCell::string(meta.gene.clone()),
            ],
            vec![
                SpreadsheetCell::string("sex"),
                SpreadsheetCell::string(meta.sex.clone()),
            ],
            vec![
                SpreadsheetCell::string("age"),
                SpreadsheetCell::string(meta.age.clone()),
            ],
        ],
        1,
    )
}

fn flatten_json_to_rows(
    obj: &serde_json::Map<String, serde_json::Value>,
    prefix: &str,
    rows: &mut Vec<Vec<SpreadsheetCell>>,
) {
    for (key, value) in obj {
        let full_key = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };

        match value {
            serde_json::Value::Null => rows.push(vec![
                SpreadsheetCell::string(full_key),
                SpreadsheetCell::string("null"),
            ]),
            serde_json::Value::Bool(v) => rows.push(vec![
                SpreadsheetCell::string(full_key),
                SpreadsheetCell::Bool(*v),
            ]),
            serde_json::Value::Number(v) => {
                let cell = if let Some(i) = v.as_i64() {
                    SpreadsheetCell::Int(i)
                } else if let Some(u) = v.as_u64() {
                    i64::try_from(u)
                        .map(SpreadsheetCell::Int)
                        .unwrap_or_else(|_| SpreadsheetCell::string(u.to_string()))
                } else if let Some(f) = v.as_f64() {
                    SpreadsheetCell::Float(f)
                } else {
                    SpreadsheetCell::string(v.to_string()) // coverage: unreachable — without arbitrary_precision every number is i64, u64 or f64
                };
                rows.push(vec![SpreadsheetCell::string(full_key), cell]);
            }
            serde_json::Value::String(v) => rows.push(vec![
                SpreadsheetCell::string(full_key),
                SpreadsheetCell::string(v.clone()),
            ]),
            serde_json::Value::Array(values) => {
                for (idx, item) in values.iter().enumerate() {
                    let indexed_key = format!("{full_key}[{idx}]");
                    if let Some(inner) = item.as_object() {
                        flatten_json_to_rows(inner, &indexed_key, rows);
                    } else {
                        rows.push(vec![
                            SpreadsheetCell::string(indexed_key),
                            SpreadsheetCell::string(item.to_string()),
                        ]);
                    }
                }
            }
            serde_json::Value::Object(inner) => {
                flatten_json_to_rows(inner, &full_key, rows);
            }
        }
    }
}

/// Recursively walk an arbitrary JSON value, invoking `visit` on every
/// primitive leaf (`Null`, `Bool`, `Number`, `String`). Objects recurse via
/// dot-joined keys; arrays recurse via `[i]` suffixes.
///
/// Used by the aggregate-export generic row extractor to flatten arbitrary
/// `detailed` blobs into typed row cells.
pub(crate) fn for_each_json_leaf(
    prefix: &str,
    value: &serde_json::Value,
    visit: &mut dyn FnMut(&str, &serde_json::Value),
) {
    match value {
        serde_json::Value::Object(inner) => {
            for (key, child) in inner {
                let full_key = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                for_each_json_leaf(&full_key, child, visit);
            }
        }
        serde_json::Value::Array(arr) => {
            for (idx, item) in arr.iter().enumerate() {
                let indexed_key = format!("{prefix}[{idx}]");
                for_each_json_leaf(&indexed_key, item, visit);
            }
        }
        _ => visit(prefix, value),
    }
}

#[cfg(test)]
#[path = "export/tests.rs"]
mod tests;
