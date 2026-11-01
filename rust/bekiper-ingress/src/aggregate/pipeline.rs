use std::collections::HashSet;

use bekiper_core::error::Result;
use indexmap::IndexMap;
use polars::prelude::*;

use super::cell::CellValue;
use super::dataframe::dataframe_from_extracted;
use super::experiment::{Experiment, experiment_for_trial};
use super::identity::{IDENTITY_COLUMNS, resolve_aggregate_identity};
use super::metadata::METADATA_COLUMNS;
use super::trial::TrialResult;
use crate::config::{AggregateConfig, AggregationRule};

type AggregateRow = IndexMap<String, CellValue>;
type ExtractedSheetRows = Vec<AggregateRow>;
type ExtractedSheet = (Vec<String>, ExtractedSheetRows);
pub type AggregateSheet = (String, DataFrame, Box<dyn Experiment>);

fn has_aggregate_metadata(rows: &[&TrialResult]) -> bool {
    rows.iter().any(|trial| {
        trial.animal_metadata.get("group_name").is_some()
            || trial.animal_metadata.get("hcar1").is_some()
            || trial.animal_metadata.get("five_xfad").is_some()
    })
}

fn metadata_cell(trial: &TrialResult, column: &str) -> CellValue {
    trial
        .animal_metadata
        .get(column)
        .map(CellValue::from_json)
        .unwrap_or(CellValue::Null)
}

fn enriched_base_columns(base_columns: Vec<String>) -> Vec<String> {
    let mut columns: Vec<String> = IDENTITY_COLUMNS
        .iter()
        .chain(METADATA_COLUMNS.iter())
        .map(|column| (*column).into())
        .collect();
    for column in base_columns {
        if matches!(
            column.as_str(),
            "animal" | "cohort" | "gene" | "sex" | "age"
        ) {
            continue;
        }
        if !columns.iter().any(|existing| existing == &column) {
            columns.push(column);
        }
    }
    columns
}

fn enriched_row(trial: &TrialResult, row: AggregateRow) -> AggregateRow {
    let mut enriched = IndexMap::with_capacity(row.len() + METADATA_COLUMNS.len() + 3);
    let identity = resolve_aggregate_identity(trial, &row)
        .expect("identity is validated before enriched rows are constructed");
    insert_identity_columns(identity, trial, &mut enriched);
    for column in METADATA_COLUMNS {
        let value = match *column {
            "animal" => {
                let metadata_value = if identity.override_animal_column {
                    identity
                        .animal_cell()
                        .unwrap_or_else(|| metadata_cell(trial, column))
                } else {
                    metadata_cell(trial, column)
                };
                if matches!(metadata_value, CellValue::Null) {
                    row.get("animal").cloned().unwrap_or(CellValue::Null)
                } else {
                    metadata_value
                }
            }
            _ => metadata_cell(trial, column),
        };
        enriched.insert((*column).to_string(), value);
    }
    for (key, value) in row {
        if IDENTITY_COLUMNS.contains(&key.as_str()) {
            if enriched
                .get(&key)
                .is_none_or(|value| matches!(value, CellValue::Null))
            {
                enriched.insert(key, value);
            }
            continue;
        }
        if matches!(key.as_str(), "animal" | "cohort" | "gene" | "sex" | "age") {
            continue;
        }
        enriched.insert(key, value);
    }
    enriched
}

fn extracted_rows_for_sheet(
    experiment: &dyn Experiment,
    rows: &[&TrialResult],
) -> Result<ExtractedSheet> {
    if !has_aggregate_metadata(rows) {
        let columns = identity_base_columns(experiment.static_columns_for_trials(rows));
        let rows = rows
            .iter()
            .map(|trial| {
                let row = experiment.extract_row(trial);
                let identity = resolve_aggregate_identity(trial, &row)?;
                Ok(with_identity_columns(identity, trial, row))
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok((columns, rows));
    }

    Ok((
        enriched_base_columns(experiment.static_columns_for_trials(rows)),
        rows.iter()
            .map(|trial| {
                let row = experiment.extract_row(trial);
                resolve_aggregate_identity(trial, &row)?;
                Ok(enriched_row(trial, row))
            })
            .collect::<Result<Vec<_>>>()?,
    ))
}

fn sheet_key_for_trial(trial: &TrialResult) -> Option<String> {
    experiment_for_trial(trial)
        .sheet_key(trial)
        .map(|sheet_key| sheet_key.routing_key())
}

fn identity_base_columns(base_columns: Vec<String>) -> Vec<String> {
    let mut columns: Vec<String> = IDENTITY_COLUMNS
        .iter()
        .map(|column| (*column).to_string())
        .collect();
    for column in base_columns {
        if !columns.iter().any(|existing| existing == &column) {
            columns.push(column);
        }
    }
    columns
}

fn with_identity_columns(
    identity: super::identity::AggregateIdentity,
    trial: &TrialResult,
    row: AggregateRow,
) -> AggregateRow {
    let mut out = IndexMap::with_capacity(row.len() + 3);
    insert_identity_columns(identity, trial, &mut out);
    for (key, value) in row {
        if out
            .get(&key)
            .is_none_or(|value| matches!(value, CellValue::Null))
        {
            out.insert(key, value);
        }
    }
    out
}

fn insert_identity_columns(
    identity: super::identity::AggregateIdentity,
    trial: &TrialResult,
    out: &mut IndexMap<String, CellValue>,
) {
    for (column, value) in identity.columns(trial) {
        out.insert(column, value);
    }
}

fn value_from_series(df: &DataFrame, column: &str, row_idx: usize) -> CellValue {
    let Ok(series) = df.column(column) else {
        return CellValue::Null;
    };
    series
        .get(row_idx)
        .ok()
        .map(|value| match value {
            AnyValue::Null => CellValue::Null,
            AnyValue::Boolean(value) => CellValue::Bool(value),
            AnyValue::Int8(value) => CellValue::Int(value as i64),
            AnyValue::Int16(value) => CellValue::Int(value as i64),
            AnyValue::Int32(value) => CellValue::Int(value as i64),
            AnyValue::Int64(value) => CellValue::Int(value),
            AnyValue::UInt8(value) => CellValue::Int(value as i64), // coverage: unreachable — polars is built without dtype-u8
            AnyValue::UInt16(value) => CellValue::Int(value as i64), // coverage: unreachable — polars is built without dtype-u16
            AnyValue::UInt32(value) => CellValue::Int(value as i64),
            AnyValue::UInt64(value) => i64::try_from(value)
                .map(CellValue::Int)
                .unwrap_or_else(|_| CellValue::String(value.to_string())),
            AnyValue::Float32(value) => CellValue::Float(value as f64),
            AnyValue::Float64(value) => CellValue::Float(value),
            AnyValue::String(value) => CellValue::String(value.to_string()),
            AnyValue::StringOwned(value) => CellValue::String(value.to_string()),
            other => CellValue::String(other.to_string()),
        })
        .unwrap_or(CellValue::Null)
}

fn cell_to_key(value: &CellValue) -> Option<String> {
    match value {
        CellValue::Int(value) => Some(value.to_string()),
        CellValue::Float(value) if value.fract() == 0.0 => Some((*value as i64).to_string()),
        CellValue::String(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
        _ => None,
    }
}

pub(super) fn build_animal_overview_dataframe(
    sheets: &[AggregateSheet],
) -> Result<Option<DataFrame>> {
    let mut rows: IndexMap<String, IndexMap<String, CellValue>> = IndexMap::new();
    let mut schema: Vec<String> = METADATA_COLUMNS
        .iter()
        .map(|column| (*column).into())
        .collect();

    for (sheet_name, df, experiment) in sheets {
        let overview_columns = experiment.animal_overview_columns();
        for column in overview_columns {
            let overview_name = format!("{sheet_name}_{column}");
            if !schema.iter().any(|existing| existing == &overview_name) {
                schema.push(overview_name);
            }
        }
        if df.column("animal").is_err() {
            continue;
        }
        for row_idx in 0..df.height() {
            let animal_cell = value_from_series(df, "animal", row_idx);
            let Some(animal_key) = cell_to_key(&animal_cell) else {
                continue;
            };
            let row = rows.entry(animal_key).or_insert_with(|| {
                let mut row = IndexMap::new();
                for column in METADATA_COLUMNS {
                    row.insert(
                        (*column).to_string(),
                        value_from_series(df, column, row_idx),
                    );
                }
                row
            });

            for column in overview_columns {
                let overview_name = format!("{sheet_name}_{column}");
                row.insert(overview_name, value_from_series(df, column, row_idx));
            }
        }
    }

    if rows.is_empty() {
        return Ok(None);
    }

    let extracted = rows.into_values().collect::<Vec<_>>();
    dataframe_from_extracted(schema, extracted).map(Some)
}

/// Group rows by recipe -> sheet name, apply per-sheet aggregation rules, and
/// return one `(sheet_name, DataFrame)` per sheet.
///
/// Sheet emission order:
/// 1. Sheets reached via `cfg.recipe_to_sheet`, iterated in config insertion
///    order, deduped (multiple recipes may map to the same sheet). Entries
///    with no matching rows are skipped.
/// 2. Any remaining sheets (recipes with no rename entry), sorted
///    alphabetically.
///
/// Experiment dispatch: the experiment of the **first** trial in a sheet wins.
/// Mixed-experiment sheets are out of scope; a warning is logged if rows disagree.
/// One row per analysis, one column per `effective_parameters` key (union of
/// keys; empty where a task lacks it). `None` when no trial carries the block.
pub(super) fn build_parameters_dataframe(trials: &[TrialResult]) -> Result<Option<DataFrame>> {
    let rows: Vec<IndexMap<String, CellValue>> = trials
        .iter()
        .filter_map(|trial| {
            let values = trial
                .detailed
                .get("effective_parameters")?
                .get("values")?
                .as_object()?;
            let mut row = IndexMap::new();
            row.insert(
                "analysis_id".to_string(),
                CellValue::String(trial.experiment.clone()),
            );
            row.insert(
                "task".to_string(),
                CellValue::String(trial.task_type.to_string()),
            );
            for (key, value) in values {
                let cell = match value {
                    serde_json::Value::Bool(b) => CellValue::Bool(*b),
                    serde_json::Value::Number(n) => n
                        .as_i64()
                        .map(CellValue::Int)
                        .or_else(|| n.as_f64().map(CellValue::Float))
                        .unwrap_or(CellValue::Null),
                    serde_json::Value::String(s) => CellValue::String(s.clone()),
                    other => CellValue::String(other.to_string()),
                };
                row.insert(key.clone(), cell);
            }
            Some(row)
        })
        .collect();
    if rows.is_empty() {
        return Ok(None);
    }
    dataframe_from_extracted(vec!["analysis_id".into(), "task".into()], rows).map(Some)
}

pub fn build_sheet_dataframes(
    rows: Vec<TrialResult>,
    cfg: &AggregateConfig,
) -> Result<Vec<AggregateSheet>> {
    let mut groups: IndexMap<String, Vec<TrialResult>> = IndexMap::new();
    for trial in rows {
        let sheet_key = match sheet_key_for_trial(&trial) {
            Some(r) => r,
            None => {
                tracing::warn!(
                    experiment = %trial.experiment,
                    task_type = %trial.task_type,
                    "could not determine aggregate sheet key, skipping"
                );
                continue;
            }
        };
        let sheet_name = cfg
            .recipe_to_sheet
            .get(&sheet_key)
            .cloned()
            .unwrap_or(sheet_key);
        groups.entry(sheet_name).or_default().push(trial);
    }

    // Determine emission order. `recipe_to_sheet` preserves config insertion
    // order; we dedupe target sheet names to handle many->one rename maps
    // cleanly.
    let mut emission_order: Vec<String> = Vec::new();
    let mut placed: HashSet<String> = HashSet::new();
    for target in cfg.recipe_to_sheet.values() {
        if groups.contains_key(target) && placed.insert(target.clone()) {
            emission_order.push(target.clone());
        }
    }
    let mut leftover: Vec<String> = groups
        .keys()
        .filter(|k| !placed.contains(k.as_str()))
        .cloned()
        .collect();
    leftover.sort();
    emission_order.extend(leftover);

    let mut result: Vec<AggregateSheet> = Vec::with_capacity(emission_order.len());
    for sheet_name in emission_order {
        let bucket = match groups.swap_remove(&sheet_name) {
            Some(b) if !b.is_empty() => b,
            _ => continue, // coverage: unreachable — emission order lists each nonempty group once
        };

        let experiment = experiment_for_trial(&bucket[0]);
        let first_experiment_id = experiment.id();
        for extra in bucket.iter().skip(1) {
            let other_experiment_id = experiment_for_trial(extra).id();
            if other_experiment_id != first_experiment_id {
                tracing::warn!(
                    sheet = %sheet_name,
                    first = ?first_experiment_id,
                    other = ?other_experiment_id,
                    experiment = %extra.experiment,
                    "heterogeneous experiments within a sheet; using first trial's experiment"
                );
                break;
            }
        }

        let row_refs: Vec<&TrialResult> = bucket.iter().collect();
        let rule = cfg
            .aggregation
            .get(&sheet_name)
            .copied()
            .unwrap_or(AggregationRule::Passthrough);
        let (base_columns, extracted_rows) =
            extracted_rows_for_sheet(experiment.as_ref(), &row_refs)?;
        let final_df = match rule {
            AggregationRule::Passthrough => dataframe_from_extracted(base_columns, extracted_rows)?,
        };

        result.push((sheet_name, final_df, experiment));
    }

    Ok(result)
}
