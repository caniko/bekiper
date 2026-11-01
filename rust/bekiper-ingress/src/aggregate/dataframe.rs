use std::collections::HashSet;

use bekiper_core::error::{BekiperError, Result};
use indexmap::IndexMap;
use polars::prelude::*;

use super::cell::CellValue;

/// Compact tag describing what cell types appeared in a column across all
/// rows. Used to pick a single [`DataType`] for the resulting [`Series`].
#[derive(Debug, Default, Clone, Copy)]
struct ColumnSeen {
    has_string: bool,
    has_float_list: bool,
    has_float: bool,
    has_int: bool,
    has_bool: bool,
}

impl ColumnSeen {
    fn observe(&mut self, v: &CellValue) {
        match v {
            CellValue::Null => {}
            CellValue::Bool(_) => self.has_bool = true,
            CellValue::Int(_) => self.has_int = true,
            CellValue::Float(_) => self.has_float = true,
            CellValue::FloatList(_) => self.has_float_list = true,
            CellValue::String(_) => self.has_string = true,
        }
    }

    /// Pick a DataType using the precedence String > Float > Int > Bool.
    /// An all-null column falls back to [`DataType::String`] so it can be
    /// written as blank cells by the spreadsheet exporter.
    fn to_dtype(self) -> DataType {
        if self.has_string {
            DataType::String
        } else if self.has_float_list {
            DataType::List(Box::new(DataType::Float64))
        } else if self.has_float {
            // Any Float forces Int-mixed columns to Float64.
            DataType::Float64
        } else if self.has_int {
            DataType::Int64
        } else if self.has_bool {
            DataType::Boolean
        } else {
            DataType::String
        }
    }
}

/// Scan all rows and decide the [`DataType`] of a single column.
fn infer_column_dtype(rows: &[IndexMap<String, CellValue>], col: &str) -> DataType {
    let mut seen = ColumnSeen::default();
    for row in rows {
        if let Some(v) = row.get(col) {
            seen.observe(v);
        }
    }
    seen.to_dtype()
}

/// Build a typed [`Column`] for one column of the union.
///
/// For numeric dtypes, any cell that doesn't fit (e.g. a stray `Int` in a
/// `Float64` column) is coerced to the appropriate numeric. Any value whose
/// variant is wholly incompatible with the chosen dtype is written as
/// `None`, which appears as a blank cell in the spreadsheet output.
fn build_column(name: &str, dtype: &DataType, rows: &[IndexMap<String, CellValue>]) -> Column {
    let n = rows.len();
    let pl_name: PlSmallStr = name.into();
    match dtype {
        DataType::Float64 => {
            let mut vals: Vec<Option<f64>> = Vec::with_capacity(n);
            for row in rows {
                let v = match row.get(name) {
                    Some(CellValue::Float(f)) => Some(*f),
                    Some(CellValue::Int(i)) => Some(*i as f64),
                    Some(CellValue::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
                    _ => None,
                };
                vals.push(v);
            }
            Column::new(pl_name, vals)
        }
        DataType::Int64 => {
            let mut vals: Vec<Option<i64>> = Vec::with_capacity(n);
            for row in rows {
                let v = match row.get(name) {
                    Some(CellValue::Int(i)) => Some(*i),
                    _ => None,
                };
                vals.push(v);
            }
            Column::new(pl_name, vals)
        }
        DataType::Boolean => {
            let mut vals: Vec<Option<bool>> = Vec::with_capacity(n);
            for row in rows {
                let v = match row.get(name) {
                    Some(CellValue::Bool(b)) => Some(*b),
                    _ => None,
                };
                vals.push(v);
            }
            Column::new(pl_name, vals)
        }
        DataType::List(inner) if inner.as_ref() == &DataType::Float64 => {
            let list: ListChunked = rows
                .iter()
                .map(|row| match row.get(name) {
                    Some(CellValue::FloatList(values)) => {
                        Some(Series::new(PlSmallStr::EMPTY, values.as_slice()))
                    }
                    _ => None,
                })
                .collect();
            let mut series = list.into_series();
            series.rename(pl_name);
            series.into_column()
        }
        _ => {
            // DataType::String (also the all-null fallback).
            let mut vals: Vec<Option<String>> = Vec::with_capacity(n);
            for row in rows {
                let v = match row.get(name) {
                    Some(CellValue::String(s)) => Some(s.clone()),
                    Some(CellValue::Int(i)) => Some(i.to_string()),
                    Some(CellValue::Float(f)) => Some(f.to_string()),
                    Some(CellValue::FloatList(values)) => Some(json_f64_array(values)),
                    Some(CellValue::Bool(b)) => Some(b.to_string()),
                    _ => None,
                };
                vals.push(v);
            }
            Column::new(pl_name, vals)
        }
    }
}

/// Build a Polars [`DataFrame`] from a batch of trials by dispatching to
/// `extractor` for each row and unioning the resulting column sets.
///
/// Column order: the static schema from [`RowExtractor::columns`] first,
/// then any dynamic columns in first-seen order across the rows. This keeps
/// per-object NORT / per-zone columns adjacent to their peers without
/// requiring the extractor to pre-declare them.
pub(super) fn dataframe_from_extracted(
    base_columns: Vec<String>,
    extracted: Vec<IndexMap<String, CellValue>>,
) -> Result<DataFrame> {
    // Union of column names, preserving first-seen order. Static schema
    // seeds the order so canonical fields always come first; dynamic
    // per-object columns follow in trial order.
    let mut seen: HashSet<String> = HashSet::new();
    let mut order: Vec<String> = Vec::new();
    for col in base_columns {
        if seen.insert(col.clone()) {
            order.push(col);
        }
    }
    for row in &extracted {
        for key in row.keys() {
            if seen.insert(key.clone()) {
                order.push(key.clone());
            }
        }
    }

    let columns: Vec<Column> = order
        .iter()
        .map(|name| {
            let dtype = infer_column_dtype(&extracted, name);
            build_column(name, &dtype, &extracted)
        })
        .collect();

    DataFrame::new(extracted.len(), columns).map_err(BekiperError::Polars)
}

fn json_f64_array(values: &[f64]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}
