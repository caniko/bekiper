use serde_json::Value;

use super::trial::TrialResult;

/// A single cell value in a consolidated export row.
///
/// Deliberately an owned local enum rather than `polars::AnyValue<'static>`:
/// AnyValue's string variants require non-`'static` references or
/// `PlSmallStr`, both of which are awkward to wrangle through a trait object
/// at this layer. The later `build_sheet_dataframes` step maps these variants
/// to Polars series dtypes explicitly.
#[derive(Debug, Clone, PartialEq)]
pub enum CellValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    FloatList(Vec<f64>),
    String(String),
}

impl CellValue {
    /// Convert a borrowed `serde_json::Value` into a [`CellValue`].
    ///
    /// Arrays and objects collapse to `Null` — those are not representable as
    /// a single cell and must be handled by the caller (e.g. per-object NORT
    /// columns are expanded by position in `NortExperiment::extract_row`).
    pub fn from_json(v: &Value) -> CellValue {
        match v {
            Value::Null => CellValue::Null,
            Value::Bool(b) => CellValue::Bool(*b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    CellValue::Int(i)
                } else if let Some(u) = n.as_u64() {
                    // i64 overflow is theoretically possible for u64::MAX but
                    // no bekiper metric emits unsigned counts that large.
                    CellValue::Int(u as i64)
                } else if let Some(f) = n.as_f64() {
                    CellValue::Float(f)
                } else {
                    CellValue::Null // coverage: unreachable — without arbitrary_precision every number is i64, u64 or f64
                }
            }
            Value::String(s) => CellValue::String(s.clone()),
            Value::Array(_) | Value::Object(_) => CellValue::Null,
        }
    }
}

/// Pluck a single key from a JSON value, returning `CellValue::Null` for
/// missing keys or non-object roots.
pub(super) fn pluck(v: &Value, key: &str) -> CellValue {
    v.get(key)
        .map(CellValue::from_json)
        .unwrap_or(CellValue::Null)
}

pub(super) fn pluck_trial_metadata(trial: &TrialResult, key: &str) -> CellValue {
    let value = pluck(&trial.animal_metadata, key);
    if !matches!(value, CellValue::Null) {
        return value;
    }

    let context_key = match key {
        "animal" => "animal",
        "cohort" => "group",
        "gene" => "genotype",
        "sex" => "sex",
        "age" => "age",
        _ => return CellValue::Null,
    };
    pluck(&trial.session_context, context_key)
}
