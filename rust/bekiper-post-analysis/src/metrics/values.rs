//! Values arithmetic and scoring contracts.
use super::*;

pub fn is_count_field(field: &str) -> bool {
    [
        "center_entries",
        "fecal_boli_count",
        "raw_total_arm_entries",
        "valid_total_arm_entries",
        "dwell_rejected_entries",
        "total_triads",
        "usable_triads",
        "correct_alternations",
        "incorrect_alternations",
        "neutral_triads",
        "valid_bouts",
        "removed_duration_bouts",
        "removed_speed_bouts",
        "removed_angle_bouts",
    ]
    .contains(&field)
        || ["raw", "valid"].iter().any(|stage| {
            ["A", "B", "C"]
                .iter()
                .any(|arm| field == format!("{stage}_arm_{arm}_entries"))
        })
}

pub fn nonnegative(value: &Value, name: &str, maximum: Option<f64>) -> Result<f64> {
    let number = value
        .as_f64()
        .with_context(|| format!("{name}: requires a finite nonnegative number"))?;
    ensure!(
        number.is_finite() && number >= 0.0,
        "{name}: requires a finite nonnegative number"
    );
    if let Some(maximum) = maximum {
        ensure!(number <= maximum, "{name}: exceeds {maximum}");
    }
    Ok(number)
}

/// JSON numbers compare arithmetically, while booleans never equal numbers.
/// Rule audits must retain exact inequalities and constants, without rounding.
pub fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(_), Value::Number(_)) => a.as_f64() == b.as_f64(),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_value(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, value)| b.get(key).is_some_and(|b| same_value(value, b)))
        }
        _ => a == b,
    }
}

pub(super) fn labels(sequence: &Value) -> Result<Vec<String>> {
    let labels = if let Some(sequence) = sequence.as_str() {
        sequence
            .chars()
            .filter(|c| *c != ' ')
            .map(|c| c.to_string())
            .collect()
    } else {
        sequence
            .as_array()
            .context("valid_arm_entry_sequence must be a string or list")?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("Arm labels must be strings")
            })
            .collect::<Result<Vec<_>>>()?
    };
    ensure!(
        labels
            .iter()
            .all(|label| matches!(label.as_str(), "A" | "B" | "C")),
        "valid_arm_entry_sequence: only A/B/C labels are supported"
    );
    Ok(labels)
}
pub(super) fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8_f64.max(1e-9 * a.abs().max(b.abs()))
}

pub(super) fn validate_count(value: &Value, field: &str) -> Result<f64> {
    let number = nonnegative(value, field, Some(9_007_199_254_740_991.0))?;
    ensure!(number.fract() == 0.0, "{field}: requires an integer count");
    Ok(number)
}

/// Differences retain signed historical raw-minus-scored values. They are
/// integers, but cannot share the nonnegative domain of actual entry counts.
pub(super) fn validate_entry_difference(value: &Value, field: &str) -> Result<()> {
    ensure!(
        value.as_f64().is_some_and(|number| number.is_finite()
            && number.abs() <= 9_007_199_254_740_991.0
            && number.fract() == 0.0),
        "{field}: requires an exact signed integer entry difference"
    );
    Ok(())
}

pub(super) fn optional_number(row: &Record, field: &str) -> Result<Option<f64>> {
    row.get(field)
        .filter(|value| !value.is_null())
        .map(|value| nonnegative(value, field, None))
        .transpose()
}
