//! Human numeric display without changing native estimates.
use super::*;

pub(super) fn descriptive_partition_label(partition: &Value) -> Result<String> {
    let hash = format!("{:x}", Sha256::digest(serde_json::to_vec(partition)?));
    Ok(format!("recorded partition {}", &hash[..12]))
}

pub(super) fn text(value: &Value) -> String {
    match value {
        Value::Null => "Unavailable".into(),
        Value::String(value) => value.clone(),
        Value::Number(value) if value.is_i64() || value.is_u64() => value.to_string(),
        Value::Number(value) => value
            .as_f64()
            .map(display_number)
            .unwrap_or_else(|| value.to_string()),
        Value::Array(values) if values.is_empty() => "None".into(),
        Value::Array(values) => values.iter().map(text).collect::<Vec<_>>().join("; "),
        Value::Object(values) => values
            .iter()
            .map(|(key, value)| format!("{key}: {}", text(value)))
            .collect::<Vec<_>>()
            .join("; "),
        other => other.to_string(),
    }
}

pub(super) fn p_value(row: &Value, field: &str) -> String {
    let log_field = if field == "p_adjusted" {
        "log_p_adjusted"
    } else {
        "log_p_raw"
    };
    if row[field].as_f64() == Some(0.0) {
        return format!("Underflow / undefined (log P: {})", text(&row[log_field]));
    }
    if row[field].is_null() && !row[log_field].is_null() {
        return format!("Unavailable (log P: {})", text(&row[log_field]));
    }
    text(&row[field])
}

pub(super) fn display_number(number: f64) -> String {
    if number == 0.0 {
        return "0".into();
    }
    if number.abs() < 0.001 || number.abs() >= 100_000.0 {
        let value = format!("{number:.5e}");
        return value
            .split_once('e')
            .map(|(mantissa, exponent)| {
                format!(
                    "{}e{exponent}",
                    mantissa.trim_end_matches('0').trim_end_matches('.')
                )
            })
            .unwrap_or(value);
    }
    let decimals = (5.0 - number.abs().log10().floor()) as usize;
    let value = format!("{number:.decimals$}");
    if value.contains('.') {
        value.trim_end_matches('0').trim_end_matches('.').into()
    } else {
        value // coverage: unreachable — [0.001, 1e5) always formats with at least one decimal
    }
}

pub(super) fn detail(row: &Value, fields: &[&str]) -> String {
    let values = fields
        .iter()
        .filter(|field| !row[**field].is_null())
        .map(|field| format!("{field}: {}", text(&row[*field])))
        .collect::<Vec<_>>();
    if values.is_empty() {
        "None reported".into()
    } else {
        values.join("; ")
    }
}
