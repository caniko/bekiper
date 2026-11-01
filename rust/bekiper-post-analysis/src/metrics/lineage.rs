//! Lineage arithmetic and scoring contracts.
use super::*;

pub(super) fn map_or_empty(row: &Record, field: &str) -> Result<serde_json::Map<String, Value>> {
    row.get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_object()
                .cloned()
                .with_context(|| format!("{field}: requires an object"))
        })
        .transpose()
        .map(Option::unwrap_or_default)
}

pub(super) fn add_disagreement(row: &mut Record, field: &str) -> Result<()> {
    push_unique_reason(row, "disagreements", field)
}

pub(super) fn set_lineage(row: &mut Record, source: &Record, field: &str, value: Value) {
    if let Some(previous) = source
        .get(field)
        .filter(|previous| !same_value(previous, &value))
    {
        row["prior_lineage"][field] = previous.clone();
    }
    row[field] = value;
}

pub fn record_computed(
    row: &mut Record,
    values: &Record,
    source: &Record,
    kind: &str,
) -> Result<()> {
    let values_map = values
        .as_object()
        .context("Computed measurements must be an object")?;
    for (field, computed) in values_map {
        if matches!(field.as_str(), "missing_reasons" | "disagreements") {
            continue;
        }
        let supplied = &source[field];
        let equal = computed
            .as_f64()
            .zip(supplied.as_f64())
            .map_or_else(|| same_value(computed, supplied), |(a, b)| close(a, b))
            || (matches!(
                field.as_str(),
                "valid_arm_entry_sequence" | "raw_arm_entry_sequence"
            ) && !computed.is_null()
                && !supplied.is_null()
                && labels(computed)? == labels(supplied)?);
        if !supplied.is_null() && !equal {
            set_lineage(row, source, &format!("upstream_{field}"), supplied.clone());
            set_lineage(
                row,
                source,
                &format!("recomputed_{field}"),
                computed.clone(),
            );
            add_disagreement(row, field)?;
        }
        let reason = values["missing_reasons"][field].as_str();
        let preserve = computed.is_null()
            && !supplied.is_null()
            && reason.is_some_and(|r| r.starts_with("not_measured"));
        if preserve {
            row[field] = supplied.clone();
            row["metric_sources"][field] = "upstream".into();
            row["recomputation_missing_reasons"][field] = json!(reason);
        } else {
            row[field] = computed.clone();
            row["metric_sources"][field] = kind.into();
        }
        if let Some(reason) = reason.filter(|_| !preserve) {
            row["missing_reasons"][field] = reason.into();
        } else {
            row["missing_reasons"]
                .as_object_mut()
                .context("missing reasons")?
                .remove(field);
        }
    }
    if let Some(disagreements) = values["disagreements"].as_array() {
        for field in disagreements {
            add_disagreement(row, field.as_str().context("Invalid disagreement")?)?;
        }
    }
    Ok(())
}

pub(super) fn block(row: &mut Record, reason: &str) -> Result<()> {
    push_unique_reason(row, "blocked_outputs", reason)
}

fn push_unique_reason(row: &mut Record, field: &str, reason: &str) -> Result<()> {
    if row[field].is_null() {
        row[field] = json!([]);
    }
    let reasons = row[field]
        .as_array_mut()
        .with_context(|| format!("{field} must be a list"))?;
    if !reasons.iter().any(|value| value == reason) {
        reasons.push(reason.into());
    }
    Ok(())
}

pub(super) fn samples<'a>(row: &'a Record, field: &str) -> Result<&'a [Record]> {
    row[field]
        .as_array()
        .map(Vec::as_slice)
        .with_context(|| format!("{field}: requires a list"))
}
