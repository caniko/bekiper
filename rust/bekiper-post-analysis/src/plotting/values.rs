//! Plot identities and registered selection values.
use super::*;

pub(super) fn is_primary(row: &Record, policy: &PlotPolicy<'_>) -> bool {
    policy.primary_threshold.is_some()
        && row["threshold_pct"].as_f64() == policy.primary_threshold.map(|value| value as f64)
        && row["primary_threshold"] == true
}

pub(super) fn phase_post<'a>(rows: &[&'a Record], phase: &str) -> Vec<&'a Record> {
    rows.iter()
        .copied()
        .filter(|row| row["timepoint"] == "post" && row["phase"] == phase)
        .collect()
}

pub(super) fn compact_pairing_row(row: &Record) -> Record {
    let mut compact = json!({});
    for field in [
        "animal_id",
        "assay",
        "session_id",
        "timepoint",
        "phase",
        "group",
        "sex",
        "age_treatment_window",
        "five_xfad",
        "hcar1",
        "treatment",
        "excluded",
        "exclusion_reason",
        "input_verification",
        "spontaneous_alternation_pct",
        "threshold_pct",
        "qc_flags",
        "missing_reasons",
        "recording_id",
        "analysis_id",
        "analysis_role",
        "variant",
        "scoring_provenance",
        "object_roles",
    ] {
        compact[field] = row[field].clone();
    }
    for field in ["pair_id", "presentation_partition"] {
        if let Some(value) = row.get(field) {
            compact[field] = value.clone();
        }
    }
    copy_genotype_fields(row, &mut compact);
    if let Some(reference) = row.get("scoring_provenance_ref") {
        compact
            .as_object_mut()
            .expect("compact row is an object")
            .remove("scoring_provenance");
        compact["scoring_provenance_ref"] = reference.clone();
    }
    compact
}

pub(super) fn copy_genotype_fields(source: &Record, target: &mut Record) {
    for field in [
        "literal_hcar1",
        "assigned_group",
        "source_factors",
        "genotype_eligibility",
        "analysis_factor_coding",
    ] {
        if let Some(value) = source.get(field) {
            target[field] = value.clone();
        }
    }
}

pub(super) fn number(row: &Record, field: &str) -> Result<Option<f64>> {
    if row[field].is_null() {
        return Ok(None);
    }
    let value = row[field]
        .as_f64()
        .with_context(|| format!("{field}: nonnumeric plot value"))?;
    ensure!(value.is_finite(), "{field}: nonfinite plot value");
    Ok(Some(value))
}

pub(super) fn string<'a>(row: &'a Record, field: &str) -> Result<&'a str> {
    row[field]
        .as_str()
        .filter(|value| !value.is_empty())
        .with_context(|| format!("{field}: required nonempty plot identity"))
}

pub(super) fn missing_reason(row: &Record, field: &str) -> String {
    row["missing_reasons"][field]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("not_measured:{field}"))
}

pub(super) fn strings_or_default(
    value: &Value,
    default: &[&str],
    label: &str,
) -> Result<Vec<String>> {
    if value.is_null() {
        return Ok(default.iter().map(|item| (*item).into()).collect());
    }
    value
        .as_array()
        .with_context(|| format!("{label} must be an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .with_context(|| format!("{label} requires strings"))
        })
        .collect()
}

pub(super) fn unique(values: &[String], label: &str) -> Result<()> {
    ensure!(
        values.iter().collect::<BTreeSet<_>>().len() == values.len(),
        "Duplicate {label}"
    );
    Ok(())
}

pub(super) fn thresholds(config: &Record) -> Result<Vec<u64>> {
    let value = &config["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"];
    let allowed = crate::inputs::allowed_thresholds(config);
    if value.is_null() {
        return Ok(allowed.to_vec());
    }
    let values = value
        .as_array()
        .context("Sensitivity thresholds must be an array")?
        .iter()
        .map(|item| {
            item.as_u64()
                .filter(|value| allowed.contains(value))
                .context("Sensitivity thresholds must be 25/45/65 integers (cross-line also 0)")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        values.iter().collect::<BTreeSet<_>>().len() == values.len(),
        "Duplicate sensitivity thresholds"
    );
    Ok(values)
}

pub(super) fn subgroup_name(assay: &str, field: &str, index: usize) -> String {
    format!("subgroup_{assay}_{field}_{index}")
}
