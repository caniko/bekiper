//! Animal-level eligibility without interpreting QC as exclusions.
use super::*;

pub(super) fn identity<'a>(row: &'a Record, field: &str) -> Result<&'a str> {
    row[field]
        .as_str()
        .filter(|value| !value.is_empty())
        .with_context(|| format!("{field}: explicit nonempty identity required"))
}

pub(super) fn number(row: &Record, field: &str) -> Result<f64> {
    let value = row[field]
        .as_f64()
        .with_context(|| format!("{field}: finite numeric value required"))?;
    ensure!(value.is_finite(), "{field}: nonfinite numeric value");
    Ok(value)
}

pub(super) fn optional_flag(row: &Record, field: &str) -> Result<Option<bool>> {
    match &row[field] {
        Value::Null => Ok(None),
        Value::Bool(value) => Ok(Some(*value)),
        _ => bail!("{field}: requires a boolean"),
    }
}

pub(super) fn excluded(row: &Record) -> Result<bool> {
    Ok(optional_flag(row, "excluded")?.unwrap_or(false))
}

pub(super) fn validate_requirement(requirement: Option<&str>) -> Result<()> {
    ensure!(
        matches!(
            requirement,
            None | Some("allow_unverified" | "require_verified")
        ),
        "Unknown upstream verification requirement"
    );
    Ok(())
}

pub(super) fn verification_status(row: &Record) -> &str {
    row["input_verification"]["status"]
        .as_str()
        .unwrap_or("unresolved")
}

/// A missing approval withholds inference. Descriptive callers can expose this
/// reason without inventing a verification classification.
pub fn verification_reason(row: &Record, requirement: Option<&str>) -> Option<String> {
    match requirement {
        None => Some("blocked:D12:upstream_verification_requirement".into()),
        Some("allow_unverified") => None,
        Some("require_verified")
            if matches!(
                verification_status(row),
                "verified_clarified_entry_definition" | "verified_source_rule_definition"
            ) =>
        {
            None
        }
        Some("require_verified") => {
            Some(format!("upstream_unverified:{}", verification_status(row)))
        }
        Some(_) => Some("blocked:D12:unknown_upstream_verification_requirement".into()),
    }
}

/// One eligible row per animal, with a ledger entry for every candidate row.
/// QC flags are intentionally not exclusions.
pub fn analysis_subset(
    rows: &[Record],
    fields: &[&str],
    requirement: Option<&str>,
) -> Result<(Vec<Record>, Vec<Record>)> {
    validate_requirement(requirement)?;
    let mut used = Vec::new();
    let mut ledger = Vec::new();
    let mut animals = BTreeSet::new();
    for row in rows {
        ensure!(row.is_object(), "analysis records must be objects");
        let animal = identity(row, "animal_id")?;
        identity(row, "session_id")?;
        let missing_fields: Vec<_> = fields
            .iter()
            .copied()
            .filter(|field| row[*field].is_null())
            .collect();
        let reason = if excluded(row)? {
            Some(format!(
                "excluded:{}",
                row["exclusion_reason"].as_str().unwrap_or("metadata")
            ))
        } else {
            crate::genotype::eligibility_reason(row)
                .or_else(|| verification_reason(row, requirement))
                .or_else(|| {
                    (!missing_fields.is_empty()).then(|| "model_specific_missingness".into())
                })
        };
        ledger.push(json!({"animal_id": animal, "session_id": row["session_id"], "reason": reason,
            "assay": row["assay"], "timepoint": row["timepoint"], "phase": row["phase"],
            "threshold_pct": row["threshold_pct"], "pair_id": row["pair_id"], "qc_flags": row["qc_flags"],
            "excluded": excluded(row)?, "exclusion_reason": row["exclusion_reason"],
            "missing_fields": missing_fields, "missing_reasons": row["missing_reasons"],
            "input_verification_status": verification_status(row),
            "genotype_eligibility":row["genotype_eligibility"], "analysis_factor_coding":row["analysis_factor_coding"],
            "source_factors":row["source_factors"],"qc_excluded":false}));
        if reason.is_none() {
            ensure!(
                animals.insert(animal.to_owned()),
                "Animal replicates in analysis: sessions/thresholds cannot be independent observations"
            );
            used.push(row.clone());
        }
    }
    Ok((used, ledger))
}
