//! Per-animal plotting values and eligibility reasons.
use super::*;

pub(super) fn point_for(
    row: &Record,
    field: &str,
    policy: &PlotPolicy<'_>,
    ledger: &mut Vec<Record>,
    missing_count: &mut usize,
) -> Result<Option<Record>> {
    let excluded = row["excluded"]
        .as_bool()
        .context("plot source requires explicit boolean excluded")?;
    let verification = row["input_verification"]["status"]
        .as_str()
        .unwrap_or("unresolved");
    let reason = if excluded {
        Some(format!(
            "excluded:{}",
            row["exclusion_reason"].as_str().unwrap_or("metadata")
        ))
    } else if let Some(reason) = crate::genotype::eligibility_reason(row) {
        Some(reason)
    } else if row["genotype_eligibility"].is_object() && row["group"].is_null() {
        Some("missing_group_metadata".into())
    } else if policy.verification == Some("require_verified")
        && !matches!(
            verification,
            "verified_clarified_entry_definition" | "verified_source_rule_definition"
        )
    {
        Some(format!("upstream_unverified:{verification}"))
    } else {
        None
    };
    let value = if reason.is_none() {
        number(row, field)?
    } else {
        None
    };
    let reason = reason.or_else(|| {
        if value.is_none() {
            *missing_count += 1;
            Some(missing_reason(row, field))
        } else {
            None
        }
    });
    let mut audit = json!({"animal_id":row["animal_id"],"assay":row["assay"],"session_id":row["session_id"],"recording_id":row["recording_id"],"analysis_id":row["analysis_id"],"variant":row["variant"],"analysis_role":row["analysis_role"],"scoring_provenance":row["scoring_provenance"],"timepoint":row["timepoint"],"phase":row["phase"],"threshold_pct":row["threshold_pct"],"endpoint":field,
        "reason":reason,"input_verification_status":verification,"metadata_excluded":excluded,"qc_flags":row["qc_flags"]});
    super::values::copy_genotype_fields(row, &mut audit);
    if row["genotype_eligibility"].is_object() {
        audit["qc_excluded"] = false.into();
    }
    ledger.push(audit);
    if policy.population == "recorded_descriptive" {
        let entry = ledger.last_mut().context("descriptive eligibility entry")?;
        if let Some(reference) = row.get("scoring_provenance_ref") {
            entry
                .as_object_mut()
                .context("eligibility entry")?
                .remove("scoring_provenance");
            entry["scoring_provenance_ref"] = reference.clone();
        } // coverage: unreachable — descriptive bundles attach scoring_provenance_ref to every row
        entry["presentation_partition"] = row["presentation_partition"].clone();
        entry["metric_status"] = if number(row, field)?.is_some() {
            "available"
        } else {
            "missing"
        }
        .into();
        entry["eligible_before_model_policy"] = reason.is_none().into();
        entry["missing_reason"] = if excluded { Value::Null } else { json!(reason) };
        entry["exclusion_reason"] = if excluded {
            row["exclusion_reason"]
                .as_str()
                .unwrap_or("metadata")
                .into()
        } else {
            Value::Null
        };
    }
    if reason.is_some() {
        return Ok(None);
    }
    let mut point = json!({});
    for key in [
        "animal_id",
        "assay",
        "session_id",
        "group",
        "sex",
        "age_treatment_window",
        "five_xfad",
        "hcar1",
        "treatment",
        "timepoint",
        "phase",
        "pre_session_id",
        "threshold_pct",
        "pair_id",
        "recording_id",
        "analysis_id",
        "analysis_role",
        "variant",
        "scoring_provenance",
        "object_roles",
    ] {
        point[key] = row[key].clone();
    }
    super::values::copy_genotype_fields(row, &mut point);
    if let Some(reference) = row.get("scoring_provenance_ref") {
        point
            .as_object_mut()
            .context("plot point")?
            .remove("scoring_provenance");
        point["scoring_provenance_ref"] = reference.clone();
    }
    string(row, "animal_id")?;
    string(row, "session_id")?;
    point["category"] = string(row, "group")?.into();
    point["value"] = value.context("available plot value")?.into();
    point["input_verification_status"] = verification.into();
    point["population_policy"] = policy.population.into();
    point["source_endpoint"] = field.into();
    point["censored"] = (field == "latency_to_first_center_entry_s"
        && row["latency_right_censored"] == true)
        .into();
    point["qc_flags"] = row["qc_flags"].clone();
    point["pair_series"] = false.into();
    point["plot_pair_id"] = Value::Null;
    point["pair_order"] = 0.into();
    Ok(Some(point))
}
