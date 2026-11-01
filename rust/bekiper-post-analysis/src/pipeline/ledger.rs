//! Endpoint eligibility and output status with explicit missingness.
use super::*;

pub(super) fn key_fields(row: &Record) -> Record {
    json!(
        KEY.iter()
            .map(|key| (key.to_string(), row[*key].clone()))
            .collect::<BTreeMap<_, _>>()
    )
}

pub fn inclusion_ledger(rows: &[Record]) -> Vec<Record> {
    let mut ledger = Vec::new();
    for row in rows {
        for endpoint in endpoint_fields(row) {
            let reason = row["missing_reasons"][endpoint].as_str();
            let value = row["qc_flags"].get(endpoint).unwrap_or(&row[endpoint]);
            let status = if let Some(reason) = reason {
                if reason.starts_with("blocked") {
                    "blocked"
                } else if reason.starts_with("undefined") {
                    "undefined"
                } else if reason.starts_with("not_applicable") {
                    "not_applicable"
                } else {
                    "unavailable"
                }
            } else if !value.is_null() {
                "available"
            } else {
                "unavailable"
            };
            let mut item = key_fields(row);
            item.as_object_mut().unwrap().extend(json!({"threshold_pct":row["threshold_pct"],"endpoint":endpoint,"metric_status":status,"missing_reason":reason,
                "metadata_excluded":row["excluded"],"exclusion_reason":row["exclusion_reason"],"qc_flags":row["qc_flags"],
                "metric_source":row["metric_sources"][endpoint],"eligible_before_model_policy":status == "available" && row["excluded"] != true}).as_object().unwrap().clone());
            ledger.push(item);
        }
    }
    ledger
}

pub(super) fn endpoint_fields(row: &Record) -> Vec<&'static str> {
    match row["assay"].as_str() {
        Some("oft") => crate::contracts::oft_primary_fields()
            .iter()
            .chain(crate::contracts::oft_secondary_fields())
            .copied()
            .collect(),
        Some("ymaze") => crate::contracts::ymaze_score_fields()
            .iter()
            .copied()
            .chain([
                "valid_total_arm_entries",
                "usable_triads",
                "raw_total_arm_entries",
                "removed_short_entries",
                "raw_to_scored_entry_difference",
                "dwell_rejected_entries",
                "arm_preference",
                "total_distance_m",
                "mean_speed_m_per_s",
                "freezing_s",
                "low_quality",
            ])
            .collect(),
        _ if row["phase"] == "training" => vec![
            "TA",
            "TB",
            "total_training_exploration",
            "object_bias",
            "side_bias",
            "passes_15s_exploration_threshold",
        ],
        _ => vec![
            "TN",
            "TF",
            "DI",
            "RI",
            "percent_novel",
            "LR",
            "total_test_exploration",
            "passes_15s_exploration_threshold",
        ],
    }
}

pub fn result_status(results: &[Record], success: &str) -> Record {
    if results.is_empty() {
        return json!({"status":"unavailable","reason":"no_applicable_observations_or_outputs","attempts":0,"successful_attempts":0});
    }
    let successful = results
        .iter()
        .filter(|item| {
            item["status"] == success
                && (success != "fitted"
                    || !item["contrast"]["p_adjusted"].is_null()
                    || !item["contrast"]["log_p_adjusted"].is_null())
        })
        .count();
    let status = if successful == results.len() {
        "complete"
    } else if results.iter().all(|item| item["status"] == "blocked") {
        "blocked"
    } else {
        "partial"
    };
    let reasons: BTreeSet<_> = results
        .iter()
        .flat_map(|item| {
            [
                &item["reason"],
                &item["contrast"]["reason"],
                &item["contrast"]["adjustment_reason"],
            ]
        })
        .filter(|reason| !reason.is_null())
        .map(Value::to_string)
        .collect();
    json!({"status":status,"attempts":results.len(),"successful_attempts":successful,"reasons":reasons})
}

pub(super) fn overall_status(statuses: &Record) -> &'static str {
    let values = statuses.as_object().expect("output statuses");
    if !values.is_empty() && values.values().all(|item| item["status"] == "complete") {
        "complete"
    } else if values
        .values()
        .any(|item| item["status"] == "complete" || item["status"] == "partial")
    {
        "partial"
    } else {
        "blocked"
    }
}
