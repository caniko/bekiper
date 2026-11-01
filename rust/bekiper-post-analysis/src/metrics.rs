//! Pure recipe arithmetic and session-wise scoring with explicit missingness.
//! Imported evidence and QC remain available when reconstruction disagrees.
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

use crate::contracts::{
    KEY, Record, missing, oft_primary_fields, oft_secondary_fields, ymaze_score_fields,
};
use crate::{events, inputs};

mod values;
use values::{close, labels, optional_number, validate_count};
pub use values::{is_count_field, nonnegative, same_value};

mod triads;
pub use triads::ymaze_scores;

mod nort;
pub use nort::nort_scores;
use nort::score_nort;

mod oft;
pub use oft::oft_scores;
use oft::score_oft;

mod validation;
pub use validation::validate_supplied;

mod lineage;
pub use lineage::record_computed;
use lineage::{block, map_or_empty, samples, set_lineage};

mod ymaze;
use ymaze::score_ymaze;

pub fn score_records(rows: &[Record], config: &Record) -> Result<(Vec<Record>, Vec<Record>)> {
    let mut scored = Vec::with_capacity(rows.len());
    let mut debug = Vec::new();
    for source in rows {
        inputs::session_key(source)?;
        validate_supplied(source)?;
        let mut row = Value::Object(
            source
                .as_object()
                .context("Session record must be an object")?
                .iter()
                .filter(|(field, _)| {
                    !matches!(
                        field.as_str(),
                        "coordinate_samples" | "candidate_segments" | "center_intervals"
                    )
                })
                .map(|(field, value)| (field.clone(), value.clone()))
                .collect(),
        );
        for field in [
            "missing_reasons",
            "metric_sources",
            "recomputation_missing_reasons",
        ] {
            row[field] = Value::Object(map_or_empty(source, field)?);
        }
        let prior = map_or_empty(source, "prior_lineage")?;
        row["prior_lineage"] = if prior.is_empty() {
            json!({})
        } else {
            json!({"previous":prior})
        };
        if row["disagreements"].is_null() {
            row["disagreements"] = json!([]);
        }
        ensure!(
            row["disagreements"]
                .as_array()
                .is_some_and(|items| items.iter().all(Value::is_string)),
            "disagreements must be a string list"
        );
        ensure!(
            row["measurement_available"].is_null() || row["measurement_available"].is_boolean(),
            "measurement_available must be a boolean"
        );
        let assay = row["assay"]
            .as_str()
            .context("Missing canonical assay")?
            .to_owned();
        let available = row["measurement_available"] != false;
        if available {
            let units = &config["inputs"]["units_and_calibration"][&assay];
            ensure!(
                units["time"] == "s" && units["distance"] == "m" && units["speed"] == "m/s",
                "{assay}: explicit SI time/distance/speed units required; conversion is never inferred"
            );
        }
        let upstream_flags = map_or_empty(source, "qc_flags")?;
        let mut flags = json!({});
        let (row_debug, verification) = match assay.as_str() {
            "oft" => {
                let verification = events::assay_provenance(&row, config)?;
                (
                    score_oft(&mut row, source, config, &mut flags)?,
                    verification,
                )
            }
            "ymaze" => score_ymaze(&mut row, source, config, &mut flags)?,
            "nort" => {
                let verification = events::assay_provenance(&row, config)?;
                (
                    score_nort(&mut row, source, config, &mut flags)?,
                    verification,
                )
            }
            _ => anyhow::bail!("Canonical assay must be oft, ymaze or nort"), // coverage: unreachable — session_key validated the assay
        };
        let mut merged_flags = upstream_flags.clone();
        let mut conflicts = serde_json::Map::new();
        for (field, computed) in flags.as_object().context("QC flags")? {
            if let Some(upstream) = upstream_flags.get(field)
                && !same_value(upstream, computed)
            {
                conflicts.insert(
                    field.clone(),
                    json!({"upstream":upstream,"computed":computed}),
                );
            }
            merged_flags.insert(field.clone(), computed.clone());
        }
        row["qc_flags"] = Value::Object(merged_flags);
        if !upstream_flags.is_empty() {
            set_lineage(
                &mut row,
                source,
                "upstream_qc_flags",
                Value::Object(upstream_flags),
            );
        }
        if !conflicts.is_empty() {
            set_lineage(
                &mut row,
                source,
                "qc_flag_conflicts",
                Value::Object(conflicts),
            );
        }
        if let Some(previous) = source
            .get("input_verification")
            .filter(|previous| !same_value(previous, &verification))
        {
            set_lineage(
                &mut row,
                source,
                "upstream_input_verification",
                previous.clone(),
            );
        }
        row["input_verification"] = verification;
        let identity: serde_json::Map<String, Value> = KEY
            .iter()
            .map(|field| ((*field).to_owned(), row[*field].clone()))
            .collect();
        for mut event in row_debug {
            let event_map = event
                .as_object_mut()
                .context("Debug event must be an object")?;
            event_map.extend(identity.clone());
            debug.push(event);
        }
        let row_map = row
            .as_object_mut()
            .context("Session record must be an object")?;
        for field in [
            "coordinate_samples",
            "candidate_segments",
            "center_intervals",
            "comments",
            "All comments",
            "source_path",
            "video_path",
            "_rule_audit",
            "_rule_evidence_files",
        ] {
            row_map.remove(field);
        }
        if row_map["disagreements"]
            .as_array()
            .is_some_and(Vec::is_empty)
        {
            row_map.remove("disagreements");
        }
        if row_map["prior_lineage"]
            .as_object()
            .is_some_and(serde_json::Map::is_empty)
        {
            row_map.remove("prior_lineage");
        }
        scored.push(row);
    }
    Ok((scored, debug))
}
