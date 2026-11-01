//! Oft arithmetic and scoring contracts.
use super::*;

pub fn oft_scores(values: &Record) -> Result<Record> {
    ensure!(values.is_object(), "OFT measurements must be an object");
    let mut result = json!({"missing_reasons":{}});
    for field in oft_primary_fields().iter().chain(oft_secondary_fields()).copied() {
        let value = &values[field];
        if value.is_null() {
            missing(
                &mut result,
                field,
                values["missing_reasons"][field]
                    .as_str()
                    .unwrap_or("not_measured"),
            );
        } else {
            nonnegative(
                value,
                field,
                if field == "center_time_s" {
                    Some(300.0)
                } else if field.ends_with("pct") {
                    Some(100.0)
                } else {
                    None
                },
            )?;
            if is_count_field(field) {
                validate_count(value, field)?;
            }
            result[field] = value.clone();
        }
    }
    if let Some(center_time) = result["center_time_s"].as_f64() {
        // Owner decision 2026-10-22: divide by the analysed seconds (at most
        // the 300 s window); a recording without a duration keeps 300 s.
        let denominator = values["analysis_duration_s"]
            .as_f64()
            .filter(|d| d.is_finite() && *d > 0.0)
            .map_or(300.0, |d| d.min(300.0));
        let computed = center_time / denominator * 100.0;
        result["center_time_pct"] = json!(computed);
        result["missing_reasons"]
            .as_object_mut()
            .context("missing reasons")?
            .remove("center_time_pct");
        if values["center_time_pct"]
            .as_f64()
            .is_some_and(|value| !close(value, computed))
        {
            result["upstream_center_time_pct"] = values["center_time_pct"].clone();
            result["disagreements"] = json!(["center_time_pct:analysed_duration_denominator"]);
        }
    }
    Ok(result)
}
pub(super) fn score_oft(
    row: &mut Record,
    source: &Record,
    config: &Record,
    flags: &mut Record,
) -> Result<Vec<Record>> {
    let mut debug = Vec::new();
    if source.get("center_intervals").is_some() {
        let occupancy = inputs::choice(config, "assays.oft.center_occupancy_policy", "D05")
            .and_then(Value::as_str);
        let latency = inputs::choice(config, "assays.oft.no_center_entry_latency_policy", "D05")
            .and_then(Value::as_str);
        let window = inputs::choice(config, "assays.oft.window_and_tracking_policy", "D04")
            .and_then(Value::as_str);
        let starting = inputs::choice(config, "assays.oft.starting_in_center_policy", "D05")
            .and_then(Value::as_str);
        let timestamp = inputs::choice(config, "assays.oft.entry_timestamp_policy", "D04")
            .and_then(Value::as_str);
        if let (
            Some(occupancy),
            Some(latency),
            Some("half_open_300s_from_first_detection"),
            Some("count_as_entry_at_origin"),
            Some("initial_crossing"),
        ) = (occupancy, latency, window, starting, timestamp)
        {
            let origin = nonnegative(&row["first_detection_s"], "first_detection_s", None)?;
            if let Some(duration) = optional_number(row, "analysis_duration_s")? {
                events::validate_window(
                    samples(source, "center_intervals")?,
                    &json!({"start_s":origin,"end_s":origin + duration.min(300.0)}),
                    true,
                )?;
            }
            let mut result = events::oft_zone_intervals(
                samples(source, "center_intervals")?,
                origin,
                occupancy,
                latency,
            )?;
            debug = result
                .as_object_mut()
                .context("OFT events")?
                .remove("event_debug")
                .and_then(|v| v.as_array().cloned())
                .context("OFT debug")?;
            record_computed(row, &result, source, "reconstructed")?;
        } else {
            for field in [
                "center_time_s",
                "center_time_pct",
                "center_entries",
                "latency_to_first_center_entry_s",
            ] {
                missing(row, field, "blocked:D04/D05:interval_policy");
            }
        }
    }
    let values = oft_scores(row)?;
    let reconstructed = row["metric_sources"].clone();
    record_computed(row, &values, source, "upstream")?;
    for &field in oft_primary_fields() {
        if let Some(value) = reconstructed.get(field) {
            row["metric_sources"][field] = value.clone();
        }
    }
    if !row["center_time_s"].is_null() {
        row["metric_sources"]["center_time_pct"] = "derived".into();
    }
    flags["incomplete_recording"] = json!(row["analysis_duration_s"].as_f64().map(|d| d < 300.0));
    flags["tracking_incomplete"] = json!(row["observed_zone_time_s"].as_f64().map(|d| d < 300.0));
    Ok(debug)
}
