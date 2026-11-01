//! Ymaze arithmetic and scoring contracts.
use super::*;

pub(super) fn score_ymaze(
    row: &mut Record,
    source: &Record,
    config: &Record,
    flags: &mut Record,
) -> Result<(Vec<Record>, Record)> {
    let threshold = row["threshold_pct"].as_f64();
    // 2026-10-28 cross-line plans: each threshold context carries its own
    // registered policy (`assays.ymaze.threshold_policies."<pct>"`, falling back
    // to `temporal_policy`) and may bridge gaps; radial plans keep one
    // uninterrupted principal policy.
    let crossline = config["assays"]["ymaze"]["threshold_geometry"] == events::GEOMETRY_ARM_START;
    let policy_value = threshold
        .filter(|_| crossline)
        .and_then(|t| config["assays"]["ymaze"]["threshold_policies"].get(format!("{t}")))
        .or_else(|| config["assays"]["ymaze"].get("temporal_policy"));
    let temporal_policy: bekiper_feature::tolerance::TemporalPolicy = policy_value
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()?
        .unwrap_or_else(|| {
            if crossline {
                Default::default()
            } else {
                bekiper_feature::tolerance::TemporalPolicy::ymaze_default()
            }
        });
    if crossline {
        temporal_policy.validate()
    } else {
        temporal_policy.validate_uninterrupted()
    }
    .map_err(anyhow::Error::msg)?;
    let registered_geometry = if crossline {
        events::GEOMETRY_ARM_START
    } else {
        events::GEOMETRY
    };
    let registered_zone_gap = config["assays"]["ymaze"]["zone_max_gap_seconds"]
        .as_f64()
        .unwrap_or(0.0);
    for (field, expected) in [
        ("threshold_geometry", json!(registered_geometry)),
        ("principal_entry_rule", json!(events::ENTRY_RULE)),
        (
            "principal_minimum_dwell_s",
            json!(temporal_policy.min_duration_seconds),
        ),
    ] {
        if let Some(value) = config["assays"]["ymaze"].get(field) {
            ensure!(
                same_value(value, &expected),
                "Principal Y-maze config conflicts with registered researcher amendment"
            );
        }
    }
    if row["measurement_available"] != false || threshold.is_some() {
        ensure!(
            threshold.is_some_and(|t| [25.0, 45.0, 65.0].contains(&t) || (crossline && t == 0.0)),
            "Principal Y-maze inputs need explicit threshold_pct 25/45/65"
        );
    }
    let mut reconstructed = false;
    let mut debug = Vec::new();
    let mut verification = json!({});
    if source.get("coordinate_samples").is_some() {
        let window_policy =
            inputs::choice(config, "assays.ymaze.window_and_tracking_policy", "D04");
        if let Some(window) = events::reconstruction_window(row, window_policy)? {
            let coordinates = samples(source, "coordinate_samples")?;
            events::validate_window(coordinates, &window, false)?;
            let threshold = threshold.context("Missing Y-maze threshold")? as u8;
            let sample_end_s = source
                .get("coordinate_sample_end_s")
                .map(|value| nonnegative(value, "coordinate_sample_end_s", None))
                .transpose()?;
            if let Some(end) = sample_end_s {
                ensure!(
                    end <= window["end_s"].as_f64().context("window end")?,
                    "Final coordinate sample support exceeds analysis window"
                );
            }
            let mut result = events::ymaze_coordinate_entries_with_zone_gap(
                coordinates,
                &row["geometry"],
                threshold,
                &temporal_policy,
                sample_end_s,
                registered_zone_gap,
            )?;
            debug = result
                .as_object_mut()
                .context("Y-maze events")?
                .remove("event_debug")
                .and_then(|v| v.as_array().cloned())
                .context("Y-maze debug")?;
            record_computed(row, &result, source, "reconstructed")?;
            row["reconstruction_window"] = window;
            reconstructed = true;
            verification = json!({"status":"verified_clarified_entry_definition","reasons":[],"source_level":"explicit_head_center_tail_base_samples"});
        } else {
            block(row, "ymaze_coordinate_reconstruction:D04")?;
        }
    }
    if !reconstructed {
        verification = events::ymaze_provenance_registered(
            row,
            &config["inputs"]["upstream_rule_provenance"]["ymaze"]["landmark_roles"],
            &temporal_policy,
            registered_geometry,
            registered_zone_gap,
        )?;
    }
    let policy = inputs::choice(config, "assays.ymaze.triad_policy", "D01").and_then(Value::as_str);
    let short = inputs::choice(
        config,
        "assays.ymaze.fewer_than_three_entries_policy",
        "D02",
    )
    .and_then(Value::as_str);
    let mut values = if !row["valid_arm_entry_sequence"].is_null() {
        ymaze_scores(
            &row["valid_arm_entry_sequence"],
            policy,
            short,
            inputs::choice(config, "assays.ymaze.custom_triad_mapping", "D01"),
        )?
    } else {
        if let Some(policy) = policy {
            // Validate approved mappings even when no sequence was exported.
            ymaze_scores(
                &json!([]),
                Some(policy),
                short,
                inputs::choice(config, "assays.ymaze.custom_triad_mapping", "D01"),
            )?;
        }
        let mut values = json!({"missing_reasons":{}});
        for field in ymaze_score_fields().iter().copied().chain([
            "valid_total_arm_entries",
            "usable_triads",
            "low_quality",
            "correct_alternations",
            "incorrect_alternations",
            "neutral_triads",
            "window_labels",
        ]) {
            missing(
                &mut values,
                field,
                if policy.is_none() && field != "valid_total_arm_entries" {
                    "blocked:D01"
                } else {
                    "not_measured:valid_sequence"
                },
            );
        }
        values
    };
    if let Some(primary) = inputs::choice(
        config,
        "assays.ymaze.primary_threshold_arm_length_pct",
        "D03",
    ) {
        let primary = primary
            .as_f64()
            .context("Registered primary Y-maze threshold must be 25, 45 or 65")?;
        ensure!(
            [25.0, 45.0, 65.0].contains(&primary) || (crossline && primary == 0.0),
            "Registered primary Y-maze threshold must be 25, 45 or 65 (cross-line also 0)"
        );
        if let Some(threshold) = threshold {
            values["primary_threshold"] = json!(threshold == primary);
        } else {
            missing(
                &mut values,
                "primary_threshold",
                "not_measured:threshold_identity",
            );
        }
    } else {
        missing(&mut values, "primary_threshold", "blocked:D03");
    }
    for field in [
        "raw_total_arm_entries",
        "removed_short_entries",
        "arm_preference",
        "total_distance_m",
        "mean_speed_m_per_s",
        "freezing_s",
    ] {
        if row[field].is_null() {
            missing(&mut values, field, "not_measured");
        }
    }
    if !row["raw_arm_entry_sequence"].is_null() {
        values["raw_total_arm_entries"] = json!(labels(&row["raw_arm_entry_sequence"])?.len());
        values["missing_reasons"]
            .as_object_mut()
            .context("missing reasons")?
            .remove("raw_total_arm_entries");
    }
    if !row["removed_short_entries"].is_null()
        && (verification["status"] != "verified_clarified_entry_definition"
            || row["dwell_rejected_entries"].is_null())
    {
        flags["removal_reason_unverified"] = json!(true);
    }
    record_computed(row, &values, source, "derived")?;
    let mut entry_diagnostics = json!({"missing_reasons":{}});
    if let (Some(raw), Some(scored)) = (
        row["raw_total_arm_entries"].as_f64(),
        row["valid_total_arm_entries"].as_f64(),
    ) {
        // Both totals are validated exact nonnegative counts, or sequence lengths.
        entry_diagnostics["raw_to_scored_entry_difference"] = json!(raw as i64 - scored as i64);
    } else {
        missing(
            &mut entry_diagnostics,
            "raw_to_scored_entry_difference",
            "not_measured:entry_counts",
        );
    }
    if row["dwell_rejected_entries"].is_null() {
        missing(
            &mut entry_diagnostics,
            "dwell_rejected_entries",
            "not_measured:dwell_bout_evidence",
        );
    }
    record_computed(row, &entry_diagnostics, source, "derived")?;
    if policy.is_some()
        && let (Some(entries), Some(usable)) = (
            row["valid_total_arm_entries"].as_f64(),
            row["usable_triads"].as_f64(),
        )
    {
        let quality = json!({"low_quality":entries < 8.0 || usable < 5.0});
        record_computed(row, &quality, source, "derived")?;
    }
    flags["low_quality"] = row["low_quality"].clone();
    Ok((debug, verification))
}
