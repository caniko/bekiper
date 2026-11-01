//! Nort arithmetic and scoring contracts.
use super::*;

struct NortPhase {
    components: [&'static str; 2],
    aliases: [&'static str; 2],
    total: &'static str,
    indices: &'static [&'static str],
}

const NORT_TRAINING: NortPhase = NortPhase {
    components: ["TA", "TB"],
    aliases: ["object_A_exploration_s", "object_B_exploration_s"],
    total: "total_training_exploration",
    indices: &["object_bias", "side_bias"],
};
const NORT_TEST: NortPhase = NortPhase {
    components: ["TN", "TF"],
    aliases: ["TN", "TF"],
    total: "total_test_exploration",
    indices: &["DI", "RI", "percent_novel", "LR"],
};

impl NortPhase {
    fn fields(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.components
            .into_iter()
            .chain(std::iter::once(self.total))
            .chain(self.indices.iter().copied())
    }
}
pub fn nort_scores(
    ta: Option<f64>,
    tb: Option<f64>,
    tn: Option<f64>,
    tf: Option<f64>,
) -> Result<Record> {
    let mut result = json!({"missing_reasons":{}});
    for (field, value) in [("TA", ta), ("TB", tb), ("TN", tn), ("TF", tf)] {
        if let Some(value) = value {
            ensure!(
                value.is_finite() && value >= 0.0,
                "{field}: requires a finite nonnegative number"
            );
            result[field] = json!(value);
        } else {
            missing(&mut result, field, "not_measured");
        }
    }
    for (a, b, phase) in [(ta, tb, &NORT_TRAINING), (tn, tf, &NORT_TEST)] {
        let total_name = phase.total;
        let fields = phase.indices.iter().copied();
        let (Some(a), Some(b)) = (a, b) else {
            for field in std::iter::once(total_name).chain(fields) {
                missing(&mut result, field, "not_measured:component_missing");
            }
            continue;
        };
        let total = a + b;
        ensure!(total.is_finite(), "Exploration total overflow");
        result[total_name] = json!(total);
        if total == 0.0 {
            for field in fields {
                missing(&mut result, field, "undefined:zero_exploration_total");
            }
        } else if total_name == "total_training_exploration" {
            result["object_bias"] = json!((a - b) / total);
            result["side_bias"] = json!((b - a) / total);
        } else {
            result["DI"] = json!((a - b) / total);
            result["RI"] = json!(a / total);
            result["percent_novel"] = json!(100.0 * (a / total));
            if a == 0.0 || b == 0.0 {
                missing(&mut result, "LR", "undefined:zero_component_no_pseudocount");
            } else {
                // Difference of logs avoids overflow in an otherwise finite ratio.
                result["LR"] = json!(a.ln() - b.ln());
            }
        }
    }
    Ok(result)
}
pub(super) fn score_nort(
    row: &mut Record,
    source: &Record,
    config: &Record,
    flags: &mut Record,
) -> Result<Vec<Record>> {
    let mut debug = Vec::new();
    if source.get("candidate_segments").is_some() {
        let gap = inputs::choice(config, "assays.nort.merged_gap_duration_policy", "D07")
            .and_then(Value::as_str);
        let median = inputs::choice(config, "assays.nort.merged_gap_median_policy", "D07")
            .and_then(Value::as_str);
        let roles = inputs::choice(
            config,
            "assays.nort.object_role_and_location_mapping",
            "D10",
        )
        .map(|mapping| events::nort_object_roles(row, mapping))
        .transpose()?
        .flatten();
        let window = events::reconstruction_window(
            row,
            inputs::choice(config, "assays.nort.window_and_tracking_policy", "D14"),
        )?;
        if window.is_none() {
            block(row, "nort_bout_reconstruction:D14:window_policy")?;
        }
        if let (Some(gap), Some(median), Some(roles), Some(window)) = (gap, median, &roles, window)
        {
            let parts = samples(source, "candidate_segments")?;
            events::validate_window(parts, &window, true)?;
            let roles_map = roles.as_object().context("object roles")?;
            let objects: BTreeSet<_> = roles_map.values().filter_map(Value::as_str).collect();
            ensure!(
                parts.iter().all(|s| s["object_id"]
                    .as_str()
                    .is_some_and(|id| objects.contains(id))),
                "NORT object roles do not cover candidate object identities"
            );
            let policy = config["assays"]["nort"]
                .get("bout_policy")
                .map(|value| serde_json::from_value(value.clone()))
                .transpose()?
                .unwrap_or_default();
            let mut result = events::nort_bouts_with_policy(parts, gap, median, &policy)?;
            let result_map = result.as_object_mut().context("NORT bouts")?;
            debug = result_map
                .remove("bout_debug")
                .and_then(|v| v.as_array().cloned())
                .context("NORT debug")?;
            let exploration = result_map
                .remove("object_exploration_s")
                .unwrap_or(Value::Null);
            for (field, object) in roles_map {
                let object = object.as_str().context("object identity")?;
                let mut value = json!({field:exploration[object],"missing_reasons":{}});
                if value[field].is_null() {
                    missing(&mut value, field, "not_measured:object_candidates");
                }
                record_computed(row, &value, source, "reconstructed")?;
            }
            record_computed(row, &result, source, "reconstructed")?;
            row["resolved_object_roles"] = roles.clone();
            row["reconstruction_window"] = window;
        } else if gap.is_none() || median.is_none() || roles.is_none() {
            block(row, "nort_bout_reconstruction:D07/D10")?;
        }
    }
    let phase = if row["phase"] == "training" {
        &NORT_TRAINING
    } else {
        &NORT_TEST
    };
    for (canonical, alias) in phase.components.into_iter().zip(phase.aliases) {
        if row[canonical].is_null() {
            row[canonical] = row[alias].clone();
        }
    }
    let ta = optional_number(row, "TA")?;
    let tb = optional_number(row, "TB")?;
    let tn = optional_number(row, "TN")?;
    let tf = optional_number(row, "TF")?;
    let mut values = nort_scores(ta, tb, tn, tf)?;
    // Explicit policies require recording-specific physical sides regardless of
    // their gap comparator or speed filter. Legacy plans without a bout policy
    // retain their recorded A/B convention.
    if row["phase"] == "training" && config["assays"]["nort"].get("bout_policy").is_some() {
        let sides = inputs::choice(
            config,
            "assays.nort.object_role_and_location_mapping",
            "D10",
        )
        .map(|mapping| events::nort_physical_sides(row, mapping))
        .transpose()?
        .flatten();
        if let Some(sides) = sides {
            if let (Some(ta), Some(tb)) = (ta, tb)
                && ta + tb > 0.0
            {
                values["side_bias"] = json!(if sides["TB"] == "right" {
                    (tb - ta) / (ta + tb)
                } else {
                    (ta - tb) / (ta + tb)
                });
            }
            row["physical_side_mapping"] = sides;
        } else {
            missing(
                &mut values,
                "side_bias",
                "blocked:D10:recording_specific_physical_side_mapping",
            );
            flags["physical_side_mapping_unavailable"] = json!(true);
        }
    }
    if config["assays"]["nort"]["lr_pseudocount"]["enabled"] == true && row["phase"] == "test" {
        let approved = inputs::choice(config, "assays.nort.lr_pseudocount", "D13");
        if let Some(approved) = approved
            .filter(|value| value["units"] == "s" && value["formula"] == "ln((TN+p)/(TF+p))")
        {
            let p = nonnegative(&approved["value"], "lr_pseudocount", None)?;
            ensure!(p > 0.0, "Registered LR pseudocount must be positive");
            if let (Some(tn), Some(tf)) = (tn, tf) {
                ensure!(
                    (tn + p).is_finite() && (tf + p).is_finite(),
                    "LR pseudocount overflow"
                );
                values["LR"] = json!((tn + p).ln() - (tf + p).ln());
                values["missing_reasons"]
                    .as_object_mut()
                    .context("missing reasons")?
                    .remove("LR");
            }
        } else {
            missing(&mut values, "LR", "blocked:D13:pseudocount_definition");
        }
    }
    let total = phase.total;
    // An imported zero total establishes undefined indices even if components
    // were not exported; it cannot turn into a plausible-looking supplied DI.
    if values[total].as_f64().or(row[total].as_f64()) == Some(0.0) {
        for &field in phase.indices {
            if field != "LR"
                || (values["LR"].is_null()
                    && !values["missing_reasons"]["LR"]
                        .as_str()
                        .is_some_and(|reason| reason.starts_with("blocked:")))
            {
                missing(&mut values, field, "undefined:zero_exploration_total");
            }
        }
    }
    values
        .as_object_mut()
        .context("NORT metrics")?
        .retain(|field, _| field == "missing_reasons" || phase.fields().any(|name| name == field));
    values["missing_reasons"]
        .as_object_mut()
        .context("missing reasons")?
        .retain(|field, _| phase.fields().any(|name| name == field));
    let component_sources: Vec<_> = phase
        .components
        .into_iter()
        .map(|field| {
            (
                field,
                row["metric_sources"][field]
                    .as_str()
                    .unwrap_or("upstream")
                    .to_owned(),
            )
        })
        .collect();
    record_computed(row, &values, source, "derived")?;
    for (field, kind) in component_sources {
        row["metric_sources"][field] = kind.into();
    }
    let definition = inputs::choice(config, "assays.nort.exploration_quality_flag", "D06");
    let flag = "passes_15s_exploration_threshold";
    let mut quality = json!({flag:null,"missing_reasons":{}});
    let reason = match definition {
        None => Some("blocked:D06"),
        Some(definition) if definition["phase"] != row["phase"] => {
            Some("not_applicable:quality_flag_phase")
        }
        Some(definition) => {
            let operator = definition["operator"].as_str();
            ensure!(
                definition["threshold_s"].as_f64() == Some(15.0)
                    && matches!(operator, Some(">=" | ">")),
                "NORT quality flag requires registered phase, quantity and >= or > 15s"
            );
            let quantity = definition["quantity"]
                .as_str()
                .context("NORT quality flag requires an explicit quantity")?;
            optional_number(row, quantity)?.map_or(Some("not_measured:quality_quantity"), |value| {
                quality[flag] = json!(if operator == Some(">=") {
                    value >= 15.0
                } else {
                    value > 15.0
                });
                None
            })
        }
    };
    if let Some(reason) = reason {
        quality["missing_reasons"][flag] = json!(reason);
    }
    record_computed(row, &quality, source, "derived")?;
    flags[flag] = row[flag].clone();
    if row["object_configuration"].is_null()
        || row["object_configuration"]
            .as_str()
            .is_some_and(|s| s.is_empty())
    {
        flags["object_configuration_unavailable"] = json!(true);
    }
    let protocol = row
        .get("protocol_metadata")
        .filter(|v| !v.is_null())
        .cloned()
        .unwrap_or_else(|| json!({}));
    ensure!(
        protocol.is_object(),
        "NORT protocol_metadata must be an object"
    );
    let expected = [
        ("familiarization_s", 360.0),
        ("delay_h", 5.0),
        ("test_s", 360.0),
    ];
    flags["protocol_timing_unverified"] =
        json!(expected.iter().any(|(field, _)| protocol[*field].is_null()));
    let mut departures = Vec::new();
    for (field, expected) in expected {
        if !protocol[field].is_null() && nonnegative(&protocol[field], field, None)? != expected {
            departures.push(field);
        }
    }
    row["protocol_timing_departures"] = json!(departures);
    flags["protocol_timing_departure"] = json!(!departures.is_empty());
    Ok(debug)
}
