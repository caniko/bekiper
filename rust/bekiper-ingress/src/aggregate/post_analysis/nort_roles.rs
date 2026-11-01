fn map_native_nort_roles(row: &mut Record, plan: &Record) -> Result<()> {
    // Positional workbook A/B aliases are retained as evidence but cannot stand
    // in for registered TA/TB identities in the scientific scoring engine.
    let mut positional = serde_json::Map::new();
    for field in ["object_A_exploration_s", "object_B_exploration_s"] {
        if let Some(value) = row.as_object_mut().unwrap().remove(field) {
            positional.insert(field.into(), value);
        }
    }
    row["native_unmapped_object_exploration"] = positional.into();
    let source = row.clone();
    for field in [
        "missing_reasons",
        "metric_sources",
        "recomputation_missing_reasons",
        "prior_lineage",
    ] {
        if row[field].is_null() {
            row[field] = json!({});
        }
    }
    let roles = inputs::choice(plan, "assays.nort.object_role_and_location_mapping", "D10")
        .map(|mapping| {
            native(bekiper_post_analysis::events::nort_object_roles(
                row, mapping,
            ))
        })
        .transpose()?
        .flatten();
    let mut values = json!({"missing_reasons":{}});
    if let Some(roles) = roles {
        let measurements = row["native_object_measurements"]
            .as_array()
            .ok_or_else(|| {
                BekiperError::Data("native NORT needs labelled object measurements".into())
            })?;
        for (role, object) in roles.as_object().unwrap() {
            let measurement = measurements
                .iter()
                .find(|item| item["object_id"] == *object)
                .ok_or_else(|| {
                    BekiperError::Data(format!(
                        "registered NORT object has no native measurement: {object}"
                    ))
                })?;
            values[role] = measurement["exploration_s"].clone();
        }
        row["resolved_object_roles"] = roles;
    } else {
        let fields: &[&str] = if row["phase"] == "training" {
            &[
                "TA",
                "TB",
                "object_bias",
                "side_bias",
                "total_training_exploration",
            ]
        } else {
            &[
                "TN",
                "TF",
                "DI",
                "RI",
                "LR",
                "percent_novel",
                "total_test_exploration",
            ]
        };
        for field in fields {
            values[*field] = Value::Null;
            values["missing_reasons"][*field] = "blocked:D10:native_object_role_mapping".into();
        }
        if row["blocked_outputs"].is_null() {
            row["blocked_outputs"] = json!([]);
        }
        row["blocked_outputs"]
            .as_array_mut()
            .ok_or_else(|| BekiperError::Data("blocked_outputs must be a list".into()))?
            .push("nort_object_role_mapping:D10".into());
    }
    native(bekiper_post_analysis::metrics::record_computed(
        row,
        &values,
        &source,
        "native_object_role_mapping",
    ))
}

fn merge_native_qc(row: &mut Record, metadata: &Record, measurement: &Record) -> Result<()> {
    let mut flags = serde_json::Map::new();
    let mut conflicts = serde_json::Map::new();
    for (origin, source) in [("metadata", metadata), ("measurement", measurement)] {
        let Some(value) = source.get("qc_flags").filter(|value| !value.is_null()) else {
            continue;
        };
        let object = value
            .as_object()
            .ok_or_else(|| BekiperError::Data("native qc_flags must be an object".into()))?;
        row[format!("{origin}_qc_flags")] = value.clone();
        for (field, value) in object {
            if let Some(previous) = flags
                .insert(field.clone(), value.clone())
                .filter(|previous| previous != value)
            {
                conflicts.insert(
                    field.clone(),
                    json!({"metadata":previous,"measurement":value}),
                );
            }
        }
    }
    if !flags.is_empty() {
        row["qc_flags"] = flags.into();
    }
    if !conflicts.is_empty() {
        row["upstream_qc_join_conflicts"] = conflicts.into();
    }
    Ok(())
}
