fn midpoint_groups_equal(left: &MidpointGroup, right: &MidpointGroup) -> bool {
    left.output_label == right.output_label && left.source_labels == right.source_labels
}

fn validate_midpoint_shape(group: &MidpointGroup, owner: &str) -> Result<()> {
    if group.output_label.is_empty() {
        return Err(BekiperError::Config(format!(
            "{owner} midpoint group output_label cannot be empty"
        )));
    }
    if group.source_labels.is_empty() {
        return Err(BekiperError::Config(format!(
            "{owner} midpoint group '{}' must define at least one source label",
            group.output_label
        )));
    }
    let mut sources = BTreeSet::new();
    for source in &group.source_labels {
        if source.is_empty() {
            return Err(BekiperError::Config(format!(
                "{owner} midpoint group '{}' contains an empty source label",
                group.output_label
            )));
        }
        if !sources.insert(source.as_str()) {
            return Err(BekiperError::Config(format!(
                "{owner} midpoint group '{}' contains duplicate source label '{}'",
                group.output_label, source
            )));
        }
    }
    Ok(())
}

fn resolve_midpoint_groups(
    labels: &[String],
    model_groups: &[MidpointGroup],
    user_groups: &[MidpointGroup],
) -> Result<Vec<MidpointGroup>> {
    let raw_labels: BTreeSet<&str> = labels.iter().map(String::as_str).collect();
    let mut available_labels: BTreeSet<String> = labels.iter().cloned().collect();
    let mut model_by_output = HashMap::new();
    let mut merged = Vec::new();

    for group in model_groups {
        validate_midpoint_shape(group, "model-required")?;
        if raw_labels.contains(group.output_label.as_str()) {
            return Err(BekiperError::Config(format!(
                "model-required midpoint group '{}' duplicates a raw pose label",
                group.output_label
            )));
        }
        if model_by_output.contains_key(&group.output_label) {
            return Err(BekiperError::Config(format!(
                "model-required midpoint group '{}' is defined more than once",
                group.output_label
            )));
        }
        let missing: Vec<_> = group
            .source_labels
            .iter()
            .filter(|source| !available_labels.contains(source.as_str()))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(BekiperError::Config(format!(
                "model-required midpoint group '{}' missing source label(s): {}",
                group.output_label,
                missing.join(", ")
            )));
        }
        available_labels.insert(group.output_label.clone());
        model_by_output.insert(group.output_label.clone(), group.clone());
        merged.push(group.clone());
    }

    for group in user_groups {
        validate_midpoint_shape(group, "user")?;
        if let Some(model_group) = model_by_output.get(&group.output_label) {
            if midpoint_groups_equal(group, model_group) {
                tracing::debug!(
                    output_label = %group.output_label,
                    "skipping user midpoint group because it duplicates a model-required group"
                );
                continue;
            }
            return Err(BekiperError::Config(format!(
                "user midpoint group '{}' conflicts with model-required midpoint group",
                group.output_label
            )));
        }
        if raw_labels.contains(group.output_label.as_str()) {
            return Err(BekiperError::Config(format!(
                "user midpoint group '{}' duplicates a raw pose label",
                group.output_label
            )));
        }
        if let Some(existing) = merged
            .iter()
            .find(|existing| existing.output_label == group.output_label)
        {
            if midpoint_groups_equal(existing, group) {
                tracing::debug!(
                    output_label = %group.output_label,
                    "skipping duplicate user midpoint group"
                );
                continue;
            }
            return Err(BekiperError::Config(format!(
                "user midpoint group '{}' is defined more than once with different sources",
                group.output_label
            )));
        }

        let missing: Vec<_> = group
            .source_labels
            .iter()
            .filter(|source| !available_labels.contains(source.as_str()))
            .cloned()
            .collect();
        if !missing.is_empty() {
            tracing::debug!(
                output_label = %group.output_label,
                sources = ?group.source_labels,
                missing = ?missing,
                "skipping user midpoint group: not all source keypoints are available"
            );
            continue;
        }

        available_labels.insert(group.output_label.clone());
        merged.push(group.clone());
    }

    Ok(merged)
}
