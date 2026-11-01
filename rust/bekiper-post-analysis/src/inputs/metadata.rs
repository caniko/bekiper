//! Registered animal factors and session metadata contracts.
use super::*;

pub fn validate_metadata(rows: &[Record], config: &Record) -> Result<BTreeMap<SessionKey, Record>> {
    let mut indexed = BTreeMap::new();
    let mut biology: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let levels = &config["inputs"]["factor_levels"];
    let groups = &config["inputs"]["group_mapping"];
    let genotype_policies = crate::genotype::policy_plan(config)?.is_some();
    ensure!(
        levels.is_null() || levels.is_object(),
        "factor_levels must be an object"
    );
    ensure!(
        groups.is_null() || groups.is_object(),
        "group_mapping must be an object"
    );
    let mut registered_levels = BTreeMap::new();
    if let Some(levels) = levels.as_object() {
        for (factor, values) in levels {
            ensure!(
                FACTORS.contains(&factor.as_str()),
                "Unknown factor in factor_levels: {factor}"
            );
            registered_levels.insert(factor.as_str(), unique_strings(values, "factor levels")?);
        }
    }
    if let Some(groups) = groups.as_object() {
        for (group, mapping) in groups {
            ensure!(
                !group.trim().is_empty() && mapping.is_object(),
                "group_mapping requires nonempty names and factor dictionaries"
            );
            for factor in FACTORS {
                let value = mapping[factor]
                    .as_str()
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| anyhow!("{group}: missing categorical {factor}"))?;
                if let Some(levels) = registered_levels.get(factor) {
                    ensure!(
                        levels.iter().any(|level| level == value),
                        "{group}: unmapped factor level for {factor}"
                    );
                }
            }
        }
    }
    for row in rows {
        let key = session_key(row)?;
        ensure!(
            !indexed.contains_key(&key),
            "Duplicate animal/assay/session/timepoint/phase metadata key"
        );
        for name in &METADATA_FIELDS[..8] {
            ensure!(row.get(*name).is_some(), "Metadata requires {name}");
        }
        let excluded = row["excluded"]
            .as_bool()
            .ok_or_else(|| anyhow!("Exclusion needs a boolean and an explicit reason"))?;
        ensure!(
            !excluded
                || row["exclusion_reason"]
                    .as_str()
                    .is_some_and(|s| !s.trim().is_empty()),
            "Exclusion needs an explicit reason"
        );
        for field in ["sex", "age_treatment_window", "exclusion_reason"] {
            ensure!(
                row[field].is_null() || row[field].as_str().is_some_and(|s| !s.trim().is_empty()),
                "{field} must be a nonempty string or explicit null"
            );
        }
        if let Some(value) = row.get("pair_id").filter(|v| !v.is_null()) {
            ensure!(
                !canonical_string(value)?.trim().is_empty(),
                "pair_id must be nonempty"
            );
        }
        ensure!(
            (genotype_policies && row["group"].is_null())
                || row["group"].as_str().is_some_and(|s| !s.trim().is_empty()),
            "Explicit group identity is required"
        );
        for factor in FACTORS {
            ensure!(
                (genotype_policies && row[factor].is_null())
                    || row[factor].as_str().is_some_and(|s| !s.trim().is_empty()),
                "Unmapped factor level for {factor}"
            );
            if let Some(allowed) = registered_levels.get(factor)
                && !(genotype_policies && (factor == "hcar1" || row[factor].is_null()))
            {
                ensure!(
                    allowed.iter().any(|v| row[factor] == v.as_str()),
                    "Unmapped factor level for {factor}"
                );
            }
        }
        if groups.as_object().is_some_and(|g| !g.is_empty()) && !row["group"].is_null() {
            let mapping = groups
                .get(row["group"].as_str().unwrap())
                .ok_or_else(|| anyhow!("Group/factor mapping conflicts with metadata"))?;
            ensure!(
                FACTORS.iter().all(
                    |f| (genotype_policies && (*f == "hcar1" || row[*f].is_null()))
                        || mapping.get(*f) == row.get(*f)
                ),
                "Group/factor mapping conflicts with metadata"
            );
        }
        let identity = ["five_xfad", "hcar1", "treatment", "sex", "group"]
            .map(|f| row[f].clone())
            .to_vec();
        if let Some(previous) = biology.insert(key[0].clone(), identity.clone()) {
            ensure!(
                previous == identity,
                "Inconsistent animal factors across sessions"
            );
        }
        indexed.insert(key, row.clone());
    }
    Ok(indexed)
}
