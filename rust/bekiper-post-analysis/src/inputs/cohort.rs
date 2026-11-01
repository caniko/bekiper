//! Cohort joining with explicit input and QC lineage.
use super::normalize::apply_session_value_mappings;
use super::*;

pub(super) fn table(
    spec: &Value,
    name: &str,
    inputs: &Value,
    base: &Path,
    descriptors: &mut Vec<Record>,
) -> Result<Vec<Record>> {
    let descriptor = if spec.is_string() {
        json!({"path": spec})
    } else {
        spec.clone()
    };
    let path = base.join(
        descriptor["path"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow!("{name}: table descriptor requires path"))?,
    );
    let sheet = descriptor
        .get("sheet")
        .filter(|v| !v.is_null())
        .map(|v| v.as_str().ok_or_else(|| anyhow!("sheet must be a string")))
        .transpose()?;
    let (data, mut fingerprint) = read_table_pinned(&path, sheet)?;
    if let Some(expected) = descriptor.get("sha256") {
        ensure!(
            expected.as_str().is_some_and(valid_sha256) && expected == &fingerprint["sha256"],
            "{name}: table fingerprint mismatch"
        );
    }
    fingerprint["role"] = json!(name);
    fingerprint["sheet"] = json!(sheet);
    descriptors.push(fingerprint);
    let constants = descriptor.get("constants").cloned().unwrap_or(json!({}));
    ensure!(constants.is_object(), "constants must be an object");
    let mappings = inputs
        .get("session_key_mapping")
        .cloned()
        .unwrap_or(json!({}));
    apply_session_value_mappings(&mut json!({}), &mappings)?;
    let column_mapping = &inputs["column_mapping"][name];
    ensure!(
        column_mapping.is_null() || column_mapping.is_object(),
        "Column mapping must be an object"
    );
    if let Some(mapping) = column_mapping.as_object() {
        for (source, target) in mapping {
            ensure!(
                target.as_str().is_some_and(|s| !s.trim().is_empty()),
                "Column mapping targets must be strings"
            );
            ensure!(
                !adapter_owned(source) && !adapter_owned(target.as_str().unwrap()),
                "Rule evidence is adapter-owned and cannot be supplied by a measurement row"
            );
        }
    }
    ensure!(
        constants
            .as_object()
            .unwrap()
            .keys()
            .all(|k| !adapter_owned(k)),
        "Rule evidence is adapter-owned and cannot be supplied by constants"
    );
    data.into_iter()
        .map(|row| {
            let Value::Object(fields) = row else {
                bail!("Table row must be an object"); // coverage: unreachable — decode_table only yields objects
            };
            ensure!(
                fields.keys().all(|k| !adapter_owned(k)),
                "Rule evidence is adapter-owned and cannot be supplied by a measurement row"
            );
            let mut combined = constants.as_object().unwrap().clone();
            combined.extend(fields);
            let mut row = normalize_owned(Value::Object(combined), column_mapping)?;
            ensure!(
                row.as_object().unwrap().keys().all(|k| !adapter_owned(k)),
                "Rule evidence is adapter-owned and cannot be supplied by a measurement row"
            );
            apply_session_value_mappings(&mut row, &mappings)?;
            Ok(row)
        })
        .collect()
}

pub(super) fn adapter_owned(field: &str) -> bool {
    ["_rule_audit", "_rule_evidence_files"].contains(&field)
}

pub(super) fn valid_sha256(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}

pub fn load_inputs(config: &Record, base: &Path) -> Result<(Vec<Record>, Vec<Record>)> {
    let inputs = config
        .get("inputs")
        .filter(|v| v.is_object())
        .ok_or_else(|| anyhow!("inputs must be an object"))?;
    for field in [
        "column_mapping",
        "factor_levels",
        "group_mapping",
        "session_key_mapping",
        "units_and_calibration",
        "upstream_rule_provenance",
    ] {
        ensure!(
            inputs.get(field).is_none_or(Value::is_object),
            "inputs.{field} must be an object"
        );
    }
    let metadata = inputs
        .get("metadata_path")
        .filter(|v| !v.is_null())
        .ok_or_else(|| {
            anyhow!("metadata_path is required; identifiers are never inferred from row order")
        })?;
    let mut descriptors = Vec::new();
    if let Some(source) = config.get(CONFIG_FINGERPRINT) {
        ensure!(
            source.is_object()
                && source["role"] == "post_analysis_plan"
                && source["sha256"].as_str().is_some_and(valid_sha256)
                && source["bytes"].is_u64()
                && source["name"].is_string(),
            "Invalid adapter plan fingerprint"
        );
        descriptors.push(source.clone());
    }
    let mut evidence_by_assay = BTreeMap::new();
    for assay in ["oft", "nort"] {
        ensure!(
            inputs["upstream_rule_provenance"]
                .get(assay)
                .is_none_or(Value::is_object),
            "{assay}: upstream_rule_provenance must be an object"
        );
        let evidence = inputs["upstream_rule_provenance"][assay]
            .get("evidence_files")
            .cloned()
            .unwrap_or(json!([]));
        let evidence = evidence
            .as_array()
            .ok_or_else(|| anyhow!("Rule evidence_files must be an array"))?;
        let mut roles = BTreeSet::new();
        let mut files = Vec::new();
        let mut audit = Value::Null;
        for spec in evidence {
            let role = spec["role"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow!("Rule evidence requires unique roles and pinned SHA-256 fingerprints")
                })?;
            let hash = spec["sha256"]
                .as_str()
                .filter(|s| valid_sha256(s))
                .ok_or_else(|| anyhow!("Rule evidence requires pinned SHA-256 fingerprints"))?;
            ensure!(roles.insert(role), "Rule evidence requires unique roles");
            let path = base.join(
                spec["path"]
                    .as_str()
                    .ok_or_else(|| anyhow!("Rule evidence requires a path"))?,
            );
            let (contents, mut actual) = snapshot::read_pinned(&path, |file| {
                if role == "rule_audit" {
                    Ok(json::from_reader(file)?)
                } else {
                    Ok(Value::Null)
                }
            })?;
            ensure!(
                actual["sha256"] == hash,
                "{assay}: rule evidence fingerprint mismatch"
            );
            actual["role"] = json!(role);
            files.push(actual.clone());
            actual["role"] = json!(format!("{assay}_evidence:{role}"));
            descriptors.push(actual);
            if role == "rule_audit" {
                ensure!(
                    contents.is_object(),
                    "{assay}: rule audit must be a JSON object"
                );
                audit = contents;
            }
        }
        evidence_by_assay.insert(
            assay,
            json!({"_rule_evidence_files": files, "_rule_audit": audit}),
        );
    }
    let meta = validate_metadata(
        &table(metadata, "metadata", inputs, base, &mut descriptors)?,
        config,
    )?;
    let mut measurements: BTreeMap<SessionKey, Vec<Record>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for assay in crate::contracts::assays_in_report_order().iter().copied() {
        let Some(spec) = inputs.get(format!("{assay}_path")).filter(|v| !v.is_null()) else {
            continue;
        };
        let specs = spec
            .as_array()
            .cloned()
            .unwrap_or_else(|| vec![spec.clone()]);
        for descriptor in specs {
            for mut row in table(&descriptor, assay, inputs, base, &mut descriptors)? {
                let key = session_key(&row)?;
                let metadata = meta.get(&key).ok_or_else(|| {
                    anyhow!("Measurement has no unique matching metadata session")
                })?;
                ensure!(
                    row["assay"] == assay,
                    "Measurement assay does not match adapter"
                );
                let mut threshold_key = None;
                if assay == "ymaze" {
                    let mut threshold = row.get("threshold_pct").cloned().unwrap_or(Value::Null);
                    if threshold.is_null()
                        && let Some(value) = row["proximal_threshold"].as_f64()
                    {
                        threshold = json!(value * 100.0);
                        row["threshold_pct"] = threshold.clone();
                    }
                    let crossline = row["scoring_provenance"]["geometry_version"]
                        == crate::events::GEOMETRY_ARM_START
                        || row["geometry"]["version"] == crate::events::GEOMETRY_ARM_START;
                    ensure!(
                        threshold.as_f64().is_some_and(
                            |v| [25.0, 45.0, 65.0].contains(&v) || (crossline && v == 0.0)
                        ),
                        "Principal Y-maze inputs need explicit threshold_pct 25/45/65"
                    );
                    if let Some(value) = row.get("proximal_threshold").filter(|v| !v.is_null()) {
                        ensure!(
                            value
                                .as_f64()
                                .is_some_and(|v| Some(v * 100.0) == threshold.as_f64()),
                            "Conflicting Y-maze threshold_pct and proximal_threshold"
                        );
                    }
                    threshold_key = threshold.as_f64().map(|v| v as u8);
                }
                ensure!(
                    seen.insert((key.clone(), threshold_key)),
                    "Duplicate measurement session/threshold; many-to-many join forbidden"
                );
                for field in METADATA_FIELDS {
                    if let Some(value) = row.get(field) {
                        ensure!(
                            metadata.get(field) == Some(value),
                            "Measurement metadata mismatch for {field}"
                        );
                    }
                }
                measurements.entry(key).or_default().push(row);
            }
        }
    }
    let mut records = Vec::new();
    for (key, metadata) in meta {
        let rows = measurements.remove(&key).unwrap_or_else(|| vec![json!({})]);
        for measurement in rows {
            let available = !measurement.as_object().unwrap().is_empty();
            let mut record = metadata.as_object().unwrap().clone();
            let joined_flags = join_qc_flags(&metadata, &measurement)?;
            let Value::Object(measurement) = measurement else {
                bail!("Measurement must be an object"); // coverage: unreachable — table rows and the `{}` placeholder are objects
            };
            record.extend(measurement);
            record.extend(joined_flags);
            if let Some(evidence) = evidence_by_assay.get(key[1].as_str()) {
                record.extend(evidence.as_object().unwrap().clone());
            }
            record.insert("measurement_available".into(), json!(available));
            records.push(Value::Object(record));
        }
    }
    Ok((records, descriptors))
}

// Metadata and measurements can each carry upstream QC. Retain both claims and
// make conflicts explicit before the scorer adds its independent computations.
pub(super) fn join_qc_flags(metadata: &Record, measurement: &Record) -> Result<Map<String, Value>> {
    let mut result = Map::new();
    let mut flags = Map::new();
    let mut conflicts = Map::new();
    for (origin, source) in [("metadata", metadata), ("measurement", measurement)] {
        let Some(value) = source.get("qc_flags").filter(|v| !v.is_null()) else {
            continue;
        };
        let object = value.as_object().context("qc_flags must be an object")?;
        result.insert(format!("{origin}_qc_flags"), value.clone());
        for (field, value) in object {
            if let Some(previous) = flags
                .insert(field.clone(), value.clone())
                .filter(|v| v != value)
            {
                conflicts.insert(
                    field.clone(),
                    json!({"metadata":previous,"measurement":value}),
                );
            }
        }
    }
    if !result.is_empty() {
        result.insert("qc_flags".into(), Value::Object(flags));
    }
    if !conflicts.is_empty() {
        result.insert(
            "upstream_qc_join_conflicts".into(),
            Value::Object(conflicts),
        );
    }
    Ok(result)
}
