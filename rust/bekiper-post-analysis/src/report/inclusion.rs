//! Distinct exclusion, QC and missingness accounting.
use super::*;

#[derive(Default)]
struct MeasurementAvailability {
    animals: BTreeSet<String>,
    available: usize,
    excluded: usize,
    missing: usize,
}

pub(super) fn inclusion_section(
    directory: &Path,
    rows: &[Record],
    ledger: &[Record],
) -> Result<Section> {
    let mut inclusion = Section::new("Inclusion and measurement availability");
    let mut counts: BTreeMap<(String, String), MeasurementAvailability> = BTreeMap::new();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for entry in ledger {
        let specific_reasons = ["missing_reason", "exclusion_reason"]
            .into_iter()
            .filter_map(|field| entry[field].as_str().filter(|value| !value.is_empty()))
            .collect::<Vec<_>>();
        for reason in specific_reasons.iter().copied().chain(
            entry["reason"]
                .as_str()
                .filter(|_| specific_reasons.is_empty()),
        ) {
            *reasons.entry(reason.into()).or_default() += 1;
        }
        // Metadata omissions are evidence, not additional endpoint observations.
        if !entry["endpoint"].is_string() {
            continue;
        }
        let mut key = format!(
            "{} / {} / {} / {} / threshold {}",
            text(&entry["assay"]),
            text(&entry["endpoint"]),
            text(&entry["timepoint"]),
            text(&entry["phase"]),
            text(&entry["threshold_pct"])
        );
        if entry["presentation_partition"].is_object() {
            key.push_str(&format!(
                " / variant {} / {}",
                text(&entry["variant"]),
                descriptive_partition_label(&entry["presentation_partition"])?
            ));
        }
        // Group by the exact partition, not its abbreviated display fingerprint.
        let count = counts
            .entry((
                key,
                serde_json::to_string(&entry["presentation_partition"])?,
            ))
            .or_default();
        if entry["eligible_before_model_policy"] == true
            && let Some(animal) = entry["animal_id"].as_str()
        {
            count.animals.insert(animal.into());
        }
        if entry["metric_status"] == "available" {
            count.available += 1;
        }
        if entry["metadata_excluded"] == true {
            count.excluded += 1;
        }
        if !matches!(
            entry["metric_status"].as_str(),
            Some("available" | "not_applicable")
        ) {
            count.missing += 1;
        }
    }
    inclusion.paragraph("Eligible animal counts precede any model-specific pairing, covariate missingness and verification policy. Full per-session lineage is in inclusion_ledger.json; pairing_ledger.json, when present, records registered pairing.");
    if ledger
        .iter()
        .any(|entry| entry["presentation_partition"].is_object())
    {
        inclusion.paragraph("Recorded descriptive counts retain separate variants, stages, sequences and scoring policies. Partition fingerprints identify the full recorded partitions in inclusion_ledger.json.");
    }
    inclusion.table(
        &[
            "Endpoint / context",
            "Eligible animals",
            "Available records",
            "Metadata excluded records",
            "Missing / blocked records",
        ],
        counts
            .into_iter()
            .map(|((name, _), count)| {
                vec![
                    name,
                    count.animals.len().to_string(),
                    count.available.to_string(),
                    count.excluded.to_string(),
                    count.missing.to_string(),
                ]
            })
            .collect(),
    );
    inclusion.table(
        &["Reason", "Ledger records"],
        reasons
            .into_iter()
            .map(|(name, count)| vec![name, count.to_string()])
            .collect(),
    );
    let pairing = read_rows(directory, "pairing_ledger.json")?;
    let mut pairing_counts: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entry in &pairing {
        let key = format!(
            "{} / {} / {}",
            text(&entry["endpoint"]),
            text(&entry["status"]),
            text(&entry["reason"])
        );
        if let Some(animal) = entry["animal_id"].as_str() {
            pairing_counts.entry(key).or_default().insert(animal.into());
        }
    }
    inclusion.table(
        &["Pairing endpoint / status / reason", "Unique animals"],
        pairing_counts
            .into_iter()
            .map(|(key, animals)| vec![key, animals.len().to_string()])
            .collect(),
    );
    let mut flags: BTreeMap<String, usize> = BTreeMap::new();
    let mut verification: BTreeMap<String, usize> = BTreeMap::new();
    for row in rows {
        *verification
            .entry(text(&row["input_verification"]["status"]))
            .or_default() += 1;
        if let Some(values) = row["qc_flags"].as_object() {
            for (name, value) in values {
                if *value == true {
                    *flags.entry(name.clone()).or_default() += 1;
                }
            }
        }
    }
    inclusion.table(
        &["QC flag (true)", "Session / threshold records"],
        flags
            .into_iter()
            .map(|(name, count)| vec![name, count.to_string()])
            .collect(),
    );
    inclusion.table(
        &[
            "Input verification classification",
            "Session / threshold records",
        ],
        verification
            .into_iter()
            .map(|(name, count)| vec![name, count.to_string()])
            .collect(),
    );
    Ok(inclusion)
}
