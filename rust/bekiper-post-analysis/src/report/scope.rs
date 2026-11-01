//! Run scope and registered scoring rules.
use super::*;

pub(super) fn overview_section(manifest: &Record, rows: &[Record]) -> Section {
    let mut overview = Section::new("Run scope and completion");
    overview.paragraph(format!(
        "Run status: {}. Run mode: {}. Real-data execution: {}.",
        text(&manifest["status"]),
        text(&manifest["run_mode"]),
        text(&manifest["real_data_execution"])
    ));
    if manifest["run_mode"] == "synthetic" || manifest["synthetic_fixture"] == true {
        overview.paragraph("Synthetic fixture: these observations exercise software only; no biological conclusion is implied.");
    }
    overview.paragraph(format!(
        "Requested outputs: {}.",
        text(&manifest["requested_outputs"])
    ));
    let animals = rows
        .iter()
        .filter_map(|row| row["animal_id"].as_str())
        .collect::<BTreeSet<_>>();
    let excluded = rows
        .iter()
        .filter(|row| row["excluded"] == true)
        .filter_map(|row| row["animal_id"].as_str())
        .collect::<BTreeSet<_>>();
    overview.paragraph(format!("{} session/threshold records describe {} unique animals; {} animals have metadata excluded records. Sessions, bouts and sensitivity thresholds are not independent animal replicates.", rows.len(), animals.len(), excluded.len()));
    overview.paragraph("QC flags are not exclusions. Metadata exclusions, unavailable measurements and model-specific missingness are retained separately; actual model animal n is reported with each attempt.");
    if let Some(statuses) = manifest["output_statuses"].as_object() {
        overview.table(
            &["Output", "Status", "Reason / accounting"],
            statuses
                .iter()
                .map(|(name, value)| {
                    vec![
                        name.clone(),
                        text(&value["status"]),
                        detail(
                            value,
                            &[
                                "reason",
                                "reasons",
                                "missing_metric_count",
                                "attempts",
                                "successful_attempts",
                            ],
                        ),
                    ]
                })
                .collect(),
        );
    }
    overview
}

pub(super) fn rules_section(directory: &Path, config: &Record) -> Result<Section> {
    let mut rules = Section::new("Scientific definitions and registered scope");
    rules.paragraph("The planned contrast sign is (G5-G6)-(G1-G2): G5 +1, G6 -1, G1 -1, G2 +1. Biological factors are categorical and the full 5XFAD status * HCAR1 status * treatment interaction is retained.");
    rules.paragraph("OFT centre percentages divide centre time by 300 s, rather than tracked time. Locomotion accompanies centre-avoidance readouts. Principal Y-maze 25/45/65% entries use the same-arm head-centre AND tail-base radial vertex-span predicate. The 2026-10-07 instruction sets the default to at least one second of uninterrupted membership; explicit policies can override it. Accepted visits retain the original crossing timestamp separately from qualification timing. Historical zero-dwell outputs remain identifiable by their effective policy.");
    rules.paragraph(format!("Effective Y-maze temporal policy: {}. A zero minimum disables dwell. Accepted and rejected visits, raw crossings and qualification timestamps remain auditable.", text(&config["assays"]["ymaze"]["temporal_policy"])));
    rules.paragraph("NORT raw exploration times accompany recognition indices. Source side_bias remains algebraic (-object_bias), not independently established spatial bias. Imported measurements do not by themselves verify upstream event sets or geometry.");
    rules.paragraph(format!("Primary Y-maze threshold in configuration: {}. Sensitivity selection: {}. Applicability is determined by the registered decisions and output statuses.",
        text(&config["assays"]["ymaze"]["primary_threshold_arm_length_pct"]), text(&config["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"])));
    let decisions = read_rows(directory, "decision_report.json")?;
    let mut decision_rows = Vec::new();
    for decision in &decisions {
        let resolution = &decision["resolution"];
        decision_rows.push(vec![
            text(&decision["decision_id"]),
            text(&decision["status"]),
            text(&resolution["origin"]),
            text(&resolution["approval_or_plan_reference"]),
            text(&resolution["rationale"]),
        ]);
    }
    if decision_rows.is_empty() {
        for number in 1..=14 {
            let id = format!("D{number:02}");
            let resolution = &config["decision_resolutions"][&id];
            decision_rows.push(vec![
                id,
                if resolution.is_object() {
                    "Record present; see dependency-scoped statuses".into()
                } else {
                    "Unresolved".into()
                },
                text(&resolution["origin"]),
                text(&resolution["approval_or_plan_reference"]),
                text(&resolution["rationale"]),
            ]);
        }
    }
    rules.table(
        &[
            "Decision",
            "Recorded status",
            "Origin",
            "Approval / plan",
            "Rationale",
        ],
        decision_rows,
    );
    Ok(rules)
}
