//! Supplementary attempts, descriptive summaries and limitations.
use super::*;

pub(super) fn correlation_section(correlations: &[Record]) -> Section {
    let mut correlation_section = Section::new("Pairwise Y-maze QC correlations");
    correlation_section.table(
        &[
            "Pair / method",
            "Animal n",
            "Rho",
            "Raw P",
            "Adjusted P",
            "Status / reason",
        ],
        correlations
            .iter()
            .map(|row| {
                vec![
                    format!(
                        "{} vs {} ({})",
                        text(&row["x"]),
                        text(&row["y"]),
                        text(&row["method"])
                    ),
                    text(&row["n_animals"]),
                    text(&row["rho"]),
                    p_value(row, "p_raw"),
                    p_value(row, "p_adjusted"),
                    format!("{}: {}", text(&row["status"]), text(&row["reason"])),
                ]
            })
            .collect(),
    );
    if correlations.is_empty() {
        correlation_section.paragraph("No correlation attempts supplied; requested-output status identifies applicability or blockers.");
    }
    correlation_section
}

pub(super) fn subgroup_sections(directory: &Path) -> Result<Vec<Section>> {
    let mut sections = Vec::new();
    let subgroups = read_rows(directory, "subgroup_analyses.json")?;
    for subgroup in subgroups {
        let heading = format!(
            "Registered subgroup: {} = {}",
            text(&subgroup["field"]),
            text(&subgroup["level"])
        );
        let subgroup_models = subgroup["models"].as_array().cloned().unwrap_or_default();
        let mut parts = model_sections(&heading, &subgroup_models);
        if let Some(first) = parts.first_mut() {
            first.paragraph(format!(
                "Subgroup status: {}. Registered assays: {}. {}",
                text(&subgroup["status"]),
                text(&subgroup["assays"]),
                text(&subgroup["reason"])
            ));
        }
        sections.extend(parts);
        if let Some(checks) = subgroup["within_group_tests"]
            .as_array()
            .filter(|rows| !rows.is_empty())
        {
            sections.push(check_section(
                &format!("{heading}: within-group checks"),
                checks,
            ));
        }
    }
    Ok(sections)
}

pub(super) fn summary_section(directory: &Path) -> Result<Section> {
    let summaries = read_rows(directory, "summaries.json")?;
    let mut summary_section = Section::new("Figure population summaries");
    summary_section.paragraph("These are the Rust-selected plotting populations and registered uncertainty values; the renderer and report do not recompute statistics. Numeric display uses six significant digits, preserving scientific notation for small values. Full-precision values remain in summaries.json and models.json. Individual animal points and pairing identities are retained in plotting_data.json.");
    summary_section.table(
        &[
            "Figure / panel",
            "Category",
            "Animal n",
            "Mean",
            "Uncertainty",
            "Half width / status",
        ],
        summaries
            .iter()
            .map(|row| {
                vec![
                    format!("{} / {}", text(&row["figure"]), text(&row["panel"])),
                    text(&row["category"]),
                    text(&row["n_animals"]),
                    text(&row["mean"]),
                    text(&row["uncertainty"]),
                    format!(
                        "{} / {}",
                        text(&row["error_half_width"]),
                        text(&row["status"])
                    ),
                ]
            })
            .collect(),
    );
    if summaries.is_empty() {
        summary_section.paragraph("No figure-summary rows supplied.");
    }
    Ok(summary_section)
}

pub(super) fn limitations_section(
    manifest: &Record,
    models: &[Record],
    ledger: &[Record],
) -> Section {
    let mut limitations = Section::new("Blocked, unavailable and non-estimable outputs");
    let mut messages = BTreeSet::new();
    if let Some(values) = manifest["limitations"].as_array() {
        for value in values {
            messages.insert(text(value));
        }
    }
    for entry in ledger {
        if !entry["missing_reason"].is_null() {
            messages.insert(text(&entry["missing_reason"]));
        }
    }
    for model in models {
        if !model["reason"].is_null() {
            messages.insert(text(&model["reason"]));
        }
    }
    for message in messages {
        limitations.paragraph(message);
    }
    limitations.paragraph("Completion is output-specific. Unsupported or unregistered models are not simplified into alternative confirmatory analyses. Exact fit subsets, coefficient covariance and residual diagnostics remain in models.json.");
    limitations
}
