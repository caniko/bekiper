//! Registered within-group checks and correction families.
use super::*;

pub(super) fn check_section(heading: &str, checks: &[Record]) -> Section {
    let mut section = Section::new(heading);
    section.paragraph("Unavailable registered attempts remain members of the planned multiplicity family; null P values do not reduce family size.");
    section.table(
        &[
            "Group / endpoint / null",
            "Animal n",
            "Estimate / SE",
            "Raw P",
            "Adjusted P",
            "Family / m / status",
        ],
        checks
            .iter()
            .map(|row| {
                vec![
                    format!(
                        "{} / {} / {}",
                        text(&row["group"]),
                        text(&row["endpoint"]),
                        text(&row["null"])
                    ),
                    text(&row["n_animals"]),
                    format!(
                        "{} / {}",
                        text(&row["estimate"]),
                        text(&row["standard_error"])
                    ),
                    p_value(row, "p_raw"),
                    p_value(row, "p_adjusted"),
                    format!(
                        "{} / {} / {}: {}",
                        text(&row["correction_family"]),
                        text(&row["family_size"]),
                        text(&row["status"]),
                        text(&row["reason"])
                    ),
                ]
            })
            .collect(),
    );
    section.table(
        &[
            "Group / endpoint",
            "Method",
            "Confidence interval",
            "Statistic / df",
            "Correction method",
        ],
        checks
            .iter()
            .map(|row| {
                vec![
                    format!("{} / {}", text(&row["group"]), text(&row["endpoint"])),
                    text(&row["method"]),
                    format!("[{}, {}]", text(&row["ci_lower"]), text(&row["ci_upper"])),
                    format!("{} / {}", text(&row["statistic"]), text(&row["df"])),
                    text(&row["correction_method"]),
                ]
            })
            .collect(),
    );
    if checks.is_empty() {
        section.paragraph("No within-group attempts supplied.");
    }
    section
}
