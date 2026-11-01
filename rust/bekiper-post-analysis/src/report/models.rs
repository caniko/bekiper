//! Registered model estimates and planned contrasts.
use super::*;

pub(super) fn model_sections(heading: &str, models: &[Record]) -> Vec<Section> {
    let mut overview = Section::new(heading);
    overview.table(
        &[
            "Model",
            "Status",
            "Animal n",
            "Contrast estimate / CI",
            "Raw P",
            "Adjusted P",
        ],
        models
            .iter()
            .map(|model| {
                let contrast = &model["contrast"];
                vec![
                    text(&model["model_id"]),
                    text(&model["status"]),
                    text(&model["n_animals"]),
                    format!(
                        "{} [{} , {}]",
                        text(&contrast["estimate"]),
                        text(&contrast["ci_lower"]),
                        text(&contrast["ci_upper"])
                    ),
                    p_value(contrast, "p_raw"),
                    p_value(contrast, "p_adjusted"),
                ]
            })
            .collect(),
    );
    if models.is_empty() {
        overview.paragraph("No model attempts supplied; requested-output status identifies applicability or blockers.");
    }
    let mut sections = vec![overview];
    for model in models {
        let mut section = Section::new(format!("Model details: {}", text(&model["model_id"])));
        section.paragraph(format!(
            "Status: {}; animal n: {}; formula: {}. Reason: {}.",
            text(&model["status"]),
            text(&model["n_animals"]),
            text(&model["formula"]),
            text(&model["reason"])
        ));
        section.paragraph(format!(
            "Registered settings: {}. Factor levels: {}.",
            text(&model["settings"]),
            text(&model["factor_levels"])
        ));
        section.paragraph(format!(
            "Design rank: {}; columns: {}; residual degrees of freedom: {}; residual variance: {}.",
            text(&model["rank"]),
            text(&model["columns"]),
            text(&model["residual_df"]),
            text(&model["residual_variance"])
        ));
        let contrast = &model["contrast"];
        if contrast.is_object() {
            section.table(
                &["Planned contrast quantity", "Value"],
                vec![
                    vec![
                        "Sign / weights".into(),
                        format!(
                            "{}; {}",
                            text(&contrast["name"]),
                            text(&contrast["weights"])
                        ),
                    ],
                    vec!["Estimate".into(), text(&contrast["estimate"])],
                    vec!["Standard error".into(), text(&contrast["standard_error"])],
                    vec![
                        "Confidence interval".into(),
                        format!(
                            "[{}, {}]",
                            text(&contrast["ci_lower"]),
                            text(&contrast["ci_upper"])
                        ),
                    ],
                    vec![
                        "Statistic / degrees of freedom".into(),
                        format!(
                            "{} / {}",
                            text(&contrast["statistic"]),
                            text(&contrast["df"])
                        ),
                    ],
                    vec![
                        "Raw P / adjusted P".into(),
                        format!(
                            "{} / {}",
                            p_value(contrast, "p_raw"),
                            p_value(contrast, "p_adjusted")
                        ),
                    ],
                    vec![
                        "Correction / family / planned family size".into(),
                        format!(
                            "{} / {} / {}",
                            text(&contrast["correction_method"]),
                            text(&contrast["correction_family"]),
                            text(&contrast["family_size"])
                        ),
                    ],
                    vec![
                        "Covariate references / estimand".into(),
                        text(&contrast["reference_values"]),
                    ],
                    vec![
                        "Contrast limitation".into(),
                        detail(contrast, &["reason", "adjustment_reason"]),
                    ],
                ],
            );
        }
        if let Some(coefficients) = model["coefficients"].as_array() {
            section.table(
                &[
                    "Coefficient",
                    "Estimability",
                    "Estimate",
                    "SE",
                    "Confidence interval",
                    "Raw P",
                ],
                coefficients
                    .iter()
                    .map(|row| {
                        vec![
                            text(&row["term"]),
                            text(&row["status"]),
                            text(&row["estimate"]),
                            text(&row["standard_error"]),
                            format!("[{}, {}]", text(&row["ci_lower"]), text(&row["ci_upper"])),
                            p_value(row, "p_raw"),
                        ]
                    })
                    .collect(),
            );
        }
        if let Some(effects) = model["genotype_lactate_effects"].as_array() {
            section.table(
                &[
                    "AD lactate - vehicle by recorded/assigned HCAR1",
                    "Estimability",
                    "Estimate [CI]",
                    "Raw / adjusted P",
                ],
                effects
                    .iter()
                    .map(|effect| {
                        vec![
                            text(&effect["hcar1"]),
                            text(&effect["status"]),
                            format!(
                                "{} [{}, {}]",
                                text(&effect["estimate"]),
                                text(&effect["ci_lower"]),
                                text(&effect["ci_upper"])
                            ),
                            format!(
                                "{} / {}",
                                p_value(effect, "p_raw"),
                                p_value(effect, "p_adjusted")
                            ),
                        ]
                    })
                    .collect(),
            );
        }
        sections.push(section);
    }
    sections
}
