//! Policy comparisons and complete sensitivity attempts share both report formats.
use super::*;

pub(super) fn sections(directory: &Path, manifest: &Record) -> Result<Vec<Section>> {
    if manifest["hcar1_policies"].is_null() {
        return Ok(Vec::new());
    }
    let bytes = fs::read(crate::export::input_path(
        directory,
        "genotype_policies.json",
    ))?;
    let record: Record = serde_json::from_slice(&bytes)?;
    let comparison = &record["comparison"];
    let mut section = Section::new("HCAR1 handling: primary A and sensitivity B/C");
    section.paragraph(
        comparison["interpretation"]
            .as_str()
            .context("Policy interpretation")?,
    );
    section.paragraph("Scored endpoints and QC are shared unchanged. Recorded genotypes are not independently verified. Genotype eligibility, endpoint/model missingness, pairing and metadata exclusions are recorded separately per animal, endpoint and policy in statistics/genotype_policies.json. Neither genotype policy nor QC flags globally remove recordings. Animal counts below distinguish the full model from the four contrast-target cells; additional cells contribute to residual-variance estimation.");
    section.table(
        &[
            "Model / policy",
            "Model n / target n",
            "Estimability / model status",
            "Estimate [CI; level]",
            "Raw / adjusted P",
        ],
        comparison["rows"]
            .as_array()
            .context("Policy comparison rows")?
            .iter()
            .map(|row| {
                vec![
                    format!("{} / {}", text(&row["model_id"]), text(&row["policy"])),
                    format!(
                        "{} / {}",
                        text(&row["n_model_animals"]),
                        text(&row["n_contrast_target_animals"])
                    ),
                    format!(
                        "{} / {}",
                        text(&row["estimability_status"]),
                        text(&row["model_status"])
                    ),
                    format!(
                        "{} [{}, {}]; {}",
                        text(&row["estimate"]),
                        text(&row["ci_lower"]),
                        text(&row["ci_upper"]),
                        text(&row["confidence_level"])
                    ),
                    format!("{} / {}", p_value(row, "p_raw"), p_value(row, "p_adjusted")),
                ]
            })
            .collect(),
    );
    section.table(
        &[
            "Endpoint / sensitivity",
            "Direction changed",
            "Estimate difference from A",
            "CI excludes zero: A / sensitivity",
            "Adjusted P: A / sensitivity",
        ],
        comparison["endpoint_interpretation"]
            .as_array()
            .context("Policy interpretation rows")?
            .iter()
            .map(|row| {
                vec![
                    format!("{} / {}", text(&row["model_id"]), text(&row["policy"])),
                    if row["status"] == "compared" {
                        text(&row["direction_changed"])
                    } else {
                        text(&row["status"])
                    },
                    text(&row["estimate_difference_from_A"]),
                    format!(
                        "{} / {}",
                        text(&row["primary_interval_excludes_zero"]),
                        text(&row["sensitivity_interval_excludes_zero"])
                    ),
                    format!(
                        "{} / {}",
                        p_value(&json!({"p_adjusted":row["primary_adjusted_p"],"log_p_adjusted":row["primary_log_adjusted_p"]}),"p_adjusted"),
                        p_value(&json!({"p_adjusted":row["sensitivity_adjusted_p"],"log_p_adjusted":row["sensitivity_log_adjusted_p"]}),"p_adjusted")
                    ),
                ]
            })
            .collect(),
    );
    let mut sections = vec![section];
    for policy in record["policies"].as_array().context("Policy results")? {
        let models = policy["models"].as_array().context("Policy models")?;
        sections.extend(model_sections(
            &format!("HCAR1 {}", text(&policy["label"])),
            models,
        ));
        sections.push(check_section(
            &format!("HCAR1 {} preference checks", text(&policy["policy"])),
            policy["within_group_tests"]
                .as_array()
                .context("Policy preference checks")?,
        ));
        let mut coding = Section::new(format!(
            "HCAR1 {} eligibility and coding",
            text(&policy["policy"])
        ));
        let mut counts: BTreeMap<(String, String, String), BTreeSet<String>> = BTreeMap::new();
        for row in policy["eligibility_and_coding_audit"]
            .as_array()
            .context("Policy audit")?
        {
            let key = (
                text(&row["endpoint"]),
                text(&row["genotype_eligibility"]["eligible"]),
                text(&row["genotype_eligibility"]["reason"]),
            );
            counts
                .entry(key)
                .or_default()
                .insert(text(&row["animal_id"]));
        }
        coding.table(
            &[
                "Endpoint",
                "Genotype eligible",
                "Genotype reason",
                "Distinct animals",
            ],
            counts
                .into_iter()
                .map(|((endpoint, eligible, reason), animals)| {
                    vec![endpoint, eligible, reason, animals.len().to_string()]
                })
                .collect(),
        );
        coding.paragraph(format!("Correlations and subgroup attempts, membership/pairing/model eligibility and literal-to-analytical coding are retained in the policy artifact. Subgroup status: {}.",text(&policy["subgroup_status"])));
        sections.push(coding);
    }
    Ok(sections)
}
