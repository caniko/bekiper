//! Registered within-group checks.
use super::*;

/// All registered groups are materialized before selecting observations, so an
/// empty group cannot reduce the declared multiplicity family.
pub fn within_group(
    rows: &[Record],
    checks: &[String],
    confidence: f64,
    requirement: Option<&str>,
    planned_groups: &[String],
) -> Result<Vec<Record>> {
    ensure!(
        !checks.is_empty()
            && checks
                .iter()
                .all(|check| matches!(check.as_str(), "DI" | "percent_novel"))
            && checks.iter().collect::<BTreeSet<_>>().len() == checks.len(),
        "Register unique DI and/or percent_novel checks"
    );
    ensure!(
        confidence.is_finite() && confidence > 0.0 && confidence < 1.0,
        "Register a valid confidence level"
    );
    ensure!(
        !planned_groups.is_empty()
            && planned_groups.iter().all(|group| !group.is_empty())
            && planned_groups.iter().collect::<BTreeSet<_>>().len() == planned_groups.len(),
        "Register unique explicit planned groups"
    );
    validate_requirement(requirement)?;
    for row in rows {
        ensure!(
            (row["genotype_eligibility"].is_object() && row["group"].is_null())
                || planned_groups.iter().any(|group| row["group"] == *group),
            "Observed group is outside registered within-group plan"
        );
    }
    for endpoint in checks {
        // Also forbid replication across groups, not only inside a group.
        analysis_subset(rows, &[endpoint.as_str(), "group"], requirement)?;
    }
    let mut results = Vec::new();
    for group in planned_groups {
        let members: Vec<_> = rows
            .iter()
            .filter(|row| row["group"] == *group)
            .cloned()
            .collect();
        for endpoint in checks {
            let chance_reference = if endpoint == "DI" { 0.0 } else { 50.0 };
            let (used, subset) = analysis_subset(&members, &[endpoint.as_str()], requirement)?;
            let values = used
                .iter()
                .map(|row| number(row, endpoint))
                .collect::<Result<Vec<_>>>()?;
            let mut result = json!({"group":group,"endpoint":endpoint,"null":chance_reference,"method":"one_sample_t",
                "n_animals":values.len(),"p_raw":null,"log_p_raw":null,"reason":"insufficient_data",
                "subset":subset,"status":"unavailable"});
            if values.len() >= 2 {
                let mut mean = 0.0;
                let mut deviations = 0.0;
                for (index, value) in values.iter().enumerate() {
                    let difference = value - mean;
                    mean += difference / (index + 1) as f64;
                    deviations += difference * (value - mean);
                }
                let sd = (deviations / (values.len() - 1) as f64).sqrt();
                let summary = t_summary(
                    mean - chance_reference,
                    sd / (values.len() as f64).sqrt(),
                    values.len() - 1,
                    confidence,
                )?;
                for (key, value) in summary.as_object().context("t summary object")? {
                    result[key] = value.clone();
                }
                result["mean"] = mean.into();
                result["sample_standard_deviation"] = sd.into();
                if result["reason"].is_null() {
                    result["status"] = "computed".into();
                }
            }
            results.push(result);
        }
    }
    Ok(results)
}
