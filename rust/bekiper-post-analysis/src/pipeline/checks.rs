//! Registered NORT within-group families.
use super::*;

pub(super) fn nort_within_group_tests(
    subset: &[Record],
    config: &Record,
    requirement: Option<&str>,
) -> Result<Vec<Record>> {
    let mut tests = Vec::new();
    let mut requirements = INCLUSION_REQUIREMENTS.to_vec();
    requirements.extend([
        ("inputs.group_mapping", "D10", None),
        ("statistics.nort_within_group_checks", "D09", None),
        (
            "statistics.nort_within_group_test_method",
            "D09",
            Some("one_sample_t"),
        ),
        ("statistics.test_tails", "D09", Some("two-sided")),
        (
            "statistics.confidence_interval_method",
            "D09",
            Some("student_t"),
        ),
        ("statistics.confidence_level", "D09", None),
        ("statistics.multiplicity", "D09", None),
    ]);
    let mut failed = failed_choices(config, &requirements);
    let correction = choice(config, "statistics.multiplicity", "D09");
    let family = correction.and_then(|value| value["families"]["nort_within_group"].as_str());
    if family.is_none_or(str::is_empty) || correction.is_none_or(|value| value["method"] != "holm")
    {
        failed.push("D09:NORT_multiple_comparison_correction".into());
    }
    let groups: Vec<_> = config["inputs"]["group_mapping"]
        .as_object()
        .map(|value| value.keys().cloned().collect())
        .unwrap_or_default();
    let checks: Vec<_> = config["statistics"]["nort_within_group_checks"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if failed.is_empty() {
        let confidence = choice(config, "statistics.confidence_level", "D09")
            .and_then(Value::as_f64)
            .context("confidence level")?;
        let post: Vec<_> = subset
            .iter()
            .filter(|row| row["timepoint"] == "post")
            .cloned()
            .collect();
        tests.extend(statistics::within_group(
            &post,
            &checks,
            confidence,
            requirement,
            &groups,
        )?);
    } else if !groups.is_empty() && !checks.is_empty() {
        for group in groups {
            for endpoint in &checks {
                tests.push(json!({"group":group,"endpoint":endpoint,"status":"blocked","reason":failed,"n_animals":0,"p_raw":null}));
            }
        }
    } else {
        tests.push(json!({"status":"blocked","reason":failed,"n_animals":0,"p_raw":null}));
    }
    if correction.is_some_and(|value| value["method"] == "holm")
        && family.is_some_and(|name| !name.is_empty())
    {
        statistics::adjust_family(&mut tests, "holm", family.unwrap())?;
    } else {
        let size = tests.len();
        for test in &mut tests {
            test["family_size"] = size.into();
            test["p_adjusted"] = Value::Null;
            test["adjustment_reason"] = "blocked:D09:multiplicity_family".into();
        }
    }
    Ok(tests)
}
