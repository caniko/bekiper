//! Triads arithmetic and scoring contracts.
use super::*;

const CORRECT: [&str; 6] = ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"];
const INCORRECT: [&str; 12] = [
    "AAB", "AAC", "ABA", "ACA", "BAB", "BBA", "BBC", "BCB", "CAC", "CBC", "CCA", "CCB",
];
const NEUTRAL: [&str; 9] = [
    "AAA", "ABB", "ACC", "BAA", "BBB", "BCC", "CAA", "CBB", "CCC",
];
pub fn ymaze_scores(
    sequence: &Value,
    policy: Option<&str>,
    short_policy: Option<&str>,
    mapping: Option<&Value>,
) -> Result<Record> {
    let labels = labels(sequence)?;
    let mut result = json!({"valid_total_arm_entries":labels.len(),"valid_arm_entry_sequence":labels,"missing_reasons":{}});
    for arm in ["A", "B", "C"] {
        result[format!("valid_arm_{arm}_entries")] =
            json!(labels.iter().filter(|label| label.as_str() == arm).count());
    }
    let Some(policy) = policy else {
        for field in ymaze_score_fields().iter().copied().chain([
            "correct_alternations",
            "incorrect_alternations",
            "neutral_triads",
            "usable_triads",
            "low_quality",
            "window_labels",
        ]) {
            missing(&mut result, field, "blocked:D01");
        }
        if labels.len() >= 3 {
            result["total_triads"] = json!(labels.len() - 2);
        } else if short_policy == Some("zero_windows") {
            result["total_triads"] = json!(0);
        } else {
            missing(
                &mut result,
                "total_triads",
                "blocked:D02:insufficient_entries",
            );
        }
        return Ok(result);
    };
    let categories: BTreeMap<&str, &str> = if policy == "custom" {
        let mapping = mapping
            .and_then(Value::as_object)
            .context("Custom triad policy requires a full mapping")?;
        let keys: BTreeSet<_> = CORRECT
            .into_iter()
            .chain(INCORRECT)
            .chain(NEUTRAL)
            .collect();
        ensure!(
            mapping.keys().map(String::as_str).collect::<BTreeSet<_>>() == keys
                && mapping
                    .values()
                    .all(|v| matches!(v.as_str(), Some("correct" | "incorrect" | "neutral"))),
            "Invalid custom triad categories"
        );
        mapping
            .iter()
            .map(|(triad, category)| (triad.as_str(), category.as_str().unwrap()))
            .collect()
    } else {
        ensure!(policy == "source_table", "Unknown triad policy");
        CORRECT
            .into_iter()
            .map(|triad| (triad, "correct"))
            .chain(INCORRECT.into_iter().map(|triad| (triad, "incorrect")))
            .chain(NEUTRAL.into_iter().map(|triad| (triad, "neutral")))
            .collect()
    };
    let windows: Vec<_> = labels
        .windows(3)
        .map(|window| {
            categories
                .get(window.concat().as_str())
                .map(|category| (*category).to_owned())
                .context("Invalid triad")
        })
        .collect::<Result<_>>()?;
    let correct = windows.iter().filter(|v| v.as_str() == "correct").count();
    let incorrect = windows.iter().filter(|v| v.as_str() == "incorrect").count();
    let neutral = windows.iter().filter(|v| v.as_str() == "neutral").count();
    let total = windows.len();
    let usable = correct + incorrect;
    for (field, count) in [
        ("correct_alternations", correct),
        ("incorrect_alternations", incorrect),
        ("neutral_triads", neutral),
        ("usable_triads", usable),
        ("total_triads", total),
    ] {
        result[field] = json!(count);
    }
    result["window_labels"] = json!(windows);
    result["low_quality"] = json!(labels.len() < 8 || usable < 5);
    if labels.len() < 3 && short_policy != Some("zero_windows") {
        missing(
            &mut result,
            "total_triads",
            "blocked:D02:insufficient_entries",
        );
    }
    for (field, denominator, numerator) in [
        (ymaze_score_fields()[0], usable, correct),
        (ymaze_score_fields()[1], total, correct),
        (ymaze_score_fields()[2], total, neutral),
    ] {
        if denominator > 0 {
            result[field] = json!(numerator as f64 / denominator as f64 * 100.0);
        } else {
            missing(
                &mut result,
                field,
                if labels.len() < 3 {
                    "undefined:insufficient_entries"
                } else {
                    "undefined:zero_usable_triads"
                },
            );
        }
    }
    Ok(result)
}
