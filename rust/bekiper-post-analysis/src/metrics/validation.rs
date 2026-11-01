//! Validation arithmetic and scoring contracts.
use super::*;

const TRIAD_CATEGORIES: [&str; 3] = [
    "correct_alternations",
    "incorrect_alternations",
    "neutral_triads",
];

/// Missing components leave an unmeasured remainder; a complete partition
/// must account for its total exactly.
fn partition_within(values: &[Option<f64>], total: f64) -> bool {
    let sum = values.iter().flatten().sum::<f64>();
    sum <= total && (values.contains(&None) || sum == total)
}

fn validate_bounds(row: &Record, fields: &[&str], maximum: f64) -> Result<()> {
    for &field in fields.iter().filter(|&&field| !row[field].is_null()) {
        nonnegative(&row[field], field, Some(maximum))?;
    }
    Ok(())
}

/// Read-only view created after the primitive count checks. Rechecking those
/// same fields during each relationship check adds duplicate failure paths;
/// the immutable row cannot invalidate the already established numeric domain.
struct ValidatedCounts<'a>(&'a Record);

impl ValidatedCounts<'_> {
    fn get(&self, field: &str) -> Option<f64> {
        debug_assert!(is_count_field(field));
        self.0[field].as_f64()
    }
}

fn validate_entry_counts(counts: &ValidatedCounts<'_>, stage: &str) -> Result<Option<f64>> {
    let row = counts.0;
    let sequence = row
        .get(format!("{stage}_arm_entry_sequence"))
        .filter(|value| !value.is_null())
        .map(labels)
        .transpose()?;
    let total_field = format!("{stage}_total_arm_entries");
    let supplied_total = counts.get(&total_field);
    if let (Some(total), Some(sequence)) = (supplied_total, &sequence) {
        ensure!(
            total == sequence.len() as f64,
            "{total_field} contradicts its supplied sequence"
        );
    }
    let total = supplied_total.or_else(|| sequence.as_ref().map(|s| s.len() as f64));
    let mut per_arm = [None; 3];
    for (index, arm) in ["A", "B", "C"].into_iter().enumerate() {
        let field = format!("{stage}_arm_{arm}_entries");
        per_arm[index] = counts.get(&field);
        if let (Some(count), Some(total)) = (per_arm[index], total) {
            ensure!(count <= total, "{field} exceeds {total_field}");
        }
        if let (Some(count), Some(sequence)) = (per_arm[index], &sequence) {
            ensure!(
                count
                    == sequence
                        .iter()
                        .filter(|label| label.as_str() == arm)
                        .count() as f64,
                "{field} contradicts its supplied sequence"
            );
        }
    }
    if let Some(total) = total {
        ensure!(
            partition_within(&per_arm, total),
            "{stage} per-arm counts contradict {total_field}"
        );
    }
    Ok(total)
}

/// Feasibility checks are separate from disagreement detection. A different
/// plausible imported estimate survives; a count/duration outside its possible
/// domain cannot be hidden by subsequent recomputation.
pub fn validate_supplied(row: &Record) -> Result<()> {
    let fields = row
        .as_object()
        .context("Session record must be an object")?;
    for (field, value) in fields {
        if value.is_null() {
            continue;
        }
        let count = is_count_field(field);
        if ["raw_to_scored_entry_difference", "removed_short_entries"].contains(&field.as_str()) {
            super::values::validate_entry_difference(value, field)?;
        } else if count {
            validate_count(value, field)?;
        } else if field.ends_with("_s") || field.ends_with("_m") || field.ends_with("_m_per_s") {
            nonnegative(value, field, None)?;
        }
        if let Some(number) = value.as_f64() {
            ensure!(number.is_finite(), "{field}: nonfinite supplied value");
        }
    }
    for (field, lower, upper) in [
        ("DI", -1.0, 1.0),
        ("RI", 0.0, 1.0),
        ("percent_novel", 0.0, 100.0),
        ("object_bias", -1.0, 1.0),
        ("side_bias", -1.0, 1.0),
    ]
    .into_iter()
    .chain(ymaze_score_fields().iter().map(|&field| (field, 0.0, 100.0)))
    {
        if row[field].is_null() {
            continue;
        }
        let value = row[field]
            .as_f64()
            .with_context(|| format!("Impossible supplied {field}"))?;
        ensure!(
            value.is_finite() && value >= lower && value <= upper,
            "Impossible supplied {field}"
        );
    }
    if !row["LR"].is_null() {
        ensure!(
            row["LR"].as_f64().is_some_and(f64::is_finite),
            "LR must be finite"
        );
    }
    for field in [
        "low_quality",
        "passes_15s_exploration_threshold",
        "latency_right_censored",
    ] {
        ensure!(
            row[field].is_null() || row[field].is_boolean(),
            "{field}: requires a boolean"
        );
    }
    // The _s primitive domain was checked above and this row is immutable.
    let duration = row["analysis_duration_s"].as_f64();
    if row["assay"] == "oft" {
        let bound = duration.unwrap_or(300.0).min(300.0);
        for field in [
            "center_time_s",
            "latency_to_first_center_entry_s",
            "freezing_s",
            "grooming_s",
            "rearing_s",
            "immobility_s",
            "observed_zone_time_s",
        ] {
            if !row[field].is_null() {
                let maximum = if field == "latency_to_first_center_entry_s"
                    && row["latency_right_censored"] == true
                {
                    300.0
                } else {
                    bound
                };
                nonnegative(&row[field], field, Some(maximum))?;
            }
        }
        validate_bounds(row, &["center_time_pct", "periphery_time_pct"], 100.0)?;
    } else if row["assay"] == "nort" {
        let bound = duration.unwrap_or(360.0);
        validate_bounds(
            row,
            &[
                "TA",
                "TB",
                "TN",
                "TF",
                "object_A_exploration_s",
                "object_B_exploration_s",
                "total_training_exploration",
                "total_test_exploration",
            ],
            bound,
        )?;
        // All canonical/alias components and totals passed validate_bounds
        // above. Read that established numeric domain once for relationships.
        for ([a, alias_a], [b, alias_b], total) in [
            (
                ["TA", "object_A_exploration_s"],
                ["TB", "object_B_exploration_s"],
                "total_training_exploration",
            ),
            (["TN", "TN"], ["TF", "TF"], "total_test_exploration"),
        ] {
            let a = row[a].as_f64().or(row[alias_a].as_f64());
            let b = row[b].as_f64().or(row[alias_b].as_f64());
            let supplied_total = row[total].as_f64();
            if let Some(total) = supplied_total {
                ensure!(
                    a.into_iter().chain(b).all(|component| component <= total),
                    "NORT component exploration exceeds its supplied total"
                );
            }
            if let (Some(a), Some(b)) = (a, b) {
                ensure!(
                    (a + b).is_finite() && a + b <= bound,
                    "NORT total exploration exceeds session duration {bound}"
                );
                if let Some(total) = supplied_total {
                    ensure!(
                        close(a + b, total),
                        "NORT supplied total contradicts its components"
                    );
                }
            }
        }
    } else if row["assay"] == "ymaze" {
        let counts = ValidatedCounts(row);
        let valid = validate_entry_counts(&counts, "valid")?;
        let raw = validate_entry_counts(&counts, "raw")?;
        if let (Some(valid), Some(raw)) = (valid, raw)
            && let Some(difference) = row["raw_to_scored_entry_difference"].as_f64()
        {
            ensure!(
                difference == raw - valid,
                "raw_to_scored_entry_difference contradicts the raw-scored counts"
            );
        }
        if let (Some(rejected), Some(raw)) = (counts.get("dwell_rejected_entries"), raw) {
            ensure!(rejected <= raw, "Dwell-rejected entries exceed raw entries");
        }
        let total = counts.get("total_triads");
        if let Some(valid) = valid {
            let possible = (valid - 2.0).max(0.0);
            if let Some(total) = total {
                ensure!(
                    total == possible,
                    "total_triads contradicts the supplied entry count"
                );
            }
            for field in [
                "total_triads",
                "usable_triads",
                "correct_alternations",
                "incorrect_alternations",
                "neutral_triads",
            ] {
                if let Some(count) = counts.get(field) {
                    ensure!(
                        count <= possible,
                        "{field} exceeds possible triads for {valid} entries"
                    );
                }
            }
            let categories: f64 = TRIAD_CATEGORIES
                .into_iter()
                .filter_map(|field| counts.get(field))
                .sum();
            ensure!(
                categories <= possible,
                "Triad category counts exceed possible triads"
            );
        }
        if let Some(total) = total {
            for field in [
                "usable_triads",
                "correct_alternations",
                "incorrect_alternations",
                "neutral_triads",
            ] {
                if let Some(count) = counts.get(field) {
                    ensure!(count <= total, "{field} exceeds total_triads");
                }
            }
            let categories = TRIAD_CATEGORIES.map(|field| counts.get(field));
            ensure!(
                partition_within(&categories, total),
                "Triad category counts contradict total_triads"
            );
            if let (Some(usable), Some(neutral)) =
                (counts.get("usable_triads"), counts.get("neutral_triads"))
            {
                ensure!(
                    usable + neutral == total,
                    "Usable and neutral counts contradict total_triads"
                );
            }
        }
        if let Some(usable) = counts.get("usable_triads") {
            let correct = counts.get("correct_alternations");
            let incorrect = counts.get("incorrect_alternations");
            ensure!(
                partition_within(&[correct, incorrect], usable),
                "Correct and incorrect counts contradict usable_triads"
            );
        }
    }
    Ok(())
}
