//! Figure status, captions and endpoint display labels.
use super::*;

pub(super) fn figure(name: &str, panels: Vec<Record>, policy: &PlotPolicy<'_>) -> Record {
    let complete = !panels.is_empty() && panels.iter().all(|panel| panel["status"] == "rendered");
    let categorical = panels.iter().any(|panel| panel["kind"] == "categorical");
    let paired = panels.iter().any(|panel| {
        panel["points"]
            .as_array()
            .is_some_and(|points| points.iter().any(|point| point["pair_series"] == true))
    });
    let mut summary_label = if categorical {
        match policy.uncertainty {
            Some("SEM") => "Individual animals; mean ± SEM. Uncertainty requires n≥2.",
            Some("95% CI") => "Individual animals; mean and 95% CI. Uncertainty requires n≥2.",
            _ => "Individual animals; means shown; uncertainty blocked (D09).",
        }
    } else {
        "Pairwise individual animals; n denotes animals."
    }
    .to_owned();
    if paired {
        summary_label.push_str(" Lines connect registered within-animal observations.");
    }
    if panels.iter().any(|panel| {
        panel["points"]
            .as_array()
            .is_some_and(|points| points.iter().any(|point| point["censored"] == true))
    }) {
        summary_label.push_str(" Upward triangles denote right-censored latency bounds; affected means/uncertainty are withheld.");
    }
    json!({"name":name,"status":if complete {"rendered"} else {"partial"},"formats":policy.formats,"dpi":policy.dpi,"panels":panels,
        "population_policy":policy.population,"summary_uncertainty":policy.uncertainty,"summary_label":summary_label})
}

pub(super) fn endpoint_label(field: &str) -> &str {
    const LABELS: &[(&str, &str)] = &[
        ("total_distance_m", "Distance (m)"),
        ("mean_speed_m_per_s", "Mean speed (m/s)"),
        ("center_time_s", "Center time (s)"),
        ("center_time_pct", "Center time (%)"),
        ("center_entries", "Center entries"),
        ("latency_to_first_center_entry_s", "Latency to center (s)"),
        ("spontaneous_alternation_pct", "SA (%)"),
        ("spontaneous_alternation_pct_standard", "Standard SA (%)"),
        ("DeltaSA", "DeltaSA (percentage points)"),
        ("valid_total_arm_entries", "Valid arm entries"),
        ("neutral_triads_pct", "Neutral triads (%)"),
        ("exploration_s", "Object exploration (s)"),
        ("TA", "Training A exploration (s)"),
        ("TB", "Training B exploration (s)"),
        ("TF", "Familiar exploration (s)"),
        ("TN", "Novel exploration (s)"),
        ("DI", "DI"),
        ("percent_novel", "Percent novel (%)"),
        ("total_test_exploration", "Total test exploration (s)"),
        ("object_bias", "Object bias (ratio)"),
        ("side_bias", "Source algebraic side_bias (ratio)"),
        ("valid_bouts", "Valid bouts (count)"),
        ("removed_duration_bouts", "Removed duration bouts (count)"),
        ("removed_speed_bouts", "Removed speed bouts (count)"),
        ("removed_angle_bouts", "Removed angle bouts (count)"),
    ];
    LABELS
        .iter()
        .find(|(endpoint, _)| *endpoint == field)
        .map_or(field, |(_, label)| *label)
}
