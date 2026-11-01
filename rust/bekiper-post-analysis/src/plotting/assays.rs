//! Assay-specific populations and registered sensitivity panels.
use super::*;

pub(super) fn oft_panels(source: &[&Record], policy: &PlotPolicy<'_>) -> Result<Vec<Record>> {
    let post: Vec<_> = source
        .iter()
        .copied()
        .filter(|row| row["timepoint"] == "post")
        .collect();
    [
        ("total_distance_m", "Total distance (m)"),
        ("mean_speed_m_per_s", "Mean speed (m/s)"),
        ("center_time_pct", "Center time (%) / 300 s"),
        ("center_entries", "Center entries"),
        ("latency_to_first_center_entry_s", "Latency to center (s)"),
    ]
    .iter()
    .map(|(field, label)| categorical_panel(&post, field, label, None, policy))
    .collect()
}

pub(super) fn ymaze_panels(source: &[&Record], policy: &PlotPolicy<'_>) -> Result<Vec<Record>> {
    let primary: Vec<_> = source
        .iter()
        .copied()
        .filter(|row| is_primary(row, policy))
        .collect();
    let post: Vec<_> = primary
        .iter()
        .copied()
        .filter(|row| row["timepoint"] == "post")
        .collect();
    // Only compact projected fields cross the pairing boundary; never copy a
    // coordinate array or candidate-bout payload to construct a graph.
    let compact = primary
        .iter()
        .map(|row| compact_pairing_row(row))
        .collect::<Vec<_>>();
    let (pairs, ledger) = pairing(
        &compact,
        "spontaneous_alternation_pct",
        Some(policy.verification.unwrap_or("allow_unverified")),
    )?;
    let paired: Vec<_> = pairs.iter().collect();
    let mut panels = vec![
        categorical_panel(
            &post,
            "spontaneous_alternation_pct",
            "Post-treatment SA (%) — primary",
            None,
            policy,
        )?,
        categorical_panel(
            &paired,
            "DeltaSA",
            "DeltaSA (post − pre; secondary)",
            None,
            policy,
        )?,
    ];
    panels[1]["pairing_ledger"] = json!(ledger);
    let mut points = Vec::new();
    let mut eligibility = Vec::new();
    let mut missing_count = 0;
    for pair in &pairs {
        let pre = primary
            .iter()
            .find(|row| {
                row["animal_id"] == pair["animal_id"]
                    && row["session_id"] == pair["pre_session_id"]
                    && row["timepoint"] == "pre"
            })
            .context("paired Y-maze record has no matching pre session")?;
        let identity = json!([
            pair["animal_id"],
            pair["pair_id"],
            pre["session_id"],
            pair["session_id"]
        ])
        .to_string();
        for (order, row) in [*pre, pair].iter().enumerate() {
            if let Some(mut point) = point_for(
                row,
                "spontaneous_alternation_pct",
                policy,
                &mut eligibility,
                &mut missing_count,
            )? {
                point["category"] =
                    format!("{}:{}", string(row, "group")?, string(row, "timepoint")?).into();
                point["pair_series"] = true.into();
                point["plot_pair_id"] = identity.clone().into();
                point["pair_order"] = order.into();
                points.push(point);
            }
        }
    }
    let mut paired_panel = finish_categorical(
        "spontaneous_alternation_pct",
        "Paired pre/post SA (%)",
        None,
        points,
        eligibility,
        missing_count,
        policy,
    )?;
    paired_panel["pairing_ledger"] = json!(ledger);
    panels.push(paired_panel);
    let unavailable_pairs = ledger
        .iter()
        .filter(|entry| !entry["reason"].is_null())
        .count();
    for panel in &mut panels[1..3] {
        panel["unavailable_pairs"] = unavailable_pairs.into();
        if unavailable_pairs > 0 && panel["status"] == "rendered" {
            panel["status"] = "partial".into();
        }
    }
    for (field, label) in [
        ("valid_total_arm_entries", "Valid arm entries"),
        ("total_distance_m", "Distance (m)"),
        ("mean_speed_m_per_s", "Mean speed (m/s)"),
        ("neutral_triads_pct", "Neutral triads (%)"),
    ] {
        panels.push(categorical_panel(&post, field, label, None, policy)?);
    }
    if policy.primary_threshold.is_none() {
        for panel in &mut panels {
            panel["reason"] = json!("blocked:D03:primary_threshold");
        }
    }
    Ok(panels)
}

pub(super) fn nort_panels(source: &[&Record], policy: &PlotPolicy<'_>) -> Result<Vec<Record>> {
    let training = phase_post(source, "training");
    let test = phase_post(source, "test");
    Ok(vec![
        object_panel(
            &training,
            [("TA", "A"), ("TB", "B")],
            "Training A vs B exploration (s)",
            policy,
        )?,
        object_panel(
            &test,
            [("TF", "familiar"), ("TN", "novel")],
            "Test familiar vs novel exploration (s)",
            policy,
        )?,
        categorical_panel(&test, "DI", "DI (primary)", Some(0.0), policy)?,
        categorical_panel(
            &test,
            "percent_novel",
            "Percent novel (%)",
            Some(50.0),
            policy,
        )?,
        categorical_panel(
            &test,
            "total_test_exploration",
            "Total test exploration (s)",
            None,
            policy,
        )?,
    ])
}

pub(super) fn nort_distribution_panels(
    rows: &[Record],
    policy: &PlotPolicy<'_>,
) -> Result<Vec<Record>> {
    let mut panels = Vec::new();
    for timepoint in ["pre", "post"] {
        for phase in ["training", "test"] {
            let source = rows
                .iter()
                .filter(|r| {
                    r["assay"] == "nort" && r["timepoint"] == timepoint && r["phase"] == phase
                })
                .collect::<Vec<_>>();
            if source.is_empty() {
                continue;
            }
            let endpoints = if phase == "training" {
                ["TA", "TB", "object_bias", "side_bias"]
            } else {
                ["TF", "TN", "DI", "percent_novel"]
            };
            for endpoint in endpoints {
                let (unit, formula, reference) = match endpoint {
                    "DI" => ("ratio", "(TN - TF) / (TN + TF)", Some(0.0)),
                    "percent_novel" => ("%", "100 * TN / (TN + TF)", Some(50.0)),
                    "object_bias" => ("ratio", "(TA - TB) / (TA + TB)", Some(0.0)),
                    "side_bias" => (
                        "ratio",
                        "(TB - TA) / (TA + TB); source algebraic bias, not spatial laterality",
                        Some(0.0),
                    ),
                    _ => ("s", "native measured exploration duration", None),
                };
                let mut panel = categorical_panel(
                    &source,
                    endpoint,
                    &format!("{timepoint} {phase} — {}", endpoint_label(endpoint)),
                    reference,
                    policy,
                )?;
                panel["unit"] = unit.into();
                panel["formula"] = formula.into();
                panel["session_partition"] = json!({"timepoint":timepoint,"phase":phase});
                panels.push(panel);
            }
        }
    }
    Ok(panels)
}
