//! Cohort comparisons of recorded whole-session state fractions. No masks are
//! reconstructed and each pre/post × training/test population stays separate.
use super::*;

pub(super) fn panels(rows: &[Record], policy: &PlotPolicy<'_>) -> Result<Vec<Record>> {
    let mut inventory: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for row in rows.iter().filter(|r| r["assay"] == "nort") {
        let Some(states) = row
            .get("native_state_distributions")
            .filter(|v| !v.is_null())
        else {
            continue;
        };
        let mut seen = BTreeSet::new();
        for state in states
            .as_array()
            .context("native_state_distributions must be an array")?
        {
            ensure!(
                state["kind"] == "state_distribution",
                "Unknown native state payload"
            );
            let label = string(state, "label")?;
            ensure!(
                seen.insert(label),
                "Duplicate native state binding within session"
            );
            string(state, "basis")?;
            strings_or_default(&state["source_columns"], &[], "native source columns")?;
            let categories =
                strings_or_default(&state["category_labels"], &[], "state category labels")?;
            unique(&categories, "state categories")?;
            let seconds = state["seconds"].as_array().context("state seconds")?;
            let percentages = state["percentages"]
                .as_array()
                .context("state percentages")?;
            let counts = state["frame_counts"]
                .as_array()
                .context("state frame counts")?;
            let total =
                number(state, "denominator_seconds")?.context("state denominator_seconds")?;
            let total_frames = state["denominator_frames"]
                .as_u64()
                .context("state denominator_frames")?;
            ensure!(
                total >= 0.0
                    && seconds.len() == categories.len()
                    && percentages.len() == categories.len()
                    && counts.len() == categories.len(),
                "State distribution shape mismatch"
            );
            let mut sum = 0.0;
            let mut frames = 0_u64;
            for ((seconds, percentage), count) in seconds.iter().zip(percentages).zip(counts) {
                let seconds = seconds
                    .as_f64()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .context("state duration")?;
                let percentage = percentage
                    .as_f64()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .context("state percentage")?;
                let expected = if total > 0.0 {
                    100.0 * seconds / total
                } else {
                    0.0
                };
                ensure!(
                    (percentage - expected).abs() < 1e-8,
                    "State percentage disagrees with recorded denominator"
                );
                sum += seconds;
                frames = frames
                    .checked_add(count.as_u64().context("state count")?)
                    .context("state count overflow")?;
            }
            ensure!(
                (sum - total).abs() < 1e-8 && frames == total_frames,
                "State partition does not cover its denominator"
            );
            inventory
                .entry(label.into())
                .or_default()
                .extend(categories);
        }
    }
    let mut panels = Vec::new();
    let partitions = rows
        .iter()
        .filter(|r| r["assay"] == "nort")
        .map(|r| {
            Ok((
                string(r, "timepoint")?.to_owned(),
                string(r, "phase")?.to_owned(),
            ))
        })
        .collect::<Result<BTreeSet<_>>>()?;
    for (timepoint, phase) in partitions {
        let sessions = rows
            .iter()
            .filter(|r| r["assay"] == "nort" && r["timepoint"] == timepoint && r["phase"] == phase)
            .collect::<Vec<_>>();
        if sessions.is_empty() {
            continue; // coverage: unreachable — partitions are drawn from these same NORT sessions
        }
        for (binding, categories) in &inventory {
            for (index, category) in categories.iter().enumerate() {
                let endpoint = format!("{binding}_state_{index}_pct");
                let source = sessions
                    .iter()
                    .map(|row| {
                        let mut projected = compact_pairing_row(row);
                        let state = row["native_state_distributions"]
                            .as_array()
                            .and_then(|states| states.iter().find(|s| s["label"] == *binding));
                        let category_index = state.and_then(|s| {
                            s["category_labels"]
                                .as_array()?
                                .iter()
                                .position(|c| c == category)
                        });
                        let available = state
                            .filter(|s| s["denominator_seconds"].as_f64().is_some_and(|v| v > 0.0));
                        projected[&endpoint] = match (available, category_index) {
                            (Some(state), Some(i)) => state["percentages"][i].clone(),
                            _ => Value::Null,
                        };
                        if projected[&endpoint].is_null() {
                            if !projected["missing_reasons"].is_object() {
                                projected["missing_reasons"] = json!({});
                            }
                            projected["missing_reasons"][&endpoint] = match (state, category_index)
                            {
                                (None, _) => "native_state_not_recorded",
                                (Some(_), None) => "category_not_recorded_for_session",
                                _ => "zero_session_duration",
                            }
                            .into();
                        }
                        projected
                    })
                    .collect::<Vec<_>>();
                let sources = source.iter().collect::<Vec<_>>();
                let mut panel = categorical_panel(
                    &sources,
                    &endpoint,
                    &format!(
                        "{timepoint} {phase} · {binding}\n{category} — whole-session duration (%)"
                    ),
                    None,
                    policy,
                )?;
                panel["session_partition"] = sessions[0]
                    .get("presentation_partition")
                    .filter(|partition| partition.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({"timepoint":timepoint,"phase":phase}));
                panel["native_state_binding"] = binding.clone().into();
                panel["native_state_category"] = category.clone().into();
                panel["y_label"] = "Whole-session duration (%)".into();
                for point in panel["points"].as_array_mut().context("state points")? {
                    let row = sessions
                        .iter()
                        .find(|r| {
                            KEY.iter().all(|key| r[*key] == point[*key])
                                && (point["analysis_id"].is_null()
                                    || r["analysis_id"] == point["analysis_id"])
                        })
                        .context("native state source identity")?;
                    let state = row["native_state_distributions"]
                        .as_array()
                        .and_then(|states| states.iter().find(|s| s["label"] == *binding))
                        .context("native state source")?;
                    let i = state["category_labels"]
                        .as_array()
                        .and_then(|labels| labels.iter().position(|c| c == category))
                        .context("native state category")?;
                    point["native_state_provenance"] = json!({"binding":binding,"category":category,"seconds":state["seconds"][i],"frame_count":state["frame_counts"][i],"denominator_seconds":state["denominator_seconds"],"denominator_frames":state["denominator_frames"],"basis":state["basis"],"source_columns":state["source_columns"]});
                }
                distributions::attach(&mut panel)?;
                let styles: &[&str] = if binding.starts_with("attention_state_distribution_") {
                    &["violin", "boxplot", "histogram"]
                } else {
                    &["boxplot"]
                };
                for style in styles {
                    let mut copy = panel.clone();
                    copy["style"] = (*style).into();
                    copy["label"] = format!(
                        "{} — {style}",
                        panel["label"].as_str().context("state label")?
                    )
                    .into();
                    panels.push(copy);
                }
            }
        }
    }
    Ok(panels)
}
