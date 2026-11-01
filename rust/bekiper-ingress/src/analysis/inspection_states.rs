//! Native descriptive state partitions. No new scoring or tolerance is applied.
use bekiper_core::inspection::{FeatureInspection, HeuristicDecompositionSpec, PerimeterSpec};
use polars::prelude::*;

/// Half-open source-row durations, with a final nominal frame interval.
/// Refuse an invalid recorded clock rather than silently substituting CFR.
fn durations(df: &DataFrame, fps: f64) -> Option<(Vec<f64>, String)> {
    if let Ok(support) = df.column("sample_duration_s") {
        let values = support
            .f64()
            .ok()?
            .into_iter()
            .collect::<Option<Vec<_>>>()?;
        if values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return None;
        }
        return Some((values, "native selected half-open sample support; window-clipped final interval; evaluated masks, no extra tolerance".into()));
    }
    if !(fps.is_finite() && fps > 0.0) {
        return None;
    }
    for column in ["elapsed_s", "elapsed"] {
        if let Ok(clock) = df.column(column) {
            let clock = clock.cast(&DataType::Float64).ok()?;
            let clock = clock.f64().ok()?.into_iter().collect::<Option<Vec<_>>>()?;
            if clock.iter().any(|t| !t.is_finite()) || clock.windows(2).any(|w| w[1] <= w[0]) {
                return None;
            }
            let mut durations = clock.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>();
            if !clock.is_empty() {
                durations.push(1.0 / fps);
            }
            return Some((
                durations,
                format!(
                    "half-open {column}; final interval 1/fps; evaluated masks, no extra tolerance"
                ),
            ));
        }
    }
    Some((
        vec![1.0 / fps; df.height()],
        "half-open CFR 1/fps; evaluated masks, no extra tolerance".into(),
    ))
}

struct StatePresentation<'a> {
    label: String,
    style: &'a str,
    description: String,
    story: String,
}

fn partition(
    df: &DataFrame,
    weights: &[f64],
    basis: &str,
    columns: Vec<String>,
    labels: Vec<String>,
    classify: impl Fn(&[Option<bool>]) -> usize,
    presentation: StatePresentation<'_>,
) -> Option<FeatureInspection> {
    let StatePresentation {
        label,
        style,
        description,
        story,
    } = presentation;
    let states = columns
        .iter()
        .map(|c| df.column(c).ok()?.bool().ok())
        .collect::<Option<Vec<_>>>()?;
    let mut counts = vec![0_u64; labels.len()];
    let mut seconds = vec![0.0; labels.len()];
    for (i, weight) in weights.iter().enumerate() {
        let row = states.iter().map(|s| s.get(i)).collect::<Vec<_>>();
        let category = classify(&row);
        counts[category] += 1;
        seconds[category] += weight;
    }
    let denominator_seconds = weights.iter().sum::<f64>();
    let percentages = seconds
        .iter()
        .map(|s| {
            if denominator_seconds > 0.0 {
                100.0 * s / denominator_seconds
            } else {
                0.0
            }
        })
        .collect();
    Some(FeatureInspection::StateDistribution {
        label,
        category_labels: labels,
        frame_counts: counts,
        seconds,
        percentages,
        denominator_frames: df.height() as u64,
        denominator_seconds,
        source_columns: columns,
        basis: basis.into(),
        style: style.into(),
        description,
        representative_frame_index: None,
        color: None,
        linewidth: None,
        title_override: None,
        story_key: Some(story),
    })
}

pub(super) fn state_inspections(
    df: &DataFrame,
    specs: &[HeuristicDecompositionSpec],
    perimeters: &[PerimeterSpec],
    fps: f64,
) -> Vec<FeatureInspection> {
    let Some((weights, basis)) = durations(df, fps) else {
        return vec![];
    };
    let mut features = Vec::new();
    let target_for = |spec: &HeuristicDecompositionSpec, family: &str| {
        perimeters
            .iter()
            .find(|p| {
                !["arena", "center", "center_zone"].contains(&p.label().as_str())
                    && spec.heuristic_name == format!("{family}_{}", p.label())
            })
            .map(|p| p.label().to_string())
    };
    for spec in specs {
        let pairs: &[(&str, &str, &str)] = if spec.heuristic_name.starts_with("olfaction_") {
            &[("nose_near", "ray_hits_object", "nose")]
        } else if spec.heuristic_name.starts_with("whisker_interaction_") {
            &[
                ("left_ear_near", "left_ray_hits_object", "left"),
                ("right_ear_near", "right_ray_hits_object", "right"),
            ]
        } else {
            &[]
        };
        for (near, gaze, side) in pairs {
            let find = |name: &str| {
                spec.components
                    .iter()
                    .find(|c| c.label == name)
                    .map(|c| c.result_column.clone())
            };
            let (Some(near), Some(gaze)) = (find(near), find(gaze)) else {
                continue;
            };
            let labels = (0..8)
                .map(|state| format!("P{} G{} F{}", state & 1, (state >> 1) & 1, (state >> 2) & 1))
                .chain(std::iter::once("Missing".into()))
                .collect();
            if let Some(feature) = partition(
                df,
                &weights,
                &basis,
                vec![near.clone(), gaze.clone(), spec.result_column.clone()],
                labels,
                |row| {
                    if row.iter().any(Option::is_none) {
                        8
                    } else {
                        row.iter()
                            .enumerate()
                            .map(|(bit, state)| usize::from(state == &Some(true)) << bit)
                            .sum()
                    }
                },
                StatePresentation {
                    label: format!("sensory_disagreement_{}_{side}", spec.heuristic_name),
                    style: "stacked",
                    description: format!(
                        "{} · {side}\nP proximity · G gaze · F final; 0=false, 1=true",
                        spec.heuristic_name
                    ),
                    story: format!(
                        "sensory_disagreement_{}",
                        target_for(
                            spec,
                            if spec.heuristic_name.starts_with("olfaction_") {
                                "olfaction"
                            } else {
                                "whisker_interaction"
                            }
                        )
                        .unwrap_or_else(|| "targets".into())
                    ),
                },
            ) {
                features.push(feature);
            }
            let family = if spec.heuristic_name.starts_with("olfaction_") {
                "olfaction"
            } else {
                "whisker_interaction"
            };
            if let Some(target) = target_for(spec, family)
                && let Some(observation) = specs
                    .iter()
                    .find(|s| s.heuristic_name == format!("observation_{target}"))
            {
                let labels = (0..8)
                    .map(|state| {
                        format!("P{} G{} O{}", state & 1, (state >> 1) & 1, (state >> 2) & 1)
                    })
                    .chain(std::iter::once("Missing".into()))
                    .collect();
                if let Some(feature) = partition(
                    df,
                    &weights,
                    &basis,
                    vec![near, gaze, observation.result_column.clone()],
                    labels,
                    |row| {
                        if row.iter().any(Option::is_none) {
                            8
                        } else {
                            row.iter()
                                .enumerate()
                                .map(|(bit, state)| usize::from(state == &Some(true)) << bit)
                                .sum()
                        }
                    },
                    StatePresentation {
                        label: format!(
                            "attention_state_distribution_{}_{side}",
                            spec.heuristic_name
                        ),
                        style: "stacked",
                        description: format!(
                            "{} · {side}\nP proximity · G gaze · O final observation; 0=false, 1=true",
                            spec.heuristic_name
                        ),
                        story: format!("attention_state_distribution_{target}"),
                    },
                ) {
                    features.push(feature);
                }
            }
        }
    }
    for family in [
        "body_proximity",
        "olfaction",
        "whisker_interaction",
        "observation",
        "exploring",
    ] {
        let mut objects = specs
            .iter()
            .filter_map(|s| Some((target_for(s, family)?, s.result_column.clone())))
            .collect::<Vec<_>>();
        objects.sort();
        objects.dedup();
        if objects.is_empty() {
            continue;
        }
        let n = objects.len();
        let labels = objects
            .iter()
            .map(|(label, _)| label.clone())
            .chain([
                "Overlapping targets".into(),
                "Unassigned".into(),
                "Missing".into(),
            ])
            .collect();
        if let Some(feature) = partition(
            df,
            &weights,
            &basis,
            objects.into_iter().map(|(_, c)| c).collect(),
            labels,
            |row| {
                // Unknown target membership cannot establish an exclusive assignment.
                if row.iter().any(Option::is_none) {
                    return n + 2;
                }
                let active = row
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| **s == Some(true))
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>();
                match active.as_slice() {
                    [] => n + 1,
                    [i] => *i,
                    _ => n,
                }
            },
            StatePresentation {
                label: format!("session_composition_{family}"),
                style: "pie",
                description: format!(
                    "Whole-session {family}; exclusive targets, overlap, unassigned and missing"
                ),
                story: "session_composition".into(),
            },
        ) {
            features.push(feature);
        }
    }
    features
}

/// Compact geometric witnesses retain native boundary targets and numerical
/// angles. Sampling chooses artists only and never recomputes an angle.
pub(super) fn vector_examples(
    df: &DataFrame,
    specs: &[HeuristicDecompositionSpec],
) -> Vec<FeatureInspection> {
    let mut features = Vec::new();
    for spec in specs {
        for component in &spec.components {
            if component.ray_origin_label.is_none()
                || component.value_unit.as_deref() != Some("degrees")
            {
                continue;
            }
            let columns = [
                &component.ray_direction_x_column,
                &component.ray_direction_y_column,
                &component.ray_target_x_column,
                &component.ray_target_y_column,
                &component.value_column,
            ];
            let Some(values) = columns
                .iter()
                .map(|c| df.column(c.as_deref()?).ok()?.f64().ok())
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let candidates = (0..df.height())
                .filter(|i| {
                    values.iter().all(|c| c.get(*i).is_some_and(f64::is_finite))
                        && values[0].get(*i).unwrap().hypot(values[1].get(*i).unwrap()) > 0.0
                })
                .collect::<Vec<_>>();
            let count = candidates.len().min(6);
            for sample in 0..count {
                let index = candidates[if count > 1 {
                    sample * (candidates.len() - 1) / (count - 1)
                } else {
                    0
                }];
                features.push(FeatureInspection::HeuristicDecomposition {
                    label: format!(
                        "geometry_vectors_{}_{}_{sample}",
                        spec.heuristic_name, component.label
                    ),
                    heuristic_name: spec.heuristic_name.clone(),
                    result_column: component.result_column.clone(),
                    coordinate_labels: component.coordinate_labels.clone(),
                    perimeter_label: component.perimeter_label.clone(),
                    components: vec![component.clone()],
                    show_final_result: false,
                    description: "Native heading / selected boundary target and numerical angle"
                        .into(),
                    representative_frame_index: Some(index as u64),
                    color: None,
                    linewidth: None,
                    title_override: None,
                    story_key: Some(format!(
                        "geometry_vectors_{}_{}",
                        spec.heuristic_name, component.label
                    )),
                });
            }
        }
    }
    features
}

#[cfg(test)]
#[path = "inspection_states/tests.rs"]
mod tests;
