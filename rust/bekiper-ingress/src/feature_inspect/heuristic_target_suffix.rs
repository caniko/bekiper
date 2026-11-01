use std::cmp::Reverse;

/// Extract the target suffix from a heuristic name by checking against
/// known perimeter labels. For `olfaction_object_1` returns `"object_1"`,
/// for `freezing` returns `None`.
///
/// Port of Python `_heuristic_target_suffix()` from `_grouping.py:50-63`.
fn heuristic_target_suffix(name: &str, perimeter_labels: &HashSet<String>) -> Option<String> {
    // Check longest labels first to avoid false positives.
    let mut sorted_labels: Vec<&String> = perimeter_labels.iter().collect();
    sorted_labels.sort_by_key(|label| Reverse(label.len()));

    for plabel in sorted_labels {
        if name.ends_with(&format!("_{plabel}")) {
            return Some(plabel.clone());
        }
        if name == format!("in_{plabel}") {
            return Some(plabel.clone());
        }
    }
    None
}

const CONTEXT_TARGETS: &[&str] = &["arena", "center", "center_zone"];

fn is_physical_object_target(target: &str) -> bool {
    target.starts_with("object_") || matches!(target, "novel" | "familiar")
}

fn is_angle_family(name: &str) -> bool {
    name.contains("olfaction") || name.contains("whisker_interaction")
}

fn decomposition_story_key(
    heuristic_name: &str,
    perimeter_labels: &HashSet<String>,
    object_targets: &HashSet<String>,
) -> Option<String> {
    let target = heuristic_target_suffix(heuristic_name, perimeter_labels)?;
    if CONTEXT_TARGETS.contains(&target.as_str()) {
        return None;
    }

    let suffix = format!("_{target}");
    let family = heuristic_name.strip_suffix(&suffix)?;
    if family.is_empty() {
        return None;
    }

    if object_targets.contains(&target) && is_angle_family(family) {
        return Some(format!("object_decomposition_{family}_{target}"));
    }

    Some(format!("object_decomposition_{family}"))
}

/// Derive the story grouping key for a single feature.
///
/// Port of Python `_derive_story_key()` from `_grouping.py:153-200`.
fn derive_story_key(
    feature: &FeatureInspection,
    perimeter_labels: &HashSet<String>,
    heuristic_targets: &HashMap<String, Option<String>>,
    result_col_to_name: &HashMap<String, String>,
    object_targets: &HashSet<String>,
) -> Option<String> {
    match feature {
        FeatureInspection::NortDiscrimination { .. } => Some("discrimination".into()),
        FeatureInspection::ArmSequence { .. } => Some("arm_sequence".into()),
        FeatureInspection::YMazeArmSections { .. } => Some("arm_sequence".into()),
        FeatureInspection::CenterPeriphery { .. } => Some("center_periphery".into()),
        FeatureInspection::Perimeter { .. } => Some("geometry".into()),
        FeatureInspection::Angle { .. } => Some("geometry".into()),
        FeatureInspection::HeuristicDecomposition { heuristic_name, .. } => {
            decomposition_story_key(heuristic_name, perimeter_labels, object_targets)
        }

        FeatureInspection::Heuristic { result_column, .. }
        | FeatureInspection::HeuristicCloud { result_column, .. }
        | FeatureInspection::Zone { result_column, .. } => {
            // Check if this heuristic's result column maps to a known heuristic name.
            if let Some(hname) = result_col_to_name.get(result_column.as_str()) {
                if let Some(Some(target)) = heuristic_targets.get(hname.as_str()) {
                    return Some(if object_targets.contains(target) {
                        target.clone()
                    } else {
                        "objects".into()
                    });
                }
                // Standalone heuristic → use heuristic name as key.
                return Some(hname.clone());
            }
            // No matching heuristic meta — try perimeter_label for Zone variant.
            if let FeatureInspection::Zone {
                perimeter_label, ..
            } = feature
                && perimeter_labels.contains(perimeter_label.as_str())
            {
                return Some(if object_targets.contains(perimeter_label) {
                    perimeter_label.clone()
                } else {
                    "objects".into()
                });
            }
            None
        }

        FeatureInspection::Proximity {
            result_column,
            perimeter_label,
            ..
        } => {
            if let Some(hname) = result_col_to_name.get(result_column.as_str()) {
                if let Some(Some(target)) = heuristic_targets.get(hname.as_str()) {
                    return Some(if object_targets.contains(target) {
                        target.clone()
                    } else {
                        "objects".into()
                    });
                }
                return Some(hname.clone());
            }
            if perimeter_labels.contains(perimeter_label.as_str()) {
                return Some(if object_targets.contains(perimeter_label) {
                    perimeter_label.clone()
                } else {
                    "objects".into()
                });
            }
            None
        }

        // ScaleDerivation, TimeSeries, Trajectory, Quadrant, RayCast: ungrouped.
        _ => None,
    }
}

/// Populate `story_key` for every feature that doesn't already have one.
/// This pre-computes the grouping that Python's `_grouping.py` would
/// otherwise derive at render time from naming conventions.
pub fn populate_story_keys(
    features: &mut [FeatureInspection],
    heuristics: &[HeuristicMeta],
    perimeters: &[PerimeterSpec],
) {
    let perimeter_labels: HashSet<String> =
        perimeters.iter().map(|p| p.label().to_string()).collect();
    let object_targets = perimeter_labels
        .iter()
        .filter(|target| {
            is_physical_object_target(target)
                || (!CONTEXT_TARGETS.contains(&target.as_str())
                    && heuristics.iter().any(|meta| {
                        [
                            "olfaction_",
                            "whisker_interaction_",
                            "body_proximity_",
                            "exploring_",
                        ]
                        .iter()
                        .any(|family| meta.name.starts_with(family))
                            && meta.name.ends_with(&format!("_{target}"))
                    }))
        })
        .cloned()
        .collect();

    // Build lookup: heuristic name → target suffix.
    let heuristic_targets: HashMap<String, Option<String>> = heuristics
        .iter()
        .map(|h| {
            let target = heuristic_target_suffix(&h.name, &perimeter_labels);
            (h.name.clone(), target)
        })
        .collect();

    // Build lookup: result_column → heuristic name.
    let result_col_to_name: HashMap<String, String> = heuristics
        .iter()
        .map(|h| (h.result_column.clone(), h.name.clone()))
        .collect();

    for feature in features.iter_mut() {
        if feature.story_key().is_some() {
            continue;
        }
        if let Some(key) = derive_story_key(
            feature,
            &perimeter_labels,
            &heuristic_targets,
            &result_col_to_name,
            &object_targets,
        ) {
            feature.set_story_key(key);
        }
    }
}
