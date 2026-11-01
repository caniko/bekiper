use std::collections::HashMap;

use bekiper_behaviour::TaskDetail;
use bekiper_behaviour::object_recognition::NortResult;
use bekiper_behaviour::open_field::OftResult;
use bekiper_behaviour::radial_arm::DEFAULT_PROXIMAL_THRESHOLD;
use bekiper_behaviour::radial_arm::YMazeResult;
use bekiper_core::inspection::{
    ArmThresholdLine, CoordinateColumnSpec, FeatureInspection, HeuristicDecompositionSpec,
    HeuristicMeta, PerimeterSpec, QuadrantSpec, QualiaChannelSpec,
};
use bekiper_core::scale::ScaleDerivationInfo;
use bekiper_core::types::{Label, TaskType};

/// Common body-axis pairs we know how to draw a heading vector for, in
/// priority order. The first pair where both labels exist in
/// `coordinate_columns` is used to emit an `Angle` feature.
const BODY_AXIS_PAIRS: &[(&str, &str)] = &[
    ("snout", "tail_base"),
    ("nose", "tail_base"),
    ("snout", "tail"),
    ("nose", "tail"),
    ("head", "tail_base"),
    ("ear_midpoint", "tail_base"),
];

/// Heuristic name prefixes/substrings that indicate a heuristic is
/// spatially anchored to a perimeter (so we should overlay it).
const SPATIAL_HEURISTIC_HINTS: &[&str] = &["in_", "_zone", "zone_", "perimeter", "confined"];

/// Build the full feature inspection list for a single experiment's
/// `AnalysisResult` plus its associated perimeters/quadrants.
///
/// The output ordering is intentional and stable: perimeters first
/// (always present), then quadrants, then heuristics (in input order),
/// then optional body-axis angles, then ray-cast cones, then a
/// primary-label trajectory, then scale derivation, then time series.
///
/// Specification for a ray-cast sensing cone to emit as a `FeatureInspection::RayCast`.
/// Each tuple is `(origin_label, angle_column, length_meters, cone_width_deg)`.
pub type RayCastSpec = (String, String, f64, f64);

/// Specification for a proximity buffer to emit as a `FeatureInspection::Proximity`.
/// Each tuple is `(perimeter_label, result_column, buffer_meters, coordinate_label)`.
pub type ProximitySpec = (String, String, f64, String);

#[allow(clippy::too_many_arguments)]
pub fn collect_feature_inspections(
    perimeters: &[PerimeterSpec],
    quadrants: &[QuadrantSpec],
    heuristics: &[HeuristicMeta],
    decompositions: &[HeuristicDecompositionSpec],
    coordinate_columns: &HashMap<Label, CoordinateColumnSpec>,
    primary_label: Option<&Label>,
    scale_derivation: Option<&ScaleDerivationInfo>,
    ray_cast_specs: &[RayCastSpec],
    motion_columns: &[(&str, &str)],
    proximity_specs: &[ProximitySpec],
) -> Vec<FeatureInspection> {
    let mut features: Vec<FeatureInspection> = Vec::new();

    // Resolve a primary label: prefer the explicit one, otherwise pick a
    // sensible default from the coordinate columns (snout > nose > first).
    let primary = primary_label
        .cloned()
        .or_else(|| pick_default_primary_label(coordinate_columns));
    let decomposition_by_name: HashMap<&str, &HeuristicDecompositionSpec> = decompositions
        .iter()
        .map(|spec| (spec.heuristic_name.as_str(), spec))
        .collect();

    // 1. Bare perimeter overlays.
    for p in perimeters {
        let perimeter_label = p.label().to_string();
        features.push(FeatureInspection::Perimeter {
            label: format!("perimeter_{perimeter_label}"),
            perimeter_label: perimeter_label.clone(),
            description: format!("Perimeter '{perimeter_label}' on experiment frame"),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        });
    }

    // 2. Quadrant grids.
    for q in quadrants {
        let coordinate_label = primary.clone().unwrap_or_else(|| "primary".to_string());
        features.push(FeatureInspection::Quadrant {
            label: format!("quadrant_{}", q.label),
            quadrant_label: q.label.clone(),
            coordinate_labels: vec![coordinate_label],
            description: format!(
                "{}x{} quadrant grid '{}' on experiment frame",
                q.rows, q.cols, q.label
            ),
            occupancy_grid: None,
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        });
    }

    // 3. Heuristics: spatial heuristics get trajectory-on-frame figures;
    //    temporal heuristics such as freezing may emit a point-cloud
    //    figure instead.
    //    If the heuristic name maps to a known perimeter (or contains one
    //    of `SPATIAL_HEURISTIC_HINTS`), the matching perimeter is overlaid
    //    too — these become the "zone confinement" inspection figures from
    //    the legacy stack.
    for h in heuristics {
        let coordinate_label = primary.clone().unwrap_or_else(|| "primary".to_string());
        let perimeter_label = decomposition_by_name
            .get(h.name.as_str())
            .and_then(|spec| spec.perimeter_label.clone())
            .filter(|label| {
                perimeters
                    .iter()
                    .any(|perimeter| perimeter.label() == label)
            })
            .or_else(|| match_heuristic_to_perimeter(h, perimeters));

        // If we resolved both a perimeter and the heuristic looks like a
        // confinement check, emit the richer Zone variant; otherwise the
        // generic Heuristic variant.
        let is_zone = perimeter_label.is_some() && looks_spatial(&h.name);
        if is_zone {
            let perim = perimeter_label.clone().unwrap();
            features.push(FeatureInspection::Zone {
                label: format!("zone_{}", h.name),
                perimeter_label: perim.clone(),
                result_column: h.result_column.clone(),
                coordinate_labels: vec![coordinate_label.clone()],
                description: format!("Zone confinement '{}' inside perimeter '{}'", h.name, perim),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        } else if h.name == "freezing" {
            features.push(FeatureInspection::HeuristicCloud {
                label: format!("heuristic_{}", h.name),
                result_column: h.result_column.clone(),
                coordinate_labels: vec![coordinate_label],
                perimeter_label,
                description: format!("Heuristic '{}' over experiment frame", h.name),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        } else {
            features.push(FeatureInspection::Heuristic {
                label: format!("heuristic_{}", h.name),
                result_column: h.result_column.clone(),
                coordinate_labels: vec![coordinate_label],
                perimeter_label,
                description: format!("Heuristic '{}' over experiment frame", h.name),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        }

        if let Some(spec) = decomposition_by_name.get(h.name.as_str()) {
            features.push(FeatureInspection::HeuristicDecomposition {
                label: format!("heuristic_decomposition_{}", spec.heuristic_name),
                heuristic_name: spec.heuristic_name.clone(),
                result_column: spec.result_column.clone(),
                coordinate_labels: spec.coordinate_labels.clone(),
                perimeter_label: spec.perimeter_label.clone(),
                components: spec.components.clone(),
                show_final_result: spec.show_final_result,
                description: format!("Heuristic decomposition for '{}'", spec.heuristic_name),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        }
    }

    // Explicit full-trial diagnostic outputs also work in standalone mode.
    // They keep their native masks and bindings; no renderer has to rescore.
    for spec in decompositions {
        if !heuristics.iter().any(|h| h.name == spec.heuristic_name) {
            features.push(FeatureInspection::HeuristicDecomposition {
                label: format!("heuristic_decomposition_{}", spec.heuristic_name),
                heuristic_name: spec.heuristic_name.clone(),
                result_column: spec.result_column.clone(),
                coordinate_labels: spec.coordinate_labels.clone(),
                perimeter_label: spec.perimeter_label.clone(),
                components: spec.components.clone(),
                show_final_result: spec.show_final_result,
                description:
                    "Native diagnostic comparison; descriptive states with recorded policy".into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        }
        for component in &spec.components {
            let family = if component.ray_origin_label.is_some() {
                "gaze_field"
            } else if component.buffer_meters.is_some() {
                "proximity_diagnostic"
            } else {
                continue;
            };
            let label = format!("{family}_{}_{}", spec.heuristic_name, component.label);
            features.push(FeatureInspection::HeuristicDecomposition {
                label: label.clone(),
                heuristic_name: label,
                result_column: component.result_column.clone(),
                coordinate_labels: component.coordinate_labels.clone(),
                perimeter_label: component.perimeter_label.clone(),
                components: vec![component.clone()],
                show_final_result: false,
                description: format!(
                    "Full-trial {family} for {}: {}",
                    spec.heuristic_name, component.label
                ),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        }
        if spec.heuristic_name.starts_with("olfaction_") {
            let components: Vec<_> = spec
                .components
                .iter()
                .filter(|c| matches!(c.label.as_str(), "nose_near" | "ray_hits_object"))
                .cloned()
                .collect();
            if components.len() == 2 {
                features.push(FeatureInspection::HeuristicDecomposition {
                    label: format!("recipe_{}", spec.heuristic_name),
                    heuristic_name: format!("recipe_{}", spec.heuristic_name),
                    result_column: spec.result_column.clone(), coordinate_labels: spec.coordinate_labels.clone(),
                    perimeter_label: spec.perimeter_label.clone(), components, show_final_result: true,
                    description: "Historical stacked proximity / direction / final olfaction recipe; contact and outside gates remain in the detailed decomposition".into(),
                    representative_frame_index: None, color: None, linewidth: None,
                    title_override: None, story_key: None,
                });
            }
        }
        if spec.heuristic_name.starts_with("body_proximity_") {
            let components = ["center_ear_near", "torso_near", "tail_base_near"]
                .iter()
                .filter_map(|label| {
                    spec.components
                        .iter()
                        .find(|component| component.label == *label)
                })
                .cloned()
                .collect();
            features.push(FeatureInspection::HeuristicDecomposition {
                label: format!("recipe_{}", spec.heuristic_name),
                heuristic_name: format!("recipe_{}", spec.heuristic_name),
                result_column: spec.result_column.clone(),
                coordinate_labels: spec.coordinate_labels.clone(),
                perimeter_label: spec.perimeter_label.clone(),
                components,
                show_final_result: true,
                description:
                    "Historical head / torso (when available) / tail / final body proximity recipe"
                        .into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        }
    }

    // Heuristic-specific multi-object ray overlays are usable as figures and
    // native inspection videos. Each target retains its own acceptance mask.
    for family in ["olfaction", "whisker_interaction"] {
        let mut components = Vec::new();
        let mut targets = std::collections::BTreeSet::new();
        for spec in decompositions
            .iter()
            .filter(|s| s.heuristic_name.starts_with(&format!("{family}_")))
        {
            for component in spec
                .components
                .iter()
                .filter(|c| c.ray_origin_label.is_some())
            {
                let mut component = component.clone();
                component.label = format!("{} — {}", spec.heuristic_name, component.label);
                component.result_column = spec.result_column.clone();
                components.push(component);
                targets.insert(spec.perimeter_label.clone());
            }
        }
        if targets.len() > 1 {
            let name = format!("multiobject_{family}");
            features.push(FeatureInspection::HeuristicDecomposition {
                label: name.clone(),
                heuristic_name: name,
                result_column: components[0].result_column.clone(),
                coordinate_labels: components[0].coordinate_labels.clone(),
                perimeter_label: None,
                components,
                show_final_result: false,
                description: "All objects; explicit native directions and per-target final masks"
                    .into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        }
    }

    // Cumulative 2×2 sensory observation: three channels and the actual
    // configured observation/fusion output, with its declared result binding.
    for perimeter in perimeters {
        let target = perimeter.label();
        let channels: Vec<_> = ["body_proximity", "olfaction", "whisker_interaction"]
            .iter()
            .filter_map(|family| {
                decompositions
                    .iter()
                    .find(|s| s.heuristic_name == format!("{family}_{target}"))
            })
            .collect();
        let final_spec = decompositions
            .iter()
            .find(|s| s.heuristic_name == format!("observation_{target}"))
            .or_else(|| {
                decompositions
                    .iter()
                    .find(|s| s.heuristic_name == format!("exploring_{target}"))
            });
        if channels.len() == 3
            && let Some(final_spec) = final_spec
        {
            let name = format!("cumulative_observation_{target}");
            features.push(FeatureInspection::HeuristicDecomposition {
                label: name.clone(),
                heuristic_name: name,
                result_column: final_spec.result_column.clone(),
                coordinate_labels: final_spec.coordinate_labels.clone(),
                perimeter_label: Some(target.to_string()),
                components: channels
                    .iter()
                    .map(|s| {
                        let mut c = s
                            .components
                            .iter()
                            .find(|c| c.label == "raw_combined")
                            .cloned()
                            .unwrap_or_default();
                        c.label = s.heuristic_name.clone();
                        c.result_column = s.result_column.clone();
                        c.coordinate_labels = s.coordinate_labels.clone();
                        c.perimeter_label = s.perimeter_label.clone();
                        c
                    })
                    .collect(),
                show_final_result: true,
                description: "Cumulative sensory channels and evaluated final observation".into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        }
    }

    // 4. Body-axis angle if a known label pair exists.
    if let Some((from, to)) = pick_body_axis_pair(coordinate_columns) {
        features.push(FeatureInspection::Angle {
            label: format!("angle_{from}_{to}"),
            from_coordinate_label: from.clone(),
            to_coordinate_label: to.clone(),
            description: format!("Body axis vector ({from} → {to}) on experiment frame"),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        });
    }

    // 5. Ray-cast sensing cones.
    for (origin_label, angle_column, length_meters, cone_width_deg) in ray_cast_specs {
        if coordinate_columns.contains_key(origin_label.as_str()) {
            features.push(FeatureInspection::RayCast {
                label: format!("raycast_{origin_label}"),
                origin_label: origin_label.clone(),
                angle_column: angle_column.clone(),
                length_meters: *length_meters,
                cone_width_deg: *cone_width_deg,
                description: format!(
                    "Sensing cone at '{origin_label}' ({cone_width_deg}° width, {length_meters} m)"
                ),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        }
    }

    // 6. Proximity buffer figures.
    for (perimeter_label, result_column, buffer_meters, coordinate_label) in proximity_specs {
        features.push(FeatureInspection::Proximity {
            label: format!("proximity_{perimeter_label}"),
            perimeter_label: perimeter_label.clone(),
            buffer_meters: *buffer_meters,
            result_column: result_column.clone(),
            coordinate_labels: vec![coordinate_label.clone()],
            description: format!("Proximity to {perimeter_label} within {buffer_meters}m buffer"),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        });
    }

    // 7. Baseline trajectory for the primary label — but only if no
    //    earlier Heuristic/Zone already draws the same coordinate label.
    //    Those variants already render the trajectory (colored by a
    //    boolean column), so an extra plain-trajectory figure for the
    //    same label would be redundant clutter in the output dir.
    if let Some(label) = primary {
        let already_covered = features.iter().any(|f| match f {
            FeatureInspection::Heuristic {
                coordinate_labels, ..
            }
            | FeatureInspection::HeuristicCloud {
                coordinate_labels, ..
            }
            | FeatureInspection::Zone {
                coordinate_labels, ..
            }
            | FeatureInspection::Proximity {
                coordinate_labels, ..
            } => coordinate_labels.contains(&label),
            _ => false,
        });
        if !already_covered {
            features.push(FeatureInspection::Trajectory {
                label: format!("trajectory_{label}"),
                coordinate_labels: vec![label.clone()],
                description: format!("Trajectory of '{label}' on experiment frame"),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            });
        }
    }

    // 7. Scale derivation — how meters-per-pixel was derived.
    //    Skipped when the method is "manual" (nothing interesting to visualize).
    if let Some(sd) = scale_derivation
        && sd.method != "manual"
    {
        features.push(FeatureInspection::ScaleDerivation {
            label: "scale_derivation".into(),
            description: "How meters-per-pixel was derived".into(),
            representative_frame_index: Some(0),
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        });
    }

    // 8. Time series for motion-derived scalar columns (e.g. speed, displacement).
    for (col, unit) in motion_columns {
        features.push(FeatureInspection::TimeSeries {
            label: format!("timeseries_{col}"),
            value_column: col.to_string(),
            unit: unit.to_string(),
            description: format!("{col} over time ({unit})"),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        });
    }

    features
}

/// Pick the most informative coordinate label for trajectory overlays.
fn pick_default_primary_label(
    coordinate_columns: &HashMap<Label, CoordinateColumnSpec>,
) -> Option<Label> {
    for preferred in ["snout", "nose", "head", "body_center", "centroid"] {
        if coordinate_columns.contains_key(preferred) {
            return Some(preferred.to_string());
        }
    }
    // Fall back to any label, deterministically (sorted).
    let mut keys: Vec<&Label> = coordinate_columns.keys().collect();
    keys.sort();
    keys.first().map(|k| (*k).clone())
}

/// Find the first body-axis pair where both labels exist.
fn pick_body_axis_pair(
    coordinate_columns: &HashMap<Label, CoordinateColumnSpec>,
) -> Option<(Label, Label)> {
    for (from, to) in BODY_AXIS_PAIRS {
        if coordinate_columns.contains_key(*from) && coordinate_columns.contains_key(*to) {
            return Some(((*from).to_string(), (*to).to_string()));
        }
    }
    None
}

/// Heuristic-to-perimeter resolution: try exact name match, then
/// substring match (e.g. `in_center` → `center`).
fn match_heuristic_to_perimeter(
    heuristic: &HeuristicMeta,
    perimeters: &[PerimeterSpec],
) -> Option<String> {
    let name = heuristic.name.to_lowercase();
    let mut candidates = perimeters.iter().collect::<Vec<_>>();
    candidates.sort_by_key(|p| std::cmp::Reverse(p.label().len()));
    for p in &candidates {
        let label = p.label().to_lowercase();
        if !label.is_empty() && name.ends_with(&format!("_{label}")) {
            return Some(p.label().to_string());
        }
    }
    // Exact match.
    for p in &candidates {
        if p.label().to_lowercase() == name {
            return Some(p.label().to_string());
        }
    }
    // Substring match: a perimeter label appears inside the heuristic name.
    for p in &candidates {
        let plabel = p.label().to_lowercase();
        if !plabel.is_empty() && name.contains(&plabel) {
            return Some(p.label().to_string());
        }
    }
    None
}

fn looks_spatial(heuristic_name: &str) -> bool {
    let lower = heuristic_name.to_lowercase();
    SPATIAL_HEURISTIC_HINTS
        .iter()
        .any(|hint| lower.contains(hint))
}

/// Emit task-specific `FeatureInspection` variants from the detailed result
/// blob. Returns an empty vec when the task type has no detailed figure or
/// the detailed result is missing / unparseable.
pub fn collect_task_detailed_inspections(
    task_type: TaskType,
    detailed: Option<&TaskDetail>,
    primary_label: Option<&Label>,
) -> Vec<FeatureInspection> {
    let Some(detail) = detailed else {
        return vec![];
    };

    match (task_type, detail) {
        (TaskType::ObjectRecognition, TaskDetail::ObjectRecognition(detail)) => {
            let mut features = emit_nort_discrimination(detail, primary_label);
            features.extend(emit_nort_center_periphery(detail, primary_label));
            features
        }
        (TaskType::YMaze, TaskDetail::YMaze(detail)) => emit_arm_sequence(detail, primary_label),
        (TaskType::OpenField, TaskDetail::OpenField(detail)) => {
            emit_center_periphery(detail, primary_label)
        }
        (TaskType::Cheeseboard, TaskDetail::Cheeseboard(_)) => vec![],
        _ => {
            tracing::warn!(
                expected = %task_type,
                actual = %detail.task_type(),
                "task-detailed inspection type mismatch; skipping detailed feature emission"
            );
            vec![]
        }
    }
}

/// The scored cross-line per arm from the trial's scoring provenance; empty
/// unless it was scored with the arm-start cross-line geometry.
fn crossline_threshold_lines(detailed: Option<&TaskDetail>) -> Vec<ArmThresholdLine> {
    let Some(TaskDetail::YMaze(detail)) = detailed else {
        return vec![];
    };
    let Some(prov) = detail.scoring_provenance.as_ref() else {
        return vec![];
    };
    if prov["geometry_version"] != "imen_arm_start_crossline_v1" {
        return vec![];
    }
    let point = |v: &serde_json::Value| Some((v[0].as_f64()?, v[1].as_f64()?));
    let lines: Option<Vec<_>> = prov["crossline_endpoints"]
        .as_array()
        .into_iter()
        .flatten()
        .zip(["A", "B", "C"])
        .map(|(e, arm)| {
            Some(ArmThresholdLine {
                arm: arm.into(),
                p1: point(&e["p1"])?,
                p2: point(&e["p2"])?,
            })
        })
        .collect();
    lines.unwrap_or_default()
}

/// Emit the Y-maze arm-sectioning overlay when a radial-maze perimeter is
/// present. Thresholds are fractions of arm depth from the maze center.
pub fn collect_ymaze_arm_section_inspections(
    task_type: TaskType,
    perimeters: &[PerimeterSpec],
    primary_label: Option<&Label>,
    proximal_threshold: Option<f64>,
    detailed: Option<&TaskDetail>,
) -> Vec<FeatureInspection> {
    if task_type != TaskType::YMaze {
        return vec![];
    }

    let Some(perimeter_label) = perimeters.iter().find_map(|perimeter| match perimeter {
        PerimeterSpec::RadialMaze { label, .. } => Some(label.clone()),
        _ => None,
    }) else {
        return vec![];
    };

    let coordinate_label = primary_label.cloned().unwrap_or_else(|| "nose".to_string());

    vec![FeatureInspection::YMazeArmSections {
        label: "ymaze_arm_sections".into(),
        perimeter_label,
        proximal_threshold: proximal_threshold.unwrap_or(DEFAULT_PROXIMAL_THRESHOLD),
        threshold_lines: crossline_threshold_lines(detailed),
        coordinate_labels: vec![coordinate_label],
        description: "Y-maze arm section thresholds".into(),
        representative_frame_index: None,
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    }]
}

/// Emit the NORT detection-field preview (nose cone, contact circle and, when
/// the whisker fan is enabled, both fans) from the effective scoring
/// provenance and the per-object `exploring_<obj>` decompositions. Driven only
/// by the configured nose/head coordinates, so it needs no task-specific
/// geometry; the head label falls back to `head_label` when the fan is off.
pub fn collect_detection_field_inspections(
    task_type: TaskType,
    detailed: Option<&TaskDetail>,
    decompositions: &[HeuristicDecompositionSpec],
    head_label: &str,
) -> Vec<FeatureInspection> {
    let (TaskType::ObjectRecognition, Some(TaskDetail::ObjectRecognition(detail))) =
        (task_type, detailed)
    else {
        return vec![];
    };
    let Some(prov) = detail.scoring_provenance.as_ref() else {
        return vec![];
    };
    let (Some(reach), Some(contact), Some(angle)) = (
        prov["zone_extension_m"].as_f64(),
        prov["contact_distance_m"].as_f64(),
        prov["angle_max_degrees"].as_f64(),
    ) else {
        return vec![];
    };
    let mut nose_label = None;
    let mut head = None;
    let (mut objects, mut near, mut gate, mut whisker) = (vec![], vec![], vec![], vec![]);
    for spec in decompositions {
        let Some(object) = spec.heuristic_name.strip_prefix("exploring_") else {
            continue;
        };
        let column = |name: &str| {
            spec.components.iter().find(|c| c.label == name).map(|c| {
                (
                    c.result_column.clone(),
                    c.coordinate_labels.first().cloned(),
                )
            })
        };
        let (Some((near_col, nose)), Some((gate_col, _))) =
            (column("nose_near"), column("contact_or_angle"))
        else {
            continue;
        };
        nose_label = nose_label.or(nose);
        let whisker_hit = column("whisker_hit");
        head = head.or(whisker_hit.as_ref().and_then(|(_, label)| label.clone()));
        objects.push(object.to_string());
        near.push(near_col);
        gate.push(gate_col);
        whisker.extend(whisker_hit.map(|(col, _)| col));
    }
    let Some(nose_label) = nose_label else {
        return vec![];
    };
    let fan = &prov["whisker_fan"];
    let whisker_on = fan["enabled"].as_bool() == Some(true) && whisker.len() == objects.len();
    let fan_value = |key: &str| whisker_on.then(|| fan[key].as_f64()).flatten();
    vec![FeatureInspection::DetectionFields {
        label: "detection_fields".into(),
        nose_label,
        head_label: head.unwrap_or_else(|| head_label.to_string()),
        cone_half_angle_deg: angle,
        nose_reach_m: reach,
        contact_distance_m: contact,
        whisker_apex_fraction: fan_value("apex_fraction"),
        whisker_half_angle_deg: fan_value("half_angle_deg"),
        whisker_length_m: fan_value("length_m"),
        object_labels: objects,
        nose_near_columns: near,
        nose_gate_columns: gate,
        whisker_hit_columns: if whisker_on { whisker } else { vec![] },
        description: format!(
            "Detection fields: nose cone ±{angle}° / {reach} m, contact {contact} m{}",
            if whisker_on { ", whisker fans" } else { "" }
        ),
        representative_frame_index: None,
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    }]
}
