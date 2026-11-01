fn emit_nort_discrimination(
    detail: &NortResult,
    primary_label: Option<&Label>,
) -> Vec<FeatureInspection> {
    let object_labels = detail.object_labels.clone();
    let exploration_times = detail.exploration_times.clone();

    if object_labels.is_empty() || exploration_times.is_empty() {
        return vec![];
    }

    let discrimination_index = detail.indices.as_ref().and_then(|i| i.discrimination_index);
    let recognition_index = detail.indices.as_ref().and_then(|i| i.recognition_index);
    let protocol = Some(detail.protocol.to_string());
    let title_override = Some(match detail.protocol {
        bekiper_core::types::ObjectRecognitionProtocol::NortTraining => {
            "Object Discrimination: NORT Training".to_string()
        }
        bekiper_core::types::ObjectRecognitionProtocol::NortNovelty => {
            "Object Discrimination: NORT Novelty".to_string()
        }
        bekiper_core::types::ObjectRecognitionProtocol::OulTraining => {
            "Object Discrimination: OUL Training".to_string()
        }
        bekiper_core::types::ObjectRecognitionProtocol::OulUpdate => {
            "Object Discrimination: OUL Update".to_string()
        }
        bekiper_core::types::ObjectRecognitionProtocol::OulTest => {
            "Object Discrimination: OUL Test".to_string()
        }
    });

    let coordinate_label = primary_label.cloned().unwrap_or_else(|| "nose".to_string());

    // Extract qualia decomposition if present.
    let qualia_channels = detail.qualia.as_ref().and_then(|q| {
        let specs: Vec<QualiaChannelSpec> = q
            .channels
            .iter()
            .map(|channel| QualiaChannelSpec {
                heuristic_type: channel.heuristic_type.clone(),
                per_object_seconds: channel.per_object_seconds.clone(),
                per_object_bouts: channel.per_object_bouts.clone(),
                relative_bias: q
                    .relative_bias
                    .get(&channel.heuristic_type)
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect();

        if specs.is_empty() { None } else { Some(specs) }
    });

    // Extract alternation count and merged channel results.
    let alternation_count = detail.qualia.as_ref().map(|q| q.alternation_count);

    let merged_channels = detail.qualia.as_ref().and_then(|q| {
        if q.merged.is_empty() {
            None
        } else {
            Some(q.merged.clone())
        }
    });

    vec![FeatureInspection::NortDiscrimination {
        label: "object_discrimination".into(),
        protocol,
        object_labels,
        exploration_times,
        discrimination_index,
        recognition_index,
        qualia_channels,
        alternation_count,
        merged_channels,
        coordinate_labels: vec![coordinate_label],
        description: format!("Object discrimination ({})", detail.protocol),
        representative_frame_index: None,
        color: None,
        linewidth: None,
        title_override,
        story_key: None,
    }]
}

fn emit_arm_sequence(
    detail: &YMazeResult,
    primary_label: Option<&Label>,
) -> Vec<FeatureInspection> {
    let arm_sequence = detail.analysis.arm_sequence.clone();

    if arm_sequence.is_empty() {
        return vec![];
    }

    let alternation_pct = detail.analysis.spontaneous_alternation_pct;

    let coordinate_label = primary_label.cloned().unwrap_or_else(|| "nose".to_string());

    vec![FeatureInspection::ArmSequence {
        label: "arm_sequence".into(),
        arm_sequence,
        coordinate_labels: vec![coordinate_label],
        alternation_pct,
        description: "Arm entry sequence".into(),
        representative_frame_index: None,
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    }]
}

fn emit_center_periphery(
    detail: &OftResult,
    primary_label: Option<&Label>,
) -> Vec<FeatureInspection> {
    let center_time_pct = detail.center_time_pct;
    let periphery_time_pct = detail.periphery_time_pct;

    // The native OFT result names its evaluated mask `in_center`; the
    // corresponding canonical geometry is named `center_zone`.
    let center_label = if detail.center_summary.zone_name == "in_center" {
        "center_zone".into()
    } else {
        detail.center_summary.zone_name.clone()
    };
    center_periphery_feature(
        center_label,
        center_time_pct,
        periphery_time_pct,
        primary_label,
    )
}

fn emit_nort_center_periphery(
    detail: &NortResult,
    primary_label: Option<&Label>,
) -> Vec<FeatureInspection> {
    let Some(zones) = &detail.zone_metrics else {
        return vec![];
    };
    let (Some(center), Some(periphery)) = (&zones.center_summary, &zones.periphery_summary) else {
        return vec![];
    };
    if detail.total_duration_seconds <= 0.0 {
        return vec![];
    }
    center_periphery_feature(
        "center_zone".into(),
        100.0 * center.duration_seconds / detail.total_duration_seconds,
        100.0 * periphery.duration_seconds / detail.total_duration_seconds,
        primary_label,
    )
}

fn center_periphery_feature(
    center_label: String,
    center_time_pct: f64,
    periphery_time_pct: f64,
    primary_label: Option<&Label>,
) -> Vec<FeatureInspection> {
    let arena_label = "arena".to_string();

    let coordinate_label = primary_label.cloned().unwrap_or_else(|| "nose".to_string());

    vec![FeatureInspection::CenterPeriphery {
        label: "center_periphery".into(),
        center_perimeter_label: center_label,
        arena_perimeter_label: arena_label,
        coordinate_labels: vec![coordinate_label],
        center_time_pct,
        periphery_time_pct,
        description: "Center vs. periphery occupancy".into(),
        representative_frame_index: None,
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    }]
}

// ---------------------------------------------------------------------------
// Post-processing: populate representative frame indices from DataFrame
// ---------------------------------------------------------------------------

/// Find contiguous True spans in a boolean Series, returning (start, end)
/// pairs where `end` is exclusive (matching Python's convention).
///
/// This is the Rust equivalent of Python's `boolean_index_truth_sequences()`
/// in `bekiper_inspect.plot.generic`.
pub fn truth_spans(series: &polars::prelude::Series) -> Vec<(u64, u64)> {
    let Some(ca) = series.bool().ok() else {
        return vec![];
    };
    let mut spans = Vec::new();
    let mut in_run = false;
    let mut start: usize = 0;

    for (i, opt_val) in ca.into_iter().enumerate() {
        let val = opt_val.unwrap_or(false);
        if val && !in_run {
            start = i;
            in_run = true;
        } else if !val && in_run {
            spans.push((start as u64, i as u64));
            in_run = false;
        }
    }
    if in_run {
        spans.push((start as u64, ca.len() as u64));
    }
    spans
}

/// Find the midpoint of the longest contiguous `true` bout in a boolean
/// Series. Returns `None` if the series is empty or contains no `true` values.
///
/// This is the Rust equivalent of Python's `find_representative_frame()` in
/// `bekiper_inspect.manifest._features`.
pub fn longest_true_bout_midpoint(series: &polars::prelude::Series) -> Option<u64> {
    let ca = series.bool().ok()?;
    let mut best_start: usize = 0;
    let mut best_len: usize = 0;
    let mut cur_start: usize = 0;
    let mut cur_len: usize = 0;
    let mut in_run = false;

    for (i, opt_val) in ca.into_iter().enumerate() {
        let val = opt_val.unwrap_or(false);
        if val {
            if !in_run {
                cur_start = i;
                cur_len = 0;
                in_run = true;
            }
            cur_len += 1;
            if cur_len > best_len {
                best_len = cur_len;
                best_start = cur_start;
            }
        } else {
            in_run = false;
        }
    }

    if best_len == 0 {
        return None;
    }
    Some(((best_start + best_start + best_len - 1) / 2) as u64)
}

/// Find the median index of non-null entries in a numeric Series.
/// Used to pick a representative frame from trajectory coordinate data.
pub fn median_non_null_index(series: &polars::prelude::Series) -> Option<u64> {
    let null_mask = series.is_null();
    let non_null_indices: Vec<usize> = (0..series.len())
        .filter(|&i| !null_mask.get(i).unwrap_or(true))
        .collect();
    if non_null_indices.is_empty() {
        return None;
    }
    let mid = non_null_indices.len() / 2;
    Some(non_null_indices[mid] as u64)
}

/// Populate `representative_frame_index` for every feature that doesn't
/// already have one. This eliminates the Python-side fallback heuristic.
///
/// Strategy per variant:
/// - Variants with a `result_column` (Zone, Heuristic, Proximity): midpoint
///   of the longest `true` bout in that boolean column.
/// - Variants with `coordinate_labels` (Trajectory, Quadrant, Angle, etc.):
///   median non-null index of the first coordinate label's x-column.
/// - ScaleDerivation: already set to frame 0 by the emitter.
/// - Fallback: frame 0.
pub fn populate_representative_frames(
    features: &mut [FeatureInspection],
    df: &polars::prelude::DataFrame,
    coordinate_columns: &HashMap<Label, CoordinateColumnSpec>,
    total_frames: u64,
) {
    let clamp = |idx: u64| -> u64 {
        if total_frames > 0 {
            idx.min(total_frames - 1)
        } else {
            idx
        }
    };

    for feature in features.iter_mut() {
        // Skip if already set.
        if feature.representative_frame_index().is_some() {
            continue;
        }
        // The Python picker prefers whisker-only, then nose hit frames.
        if matches!(feature, FeatureInspection::DetectionFields { .. }) {
            continue;
        }

        // Try 1: boolean result column → longest true bout midpoint.
        if let Some(rc) = feature.result_column()
            && let Ok(col) = df.column(rc)
            && let Some(idx) = longest_true_bout_midpoint(col.as_materialized_series())
        {
            feature.set_representative_frame_index(clamp(idx));
            continue;
        }

        // Try 2: coordinate labels → median non-null index.
        let first_coord_label = feature
            .coordinate_labels()
            .and_then(|labels| labels.first())
            .cloned();
        // Also handle Angle's from_coordinate_label and RayCast's origin_label.
        let coord_label = first_coord_label.or_else(|| match feature {
            FeatureInspection::Angle {
                from_coordinate_label,
                ..
            } => Some(from_coordinate_label.clone()),
            FeatureInspection::RayCast { origin_label, .. } => Some(origin_label.clone()),
            _ => None,
        });

        if let Some(label) = coord_label
            && let Some(spec) = coordinate_columns.get(&label)
            && let Ok(col) = df.column(&spec.x)
            && let Some(idx) = median_non_null_index(col.as_materialized_series())
        {
            feature.set_representative_frame_index(clamp(idx));
            continue;
        }

        // Fallback: frame 0.
        feature.set_representative_frame_index(0);
    }
}

// ---------------------------------------------------------------------------
// Post-processing: populate quadrant occupancy grids
// ---------------------------------------------------------------------------

/// Compute the occupancy grid for a quadrant: for each cell in the NxN grid,
/// compute the fraction of frames where the coordinate falls inside that cell.
/// Normalise by the max cell value so the output is 0.0–1.0.
fn compute_quadrant_occupancy(
    spec: &QuadrantSpec,
    x_values: &[f64],
    y_values: &[f64],
) -> Vec<Vec<f64>> {
    let rows = spec.rows as usize;
    let cols = spec.cols as usize;
    let cell_w = (spec.x_max - spec.x_min) / cols as f64;
    let cell_h = (spec.y_max - spec.y_min) / rows as f64;

    let mut grid = vec![vec![0.0_f64; cols]; rows];
    let n = x_values.len().min(y_values.len());

    for i in 0..n {
        let x = x_values[i];
        let y = y_values[i];
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let c = ((x - spec.x_min) / cell_w).floor() as isize;
        let r = ((y - spec.y_min) / cell_h).floor() as isize;
        if c >= 0 && c < cols as isize && r >= 0 && r < rows as isize {
            grid[r as usize][c as usize] += 1.0;
        }
    }

    // Normalize: divide by count to get fraction, then by max.
    if n > 0 {
        for row in &mut grid {
            for val in row.iter_mut() {
                *val /= n as f64;
            }
        }
    }
    let max_val = grid
        .iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(0.0_f64, f64::max);
    if max_val > 0.0 {
        for row in &mut grid {
            for val in row.iter_mut() {
                *val /= max_val;
            }
        }
    }

    grid
}

/// Populate `occupancy_grid` for every Quadrant feature from the evaluated
/// DataFrame. Skips features that already have a grid.
pub fn populate_occupancy_grids(
    features: &mut [FeatureInspection],
    df: &polars::prelude::DataFrame,
    coordinate_columns: &HashMap<Label, CoordinateColumnSpec>,
    quadrants: &[QuadrantSpec],
) {
    for feature in features.iter_mut() {
        if let FeatureInspection::Quadrant {
            quadrant_label,
            coordinate_labels,
            occupancy_grid,
            ..
        } = feature
        {
            if occupancy_grid.is_some() {
                continue;
            }
            let Some(spec) = quadrants.iter().find(|q| &q.label == quadrant_label) else {
                continue;
            };
            let coord_label = coordinate_labels.first();
            let Some(label) = coord_label else {
                continue;
            };
            let Some(col_spec) = coordinate_columns.get(label.as_str()) else {
                continue;
            };
            let (Ok(x_col), Ok(y_col)) = (df.column(&col_spec.x), df.column(&col_spec.y)) else {
                continue;
            };
            let x_series = x_col.as_materialized_series();
            let y_series = y_col.as_materialized_series();
            let Ok(x_ca) = x_series.f64() else { continue };
            let Ok(y_ca) = y_series.f64() else { continue };

            let x_vals: Vec<f64> = x_ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
            let y_vals: Vec<f64> = y_ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();

            let grid = compute_quadrant_occupancy(spec, &x_vals, &y_vals);
            *occupancy_grid = Some(grid);
        }
    }
}

/// Populate `truth_spans` on each `HeuristicMeta` from the evaluated
/// DataFrame's boolean columns. Skips heuristics whose `result_column`
/// is not found in the DataFrame.
pub fn populate_truth_spans(heuristics: &mut [HeuristicMeta], df: &polars::prelude::DataFrame) {
    for h in heuristics.iter_mut() {
        if !h.truth_spans.is_empty() {
            continue;
        }
        if let Ok(col) = df.column(&h.result_column) {
            h.truth_spans = truth_spans(col.as_materialized_series());
        }
    }
}

// ---------------------------------------------------------------------------
// Post-processing: populate story grouping keys
// ---------------------------------------------------------------------------

use std::collections::HashSet;
