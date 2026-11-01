//! Audit alignment responsibilities.
use super::*;

pub(super) fn ymaze_alignment_diagnostics(
    experiment: &ExperimentConfig,
    augmented_df: Option<&DataFrame>,
    exp_dir: &ExperimentOutputDir,
    validation_command: &str,
) -> Result<Vec<AuditDiagnostic>> {
    if experiment.task_type != TaskType::YMaze {
        return Ok(Vec::new());
    }
    let Some(df) = augmented_df else {
        return Ok(Vec::new());
    };
    let perimeters_path = exp_dir.perimeters_toml();
    if !perimeters_path.exists() {
        return Ok(Vec::new());
    }

    let content = std::fs::read_to_string(&perimeters_path).map_err(|e| {
        BekiperError::Other(format!("failed to read {}: {e}", perimeters_path.display()))
    })?;
    let file: PerimetersFile = toml::from_str(&content)
        .map_err(|e| BekiperError::Config(format!("invalid {}: {e}", perimeters_path.display())))?;

    let Some((center_vertices, arms)) = file.perimeters.iter().find_map(|perimeter| {
        if let PerimeterSpec::RadialMaze {
            center_vertices,
            arms,
            ..
        } = perimeter
        {
            Some((center_vertices, arms))
        } else {
            None
        }
    }) else {
        return Ok(Vec::new());
    };

    let Some(coords) = coordinate_pairs(df, "center_ear_x", "center_ear_y") else {
        return Ok(vec![
            AuditDiagnostic::new(
                AuditSeverity::Warning,
                "y-maze alignment",
                "center_ear coordinate columns are missing; cannot check Y-maze alignment",
            )
            .experiment(&experiment.name)
            .artifact_path(exp_dir.augmented_parquet())
            .cause("Y-maze alignment diagnostics need center_ear_x and center_ear_y")
            .fix_hint("check task_params and pose label mapping for this dataset")
            .validation_command(validation_command),
        ]);
    };

    let center_count = polygon_overlap_count(center_vertices, &coords);
    let arm_counts = arms
        .iter()
        .take(3)
        .map(|arm| polygon_overlap_count(arm, &coords))
        .collect::<Vec<_>>();
    let zero_regions = std::iter::once(("center".to_string(), center_count))
        .chain(arm_counts.iter().enumerate().map(|(idx, count)| {
            (
                format!("arm_{}", ["A", "B", "C"].get(idx).copied().unwrap_or("?")),
                *count,
            )
        }))
        .filter_map(|(label, count)| (count == 0).then_some(label))
        .collect::<Vec<_>>();

    if zero_regions.len() < 2 {
        return Ok(Vec::new());
    }

    let mut details = vec![
        format!("center overlap frames={center_count}"),
        format!(
            "arm overlap frames A/B/C={}/{}/{}",
            arm_counts.first().copied().unwrap_or(0),
            arm_counts.get(1).copied().unwrap_or(0),
            arm_counts.get(2).copied().unwrap_or(0)
        ),
    ];
    if let Some(bbox) = coordinate_bbox(df, "center_ear_x", "center_ear_y") {
        details.push(format!(
            "center_ear bbox x={:.4}..{:.4}, y={:.4}..{:.4}",
            bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y
        ));
    }
    details.push(format!(
        "center bbox {}",
        vertex_bbox_string(center_vertices)
    ));
    for (idx, arm) in arms.iter().take(3).enumerate() {
        details.push(format!(
            "arm_{} bbox {}",
            ["A", "B", "C"].get(idx).copied().unwrap_or("?"),
            vertex_bbox_string(arm)
        ));
    }

    Ok(vec![
        AuditDiagnostic::new(
            AuditSeverity::Error,
            "y-maze alignment",
            format!(
                "Y-maze pose coordinates do not overlap expected maze regions: {}",
                zero_regions.join(", ")
            ),
        )
        .experiment(&experiment.name)
        .artifact_path(&perimeters_path)
        .cause(details.join("; "))
        .fix_hint("fix upstream DLC pose coordinates or Y-maze manual annotation reference mapping; do not synthesize arm occupancy values")
        .validation_command(validation_command),
    ])
}

pub(super) fn coordinate_pairs(
    df: &DataFrame,
    x_col: &str,
    y_col: &str,
) -> Option<Vec<(f64, f64)>> {
    let xs = df.column(x_col).ok()?.f64().ok()?;
    let ys = df.column(y_col).ok()?.f64().ok()?;
    Some(
        xs.into_iter()
            .zip(ys)
            .filter_map(|(x, y)| Some((x?, y?)))
            .collect(),
    )
}

pub(super) fn polygon_overlap_count(vertices: &[(f64, f64)], coords: &[(f64, f64)]) -> usize {
    let polygon = Polygon::new(vertices.to_vec());
    coords
        .iter()
        .filter(|(x, y)| polygon.contains(*x, *y))
        .count()
}
