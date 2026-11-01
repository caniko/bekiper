fn canonical_reference_to_editable(
    path: &Path,
    reference: Option<CanonicalReferenceDocument>,
    target_image: Option<&str>,
) -> (
    Option<ReferenceDocument>,
    Option<ReferenceDocument>,
    Option<usize>,
) {
    let Some(reference) = reference else {
        return (None, None, None);
    };

    let source_reference = reference.source.map(|point| ReferenceDocument {
        path: path.to_path_buf(),
        rows: vec![ReferenceRow {
            label: "reference".into(),
            x: point.x,
            y: point.y,
            image_name: point.image_name,
            extras: Vec::new(),
        }],
    });

    let target_rows = reference
        .targets
        .into_values()
        .map(|point| ReferenceRow {
            label: "reference".into(),
            x: point.x,
            y: point.y,
            image_name: point.image_name,
            extras: Vec::new(),
        })
        .collect::<Vec<_>>();
    let target_index = target_image.and_then(|target| {
        target_rows
            .iter()
            .position(|row| strip_png_extension(&row.image_name) == target)
    });
    let change_reference =
        (target_image.is_some() || !target_rows.is_empty()).then(|| ReferenceDocument {
            path: path.to_path_buf(),
            rows: target_rows,
        });

    (source_reference, change_reference, target_index)
}

fn editable_references_to_canonical(
    source: Option<&ReferenceDocument>,
    targets: Option<&ReferenceDocument>,
) -> CanonicalReferenceDocument {
    CanonicalReferenceDocument {
        source: source
            .and_then(ReferenceDocument::first_point)
            .map(|point| crate::canonical::ReferencePointDocument {
                x: point.x,
                y: point.y,
                image_name: point.image_name,
            }),
        targets: targets
            .map(|doc| {
                doc.rows
                    .iter()
                    .map(|row| {
                        (
                            strip_png_extension(&row.image_name).to_string(),
                            crate::canonical::ReferencePointDocument {
                                x: row.x,
                                y: row.y,
                                image_name: row.image_name.clone(),
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn canonical_group_summary(
    info: &AnnotationGroupInfo,
    session: &CanonicalSession,
) -> AnnotationGroupSummary {
    AnnotationGroupSummary {
        source_perimeter_id: info.key.source_perimeter_id.clone(),
        target_image: info.key.target_image.clone(),
        member_count: info.members.len(),
        member_labels: info
            .members
            .iter()
            .map(|member| member.key.clone())
            .collect(),
        correction_kind: session.correction_kind().into(),
    }
}

fn bounding_rect_from_spec(spec: &PerimeterSpec) -> (String, f64, f64, f64, f64) {
    match spec {
        PerimeterSpec::Rectangle {
            label,
            center_x,
            center_y,
            width,
            height,
        } => (
            label.clone(),
            center_x - width / 2.0,
            center_y - height / 2.0,
            *width,
            *height,
        ),
        PerimeterSpec::Polygon { label, vertices } => {
            let (x, y, width, height) = vertices_bounding_box(vertices);
            (label.clone(), x, y, width, height)
        }
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } => (
            label.clone(),
            center_x - radius,
            center_y - radius,
            radius * 2.0,
            radius * 2.0,
        ),
        PerimeterSpec::Triangle { label, vertices } => {
            let vertices: Vec<(f64, f64)> = vertices.to_vec();
            let (x, y, width, height) = vertices_bounding_box(&vertices);
            (label.clone(), x, y, width, height)
        }
        PerimeterSpec::RadialMaze { label, .. } => (label.clone(), 0.0, 0.0, 0.0, 0.0),
    }
}

fn circle_from_spec(spec: &PerimeterSpec) -> (String, f64, f64, f64) {
    match spec {
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } => (label.clone(), *center_x, *center_y, *radius),
        _ => {
            let (label, x, y, width, height) = bounding_rect_from_spec(spec);
            (
                label,
                x + width / 2.0,
                y + height / 2.0,
                width.max(height) / 2.0,
            )
        }
    }
}

fn segmentation_vertices_from_spec(
    spec: &PerimeterSpec,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> Vec<(f64, f64)> {
    match spec {
        PerimeterSpec::Polygon { vertices, .. } => canonicalize_vertices(vertices.clone()),
        PerimeterSpec::Triangle { vertices, .. } => canonicalize_vertices(vertices.to_vec()),
        _ => rectangle_vertices(x, y, width, height),
    }
}

fn rectangle_vertices(x: f64, y: f64, width: f64, height: f64) -> Vec<(f64, f64)> {
    vec![
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    ]
}

fn vertices_bounding_box(vertices: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let min_x = vertices
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::INFINITY, f64::min);
    let min_y = vertices
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min);
    let max_x = vertices
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_y = vertices
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    (
        min_x,
        min_y,
        (max_x - min_x).max(0.0),
        (max_y - min_y).max(0.0),
    )
}

fn polygon_area(vertices: &[(f64, f64)]) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for index in 0..vertices.len() {
        let (x1, y1) = vertices[index];
        let (x2, y2) = vertices[(index + 1) % vertices.len()];
        area += x1 * y2 - x2 * y1;
    }
    area / 2.0
}

/// Select the arm edge farthest from the arithmetic mean of centre vertices.
/// Empty arms have no edge; one/two-point arms return their endpoints. When
/// centre vertices are absent, the arm's own vertex mean is used. Equal
/// distances retain the existing last-maximum edge selection.
pub fn radial_arm_distal_edge(
    center_vertices: &[(f64, f64)],
    arm: &[(f64, f64)],
) -> Option<((f64, f64), (f64, f64))> {
    if arm.len() <= 2 {
        return Some((*arm.first()?, *arm.last()?));
    }

    let center = vertices_centroid(center_vertices).or_else(|| vertices_centroid(arm))?;
    (0..arm.len())
        .map(|idx| {
            let next_idx = (idx + 1) % arm.len();
            let midpoint = (
                (arm[idx].0 + arm[next_idx].0) / 2.0,
                (arm[idx].1 + arm[next_idx].1) / 2.0,
            );
            let dx = midpoint.0 - center.0;
            let dy = midpoint.1 - center.1;
            (idx, next_idx, dx * dx + dy * dy)
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .map(|(idx, next_idx, _)| (arm[idx], arm[next_idx]))
}

fn vertices_centroid(vertices: &[(f64, f64)]) -> Option<(f64, f64)> {
    if vertices.is_empty() {
        return None;
    }
    let (sx, sy) = vertices
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x, sy + y));
    let n = vertices.len() as f64;
    Some((sx / n, sy / n))
}

fn strip_png_extension(image_name: &str) -> &str {
    image_name.strip_suffix(PNG_EXTENSION).unwrap_or(image_name)
}

fn ensure_png_extension(image_stem: &str) -> String {
    if image_stem.ends_with(PNG_EXTENSION) {
        image_stem.to_string()
    } else {
        format!("{image_stem}{PNG_EXTENSION}")
    }
}

fn format_number(value: f64) -> String {
    let rounded = value.round();
    if (value - rounded).abs() < 1e-9 {
        format!("{rounded:.0}")
    } else {
        format!("{value:.6}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

fn sole_perimeter_id(plugin_dir: &Path) -> Result<Option<String>> {
    let mut ids = Vec::new();
    for entry in std::fs::read_dir(plugin_dir)
        .map_err(|e| BekiperError::Data(format!("{}: {e}", plugin_dir.display())))?
    {
        let entry =
            entry.map_err(|e| BekiperError::Data(format!("{}: {e}", plugin_dir.display())))?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(id) = name.strip_prefix("perimeter-rectangle-").and_then(|value| {
            value
                .strip_suffix(".csv")
                .or_else(|| value.strip_suffix(".json"))
        }) {
            ids.push(id.to_string());
        } else if path.is_dir()
            && let Some(id) = name.strip_prefix("radial-maze_")
        {
            ids.push(id.to_string());
        }
    }

    if ids.len() == 1 {
        Ok(ids.into_iter().next())
    } else {
        Ok(None)
    }
}
