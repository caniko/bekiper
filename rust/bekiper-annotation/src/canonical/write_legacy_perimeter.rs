fn write_legacy_perimeter(
    plugin_dir: &Path,
    perimeter_id: &str,
    perimeters: &[PerimeterSpec],
) -> Result<()> {
    if perimeters
        .iter()
        .all(|spec| matches!(spec, PerimeterSpec::Rectangle { .. }))
    {
        write_rectangle_csv(
            &plugin_dir.join(format!("perimeter-rectangle-{perimeter_id}.csv")),
            perimeters,
        )
    } else if perimeters
        .iter()
        .all(|spec| matches!(spec, PerimeterSpec::Circle { .. }))
    {
        write_circle_csv(
            &plugin_dir.join(format!("perimeter-circle-{perimeter_id}.csv")),
            perimeters,
        )
    } else {
        write_coco_json(
            &plugin_dir.join(format!("perimeter-rectangle-{perimeter_id}.json")),
            perimeters,
        )
    }
}

fn write_reference_csv(path: &Path, points: &[ReferencePointDocument]) -> Result<()> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to write {}: {e}", path.display())))?;
    for point in points {
        writer
            .write_record([
                "reference".to_string(),
                format_number(point.x),
                format_number(point.y),
                point.image_name.clone(),
            ])
            .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
    }
    writer
        .flush()
        .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))
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
        PerimeterSpec::Polygon { label, vertices } => bounds(label, vertices),
        PerimeterSpec::Triangle { label, vertices } => bounds(label, vertices),
        PerimeterSpec::RadialMaze {
            label,
            center_vertices,
            arms,
        } => {
            let points = center_vertices
                .iter()
                .copied()
                .chain(arms.iter().flat_map(|arm| arm.iter().copied()))
                .collect::<Vec<_>>();
            bounds(label, &points)
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
        PerimeterSpec::Polygon { vertices, .. } => vertices.clone(),
        PerimeterSpec::Triangle { vertices, .. } => vertices.to_vec(),
        _ => vec![
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ],
    }
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

fn bounds(label: &str, vertices: &[(f64, f64)]) -> (String, f64, f64, f64, f64) {
    let min_x = vertices
        .iter()
        .map(|(x, _)| *x)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let min_y = vertices
        .iter()
        .map(|(_, y)| *y)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let max_x = vertices
        .iter()
        .map(|(x, _)| *x)
        .reduce(f64::max)
        .unwrap_or(min_x);
    let max_y = vertices
        .iter()
        .map(|(_, y)| *y)
        .reduce(f64::max)
        .unwrap_or(min_y);
    (
        label.to_string(),
        min_x,
        min_y,
        max_x - min_x,
        max_y - min_y,
    )
}

fn format_number(value: f64) -> String {
    if (value.fract()).abs() < 1e-9 {
        format!("{value:.0}")
    } else {
        let mut s = format!("{value:.6}");
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        s
    }
}

#[cfg(test)]
fn test_individual_target_id(source_id: &str, target_image: &str) -> String {
    individual_target_perimeter_id(
        source_id,
        target_image,
        "metadata_row_2_animal_1_stage_1_part_2",
    )
}

/// Canonical ID for a session-local correction, also used by correction classification.
pub fn individual_target_perimeter_id(
    source_id: &str,
    target_image: &str,
    session_key: &str,
) -> String {
    format!(
        "{}__target__{}__session__{}",
        sanitize_id(source_id),
        sanitize_id(strip_png_extension(target_image)),
        sanitize_id(session_key)
    )
}

pub(crate) fn group_target_perimeter_id(source_id: &str, target_image: &str) -> String {
    format!(
        "{}__group_target__{}",
        sanitize_id(source_id),
        sanitize_id(strip_png_extension(target_image))
    )
}

fn sanitize_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn strip_png_extension(image_name: &str) -> &str {
    image_name.strip_suffix(".png").unwrap_or(image_name)
}

fn translate_perimeters(specs: &mut [PerimeterSpec], dx: f64, dy: f64) {
    for spec in specs {
        match spec {
            PerimeterSpec::Rectangle {
                center_x, center_y, ..
            }
            | PerimeterSpec::Circle {
                center_x, center_y, ..
            } => {
                *center_x += dx;
                *center_y += dy;
            }
            PerimeterSpec::Polygon { vertices, .. } => {
                for (x, y) in vertices {
                    *x += dx;
                    *y += dy;
                }
            }
            PerimeterSpec::Triangle { vertices, .. } => {
                for (x, y) in vertices {
                    *x += dx;
                    *y += dy;
                }
            }
            PerimeterSpec::RadialMaze {
                center_vertices,
                arms,
                ..
            } => {
                for (x, y) in center_vertices {
                    *x += dx;
                    *y += dy;
                }
                for arm in arms {
                    for (x, y) in arm {
                        *x += dx;
                        *y += dy;
                    }
                }
            }
        }
    }
}

fn sorted_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    Ok(paths)
}

fn write_toml<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let content = toml::to_string_pretty(value)
        .map_err(|e| BekiperError::Other(format!("failed to serialize {}: {e}", path.display())))?;
    atomic_write(path, content.as_bytes())
}

fn atomic_write(path: &Path, content: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(content)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn write_conflict_report(
    annotation_dir: &Path,
    legacy_changed: bool,
    canonical_changed: bool,
) -> Result<PathBuf> {
    #[derive(Serialize)]
    struct ConflictReport {
        legacy_changed: bool,
        canonical_changed: bool,
        written_unix_s: u64,
    }

    let root = CanonicalAnnotationStore::path_for(annotation_dir).join("conflicts");
    fs::create_dir_all(&root)?;
    let path = root.join(format!("{}.toml", now_unix_s()));
    write_toml(
        &path,
        &ConflictReport {
            legacy_changed,
            canonical_changed,
            written_unix_s: now_unix_s(),
        },
    )?;
    Ok(path)
}

fn now_unix_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
