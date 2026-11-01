/// Parse a legacy circle-point plugin CSV file.
///
/// Each row defines a point annotation in Makesense format:
/// `label, center_x, center_y, image_name, image_w, image_h`.
///
/// The legacy Rust manifest model does not have a separate point-shaped
/// perimeter variant, so the point is preserved as a zero-radius circle.
pub fn parse_circle_point_csv(path: &Path) -> Result<Vec<PerimeterSpec>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

    let mut circles = Vec::new();

    for record in reader.records() {
        let record = record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
        if record.len() < CIRCLE_POINT_CSV_MIN_FIELDS {
            return Err(BekiperError::Data(format!(
                "{}: expected at least {CIRCLE_POINT_CSV_MIN_FIELDS} fields, got {}",
                path.display(),
                record.len()
            )));
        }

        if record[1].trim().parse::<f64>().is_err() {
            continue;
        }

        let label = record[0].to_string();
        let center_x: f64 = record[1]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: center_x: {e}", path.display())))?;
        let center_y: f64 = record[2]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: center_y: {e}", path.display())))?;

        circles.push(PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius: 0.0,
        });
    }

    if circles.is_empty() {
        return Err(BekiperError::Data(format!(
            "{}: no circle-point data rows found",
            path.display()
        )));
    }

    Ok(circles)
}

// ── Center point parser ────────────────────────────────────────────────────

/// Arena center point loaded from a `center-{id}.csv` plugin file.
#[derive(Debug, Clone)]
pub struct CenterPoint {
    pub x: f64,
    pub y: f64,
}

/// Parse an arena center CSV file.
///
/// Format: single row, no header: `"center", x, y, image_name, image_w, image_h`
pub fn parse_center_csv(path: &Path) -> Result<CenterPoint> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

    let record = reader
        .records()
        .next()
        .ok_or_else(|| BekiperError::Data(format!("{}: empty file", path.display())))?
        .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;

    if record.len() < 3 {
        return Err(BekiperError::Data(format!(
            "{}: expected at least 3 fields, got {}",
            path.display(),
            record.len()
        )));
    }

    let x: f64 = record[1]
        .parse()
        .map_err(|e| BekiperError::Data(format!("{}: x: {e}", path.display())))?;
    let y: f64 = record[2]
        .parse()
        .map_err(|e| BekiperError::Data(format!("{}: y: {e}", path.display())))?;

    Ok(CenterPoint { x, y })
}

// ── Scale calibration parser ────────────────────────────────────────────────

/// Scale calibration from a `meters_per_pixel-{method}-{value}.csv` plugin file.
///
/// The physical measurement (meters) is encoded in the filename.
/// The CSV contains the pixel-space measurement endpoints.
#[derive(Debug, Clone)]
pub struct ScaleCalibration {
    /// Derivation method (e.g. "diagonal").
    pub method: String,
    /// Physical measurement in meters (from filename).
    pub physical_meters: f64,
    /// Two endpoints defining the pixel measurement.
    pub p1: (f64, f64),
    pub p2: (f64, f64),
    /// Computed meters-per-pixel value.
    pub meters_per_pixel: MetersPerPixel,
}

/// Search the plugin directory for `meters_per_pixel-*.csv` files and parse them.
///
/// Filename format: `meters_per_pixel-{method}-{value}.csv`
/// CSV format (no header): `mpp,x1,y1,x2,y2,image_name,image_w,image_h`
pub fn load_scale_calibrations(plugin_dir: &Path) -> Result<Vec<ScaleCalibration>> {
    let mut calibrations = Vec::new();

    let entries = std::fs::read_dir(plugin_dir)
        .map_err(|e| BekiperError::Data(format!("{}: {e}", plugin_dir.display())))?;

    for entry in entries {
        let entry =
            entry.map_err(|e| BekiperError::Data(format!("{}: {e}", plugin_dir.display())))?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if !name.starts_with("meters_per_pixel-") || !name.ends_with(".csv") {
            continue;
        }

        // Parse filename: meters_per_pixel-{method}-{value}.csv
        let stem = name.trim_end_matches(".csv");
        let parts: Vec<&str> = stem.splitn(SCALE_FILENAME_PARTS, '-').collect();
        if parts.len() < SCALE_FILENAME_PARTS {
            tracing::warn!(file = %name, "unexpected meters_per_pixel filename format");
            continue;
        }
        let method = parts[1].to_string();
        let physical_meters: f64 = match parts[2].parse() {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(file = %name, error = %e, "could not parse physical meters from filename");
                continue;
            }
        };

        // Parse CSV: mpp,x1,y1,x2,y2,image_name,image_w,image_h
        let path = entry.path();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(&path)
            .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;

        if let Some(record) = reader.records().next() {
            let record =
                record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
            if record.len() < SCALE_CSV_MIN_FIELDS {
                continue;
            }
            let x1: f64 = record[1]
                .parse()
                .map_err(|e| BekiperError::Data(format!("{}: x1: {e}", path.display())))?;
            let y1: f64 = record[2]
                .parse()
                .map_err(|e| BekiperError::Data(format!("{}: y1: {e}", path.display())))?;
            let x2: f64 = record[3]
                .parse()
                .map_err(|e| BekiperError::Data(format!("{}: x2: {e}", path.display())))?;
            let y2: f64 = record[4]
                .parse()
                .map_err(|e| BekiperError::Data(format!("{}: y2: {e}", path.display())))?;

            let pixel_distance = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
            if pixel_distance <= 0.0 {
                tracing::warn!(file = %name, "scale calibration has zero pixel distance");
                continue;
            }

            let mpp = physical_meters / pixel_distance;
            calibrations.push(ScaleCalibration {
                method,
                physical_meters,
                p1: (x1, y1),
                p2: (x2, y2),
                meters_per_pixel: MetersPerPixel(mpp),
            });
        }
    }

    Ok(calibrations)
}

// ── Reference point parsers ──────────────────────────────────────────────────

/// A reference point: a known physical location's pixel coordinates in an image.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferencePoint {
    pub x: f64,
    pub y: f64,
    pub image_name: String,
}

/// Parse a reference point CSV (single-row): `reference,x,y,image_name,w,h`
fn parse_reference_csv(path: &Path) -> Result<Vec<ReferencePoint>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

    let mut points = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
        if record.len() < REFERENCE_CSV_MIN_FIELDS {
            continue;
        }
        let x: f64 = record[1]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: x: {e}", path.display())))?;
        let y: f64 = record[2]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: y: {e}", path.display())))?;
        let image_name = record[3].to_string();
        points.push(ReferencePoint { x, y, image_name });
    }
    Ok(points)
}

/// Load the source reference point for a perimeter ID.
///
/// Reads `reference-{id}.csv` from the plugin directory.
pub fn load_source_reference(plugin_dir: &Path, perimeter_id: &str) -> Result<ReferencePoint> {
    let path = plugin_dir.join(format!("reference-{perimeter_id}.csv"));
    let points = parse_reference_csv(&path)?;
    points.into_iter().next().ok_or_else(|| {
        BekiperError::Data(format!(
            "empty reference file for '{perimeter_id}' in {}",
            plugin_dir.display()
        ))
    })
}

/// Load target reference points for a perimeter ID.
///
/// Reads `change_reference-{id}.csv` and returns a map: image_name_stem → ReferencePoint.
/// Image name stems are like "A1_training_1" (without .png extension).
fn load_change_references(
    plugin_dir: &Path,
    perimeter_id: &str,
) -> Result<HashMap<String, ReferencePoint>> {
    let path = plugin_dir.join(format!("change_reference-{perimeter_id}.csv"));
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let points = parse_reference_csv(&path)?;
    Ok(points
        .into_iter()
        .map(|p| {
            let stem = p
                .image_name
                .strip_suffix(".png")
                .unwrap_or(&p.image_name)
                .to_string();
            (stem, p)
        })
        .collect())
}

/// Apply a (dx, dy) translation to all perimeter specs.
fn translate_perimeters(specs: &mut [PerimeterSpec], dx: f64, dy: f64) {
    for spec in specs.iter_mut() {
        match spec {
            PerimeterSpec::Rectangle {
                center_x, center_y, ..
            } => {
                *center_x += dx;
                *center_y += dy;
            }
            PerimeterSpec::Polygon { vertices, .. } => {
                for (x, y) in vertices.iter_mut() {
                    *x += dx;
                    *y += dy;
                }
            }
            PerimeterSpec::Circle {
                center_x, center_y, ..
            } => {
                *center_x += dx;
                *center_y += dy;
            }
            PerimeterSpec::RadialMaze {
                center_vertices,
                arms,
                ..
            } => {
                for (x, y) in center_vertices.iter_mut() {
                    *x += dx;
                    *y += dy;
                }
                for arm in arms.iter_mut() {
                    for (x, y) in arm.iter_mut() {
                        *x += dx;
                        *y += dy;
                    }
                }
            }
            PerimeterSpec::Triangle { vertices, .. } => {
                for (x, y) in vertices.iter_mut() {
                    *x += dx;
                    *y += dy;
                }
            }
        }
    }
}

// ── Perimeter ID → plugin file resolution ───────────────────────────────────

/// Load perimeters from a plugin file identified by its perimeter ID.
///
/// Searches for `perimeter-rectangle-{id}.csv` or `.json` in the plugin_dir.
pub fn load_perimeter_by_id(plugin_dir: &Path, perimeter_id: &str) -> Result<Vec<PerimeterSpec>> {
    // Try CSV first, then JSON
    let csv_path = plugin_dir.join(format!("perimeter-rectangle-{perimeter_id}.csv"));
    if csv_path.exists() {
        return parse_rectangle_csv(&csv_path);
    }

    let json_path = plugin_dir.join(format!("perimeter-rectangle-{perimeter_id}.json"));
    if json_path.exists() {
        return parse_coco_json(&json_path);
    }

    // Try legacy circle-point CSV before the line-based circle format.
    for name in [
        format!("perimeter-circle_point-{perimeter_id}.csv"),
        format!("perimeter-circle-point-{perimeter_id}.csv"),
    ] {
        let path = plugin_dir.join(name);
        if path.exists() {
            return parse_circle_point_csv(&path);
        }
    }

    // Try circle CSV
    let circle_path = plugin_dir.join(format!("perimeter-circle-{perimeter_id}.csv"));
    if circle_path.exists() {
        let mut specs = parse_circle_csv(&circle_path)?;
        if specs.len() == 1 {
            match &mut specs[0] {
                PerimeterSpec::Circle { label, .. } => *label = perimeter_id.to_string(),
                PerimeterSpec::Rectangle { label, .. } => *label = perimeter_id.to_string(),
                PerimeterSpec::Polygon { label, .. } => *label = perimeter_id.to_string(),
                PerimeterSpec::RadialMaze { label, .. } => *label = perimeter_id.to_string(),
                PerimeterSpec::Triangle { label, .. } => *label = perimeter_id.to_string(),
            }
        }
        return Ok(specs);
    }

    // Try radial-maze directory
    let maze_dir = plugin_dir.join(format!("radial-maze_{perimeter_id}"));
    if maze_dir.is_dir() {
        return parse_radial_maze(&maze_dir);
    }

    Err(BekiperError::Data(format!(
        "no perimeter plugin file found for ID '{perimeter_id}' in {}",
        plugin_dir.display()
    )))
}

// ── Video context extraction ────────────────────────────────────────────────

/// Contextual information extracted from a video file path.
///
/// Used to look up the correct perimeter in the metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VideoContext {
    /// Animal / video number (from `video-N` in filename).
    pub animal: u32,
    /// Experiment stage: 1 = T1 (training), 2 = T2 (novelty).
    pub stage: u32,
    /// Phase part: 1 = pre(-treatment), 2 = post(-treatment).
    pub phase_part: u32,
}
