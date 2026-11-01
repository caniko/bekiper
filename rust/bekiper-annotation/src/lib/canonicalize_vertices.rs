use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::PerimeterSpec;
use bekiper_core::types::MetersPerPixel;
use bekiper_math::geometry::canonicalize_polygon_vertices as canonicalize_vertices;

pub use editable::*;

type Point = (f64, f64);
type Line = (Point, Point);

// ── Format contract constants ───────────────────────────────────────────────
// These are not user-tunable — they describe the *external* file formats
// this module parses. Named for readability and to make the "why 6?" /
// "why 3?" in contract checks self-documenting.

/// Minimum coordinate count in a COCO segmentation polygon: 3 vertices × 2
/// (x, y) = 6 flat values. Anything fewer cannot form a triangle.
const COCO_MIN_POLYGON_COORDS: usize = 6;

/// COCO bounding-box length: `[x, y, width, height]`.
const COCO_BBOX_LEN: usize = 4;

/// Number of `-` separated parts in a `meters_per_pixel-{method}-{value}.csv`
/// filename: `"meters_per_pixel"`, method, numeric value.
const SCALE_FILENAME_PARTS: usize = 3;

/// Minimum field count in a rectangle-perimeter plugin CSV row:
/// `label, bbox_x, bbox_y, bbox_w, bbox_h, …`.
const RECTANGLE_CSV_MIN_FIELDS: usize = 5;

/// Minimum field count in a circle-perimeter plugin CSV row:
/// `label, center_x, center_y, radius, image, …`.
const CIRCLE_CSV_MIN_FIELDS: usize = 5;

/// Minimum field count in a legacy circle-point plugin CSV row:
/// `label, center_x, center_y, image, …`.
const CIRCLE_POINT_CSV_MIN_FIELDS: usize = 4;

/// Minimum field count in an arm-lines plugin CSV row:
/// `label, x1, y1, x2, y2, …`.
const ARM_LINES_CSV_MIN_FIELDS: usize = 5;

/// Minimum field count in a meters-per-pixel calibration CSV row:
/// `mpp, x1, y1, x2, y2, …`.
const SCALE_CSV_MIN_FIELDS: usize = 5;

/// Minimum field count in a reference-point CSV row:
/// `reference, x, y, image_name, …`.
const REFERENCE_CSV_MIN_FIELDS: usize = 4;

// ── Plugin file parsers ─────────────────────────────────────────────────────

/// Parse a rectangle-perimeter plugin CSV file.
///
/// Handles both header and headerless formats:
/// - With header: `label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,...`
/// - Without:     `label,bbox_x,bbox_y,bbox_w,bbox_h,image_name,...`
///
/// Returns `PerimeterSpec::Rectangle` entries with center-based coordinates.
pub fn parse_rectangle_csv(path: &Path) -> Result<Vec<PerimeterSpec>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

    let mut perimeters = Vec::new();

    for record in reader.records() {
        let record = record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
        if record.len() < RECTANGLE_CSV_MIN_FIELDS {
            return Err(BekiperError::Data(format!(
                "{}: expected at least {RECTANGLE_CSV_MIN_FIELDS} fields, got {}",
                path.display(),
                record.len()
            )));
        }

        // Skip header row: if field[1] isn't a number, it's a column name
        if record[1].trim().parse::<f64>().is_err() {
            continue;
        }

        let label: String = record[0].to_string();
        let bbox_x: f64 = record[1]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: bbox_x: {e}", path.display())))?;
        let bbox_y: f64 = record[2]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: bbox_y: {e}", path.display())))?;
        let bbox_w: f64 = record[3]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: bbox_w: {e}", path.display())))?;
        let bbox_h: f64 = record[4]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: bbox_h: {e}", path.display())))?;

        perimeters.push(PerimeterSpec::Rectangle {
            label,
            center_x: bbox_x + bbox_w / 2.0,
            center_y: bbox_y + bbox_h / 2.0,
            width: bbox_w,
            height: bbox_h,
        });
    }

    Ok(perimeters)
}

/// Parse a COCO-format JSON annotation file.
///
/// Each annotation becomes either a `Polygon` (if segmentation has >4 vertices)
/// or a `Rectangle` (from bbox).
pub fn parse_coco_json(path: &Path) -> Result<Vec<PerimeterSpec>> {
    let content = std::fs::read_to_string(path)?;
    let coco: CocoAnnotation = serde_json::from_str(&content)
        .map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;

    // Build category_id → name lookup
    let cat_map: HashMap<u64, String> = coco
        .categories
        .iter()
        .map(|c| (c.id, c.name.clone()))
        .collect();

    let mut perimeters = Vec::new();

    for ann in &coco.annotations {
        let label = cat_map
            .get(&ann.category_id)
            .cloned()
            .unwrap_or_else(|| format!("category_{}", ann.category_id));

        // Prefer segmentation polygons if available
        if let Some(ref segs) = ann.segmentation
            && let Some(seg) = segs.first()
            && seg.len() >= COCO_MIN_POLYGON_COORDS
        {
            // Flat [x1,y1,x2,y2,...] → Vec<(f64,f64)>
            let (pairs, _) = seg.as_chunks::<2>();
            let vertices: Vec<(f64, f64)> = pairs.iter().map(|[x, y]| (*x, *y)).collect();
            perimeters.push(PerimeterSpec::Polygon {
                label,
                vertices: canonicalize_vertices(vertices),
            });
            continue;
        }

        // Fall back to bbox [x, y, w, h]
        if ann.bbox.len() == COCO_BBOX_LEN {
            perimeters.push(PerimeterSpec::Rectangle {
                label,
                center_x: ann.bbox[0] + ann.bbox[2] / 2.0,
                center_y: ann.bbox[1] + ann.bbox[3] / 2.0,
                width: ann.bbox[2],
                height: ann.bbox[3],
            });
        }
    }

    Ok(perimeters)
}
/// Parse Y-maze annotations: arm-lines CSV + center-triangle JSON.
///
/// The external annotation format stores each arm as a distal cross-line
/// (`a`, `b`, `c`) plus a center triangle. Analysis needs real arm areas, so
/// each distal line is paired with the best-matching center edge and expanded
/// into a parallelogram polygon.
pub fn parse_radial_maze(maze_dir: &Path) -> Result<Vec<PerimeterSpec>> {
    let arm_lines_path = maze_dir.join("arm-lines.csv");
    let center_path = maze_dir.join("center-triangle.json");

    if !arm_lines_path.exists() {
        return Err(BekiperError::Data(format!(
            "arm-lines.csv not found in {}",
            maze_dir.display()
        )));
    }

    // Parse arm lines: label,x1,y1,x2,y2,image,w,h (no header)
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&arm_lines_path)
        .map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;

    let mut arm_lines: Vec<Line> = Vec::new();

    for record in reader.records() {
        let record =
            record.map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;
        if record.len() < ARM_LINES_CSV_MIN_FIELDS {
            continue;
        }
        let x1: f64 = record[1]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;
        let y1: f64 = record[2]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;
        let x2: f64 = record[3]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;
        let y2: f64 = record[4]
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: {e}", arm_lines_path.display())))?;
        arm_lines.push(((x1, y1), (x2, y2)));
    }

    // Parse center triangle from COCO JSON
    let center_vertices = if center_path.exists() {
        let coco_specs = parse_coco_json(&center_path)?;
        match coco_specs.into_iter().next() {
            Some(PerimeterSpec::Polygon { vertices, .. }) => vertices,
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };

    let center_vertices = canonicalize_vertices(center_vertices);
    let arms = build_radial_maze_arm_polygons(&center_vertices, &arm_lines);

    Ok(vec![PerimeterSpec::RadialMaze {
        label: "maze".into(),
        center_vertices,
        arms,
    }])
}

fn build_radial_maze_arm_polygons(
    center_vertices: &[(f64, f64)],
    arm_lines: &[Line],
) -> Vec<Vec<(f64, f64)>> {
    if center_vertices.len() < 3 {
        return arm_lines.iter().map(|&(a, b)| vec![a, b]).collect();
    }

    let center_edges: Vec<Line> = (0..center_vertices.len())
        .map(|idx| {
            (
                center_vertices[idx],
                center_vertices[(idx + 1) % center_vertices.len()],
            )
        })
        .collect();
    let mut used_edges = vec![false; center_edges.len()];

    arm_lines
        .iter()
        .map(|&(d1, d2)| {
            let edge_idx = best_center_edge_for_arm_line(&center_edges, &used_edges, (d1, d2));
            used_edges[edge_idx] = true;
            let (c1, c2) = center_edges[edge_idx];
            let (d1, d2) = orient_distal_line_to_center_edge(c1, c2, d1, d2);
            canonicalize_vertices(vec![c1, c2, d2, d1])
        })
        .collect()
}

fn best_center_edge_for_arm_line(
    center_edges: &[Line],
    used_edges: &[bool],
    arm_line: Line,
) -> usize {
    center_edges
        .iter()
        .enumerate()
        .filter(|(idx, _)| !used_edges.get(*idx).copied().unwrap_or(false))
        .max_by(|(_, a), (_, b)| {
            line_orientation_similarity(**a, arm_line)
                .total_cmp(&line_orientation_similarity(**b, arm_line))
        })
        .map(|(idx, _)| idx)
        .or_else(|| {
            center_edges
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    line_orientation_similarity(**a, arm_line)
                        .total_cmp(&line_orientation_similarity(**b, arm_line))
                })
                .map(|(idx, _)| idx)
        })
        .unwrap_or(0)
}

fn line_orientation_similarity(a: Line, b: Line) -> f64 {
    let av = (a.1.0 - a.0.0, a.1.1 - a.0.1);
    let bv = (b.1.0 - b.0.0, b.1.1 - b.0.1);
    let amag = (av.0 * av.0 + av.1 * av.1).sqrt();
    let bmag = (bv.0 * bv.0 + bv.1 * bv.1).sqrt();
    if amag <= f64::EPSILON || bmag <= f64::EPSILON {
        return 0.0;
    }
    ((av.0 * bv.0 + av.1 * bv.1) / (amag * bmag)).abs()
}

fn orient_distal_line_to_center_edge(c1: Point, c2: Point, d1: Point, d2: Point) -> Line {
    let direct = squared_distance(c1, d1) + squared_distance(c2, d2);
    let swapped = squared_distance(c1, d2) + squared_distance(c2, d1);
    if swapped < direct { (d2, d1) } else { (d1, d2) }
}

fn squared_distance(a: Point, b: Point) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

// ── Circle perimeter parser ─────────────────────────────────────────────────

/// Parse a circle-perimeter plugin CSV file.
///
/// Each row defines one circle via a single diameter line:
/// `object_index, x1, y1, x2, y2, image_name, image_w, image_h`
///
/// The midpoint of (x1,y1)→(x2,y2) is the center, half the length is the radius.
/// Each row becomes a separate `PerimeterSpec::Circle` labeled `object_{index}`.
pub fn parse_circle_csv(path: &Path) -> Result<Vec<PerimeterSpec>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

    let mut circles = Vec::new();

    for record in reader.records() {
        let record = record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
        if record.len() < CIRCLE_CSV_MIN_FIELDS {
            return Err(BekiperError::Data(format!(
                "{}: expected at least {CIRCLE_CSV_MIN_FIELDS} fields, got {}",
                path.display(),
                record.len()
            )));
        }

        // Skip header row if present
        if record[1].trim().parse::<f64>().is_err() {
            continue;
        }

        let index: u32 = record[0]
            .trim()
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: index: {e}", path.display())))?;
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

        // MakeSense circle CSV: (x1, y1) is the center, (x2, y2) is a point
        // on the edge. Radius = distance from center to edge.
        let center_x = x1;
        let center_y = y1;
        let radius = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();

        circles.push(PerimeterSpec::Circle {
            label: format!("object_{index}"),
            center_x,
            center_y,
            radius,
        });
    }

    if circles.is_empty() {
        return Err(BekiperError::Data(format!(
            "{}: no circle data rows found",
            path.display()
        )));
    }

    Ok(circles)
}
