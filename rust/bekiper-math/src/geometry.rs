use polars::prelude::*;

use crate::vector::dot_product_expr;

/// Remove a repeated closing vertex and normalize winding and starting vertex.
/// Degenerate inputs of at most two vertices retain their original order.
pub fn canonicalize_polygon_vertices(mut vertices: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    if vertices.len() <= 2 {
        return vertices;
    }
    if vertices.first() == vertices.last() {
        vertices.pop();
    }
    if vertices.len() <= 2 {
        return vertices;
    }
    let mut signed_area = 0.0;
    for i in 0..vertices.len() {
        let (x1, y1) = vertices[i];
        let (x2, y2) = vertices[(i + 1) % vertices.len()];
        signed_area += x1 * y2 - x2 * y1;
    }
    if signed_area / 2.0 < 0.0 {
        vertices.reverse();
    }
    let start = vertices
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    vertices.rotate_left(start);
    vertices
}

/// Euclidean distance expression between two 2D coordinate column pairs.
pub fn euclidean_distance_expr(x1: &str, y1: &str, x2: &str, y2: &str) -> Expr {
    let dx = col(x1) - col(x2);
    let dy = col(y1) - col(y2);
    (dx.pow(2) + dy.pow(2)).sqrt()
}

/// Convert pixel coordinates to meters.
pub fn pixel_to_meters_expr(col_name: &str, meters_per_pixel: f64) -> Expr {
    col(col_name) * lit(meters_per_pixel)
}

/// Invert the Y axis (for coordinate systems where Y increases downward).
pub fn invert_y_expr(col_name: &str, height: f64) -> Expr {
    lit(height) - col(col_name)
}

/// Compute the perpendicular distance from each point to a line defined
/// by two points (lx1, ly1) → (lx2, ly2). Fully vectorized.
pub fn perpendicular_distance_to_line_expr(
    px: &str,
    py: &str,
    lx1: f64,
    ly1: f64,
    lx2: f64,
    ly2: f64,
) -> Expr {
    let dx = lx2 - lx1;
    let dy = ly2 - ly1;
    let length = (dx * dx + dy * dy).sqrt();

    // |(p - l1) × (l2 - l1)| / |l2 - l1|
    let cross = (col(px) - lit(lx1)) * lit(dy) - (col(py) - lit(ly1)) * lit(dx);
    cross.abs() / lit(length)
}

/// Signed angle expression (radians) between direction vectors.
/// Uses atan2(cross, dot) for signed result.
pub fn signed_angle_expr(x1: &str, y1: &str, x2: &str, y2: &str) -> Expr {
    let cross = col(x1) * col(y2) - col(y1) * col(x2);
    let dot = dot_product_expr(x1, y1, x2, y2);
    cross.arctan2(dot)
}

/// Compute total path distance from (x, y) coordinate columns in a collected DataFrame.
///
/// Sums frame-to-frame Euclidean displacements. Optionally accepts a boolean mask
/// to only accumulate distance for frames where the mask is true. Null frames are
/// skipped but the previous valid position is still tracked for accurate displacement.
pub fn path_distance(
    df: &DataFrame,
    x_col: &str,
    y_col: &str,
    mask: Option<&BooleanChunked>,
) -> f64 {
    let xs = match df.column(x_col).and_then(|s| Ok(s.f64()?.clone())) {
        Ok(v) => v,
        Err(_) => return 0.0,
    };
    let ys = match df.column(y_col).and_then(|s| Ok(s.f64()?.clone())) {
        Ok(v) => v,
        Err(_) => return 0.0,
    };

    let mut total = 0.0;
    let mut prev: Option<(f64, f64)> = None;

    for i in 0..df.height() {
        let (x, y) = match (xs.get(i), ys.get(i)) {
            (Some(x), Some(y)) if x.is_finite() && y.is_finite() => (x, y),
            _ => {
                // Missing observations are not a measured path segment.
                prev = None;
                continue;
            }
        };

        let in_zone = mask.is_none_or(|m| m.get(i).unwrap_or(false));
        if in_zone && let Some((px, py)) = prev {
            let dx = x - px;
            let dy = y - py;
            total += (dx * dx + dy * dy).sqrt();
        }

        prev = Some((x, y));
    }

    total
}

#[cfg(test)]
mod polygon_tests {
    use super::canonicalize_polygon_vertices;

    #[test]
    fn closed_clockwise_polygons_and_degenerate_vertices_keep_their_contract() {
        assert_eq!(
            canonicalize_polygon_vertices(vec![
                (1.0, 1.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
            ]),
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        );
        let line = vec![(2.0, 1.0), (0.0, 1.0)];
        assert_eq!(canonicalize_polygon_vertices(line.clone()), line);
        assert!(canonicalize_polygon_vertices(Vec::new()).is_empty());
        // A closed two-vertex outline stays a degenerate segment.
        assert_eq!(
            canonicalize_polygon_vertices(vec![(1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]),
            vec![(1.0, 0.0), (0.0, 0.0)]
        );
    }
}
