use polars::prelude::*;

pub const GEOMETRY_EPSILON: f64 = 1e-9;

/// Compute the unit (normalized) vector from (x, y) columns.
/// Returns two new columns: `{prefix}_ux`, `{prefix}_uy`.
pub fn unit_vector_exprs(x_col: &str, y_col: &str, prefix: &str) -> Vec<Expr> {
    let magnitude = (col(x_col).pow(2) + col(y_col).pow(2)).sqrt();
    vec![
        (col(x_col) / magnitude.clone()).alias(format!("{prefix}_ux")),
        (col(y_col) / magnitude).alias(format!("{prefix}_uy")),
    ]
}

/// Compute the orthogonal (perpendicular) vector: (-y, x).
pub fn orthogonal_vector_exprs(x_col: &str, y_col: &str, prefix: &str) -> Vec<Expr> {
    vec![
        (lit(0.0) - col(y_col)).alias(format!("{prefix}_ortho_x")),
        col(x_col).alias(format!("{prefix}_ortho_y")),
    ]
}

/// Dot product of two 2D vector column pairs.
pub fn dot_product_expr(x1: &str, y1: &str, x2: &str, y2: &str) -> Expr {
    col(x1) * col(x2) + col(y1) * col(y2)
}

/// Direction vector from point A to point B columns.
pub fn direction_exprs(
    from_x: &str,
    from_y: &str,
    to_x: &str,
    to_y: &str,
    prefix: &str,
) -> Vec<Expr> {
    vec![
        (col(to_x) - col(from_x)).alias(format!("{prefix}_dx")),
        (col(to_y) - col(from_y)).alias(format!("{prefix}_dy")),
    ]
}

/// Rotate 2D vectors by an angle (in radians) using rotation matrix.
/// Returns rotated (x, y) as two new columns.
pub fn rotate_exprs(x_col: &str, y_col: &str, angle_rad: f64, prefix: &str) -> Vec<Expr> {
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    vec![
        (col(x_col) * lit(cos_a) - col(y_col) * lit(sin_a)).alias(format!("{prefix}_rot_x")),
        (col(x_col) * lit(sin_a) + col(y_col) * lit(cos_a)).alias(format!("{prefix}_rot_y")),
    ]
}

/// Test whether a ray from `origin` in `direction` intersects a line segment
/// from `seg_start` to `seg_end`. Uses the parametric intersection method.
///
/// This is a scalar function used inside `Shape::ray_intersects` implementations.
pub fn ray_line_segment_intersection(
    origin: (f64, f64),
    direction: (f64, f64),
    seg_start: (f64, f64),
    seg_end: (f64, f64),
) -> bool {
    ray_line_segment_intersection_point(origin, direction, seg_start, seg_end).is_some()
}

/// Return the first intersection point between a ray and a line segment.
pub fn ray_line_segment_intersection_point(
    origin: (f64, f64),
    direction: (f64, f64),
    seg_start: (f64, f64),
    seg_end: (f64, f64),
) -> Option<(f64, f64)> {
    let (ox, oy) = origin;
    let (dx, dy) = direction;
    let (sx, sy) = seg_start;
    let (ex, ey) = seg_end;

    let ray_len_sq = dx * dx + dy * dy;
    if ray_len_sq <= GEOMETRY_EPSILON {
        return None;
    }

    let seg_dx = ex - sx;
    let seg_dy = ey - sy;
    let offset_x = sx - ox;
    let offset_y = sy - oy;

    let denom = dx * seg_dy - dy * seg_dx;
    if denom.abs() <= GEOMETRY_EPSILON {
        let cross = offset_x * dy - offset_y * dx;
        if cross.abs() > GEOMETRY_EPSILON {
            return None;
        }

        let t0 = (offset_x * dx + offset_y * dy) / ray_len_sq;
        let t1 = ((ex - ox) * dx + (ey - oy) * dy) / ray_len_sq;
        let t_min = t0.min(t1);
        let t_max = t0.max(t1);
        if t_max < -GEOMETRY_EPSILON {
            return None;
        }
        let t = if t_min >= -GEOMETRY_EPSILON {
            t_min.max(0.0)
        } else {
            0.0
        };
        return Some((ox + t * dx, oy + t * dy));
    }

    let t = (offset_x * seg_dy - offset_y * seg_dx) / denom;
    let u = (offset_x * dy - offset_y * dx) / denom;
    if t >= -GEOMETRY_EPSILON && (-GEOMETRY_EPSILON..=1.0 + GEOMETRY_EPSILON).contains(&u) {
        Some((ox + t.max(0.0) * dx, oy + t.max(0.0) * dy))
    } else {
        None
    }
}

/// Compute the angle between two 2D vectors in radians.
pub fn angle_between(v1: (f64, f64), v2: (f64, f64)) -> f64 {
    let dot = v1.0 * v2.0 + v1.1 * v2.1;
    let mag1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
    let mag2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
    let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    cos_angle.acos()
}

/// Signed angle from `v1` to `v2` in radians, in `[-pi, pi]`.
pub fn signed_angle_between(v1: (f64, f64), v2: (f64, f64)) -> f64 {
    let cross = v1.0 * v2.1 - v1.1 * v2.0;
    let dot = v1.0 * v2.0 + v1.1 * v2.1;
    cross.atan2(dot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_hits_segment() {
        assert!(ray_line_segment_intersection(
            (0.0, 0.0),
            (1.0, 0.0),
            (5.0, -1.0),
            (5.0, 1.0),
        ));
    }

    #[test]
    fn test_ray_hit_point() {
        let point =
            ray_line_segment_intersection_point((0.0, 0.0), (1.0, 0.0), (5.0, -1.0), (5.0, 1.0))
                .unwrap();
        assert!((point.0 - 5.0).abs() < 1e-10);
        assert!(point.1.abs() < 1e-10);
    }

    #[test]
    fn test_ray_misses_segment() {
        assert!(!ray_line_segment_intersection(
            (0.0, 0.0),
            (0.0, 1.0),
            (5.0, -1.0),
            (5.0, 1.0),
        ));
    }

    #[test]
    fn test_ray_hits_collinear_segment_overlap() {
        let point =
            ray_line_segment_intersection_point((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (3.0, 0.0))
                .unwrap();
        assert!((point.0 - 1.0).abs() < 1e-10);
        assert!(point.1.abs() < 1e-10);
    }

    #[test]
    fn test_ray_hits_collinear_segment_when_origin_is_on_segment() {
        let point =
            ray_line_segment_intersection_point((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (2.0, 0.0))
                .unwrap();
        assert!(point.0.abs() < 1e-10);
        assert!(point.1.abs() < 1e-10);
    }

    #[test]
    fn test_ray_misses_collinear_segment_behind_origin() {
        assert!(
            ray_line_segment_intersection_point((0.0, 0.0), (1.0, 0.0), (-3.0, 0.0), (-1.0, 0.0))
                .is_none()
        );
    }

    #[test]
    fn test_angle_between_orthogonal() {
        let angle = angle_between((1.0, 0.0), (0.0, 1.0));
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_signed_angle_between() {
        let angle = signed_angle_between((1.0, 0.0), (0.0, 1.0));
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }
}
