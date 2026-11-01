use polars::prelude::*;

use crate::midpoint::CoordinatePairSource;
use bekiper_math::vector::direction_exprs;

const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

/// Compute the heading angle of the animal from two body-part coordinate pairs.
/// E.g., center_ear → nose defines the snout direction.
pub fn heading_angle_expr(from_x: &str, from_y: &str, to_x: &str, to_y: &str) -> Expr {
    (col(to_y) - col(from_y)).arctan2(col(to_x) - col(from_x))
}

/// Heading angle in degrees, derived from [`heading_angle_expr`].
pub fn heading_angle_deg_expr(from_x: &str, from_y: &str, to_x: &str, to_y: &str) -> Expr {
    heading_angle_expr(from_x, from_y, to_x, to_y) * lit(RAD_TO_DEG)
}

/// Compute a heading angle from two coordinate-pair sources.
pub fn heading_angle_from<A: CoordinatePairSource, B: CoordinatePairSource>(
    from: &A,
    to: &B,
) -> Expr {
    heading_angle_expr(from.x_col(), from.y_col(), to.x_col(), to.y_col())
}

/// Add direction vector columns for a heading (from → to).
pub fn add_direction_columns(
    lf: LazyFrame,
    from_x: &str,
    from_y: &str,
    to_x: &str,
    to_y: &str,
    prefix: &str,
) -> LazyFrame {
    let exprs = direction_exprs(from_x, from_y, to_x, to_y, prefix);
    let mut lf = lf;
    for expr in exprs {
        lf = lf.with_column(expr);
    }
    lf
}

/// Add direction vector columns from two coordinate-pair sources.
pub fn add_direction_columns_from<A: CoordinatePairSource, B: CoordinatePairSource>(
    lf: LazyFrame,
    from: &A,
    to: &B,
    prefix: &str,
) -> LazyFrame {
    add_direction_columns(
        lf,
        from.x_col(),
        from.y_col(),
        to.x_col(),
        to.y_col(),
        prefix,
    )
}
