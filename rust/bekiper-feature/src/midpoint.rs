use polars::prelude::*;

use bekiper_core::types::CoordinateColumns;

/// Anything that exposes an `(x, y)` coordinate column pair.
pub trait CoordinatePairSource {
    fn x_col(&self) -> &str;
    fn y_col(&self) -> &str;
}

impl CoordinatePairSource for CoordinateColumns {
    fn x_col(&self) -> &str {
        &self.x
    }

    fn y_col(&self) -> &str {
        &self.y
    }
}

impl CoordinatePairSource for (&str, &str) {
    fn x_col(&self) -> &str {
        self.0
    }

    fn y_col(&self) -> &str {
        self.1
    }
}

/// Compute midpoint of two coordinate column pairs and add as new columns.
pub fn midpoint_exprs(
    x1: &str,
    y1: &str,
    x2: &str,
    y2: &str,
    out_x: &str,
    out_y: &str,
) -> Vec<Expr> {
    vec![
        ((col(x1) + col(x2)) / lit(2.0)).alias(out_x),
        ((col(y1) + col(y2)) / lit(2.0)).alias(out_y),
    ]
}

/// Compute midpoint expressions from two coordinate-pair sources.
pub fn midpoint_exprs_from<A: CoordinatePairSource, B: CoordinatePairSource>(
    a: &A,
    b: &B,
    out_x: &str,
    out_y: &str,
) -> Vec<Expr> {
    midpoint_exprs(a.x_col(), a.y_col(), b.x_col(), b.y_col(), out_x, out_y)
}
