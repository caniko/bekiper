use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::shape::{Expandable, RayIntersectable, Shape};
use bekiper_core::types::{CoordinateColumns, Label};

/// A labeled perimeter wrapping any geometric shape.
/// Generic over `S` so the compiler monomorphizes all operations
/// per concrete shape — no dynamic dispatch in hot paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perimeter<S: Shape> {
    pub label: Label,
    pub shape: S,
}

impl<S: Shape + Clone + 'static> Perimeter<S> {
    pub fn new(label: impl Into<Label>, shape: S) -> Self {
        Self {
            label: label.into(),
            shape,
        }
    }

    /// Polars boolean expression: true where coordinates are inside this perimeter.
    pub fn confinement_mask(&self, coords: &CoordinateColumns) -> Expr {
        self.shape.confinement_expr(&coords.x, &coords.y)
    }

    /// Filter a LazyFrame to only rows inside this perimeter.
    pub fn filter_confined(&self, lf: LazyFrame, coords: &CoordinateColumns) -> LazyFrame {
        lf.filter(self.confinement_mask(coords))
    }

    /// Null coordinate columns where the reference point is outside this perimeter.
    ///
    /// Unlike `filter_confined` (which drops rows), this preserves all frames
    /// and sets the listed columns to null where the reference coords fall outside.
    /// This keeps frame indices and timing intact.
    pub fn null_outside(
        &self,
        lf: LazyFrame,
        reference: &CoordinateColumns,
        columns_to_null: &[String],
    ) -> LazyFrame {
        let outside = self.confinement_mask(reference).not();
        let exprs: Vec<Expr> = columns_to_null
            .iter()
            .map(|c| {
                when(outside.clone())
                    .then(lit(NULL))
                    .otherwise(col(c))
                    .alias(c)
            })
            .collect();
        if exprs.is_empty() {
            return lf;
        }
        lf.with_columns(exprs)
    }
}

impl<S: Expandable + Clone + 'static> Perimeter<S>
where
    S::Output: Clone + 'static,
{
    /// Create an expanded perimeter (for proximity detection).
    pub fn expanded(&self, distance: f64) -> Perimeter<S::Output> {
        Perimeter {
            label: format!("{}_expanded_{distance}", self.label),
            shape: self.shape.expand(distance),
        }
    }
}

impl<S: RayIntersectable + Clone + 'static> Perimeter<S> {
    /// Polars boolean expression: true where a ray from the given origin
    /// in the given direction intersects this perimeter.
    pub fn ray_intersection_mask(
        &self,
        origin: &CoordinateColumns,
        direction: &CoordinateColumns,
    ) -> Expr {
        self.shape
            .ray_filter_expr(&origin.x, &origin.y, &direction.x, &direction.y)
    }

    /// Polars boolean expression: true where a cone centred on the given
    /// direction intersects this perimeter.
    pub fn cone_intersection_mask(
        &self,
        origin: &CoordinateColumns,
        direction: &CoordinateColumns,
        half_angle_degrees: f64,
    ) -> Expr {
        self.shape.cone_filter_expr(
            &origin.x,
            &origin.y,
            &direction.x,
            &direction.y,
            half_angle_degrees,
        )
    }

    /// Float expression: minimum angular offset from the direction vector to
    /// the perimeter boundary, in degrees.
    pub fn ray_min_angle_deg_expr(
        &self,
        origin: &CoordinateColumns,
        direction: &CoordinateColumns,
    ) -> Expr {
        self.shape
            .ray_min_angle_deg_expr(&origin.x, &origin.y, &direction.x, &direction.y)
    }

    /// Float expression: chosen boundary target X coordinate.
    pub fn ray_target_x_expr(
        &self,
        origin: &CoordinateColumns,
        direction: &CoordinateColumns,
    ) -> Expr {
        self.shape
            .ray_target_x_expr(&origin.x, &origin.y, &direction.x, &direction.y)
    }

    /// Float expression: chosen boundary target Y coordinate.
    pub fn ray_target_y_expr(
        &self,
        origin: &CoordinateColumns,
        direction: &CoordinateColumns,
    ) -> Expr {
        self.shape
            .ray_target_y_expr(&origin.x, &origin.y, &direction.x, &direction.y)
    }
}
