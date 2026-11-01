use polars::prelude::*;

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_core::shape::Shape;
use bekiper_core::types::CoordinateColumns;
use bekiper_perimeter::perimeter::Perimeter;

use crate::heuristic::{Heuristic, coordinate_label_from_xy, decomposition_component};

/// Helper heuristic: true when coordinates are OUTSIDE the object perimeter.
///
/// Used to ensure the animal is investigating from outside, not from within.
/// Generic over shape — monomorphized.
pub struct OutsidePerimeterHeuristic<S: Shape> {
    pub perimeter: Perimeter<S>,
    pub coords: CoordinateColumns,
    /// Column name. Defaults to `"outside_perimeter"`; set to e.g.
    /// `"outside_novel"` for per-object disambiguation.
    pub name: String,
}

impl<S: Shape + Clone + Send + Sync + 'static> Heuristic for OutsidePerimeterHeuristic<S> {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let inside = self.perimeter.confinement_mask(&self.coords);
        lf.with_column(inside.not().fill_null(lit(false)).alias(self.name()))
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.name.clone(),
            result_column: self.name.clone(),
            coordinate_labels: vec![coordinate_label_from_xy(&self.coords.x)],
            perimeter_label: Some(self.perimeter.label.clone()),
            components: vec![decomposition_component(
                "outside_perimeter",
                self.name.clone(),
                vec![coordinate_label_from_xy(&self.coords.x)],
                Some(self.perimeter.label.clone()),
            )],
            show_final_result: false,
        })
    }
}
