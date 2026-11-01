use polars::prelude::*;

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_core::shape::Shape;
use bekiper_core::types::CoordinateColumns;
use bekiper_perimeter::perimeter::Perimeter;

use crate::heuristic::{
    Heuristic, coordinate_label_from_xy, decomposition_component, diag_column_name,
};

/// Coordinates inside the arena but outside its center, for any arena shape.
pub struct PeripheryHeuristic<S: Shape> {
    pub arena: Perimeter<S>,
    pub center: Perimeter<S>,
    pub coords: CoordinateColumns,
}

impl<S: Shape + Clone + 'static> Heuristic for PeripheryHeuristic<S> {
    fn name(&self) -> &str {
        "in_periphery"
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let in_arena = self
            .arena
            .confinement_mask(&self.coords)
            .fill_null(lit(false));
        let in_center = self
            .center
            .confinement_mask(&self.coords)
            .fill_null(lit(false));
        let in_periphery = in_arena
            .clone()
            .and(in_center.clone().not())
            .fill_null(lit(false));
        lf.with_columns([
            in_arena.alias(diag_column_name(self.name(), "in_arena")),
            in_center.alias(diag_column_name(self.name(), "in_center")),
            in_periphery.alias(self.name()),
        ])
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        let coordinates = vec![coordinate_label_from_xy(&self.coords.x)];
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.name().to_string(),
            result_column: self.name().to_string(),
            coordinate_labels: coordinates.clone(),
            perimeter_label: Some(self.arena.label.clone()),
            components: [("in_arena", &self.arena), ("in_center", &self.center)]
                .into_iter()
                .map(|(label, perimeter)| {
                    bekiper_core::inspection::HeuristicDecompositionComponent {
                        evaluated_perimeter: perimeter.shape.inspection_perimeter(&perimeter.label),
                        ..decomposition_component(
                            label,
                            diag_column_name(self.name(), label),
                            coordinates.clone(),
                            Some(perimeter.label.clone()),
                        )
                    }
                })
                .collect(),
            show_final_result: true,
        })
    }
}
