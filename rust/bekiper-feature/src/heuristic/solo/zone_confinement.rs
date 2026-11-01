use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_core::shape::Shape;
use bekiper_core::types::CoordinateColumns;
use bekiper_perimeter::perimeter::Perimeter;

use crate::heuristic::Heuristic;
use crate::heuristic::{coordinate_label_from_xy, decomposition_component, diag_column_name};

/// Detects when coordinates are inside a named zone.
///
/// Produces a boolean column `in_{zone_name}` that is true when the
/// tracked point is within the zone perimeter. Used for zone-based
/// analysis in OFT (center/periphery), Y-maze (arms/center), etc.
///
/// For multi-node zone entry (e.g. Y-maze requiring both head AND tail
/// to be in the arm), compose multiple `ZoneConfinementHeuristic`s
/// and AND their results together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneConfinementHeuristic<S: Shape> {
    pub zone_name: String,
    pub perimeter: Perimeter<S>,
    pub coords: CoordinateColumns,
}

impl<S: Shape + Clone + Send + Sync + 'static> Heuristic for ZoneConfinementHeuristic<S> {
    fn name(&self) -> &str {
        &self.zone_name
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let mask = self
            .perimeter
            .confinement_mask(&self.coords)
            .fill_null(lit(false));
        let component_col = diag_column_name(self.name(), "in_zone");
        lf.with_columns([mask.clone().alias(&component_col), mask.alias(self.name())])
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.zone_name.clone(),
            result_column: self.zone_name.clone(),
            coordinate_labels: vec![coordinate_label_from_xy(&self.coords.x)],
            perimeter_label: Some(self.perimeter.label.clone()),
            components: vec![bekiper_core::inspection::HeuristicDecompositionComponent {
                evaluated_perimeter: self
                    .perimeter
                    .shape
                    .inspection_perimeter(&self.perimeter.label),
                ..decomposition_component(
                    "in_zone",
                    diag_column_name(self.name(), "in_zone"),
                    vec![coordinate_label_from_xy(&self.coords.x)],
                    Some(self.perimeter.label.clone()),
                )
            }],
            show_final_result: false,
        })
    }
}
