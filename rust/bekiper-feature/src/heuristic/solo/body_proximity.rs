use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_core::shape::Expandable;
use bekiper_core::types::CoordinateColumns;
use bekiper_perimeter::perimeter::Perimeter;

use crate::heuristic::Heuristic;
use crate::heuristic::proximity::ProximityHeuristic;
use crate::heuristic::{coordinate_label_from_xy, decomposition_component, diag_column_name};

/// Detects when center_ear or tail_base are within proximity of an object.
///
/// Proximity-only heuristic — no ray check needed.
/// Generic over shape `S` — monomorphized per concrete shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyProximityHeuristic<S: Expandable> {
    pub perimeter: Perimeter<S>,
    pub max_distance: f64,
    #[serde(default = "default_head_coords")]
    pub head_coords: CoordinateColumns,
    #[serde(default = "default_tail_coords")]
    pub tail_coords: CoordinateColumns,
    /// Optional torso/centroid binding, retained as a descriptive gate only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub torso_coords: Option<CoordinateColumns>,
    /// Column name for the boolean result. Defaults to `"body_proximity"`;
    /// set to e.g. `"body_proximity_novel"` for per-object disambiguation.
    #[serde(default = "default_body_proximity_name")]
    pub name: String,
    /// Optional output column for the combined raw boolean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_column: Option<String>,
}

fn default_body_proximity_name() -> String {
    "body_proximity".into()
}

fn default_head_coords() -> CoordinateColumns {
    CoordinateColumns::new("center_ear_x", "center_ear_y")
}

fn default_tail_coords() -> CoordinateColumns {
    CoordinateColumns::new("tail_base_x", "tail_base_y")
}

impl<S: Expandable> BodyProximityHeuristic<S> {
    fn result_column_name(&self) -> &str {
        self.result_column.as_deref().unwrap_or(self.name.as_str())
    }

    fn center_ear_col(&self) -> String {
        diag_column_name(&self.name, "center_ear_near")
    }

    fn tail_base_col(&self) -> String {
        diag_column_name(&self.name, "tail_base_near")
    }
}

impl<S> ProximityHeuristic<S> for BodyProximityHeuristic<S>
where
    S: Expandable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn max_distance_meters(&self) -> f64 {
        self.max_distance
    }

    fn perimeter(&self) -> &Perimeter<S> {
        &self.perimeter
    }

    fn coordinate_columns(&self) -> Vec<CoordinateColumns> {
        vec![self.head_coords.clone(), self.tail_coords.clone()]
    }
}

impl<S> Heuristic for BodyProximityHeuristic<S>
where
    S: Expandable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn result_column(&self) -> &str {
        self.result_column_name()
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let center_ear = self
            .perimeter
            .expanded(self.max_distance)
            .confinement_mask(&self.head_coords);
        let tail_base = self
            .perimeter
            .expanded(self.max_distance)
            .confinement_mask(&self.tail_coords);
        // Keep tail proximity as a diagnostic channel, but treat investigation
        // as head-led contact so backing into an object does not count.
        let combined = center_ear.clone().fill_null(lit(false));

        let mut exprs = vec![
            center_ear.alias(self.center_ear_col()),
            tail_base.alias(self.tail_base_col()),
            combined.alias(self.result_column_name()),
        ];
        if let Some(torso) = &self.torso_coords {
            exprs.push(
                self.perimeter
                    .expanded(self.max_distance)
                    .confinement_mask(torso)
                    .fill_null(lit(false))
                    .alias(diag_column_name(&self.name, "torso_near")),
            );
        }
        lf.with_columns(exprs)
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        let mut spec = HeuristicDecompositionSpec {
            heuristic_name: self.name.clone(),
            result_column: self.result_column_name().to_string(),
            coordinate_labels: vec![coordinate_label_from_xy(&self.head_coords.x)],
            perimeter_label: Some(self.perimeter.label.clone()),
            components: vec![
                decomposition_component(
                    "center_ear_near",
                    self.center_ear_col(),
                    vec![coordinate_label_from_xy(&self.head_coords.x)],
                    Some(self.perimeter.label.clone()),
                )
                .with_buffer(self.max_distance),
                decomposition_component(
                    "tail_base_near",
                    self.tail_base_col(),
                    vec![coordinate_label_from_xy(&self.tail_coords.x)],
                    Some(self.perimeter.label.clone()),
                )
                .with_buffer(self.max_distance),
                decomposition_component(
                    "raw_combined",
                    self.result_column_name().to_string(),
                    vec![coordinate_label_from_xy(&self.head_coords.x)],
                    Some(self.perimeter.label.clone()),
                ),
            ]
            .into_iter()
            .map(|c| c.with_geometry(&self.perimeter.shape, &self.perimeter.label))
            .collect(),
            show_final_result: false,
        };
        if let Some(torso) = &self.torso_coords {
            spec.components.insert(
                1,
                decomposition_component(
                    "torso_near",
                    diag_column_name(&self.name, "torso_near"),
                    vec![coordinate_label_from_xy(&torso.x)],
                    Some(self.perimeter.label.clone()),
                )
                .with_buffer(self.max_distance)
                .with_geometry(&self.perimeter.shape, &self.perimeter.label),
            );
        }
        Some(spec)
    }
}
