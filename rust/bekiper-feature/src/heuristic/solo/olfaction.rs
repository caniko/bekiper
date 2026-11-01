use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_core::shape::{Expandable, RayIntersectable};
use bekiper_core::types::{CoordinateColumns, RaySpec};
use bekiper_perimeter::perimeter::Perimeter;

use crate::heuristic::Heuristic;
use crate::heuristic::proximity::ProximityHeuristic;
use crate::heuristic::ray::RayHeuristic;
use crate::heuristic::{coordinate_label_from_xy, decomposition_component, diag_column_name};

/// Detects olfactory investigation of objects.
///
/// Combines proximity (nose within distance) AND
/// field-of-view (ray from snout toward object intersects perimeter).
///
/// Reuses shared trait defaults — DRY. Monomorphized per shape type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlfactionHeuristic<S: Expandable + RayIntersectable> {
    pub perimeter: Perimeter<S>,
    pub max_distance: f64,
    pub max_angle: f64,
    /// Distance below which close-range olfaction passes even when the
    /// heading vector is noisy. Direction estimates are least reliable once
    /// the nose is already nearly touching the target. Default: 0.01.
    #[serde(default = "default_olfaction_contact_distance")]
    pub contact_distance: f64,
    /// Column name for the boolean result. Defaults to `"olfaction"`;
    /// set to e.g. `"olfaction_novel"` for per-object disambiguation.
    #[serde(default = "default_olfaction_name")]
    pub name: String,
    /// Optional output column for the combined raw boolean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_column: Option<String>,
}

fn default_olfaction_name() -> String {
    "olfaction".into()
}

fn default_olfaction_contact_distance() -> f64 {
    0.01
}

impl<S: Expandable + RayIntersectable> OlfactionHeuristic<S> {
    fn result_column_name(&self) -> &str {
        self.result_column.as_deref().unwrap_or(self.name.as_str())
    }

    fn nose_near_col(&self) -> String {
        diag_column_name(&self.name, "nose_near")
    }

    fn ray_hits_col(&self) -> String {
        diag_column_name(&self.name, "ray_hits_object")
    }

    fn nose_contact_col(&self) -> String {
        diag_column_name(&self.name, "nose_contact")
    }
}

impl<S> ProximityHeuristic<S> for OlfactionHeuristic<S>
where
    S: Expandable + RayIntersectable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn max_distance_meters(&self) -> f64 {
        self.max_distance
    }

    fn perimeter(&self) -> &Perimeter<S> {
        &self.perimeter
    }

    fn coordinate_columns(&self) -> Vec<CoordinateColumns> {
        vec![CoordinateColumns::new("nose_x", "nose_y")]
    }
}

impl<S> RayHeuristic<S> for OlfactionHeuristic<S>
where
    S: Expandable + RayIntersectable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn max_angle_degrees(&self) -> f64 {
        self.max_angle
    }

    fn perimeter(&self) -> &Perimeter<S> {
        &self.perimeter
    }

    fn diagnostic_name(&self) -> &str {
        &self.name
    }

    fn ray_specs(&self) -> Vec<RaySpec> {
        vec![
            // Ray from nose in the direction the snout is pointing
            // (center_ear → nose direction)
            RaySpec {
                label: "ray".into(),
                origin: CoordinateColumns::new("nose_x", "nose_y"),
                direction: CoordinateColumns::new("snout_dx", "snout_dy"),
            },
        ]
    }
}

impl<S> Heuristic for OlfactionHeuristic<S>
where
    S: Expandable + RayIntersectable + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn result_column(&self) -> &str {
        self.result_column_name()
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let nose_near = self.proximity_expr().fill_null(lit(false));
        let nose_contact = self
            .perimeter
            .expanded(self.contact_distance)
            .confinement_mask(&CoordinateColumns::new("nose_x", "nose_y"))
            .fill_null(lit(false));
        let ray_hits = self.ray_expr().fill_null(lit(false));
        let contact_or_ray = nose_contact
            .clone()
            .or(ray_hits.clone())
            .fill_null(lit(false));
        let combined = nose_near.clone().and(contact_or_ray).fill_null(lit(false));
        let mut exprs = vec![
            nose_near.alias(self.nose_near_col()),
            nose_contact.alias(self.nose_contact_col()),
            ray_hits.alias(self.ray_hits_col()),
            combined.alias(self.result_column_name()),
        ];
        exprs.extend(self.ray_diagnostic_exprs());
        lf.with_columns(exprs)
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.name.clone(),
            result_column: self.result_column_name().to_string(),
            coordinate_labels: vec![coordinate_label_from_xy("nose_x")],
            perimeter_label: Some(self.perimeter.label.clone()),
            components: vec![
                decomposition_component(
                    "nose_near",
                    self.nose_near_col(),
                    vec![coordinate_label_from_xy("nose_x")],
                    Some(self.perimeter.label.clone()),
                )
                .with_buffer(self.max_distance),
                decomposition_component(
                    "ray_hits_object",
                    self.ray_hits_col(),
                    vec![coordinate_label_from_xy("nose_x")],
                    Some(self.perimeter.label.clone()),
                )
                .with_ray(
                    &self.ray_specs()[0],
                    diag_column_name(&self.name, "ray_target_x"),
                    diag_column_name(&self.name, "ray_target_y"),
                )
                .with_values(diag_column_name(&self.name, "ray_min_angle_deg"), "degrees"),
                decomposition_component(
                    "nose_contact",
                    self.nose_contact_col(),
                    vec![coordinate_label_from_xy("nose_x")],
                    Some(self.perimeter.label.clone()),
                )
                .with_buffer(self.contact_distance),
                decomposition_component(
                    "raw_combined",
                    self.result_column_name().to_string(),
                    vec![coordinate_label_from_xy("nose_x")],
                    Some(self.perimeter.label.clone()),
                ),
            ]
            .into_iter()
            .map(|c| c.with_geometry(&self.perimeter.shape, &self.perimeter.label))
            .collect(),
            show_final_result: false,
        })
    }
}
