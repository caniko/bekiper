use polars::prelude::*;

use bekiper_core::shape::RayIntersectable;
use bekiper_core::types::RaySpec;
use bekiper_math::discrete::any_of;
use bekiper_perimeter::perimeter::Perimeter;

use crate::heuristic::diag_column_name;

fn ray_component_name(spec: &RaySpec, suffix: &str) -> String {
    if spec.label.is_empty() {
        suffix.to_string()
    } else {
        format!("{}_{}", spec.label, suffix)
    }
}

pub fn ray_min_angle_col(heuristic_name: &str, spec: &RaySpec) -> String {
    diag_column_name(heuristic_name, &ray_component_name(spec, "min_angle_deg"))
}

pub fn ray_angle_margin_col(heuristic_name: &str, spec: &RaySpec) -> String {
    diag_column_name(
        heuristic_name,
        &ray_component_name(spec, "angle_margin_deg"),
    )
}

pub fn ray_target_x_col(heuristic_name: &str, spec: &RaySpec) -> String {
    diag_column_name(heuristic_name, &ray_component_name(spec, "target_x"))
}

pub fn ray_target_y_col(heuristic_name: &str, spec: &RaySpec) -> String {
    diag_column_name(heuristic_name, &ray_component_name(spec, "target_y"))
}

/// Trait for ray-based (field-of-view) heuristics.
///
/// Shared default `ray_expr()` — DRY across Whisker and Olfaction.
/// Generic over shape `S` — monomorphized per concrete shape.
pub trait RayHeuristic<S>: Send + Sync
where
    S: RayIntersectable + Clone + 'static,
{
    /// Maximum angle offset (degrees) from heading to shape.
    fn max_angle_degrees(&self) -> f64 {
        45.0
    }

    /// Heuristic name used for diagnostic column naming.
    fn diagnostic_name(&self) -> &str;

    /// The target perimeter for ray intersection.
    fn perimeter(&self) -> &Perimeter<S>;

    /// Ray specifications: origin + direction column pairs.
    fn ray_specs(&self) -> Vec<RaySpec>;

    /// Boolean expression: true where ANY sensing cone intersects the perimeter.
    ///
    /// Uses `max_angle_degrees()` as the cone half-angle, so a heuristic can
    /// accept near-miss headings inside its configured field of view instead
    /// of only exact line intersections.
    fn ray_expr(&self) -> Expr {
        let specs = self.ray_specs();
        let exprs: Vec<Expr> = specs.iter().map(|spec| self.ray_hit_expr(spec)).collect();
        any_of(exprs)
    }

    /// Boolean expression for one named ray spec.
    fn ray_hit_expr(&self, spec: &RaySpec) -> Expr {
        self.perimeter().cone_intersection_mask(
            &spec.origin,
            &spec.direction,
            self.max_angle_degrees(),
        )
    }

    /// Minimum-angle expression for one named ray spec.
    fn ray_min_angle_expr(&self, spec: &RaySpec) -> Expr {
        self.perimeter()
            .ray_min_angle_deg_expr(&spec.origin, &spec.direction)
    }

    /// Angle-margin expression (`threshold - min_angle`) for one named ray spec.
    fn ray_angle_margin_expr(&self, spec: &RaySpec) -> Expr {
        lit(self.max_angle_degrees()) - self.ray_min_angle_expr(spec)
    }

    /// Target-X expression for one named ray spec.
    fn ray_target_x_expr(&self, spec: &RaySpec) -> Expr {
        self.perimeter()
            .ray_target_x_expr(&spec.origin, &spec.direction)
    }

    /// Target-Y expression for one named ray spec.
    fn ray_target_y_expr(&self, spec: &RaySpec) -> Expr {
        self.perimeter()
            .ray_target_y_expr(&spec.origin, &spec.direction)
    }

    /// Numeric/target diagnostics for one named ray spec.
    fn ray_diagnostic_exprs(&self) -> Vec<Expr> {
        self.ray_specs()
            .iter()
            .flat_map(|spec| {
                let min_angle = self.ray_min_angle_expr(spec);
                let margin = self.ray_angle_margin_expr(spec);
                [
                    min_angle.alias(ray_min_angle_col(self.diagnostic_name(), spec)),
                    margin.alias(ray_angle_margin_col(self.diagnostic_name(), spec)),
                    self.ray_target_x_expr(spec)
                        .alias(ray_target_x_col(self.diagnostic_name(), spec)),
                    self.ray_target_y_expr(spec)
                        .alias(ray_target_y_col(self.diagnostic_name(), spec)),
                ]
            })
            .collect()
    }
}
