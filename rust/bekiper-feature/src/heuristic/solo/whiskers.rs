use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_core::shape::{Expandable, RayIntersectable};
use bekiper_core::types::{CoordinateColumns, RaySpec};
use bekiper_perimeter::fan::{FanHit, fan_hit, outline};
use bekiper_perimeter::object_shape::ObjectShape;
use bekiper_perimeter::perimeter::Perimeter;

use crate::heuristic::Heuristic;
use crate::heuristic::combined::exploration::{
    IMEN_WHISKER_APEX_FRACTION, IMEN_WHISKER_HALF_ANGLE_DEG, LITERATURE_WHISKER_LENGTH_M,
};
use crate::heuristic::proximity::ProximityHeuristic;
use crate::heuristic::ray::RayHeuristic;
use crate::heuristic::{coordinate_label_from_xy, decomposition_component, diag_column_name};

/// Detects whisker/ear interactions with objects.
///
/// Combines proximity (left/right ear within distance) AND
/// field-of-view (ray from center_ear through left/right ear intersects object).
///
/// Reuses `ProximityHeuristic::proximity_expr()` and `RayHeuristic::ray_expr()`
/// defaults — DRY. Monomorphized per shape type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiskerHeuristic<S: Expandable + RayIntersectable> {
    pub perimeter: Perimeter<S>,
    pub max_distance: f64,
    pub max_angle: f64,
    /// Column name for the boolean result. Defaults to `"whisker_interaction"`;
    /// set to e.g. `"whisker_interaction_novel"` for per-object disambiguation.
    #[serde(default = "default_whisker_name")]
    pub name: String,
    /// Optional output column for the combined raw boolean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_column: Option<String>,
}

fn default_whisker_name() -> String {
    "whisker_interaction".into()
}

impl<S: Expandable + RayIntersectable> WhiskerHeuristic<S> {
    fn result_column_name(&self) -> &str {
        self.result_column.as_deref().unwrap_or(self.name.as_str())
    }

    fn left_proximity_col(&self) -> String {
        diag_column_name(&self.name, "left_ear_near")
    }

    fn right_proximity_col(&self) -> String {
        diag_column_name(&self.name, "right_ear_near")
    }

    fn left_ray_col(&self) -> String {
        diag_column_name(&self.name, "left_ray_hits_object")
    }

    fn right_ray_col(&self) -> String {
        diag_column_name(&self.name, "right_ray_hits_object")
    }

    fn selected_ray_min_angle_col(&self) -> String {
        diag_column_name(&self.name, "selected_ray_min_angle_deg")
    }

    fn selected_ray_margin_col(&self) -> String {
        diag_column_name(&self.name, "selected_ray_angle_margin_deg")
    }

    fn selected_ray_target_x_col(&self) -> String {
        diag_column_name(&self.name, "selected_ray_target_x")
    }

    fn selected_ray_target_y_col(&self) -> String {
        diag_column_name(&self.name, "selected_ray_target_y")
    }

    fn selected_ray_dx_col(&self) -> String {
        diag_column_name(&self.name, "selected_ray_dx")
    }

    fn selected_ray_dy_col(&self) -> String {
        diag_column_name(&self.name, "selected_ray_dy")
    }

    fn selected_ray_side_col(&self) -> String {
        diag_column_name(&self.name, "selected_ray_side")
    }
}

impl<S> ProximityHeuristic<S> for WhiskerHeuristic<S>
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
        vec![
            CoordinateColumns::new("left_ear_x", "left_ear_y"),
            CoordinateColumns::new("right_ear_x", "right_ear_y"),
        ]
    }
}

impl<S> RayHeuristic<S> for WhiskerHeuristic<S>
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
            // Ray from center_ear through left_ear
            RaySpec {
                label: "left_ray".into(),
                origin: CoordinateColumns::new("center_ear_x", "center_ear_y"),
                direction: CoordinateColumns::new("left_ear_dx", "left_ear_dy"),
            },
            // Ray from center_ear through right_ear
            RaySpec {
                label: "right_ray".into(),
                origin: CoordinateColumns::new("center_ear_x", "center_ear_y"),
                direction: CoordinateColumns::new("right_ear_dx", "right_ear_dy"),
            },
        ]
    }
}

impl<S> Heuristic for WhiskerHeuristic<S>
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
        let specs = self.ray_specs();
        let left_spec = &specs[0];
        let right_spec = &specs[1];
        let expanded = self.perimeter.expanded(self.max_distance);
        let left_proximity = expanded
            .confinement_mask(&CoordinateColumns::new("left_ear_x", "left_ear_y"))
            .fill_null(lit(false));
        let right_proximity = expanded
            .confinement_mask(&CoordinateColumns::new("right_ear_x", "right_ear_y"))
            .fill_null(lit(false));
        let left_ray = self.ray_hit_expr(left_spec).fill_null(lit(false));
        let right_ray = self.ray_hit_expr(right_spec).fill_null(lit(false));
        let left_min_angle = self.ray_min_angle_expr(left_spec);
        let right_min_angle = self.ray_min_angle_expr(right_spec);
        let left_margin = self.ray_angle_margin_expr(left_spec);
        let right_margin = self.ray_angle_margin_expr(right_spec);
        let left_target_x = self.ray_target_x_expr(left_spec);
        let left_target_y = self.ray_target_y_expr(left_spec);
        let right_target_x = self.ray_target_x_expr(right_spec);
        let right_target_y = self.ray_target_y_expr(right_spec);

        let left_combined = left_proximity
            .clone()
            .and(left_ray.clone())
            .fill_null(lit(false));
        let right_combined = right_proximity
            .clone()
            .and(right_ray.clone())
            .fill_null(lit(false));
        let combined = left_combined
            .clone()
            .or(right_combined.clone())
            .fill_null(lit(false));
        let choose_left = when(left_proximity.clone().and(right_proximity.clone().not()))
            .then(lit(true))
            .when(right_proximity.clone().and(left_proximity.clone().not()))
            .then(lit(false))
            .otherwise(
                left_margin
                    .clone()
                    .fill_null(lit(f64::NEG_INFINITY))
                    .gt_eq(right_margin.clone().fill_null(lit(f64::NEG_INFINITY))),
            )
            .fill_null(lit(true));

        let mut exprs = vec![
            left_proximity.alias(self.left_proximity_col()),
            right_proximity.alias(self.right_proximity_col()),
            left_ray.alias(self.left_ray_col()),
            right_ray.alias(self.right_ray_col()),
            left_combined.alias(diag_column_name(&self.name, "left_combined")),
            right_combined.alias(diag_column_name(&self.name, "right_combined")),
            when(choose_left.clone())
                .then(left_min_angle)
                .otherwise(right_min_angle)
                .alias(self.selected_ray_min_angle_col()),
            when(choose_left.clone())
                .then(left_margin)
                .otherwise(right_margin)
                .alias(self.selected_ray_margin_col()),
            when(choose_left.clone())
                .then(left_target_x)
                .otherwise(right_target_x)
                .alias(self.selected_ray_target_x_col()),
            when(choose_left.clone())
                .then(left_target_y)
                .otherwise(right_target_y)
                .alias(self.selected_ray_target_y_col()),
            when(choose_left.clone())
                .then(col("left_ear_dx"))
                .otherwise(col("right_ear_dx"))
                .alias(self.selected_ray_dx_col()),
            when(choose_left.clone())
                .then(col("left_ear_dy"))
                .otherwise(col("right_ear_dy"))
                .alias(self.selected_ray_dy_col()),
            when(choose_left)
                .then(lit("left"))
                .otherwise(lit("right"))
                .alias(self.selected_ray_side_col()),
            combined.alias(self.result_column_name()),
        ];
        exprs.extend(self.ray_diagnostic_exprs());
        lf.with_columns(exprs)
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.name.clone(),
            result_column: self.result_column_name().to_string(),
            coordinate_labels: vec![coordinate_label_from_xy("center_ear_x")],
            perimeter_label: Some(self.perimeter.label.clone()),
            components: vec![
                decomposition_component(
                    "left_ear_near",
                    self.left_proximity_col(),
                    vec![coordinate_label_from_xy("left_ear_x")],
                    Some(self.perimeter.label.clone()),
                )
                .with_buffer(self.max_distance),
                decomposition_component(
                    "right_ear_near",
                    self.right_proximity_col(),
                    vec![coordinate_label_from_xy("right_ear_x")],
                    Some(self.perimeter.label.clone()),
                )
                .with_buffer(self.max_distance),
                decomposition_component(
                    "left_ray_hits_object",
                    self.left_ray_col(),
                    vec![coordinate_label_from_xy("center_ear_x")],
                    Some(self.perimeter.label.clone()),
                )
                .with_ray(
                    &self.ray_specs()[0],
                    diag_column_name(&self.name, "left_ray_target_x"),
                    diag_column_name(&self.name, "left_ray_target_y"),
                )
                .with_values(
                    diag_column_name(&self.name, "left_ray_min_angle_deg"),
                    "degrees",
                ),
                decomposition_component(
                    "right_ray_hits_object",
                    self.right_ray_col(),
                    vec![coordinate_label_from_xy("center_ear_x")],
                    Some(self.perimeter.label.clone()),
                )
                .with_ray(
                    &self.ray_specs()[1],
                    diag_column_name(&self.name, "right_ray_target_x"),
                    diag_column_name(&self.name, "right_ray_target_y"),
                )
                .with_values(
                    diag_column_name(&self.name, "right_ray_min_angle_deg"),
                    "degrees",
                ),
                decomposition_component(
                    "raw_combined",
                    self.result_column_name().to_string(),
                    vec![coordinate_label_from_xy("center_ear_x")],
                    Some(self.perimeter.label.clone()),
                ),
                decomposition_component(
                    "left_combined",
                    diag_column_name(&self.name, "left_combined"),
                    vec![coordinate_label_from_xy("left_ear_x")],
                    Some(self.perimeter.label.clone()),
                ),
                decomposition_component(
                    "right_combined",
                    diag_column_name(&self.name, "right_combined"),
                    vec![coordinate_label_from_xy("right_ear_x")],
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

/// Whisker-fan geometry: apex at `apex_fraction` of the way from the head
/// midline (inter-ear midpoint, `center_ear`) to the nose tip, and two lateral circular sectors of radius
/// `length_m`, each spanning +-`half_angle_deg` about the head-axis perpendicular.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WhiskerFanParams {
    pub apex_fraction: f64,
    pub half_angle_deg: f64,
    pub length_m: f64,
}

impl Default for WhiskerFanParams {
    fn default() -> Self {
        Self {
            apex_fraction: IMEN_WHISKER_APEX_FRACTION,
            half_angle_deg: IMEN_WHISKER_HALF_ANGLE_DEG,
            length_m: LITERATURE_WHISKER_LENGTH_M,
        }
    }
}

impl WhiskerFanParams {
    pub fn validate(&self) -> Result<(), String> {
        let ok = self.apex_fraction.is_finite()
            && (0.0..=1.0).contains(&self.apex_fraction)
            && self.half_angle_deg.is_finite()
            && self.half_angle_deg > 0.0
            && self.half_angle_deg < 90.0
            && self.length_m.is_finite()
            && self.length_m > 0.0;
        ok.then_some(()).ok_or_else(|| {
            format!(
                "whisker fan needs apex_fraction in [0,1], half_angle_deg in (0,90) and finite positive length_m; got {self:?}"
            )
        })
    }
}

/// Fan parameters bound to one object outline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiskerFanSpec {
    pub params: WhiskerFanParams,
    pub outline: Vec<(f64, f64)>,
}

impl WhiskerFanSpec {
    pub fn new(params: WhiskerFanParams, shape: &ObjectShape) -> Self {
        Self {
            params,
            outline: outline(shape),
        }
    }

    /// Scalar hit for one frame (apex, then fans). Degenerate axis is a miss.
    pub fn hit_at(&self, ear: (f64, f64), nose: (f64, f64)) -> Option<FanHit> {
        let axis = (nose.0 - ear.0, nose.1 - ear.1);
        let apex = (
            ear.0 + self.params.apex_fraction * axis.0,
            ear.1 + self.params.apex_fraction * axis.1,
        );
        fan_hit(
            apex,
            axis,
            self.params.half_angle_deg,
            self.params.length_m,
            &self.outline,
        )
    }

    /// `[hit, side, nearest_distance_m]` expressions over the coordinate
    /// columns. Missing pose is a miss (`false`, null, null).
    pub fn hit_exprs(&self, ear: &CoordinateColumns, nose: &CoordinateColumns) -> [Expr; 3] {
        [
            self.column::<BooleanChunked, bool>(
                ear,
                nose,
                "whisker_fan_hit",
                DataType::Boolean,
                |h| Some(h.is_some()),
            ),
            self.column::<StringChunked, &'static str>(
                ear,
                nose,
                "whisker_fan_side",
                DataType::String,
                |h| h.map(|h| h.side.as_str()),
            ),
            self.column::<Float64Chunked, f64>(
                ear,
                nose,
                "whisker_fan_distance_m",
                DataType::Float64,
                |h| h.map(|h| h.distance),
            ),
        ]
    }

    fn column<Out, V>(
        &self,
        ear: &CoordinateColumns,
        nose: &CoordinateColumns,
        name: &'static str,
        dtype: DataType,
        pick: fn(Option<FanHit>) -> Option<V>,
    ) -> Expr
    where
        Out: FromIterator<Option<V>> + IntoColumn,
        V: 'static,
    {
        let spec = self.clone();
        map_multiple(
            move |columns: &mut [Column]| {
                let (ex, ey) = (columns[0].f64()?, columns[1].f64()?);
                let (nx, ny) = (columns[2].f64()?, columns[3].f64()?);
                let out: Out = ex
                    .into_iter()
                    .zip(ey)
                    .zip(nx)
                    .zip(ny)
                    .map(|(((ex, ey), nx), ny)| match (ex, ey, nx, ny) {
                        (Some(ex), Some(ey), Some(nx), Some(ny)) => {
                            pick(spec.hit_at((ex, ey), (nx, ny)))
                        }
                        _ => pick(None),
                    })
                    .collect();
                Ok(out.into_column())
            },
            [col(&ear.x), col(&ear.y), col(&nose.x), col(&nose.y)],
            move |_schema: &Schema, _fields: &[Field]| Ok(Field::new(name.into(), dtype.clone())),
        )
    }
}

/// Lateral whisker-fan detector for an object outline (Neo/Imen NORT).
/// Other datasets keep the ear-ray [`WhiskerHeuristic`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiskerFanHeuristic {
    pub spec: WhiskerFanSpec,
    /// Head base (centre ear) columns.
    pub ear_coords: CoordinateColumns,
    pub nose_coords: CoordinateColumns,
    /// Result column; also the prefix of the diagnostic columns.
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_column: Option<String>,
    /// Fan computed once per recording for a variant group (`None` = per-frame fan).
    #[serde(skip)]
    pub shared:
        Option<std::sync::Arc<crate::heuristic::combined::exploration::SharedObjectGeometry>>,
}

impl WhiskerFanHeuristic {
    pub fn side_col(&self) -> String {
        diag_column_name(&self.name, "side")
    }

    pub fn distance_col(&self) -> String {
        diag_column_name(&self.name, "nearest_hit_distance_m")
    }
}

impl Heuristic for WhiskerFanHeuristic {
    fn name(&self) -> &str {
        &self.name
    }

    fn result_column(&self) -> &str {
        self.result_column.as_deref().unwrap_or(self.name.as_str())
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let [hit, side, distance] = self
            .shared
            .as_deref()
            .and_then(|s| s.fan_exprs(&self.spec.params))
            .unwrap_or_else(|| self.spec.hit_exprs(&self.ear_coords, &self.nose_coords));
        lf.with_columns([
            hit.alias(self.result_column()),
            side.alias(self.side_col()),
            distance.alias(self.distance_col()),
        ])
    }
}
