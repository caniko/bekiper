use polars::prelude::*;

use crate::types::BoundingBox;

const RAY_ANGLE_EPSILON_RAD: f64 = 1e-9;

type Heading = ((f64, f64), (f64, f64));

/// Decode the four nullable input columns once for every directional metric.
/// Each caller retains its own missing-value policy (false mask or null value).
fn heading_expr<Output, Value>(
    names: [&str; 4],
    name: &'static str,
    dtype: DataType,
    compute: impl Fn(Option<Heading>) -> Value + Send + Sync + 'static,
) -> Expr
where
    Output: FromIterator<Value> + IntoColumn,
{
    map_multiple(
        move |columns: &mut [Column]| {
            let ox = columns[0].f64()?;
            let oy = columns[1].f64()?;
            let dx = columns[2].f64()?;
            let dy = columns[3].f64()?;
            let out: Output = ox
                .into_iter()
                .zip(oy)
                .zip(dx)
                .zip(dy)
                .map(|(((ox, oy), dx), dy)| match (ox, oy, dx, dy) {
                    (Some(ox), Some(oy), Some(dx), Some(dy)) => Some(((ox, oy), (dx, dy))),
                    _ => None,
                })
                .map(&compute)
                .collect();
            Ok(out.into_column())
        },
        names.map(col),
        move |_schema: &Schema, _fields: &[Field]| Ok(Field::new(name.into(), dtype.clone())),
    )
}

#[cfg(test)]
mod heading_tests {
    use super::*;

    #[derive(Clone)]
    struct Probe;

    impl Shape for Probe {
        fn contains(&self, x: f64, _y: f64) -> bool {
            x >= 9.0
        }
        fn bounding_box(&self) -> BoundingBox {
            BoundingBox {
                min_x: 0.0,
                min_y: 0.0,
                max_x: 10.0,
                max_y: 10.0,
            }
        }
    }

    impl RayIntersectable for Probe {
        fn ray_visibility(
            &self,
            origin: (f64, f64),
            direction: (f64, f64),
        ) -> Option<RayVisibility> {
            Some(RayVisibility {
                min_angle_radians: origin.0 / 2.0,
                target_point: (origin.0 + direction.0, origin.1 + direction.1),
            })
        }
    }

    #[test]
    fn directional_columns_keep_row_order_null_masks_and_numeric_missingness() {
        let frame = df! {
            "ox" => [Some(0.0), None, Some(0.0), Some(0.0), Some(0.0), Some(2.0)],
            "oy" => [Some(0.0), Some(0.0), None, Some(0.0), Some(0.0), Some(3.0)],
            "dx" => [Some(1.0), Some(1.0), Some(1.0), None, Some(1.0), Some(4.0)],
            "dy" => [Some(0.0), Some(0.0), Some(0.0), Some(0.0), None, Some(5.0)],
        }
        .unwrap();
        let expressions = vec![
            Probe.ray_filter_expr("ox", "oy", "dx", "dy").alias("ray"),
            Probe
                .cone_filter_expr("ox", "oy", "dx", "dy", 90.0)
                .alias("cone"),
            Probe
                .ray_min_angle_deg_expr("ox", "oy", "dx", "dy")
                .alias("angle"),
            Probe
                .ray_target_x_expr("ox", "oy", "dx", "dy")
                .alias("target_x"),
            Probe
                .ray_target_y_expr("ox", "oy", "dx", "dy")
                .alias("target_y"),
        ];
        let out = frame
            .clone()
            .lazy()
            .select(expressions.clone())
            .collect()
            .unwrap();
        assert_eq!(
            out["ray"].bool().unwrap().into_iter().collect::<Vec<_>>(),
            vec![
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(false)
            ]
        );
        assert_eq!(
            out["cone"].bool().unwrap().into_iter().collect::<Vec<_>>(),
            vec![
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(true)
            ]
        );
        for (name, expected) in [
            (
                "angle",
                vec![
                    Some(0.0),
                    None,
                    None,
                    None,
                    None,
                    Some(1.0_f64.to_degrees()),
                ],
            ),
            (
                "target_x",
                vec![Some(1.0), None, None, None, None, Some(6.0)],
            ),
            (
                "target_y",
                vec![Some(0.0), None, None, None, None, Some(8.0)],
            ),
        ] {
            assert_eq!(
                out[name].f64().unwrap().into_iter().collect::<Vec<_>>(),
                expected
            );
        }
        let empty = frame.clear().lazy().select(expressions).collect().unwrap();
        assert_eq!(empty.shape(), (0, 5));
        assert_eq!(empty["ray"].dtype(), &DataType::Boolean);
        assert_eq!(empty["angle"].dtype(), &DataType::Float64);
    }

    impl ClosestPoint for Probe {
        fn closest_point(&self, _x: f64, _y: f64) -> (f64, f64) {
            (1.0, 0.0)
        }
    }

    #[test]
    fn nearest_point_heading_measures_the_angle_to_the_closest_boundary_point() {
        let angle = |origin, direction| {
            Probe
                .nearest_point_heading(origin, direction)
                .map(|v| v.min_angle_radians.to_degrees())
        };
        assert_eq!(angle((0.0, 0.0), (1.0, 0.0)), Some(0.0));
        assert!((angle((0.0, 0.0), (0.0, -1.0)).unwrap() - 90.0).abs() < 1e-12);
        assert!((angle((0.0, 0.0), (-1.0, 0.0)).unwrap() - 180.0).abs() < 1e-12);
        assert_eq!(angle((0.0, 0.0), (0.0, 0.0)), None);
        // On the boundary or inside the shape there is no direction to miss.
        assert_eq!(angle((1.0, 0.0), (-1.0, 0.0)), Some(0.0));
        assert_eq!(angle((9.5, 0.0), (1.0, 0.0)), Some(0.0));

        let frame = df! {
            "ox" => [Some(0.0), None],
            "oy" => [Some(0.0), Some(0.0)],
            "dx" => [Some(0.0), Some(1.0)],
            "dy" => [Some(1.0), Some(0.0)],
        }
        .unwrap();
        let [deg, tx, ty] = Probe.nearest_point_heading_exprs("ox", "oy", "dx", "dy");
        let out = frame
            .lazy()
            .select([deg.alias("deg"), tx.alias("tx"), ty.alias("ty")])
            .collect()
            .unwrap();
        let column = |name: &str| out[name].f64().unwrap().into_iter().collect::<Vec<_>>();
        assert!((column("deg")[0].unwrap() - 90.0).abs() < 1e-12);
        assert_eq!(column("deg")[1], None);
        assert_eq!(column("tx"), vec![Some(1.0), None]);
        assert_eq!(column("ty"), vec![Some(0.0), None]);
    }

    #[test]
    fn default_confinement_maps_missing_points_to_outside() {
        assert!(Probe.inspection_perimeter("probe").is_none());
        assert_eq!(Probe.bounding_box().max_x, 10.0);
        let frame = df! {
            "x" => [Some(1.0), None],
            "y" => [Some(1.0), Some(1.0)],
        }
        .unwrap();
        let out = frame
            .lazy()
            .select([Probe.confinement_expr("x", "y").alias("inside")])
            .collect()
            .unwrap();
        // The probe contains nothing; a missing coordinate is never inside.
        assert_eq!(
            out["inside"]
                .bool()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![Some(false), Some(false)]
        );
    }
}

/// Core geometric trait. Monomorphized per concrete shape —
/// `Circle::contains()` compiles to a radius check,
/// `Polygon::contains()` compiles to ray-casting.
pub trait Shape: Send + Sync {
    /// Exact evaluated geometry for inspection (including expansions/erosions).
    /// Custom shapes can omit this while still exporting their evaluated mask.
    fn inspection_perimeter(&self, _label: &str) -> Option<crate::inspection::PerimeterSpec> {
        None
    }

    /// Test whether a single point is inside this shape.
    fn contains(&self, x: f64, y: f64) -> bool;

    /// Axis-aligned bounding box for fast spatial pre-filtering.
    fn bounding_box(&self) -> BoundingBox;

    /// Polars boolean expression: true for rows where (x_col, y_col) is inside this shape.
    ///
    /// Default implementation applies `contains` row-wise via `map_multiple`.
    /// Concrete shapes can override with vectorized implementations.
    fn confinement_expr(&self, x_col: &str, y_col: &str) -> Expr
    where
        Self: Clone + 'static,
    {
        let shape = self.clone();
        let x = x_col.to_string();
        let y = y_col.to_string();

        map_multiple(
            move |columns: &mut [Column]| {
                let xs = columns[0].f64()?;
                let ys = columns[1].f64()?;

                let out: BooleanChunked = xs
                    .into_iter()
                    .zip(ys)
                    .map(|(x, y)| match (x, y) {
                        (Some(x), Some(y)) => Some(shape.contains(x, y)),
                        _ => Some(false),
                    })
                    .collect();

                Ok(out.into_column())
            },
            [col(&x), col(&y)],
            |_schema: &Schema, _fields: &[Field]| {
                Ok(Field::new("confinement".into(), DataType::Boolean))
            },
        )
    }
}

/// Shapes that can be offset (expanded or contracted) by a distance.
pub trait Expandable: Shape {
    type Output: Shape;

    /// Create a new shape expanded outward by `distance` meters.
    fn expand(&self, distance: f64) -> Self::Output;
}

/// Shapes that can return the closest boundary point to a coordinate.
pub trait ClosestPoint: Shape {
    /// Return the closest point on the shape boundary to `(x, y)`.
    fn closest_point(&self, x: f64, y: f64) -> (f64, f64);

    /// Angle between `direction` and the vector from `origin` to the nearest
    /// boundary point, with that point. Zero when `origin` is inside or on
    /// the boundary; `None` for a zero heading.
    fn nearest_point_heading(
        &self,
        origin: (f64, f64),
        direction: (f64, f64),
    ) -> Option<RayVisibility> {
        let (dx, dy) = direction;
        if dx == 0.0 && dy == 0.0 {
            return None;
        }
        let target_point = self.closest_point(origin.0, origin.1);
        let (tx, ty) = (target_point.0 - origin.0, target_point.1 - origin.1);
        let min_angle_radians = if self.contains(origin.0, origin.1) || (tx == 0.0 && ty == 0.0) {
            0.0
        } else {
            (dx * ty - dy * tx).abs().atan2(dx * tx + dy * ty)
        };
        Some(RayVisibility {
            min_angle_radians,
            target_point,
        })
    }

    /// Polars float expressions `[angle_deg, target_x, target_y]` of
    /// [`Self::nearest_point_heading`]. Null when no valid heading exists.
    fn nearest_point_heading_exprs(
        &self,
        origin_x: &str,
        origin_y: &str,
        dir_x: &str,
        dir_y: &str,
    ) -> [Expr; 3]
    where
        Self: Clone + 'static,
    {
        let names = [origin_x, origin_y, dir_x, dir_y];
        let field = |name: &'static str, pick: fn(RayVisibility) -> f64| {
            let shape = self.clone();
            heading_expr::<Float64Chunked, _>(names, name, DataType::Float64, move |heading| {
                heading
                    .and_then(|(origin, direction)| shape.nearest_point_heading(origin, direction))
                    .map(pick)
            })
        };
        [
            field("nearest_angle_deg", |v| v.min_angle_radians.to_degrees()),
            field("nearest_target_x", |v| v.target_point.0),
            field("nearest_target_y", |v| v.target_point.1),
        ]
    }
}

/// Continuous ray-to-shape visibility result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayVisibility {
    /// Minimum angular offset from the heading vector to any boundary point.
    pub min_angle_radians: f64,
    /// Boundary point that attains `min_angle_radians`.
    pub target_point: (f64, f64),
}

/// Shapes that support ray intersection tests for directional filtering.
pub trait RayIntersectable: Shape {
    /// Return the minimum angular offset from the heading vector to any
    /// boundary point together with the matching target point.
    fn ray_visibility(&self, origin: (f64, f64), direction: (f64, f64)) -> Option<RayVisibility>;

    /// Test whether a ray from `origin` in `direction` intersects this shape.
    fn ray_intersects(&self, origin: (f64, f64), direction: (f64, f64)) -> bool {
        self.ray_visibility(origin, direction)
            .is_some_and(|visibility| visibility.min_angle_radians <= RAY_ANGLE_EPSILON_RAD)
    }

    /// Test whether a cone centred on `direction` with the given half-angle
    /// intersects this shape. The default derives the boolean from the
    /// continuous visibility metric so diagnostics and thresholding stay aligned.
    fn cone_intersects(
        &self,
        origin: (f64, f64),
        direction: (f64, f64),
        half_angle_degrees: f64,
    ) -> bool {
        self.ray_visibility(origin, direction)
            .is_some_and(|visibility| {
                visibility.min_angle_radians
                    <= half_angle_degrees.to_radians() + RAY_ANGLE_EPSILON_RAD
            })
    }

    /// Polars boolean expression: true where the ray from origin columns
    /// in direction columns intersects this shape.
    fn ray_filter_expr(&self, origin_x: &str, origin_y: &str, dir_x: &str, dir_y: &str) -> Expr
    where
        Self: Clone + 'static,
    {
        let shape = self.clone();
        heading_expr::<BooleanChunked, _>(
            [origin_x, origin_y, dir_x, dir_y],
            "ray_filter",
            DataType::Boolean,
            move |heading| {
                Some(
                    heading
                        .is_some_and(|(origin, direction)| shape.ray_intersects(origin, direction)),
                )
            },
        )
    }

    /// Polars boolean expression: true where a cone from the origin columns
    /// centred on the direction columns intersects this shape.
    fn cone_filter_expr(
        &self,
        origin_x: &str,
        origin_y: &str,
        dir_x: &str,
        dir_y: &str,
        half_angle_degrees: f64,
    ) -> Expr
    where
        Self: Clone + 'static,
    {
        let shape = self.clone();
        heading_expr::<BooleanChunked, _>(
            [origin_x, origin_y, dir_x, dir_y],
            "cone_filter",
            DataType::Boolean,
            move |heading| {
                Some(heading.is_some_and(|(origin, direction)| {
                    shape.cone_intersects(origin, direction, half_angle_degrees)
                }))
            },
        )
    }

    /// Polars float expression: minimum angular offset in degrees from the
    /// heading vector to the shape boundary. Null when no valid heading exists.
    fn ray_min_angle_deg_expr(
        &self,
        origin_x: &str,
        origin_y: &str,
        dir_x: &str,
        dir_y: &str,
    ) -> Expr
    where
        Self: Clone + 'static,
    {
        let shape = self.clone();
        heading_expr::<Float64Chunked, _>(
            [origin_x, origin_y, dir_x, dir_y],
            "ray_min_angle_deg",
            DataType::Float64,
            move |heading| {
                heading
                    .and_then(|(origin, direction)| shape.ray_visibility(origin, direction))
                    .map(|visibility| visibility.min_angle_radians.to_degrees())
            },
        )
    }

    /// Polars float expression: target boundary X coordinate chosen by the
    /// visibility solve. Null when no valid heading exists.
    fn ray_target_x_expr(&self, origin_x: &str, origin_y: &str, dir_x: &str, dir_y: &str) -> Expr
    where
        Self: Clone + 'static,
    {
        let shape = self.clone();
        heading_expr::<Float64Chunked, _>(
            [origin_x, origin_y, dir_x, dir_y],
            "ray_target_x",
            DataType::Float64,
            move |heading| {
                heading
                    .and_then(|(origin, direction)| shape.ray_visibility(origin, direction))
                    .map(|visibility| visibility.target_point.0)
            },
        )
    }

    /// Polars float expression: target boundary Y coordinate chosen by the
    /// visibility solve. Null when no valid heading exists.
    fn ray_target_y_expr(&self, origin_x: &str, origin_y: &str, dir_x: &str, dir_y: &str) -> Expr
    where
        Self: Clone + 'static,
    {
        let shape = self.clone();
        heading_expr::<Float64Chunked, _>(
            [origin_x, origin_y, dir_x, dir_y],
            "ray_target_y",
            DataType::Float64,
            move |heading| {
                heading
                    .and_then(|(origin, direction)| shape.ray_visibility(origin, direction))
                    .map(|visibility| visibility.target_point.1)
            },
        )
    }
}
