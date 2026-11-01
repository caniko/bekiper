use polars::prelude::*;

use bekiper_core::shape::{ClosestPoint, RayIntersectable};
use bekiper_core::types::CoordinateColumns;
use bekiper_math::vector::angle_between;

/// Filter that checks whether the animal's heading (ray direction)
/// is within a maximum angle offset from the direction toward a target shape.
///
/// Generic over `S` — monomorphized per shape type.
#[derive(Debug, Clone)]
pub struct RayOffsetFilter<S: RayIntersectable + ClosestPoint> {
    pub shape: S,
    pub max_angle_rad: f64,
}

impl<S: RayIntersectable + ClosestPoint + Clone + 'static> RayOffsetFilter<S> {
    pub fn new(shape: S, max_angle_degrees: f64) -> Self {
        Self {
            shape,
            max_angle_rad: max_angle_degrees.to_radians(),
        }
    }

    /// Boolean expression: true where the heading from `origin` toward
    /// `heading_direction` stays within `max_angle_rad` of the nearest edge
    /// point on the shape.
    pub fn filter_expr(
        &self,
        origin: &CoordinateColumns,
        heading_direction: &CoordinateColumns,
    ) -> Expr {
        let shape = self.shape.clone();
        let ox = origin.x.clone();
        let oy = origin.y.clone();
        let dx = heading_direction.x.clone();
        let dy = heading_direction.y.clone();
        let max_angle_rad = self.max_angle_rad;

        map_multiple(
            move |columns: &mut [Column]| {
                let oxs = columns[0].f64()?;
                let oys = columns[1].f64()?;
                let dxs = columns[2].f64()?;
                let dys = columns[3].f64()?;

                let out: BooleanChunked = oxs
                    .into_iter()
                    .zip(oys)
                    .zip(dxs)
                    .zip(dys)
                    .map(|(((ox, oy), dx), dy)| match (ox, oy, dx, dy) {
                        (Some(ox), Some(oy), Some(dx), Some(dy)) => {
                            let heading = (dx - ox, dy - oy);
                            let heading_mag =
                                (heading.0 * heading.0 + heading.1 * heading.1).sqrt();
                            if heading_mag <= f64::EPSILON {
                                Some(false)
                            } else {
                                let heading = (heading.0 / heading_mag, heading.1 / heading_mag);
                                let closest = shape.closest_point(dx, dy);
                                let to_edge = (closest.0 - dx, closest.1 - dy);
                                let to_edge_mag =
                                    (to_edge.0 * to_edge.0 + to_edge.1 * to_edge.1).sqrt();
                                if to_edge_mag <= f64::EPSILON {
                                    Some(true)
                                } else {
                                    let to_edge =
                                        (to_edge.0 / to_edge_mag, to_edge.1 / to_edge_mag);
                                    Some(angle_between(heading, to_edge) <= max_angle_rad)
                                }
                            }
                        }
                        _ => Some(false),
                    })
                    .collect();

                Ok(out.into_column())
            },
            [col(&ox), col(&oy), col(&dx), col(&dy)],
            |_schema: &Schema, _fields: &[Field]| {
                Ok(Field::new("ray_offset".into(), DataType::Boolean))
            },
        )
    }
}
