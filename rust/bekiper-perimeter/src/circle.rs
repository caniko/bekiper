use serde::{Deserialize, Serialize};

use bekiper_core::shape::{ClosestPoint, Expandable, RayIntersectable, RayVisibility, Shape};
use bekiper_core::types::BoundingBox;
use bekiper_math::vector::{GEOMETRY_EPSILON, angle_between};

/// A circle defined by center and radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle {
    pub center_x: f64,
    pub center_y: f64,
    pub radius: f64,
}

impl Circle {
    pub fn new(center_x: f64, center_y: f64, radius: f64) -> Self {
        Self {
            center_x,
            center_y,
            radius,
        }
    }
}

impl Shape for Circle {
    fn inspection_perimeter(&self, label: &str) -> Option<bekiper_core::inspection::PerimeterSpec> {
        Some(bekiper_core::inspection::PerimeterSpec::Circle {
            label: label.into(),
            center_x: self.center_x,
            center_y: self.center_y,
            radius: self.radius,
        })
    }
    fn contains(&self, x: f64, y: f64) -> bool {
        let dx = x - self.center_x;
        let dy = y - self.center_y;
        dx * dx + dy * dy <= self.radius * self.radius
    }

    fn bounding_box(&self) -> BoundingBox {
        BoundingBox {
            min_x: self.center_x - self.radius,
            min_y: self.center_y - self.radius,
            max_x: self.center_x + self.radius,
            max_y: self.center_y + self.radius,
        }
    }
}

impl ClosestPoint for Circle {
    fn closest_point(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x - self.center_x;
        let dy = y - self.center_y;
        let mag = (dx * dx + dy * dy).sqrt();
        if mag <= GEOMETRY_EPSILON {
            (self.center_x + self.radius, self.center_y)
        } else {
            (
                self.center_x + self.radius * dx / mag,
                self.center_y + self.radius * dy / mag,
            )
        }
    }
}

impl Expandable for Circle {
    type Output = Circle;

    fn expand(&self, distance: f64) -> Circle {
        Circle {
            center_x: self.center_x,
            center_y: self.center_y,
            radius: self.radius + distance,
        }
    }
}

impl RayIntersectable for Circle {
    fn ray_visibility(&self, origin: (f64, f64), direction: (f64, f64)) -> Option<RayVisibility> {
        let (ox, oy) = origin;
        let (dx, dy) = direction;
        let dir_mag = (dx * dx + dy * dy).sqrt();
        if dir_mag <= GEOMETRY_EPSILON {
            return None;
        }
        let heading = (dx / dir_mag, dy / dir_mag);

        if let Some((_, hit_point)) =
            first_ray_circle_hit(origin, heading, (self.center_x, self.center_y), self.radius)
        {
            return Some(RayVisibility {
                min_angle_radians: 0.0,
                target_point: hit_point,
            });
        }

        let to_center = (self.center_x - ox, self.center_y - oy);
        let center_dist = (to_center.0 * to_center.0 + to_center.1 * to_center.1).sqrt();
        // Every ray from inside the circle hits it above; only an origin rounded
        // onto the boundary while heading outward gets here, and it sees nothing.
        if center_dist <= self.radius {
            return None;
        }

        let to_center_unit = (to_center.0 / center_dist, to_center.1 / center_dist);
        let center_angle = angle_between(heading, to_center_unit);
        let tangent_half_angle = (self.radius / center_dist).clamp(-1.0, 1.0).asin();
        let angle_offset = (center_angle - tangent_half_angle).max(0.0);

        let center_bearing = to_center.1.atan2(to_center.0);
        let heading_bearing = heading.1.atan2(heading.0);
        let signed_delta = wrap_angle_radians(heading_bearing - center_bearing);
        let tangent_bearing = center_bearing
            + if signed_delta >= 0.0 {
                tangent_half_angle
            } else {
                -tangent_half_angle
            };
        // The tangent point itself: re-intersecting the tangent ray loses it to
        // rounding once coordinates reach pixel scale.
        let tangent_length = (center_dist * center_dist - self.radius * self.radius)
            .max(0.0)
            .sqrt();
        let target_point = (
            ox + tangent_length * tangent_bearing.cos(),
            oy + tangent_length * tangent_bearing.sin(),
        );

        Some(RayVisibility {
            min_angle_radians: angle_offset,
            target_point,
        })
    }
}

fn first_ray_circle_hit(
    origin: (f64, f64),
    direction: (f64, f64),
    center: (f64, f64),
    radius: f64,
) -> Option<(f64, (f64, f64))> {
    let (ox, oy) = origin;
    let (dx, dy) = direction;
    let fx = ox - center.0;
    let fy = oy - center.1;

    let a = dx * dx + dy * dy;
    let b = 2.0 * (fx * dx + fy * dy);
    let c = fx * fx + fy * fy - radius * radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < -GEOMETRY_EPSILON {
        return None;
    }

    let sqrt_disc = discriminant.max(0.0).sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    let t = [t1, t2]
        .into_iter()
        .filter(|value| *value >= 0.0)
        .min_by(|a, b| a.total_cmp(b))?;
    Some((t, (ox + t * dx, oy + t * dy)))
}

fn wrap_angle_radians(angle: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    (angle + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_contains() {
        let c = Circle::new(0.0, 0.0, 1.0);
        assert!(c.contains(0.0, 0.0));
        assert!(c.contains(0.5, 0.5));
        assert!(!c.contains(1.5, 0.0));
    }

    #[test]
    fn test_circle_expand() {
        let c = Circle::new(0.0, 0.0, 1.0);
        let expanded = c.expand(0.5);
        assert!((expanded.radius - 1.5).abs() < f64::EPSILON);
        assert!(expanded.contains(1.3, 0.0));
    }

    #[test]
    fn test_circle_ray_hit() {
        let c = Circle::new(5.0, 0.0, 1.0);
        assert!(c.ray_intersects((0.0, 0.0), (1.0, 0.0)));
    }

    #[test]
    fn test_circle_ray_visibility_reports_first_hit_point() {
        let c = Circle::new(5.0, 0.0, 1.0);
        let visibility = c.ray_visibility((0.0, 0.0), (1.0, 0.0)).unwrap();
        assert!(visibility.min_angle_radians.abs() < 1e-10);
        assert!((visibility.target_point.0 - 4.0).abs() < 1e-10);
        assert!(visibility.target_point.1.abs() < 1e-10);
    }

    #[test]
    fn test_circle_ray_miss() {
        let c = Circle::new(5.0, 5.0, 1.0);
        assert!(!c.ray_intersects((0.0, 0.0), (1.0, 0.0)));
    }

    #[test]
    fn test_circle_ray_visibility_reports_tangent_offset() {
        let c = Circle::new(1.0, 0.5, 0.2);
        let visibility = c.ray_visibility((0.0, 0.0), (1.0, 0.0)).unwrap();
        let center_dist = (1.0_f64.powi(2) + 0.5_f64.powi(2)).sqrt();
        let center_angle = (0.5_f64).atan2(1.0_f64).abs();
        let tangent_half_angle = (0.2 / center_dist).asin();
        let expected = center_angle - tangent_half_angle;
        assert!((visibility.min_angle_radians - expected).abs() < 1e-10);
        let dx = visibility.target_point.0 - c.center_x;
        let dy = visibility.target_point.1 - c.center_y;
        assert!(((dx * dx + dy * dy).sqrt() - c.radius).abs() < 1e-10);
    }

    #[test]
    fn test_circle_cone_intersection_accepts_near_miss() {
        let c = Circle::new(0.4, 0.12, 0.05);
        assert!(c.cone_intersects((0.0, 0.0), (1.0, 0.0), 10.0));
    }

    #[test]
    fn test_circle_tangent_target_survives_pixel_scale_rounding() {
        // Re-intersecting this tangent ray used to miss by rounding, dropping
        // the visibility of a near-miss heading at pixel-scale coordinates.
        let c = Circle::new(-751.3939864483543, -672.2161249131393, 210.97419600116575);
        let heading = (-0.44304004935295904, -0.8965018207841675);
        assert!(!c.ray_intersects((0.0, 0.0), heading));
        let visibility = c.ray_visibility((0.0, 0.0), heading).unwrap();
        assert!(visibility.min_angle_radians > 0.0);
        let (tx, ty) = visibility.target_point;
        let on_circle = ((tx - c.center_x).hypot(ty - c.center_y) - c.radius).abs();
        assert!(on_circle < 1e-9 * c.radius, "{on_circle}");
        // The target is the tangent point: its radius is perpendicular to the sight line.
        let dot = tx * (tx - c.center_x) + ty * (ty - c.center_y);
        assert!(dot.abs() < 1e-6 * c.radius * c.radius, "{dot}");
    }

    #[test]
    fn test_circle_ray_visibility_degenerate_headings() {
        let c = Circle::new(0.0, 0.0, 1.0);
        assert!(c.ray_visibility((5.0, 0.0), (0.0, 0.0)).is_none());
        // From inside, every heading sees the boundary straight ahead.
        let inside = c.ray_visibility((0.5, 0.0), (0.0, 1.0)).unwrap();
        assert_eq!(inside.min_angle_radians, 0.0);
        assert!((inside.target_point.1 - 0.75_f64.sqrt()).abs() < 1e-12);
        // An origin rounded onto the boundary that heads outward sees nothing.
        let edge: (f64, f64) = (0.664372225255073, 0.7474018639992964);
        assert_eq!((edge.0 * edge.0 + edge.1 * edge.1).sqrt(), 1.0);
        assert!(c.ray_visibility(edge, edge).is_none());
    }

    #[test]
    fn test_circle_closest_point_and_inspection() {
        let c = Circle::new(1.0, 2.0, 0.5);
        assert_eq!(c.closest_point(4.0, 2.0), (1.5, 2.0));
        assert_eq!(c.closest_point(1.0, 1.0), (1.0, 1.5));
        // The centre is equidistant from the boundary; pick +x deterministically.
        assert_eq!(c.closest_point(1.0, 2.0), (1.5, 2.0));
        assert!(c.contains(1.5, 2.0), "boundary is inside");
        assert_eq!(
            c.bounding_box(),
            BoundingBox {
                min_x: 0.5,
                min_y: 1.5,
                max_x: 1.5,
                max_y: 2.5
            }
        );
        assert!(matches!(
            c.inspection_perimeter("obj"),
            Some(bekiper_core::inspection::PerimeterSpec::Circle { radius, .. }) if radius == 0.5
        ));
    }

    #[test]
    fn test_circle_cone_intersection_rejects_tight_cone() {
        let c = Circle::new(0.4, 0.12, 0.05);
        assert!(!c.cone_intersects((0.0, 0.0), (1.0, 0.0), 5.0));
    }
}
