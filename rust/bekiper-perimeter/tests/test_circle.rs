use bekiper_core::shape::{Expandable, RayIntersectable, Shape};
use bekiper_perimeter::circle::Circle;

#[test]
fn circle_bounding_box() {
    let c = Circle::new(5.0, 5.0, 2.0);
    let bb = c.bounding_box();
    assert!((bb.min_x - 3.0).abs() < 1e-10);
    assert!((bb.min_y - 3.0).abs() < 1e-10);
    assert!((bb.max_x - 7.0).abs() < 1e-10);
    assert!((bb.max_y - 7.0).abs() < 1e-10);
}

#[test]
fn circle_contains_edge() {
    let c = Circle::new(5.0, 5.0, 2.0);
    // Point exactly on the radius boundary (east)
    assert!(c.contains(7.0, 5.0));
    // Point exactly on the radius boundary (north)
    assert!(c.contains(5.0, 7.0));
}

#[test]
fn circle_expand_preserves_center() {
    let c = Circle::new(3.0, 7.0, 1.0);
    let expanded = c.expand(2.5);
    assert!((expanded.center_x - 3.0).abs() < 1e-10);
    assert!((expanded.center_y - 7.0).abs() < 1e-10);
    assert!((expanded.radius - 3.5).abs() < 1e-10);
}

#[test]
fn circle_ray_tangent() {
    // Circle at (5, 5) with radius 1. A ray from (0, 0) in direction (1, 0)
    // passes at y=0, which is far from the circle. Should miss.
    let c = Circle::new(5.0, 5.0, 1.0);
    assert!(!c.ray_intersects((0.0, 0.0), (1.0, 0.0)));

    // Ray that barely misses: aim just above the circle top edge.
    // Circle at (10, 0) radius 1. Ray from origin going in direction (10, 1.01).
    // At x=10 the ray y ~ 1.01, which is just outside the radius.
    let c2 = Circle::new(10.0, 0.0, 1.0);
    assert!(!c2.ray_intersects((0.0, 0.0), (10.0, 1.01)));
}

#[test]
fn circle_ray_behind_origin() {
    // Circle is entirely behind the ray origin (negative t values).
    let c = Circle::new(-5.0, 0.0, 1.0);
    // Ray fires in +x direction from origin.
    assert!(!c.ray_intersects((0.0, 0.0), (1.0, 0.0)));
}

#[test]
fn circle_serde_roundtrip() {
    let c = Circle::new(1.5, -2.3, 4.0);
    let toml_str = toml::to_string(&c).unwrap();
    let c2: Circle = toml::from_str(&toml_str).unwrap();
    assert!((c2.center_x - 1.5).abs() < 1e-10);
    assert!((c2.center_y - (-2.3)).abs() < 1e-10);
    assert!((c2.radius - 4.0).abs() < 1e-10);
}
