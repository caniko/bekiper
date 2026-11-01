use bekiper_core::shape::{ClosestPoint, Expandable, RayIntersectable, Shape};
use bekiper_perimeter::rectangle::Rectangle;

#[test]
fn rectangle_contains_center() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    assert!(r.contains(5.0, 5.0));
}

#[test]
fn rectangle_contains_edge() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    // Edge at (3, 4) to (7, 6)
    assert!(r.contains(3.5, 5.0));
}

#[test]
fn rectangle_excludes_outside() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    assert!(!r.contains(0.0, 0.0));
    assert!(!r.contains(8.0, 5.0));
}

#[test]
fn rectangle_bounding_box() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    let bb = r.bounding_box();
    assert!((bb.min_x - 3.0).abs() < 1e-10);
    assert!((bb.min_y - 4.0).abs() < 1e-10);
    assert!((bb.max_x - 7.0).abs() < 1e-10);
    assert!((bb.max_y - 6.0).abs() < 1e-10);
}

#[test]
fn rectangle_expand_grows() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    let expanded = r.expand(1.0);
    let bb = expanded.bounding_box();
    assert!((bb.min_x - 2.0).abs() < 1e-10);
    assert!((bb.min_y - 3.0).abs() < 1e-10);
    assert!((bb.max_x - 8.0).abs() < 1e-10);
    assert!((bb.max_y - 7.0).abs() < 1e-10);
    assert!(expanded.contains(2.0, 5.0));
    assert!(!r.contains(2.0, 5.0));
}

#[test]
fn rectangle_ray_hit() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    // Ray from origin pointing right should hit
    assert!(r.ray_intersects((0.0, 5.0), (1.0, 0.0)));
}

#[test]
fn rectangle_ray_miss() {
    let r = Rectangle::new(5.0, 5.0, 4.0, 2.0);
    // Ray pointing up should miss
    assert!(!r.ray_intersects((0.0, 0.0), (0.0, -1.0)));
}

#[test]
fn rectangle_serde_roundtrip() {
    let r = Rectangle::new(1.0, 2.0, 3.0, 4.0);
    let toml_str = toml::to_string(&r).unwrap();
    let r2: Rectangle = toml::from_str(&toml_str).unwrap();
    assert_eq!(r, r2);
}

#[test]
fn rectangle_closest_point_and_inspection() {
    let r = Rectangle::new(0.0, 0.0, 4.0, 2.0);
    assert_eq!(r.closest_point(5.0, 0.0), (2.0, 0.0));
    assert_eq!(r.closest_point(3.0, 3.0), (2.0, 1.0));
    assert!(matches!(
        r.inspection_perimeter("box"),
        Some(bekiper_core::inspection::PerimeterSpec::Rectangle { width, height, .. })
            if width == 4.0 && height == 2.0
    ));
}
