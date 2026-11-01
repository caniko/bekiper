use bekiper_core::shape::{Expandable, RayIntersectable, Shape};
use bekiper_perimeter::polygon::Polygon;

#[test]
fn polygon_centroid_triangle() {
    let tri = Polygon::new(vec![(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)]);
    let (cx, cy) = tri.centroid();
    assert!((cx - 1.0).abs() < 1e-10);
    assert!((cy - 1.0).abs() < 1e-10);
}

#[test]
fn polygon_edges_count() {
    let sq = Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
    assert_eq!(sq.edges().count(), 4);
}

#[test]
fn polygon_edges_wrap_around() {
    let sq = Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
    let edges: Vec<_> = sq.edges().collect();
    // Last edge should connect the last vertex back to the first vertex.
    let last_edge = edges[3];
    assert!((last_edge.0.0 - 0.0).abs() < 1e-10); // start x = 0.0
    assert!((last_edge.0.1 - 1.0).abs() < 1e-10); // start y = 1.0
    assert!((last_edge.1.0 - 0.0).abs() < 1e-10); // end x = 0.0
    assert!((last_edge.1.1 - 0.0).abs() < 1e-10); // end y = 0.0
}

#[test]
fn polygon_bounding_box() {
    let sq = Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
    let bb = sq.bounding_box();
    assert!((bb.min_x - 0.0).abs() < 1e-10);
    assert!((bb.min_y - 0.0).abs() < 1e-10);
    assert!((bb.max_x - 1.0).abs() < 1e-10);
    assert!((bb.max_y - 1.0).abs() < 1e-10);
}

#[test]
fn polygon_contains_concave() {
    // L-shaped polygon (concave):
    //   (0,0) -> (2,0) -> (2,1) -> (1,1) -> (1,2) -> (0,2)
    let l_shape = Polygon::new(vec![
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (1.0, 2.0),
        (0.0, 2.0),
    ]);
    // Inside the bottom-left part of the L
    assert!(l_shape.contains(0.5, 0.5));
    // Inside the top-left part of the L
    assert!(l_shape.contains(0.5, 1.5));
    // Inside the bottom-right part of the L
    assert!(l_shape.contains(1.5, 0.5));
    // Outside: the concave notch at (1.5, 1.5)
    assert!(!l_shape.contains(1.5, 1.5));
}

#[test]
fn polygon_expand_grows() {
    let tri = Polygon::new(vec![(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)]);
    let (cx, cy) = tri.centroid();
    let expanded = tri.expand(1.0);

    // Every expanded vertex should be farther from centroid than original
    for (orig, exp) in tri.vertices.iter().zip(expanded.vertices.iter()) {
        let orig_dist = ((orig.0 - cx).powi(2) + (orig.1 - cy).powi(2)).sqrt();
        let exp_dist = ((exp.0 - cx).powi(2) + (exp.1 - cy).powi(2)).sqrt();
        assert!(exp_dist > orig_dist);
    }
}

#[test]
fn polygon_ray_parallel() {
    // Unit square [0,1] x [0,1]. Ray parallel to bottom edge but outside (at y = -1).
    let sq = Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
    assert!(!sq.ray_intersects((-1.0, -1.0), (1.0, 0.0)));
}

#[test]
fn polygon_serde_roundtrip() {
    let poly = Polygon::new(vec![(1.0, 2.0), (3.0, 4.0), (5.0, 0.0)]);
    let toml_str = toml::to_string(&poly).unwrap();
    let poly2: Polygon = toml::from_str(&toml_str).unwrap();
    assert_eq!(poly.vertices.len(), poly2.vertices.len());
    for (a, b) in poly.vertices.iter().zip(poly2.vertices.iter()) {
        assert!((a.0 - b.0).abs() < 1e-10);
        assert!((a.1 - b.1).abs() < 1e-10);
    }
}

#[test]
fn polygon_closest_point_skips_zero_length_edges() {
    // A repeated vertex creates a zero-length edge, which reduces to its start.
    let sq = Polygon::new(vec![
        (0.0, 0.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]);
    assert_eq!(sq.closest_point(-1.0, -1.0), (0.0, 0.0));
    assert_eq!(sq.closest_point(0.5, -1.0), (0.5, 0.0));
    // A point on an edge is its own closest point.
    assert_eq!(sq.closest_point(1.0, 0.5), (1.0, 0.5));
}

#[test]
fn polygon_expand_zero_and_degenerate_return_hull() {
    // Interior and collinear points drop out of the convex hull.
    let sq = Polygon::new(vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (1.0, 1.0),
        (0.0, 2.0),
    ]);
    assert_eq!(
        sq.expand(0.0).vertices,
        vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    );
    // Fewer than three distinct points cannot be offset.
    let segment = Polygon::new(vec![(0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0)]);
    assert_eq!(segment.expand(1.0).vertices, vec![(0.0, 0.0), (1.0, 1.0)]);
    assert_eq!(segment.expand(-1.0).vertices, vec![(0.0, 0.0), (1.0, 1.0)]);
}

#[test]
fn polygon_erosion_offsets_edges_and_collapses_past_inradius() {
    let sq = Polygon::new(vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]);
    let eroded = sq.expand(-1.0);
    for (got, want) in eroded
        .vertices
        .iter()
        .zip([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])
    {
        assert!((got.0 - want.0).abs() < 1e-12 && (got.1 - want.1).abs() < 1e-12);
    }
    // Eroding by the inradius or more collapses to the centroid.
    assert_eq!(sq.expand(-2.0).vertices, vec![(2.0, 2.0); 3]);
    assert_eq!(sq.expand(-3.0).vertices, vec![(2.0, 2.0); 3]);
    assert!(!sq.expand(-3.0).contains(0.5, 0.5));
}

#[test]
fn polygon_points_on_edges_and_inspection() {
    let tri = Polygon::new(vec![(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]);
    let bb = tri.bounding_box();
    assert_eq!((bb.min_x, bb.max_y), (0.0, 2.0));
    assert!(matches!(
        tri.inspection_perimeter("tri"),
        Some(bekiper_core::inspection::PerimeterSpec::Polygon { vertices, .. }) if vertices.len() == 3
    ));
    // From inside, the ray reports the boundary straight ahead with zero angle.
    let inside = tri.ray_visibility((0.5, 0.5), (1.0, 0.0)).unwrap();
    assert_eq!(inside.min_angle_radians, 0.0);
    assert!((inside.target_point.0 - 1.5).abs() < 1e-12);
    assert!(tri.ray_visibility((0.5, 0.5), (0.0, 0.0)).is_none());
}
