use bekiper_core::inspection::PerimeterSpec;
use bekiper_core::shape::{ClosestPoint, Expandable, RayIntersectable, Shape};
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::object_shape::ObjectShape;
use bekiper_perimeter::polygon::Polygon;
use bekiper_perimeter::rectangle::Rectangle;
use bekiper_perimeter::triangle::Triangle;

/// One shape of each kind, all covering the point (1, 1) and spanning [0, 2].
fn shapes() -> [ObjectShape; 4] {
    [
        Circle::new(1.0, 1.0, 1.0).into(),
        Rectangle::new(1.0, 1.0, 2.0, 2.0).into(),
        Polygon::new(vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]).into(),
        Triangle::new((0.0, 0.0), (2.0, 0.0), (1.0, 2.0)).into(),
    ]
}

#[test]
fn every_variant_delegates_geometry() {
    let kinds = ["circle", "rectangle", "polygon", "triangle"];
    for (shape, kind) in shapes().iter().zip(kinds) {
        assert!(shape.contains(1.0, 1.0), "{kind}");
        assert!(!shape.contains(3.0, 3.0), "{kind}");
        let bb = shape.bounding_box();
        assert_eq!(
            (bb.min_x, bb.min_y, bb.max_x, bb.max_y),
            (0.0, 0.0, 2.0, 2.0),
            "{kind}"
        );
        // The bottom edge (y = 0) is closest to a point straight below x = 1.
        assert_eq!(shape.closest_point(1.0, -5.0), (1.0, 0.0), "{kind}");
        let hit = shape.ray_visibility((1.0, -5.0), (0.0, 1.0)).unwrap();
        assert_eq!(hit.min_angle_radians, 0.0, "{kind}");
        assert_eq!(hit.target_point, (1.0, 0.0), "{kind}");
        assert!(
            shape.ray_visibility((1.0, -5.0), (0.0, 0.0)).is_none(),
            "{kind}"
        );
        let spec = shape.inspection_perimeter("obj").unwrap();
        let tag = match spec {
            PerimeterSpec::Circle { .. } => "circle",
            PerimeterSpec::Rectangle { .. } => "rectangle",
            PerimeterSpec::Polygon { .. } => "polygon",
            PerimeterSpec::Triangle { .. } => "triangle",
            PerimeterSpec::RadialMaze { .. } => "radial_maze",
        };
        assert_eq!(tag, kind);
    }
}

#[test]
fn expansion_keeps_circles_and_polygonises_the_rest() {
    let [circle, rectangle, polygon, triangle] = shapes();
    match circle.expand(0.5) {
        ObjectShape::Circle(c) => assert_eq!(c.radius, 1.5),
        other => panic!("circle expanded to {other:?}"),
    }
    for shape in [rectangle, polygon, triangle] {
        let ObjectShape::Polygon(grown) = shape.expand(0.5) else {
            panic!("{shape:?} must expand to a polygon");
        };
        assert!(grown.contains(1.0, -0.49));
        assert!(!grown.contains(1.0, -0.51));
    }
}

#[test]
fn centroids_per_variant() {
    let [circle, rectangle, polygon, triangle] = shapes();
    assert_eq!(circle.centroid(), (1.0, 1.0));
    assert_eq!(rectangle.centroid(), (1.0, 1.0));
    assert_eq!(polygon.centroid(), (1.0, 1.0));
    let (tx, ty) = triangle.centroid();
    assert_eq!(tx, 1.0);
    assert!((ty - 2.0 / 3.0).abs() < 1e-12);
}

#[test]
fn serde_is_tagged_by_type() {
    for shape in shapes() {
        let text = toml::to_string(&shape).unwrap();
        let back: ObjectShape = toml::from_str(&text).unwrap();
        assert_eq!(back.bounding_box(), shape.bounding_box(), "{text}");
    }
    let circle: ObjectShape =
        toml::from_str("type = \"Circle\"\ncenter_x = 0.0\ncenter_y = 0.0\nradius = 2.0").unwrap();
    assert!(matches!(circle, ObjectShape::Circle(c) if c.radius == 2.0));
    assert!(toml::from_str::<ObjectShape>("type = \"Hexagon\"").is_err());
}

#[test]
fn degenerate_triangle_has_no_triangle_inspection() {
    // A repeated closing vertex is dropped, leaving two vertices.
    let t = Triangle::new((0.0, 0.0), (1.0, 0.0), (0.0, 0.0));
    assert_eq!(t.vertices().len(), 2);
    assert!(t.inspection_perimeter("t").is_none());
    let ok = Triangle::new((0.0, 0.0), (2.0, 0.0), (1.0, 2.0));
    assert_eq!(ok.closest_point(1.0, -1.0), (1.0, 0.0));
    assert!(matches!(
        ok.inspection_perimeter("t"),
        Some(PerimeterSpec::Triangle { vertices, .. }) if vertices[0] == (0.0, 0.0)
    ));
}
