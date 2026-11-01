use bekiper_core::types::CoordinateColumns;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::polygon::Polygon;
use bekiper_perimeter::ray_offset::RayOffsetFilter;
use polars::prelude::*;

#[test]
fn ray_offset_filter_new() {
    let filter = RayOffsetFilter::new(Circle::new(5.0, 0.0, 1.0), 45.0);
    assert!((filter.max_angle_rad - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
}

#[test]
fn ray_offset_filter_stores_shape() {
    let c = Circle::new(10.0, 10.0, 2.0);
    let filter = RayOffsetFilter::new(c, 30.0);
    assert!((filter.shape.center_x - 10.0).abs() < 1e-10);
    assert!((filter.shape.radius - 2.0).abs() < 1e-10);
}

#[test]
fn ray_offset_filter_honors_angle_tolerance() {
    let shape = Polygon::new(vec![(10.0, 10.0), (12.0, 10.0), (12.0, 12.0), (10.0, 12.0)]);
    let filter = RayOffsetFilter::new(shape, 20.0);
    let origin = CoordinateColumns::new("ox", "oy");
    let direction = CoordinateColumns::new("dx", "dy");

    let df = df! {
        "ox" => &[0.0, 0.0],
        "oy" => &[0.0, 0.0],
        "dx" => &[1.0, 2.0],
        "dy" => &[0.0, 2.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(filter.filter_expr(&origin, &direction).alias("keep"))
        .collect()
        .unwrap();

    let keep = result.column("keep").unwrap().bool().unwrap();
    assert!(!keep.get(0).unwrap());
    assert!(keep.get(1).unwrap());
}
