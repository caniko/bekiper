use bekiper_core::types::CoordinateColumns;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;
use polars::prelude::*;

#[test]
fn perimeter_new() {
    let p = Perimeter::new("object1", Circle::new(0.0, 0.0, 1.0));
    assert_eq!(p.label, "object1");
}

#[test]
fn perimeter_confinement_mask() {
    let p = Perimeter::new("obj", Circle::new(5.0, 5.0, 2.0));
    let coords = CoordinateColumns::new("x", "y");

    let df = df! {
        "x" => &[5.0, 0.0, 6.0],
        "y" => &[5.0, 0.0, 5.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(p.confinement_mask(&coords).alias("inside"))
        .collect()
        .unwrap();

    let inside = result.column("inside").unwrap().bool().unwrap();
    assert!(inside.get(0).unwrap()); // center
    assert!(!inside.get(1).unwrap()); // far away
    assert!(inside.get(2).unwrap()); // within radius
}

#[test]
fn perimeter_filter_confined() {
    let p = Perimeter::new("obj", Circle::new(0.0, 0.0, 1.0));
    let coords = CoordinateColumns::new("x", "y");

    let df = df! {
        "x" => &[0.0, 5.0, 0.5],
        "y" => &[0.0, 5.0, 0.5],
    }
    .unwrap();

    let result = p.filter_confined(df.lazy(), &coords).collect().unwrap();
    assert_eq!(result.height(), 2); // only 2 points inside
}

#[test]
fn perimeter_expanded() {
    let p = Perimeter::new("obj", Circle::new(0.0, 0.0, 1.0));
    let expanded = p.expanded(0.5);
    assert_eq!(expanded.label, "obj_expanded_0.5");
    assert!((expanded.shape.radius - 1.5).abs() < 1e-10);
}

#[test]
fn perimeter_null_outside_keeps_frames() {
    let p = Perimeter::new("arena", Circle::new(0.0, 0.0, 1.0));
    let coords = CoordinateColumns::new("x", "y");
    let df = df! {
        "x" => [Some(0.0), Some(1.0), Some(2.0), None],
        "y" => [0.0, 0.0, 0.0, 0.0],
        "v" => [1.0, 2.0, 3.0, 4.0],
    }
    .unwrap();

    let out = p
        .null_outside(df.clone().lazy(), &coords, &["v".to_string()])
        .collect()
        .unwrap();
    assert_eq!(out.height(), 4);
    // The boundary is inside; a missing reference counts as outside.
    assert_eq!(
        out.column("v")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        vec![Some(1.0), Some(2.0), None, None]
    );
    // Nothing to null leaves the frame untouched.
    let same = p
        .null_outside(df.clone().lazy(), &coords, &[])
        .collect()
        .unwrap();
    assert!(same.equals_missing(&df));
}

#[test]
fn perimeter_ray_intersection_mask() {
    let p = Perimeter::new("obj", Circle::new(5.0, 0.0, 1.0));
    let origin = CoordinateColumns::new("x", "y");
    let direction = CoordinateColumns::new("dx", "dy");
    let df = df! {
        "x" => [Some(0.0), Some(0.0), None],
        "y" => [0.0, 0.0, 0.0],
        "dx" => [1.0, -1.0, 1.0],
        "dy" => [0.0, 0.0, 0.0],
    }
    .unwrap();
    let out = df
        .lazy()
        .select([p.ray_intersection_mask(&origin, &direction).alias("hit")])
        .collect()
        .unwrap();
    assert_eq!(
        out.column("hit")
            .unwrap()
            .bool()
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        vec![Some(true), Some(false), Some(false)]
    );
}
