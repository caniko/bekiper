use polars::prelude::*;

use bekiper_core::types::CoordinateColumns;
use bekiper_feature::axiom::proximity::ComputeProximity;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;

fn make_df(xs: &[f64], ys: &[f64]) -> DataFrame {
    df! {
        "x" => xs,
        "y" => ys,
    }
    .unwrap()
}

fn coords() -> CoordinateColumns {
    CoordinateColumns::new("x", "y")
}

#[test]
fn proximity_inside_perimeter_is_true() {
    let circle = Circle::new(0.0, 0.0, 10.0);
    let perimeter = Perimeter::new("arena", circle);
    let cp = ComputeProximity::new(perimeter, 0.0);

    // Point at (1, 1) is inside the circle of radius 10
    let df = make_df(&[1.0], &[1.0]);
    let result = df
        .lazy()
        .with_column(cp.is_proximate(&coords()).alias("prox"))
        .collect()
        .unwrap();

    let prox = result.column("prox").unwrap().bool().unwrap();
    assert!(prox.get(0).unwrap());
}

#[test]
fn proximity_outside_far_is_false() {
    let circle = Circle::new(0.0, 0.0, 5.0);
    let perimeter = Perimeter::new("arena", circle);
    let cp = ComputeProximity::new(perimeter, 0.0);

    // Point at (100, 100) is far outside the circle of radius 5
    let df = make_df(&[100.0], &[100.0]);
    let result = df
        .lazy()
        .with_column(cp.is_proximate(&coords()).alias("prox"))
        .collect()
        .unwrap();

    let prox = result.column("prox").unwrap().bool().unwrap();
    assert!(!prox.get(0).unwrap());
}

#[test]
fn proximity_in_expanded_range_is_true() {
    let circle = Circle::new(0.0, 0.0, 5.0);
    let perimeter = Perimeter::new("arena", circle);
    // Point at (7, 0) is outside radius 5, but within max_distance=3 (expanded radius=8)
    let cp = ComputeProximity::new(perimeter, 3.0);

    let df = make_df(&[7.0], &[0.0]);
    let result = df
        .lazy()
        .with_column(cp.is_proximate(&coords()).alias("prox"))
        .collect()
        .unwrap();

    let prox = result.column("prox").unwrap().bool().unwrap();
    assert!(prox.get(0).unwrap());
}

#[test]
fn proximity_apply_adds_column() {
    let circle = Circle::new(0.0, 0.0, 10.0);
    let perimeter = Perimeter::new("arena", circle);
    let cp = ComputeProximity::new(perimeter, 0.0);

    let df = make_df(&[0.0, 50.0, 3.0], &[0.0, 50.0, 4.0]);
    let result = cp.apply(df.lazy(), &coords(), "is_near").collect().unwrap();

    // Verify the column exists and has the right name
    let col = result.column("is_near").unwrap().bool().unwrap();
    assert_eq!(col.len(), 3);
    assert!(col.get(0).unwrap()); // (0,0) inside
    assert!(!col.get(1).unwrap()); // (50,50) outside
    assert!(col.get(2).unwrap()); // (3,4) inside (dist=5 < 10)
}
