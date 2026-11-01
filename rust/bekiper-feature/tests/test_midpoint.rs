use polars::prelude::*;

use bekiper_core::types::CoordinateColumns;
use bekiper_feature::midpoint::{midpoint_exprs, midpoint_exprs_from};

#[test]
fn midpoint_of_two_points() {
    let df = df! {
        "x1" => &[0.0, 2.0],
        "y1" => &[0.0, 4.0],
        "x2" => &[4.0, 6.0],
        "y2" => &[4.0, 8.0],
    }
    .unwrap();

    let exprs = midpoint_exprs("x1", "y1", "x2", "y2", "mid_x", "mid_y");
    let mut lf = df.lazy();
    for expr in exprs {
        lf = lf.with_column(expr);
    }
    let result = lf.collect().unwrap();

    let mx = result.column("mid_x").unwrap().f64().unwrap();
    let my = result.column("mid_y").unwrap().f64().unwrap();

    assert!((mx.get(0).unwrap() - 2.0).abs() < 1e-10);
    assert!((my.get(0).unwrap() - 2.0).abs() < 1e-10);
    assert!((mx.get(1).unwrap() - 4.0).abs() < 1e-10);
    assert!((my.get(1).unwrap() - 6.0).abs() < 1e-10);
}

#[test]
fn midpoint_same_point() {
    let df = df! {
        "x1" => &[5.0],
        "y1" => &[5.0],
        "x2" => &[5.0],
        "y2" => &[5.0],
    }
    .unwrap();

    let exprs = midpoint_exprs("x1", "y1", "x2", "y2", "mx", "my");
    let mut lf = df.lazy();
    for expr in exprs {
        lf = lf.with_column(expr);
    }
    let result = lf.collect().unwrap();

    let mx = result.column("mx").unwrap().f64().unwrap();
    let my = result.column("my").unwrap().f64().unwrap();
    assert!((mx.get(0).unwrap() - 5.0).abs() < 1e-10);
    assert!((my.get(0).unwrap() - 5.0).abs() < 1e-10);
}

#[test]
fn midpoint_from_coordinate_pair_sources() {
    let df = df! {
        "left_x" => &[0.0],
        "left_y" => &[2.0],
        "right_x" => &[4.0],
        "right_y" => &[6.0],
    }
    .unwrap();

    let left = CoordinateColumns::new("left_x", "left_y");
    let right = CoordinateColumns::new("right_x", "right_y");
    let exprs = midpoint_exprs_from(&left, &right, "mid_x", "mid_y");
    let mut lf = df.lazy();
    for expr in exprs {
        lf = lf.with_column(expr);
    }
    let result = lf.collect().unwrap();

    let mx = result.column("mid_x").unwrap().f64().unwrap();
    let my = result.column("mid_y").unwrap().f64().unwrap();
    assert!((mx.get(0).unwrap() - 2.0).abs() < 1e-10);
    assert!((my.get(0).unwrap() - 4.0).abs() < 1e-10);
}

#[test]
fn midpoint_from_str_pairs_propagates_missing() {
    let df = df! {
        "a_x" => [Some(0.0), None],
        "a_y" => [Some(0.0), Some(1.0)],
        "b_x" => [Some(2.0), Some(2.0)],
        "b_y" => [Some(-2.0), Some(1.0)],
    }
    .unwrap();
    let out = df
        .lazy()
        .select(midpoint_exprs_from(
            &("a_x", "a_y"),
            &("b_x", "b_y"),
            "m_x",
            "m_y",
        ))
        .collect()
        .unwrap();
    let mx: Vec<_> = out
        .column("m_x")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .collect();
    let my: Vec<_> = out
        .column("m_y")
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(mx, vec![Some(1.0), None]);
    assert_eq!(my, vec![Some(-1.0), Some(1.0)]);
}
