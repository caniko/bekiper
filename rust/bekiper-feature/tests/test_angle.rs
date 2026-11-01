use polars::prelude::*;

use bekiper_core::types::CoordinateColumns;
use bekiper_feature::angle::{
    add_direction_columns, add_direction_columns_from, heading_angle_deg_expr, heading_angle_expr,
    heading_angle_from,
};

#[test]
fn add_direction_columns_computes_diff() {
    let df = df! {
        "from_x" => &[0.0, 1.0],
        "from_y" => &[0.0, 2.0],
        "to_x" => &[3.0, 4.0],
        "to_y" => &[4.0, 6.0],
    }
    .unwrap();

    let result = add_direction_columns(df.lazy(), "from_x", "from_y", "to_x", "to_y", "heading")
        .collect()
        .unwrap();

    let dx = result.column("heading_dx").unwrap().f64().unwrap();
    let dy = result.column("heading_dy").unwrap().f64().unwrap();

    assert!((dx.get(0).unwrap() - 3.0).abs() < 1e-10);
    assert!((dy.get(0).unwrap() - 4.0).abs() < 1e-10);
    assert!((dx.get(1).unwrap() - 3.0).abs() < 1e-10);
    assert!((dy.get(1).unwrap() - 4.0).abs() < 1e-10);
}

#[test]
fn heading_angle_expr_matches_expected_radians() {
    let df = df! {
        "from_x" => &[0.0, 0.0, 0.0, 0.0],
        "from_y" => &[0.0, 0.0, 0.0, 0.0],
        "to_x" => &[1.0, 0.0, -1.0, 0.0],
        "to_y" => &[0.0, 1.0, 0.0, -1.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(heading_angle_expr("from_x", "from_y", "to_x", "to_y").alias("heading"))
        .collect()
        .unwrap();

    let heading = result.column("heading").unwrap().f64().unwrap();
    assert!((heading.get(0).unwrap() - 0.0).abs() < 1e-10);
    assert!((heading.get(1).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((heading.get(2).unwrap() - std::f64::consts::PI).abs() < 1e-10);
    assert!((heading.get(3).unwrap() + std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

#[test]
fn heading_angle_deg_expr_matches_expected_degrees() {
    let df = df! {
        "from_x" => &[0.0, 0.0],
        "from_y" => &[0.0, 0.0],
        "to_x" => &[0.0, -1.0],
        "to_y" => &[1.0, 0.0],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(
            heading_angle_deg_expr("from_x", "from_y", "to_x", "to_y").alias("heading_deg"),
        )
        .collect()
        .unwrap();

    let heading_deg = result.column("heading_deg").unwrap().f64().unwrap();
    assert!((heading_deg.get(0).unwrap() - 90.0).abs() < 1e-10);
    assert!((heading_deg.get(1).unwrap() - 180.0).abs() < 1e-10);
}

#[test]
fn coordinate_pair_helpers_feed_angle_and_direction_helpers() {
    let df = df! {
        "nose_x" => &[2.0],
        "nose_y" => &[3.0],
        "tail_x" => &[1.0],
        "tail_y" => &[1.0],
    }
    .unwrap();

    let tail = CoordinateColumns::new("tail_x", "tail_y");
    let nose = CoordinateColumns::new("nose_x", "nose_y");
    let result = add_direction_columns_from(df.lazy(), &tail, &nose, "body")
        .with_column(heading_angle_from(&tail, &nose).alias("heading"))
        .collect()
        .unwrap();

    let dx = result.column("body_dx").unwrap().f64().unwrap();
    let dy = result.column("body_dy").unwrap().f64().unwrap();
    let heading = result.column("heading").unwrap().f64().unwrap();

    assert!((dx.get(0).unwrap() - 1.0).abs() < 1e-10);
    assert!((dy.get(0).unwrap() - 2.0).abs() < 1e-10);
    assert!((heading.get(0).unwrap() - (2.0f64).atan2(1.0)).abs() < 1e-10);
}
