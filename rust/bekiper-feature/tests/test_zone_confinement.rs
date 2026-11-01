use polars::prelude::*;

use bekiper_core::types::CoordinateColumns;
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::solo::zone_confinement::ZoneConfinementHeuristic;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_perimeter::rectangle::Rectangle;

#[test]
fn name_matches_zone_name() {
    let h = ZoneConfinementHeuristic {
        zone_name: "in_center".into(),
        perimeter: Perimeter::new("center", Circle::new(0.0, 0.0, 5.0)),
        coords: CoordinateColumns::new("x", "y"),
    };
    assert_eq!(h.name(), "in_center");
}

#[test]
fn evaluate_inside_circle() {
    let df = df! {
        "x" => &[0.0, 1.0, -1.0, 0.5],
        "y" => &[0.0, 1.0, -1.0, 0.5],
    }
    .unwrap();

    let h = ZoneConfinementHeuristic {
        zone_name: "in_zone".into(),
        perimeter: Perimeter::new("zone", Circle::new(0.0, 0.0, 5.0)),
        coords: CoordinateColumns::new("x", "y"),
    };

    let result = h.evaluate(df.lazy()).collect().unwrap();
    let col = result.column("in_zone").unwrap().bool().unwrap();

    for i in 0..4 {
        assert!(col.get(i).unwrap(), "point {i} should be inside circle");
    }
}

#[test]
fn evaluate_outside_circle() {
    let df = df! {
        "x" => &[10.0, 20.0, -15.0],
        "y" => &[10.0, 20.0, -15.0],
    }
    .unwrap();

    let h = ZoneConfinementHeuristic {
        zone_name: "in_zone".into(),
        perimeter: Perimeter::new("zone", Circle::new(0.0, 0.0, 5.0)),
        coords: CoordinateColumns::new("x", "y"),
    };

    let result = h.evaluate(df.lazy()).collect().unwrap();
    let col = result.column("in_zone").unwrap().bool().unwrap();

    for i in 0..3 {
        assert!(!col.get(i).unwrap(), "point {i} should be outside circle");
    }
}

#[test]
fn evaluate_mixed_circle() {
    let df = df! {
        "x" => &[0.0, 10.0, 2.0, -20.0],
        "y" => &[0.0, 10.0, 2.0, -20.0],
    }
    .unwrap();

    let h = ZoneConfinementHeuristic {
        zone_name: "in_zone".into(),
        perimeter: Perimeter::new("zone", Circle::new(0.0, 0.0, 5.0)),
        coords: CoordinateColumns::new("x", "y"),
    };

    let result = h.evaluate(df.lazy()).collect().unwrap();
    let col = result.column("in_zone").unwrap().bool().unwrap();

    assert!(col.get(0).unwrap()); // (0,0) inside
    assert!(!col.get(1).unwrap()); // (10,10) outside
    assert!(col.get(2).unwrap()); // (2,2) inside
    assert!(!col.get(3).unwrap()); // (-20,-20) outside
}

#[test]
fn evaluate_rectangle_zone() {
    // Rectangle centered at (5,5), width=10, height=10 → spans (0,0) to (10,10)
    let df = df! {
        "x" => &[5.0, 3.0, 8.0, 15.0],
        "y" => &[5.0, 3.0, 8.0, 15.0],
    }
    .unwrap();

    let h = ZoneConfinementHeuristic {
        zone_name: "in_rect".into(),
        perimeter: Perimeter::new("rect", Rectangle::new(5.0, 5.0, 10.0, 10.0)),
        coords: CoordinateColumns::new("x", "y"),
    };

    let result = h.evaluate(df.lazy()).collect().unwrap();
    let col = result.column("in_rect").unwrap().bool().unwrap();

    assert!(col.get(0).unwrap()); // (5,5) center → inside
    assert!(col.get(1).unwrap()); // (3,3) → inside
    assert!(col.get(2).unwrap()); // (8,8) → inside
    assert!(!col.get(3).unwrap()); // (15,15) → outside
}
