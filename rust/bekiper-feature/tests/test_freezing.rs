use polars::prelude::*;

use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::solo::freezing::FreezingHeuristic;

#[test]
fn new_computes_window_frames() {
    let h = FreezingHeuristic::new("x", "y", 1.0, 0.005, 30.0);
    assert_eq!(h.window_frames, 30);
    assert!((h.displacement_threshold - 0.005).abs() < 1e-10);
}

#[test]
fn name_is_freezing() {
    let h = FreezingHeuristic::default_for("x", "y", 30.0);
    assert_eq!(h.name(), "freezing");
}

#[test]
fn evaluate_stationary_is_frozen() {
    // 10 stationary frames, window=3 frames, threshold=0.005
    let df = df! {
        "x" => &[1.0; 10],
        "y" => &[2.0; 10],
    }
    .unwrap();

    let h = FreezingHeuristic::new("x", "y", 0.1, 0.005, 30.0); // window=3 frames
    let result = h.evaluate(df.lazy()).collect().unwrap();
    let col = result.column("freezing").unwrap().bool().unwrap();

    // First 2 frames: rolling_sum needs min_periods=3 → null → false
    assert!(!col.get(0).unwrap());
    assert!(!col.get(1).unwrap());
    // Frame 2 onward: rolling sum of 0.0 displacement ≤ 0.005 → frozen
    for i in 2..10 {
        assert!(col.get(i).unwrap(), "frame {i} should be frozen");
    }
}

#[test]
fn evaluate_fast_movement_not_frozen() {
    let df = df! {
        "x" => &[0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        "y" => &[0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
    }
    .unwrap();

    let h = FreezingHeuristic::new("x", "y", 0.1, 0.005, 30.0); // window=3
    let result = h.evaluate(df.lazy()).collect().unwrap();
    let col = result.column("freezing").unwrap().bool().unwrap();

    // All frames are either null (initial) or have high displacement → not frozen
    for i in 0..6 {
        assert!(!col.get(i).unwrap(), "frame {i} should not be frozen");
    }
}

#[test]
fn evaluate_mixed() {
    // Stationary for 6 frames, then big jump
    let df = df! {
        "x" => &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 200.0],
        "y" => &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 200.0, 300.0],
    }
    .unwrap();

    let h = FreezingHeuristic::new("x", "y", 0.1, 0.005, 30.0); // window=3
    let result = h.evaluate(df.lazy()).collect().unwrap();
    let col = result.column("freezing").unwrap().bool().unwrap();

    // Frames 0-1: insufficient window → false
    assert!(!col.get(0).unwrap());
    assert!(!col.get(1).unwrap());
    // Frames 2-5: rolling sum of displacement = 0 → frozen
    for i in 2..6 {
        assert!(col.get(i).unwrap(), "frame {i} should be frozen");
    }
    // Frames 6-7: include the big jump → not frozen
    assert!(!col.get(6).unwrap());
    assert!(!col.get(7).unwrap());
}

#[test]
fn single_frame_not_frozen() {
    let df = df! {
        "x" => &[0.0],
        "y" => &[0.0],
    }
    .unwrap();

    let h = FreezingHeuristic::default_for("x", "y", 30.0);
    let result = h.evaluate(df.lazy()).collect().unwrap();
    let col = result.column("freezing").unwrap().bool().unwrap();

    // Insufficient window → null → false
    assert!(!col.get(0).unwrap());
}

#[test]
fn output_columns_are_diagnostic_then_verdict() {
    let h = FreezingHeuristic::default_for("x", "y", 30.0);
    let [diag, verdict] = h.output_columns();
    assert!(diag.ends_with("thresholded_rolling_displacement") && verdict == "freezing");
}
