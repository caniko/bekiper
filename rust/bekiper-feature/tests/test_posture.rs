use polars::prelude::*;

use bekiper_core::video::FrameTiming;
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::solo::grooming::GroomingHeuristic;
use bekiper_feature::heuristic::solo::rearing::RearingHeuristic;
use bekiper_feature::heuristic::solo::{Forepaws, speed_per_frame};

fn flags(df: &DataFrame, name: &str) -> Vec<bool> {
    df.column(name)
        .unwrap()
        .bool()
        .unwrap()
        .into_iter()
        .map(|v| v.expect("heuristic columns are null-free"))
        .collect()
}

/// Stationary centroid; nose at the origin and tail base at `tail_x`.
fn posture_frame(tail_x: &[f64], centroid_x: &[f64]) -> DataFrame {
    let n = tail_x.len();
    df! {
        "nose_x" => vec![0.0; n],
        "nose_y" => vec![0.0; n],
        "tail_base_x" => tail_x,
        "tail_base_y" => vec![0.0; n],
        "mouse_center_x" => centroid_x,
        "mouse_center_y" => vec![0.0; n],
    }
    .unwrap()
}

fn with_forepaws(mut df: DataFrame, left: &[Option<f64>], right: &[Option<f64>]) -> DataFrame {
    let zeros = vec![0.0; left.len()];
    for series in [
        Series::new("left_forepaw_x".into(), left),
        Series::new("right_forepaw_x".into(), right),
        Series::new("left_forepaw_y".into(), &zeros),
        Series::new("right_forepaw_y".into(), &zeros),
    ] {
        df.with_column(series.into_column()).unwrap();
    }
    df
}

#[test]
fn forepaw_defaults_and_speed_conversion() {
    let fp = Forepaws::default();
    assert_eq!(fp.left_x, "left_forepaw_x");
    assert_eq!(fp.right_y, "right_forepaw_y");
    assert_eq!(speed_per_frame(0.3, 30.0), 0.01);
}

#[test]
fn grooming_requires_low_speed_and_body_length_in_inclusive_range() {
    let h = GroomingHeuristic::new(10.0);
    assert_eq!(h.max_speed, 0.0005);
    // Frame 0 has no previous centroid, so speed is null and never grooming.
    let df = posture_frame(
        &[0.06, 0.06, 0.04, 0.08, 0.03, 0.09, 0.06, f64::NAN],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01],
    );
    let out = h.evaluate(df.lazy()).collect().unwrap();
    assert_eq!(
        flags(&out, "grooming"),
        vec![false, true, true, true, false, false, false, false]
    );
    assert_eq!(
        flags(&out, "__bk_diag__grooming__low_speed"),
        vec![false, true, true, true, true, true, false, true]
    );
    assert!(
        out.column("__bk_diag__grooming__forepaws_near_face")
            .is_err()
    );
    assert_eq!(h.decomposition().unwrap().components.len(), 2);
}

#[test]
fn grooming_forepaws_need_either_paw_near_the_nose() {
    let h = GroomingHeuristic::new(10.0).with_forepaws();
    let df = with_forepaws(
        posture_frame(&[0.06; 5], &[0.0; 5]),
        &[Some(0.01), Some(0.01), Some(0.05), Some(0.02), None],
        &[Some(0.01), Some(0.05), Some(0.05), Some(0.05), Some(0.01)],
    );
    let out = h.evaluate(df.lazy()).collect().unwrap();
    assert_eq!(
        flags(&out, "__bk_diag__grooming__forepaws_near_face"),
        vec![true, true, false, true, true]
    );
    assert_eq!(
        flags(&out, "grooming"),
        vec![false, true, false, true, true]
    );
    let spec = h.decomposition().unwrap();
    assert_eq!(spec.components.len(), 3);
    assert_eq!(spec.components[2].label, "forepaws_near_face");
    assert_eq!(spec.coordinate_labels, vec!["nose"]);
    assert!(spec.show_final_result);
}

#[test]
fn rearing_requires_low_speed_and_compact_body() {
    let h = RearingHeuristic::new(10.0);
    let df = posture_frame(
        &[0.03, 0.03, 0.05, 0.06, 0.03, f64::NAN],
        &[0.0, 0.0, 0.0, 0.0, 0.002, 0.002],
    );
    let out = h.evaluate(df.lazy()).collect().unwrap();
    assert_eq!(
        flags(&out, "rearing"),
        vec![false, true, true, false, false, false]
    );
    assert_eq!(
        flags(&out, "__bk_diag__rearing__compact_body"),
        vec![true, true, true, false, true, false]
    );
    assert_eq!(h.decomposition().unwrap().components.len(), 2);
}

#[test]
fn rearing_forepaws_need_both_paws_near_the_nose() {
    let h = RearingHeuristic::new(10.0).with_forepaws();
    let df = with_forepaws(
        posture_frame(&[0.03; 5], &[0.0; 5]),
        &[Some(0.01), Some(0.01), Some(0.03), None, Some(0.04)],
        &[Some(0.01), Some(0.01), Some(0.03), Some(0.01), Some(0.01)],
    );
    let out = h.evaluate(df.lazy()).collect().unwrap();
    // A missing paw is not evidence of elevation.
    assert_eq!(
        flags(&out, "__bk_diag__rearing__forepaws_near_head"),
        vec![true, true, true, false, false]
    );
    assert_eq!(
        flags(&out, "rearing"),
        vec![false, true, true, false, false]
    );
    let spec = h.decomposition().unwrap();
    assert_eq!(spec.components.len(), 3);
    assert_eq!(spec.components[2].label, "forepaws_near_head");
}

struct Plain;

impl Heuristic for Plain {
    fn name(&self) -> &str {
        "plain"
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        lf.with_column(lit(true).alias("plain"))
    }
}

#[test]
fn heuristic_trait_defaults() {
    let h = Plain;
    assert_eq!(h.result_column(), "plain");
    assert!(h.decomposition().is_none());
    let empty = df! { "x" => &[1.0, 2.0] }.unwrap();
    let missing = h.summary(&empty, &FrameTiming::cfr(2.0));
    assert_eq!((missing.true_frames, missing.total_frames), (0, 0));
    assert_eq!(missing.seconds, 0.0);
    let evaluated = h.evaluate(empty.lazy()).collect().unwrap();
    let summary = h.summary(&evaluated, &FrameTiming::cfr(2.0));
    assert_eq!((summary.true_frames, summary.total_frames), (2, 2));
    assert_eq!(summary.seconds, 1.0);
}
