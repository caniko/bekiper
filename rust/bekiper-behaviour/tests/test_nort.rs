use bekiper_behaviour::nort::{
    NortIndices, build_observation_sequence, count_alternations, object_bias, side_bias,
};

#[test]
fn compute_equal_exploration() {
    let idx = NortIndices::compute(10.0, 10.0, 5, 5);
    assert!((idx.discrimination_index.unwrap() - 0.0).abs() < 1e-10);
    assert!((idx.recognition_index.unwrap() - 0.5).abs() < 1e-10);
    assert!((idx.pct_novel.unwrap() - 50.0).abs() < 1e-10);
}

#[test]
fn compute_novel_preference() {
    let idx = NortIndices::compute(15.0, 5.0, 8, 3);
    assert!((idx.discrimination_index.unwrap() - 0.5).abs() < 1e-10);
    assert!((idx.recognition_index.unwrap() - 0.75).abs() < 1e-10);
    assert!((idx.pct_novel.unwrap() - 75.0).abs() < 1e-10);
}

#[test]
fn compute_familiar_preference() {
    let idx = NortIndices::compute(5.0, 15.0, 3, 8);
    assert!((idx.discrimination_index.unwrap() - (-0.5)).abs() < 1e-10);
    assert!((idx.recognition_index.unwrap() - 0.25).abs() < 1e-10);
}

#[test]
fn compute_zero_exploration_reports_na() {
    let idx = NortIndices::compute(0.0, 0.0, 0, 0);
    assert!(idx.discrimination_index.is_none());
    assert!(idx.recognition_index.is_none());
    assert!(idx.pct_novel.is_none());
    // NA survives a result-file round trip.
    let restored: NortIndices = toml::from_str(&toml::to_string(&idx).unwrap()).unwrap();
    assert!(restored.discrimination_index.is_none());
    assert!(idx.log_ratio.is_none());
}

#[test]
fn compute_log_ratio_equal() {
    let idx = NortIndices::compute(10.0, 10.0, 5, 5);
    assert!(idx.log_ratio.is_some());
    assert!((idx.log_ratio.unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn compute_log_ratio_none_when_familiar_zero() {
    let idx = NortIndices::compute(10.0, 0.0, 5, 0);
    assert!(idx.log_ratio.is_none());
}

#[test]
fn compute_log_ratio_none_when_novel_zero() {
    let idx = NortIndices::compute(0.0, 10.0, 0, 5);
    assert!(idx.log_ratio.is_none());
}

#[test]
fn meets_criterion_above() {
    let idx = NortIndices::compute(10.0, 10.0, 5, 5);
    assert!(idx.meets_criterion(15.0));
}

#[test]
fn meets_criterion_below() {
    let idx = NortIndices::compute(5.0, 5.0, 3, 3);
    assert!(!idx.meets_criterion(15.0));
}

#[test]
fn meets_criterion_exact() {
    let idx = NortIndices::compute(10.0, 5.0, 5, 3);
    assert!(idx.meets_criterion(15.0));
}

#[test]
fn side_bias_no_bias() {
    assert!((side_bias(10.0, 10.0) - 0.0).abs() < 1e-10);
}

#[test]
fn side_bias_right_dominant() {
    assert!((side_bias(5.0, 15.0) - 0.5).abs() < 1e-10);
}

#[test]
fn side_bias_left_dominant() {
    assert!((side_bias(15.0, 5.0) - (-0.5)).abs() < 1e-10);
}

#[test]
fn side_bias_zero() {
    assert!((side_bias(0.0, 0.0) - 0.0).abs() < 1e-10);
}

#[test]
fn object_bias_no_bias() {
    assert!((object_bias(10.0, 10.0) - 0.0).abs() < 1e-10);
}

#[test]
fn object_bias_a_dominant() {
    assert!((object_bias(15.0, 5.0) - 0.5).abs() < 1e-10);
}

#[test]
fn nort_serde_roundtrip() {
    let idx = NortIndices::compute(12.0, 8.0, 6, 4);
    let toml_str = toml::to_string(&idx).unwrap();
    let restored: NortIndices = toml::from_str(&toml_str).unwrap();
    assert_eq!(restored.discrimination_index, idx.discrimination_index);
    assert_eq!(restored.recognition_index, idx.recognition_index);
    assert_eq!(restored.pct_novel, idx.pct_novel);
    assert_eq!(restored.novel_bouts, idx.novel_bouts);
    assert_eq!(restored.familiar_bouts, idx.familiar_bouts);
}

// ── Observation sequence tests ──────────────────────────────────────

#[test]
fn observation_sequence_two_objects_no_overlap() {
    use polars::prelude::*;

    // Object 1 active on frames 0-2, object 2 on frames 5-7, rest none.
    let obj1: Vec<bool> = vec![
        true, true, true, false, false, false, false, false, false, false,
    ];
    let obj2: Vec<bool> = vec![
        false, false, false, false, false, true, true, true, false, false,
    ];

    let df = df! {
        "exploring_obj1_raw" => &obj1,
        "exploring_obj2_raw" => &obj2,
    }
    .unwrap();

    let cols = vec!["exploring_obj1_raw".into(), "exploring_obj2_raw".into()];
    let seq = build_observation_sequence(&df, &cols);

    assert_eq!(seq, vec![1, 1, 1, 0, 0, 2, 2, 2, 0, 0]);
}

#[test]
fn observation_sequence_overlap_first_wins() {
    use polars::prelude::*;

    // Both active on frame 2 — first object wins (overlaps should be
    // pre-resolved, but build_observation_sequence assigns first-match).
    let obj1: Vec<bool> = vec![false, false, true, false];
    let obj2: Vec<bool> = vec![false, false, true, true];

    let df = df! {
        "col1" => &obj1,
        "col2" => &obj2,
    }
    .unwrap();

    let seq = build_observation_sequence(&df, &["col1".into(), "col2".into()]);
    assert_eq!(seq[2], 1); // first wins
    assert_eq!(seq[3], 2); // only obj2
}

#[test]
fn observation_sequence_all_false() {
    use polars::prelude::*;

    let df = df! {
        "a" => &[false, false, false],
        "b" => &[false, false, false],
    }
    .unwrap();

    let seq = build_observation_sequence(&df, &["a".into(), "b".into()]);
    assert_eq!(seq, vec![0, 0, 0]);
}

#[test]
fn observation_sequence_empty_df() {
    use polars::prelude::*;

    let df = df! {
        "a" => Vec::<bool>::new(),
    }
    .unwrap();

    let seq = build_observation_sequence(&df, &["a".into()]);
    assert!(seq.is_empty());
}

// ── Alternation count tests ─────────────────────────────────────────

#[test]
fn alternation_count_simple_switches() {
    // 1,1,1, 0,0, 2,2,2, 0,0, 1,1 → 3 observation bouts (obj1, obj2, obj1)
    let seq = vec![1, 1, 1, 0, 0, 2, 2, 2, 0, 0, 1, 1];
    assert_eq!(count_alternations(&seq, 1), 3);
}

#[test]
fn alternation_count_no_exploration() {
    let seq = vec![0, 0, 0, 0, 0];
    assert_eq!(count_alternations(&seq, 1), 0);
}

#[test]
fn alternation_count_single_bout() {
    let seq = vec![0, 0, 1, 1, 1, 0, 0];
    assert_eq!(count_alternations(&seq, 1), 1);
}

#[test]
fn alternation_count_empty() {
    assert_eq!(count_alternations(&[], 1), 0);
}

#[test]
fn alternation_count_filters_short_segments() {
    // 1,1,1,1, 2, 1,1,1 — the single frame of obj2 is too short (min_frames=3)
    // After filtering: 1,1,1,1, 1,1,1 → 1 bout (the short 2 is collapsed)
    let seq = vec![1, 1, 1, 1, 2, 1, 1, 1];
    assert_eq!(count_alternations(&seq, 3), 1);
}

#[test]
fn alternation_count_keeps_long_segments() {
    // 1,1,1, 2,2,2, 1,1,1 — all segments >= 3 frames
    let seq = vec![1, 1, 1, 2, 2, 2, 1, 1, 1];
    assert_eq!(count_alternations(&seq, 3), 3);
}

#[test]
fn alternation_count_filters_short_zero_gap() {
    // 1,1,1, 0, 2,2,2 — the single zero gap is filtered (min_frames=2)
    // After: 1,1,1, 2,2,2 → 2 bouts
    let seq = vec![1, 1, 1, 0, 2, 2, 2];
    assert_eq!(count_alternations(&seq, 2), 2);
}

#[test]
fn alternation_count_continuous_single_object() {
    let seq = vec![1, 1, 1, 1, 1];
    assert_eq!(count_alternations(&seq, 1), 1);
}

#[test]
fn object_bias_zero_exploration_is_unbiased() {
    assert_eq!(object_bias(0.0, 0.0), 0.0);
}

#[test]
fn observation_sequence_ignores_missing_or_non_boolean_columns() {
    use polars::prelude::*;

    let df = df! { "a" => &[true, false], "speed" => &[1.0, 2.0] }.unwrap();
    // Missing and non-boolean columns keep their object index unused.
    let cols = ["absent".into(), "speed".into(), "a".into()];
    assert_eq!(build_observation_sequence(&df, &cols), vec![3, 0]);
}
