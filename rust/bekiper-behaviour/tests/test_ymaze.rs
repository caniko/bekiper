use bekiper_behaviour::ymaze::*;

fn s(vals: &[&str]) -> Vec<String> {
    vals.iter().map(|v| v.to_string()).collect()
}

#[test]
fn classify_triad_correct() {
    assert_eq!(classify_triad("A", "B", "C"), TriadType::Correct);
}

#[test]
fn classify_triad_incorrect() {
    assert_eq!(classify_triad("A", "B", "A"), TriadType::Incorrect);
}

#[test]
fn classify_triad_incorrect_initial_repeat() {
    assert_eq!(classify_triad("A", "A", "B"), TriadType::Incorrect);
}

#[test]
fn classify_triad_neutral_bb() {
    assert_eq!(classify_triad("A", "B", "B"), TriadType::Neutral);
}

#[test]
fn analyze_alternations_perfect() {
    let seq = s(&["A", "B", "C", "A", "B", "C"]);
    let (c, i, n) = analyze_alternations(&seq);
    assert_eq!((c, i, n), (4, 0, 0));
}

#[test]
fn analyze_alternations_imperfect() {
    let seq = s(&["A", "B", "A", "B"]);
    let (c, i, n) = analyze_alternations(&seq);
    assert_eq!((c, i, n), (0, 2, 0));
}

#[test]
fn analyze_alternations_short() {
    let seq = s(&["A", "B"]);
    assert_eq!(analyze_alternations(&seq), (0, 0, 0));
}

#[test]
fn analyze_alternations_empty() {
    let seq: Vec<String> = vec![];
    assert_eq!(analyze_alternations(&seq), (0, 0, 0));
}

#[test]
fn permutation_distribution_counts_unique_triads() {
    let seq = s(&["A", "B", "C", "A", "C", "B"]);
    let distribution = permutation_alternation_distribution(&seq);

    assert_eq!(distribution.abc, 1);
    assert_eq!(distribution.acb, 1);
    assert_eq!(distribution.bca, 1);
    assert_eq!(distribution.total(), 3);
}

#[test]
fn triad_distributions_count_all_criteria_triads() {
    let seq = s(&["A", "B", "C", "A", "A", "B", "B", "C", "C", "A", "C", "C"]);
    let distributions = triad_distributions(&seq);

    assert_eq!(distributions.correct.abc, 1);
    assert_eq!(distributions.correct.bca, 1);
    assert_eq!(distributions.incorrect.cac, 1);
    assert_eq!(distributions.incorrect.aab, 1);
    assert_eq!(distributions.incorrect.bbc, 1);
    assert_eq!(distributions.incorrect.cca, 1);
    assert_eq!(distributions.neutral.abb, 1);
    assert_eq!(distributions.neutral.bcc, 1);
    assert_eq!(distributions.neutral.acc, 1);
    assert_eq!(distributions.correct.total(), 2);
    assert_eq!(distributions.incorrect.total(), 4);
    assert_eq!(distributions.neutral.total(), 4);
}

#[test]
fn spontaneous_alternation_excludes_neutral_triads() {
    let seq = s(&["A", "A", "B", "C", "A", "C"]);
    let (correct, incorrect, neutral) = analyze_alternations(&seq);

    assert_eq!((correct, incorrect, neutral), (2, 2, 0));
    assert!(
        (spontaneous_alternation_pct(correct, incorrect).expect("usable denominator") - 50.0).abs()
            < 1e-10
    );
    assert_eq!(spontaneous_alternation_pct(0, 0), None);
}

#[test]
fn reduce_arm_sequence_filters_center() {
    let seq = s(&["A", "Center", "B", "Center", "C"]);
    let result = reduce_to_arm_sequence(&seq, &["A", "B", "C"]);
    assert_eq!(result, s(&["A", "B", "C"]));
}

#[test]
fn reduce_arm_sequence_preserves_reentries() {
    // A → Center → A is two distinct arm visits (exit + re-entry)
    let seq = s(&["A", "Center", "A", "Center", "B"]);
    let result = reduce_to_arm_sequence(&seq, &["A", "B", "C"]);
    assert_eq!(result, s(&["A", "A", "B"]));
}

#[test]
fn reduce_arm_sequence_enables_neutral_triads() {
    // A → Center → B → Center → B should produce neutral triad ABB
    let seq = s(&["A", "Center", "B", "Center", "B"]);
    let arm_seq = reduce_to_arm_sequence(&seq, &["A", "B", "C"]);
    let (c, i, n) = analyze_alternations(&arm_seq);
    assert_eq!((c, i, n), (0, 0, 1)); // ABB is neutral
}

#[test]
fn reduce_arm_sequence_empty() {
    let seq: Vec<String> = vec![];
    let result = reduce_to_arm_sequence(&seq, &["A", "B", "C"]);
    assert!(result.is_empty());
}

#[test]
fn arm_preference_equal() {
    let pref = arm_preference(&[10.0, 10.0, 10.0]);
    assert!((pref - 0.0).abs() < 1e-10);
}

#[test]
fn arm_preference_unequal() {
    let pref = arm_preference(&[30.0, 0.0, 0.0]);
    assert!(pref > 0.0);
}

#[test]
fn extract_zone_sequence_basic() {
    let zones: Vec<(&str, Vec<bool>)> = vec![
        ("A", vec![true, false, false, false, false]),
        ("B", vec![false, true, true, false, false]),
        ("C", vec![false, false, false, true, true]),
    ];
    let seq = extract_zone_sequence(&zones, 5, 0);
    assert_eq!(seq, s(&["A", "B", "C"]));
}

#[test]
fn extract_zone_sequence_none_zones() {
    let zones: Vec<(&str, Vec<bool>)> = vec![
        ("A", vec![false, false, false]),
        ("B", vec![false, false, false]),
    ];
    let seq = extract_zone_sequence(&zones, 3, 0);
    assert!(seq.is_empty());
}

#[test]
fn extract_zone_sequence_collapses_consecutive() {
    let zones: Vec<(&str, Vec<bool>)> = vec![
        ("A", vec![true, true, false, false]),
        ("B", vec![false, false, true, true]),
    ];
    let seq = extract_zone_sequence(&zones, 4, 0);
    assert_eq!(seq, s(&["A", "B"]));
}

#[test]
fn permutation_distribution_counts_cba() {
    let distribution = permutation_alternation_distribution(&s(&["C", "B", "A"]));
    assert_eq!(distribution.cba, 1);
    assert_eq!(distribution.total(), 1);
}

#[test]
fn triads_with_labels_outside_abc_are_classified_but_not_tabulated() {
    // A fourth arm label still classifies, but no named Y-maze triad exists.
    let seq = s(&["A", "B", "D", "B", "B", "D", "D"]);
    let (correct, incorrect, neutral) = analyze_alternations(&seq);
    assert!(correct > 0 && incorrect > 0 && neutral > 0);
    let distributions = triad_distributions(&seq);
    assert_eq!(distributions.correct.total(), 0);
    assert_eq!(distributions.incorrect.total(), 0);
    assert_eq!(distributions.neutral.total(), 0);
    assert_eq!(permutation_alternation_distribution(&seq).total(), 0);
}

#[test]
fn arm_preference_with_no_arm_time_is_zero() {
    assert_eq!(arm_preference(&[0.0, 0.0, 0.0]), 0.0);
}
