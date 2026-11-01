use super::*;

fn arm_seq(labels: &[&str]) -> Vec<String> {
    labels.iter().map(|s| (*s).to_string()).collect()
}

// --- classify_triad: exhaustive 27-cell coverage ------------------------

#[test]
fn classify_triad_all_six_correct_triads() {
    for triad in ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"] {
        let chars: Vec<&str> = triad.split("").filter(|s| !s.is_empty()).collect();
        assert_eq!(
            classify_triad(chars[0], chars[1], chars[2]),
            TriadType::Correct,
            "{triad}"
        );
    }
}

#[test]
fn classify_triad_all_twelve_incorrect_triads() {
    for triad in [
        "AAB", "AAC", "ABA", "ACA", "BAB", "BBA", "BBC", "BCB", "CAC", "CBC", "CCA", "CCB",
    ] {
        let chars: Vec<&str> = triad.split("").filter(|s| !s.is_empty()).collect();
        assert_eq!(
            classify_triad(chars[0], chars[1], chars[2]),
            TriadType::Incorrect,
            "{triad}"
        );
    }
}

#[test]
fn classify_triad_all_nine_neutral_triads() {
    for triad in [
        "AAA", "ABB", "ACC", "BAA", "BBB", "BCC", "CAA", "CBB", "CCC",
    ] {
        let chars: Vec<&str> = triad.split("").filter(|s| !s.is_empty()).collect();
        assert_eq!(
            classify_triad(chars[0], chars[1], chars[2]),
            TriadType::Neutral,
            "{triad}"
        );
    }
}

#[test]
fn classify_triad_partitions_full_27_cell_space() {
    let mut correct = 0;
    let mut incorrect = 0;
    let mut neutral = 0;
    for a in ["A", "B", "C"] {
        for b in ["A", "B", "C"] {
            for c in ["A", "B", "C"] {
                match classify_triad(a, b, c) {
                    TriadType::Correct => correct += 1,
                    TriadType::Incorrect => incorrect += 1,
                    TriadType::Neutral => neutral += 1,
                }
            }
        }
    }
    assert_eq!(correct, 6);
    assert_eq!(incorrect, 12);
    assert_eq!(neutral, 9);
    assert_eq!(correct + incorrect + neutral, 27);
}

// --- analyze_alternations -----------------------------------------------

#[test]
fn analyze_alternations_below_window_returns_zero() {
    assert_eq!(analyze_alternations(&arm_seq(&[])), (0, 0, 0));
    assert_eq!(analyze_alternations(&arm_seq(&["A"])), (0, 0, 0));
    assert_eq!(analyze_alternations(&arm_seq(&["A", "B"])), (0, 0, 0));
}

#[test]
fn analyze_alternations_single_correct_triad() {
    assert_eq!(analyze_alternations(&arm_seq(&["A", "B", "C"])), (1, 0, 0));
}

#[test]
fn analyze_alternations_overlapping_sliding_window() {
    // ABCAB → windows: ABC (correct), BCA (correct), CAB (correct)
    assert_eq!(
        analyze_alternations(&arm_seq(&["A", "B", "C", "A", "B"])),
        (3, 0, 0)
    );
}

#[test]
fn analyze_alternations_mixed_triad_types() {
    // AABCA → windows: AAB (incorrect), ABC (correct), BCA (correct)
    assert_eq!(
        analyze_alternations(&arm_seq(&["A", "A", "B", "C", "A"])),
        (2, 1, 0)
    );
}

#[test]
fn analyze_alternations_neutral_triad_in_window() {
    // ABBCA → ABB (neutral), BBC (incorrect), BCA (correct)
    assert_eq!(
        analyze_alternations(&arm_seq(&["A", "B", "B", "C", "A"])),
        (1, 1, 1)
    );
}

// --- spontaneous_alternation_pct ---------------------------------------

#[test]
fn spontaneous_alternation_pct_zero_denominator_is_none() {
    assert_eq!(spontaneous_alternation_pct(0, 0), None);
}

#[test]
fn spontaneous_alternation_pct_all_correct_is_100() {
    assert_eq!(spontaneous_alternation_pct(5, 0), Some(100.0));
}

#[test]
fn spontaneous_alternation_pct_all_incorrect_is_0() {
    assert_eq!(spontaneous_alternation_pct(0, 5), Some(0.0));
}

#[test]
fn spontaneous_alternation_pct_mixed_is_fraction() {
    assert_eq!(spontaneous_alternation_pct(3, 1), Some(75.0));
    assert_eq!(spontaneous_alternation_pct(1, 1), Some(50.0));
}

// --- arm_preference -----------------------------------------------------

#[test]
fn arm_preference_equal_durations_is_zero() {
    assert_eq!(arm_preference(&[30.0, 30.0, 30.0]), 0.0);
}

#[test]
fn arm_preference_unequal_durations_rmsd() {
    // mean = 30, deviations = (-20, 0, 20), variance = 800/3 ≈ 266.67
    let result = arm_preference(&[10.0, 30.0, 50.0]);
    let expected = (800.0_f64 / 3.0).sqrt();
    assert!(
        (result - expected).abs() < 1e-9,
        "arm_preference = {result}, expected {expected}"
    );
}

#[test]
fn arm_preference_all_zero_is_zero() {
    assert_eq!(arm_preference(&[0.0, 0.0, 0.0]), 0.0);
}

// --- triad_distributions ------------------------------------------------

#[test]
fn triad_distributions_count_all_criteria_triads() {
    // Build a sequence that contains at least one of each triad type.
    let seq = arm_seq(&["A", "B", "C", "A", "B", "B", "C", "A"]);
    // windows: ABC (correct), BCA (correct), CAB (correct), ABB (neutral),
    //          BBC (incorrect), BCA (correct).
    let dist = triad_distributions(&seq);
    assert_eq!(dist.correct.total(), 4);
    assert_eq!(dist.incorrect.total(), 1);
    assert_eq!(dist.neutral.total(), 1);
    assert_eq!(dist.correct.bca, 2);
    assert_eq!(dist.correct.abc, 1);
    assert_eq!(dist.correct.cab, 1);
    assert_eq!(dist.incorrect.bbc, 1);
    assert_eq!(dist.neutral.abb, 1);
}

#[test]
fn permutation_alternation_distribution_only_counts_correct_triads() {
    let seq = arm_seq(&["A", "B", "C", "A", "B", "B"]);
    // Windows: ABC, BCA, CAB, ABB. Only the first three are correct.
    let dist = permutation_alternation_distribution(&seq);
    assert_eq!(dist.total(), 3);
    assert_eq!(dist.abc, 1);
    assert_eq!(dist.bca, 1);
    assert_eq!(dist.cab, 1);
}

// --- reduce_to_arm_sequence --------------------------------------------

#[test]
fn reduce_to_arm_sequence_drops_center_preserves_re_entries() {
    let zones = arm_seq(&["A", "Center", "A", "Center", "B"]);
    let reduced = reduce_to_arm_sequence(&zones, &["A", "B", "C"]);
    assert_eq!(reduced, arm_seq(&["A", "A", "B"]));
}

#[test]
fn reduce_to_arm_sequence_filters_unknown_labels() {
    let zones = arm_seq(&["A", "B", "X", "Y", "C"]);
    let reduced = reduce_to_arm_sequence(&zones, &["A", "B", "C"]);
    assert_eq!(reduced, arm_seq(&["A", "B", "C"]));
}

// --- TriadDistribution invariants --------------------------------------

#[test]
fn correct_triad_distribution_total_matches_increments() {
    let mut d = CorrectTriadDistribution::default();
    d.increment("ABC");
    d.increment("ABC");
    d.increment("BAC");
    d.increment("XYZ"); // unknown — ignored
    assert_eq!(d.abc, 2);
    assert_eq!(d.bac, 1);
    assert_eq!(d.total(), 3);
}

#[test]
fn incorrect_triad_distribution_total_sums_twelve_buckets() {
    let mut d = IncorrectTriadDistribution::default();
    for label in [
        "AAB", "AAC", "ABA", "ACA", "BAB", "BBA", "BBC", "BCB", "CAC", "CBC", "CCA", "CCB",
    ] {
        d.increment(label);
    }
    assert_eq!(d.total(), 12);
}

#[test]
fn neutral_triad_distribution_total_sums_nine_buckets() {
    let mut d = NeutralTriadDistribution::default();
    for label in [
        "AAA", "ABB", "ACC", "BAA", "BBB", "BCC", "CAA", "CBB", "CCC",
    ] {
        d.increment(label);
    }
    assert_eq!(d.total(), 9);
}
