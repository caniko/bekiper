//! Y-maze alternation analysis: triad classification, arm preference,
//! and zone sequence extraction.

use serde::{Deserialize, Serialize};

/// Result of Y-maze alternation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YMazeAnalysis {
    /// Full zone entry sequence (e.g. ["A", "Center", "B", "Center", "C", ...]).
    pub zone_sequence: Vec<String>,
    /// Reduced arm-only sequence (e.g. ["A", "B", "C", "A", ...]).
    pub arm_sequence: Vec<String>,
    /// Raw arm-only sequence from the original arm-threshold crossings before
    /// pose gap-fill, arm-membership hysteresis, or zone-gap bridging.
    #[serde(default)]
    pub raw_arm_sequence: Vec<String>,
    /// First arm entered in the reduced arm-only sequence.
    #[serde(default)]
    pub first_arm_entered: Option<String>,
    /// Last arm entered in the reduced arm-only sequence.
    #[serde(default)]
    pub last_arm_entered: Option<String>,
    /// Number of entries per arm.
    pub arm_entries: [u32; 3],
    /// Number of center entries.
    pub center_entries: u32,
    /// Correct alternation triads (all 3 arms visited consecutively).
    pub correct_alternations: u32,
    /// Incorrect alternation triads (repeated arm in window of 3).
    pub incorrect_alternations: u32,
    /// Neutral triads (same arm visited consecutively → can't assess).
    pub neutral_triads: u32,
    /// Spontaneous alternation percentage: correct / (correct + incorrect) * 100.
    ///
    /// `None` when no correct or incorrect triads are usable as a denominator.
    pub spontaneous_alternation_pct: Option<f64>,
    /// Arm preference: RMSD of time-per-arm around the mean.
    pub arm_preference: f64,
    /// Distribution of unique arm triad permutations (ABC, ACB, ...).
    #[serde(default)]
    pub permutation_distribution: TriadPermutationDistribution,
    /// Sum of the unique triad permutation counts.
    #[serde(default)]
    pub sum_of_permutation_distribution: u32,
    /// Distribution of correct triads (ABC, ACB, BAC, BCA, CAB, CBA).
    #[serde(default)]
    pub correct_triad_distribution: CorrectTriadDistribution,
    /// Distribution of incorrect triads (AAB, AAC, ABA, ...).
    #[serde(default)]
    pub incorrect_triad_distribution: IncorrectTriadDistribution,
    /// Distribution of neutral triads (AAA, ABB, ACC, ...).
    #[serde(default)]
    pub neutral_triad_distribution: NeutralTriadDistribution,
    // ── Parallel SA view built from `raw_arm_sequence` ────────────────────
    // These fields mirror `correct_alternations`, `incorrect_alternations`,
    // `neutral_triads`, and `spontaneous_alternation_pct` but are scored from
    // the truly raw arm-threshold crossing sequence, bypassing pose gap-fill,
    // arm-membership hysteresis, and the zone-sequence gap bridge. They are
    // always populated so downstream consumers can compare the mitigated
    // headline against the original arm-threshold view side-by-side.
    /// Correct triads scored against `raw_arm_sequence`.
    #[serde(default)]
    pub raw_correct_alternations: u32,
    /// Incorrect triads scored against `raw_arm_sequence`.
    #[serde(default)]
    pub raw_incorrect_alternations: u32,
    /// Neutral triads scored against `raw_arm_sequence`.
    #[serde(default)]
    pub raw_neutral_triads: u32,
    /// Spontaneous alternation percentage from the raw arm sequence.
    #[serde(default)]
    pub raw_spontaneous_alternation_pct: Option<f64>,
}

/// Distribution of unique Y-maze arm triads.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TriadPermutationDistribution {
    pub abc: u32,
    pub acb: u32,
    pub bac: u32,
    pub bca: u32,
    pub cab: u32,
    pub cba: u32,
}

impl TriadPermutationDistribution {
    pub fn total(&self) -> u32 {
        self.abc + self.acb + self.bac + self.bca + self.cab + self.cba
    }

    fn increment(&mut self, a: &str, b: &str, c: &str) {
        match (a, b, c) {
            ("A", "B", "C") => self.abc += 1,
            ("A", "C", "B") => self.acb += 1,
            ("B", "A", "C") => self.bac += 1,
            ("B", "C", "A") => self.bca += 1,
            ("C", "A", "B") => self.cab += 1,
            ("C", "B", "A") => self.cba += 1,
            _ => {}
        }
    }
}

/// Distribution of correct Y-maze triads.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorrectTriadDistribution {
    pub abc: u32,
    pub acb: u32,
    pub bac: u32,
    pub bca: u32,
    pub cab: u32,
    pub cba: u32,
}

impl CorrectTriadDistribution {
    pub fn total(&self) -> u32 {
        self.abc + self.acb + self.bac + self.bca + self.cab + self.cba
    }

    fn increment(&mut self, triad: &str) {
        match triad {
            "ABC" => self.abc += 1,
            "ACB" => self.acb += 1,
            "BAC" => self.bac += 1,
            "BCA" => self.bca += 1,
            "CAB" => self.cab += 1,
            "CBA" => self.cba += 1,
            _ => {}
        }
    }
}

/// Distribution of incorrect Y-maze triads.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct IncorrectTriadDistribution {
    pub aab: u32,
    pub aac: u32,
    pub aba: u32,
    pub aca: u32,
    pub bab: u32,
    pub bba: u32,
    pub bbc: u32,
    pub bcb: u32,
    pub cac: u32,
    pub cbc: u32,
    pub cca: u32,
    pub ccb: u32,
}

impl IncorrectTriadDistribution {
    pub fn total(&self) -> u32 {
        self.aab
            + self.aac
            + self.aba
            + self.aca
            + self.bab
            + self.bba
            + self.bbc
            + self.bcb
            + self.cac
            + self.cbc
            + self.cca
            + self.ccb
    }

    fn increment(&mut self, triad: &str) {
        match triad {
            "AAB" => self.aab += 1,
            "AAC" => self.aac += 1,
            "ABA" => self.aba += 1,
            "ACA" => self.aca += 1,
            "BAB" => self.bab += 1,
            "BBA" => self.bba += 1,
            "BBC" => self.bbc += 1,
            "BCB" => self.bcb += 1,
            "CAC" => self.cac += 1,
            "CBC" => self.cbc += 1,
            "CCA" => self.cca += 1,
            "CCB" => self.ccb += 1,
            _ => {}
        }
    }
}

/// Distribution of neutral Y-maze triads.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NeutralTriadDistribution {
    pub aaa: u32,
    pub abb: u32,
    pub acc: u32,
    pub baa: u32,
    pub bbb: u32,
    pub bcc: u32,
    pub caa: u32,
    pub cbb: u32,
    pub ccc: u32,
}

impl NeutralTriadDistribution {
    pub fn total(&self) -> u32 {
        self.aaa
            + self.abb
            + self.acc
            + self.baa
            + self.bbb
            + self.bcc
            + self.caa
            + self.cbb
            + self.ccc
    }

    fn increment(&mut self, triad: &str) {
        match triad {
            "AAA" => self.aaa += 1,
            "ABB" => self.abb += 1,
            "ACC" => self.acc += 1,
            "BAA" => self.baa += 1,
            "BBB" => self.bbb += 1,
            "BCC" => self.bcc += 1,
            "CAA" => self.caa += 1,
            "CBB" => self.cbb += 1,
            "CCC" => self.ccc += 1,
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TriadDistributions {
    pub correct: CorrectTriadDistribution,
    pub incorrect: IncorrectTriadDistribution,
    pub neutral: NeutralTriadDistribution,
}

/// Classify a triad of arm labels as correct, incorrect, or neutral.
///
/// - Correct: all 3 different (ABC, ACB, BAC, BCA, CAB, CBA)
/// - Neutral: consecutive visits to the same current-choice arm (AAA, ABB, etc.)
/// - Incorrect: repeats one of the previous two without completing triad
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriadType {
    Correct,
    Incorrect,
    Neutral,
}

/// Classify a sliding-window triad of three consecutive arm labels.
pub fn classify_triad(a: &str, b: &str, c: &str) -> TriadType {
    if a != b && b != c && a != c {
        // All three different → correct
        TriadType::Correct
    } else if b == c {
        // Current choice repeats the immediately previous arm → neutral.
        TriadType::Neutral
    } else {
        // Current choice repeats the first arm, or the first two entries match.
        TriadType::Incorrect
    }
}

/// Compute alternation analysis from a reduced arm sequence.
///
/// The arm sequence should only contain arm labels (no "center"),
/// with consecutive duplicates already removed.
pub fn analyze_alternations(arm_sequence: &[String]) -> (u32, u32, u32) {
    if arm_sequence.len() < 3 {
        return (0, 0, 0);
    }

    let mut correct = 0u32;
    let mut incorrect = 0u32;
    let mut neutral = 0u32;

    for window in arm_sequence.windows(3) {
        match classify_triad(&window[0], &window[1], &window[2]) {
            TriadType::Correct => correct += 1,
            TriadType::Incorrect => incorrect += 1,
            TriadType::Neutral => neutral += 1,
        }
    }

    (correct, incorrect, neutral)
}

/// Count the distribution of unique Y-maze triad permutations.
pub fn permutation_alternation_distribution(
    arm_sequence: &[String],
) -> TriadPermutationDistribution {
    let mut distribution = TriadPermutationDistribution::default();
    for window in arm_sequence.windows(3) {
        if classify_triad(&window[0], &window[1], &window[2]) == TriadType::Correct {
            distribution.increment(&window[0], &window[1], &window[2]);
        }
    }
    distribution
}

/// Count all criteria-defined correct, incorrect, and neutral triads.
pub fn triad_distributions(arm_sequence: &[String]) -> TriadDistributions {
    let mut distributions = TriadDistributions::default();
    for window in arm_sequence.windows(3) {
        let triad = format!("{}{}{}", window[0], window[1], window[2]);
        match classify_triad(&window[0], &window[1], &window[2]) {
            TriadType::Correct => distributions.correct.increment(&triad),
            TriadType::Incorrect => distributions.incorrect.increment(&triad),
            TriadType::Neutral => distributions.neutral.increment(&triad),
        }
    }
    distributions
}

/// Spontaneous alternation percentage using the criteria denominator.
///
/// Neutral triads are excluded because they prevent a valid alternation
/// assessment.
pub fn spontaneous_alternation_pct(
    correct_alternations: u32,
    incorrect_alternations: u32,
) -> Option<f64> {
    let denominator = correct_alternations + incorrect_alternations;
    if denominator == 0 {
        None
    } else {
        Some(100.0 * correct_alternations as f64 / denominator as f64)
    }
}

/// Reduce a zone sequence to arm-only entries, removing center visits
/// but preserving same-arm re-entries (e.g. A → Center → A counts as two A entries).
///
/// For example: ["A", "Center", "A", "Center", "B"] → ["A", "A", "B"]
/// Each arm crossing (exit to center then re-entry) is a distinct visit,
/// which is required for neutral triad detection (AAB, ABB, etc.).
pub fn reduce_to_arm_sequence(zone_sequence: &[String], arm_labels: &[&str]) -> Vec<String> {
    let mut result = Vec::new();
    for zone in zone_sequence {
        if arm_labels.contains(&zone.as_str()) {
            result.push(zone.clone());
        }
    }
    result
}

/// Compute arm preference as RMSD of time-per-arm around the mean.
///
/// arm_durations: [T_A, T_B, T_C] in seconds.
pub fn arm_preference(arm_durations: &[f64; 3]) -> f64 {
    let mean = arm_durations.iter().sum::<f64>() / 3.0;
    let variance = arm_durations
        .iter()
        .map(|t| (t - mean).powi(2))
        .sum::<f64>()
        / 3.0;
    variance.sqrt()
}

/// Build a zone sequence from a boolean zone columns on a collected DataFrame.
///
/// For each frame, determines which zone the animal is in based on
/// the first matching zone column. Returns the sequence of zone transitions
/// (consecutive same-zone entries collapsed).
pub fn extract_zone_sequence(
    zone_values: &[(&str, Vec<bool>)],
    n_frames: usize,
    max_gap_frames: u32,
) -> Vec<String> {
    let mut sequence = Vec::new();
    let mut prev_zone = String::new();
    let mut gap_frames = 0_u32;

    for i in 0..n_frames {
        let mut current_zone = "none".to_string();
        for (name, values) in zone_values {
            if values[i] {
                current_zone = name.to_string();
                break;
            }
        }
        if current_zone == "none" {
            if !prev_zone.is_empty() {
                gap_frames += 1;
                if gap_frames > max_gap_frames {
                    prev_zone.clear();
                    gap_frames = 0;
                }
            }
            continue;
        }
        gap_frames = 0;
        if current_zone != prev_zone {
            sequence.push(current_zone.clone());
            prev_zone = current_zone;
        }
    }

    sequence
}

#[cfg(test)]
#[path = "ymaze/tests.rs"]
mod tests;
