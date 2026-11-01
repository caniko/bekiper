//! NORT discrimination and recognition indices, plus side/object bias
//! computations.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// NORT (Novel Object Recognition Test) discrimination indices.
///
/// Computed from exploration times of novel (N) and familiar (F) objects.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NortIndices {
    /// Exploration time of novel object (seconds).
    pub novel_seconds: f64,
    /// Exploration time of familiar object (seconds).
    pub familiar_seconds: f64,
    /// Total exploration time (seconds).
    pub total_seconds: f64,
    /// Bout count for novel object.
    pub novel_bouts: u32,
    /// Bout count for familiar object.
    pub familiar_bouts: u32,
    /// Discrimination Index: (T_N - T_F) / (T_N + T_F). Range [-1, 1]. NA (`None`) when T_N + T_F = 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discrimination_index: Option<f64>,
    /// Recognition Index: T_N / (T_N + T_F). Range [0, 1]. NA (`None`) when T_N + T_F = 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recognition_index: Option<f64>,
    /// Percent novel: 100 × RI. NA (`None`) when T_N + T_F = 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pct_novel: Option<f64>,
    /// Log ratio: ln(T_N / T_F). Undefined if T_F = 0.
    pub log_ratio: Option<f64>,
}

impl NortIndices {
    /// Compute all NORT indices from exploration times and bout counts.
    pub fn compute(
        novel_seconds: f64,
        familiar_seconds: f64,
        novel_bouts: u32,
        familiar_bouts: u32,
    ) -> Self {
        let total = novel_seconds + familiar_seconds;
        let di = (total > 0.0).then(|| (novel_seconds - familiar_seconds) / total);
        let ri = (total > 0.0).then(|| novel_seconds / total);

        let log_ratio = if familiar_seconds > 0.0 && novel_seconds > 0.0 {
            Some((novel_seconds / familiar_seconds).ln())
        } else {
            None
        };

        Self {
            novel_seconds,
            familiar_seconds,
            total_seconds: total,
            novel_bouts,
            familiar_bouts,
            discrimination_index: di,
            recognition_index: ri,
            pct_novel: ri.map(|ri| 100.0 * ri),
            log_ratio,
        }
    }

    /// Check if animal meets minimum exploration criterion.
    /// Animals with total exploration < threshold are excluded.
    pub fn meets_criterion(&self, min_total_seconds: f64) -> bool {
        self.total_seconds >= min_total_seconds
    }
}

// ── Qualia decomposition ────────────────────────────────────────────

/// Per-sensory-channel statistics for qualia decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualiaChannel {
    /// Heuristic type: `"body_proximity"`, `"olfaction"`, or `"whisker_interaction"`.
    pub heuristic_type: String,
    /// Per-object exploration seconds (ordered by object_labels).
    pub per_object_seconds: Vec<f64>,
    /// Per-object bout counts (ordered by object_labels).
    pub per_object_bouts: Vec<u32>,
}

/// Full qualia decomposition: per-heuristic per-object statistics +
/// bias scores + observation sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualiaDecomposition {
    /// One entry per sensory channel.
    pub channels: Vec<QualiaChannel>,
    /// Per-heuristic relative bias: `100 * obj_seconds / channel_total_seconds`.
    /// Key = heuristic_type, value = per-object bias (ordered by object_labels).
    pub relative_bias: HashMap<String, Vec<f64>>,
    /// Per-heuristic absolute bias: `100 * obj_seconds / total_trial_seconds`.
    pub absolute_bias: HashMap<String, Vec<f64>>,
    /// Per-frame object observation sequence (0 = no object, 1..N = object index).
    /// Built from the combined exploration heuristic (not qualia channels).
    /// When multiple objects are active on the same frame, the nearest wins
    /// (already resolved by `resolve_overlapping_explorations`).
    pub observation_sequence: Vec<u8>,
    /// Number of distinct observation bouts (switches between objects),
    /// after filtering short fluctuations via temporal tolerance.
    pub alternation_count: u32,
    /// Optional merged heuristic results (from `MergeRule` combinations).
    /// Key = merge name, value = per-object seconds.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub merged: HashMap<String, Vec<f64>>,
}

/// Build a per-frame observation sequence from per-object boolean columns.
///
/// Each frame gets the 1-based index of the object being explored (0 = none).
/// If multiple objects are true on the same frame, the first one wins
/// (caller should resolve overlaps before calling this).
pub fn build_observation_sequence(df: &polars::prelude::DataFrame, raw_cols: &[String]) -> Vec<u8> {
    let n = df.height();
    let mut seq = vec![0u8; n];
    for (obj_idx, col_name) in raw_cols.iter().enumerate() {
        if let Ok(series) = df.column(col_name)
            && let Ok(bools) = series.bool()
        {
            for (i, val) in bools.into_iter().enumerate() {
                if val == Some(true) && seq[i] == 0 {
                    seq[i] = (obj_idx + 1) as u8;
                }
            }
        }
    }
    seq
}

/// Count distinct observation bouts after collapsing short repeating segments.
///
/// Mirrors the legacy `reduce_repeating_sequences` + count pattern:
/// segments shorter than `min_frames` are merged into the preceding segment,
/// then we count how many non-zero segments remain.
pub fn count_alternations(sequence: &[u8], min_frames: usize) -> u32 {
    if sequence.is_empty() {
        return 0;
    }

    // Run-length encode: (value, length) pairs.
    let mut runs: Vec<(u8, usize)> = Vec::new();
    let mut current = sequence[0];
    let mut length = 1usize;
    for &val in &sequence[1..] {
        if val == current {
            length += 1;
        } else {
            runs.push((current, length));
            current = val;
            length = 1;
        }
    }
    runs.push((current, length));

    // Collapse short segments: merge into previous run, then deduplicate
    // adjacent same-value entries.
    let mut reduced: Vec<u8> = Vec::new();
    for (val, len) in &runs {
        if *len < min_frames && !reduced.is_empty() {
            // Too short — skip (effectively merges into previous).
            continue;
        }
        // Deduplicate: if same value as previous, don't push a new entry.
        if reduced.last() != Some(val) {
            reduced.push(*val);
        }
    }

    // Count non-zero entries (distinct observation bouts).
    reduced.iter().filter(|&&v| v != 0).count() as u32
}

/// Side bias: T_B - T_A / (T_B + T_A) for familiarization phase.
/// Range [-1, 1]. 0 = no bias.
pub fn side_bias(left_seconds: f64, right_seconds: f64) -> f64 {
    let total = left_seconds + right_seconds;
    if total > 0.0 {
        (right_seconds - left_seconds) / total
    } else {
        0.0
    }
}

/// Object bias: T_A - T_B / (T_A + T_B) for identical objects in T1.
pub fn object_bias(object_a_seconds: f64, object_b_seconds: f64) -> f64 {
    let total = object_a_seconds + object_b_seconds;
    if total > 0.0 {
        (object_a_seconds - object_b_seconds) / total
    } else {
        0.0
    }
}
