//! Assay-independent segmentation and duration selection. Units are supplied
//! by the adapter, so the same kernel handles frames and elapsed seconds.
use std::ops::Range;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DurationComparison {
    #[default]
    AtLeast,
    GreaterThan,
}

impl DurationComparison {
    pub fn accepts(self, duration: f64, minimum: f64) -> bool {
        match self {
            Self::AtLeast => duration >= minimum,
            Self::GreaterThan => duration > minimum,
        }
    }
}

impl std::str::FromStr for DurationComparison {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "at_least" => Ok(Self::AtLeast),
            "greater_than" => Ok(Self::GreaterThan),
            _ => Err("dwell comparison must be at_least or greater_than".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectedBout {
    pub start: usize,
    /// Exclusive sample boundary.
    pub end: usize,
    pub duration: f64,
    pub accepted: bool,
    /// First sample whose ending boundary qualifies the original crossing.
    pub qualification_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemporalSelection {
    pub mask: Vec<bool>,
    pub post_merge_mask: Vec<bool>,
    pub raw_spans: Vec<Range<usize>>,
    pub bouts: Vec<SelectedBout>,
}

/// Half-open runs of the requested state, including recording-edge runs.
pub fn state_spans(values: &[bool], state: bool) -> Vec<Range<usize>> {
    let mut spans = Vec::new();
    let mut i = 0;
    while i < values.len() {
        if values[i] != state {
            i += 1;
            continue;
        }
        let start = i;
        while i < values.len() && values[i] == state {
            i += 1;
        }
        spans.push(start..i);
    }
    spans
}

/// Partition ordered components into bouts. Assay adapters own identity and
/// exclusion barriers; this owns the common adjacent-component merge walk.
pub fn partition_adjacent<T: Clone>(
    components: &[T],
    can_merge: impl Fn(&T, &T) -> bool,
) -> Vec<Vec<T>> {
    let mut batches: Vec<Vec<T>> = Vec::new();
    for component in components {
        if let Some(batch) = batches.last_mut()
            && batch
                .last()
                .is_some_and(|previous| can_merge(previous, component))
        {
            batch.push(component.clone());
        } else {
            batches.push(vec![component.clone()]);
        }
    }
    batches
}

/// Bridge bounded gaps, then select whole bouts. `duration(start, end)` must
/// measure monotonically increasing sample boundaries in the caller's units.
/// A zero minimum disables duration filtering, including strict comparison.
/// Density is measured in the original gap, preserving the legacy contract.
pub fn select_bouts(
    raw: &[bool],
    duration: impl Fn(usize, usize) -> f64,
    minimum: f64,
    comparison: DurationComparison,
    maximum_gap: f64,
    gap_density: f64,
) -> TemporalSelection {
    select_bouts_with_gap_bound(
        raw,
        duration,
        minimum,
        comparison,
        maximum_gap,
        gap_density,
        false,
    )
}

/// Explicit strict upper gap bound; the original entry point retains inclusive
/// historical semantics. Minimum duration comparison is independent.
pub fn select_bouts_with_gap_bound(
    raw: &[bool],
    duration: impl Fn(usize, usize) -> f64,
    minimum: f64,
    comparison: DurationComparison,
    maximum_gap: f64,
    gap_density: f64,
    strict_gap: bool,
) -> TemporalSelection {
    let raw_spans = state_spans(raw, true);
    let mut mask = raw.to_vec();
    let merged = partition_adjacent(&raw_spans, |previous, next| {
        let gap = previous.end..next.start;
        let density = raw[gap.clone()].iter().filter(|&&v| v).count() as f64 / gap.len() as f64;
        let elapsed = duration(gap.start, gap.end);
        maximum_gap > 0.0
            && (if strict_gap {
                elapsed < maximum_gap
            } else {
                elapsed <= maximum_gap
            })
            && density >= gap_density
    });
    let spans: Vec<_> = merged
        .iter()
        .map(|batch| batch[0].start..batch[batch.len() - 1].end)
        .collect();
    for span in &spans {
        mask[span.clone()].fill(true);
    }
    let post_merge_mask = mask.clone();
    let bouts = spans
        .into_iter()
        .map(|span| {
            let measured = duration(span.start, span.end);
            let accepted = minimum == 0.0 || comparison.accepts(measured, minimum);
            let qualification_index = if accepted {
                (span.start..span.end).find(|&i| {
                    minimum == 0.0 || comparison.accepts(duration(span.start, i + 1), minimum)
                })
            } else {
                mask[span.clone()].fill(false);
                None
            };
            SelectedBout {
                start: span.start,
                end: span.end,
                duration: measured,
                accepted,
                qualification_index,
            }
        })
        .collect();
    TemporalSelection {
        mask,
        post_merge_mask,
        raw_spans,
        bouts,
    }
}
