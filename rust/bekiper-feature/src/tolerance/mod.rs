use polars::prelude::*;

mod policy;
pub use policy::{
    BoutTiming, DEFAULT_MIN_DWELL_SECONDS, DurationComparison, DwellOverrides, TemporalPolicy,
    TemporalSelection,
};

/// Bout selection on a collected `DataFrame` boolean column. Durations follow
/// [`TemporalPolicy`] and [`BoutTiming`] (seconds, half-open intervals).
pub trait TemporalBoutOps {
    fn select_temporal_bouts(
        &self,
        bool_col: &str,
        policy: &TemporalPolicy,
        timing: &BoutTiming,
    ) -> PolarsResult<TemporalSelection>;
}

impl TemporalBoutOps for DataFrame {
    fn select_temporal_bouts(
        &self,
        bool_col: &str,
        policy: &TemporalPolicy,
        timing: &BoutTiming,
    ) -> PolarsResult<TemporalSelection> {
        policy.filter_series(self.column(bool_col)?.bool()?, timing)
    }
}
