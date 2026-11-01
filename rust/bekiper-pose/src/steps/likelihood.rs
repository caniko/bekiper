use polars::prelude::*;

use crate::pipeline::PipelineStep;

/// Filter rows where the likelihood of tracked body parts falls below a threshold.
/// Sets coordinates to null where likelihood is too low.
pub struct LikelihoodFilter {
    pub threshold: f64,
    pub likelihood_columns: Vec<String>,
    pub coordinate_columns: Vec<String>,
}

impl LikelihoodFilter {
    pub fn new(
        threshold: f64,
        likelihood_columns: Vec<String>,
        coordinate_columns: Vec<String>,
    ) -> Self {
        Self {
            threshold,
            likelihood_columns,
            coordinate_columns,
        }
    }
}

impl PipelineStep for LikelihoodFilter {
    fn name(&self) -> &str {
        "likelihood_filter"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        // For each likelihood column, null out the corresponding coordinate
        // columns where likelihood < threshold.
        let mut lf = lf;
        for (lik_col, coord_col) in self
            .likelihood_columns
            .iter()
            .zip(self.coordinate_columns.iter())
        {
            lf = lf.with_column(
                when(col(lik_col).lt(lit(self.threshold)))
                    .then(lit(NULL))
                    .otherwise(col(coord_col))
                    .alias(coord_col),
            );
        }
        lf
    }
}

/// Trim trial boundaries based on likelihood.
///
/// Finds the first and last frame where ALL keypoints have likelihood ≥ threshold,
/// and slices the trial to that range. Frames outside this range are dropped
/// entirely (they represent periods where pose estimation was unreliable,
/// typically at recording start/end).
///
/// This is opt-in — use only when edge frames are known to be unreliable.
/// The default `LikelihoodFilter` (nulling) is sufficient for most datasets.
pub struct LikelihoodTrimmer {
    pub threshold: f64,
    pub likelihood_columns: Vec<String>,
}

impl PipelineStep for LikelihoodTrimmer {
    fn name(&self) -> &str {
        "likelihood_trimmer"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        if self.likelihood_columns.is_empty() {
            return lf;
        }

        // Build expression: ALL likelihood columns ≥ threshold
        let mut all_good = col(&self.likelihood_columns[0]).gt_eq(lit(self.threshold));
        for lik_col in &self.likelihood_columns[1..] {
            all_good = all_good.and(col(lik_col).gt_eq(lit(self.threshold)));
        }

        // Add a temporary boolean column, collect, find boundaries, slice
        let df = lf
            .with_column(all_good.alias("_lik_ok"))
            .collect()
            .expect("likelihood_trimmer: collect failed");

        let ok_col = df.column("_lik_ok").unwrap().bool().unwrap();
        let first = ok_col.into_iter().position(|v| v.unwrap_or(false));
        let last = ok_col
            .into_iter()
            .enumerate()
            .rev()
            .find_map(|(i, v)| if v.unwrap_or(false) { Some(i) } else { None });

        let (start, end) = match (first, last) {
            (Some(s), Some(e)) => (s, e + 1),
            // No good frames → keep every row, still dropping `_lik_ok`.
            _ => (0, df.height()),
        };

        let mut trimmed = df.slice(start as i64, end - start);
        let _ = trimmed.drop_in_place("_lik_ok");
        trimmed.lazy()
    }
}
