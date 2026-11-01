use polars::prelude::*;

use crate::pipeline::PipelineStep;
use bekiper_math::velocity::high_velocity_mask_expr;

/// Null out coordinate values where frame-to-frame velocity exceeds a threshold.
///
/// Uses iterative re-evaluation: after nulling high-velocity frames, recomputes
/// velocity against interpolated neighbors and nulls again. Repeats until
/// convergence or `max_iterations` is reached. This handles multi-frame
/// teleportation artifacts that a single pass would miss.
///
/// Optionally applies a σ-based outlier gate: frames where displacement
/// exceeds `sigma_threshold * std(displacement)` are also nulled.
pub struct HighVelocityFilter {
    pub max_velocity: f64,
    pub x_columns: Vec<String>,
    pub y_columns: Vec<String>,
    /// Maximum iterations for cascade resolution. Default: 3.
    pub max_iterations: u32,
    /// Optional: null frames where displacement > N * std(displacement).
    /// Set to 0.0 to disable (default).
    pub sigma_threshold: f64,
}

impl HighVelocityFilter {
    /// Null high-velocity frames for a single coordinate pair, iteratively.
    fn null_iterative(&self, df: DataFrame, x_col: &str, y_col: &str) -> DataFrame {
        let mut df = df;
        let temp_x = format!("_hv_interp_{x_col}");
        let temp_y = format!("_hv_interp_{y_col}");

        for _ in 0..self.max_iterations {
            // Interpolate nulls to compute velocity against nearest non-null neighbors.
            let evaluated = df
                .clone()
                .lazy()
                .with_columns([
                    col(x_col)
                        .interpolate(InterpolationMethod::Linear)
                        .alias(&temp_x),
                    col(y_col)
                        .interpolate(InterpolationMethod::Linear)
                        .alias(&temp_y),
                ])
                .with_column(
                    high_velocity_mask_expr(&temp_x, &temp_y, self.max_velocity).alias("_hv_mask"),
                )
                .collect()
                .expect("high_velocity_filter: collect failed");

            // Count frames that are newly flagged (not already null)
            let mask_col = evaluated.column("_hv_mask").unwrap().bool().unwrap();
            let null_col = df.column(x_col).unwrap().is_null();
            let new_violations: u32 = mask_col
                .into_iter()
                .zip(&null_col)
                .map(|(m, already)| u32::from(m.unwrap_or(false) && !already.unwrap_or(false)))
                .sum();

            if new_violations == 0 {
                break;
            }

            // Null original coordinates where mask is true
            let mut result = evaluated
                .lazy()
                .with_columns([
                    when(col("_hv_mask"))
                        .then(lit(NULL))
                        .otherwise(col(x_col))
                        .alias(x_col),
                    when(col("_hv_mask"))
                        .then(lit(NULL))
                        .otherwise(col(y_col))
                        .alias(y_col),
                ])
                .collect()
                .expect("high_velocity_filter: apply failed");

            let _ = result.drop_in_place("_hv_mask");
            let _ = result.drop_in_place(&temp_x);
            let _ = result.drop_in_place(&temp_y);
            df = result;
        }

        df
    }

    /// Apply σ-based displacement outlier gate.
    fn apply_sigma_gate(&self, mut df: DataFrame) -> DataFrame {
        if self.sigma_threshold <= 0.0 {
            return df;
        }

        for (x_col, y_col) in self.x_columns.iter().zip(self.y_columns.iter()) {
            // Compute displacement
            let dx = col(x_col) - col(x_col).shift(lit(1));
            let dy = col(y_col) - col(y_col).shift(lit(1));
            let disp = (dx.pow(2) + dy.pow(2)).sqrt();

            let with_disp = df
                .clone()
                .lazy()
                .with_column(disp.alias("_disp"))
                .collect()
                .expect("sigma_gate: disp failed");

            // Compute threshold = mean + sigma * std
            let disp_series = with_disp.column("_disp").unwrap().f64().unwrap();
            let mean = disp_series.mean().unwrap_or(0.0);
            let std = disp_series.std(1).unwrap_or(0.0);
            let threshold = mean + self.sigma_threshold * std;

            let outlier = col("_disp").gt(lit(threshold));
            let mut result = with_disp
                .lazy()
                .with_columns([
                    when(outlier.clone())
                        .then(lit(NULL))
                        .otherwise(col(x_col))
                        .alias(x_col),
                    when(outlier)
                        .then(lit(NULL))
                        .otherwise(col(y_col))
                        .alias(y_col),
                ])
                .collect()
                .expect("sigma_gate: apply failed");

            let _ = result.drop_in_place("_disp");
            df = result;
        }

        df
    }
}

impl PipelineStep for HighVelocityFilter {
    fn name(&self) -> &str {
        "high_velocity_filter"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let mut df = lf
            .collect()
            .expect("high_velocity_filter: initial collect failed");

        for (x_col, y_col) in self.x_columns.iter().zip(self.y_columns.iter()) {
            df = self.null_iterative(df, x_col, y_col);
        }

        df = self.apply_sigma_gate(df);

        df.lazy()
    }
}
