use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_math::velocity::velocity_expr;

use crate::heuristic::Heuristic;
use crate::heuristic::{coordinate_label_from_xy, decomposition_component, diag_column_name};

// ── Freezing defaults ─────────────────────────────────────────────────────
// Canonical defaults; task layer re-exports them via `TaskParams`.

/// Rolling-window duration (s) over which cumulative displacement is summed.
pub const DEFAULT_WINDOW_SECONDS: f64 = 1.0;

/// Maximum cumulative displacement (m) within the rolling window for a
/// frame to be classified as frozen. 5 mm is the standard mouse default.
pub const DEFAULT_DISPLACEMENT_THRESHOLD: f64 = 0.005;

/// Detects freezing via rolling cumulative displacement.
///
/// A frame is marked frozen when the total displacement over a sliding window
/// of `window_seconds` is below `displacement_threshold` meters. This captures
/// "the animal hasn't moved much recently" rather than just "the animal isn't
/// moving right now" (which an instantaneous speed threshold would do).
///
/// The minimum-bout-duration constraint is applied separately via the tolerance
/// module — this heuristic only marks individual frames.
///
/// Improvement over the legacy algorithm:
/// - Single centroid instead of multi-node AND (centroid already captures body
///   movement; per-node AND is redundant and the legacy implementation had a
///   premature-break bug in its post-AND cleanup).
/// - Rolling sum via Polars instead of O(n²) manual sliding window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreezingHeuristic {
    /// Coordinate columns to compute displacement from (e.g. center_ear).
    pub x_col: String,
    pub y_col: String,
    /// Window size in frames over which displacement is summed.
    pub window_frames: u32,
    /// Maximum cumulative displacement (meters) within the window
    /// for the frame to be classified as frozen.
    pub displacement_threshold: f64,
}

impl FreezingHeuristic {
    /// Create from physical parameters (seconds + m/s), converting to frames.
    pub fn new(
        x_col: impl Into<String>,
        y_col: impl Into<String>,
        window_seconds: f64,
        displacement_threshold: f64,
        fps: f64,
    ) -> Self {
        Self {
            x_col: x_col.into(),
            y_col: y_col.into(),
            window_frames: (window_seconds * fps).round() as u32,
            displacement_threshold,
        }
    }

    /// Convenience: default window and displacement threshold.
    pub fn default_for(x_col: impl Into<String>, y_col: impl Into<String>, fps: f64) -> Self {
        Self::new(
            x_col,
            y_col,
            DEFAULT_WINDOW_SECONDS,
            DEFAULT_DISPLACEMENT_THRESHOLD,
            fps,
        )
    }
}

impl FreezingHeuristic {
    /// Rolling sum of per-frame displacement (meters) over the window. It is the
    /// critical value of the threshold: a frame is frozen iff it is `<=` the
    /// threshold, so a threshold family shares this one column.
    pub fn rolling_displacement_expr(&self) -> Expr {
        let displacement = velocity_expr(&self.x_col, &self.y_col).fill_null(lit(0.0));
        let window = self.window_frames as usize;
        let rolling_opts = RollingOptionsFixedWindow {
            window_size: window,
            min_periods: window,
            ..Default::default()
        };
        displacement.rolling_sum(rolling_opts)
    }

    /// Names of the columns [`Heuristic::evaluate`] writes for the verdict.
    pub fn output_columns(&self) -> [String; 2] {
        [
            diag_column_name(self.name(), "thresholded_rolling_displacement"),
            self.name().to_string(),
        ]
    }
}

impl Heuristic for FreezingHeuristic {
    fn name(&self) -> &str {
        "freezing"
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        // Frozen when cumulative displacement is at or below threshold.
        let frozen = self
            .rolling_displacement_expr()
            .lt_eq(lit(self.displacement_threshold))
            .fill_null(lit(false));
        let component_col = diag_column_name(self.name(), "thresholded_rolling_displacement");
        lf.with_columns([
            frozen.clone().alias(&component_col),
            frozen.alias(self.name()),
        ])
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.name().to_string(),
            result_column: self.name().to_string(),
            coordinate_labels: vec![coordinate_label_from_xy(&self.x_col)],
            perimeter_label: None,
            components: vec![decomposition_component(
                "thresholded_rolling_displacement",
                diag_column_name(self.name(), "thresholded_rolling_displacement"),
                vec![coordinate_label_from_xy(&self.x_col)],
                None,
            )],
            show_final_result: false,
        })
    }
}
