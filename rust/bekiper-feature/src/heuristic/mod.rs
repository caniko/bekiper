pub mod combined;
pub mod helper;
pub mod proximity;
pub mod ray;
pub mod solo;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::{HeuristicDecompositionComponent, HeuristicDecompositionSpec};
use bekiper_core::video::FrameTiming;

/// Summary of a heuristic evaluation: how many frames/seconds the behavior occurred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeuristicSummary {
    pub name: String,
    pub true_frames: u64,
    pub total_frames: u64,
    pub seconds: f64,
}

pub const DIAGNOSTIC_COLUMN_PREFIX: &str = "__bk_diag__";

pub fn diag_column_name(heuristic_name: &str, component: &str) -> String {
    format!("{DIAGNOSTIC_COLUMN_PREFIX}{heuristic_name}__{component}")
}

pub fn coordinate_label_from_xy(x_col: &str) -> String {
    x_col.strip_suffix("_x").unwrap_or(x_col).to_string()
}

pub fn decomposition_component(
    label: impl Into<String>,
    result_column: impl Into<String>,
    coordinate_labels: Vec<String>,
    perimeter_label: Option<String>,
) -> HeuristicDecompositionComponent {
    HeuristicDecompositionComponent {
        label: label.into(),
        result_column: result_column.into(),
        coordinate_labels,
        perimeter_label,
        ..Default::default()
    }
}

/// Core heuristic trait. Each heuristic adds one or more boolean columns
/// to a LazyFrame indicating frames where the behavior is detected.
///
/// The `summary` default method is shared by all heuristics — DRY.
pub trait Heuristic: Send + Sync {
    /// Name of the boolean result column added to the DataFrame.
    fn name(&self) -> &str;

    /// Actual boolean column written by `evaluate()`.
    ///
    /// Most heuristics use `name()` directly. Some helper/combined
    /// heuristics persist an internal raw mask under a different column.
    fn result_column(&self) -> &str {
        self.name()
    }

    /// Apply this heuristic to the LazyFrame, adding a boolean result column.
    fn evaluate(&self, lf: LazyFrame) -> LazyFrame;

    /// Optional metadata for a standalone heuristic decomposition figure.
    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        None
    }

    /// Compute summary statistics from the evaluated boolean column.
    /// Uses `FrameTiming` for VFR-accurate seconds; for CFR falls back to `timing.fps`.
    fn summary(&self, df: &DataFrame, timing: &FrameTiming) -> HeuristicSummary {
        let col = df.column(self.name());
        let (true_frames, total_frames) = match col {
            Ok(series) => {
                let bool_col = series.bool().unwrap();
                let total = bool_col.len() as u64;
                let trues = bool_col.sum().unwrap_or(0) as u64;
                (trues, total)
            }
            Err(_) => (0, 0),
        };

        HeuristicSummary {
            name: self.name().to_string(),
            true_frames,
            total_frames,
            seconds: true_frames as f64 / timing.fps,
        }
    }
}
