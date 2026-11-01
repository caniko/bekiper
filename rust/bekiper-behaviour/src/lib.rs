//! Behavioural task definitions for rodent neuroscience experiments.
//!
//! Each task (Y-maze, open field, NORT, cheeseboard) implements the
//! [`BehaviouralTask`] trait, which provides a unified `analyze` method
//! that evaluates heuristics on a Polars `LazyFrame` and returns an
//! [`AnalysisResult`].

pub mod factory;
pub mod mapping;
pub mod motion_detail;
pub mod nort;
pub mod object_recognition;
pub mod open_field;
pub mod params;
pub mod radial_arm;
pub mod reward_tracing;
pub mod ymaze;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::{HeuristicDecompositionSpec, QuadrantSpec};
use bekiper_core::types::TaskType;
use bekiper_core::video::FrameTiming;
use bekiper_feature::heuristic::{Heuristic, HeuristicSummary};

pub use factory::TaskOverrides;
pub use params::TaskParams;

/// Typed task-specific detailed payload.
///
/// The per-task detail stays strongly typed inside Rust, but callers can still
/// serialize it through [`as_json_value`](Self::as_json_value) for generic
/// export and aggregation code that expects a schema-free tree.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum TaskDetail {
    ObjectRecognition(object_recognition::NortResult),
    YMaze(radial_arm::YMazeResult),
    OpenField(open_field::OftResult),
    Cheeseboard(reward_tracing::CheeseboardResult),
}

impl TaskDetail {
    pub fn task_type(&self) -> TaskType {
        match self {
            Self::ObjectRecognition(_) => TaskType::ObjectRecognition,
            Self::YMaze(_) => TaskType::YMaze,
            Self::OpenField(_) => TaskType::OpenField,
            Self::Cheeseboard(_) => TaskType::Cheeseboard,
        }
    }

    pub fn as_json_value(&self) -> serde_json::Value {
        match self {
            Self::ObjectRecognition(detail) => {
                serde_json::to_value(detail).expect("ObjectRecognition detail should serialize")
            }
            Self::YMaze(detail) => {
                serde_json::to_value(detail).expect("YMaze detail should serialize")
            }
            Self::OpenField(detail) => {
                serde_json::to_value(detail).expect("OpenField detail should serialize")
            }
            Self::Cheeseboard(detail) => {
                serde_json::to_value(detail).expect("Cheeseboard detail should serialize")
            }
        }
    }

    pub fn from_json_value(task_type: TaskType, value: serde_json::Value) -> Result<Self> {
        match task_type {
            TaskType::ObjectRecognition => serde_json::from_value(value)
                .map(Self::ObjectRecognition)
                .map_err(|e| {
                    BekiperError::Config(format!("invalid object-recognition detail: {e}"))
                }),
            TaskType::YMaze => serde_json::from_value(value)
                .map(Self::YMaze)
                .map_err(|e| BekiperError::Config(format!("invalid y-maze detail: {e}"))),
            TaskType::OpenField => serde_json::from_value(value)
                .map(Self::OpenField)
                .map_err(|e| BekiperError::Config(format!("invalid open-field detail: {e}"))),
            TaskType::Cheeseboard => serde_json::from_value(value)
                .map(Self::Cheeseboard)
                .map_err(|e| BekiperError::Config(format!("invalid cheeseboard detail: {e}"))),
        }
    }
}

/// Result of analyzing a trial with a behavioural task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub task_type: TaskType,
    pub summaries: Vec<HeuristicSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decompositions: Vec<HeuristicDecompositionSpec>,
    /// The fully evaluated DataFrame including all heuristic boolean columns
    /// and coordinate data. Used for inspection export.
    #[serde(skip)]
    pub evaluated_df: Option<DataFrame>,
}

/// Trait for behavioural tasks. Each task defines which heuristics to run
/// and how to analyze the data.
///
/// The `analyze` default method is shared by all tasks — DRY.
/// Associated type `EnclosureShape` drives monomorphization of the
/// entire heuristic pipeline per experiment type.
pub trait BehaviouralTask: Send + Sync {
    /// Human-readable task name.
    fn task_type(&self) -> TaskType;

    /// Build the set of heuristics for this task.
    fn heuristics(&self) -> Vec<Box<dyn Heuristic>>;

    /// Auto-generated quadrant grids (NxN) produced by this task, if any.
    /// Emitted alongside the `AnalysisResult` so the inspection manifest
    /// can render a per-cell occupancy overlay. Default: none.
    fn quadrants(&self) -> Vec<QuadrantSpec> {
        Vec::new()
    }

    /// Run all heuristics and collect summaries.
    /// Default implementation — DRY across all tasks.
    fn analyze(&self, lf: LazyFrame, timing: &FrameTiming) -> AnalysisResult {
        let heuristics = self.heuristics();
        let decompositions = decompositions_from_heuristics(&heuristics);

        // Apply all heuristics to the lazy frame.
        // Polars fuses them into a single optimized plan.
        let evaluated = heuristics.iter().fold(lf, |lf, h| h.evaluate(lf));

        let df = evaluated
            .collect()
            .expect("failed to collect heuristic results");

        let summaries = heuristics.iter().map(|h| h.summary(&df, timing)).collect();

        AnalysisResult {
            task_type: self.task_type(),
            summaries,
            decompositions,
            evaluated_df: Some(df),
        }
    }

    /// Run task-specific detailed analysis, returning the generic result
    /// plus a task-specific JSON blob with rich metrics (zone summaries,
    /// indices, etc.). Default returns None for the detailed result.
    fn analyze_detailed(
        &self,
        lf: LazyFrame,
        timing: &FrameTiming,
    ) -> (AnalysisResult, Option<TaskDetail>) {
        (self.analyze(lf, timing), None)
    }

    /// Resolved knobs (defaults included), nested exactly like `task_params`
    /// config keys (`exploration.min_bout_seconds`, `ymaze.proximal_threshold`).
    /// Null entries mean "not set"; consumers skip them. Default: none.
    fn effective_parameters(&self) -> Option<serde_json::Value> {
        None
    }
}

pub fn decompositions_from_heuristics(
    heuristics: &[Box<dyn Heuristic>],
) -> Vec<HeuristicDecompositionSpec> {
    heuristics
        .iter()
        .filter_map(|h| h.decomposition())
        .collect()
}

/// Extract `HeuristicSummary` entries from a collected DataFrame by scanning
/// boolean columns whose names match the given heuristic names.
///
/// This avoids re-creating heuristic objects just to call `summary()`.
pub fn summaries_from_df(
    df: &DataFrame,
    heuristic_names: &[&str],
    timing: &FrameTiming,
) -> Vec<HeuristicSummary> {
    heuristic_names
        .iter()
        .map(|&name| {
            let (true_frames, total_frames) = if let Ok(series) = df.column(name)
                && let Ok(bools) = series.bool()
            {
                (bools.sum().unwrap_or(0) as u64, bools.len() as u64)
            } else {
                (0, 0)
            };
            HeuristicSummary {
                name: (*name).into(),
                true_frames,
                total_frames,
                seconds: true_frames as f64 / timing.fps,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small DataFrame with two boolean columns for testing.
    fn test_df() -> DataFrame {
        let col_a =
            BooleanChunked::from_slice("heuristic_a".into(), &[true, true, false, true, false]);
        let col_b =
            BooleanChunked::from_slice("heuristic_b".into(), &[false, false, false, false, true]);
        DataFrame::new(5, vec![col_a.into_column(), col_b.into_column()]).unwrap()
    }

    #[test]
    fn summaries_from_df_counts_true_frames() {
        let df = test_df();
        let timing = FrameTiming::cfr(10.0);
        let summaries = summaries_from_df(&df, &["heuristic_a", "heuristic_b"], &timing);

        assert_eq!(summaries.len(), 2);

        assert_eq!(summaries[0].name, "heuristic_a");
        assert_eq!(summaries[0].true_frames, 3);
        assert_eq!(summaries[0].total_frames, 5);
        assert!((summaries[0].seconds - 0.3).abs() < 1e-10); // 3 / 10.0

        assert_eq!(summaries[1].name, "heuristic_b");
        assert_eq!(summaries[1].true_frames, 1);
        assert_eq!(summaries[1].total_frames, 5);
        assert!((summaries[1].seconds - 0.1).abs() < 1e-10); // 1 / 10.0
    }

    #[test]
    fn summaries_from_df_missing_column_returns_zeros() {
        let df = test_df();
        let timing = FrameTiming::cfr(30.0);
        let summaries = summaries_from_df(&df, &["nonexistent"], &timing);

        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].true_frames, 0);
        assert_eq!(summaries[0].total_frames, 0);
    }

    #[test]
    fn summaries_from_df_empty_names() {
        let df = test_df();
        let timing = FrameTiming::cfr(30.0);
        let summaries = summaries_from_df(&df, &[], &timing);
        assert!(summaries.is_empty());
    }

    /// A task relying on every trait default.
    struct DefaultTask;
    impl BehaviouralTask for DefaultTask {
        fn task_type(&self) -> TaskType {
            TaskType::OpenField
        }
        fn heuristics(&self) -> Vec<Box<dyn Heuristic>> {
            vec![Box::new(
                bekiper_feature::heuristic::solo::freezing::FreezingHeuristic::new(
                    "mouse_center_x",
                    "mouse_center_y",
                    1.0,
                    0.01,
                    2.0,
                ),
            )]
        }
    }

    #[test]
    fn default_analysis_evaluates_heuristics_without_task_detail() {
        let df = df! { "mouse_center_x" => [0.0; 4], "mouse_center_y" => [0.0; 4] }.unwrap();
        let task = DefaultTask;
        assert!(task.quadrants().is_empty());
        let (result, detail) = task.analyze_detailed(df.lazy(), &FrameTiming::cfr(2.0));
        assert!(detail.is_none());
        assert!(task.effective_parameters().is_none());
        assert_eq!(result.task_type, TaskType::OpenField);
        assert_eq!(result.summaries.len(), 1);
        assert_eq!(result.summaries[0].total_frames, 4);
        assert_eq!(result.decompositions.len(), 1);
        assert!(result.evaluated_df.unwrap().column("freezing").is_ok());
    }

    #[test]
    fn task_detail_reports_its_task_and_rejects_foreign_json() {
        use bekiper_perimeter::{circle::Circle, perimeter::Perimeter};
        let task = reward_tracing::Cheeseboard {
            start_area: None,
            enclosure: Perimeter::new("board", Circle::new(0.0, 0.0, 5.0)),
            reward_locations: vec![Perimeter::new("reward_a", Circle::new(3.0, 0.0, 0.5))],
        };
        let df = df! { "mouse_center_x" => [4.0, 3.0], "mouse_center_y" => [0.0, 0.0] }.unwrap();
        let (_, detail) = task.analyze_detailed(df.lazy(), &FrameTiming::cfr(4.0));
        let detail = detail.unwrap();
        assert_eq!(detail.task_type(), TaskType::Cheeseboard);
        let json = detail.as_json_value();
        let restored = TaskDetail::from_json_value(TaskType::Cheeseboard, json.clone()).unwrap();
        assert_eq!(restored.task_type(), TaskType::Cheeseboard);
        for (task_type, expected) in [
            (TaskType::ObjectRecognition, "object-recognition"),
            (TaskType::YMaze, "y-maze"),
            (TaskType::OpenField, "open-field"),
        ] {
            let error = TaskDetail::from_json_value(task_type, json.clone())
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
        assert!(TaskDetail::from_json_value(TaskType::Cheeseboard, serde_json::json!(1)).is_err());
    }
}
