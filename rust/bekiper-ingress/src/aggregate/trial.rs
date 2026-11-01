use std::path::Path;

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::types::TaskType;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Deserialized `result.toml` for a single trial.
///
/// `animal_metadata`, `summaries`, and `detailed` are kept as
/// `serde_json::Value` so per-task extractors can pluck whatever keys they
/// need without forcing every task schema into a single monolithic struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub experiment: String,
    pub task_type: TaskType,
    #[serde(default)]
    pub analysis_role: Option<String>,
    pub fps: f64,
    pub meters_per_pixel: f64,
    #[serde(default)]
    pub animal_metadata: Value,
    #[serde(default)]
    pub session_context: Value,
    #[serde(default)]
    pub summaries: Vec<Value>,
    #[serde(default)]
    pub detailed: Value,
}

impl TrialResult {
    /// Read and parse a `result.toml` file from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| BekiperError::Other(format!("failed to read {}: {e}", path.display())))?;
        toml::from_str(&contents).map_err(|e| {
            BekiperError::Config(format!(
                "failed to parse {} as result.toml: {e}",
                path.display()
            ))
        })
    }
}

/// Parse the recipe segment from an experiment name of the form
/// `{animal}_seq{N}_{recipe}` — e.g. `"1035_seq3_oulupdate"` → `"oulupdate"`.
///
/// The recipe segment is everything after the second underscore, so compound
/// recipe names containing underscores (`nort_training`) round-trip
/// correctly. Returns `None` if the name has fewer than three underscore-
/// separated segments or the second segment is not `seq{N}` with `N` a
/// non-negative integer.
pub fn parse_recipe_from_experiment(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.splitn(3, '_').collect();
    if parts.len() < 3 {
        return None;
    }
    let seq = parts[1];
    if !seq.starts_with("seq") {
        return None;
    }
    seq[3..].parse::<u32>().ok()?;
    Some(parts[2].to_string())
}

#[cfg(test)]
mod parse_recipe {
    use super::*;

    #[test]
    fn legacy_ymaze_without_variant() {
        assert_eq!(
            parse_recipe_from_experiment("1035_seq3_ymaze"),
            Some("ymaze".to_string())
        );
    }

    #[test]
    fn legacy_ymaze_with_variant_suffix() {
        assert_eq!(
            parse_recipe_from_experiment("1035_seq3_ymaze__proximal65"),
            Some("ymaze__proximal65".to_string())
        );
    }

    #[test]
    fn descriptive_ymaze_variant_uses_analysis_role() {
        let trial = TrialResult {
            experiment: "y_maze_videos_y_maze_pre_pre_video_14__proximal45".to_string(),
            task_type: TaskType::YMaze,
            analysis_role: Some("ymaze__proximal45".to_string()),
            fps: 15.0,
            meters_per_pixel: 0.001,
            animal_metadata: Value::Null,
            session_context: Value::Null,
            summaries: Vec::new(),
            detailed: Value::Null,
        };

        assert_eq!(parse_recipe_from_experiment(&trial.experiment), None);
    }
}
