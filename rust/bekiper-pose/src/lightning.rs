use std::io::BufRead;
use std::path::Path;

use polars::prelude::*;

use bekiper_core::error::{BekiperError, Result};

use crate::{PoseBackend, PoseReader};

/// Reader for Lightning Pose CSV output.
///
/// Lightning Pose outputs a CSV with columns:
///   `{label}_x`, `{label}_y`, `{label}_likelihood`
///
/// This is already the flat format used by the pipeline, so the reader
/// only needs to ensure a `frame` column is present.
pub struct LightningPoseReader {
    labels: Vec<String>,
}

impl LightningPoseReader {
    /// Create a reader with known labels.
    pub fn with_labels(labels: Vec<String>) -> Self {
        Self { labels }
    }

    /// Create a reader that auto-detects labels from the CSV header.
    pub fn from_file(path: &Path) -> Result<Self> {
        let labels = parse_lp_header(path)?;
        Ok(Self { labels })
    }

    fn read_csv(&self, path: &Path) -> Result<LazyFrame> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path.into()))?
            .finish()?;

        let col_names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let mut result = df;

        // Add frame column if missing (use row index)
        if !col_names.contains(&"frame".to_string()) {
            let n = result.height();
            let frame_col = Column::new("frame".into(), (0..n as i64).collect::<Vec<_>>());
            result = result.with_column(frame_col)?.clone();
        }

        Ok(result.lazy())
    }
}

impl PoseReader for LightningPoseReader {
    fn backend(&self) -> PoseBackend {
        PoseBackend::LightningPose
    }

    fn read(&self, path: &Path) -> Result<LazyFrame> {
        self.read_csv(path)
    }

    fn labels(&self) -> &[String] {
        &self.labels
    }
}

/// Parse Lightning Pose header: columns ending in `_x` give us the label names.
fn parse_lp_header(path: &Path) -> Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let header = reader
        .lines()
        .next()
        .ok_or_else(|| BekiperError::Data("Lightning Pose CSV is empty".into()))??;

    let mut labels = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for col in header.split(',') {
        let col = col.trim();
        if let Some(label) = col.strip_suffix("_x")
            && !label.is_empty()
            && label != "frame"
            && seen.insert(label.to_string())
        {
            labels.push(label.to_string());
        }
    }

    if labels.is_empty() {
        return Err(BekiperError::Data(
            "No labels found in Lightning Pose CSV header".into(),
        ));
    }

    Ok(labels)
}
