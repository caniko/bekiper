use std::io::BufRead;
use std::path::Path;

use polars::prelude::*;

use bekiper_core::error::{BekiperError, Result};

use crate::{PoseBackend, PoseReader};

/// Reader for SLEAP analysis CSV exports.
///
/// SLEAP exports tracking results via `File → Export Analysis CSV...` with columns:
///   `track`, `frame_idx`, `node1.x`, `node1.y`, `node1.score`, `node2.x`, ...
///
/// Or via `sleap-convert --format analysis.csv` which uses the same layout.
///
/// For single-animal projects the `track` column may be absent, leaving just
/// `frame_idx` followed by the node coordinate columns.
pub struct SleapCsvReader {
    labels: Vec<String>,
}

impl SleapCsvReader {
    /// Create a reader with known labels (skips header parsing).
    pub fn with_labels(labels: Vec<String>) -> Self {
        Self { labels }
    }

    /// Create a reader that auto-detects labels from the SLEAP CSV header.
    pub fn from_file(path: &Path) -> Result<Self> {
        let labels = parse_sleap_csv_header(path)?;
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

        let mut renamed = df;

        // Rename SLEAP columns to our flat format
        for label in &self.labels {
            // SLEAP uses "node.x", "node.y", "node.score"
            let sleap_x = format!("{label}.x");
            let sleap_y = format!("{label}.y");
            let sleap_score = format!("{label}.score");

            if col_names.contains(&sleap_x) {
                renamed.rename(&sleap_x, format!("{label}_x").into())?;
            }
            if col_names.contains(&sleap_y) {
                renamed.rename(&sleap_y, format!("{label}_y").into())?;
            }
            if col_names.contains(&sleap_score) {
                renamed.rename(&sleap_score, format!("{label}_likelihood").into())?;
            }
        }

        // Normalize frame column name
        if col_names.contains(&"frame_idx".to_string()) {
            renamed.rename("frame_idx", "frame".into())?;
        }

        // Drop track column if present (single-animal assumption)
        let final_cols: Vec<String> = renamed
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .filter(|s| s != "track")
            .collect();

        let exprs: Vec<Expr> = final_cols.iter().map(|c| col(c.as_str())).collect();
        Ok(renamed.lazy().select(exprs))
    }
}

impl PoseReader for SleapCsvReader {
    fn backend(&self) -> PoseBackend {
        PoseBackend::Sleap
    }

    fn read(&self, path: &Path) -> Result<LazyFrame> {
        self.read_csv(path)
    }

    fn labels(&self) -> &[String] {
        &self.labels
    }
}

/// Parse SLEAP CSV header to extract node names.
///
/// Header columns follow the pattern: `node_name.{x|y|score}`
fn parse_sleap_csv_header(path: &Path) -> Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let header = reader
        .lines()
        .next()
        .ok_or_else(|| BekiperError::Data("SLEAP CSV is empty".into()))??;

    let mut labels = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for col in header.split(',') {
        let col = col.trim();
        // Match columns ending in ".x" to extract node names
        if let Some(node_name) = col.strip_suffix(".x")
            && !node_name.is_empty()
            && seen.insert(node_name.to_string())
        {
            labels.push(node_name.to_string());
        }
    }

    if labels.is_empty() {
        return Err(BekiperError::Data(
            "No node labels found in SLEAP CSV header".into(),
        ));
    }

    Ok(labels)
}
