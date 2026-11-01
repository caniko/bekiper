use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::annotation::{AnnotationResolver, AnnotationSession};
use crate::config::{ExperimentConfig, IngressConfig, PoseLabelModel, PoseQualityConfig};
use crate::coordinate_diagnostics::{coordinate_bbox, vertex_bbox_string};
use crate::stage::{ExperimentOutputDir, Stage};
#[cfg(feature = "pose-gpu")]
use crate::workflow::resolve_model_spec;
use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::{PerimeterSpec, PerimetersFile};
use bekiper_core::shape::Shape;
use bekiper_core::types::PoseModelPreset;
use bekiper_core::types::{DetectionBackend, TaskType};
use bekiper_perimeter::polygon::Polygon;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

mod likelihood;
pub use likelihood::check_likelihood;
mod population;
pub use population::{PopulationLedger, population_ledger_at};
pub mod reconciliation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditSeverity {
    Ok,
    Skipped,
    Warning,
    Error,
}

impl AuditSeverity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Skipped => "SKIP",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditDiagnostic {
    pub severity: AuditSeverity,
    pub category: String,
    pub experiment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<PathBuf>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix_hint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_command: Option<String>,
}

impl AuditDiagnostic {
    pub fn new(
        severity: AuditSeverity,
        category: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            category: category.into(),
            experiment: None,
            artifact_path: None,
            message: message.into(),
            cause: None,
            fix_hint: None,
            validation_command: None,
        }
    }

    pub fn experiment(mut self, experiment: impl Into<String>) -> Self {
        self.experiment = Some(experiment.into());
        self
    }

    pub fn artifact_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.artifact_path = Some(path.into());
        self
    }

    pub fn cause(mut self, cause: impl Into<String>) -> Self {
        self.cause = Some(cause.into());
        self
    }

    pub fn fix_hint(mut self, fix_hint: impl Into<String>) -> Self {
        self.fix_hint = Some(fix_hint.into());
        self
    }

    pub fn validation_command(mut self, command: impl Into<String>) -> Self {
        self.validation_command = Some(command.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditReport {
    pub name: String,
    pub config_path: Option<PathBuf>,
    pub diagnostics: Vec<AuditDiagnostic>,
    /// Pre-filter population from the same bounded observation as this audit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population: Option<PopulationLedger>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodAuditReport {
    pub name: String,
    pub config_path: Option<PathBuf>,
    pub diagnostics: Vec<AuditDiagnostic>,
    pub summary: LikelihoodSummary,
    pub worst_pairs: Vec<LikelihoodPairRow>,
    pub worst_labels: Vec<LikelihoodAggregateRow>,
    pub worst_experiments: Vec<LikelihoodAggregateRow>,
    pub task_phase_summary: Vec<LikelihoodAggregateRow>,
}

impl LikelihoodAuditReport {
    pub fn new(name: impl Into<String>, config_path: Option<PathBuf>) -> Self {
        Self {
            name: name.into(),
            config_path,
            diagnostics: Vec::new(),
            summary: LikelihoodSummary::default(),
            worst_pairs: Vec::new(),
            worst_labels: Vec::new(),
            worst_experiments: Vec::new(),
            task_phase_summary: Vec::new(),
        }
    }

    pub fn push(&mut self, diagnostic: AuditDiagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn counts(&self) -> AuditCounts {
        AuditCounts::from_diagnostics(&self.diagnostics)
    }

    pub fn has_errors(&self) -> bool {
        self.counts().errors > 0
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodSummary {
    pub all_labels: bool,
    pub pose_label_model: PoseLabelModel,
    pub focused_labels: Vec<String>,
    pub experiments_expected: usize,
    pub experiments_inspected: usize,
    pub experiments_missing_poses: usize,
    pub total_frames: usize,
    pub likelihood_threshold: f64,
    pub labels_scored: usize,
    pub labels_skipped_missing_likelihood: usize,
    pub low_likelihood_frames: usize,
    pub scored_likelihood_values: usize,
    pub augmented_missing_count: usize,
}

impl LikelihoodSummary {
    pub fn low_likelihood_fraction(&self) -> f64 {
        fraction(self.low_likelihood_frames, self.scored_likelihood_values)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodPairRow {
    pub experiment: String,
    pub label: String,
    pub frames: usize,
    pub low_likelihood_frames: usize,
    pub low_likelihood_fraction: f64,
    pub mean_likelihood: Option<f64>,
    pub min_likelihood: Option<f64>,
    pub max_likelihood: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodAggregateRow {
    pub key: String,
    pub experiments: usize,
    pub labels: usize,
    pub frames: usize,
    pub low_likelihood_frames: usize,
    pub low_likelihood_fraction: f64,
    pub mean_likelihood: Option<f64>,
}

impl AuditReport {
    pub fn new(name: impl Into<String>, config_path: Option<PathBuf>) -> Self {
        Self {
            name: name.into(),
            config_path,
            diagnostics: Vec::new(),
            population: None,
        }
    }

    pub fn push(&mut self, diagnostic: AuditDiagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn extend(&mut self, diagnostics: impl IntoIterator<Item = AuditDiagnostic>) {
        self.diagnostics.extend(diagnostics);
    }

    pub fn counts(&self) -> AuditCounts {
        AuditCounts::from_diagnostics(&self.diagnostics)
    }

    pub fn has_errors(&self) -> bool {
        self.counts().errors > 0
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditCounts {
    pub ok: usize,
    pub skipped: usize,
    pub warnings: usize,
    pub errors: usize,
}

impl AuditCounts {
    fn from_diagnostics(diagnostics: &[AuditDiagnostic]) -> Self {
        let mut counts = Self::default();
        for diagnostic in diagnostics {
            match diagnostic.severity {
                AuditSeverity::Ok => counts.ok += 1,
                AuditSeverity::Skipped => counts.skipped += 1,
                AuditSeverity::Warning => counts.warnings += 1,
                AuditSeverity::Error => counts.errors += 1,
            }
        }
        counts
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigAuditSummary {
    pub config_path: PathBuf,
    pub experiments: usize,
    pub scan_entries: usize,
    pub animal_scan_entries: usize,
    pub ingress_profile_entries: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManualAnnotationCoverageSummary {
    pub config_path: PathBuf,
    pub total_manual_sessions: usize,
    pub assigned_sessions: usize,
    pub missing_sessions: Vec<ManualAnnotationCoverageFailure>,
}

impl ManualAnnotationCoverageSummary {
    pub fn is_complete(&self) -> bool {
        self.missing_sessions.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManualAnnotationCoverageFailure {
    pub experiment: String,
    pub video_path: Option<PathBuf>,
    pub annotation_dir: Option<PathBuf>,
    pub error: String,
}

fn load_audit_config(
    path: &Path,
) -> std::result::Result<(IngressConfig, PathBuf), Box<AuditDiagnostic>> {
    let base_dir = path.parent().unwrap_or(Path::new("."));
    let mut config = IngressConfig::from_file(path).map_err(|error| {
        Box::new(
            AuditDiagnostic::new(AuditSeverity::Error, "config", "config validation failed")
                .artifact_path(path)
                .cause(error.to_string())
                .fix_hint("restore the project config or pass the correct path with -c")
                .validation_command(format!("bekiper check readiness -c {}", path.display())),
        )
    })?;
    config.expand_for_validation(base_dir, Stage::all()).map_err(|error| {
        Box::new(AuditDiagnostic::new(AuditSeverity::Error, "config", "config expansion failed")
                .artifact_path(path)
                .cause(error.to_string())
                .fix_hint("fix the referenced config path, model, data, annotation, or perimeter artifact")
                .validation_command(format!("bekiper check readiness -c {}", path.display())))
    })?;
    let output_root = base_dir.join(
        config
            .project
            .output_dir
            .as_deref()
            .unwrap_or(Path::new("output")),
    );
    Ok((config, output_root))
}

fn focused_likelihood_labels_for_config(config: &IngressConfig) -> Vec<String> {
    let mut labels = HashSet::new();
    for experiment in &config.experiments {
        for label in focused_likelihood_labels_for_task(
            config.pipeline.pose_label_model,
            experiment.task_type,
        ) {
            labels.insert(label.to_string());
        }
    }
    let mut labels = labels.into_iter().collect::<Vec<_>>();
    labels.sort();
    labels
}

fn focused_likelihood_labels_for_task(
    model: PoseLabelModel,
    task_type: TaskType,
) -> &'static [&'static str] {
    match (model, task_type) {
        (PoseLabelModel::ImenDlc, TaskType::ObjectRecognition | TaskType::OpenField) => {
            &["nose", "left_ear", "right_ear", "tail"]
        }
        (PoseLabelModel::ImenDlc, TaskType::YMaze) => {
            &["nose", "left_ear", "right_ear", "base_tail"]
        }
        (PoseLabelModel::ImenDlc, TaskType::Cheeseboard) => &["nose", "left_ear", "right_ear"],
        (_, TaskType::ObjectRecognition) => &[
            "mouse_center",
            "nose",
            "center_ear",
            "head_midpoint",
            "tail_base",
        ],
        (_, TaskType::OpenField) => &["mouse_center", "nose", "tail_base"],
        (_, TaskType::YMaze) => &["center_ear", "head_midpoint", "tail_base"],
        (_, TaskType::Cheeseboard) => &[
            "mouse_center",
            "nose",
            "center_ear",
            "head_midpoint",
            "tail_base",
        ],
    }
}

fn likelihood_pair_row(
    df: &DataFrame,
    experiment: &str,
    label: &str,
    threshold: f64,
) -> Option<LikelihoodPairRow> {
    let name = format!("{label}_likelihood");
    let series = df.column(&name).ok()?;
    let values = series.f64().ok()?;

    let mut frames = 0usize;
    let mut low = 0usize;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for value in values {
        frames += 1;
        match value {
            Some(v) => {
                if v < threshold {
                    low += 1;
                }
                sum += v;
                min = min.min(v);
                max = max.max(v);
            }
            None => low += 1,
        }
    }

    let finite = min.is_finite() && max.is_finite();
    Some(LikelihoodPairRow {
        experiment: experiment.to_string(),
        label: label.to_string(),
        frames,
        low_likelihood_frames: low,
        low_likelihood_fraction: fraction(low, frames),
        mean_likelihood: finite.then_some(sum / frames as f64),
        min_likelihood: finite.then_some(min),
        max_likelihood: finite.then_some(max),
    })
}

fn top_likelihood_pairs(mut rows: Vec<LikelihoodPairRow>, limit: usize) -> Vec<LikelihoodPairRow> {
    rows.sort_by(|a, b| {
        b.low_likelihood_fraction
            .total_cmp(&a.low_likelihood_fraction)
            .then_with(|| b.low_likelihood_frames.cmp(&a.low_likelihood_frames))
            .then_with(|| a.experiment.cmp(&b.experiment))
            .then_with(|| a.label.cmp(&b.label))
    });
    rows.truncate(limit);
    rows
}

fn task_phase_key(experiment: &str) -> Option<String> {
    let task = if experiment.starts_with("nort_videos_") {
        "nort"
    } else if experiment.starts_with("oft_videos_") {
        "oft"
    } else if experiment.starts_with("y_maze_videos_") {
        "y_maze"
    } else {
        return None;
    };

    let phase = if experiment.contains("_pre_") {
        "pre"
    } else if experiment.contains("_post_") {
        "post"
    } else {
        "unknown"
    };

    Some(format!("{task}/{phase}"))
}

fn scan_parquet_df(path: &Path) -> Result<DataFrame> {
    LazyFrame::scan_parquet(path.to_str().unwrap().into(), Default::default())
        .map_err(|e| BekiperError::Data(format!("scan {}: {e}", path.display())))?
        .collect()
        .map_err(|e| BekiperError::Data(format!("collect {}: {e}", path.display())))
}

fn audit_exp_output_dir(output_root: &Path, experiment: &ExperimentConfig) -> ExperimentOutputDir {
    match experiment.animal_id {
        Some(id) => ExperimentOutputDir::for_animal(output_root, id, &experiment.name),
        None => ExperimentOutputDir::new(output_root, &experiment.name),
    }
}

fn discover_pose_labels(df: &DataFrame) -> Vec<String> {
    let names: HashSet<String> = df
        .get_column_names()
        .iter()
        .map(|name| name.as_str().to_string())
        .collect();
    let mut labels = Vec::new();
    for name in &names {
        if let Some(label) = name.strip_suffix("_x")
            && names.contains(&format!("{label}_y"))
        {
            labels.push(label.to_string());
        }
    }
    labels.sort();
    labels
}

mod validation;
use validation::{clearer_manual_annotation_error, resolve_config_path};
pub use validation::{validate_config_file, validate_config_file_at};

mod coverage;
pub use coverage::check_manual_annotation_coverage;

mod readiness;
pub use readiness::check_readiness;
use readiness::path_readiness_diagnostic;

mod external;
pub use external::check_external;
#[cfg(test)]
use external::external_command_result;
use external::stderr_tail;

mod quality;
pub use quality::check_pose_quality;
#[cfg(test)]
use quality::compute_pose_quality_metrics;
use quality::fraction;

mod alignment;
use alignment::ymaze_alignment_diagnostics;

mod doctor;
use doctor::doctor_diagnostics;
pub use doctor::doctor_report;

#[cfg(test)]
#[path = "audit/regressions.rs"]
mod tests;
