use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::{PerimeterSpec, PerimetersFile, json_value_to_toml_string};
use bekiper_core::types::DetectionBackend;
use bekiper_pose::video::VideoDecoder;

use crate::audit::{AuditDiagnostic, AuditSeverity};
use crate::config::{ExperimentConfig, IngressConfig};
use crate::stage::{ExperimentOutputDir, Stage};

use super::helpers::resolve_python;
use super::inputs::{PerimeterInputSource, PipelineInputResolver};
use super::{normalize_runtime_paths, resolve_runtime_path};

// ---------------------------------------------------------------------------
// Dry-run: validate detection plan without running anything
// ---------------------------------------------------------------------------

pub fn dry_run_detect(config_path: &Path) -> Result<()> {
    let summary = analyze_detection_dry_run(config_path)?;
    summary.print();
    if summary.errors > 0 {
        return Err(BekiperError::Data(format!(
            "detection dry run found {} error(s)",
            summary.errors
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DryRunSeverity {
    Ok,
    Skipped,
    #[allow(dead_code)]
    Warning,
    Error,
}

impl DryRunSeverity {
    fn label(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Skipped => "SKIP",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DryRunDiagnostic {
    severity: DryRunSeverity,
    category: String,
    experiment: String,
    message: String,
}

#[derive(Debug, Default)]
pub struct DetectionDryRunSummary {
    pub total_experiments: usize,
    pub skipped_experiments: usize,
    pub ok: usize,
    pub warnings: usize,
    pub errors: usize,
    pub diagnostics: Vec<DryRunDiagnostic>,
}

include!("dry_run_detect/summary.rs");

include!("dry_run_detect/analysis.rs");

include!("dry_run_detect/video_probe.rs");

include!("dry_run_detect/manual_precomputed.rs");

include!("dry_run_detect/automatic.rs");

include!("dry_run_detect/overlay.rs");

include!("dry_run_detect/backend_dispatch.rs");
