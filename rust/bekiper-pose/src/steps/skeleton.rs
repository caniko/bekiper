use std::collections::{BTreeSet, HashMap};

use polars::prelude::*;

use bekiper_core::error::{BekiperError, Result};

use crate::pipeline::PipelineStep;
use crate::skeleton::Skeleton;

#[derive(Debug, Clone)]
struct FittedEdgeBaseline {
    source_idx: usize,
    target_idx: usize,
    median: f64,
    mad: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameSkeletonViolations {
    pub row_index: usize,
    pub frame: i64,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SkeletonViolationEvaluation {
    label_order: Vec<String>,
    label_to_index: HashMap<String, usize>,
    flags: Vec<Vec<bool>>,
    counts: Vec<u32>,
}

include!("skeleton/violations.rs");

include!("skeleton/fitted_repair.rs");

#[derive(Debug, Clone)]
struct LabelCoordinates {
    xs: Vec<Option<f64>>,
    ys: Vec<Option<f64>>,
}

/// Null out anatomically implausible keypoints using the configured skeleton topology.
#[derive(Debug, Clone)]
pub struct SkeletonRepairStep {
    pub skeleton: Skeleton,
    pub relative_tolerance: f64,
    pub mad_multiplier: f64,
    pub min_baseline_frames: usize,
}

include!("skeleton/fitting.rs");

include!("skeleton/pipeline_step.rs");

include!("skeleton/statistics.rs");
