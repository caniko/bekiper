use std::collections::HashMap;
#[cfg(feature = "pose-video")]
use std::path::Path;

use bekiper_behaviour::AnalysisResult;
use bekiper_behaviour::TaskDetail;
use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::{
    CoordinateColumnSpec, HeuristicDecompositionSpec, HeuristicMeta, InspectionManifest,
    InspectionSettings, PerimeterSpec,
};
#[cfg(feature = "pose-gpu")]
use bekiper_core::inspection::{json_value_to_toml_string, toml_str_to_json_value};
use bekiper_core::scale::ScaleDerivationInfo;
use bekiper_core::types::{MetersPerPixel, TaskType};
use bekiper_core::video::VideoMetadata;
use bekiper_feature::heuristic::HeuristicSummary;
#[cfg(feature = "pose-video")]
use bekiper_pose::video::VideoDecoder;
use polars::prelude::DataFrame;

use crate::config::ExperimentConfig;
#[cfg(feature = "pose-gpu")]
use crate::config::IngressConfig;
use crate::coordinate_diagnostics::{coordinate_bbox, vertex_bbox_string};
#[path = "inspection_states.rs"]
mod inspection_states;
#[cfg(feature = "pose-gpu")]
use crate::export::export_inspection;
#[cfg(feature = "pose-video")]
use crate::stage::ExperimentOutputDir;

#[cfg(feature = "pose-gpu")]
use super::helpers::{load_perimeters, load_scale_info};

// ---------------------------------------------------------------------------
// Stage 4: Summary statistics from behavioural results
// ---------------------------------------------------------------------------

include!("summary/run.rs");

include!("summary/ymaze_validation.rs");

include!("summary/manifest.rs");

include!("summary/tracking.rs");

include!("summary/trial_assembly.rs");

include!("summary/frames.rs");

#[cfg(test)]
#[path = "summary/skeleton_inspection_tests.rs"]
mod skeleton_inspection_tests;
