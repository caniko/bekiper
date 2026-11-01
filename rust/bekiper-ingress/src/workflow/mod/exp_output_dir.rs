use std::path::{Path, PathBuf};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::video::{FpsConfig, FrameTiming};
use bekiper_pose::infer::OnnxPoseModel;

use crate::aggregate::{is_aggregate_fresh, run_aggregate};
use crate::config::{ExperimentConfig, IngressConfig};
use crate::parallel::ParallelScheduler;
use crate::stage::{ExperimentOutputDir, Stage, aggregate_output_path, aggregate_state_path};
use crate::stage_share::{Shared, SharedRun};
use inputs::{InputArtifactState, PerimeterInputSource, PipelineInputResolver};

/// Create the output directory for an experiment, grouping by animal ID
/// when the experiment was produced by `[[animal_scan]]`.
fn exp_output_dir(output_root: &Path, experiment: &ExperimentConfig) -> ExperimentOutputDir {
    match experiment.animal_id {
        Some(id) => ExperimentOutputDir::for_animal(output_root, id, &experiment.name),
        None => ExperimentOutputDir::new(output_root, &experiment.name),
    }
}

pub use detect::{analyze_detection_dry_run, dry_run_detect, overlay_annotation};

use augment::{run_augment, run_augment_with_model};
use behaviour::run_behaviour;
use detect::run_detect_filtered;
use inspect::run_inspect;
use pose::{run_pose, run_pose_with_model_gpu};
use summary::run_summary;

pub(crate) use pose::resolve_model_spec;

#[derive(Debug, Clone, PartialEq, Eq)]
struct StageExecutionPlan {
    sequential_shared_model: Vec<Stage>,
    parallel_cpu: Vec<Stage>,
}

/// Options controlling which stages to run.
#[derive(Default)]
pub struct RunOptions {
    pub measurements: crate::config::MeasurementOverrides,
    pub ymaze_dwell: bekiper_feature::tolerance::DwellOverrides,
    /// Generic `--set KEY=VALUE` overrides (highest precedence).
    pub set: Vec<String>,
    /// Run only stages from this one onward.
    pub from_stage: Option<Stage>,
    /// Run only this single stage.
    pub only_stage: Option<Stage>,
    /// Skip the inspect stage.
    pub no_inspect: bool,
    /// Constructor/CLI selection overrides environment and TOML selections.
    pub inspection_video_trials: Option<Vec<String>>,
    /// Skip the aggregate stage even if selected by the stage plan.
    pub no_aggregate: bool,
    /// Override the aggregate spreadsheet filename.
    pub aggregate_output: Option<String>,
    /// Override pipeline Y-axis inversion.
    pub invert_y: Option<bool>,
    /// Force re-run even if outputs are fresh.
    pub force: bool,
    /// Delete augment and downstream outputs before running from augment.
    pub refresh_augment: bool,
    /// Do not run detection. Required detection artifacts must already exist.
    pub no_detect: bool,
    /// Do not run automatic detection backends; manual/precomputed sources are allowed.
    pub no_auto_detect: bool,
    /// Only process experiments belonging to the animal at this index (0-based).
    pub animal_index: Option<usize>,
    /// Override the experiment-level Rayon thread cap.
    pub num_threads: Option<usize>,
    /// Override host-RAM-aware scheduling.
    pub memory_scheduler_enabled: Option<bool>,
    /// Override the RAM admission threshold.
    pub max_ram_usage_fraction: Option<f64>,
}

include!("exp_output_dir/run.rs");

include!("exp_output_dir/purge.rs");

include!("exp_output_dir/detection_outputs.rs");

include!("exp_output_dir/exclusions.rs");

include!("exp_output_dir/manual_preflight.rs");

include!("exp_output_dir/auto_preflight.rs");

include!("exp_output_dir/runtime_paths.rs");

include!("exp_output_dir/video_timing.rs");

include!("exp_output_dir/stage_selection.rs");

include!("exp_output_dir/runtime_path_tests.rs");

include!("exp_output_dir/shared_tests.rs");
