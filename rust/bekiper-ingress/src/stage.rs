use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use bekiper_core::error::{BekiperError, Result};

use crate::config::normalize_spreadsheet_filename;
use crate::stage_share::{Shared, SharedLayout};

/// Named pipeline stages in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    Detect,
    Pose,
    Augment,
    Behaviour,
    Summary,
    Inspect,
    /// Cross-trial consolidation: walks the run root after every per-trial
    /// stage finishes and writes a single spreadsheet combining all `result.toml`
    /// files. Unlike the other stages, this is a batch step that runs once
    /// per project, not once per experiment.
    Aggregate,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Stage {
    pub fn name(&self) -> &'static str {
        match self {
            Stage::Detect => "detect",
            Stage::Pose => "pose",
            Stage::Augment => "augment",
            Stage::Behaviour => "behaviour",
            Stage::Summary => "summary",
            Stage::Inspect => "inspect",
            Stage::Aggregate => "aggregate",
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "detect" => Ok(Stage::Detect),
            "pose" => Ok(Stage::Pose),
            "augment" => Ok(Stage::Augment),
            "behaviour" => Ok(Stage::Behaviour),
            "summary" => Ok(Stage::Summary),
            "inspect" => Ok(Stage::Inspect),
            "aggregate" => Ok(Stage::Aggregate),
            _ => Err(BekiperError::Config(format!("unknown stage: {name}"))),
        }
    }

    /// All stages in execution order.
    pub fn all() -> &'static [Stage] {
        &[
            Stage::Detect,
            Stage::Pose,
            Stage::Augment,
            Stage::Behaviour,
            Stage::Summary,
            Stage::Inspect,
            Stage::Aggregate,
        ]
    }

    /// Stages from `start` onward (inclusive).
    pub fn from_onward(start: Stage) -> Vec<Stage> {
        Stage::all()
            .iter()
            .copied()
            .filter(|s| *s >= start)
            .collect()
    }
}

/// All outputs materially owned by one pipeline stage.
///
/// `primary_output` is the path used for freshness checks. `owned_paths`
/// is the full cleanup surface for `bekiper data purge-cache`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageArtifacts {
    pub primary_output: PathBuf,
    pub owned_paths: Vec<PathBuf>,
}

/// Output directory layout for a single experiment.
///
/// ```text
/// output/data/<experiment>/
///   augmented.parquet          ← augment
///   behaviour.parquet          ← behaviour
///   <experiment>_evaluated.parquet  ← summary (inspection export)
///   result.toml                ← summary (consolidated metrics + metadata)
///   figures/                   ← inspect (only if [project.figures] enabled = true)
///
/// checkpoint/<experiment>/
///   perimeters.toml            ← detect (intermediate)
///   poses.parquet              ← pose (checkpoint)
///   detection_frame.png        ← detect (debug)
///   frame_timing.toml          ← video probe (cache)
///   scale.toml                 ← augment (intermediate stage handoff)
///   behaviour_summaries.toml   ← behaviour (intermediate stage handoff)
///   behaviour_result.toml      ← behaviour (intermediate stage handoff)
///   pose_chunks/               ← pose inference (incremental)
///   debug/                     ← detection debug artifacts
///     debug.toml               ← metadata (backend, blob counts/areas)
///     threshold.png            ← (classical CV) binary threshold mask
///     post_mouse.png           ← (classical CV) binary mask after mouse removal
///     candidates.png           ← (classical CV) overlay of all candidate blobs
///     masks/                   ← (SAM2) individual mask PNGs + scores
///     overlay.png              ← (SAM2) all masks overlaid on detection frame
/// ```
#[derive(Debug, Clone)]
pub struct ExperimentOutputDir {
    root: PathBuf,
    /// The top-level output directory, used to derive sibling checkpoint paths.
    output_root: PathBuf,
    /// Full experiment name (e.g. `1035_seq0_oultraining`), used for
    /// checkpoint subdirectory naming.
    experiment_name: String,
    /// Set on a shared view: probe/detect/pose/augment outputs live in
    /// `checkpoint/<recording>/<stage>-<fp8>/` instead of `checkpoint/<experiment>/`.
    shared: Option<SharedLayout>,
    /// Directory holding representative frames shared by the variants of a recording.
    frame_cache: Option<PathBuf>,
}

impl AsRef<Path> for ExperimentOutputDir {
    fn as_ref(&self) -> &Path {
        &self.root
    }
}

impl ExperimentOutputDir {
    pub fn new(output_root: &Path, experiment_name: &str) -> Self {
        Self {
            root: output_root.join("data").join(experiment_name),
            output_root: output_root.to_path_buf(),
            experiment_name: experiment_name.to_string(),
            shared: None,
            frame_cache: None,
        }
    }

    /// Create an output dir grouped under the animal ID:
    /// `{output_root}/data/{animal_id}/{sequence_name}/`.
    ///
    /// The *experiment_name* has the format `{animal_id}_seq{N}_{recipe}`;
    /// the `{animal_id}_` prefix is stripped to form the subdirectory name.
    pub fn for_animal(output_root: &Path, animal_id: u32, experiment_name: &str) -> Self {
        let prefix = format!("{}_", animal_id);
        let subdir = experiment_name
            .strip_prefix(&prefix)
            .unwrap_or(experiment_name);
        Self {
            root: output_root
                .join("data")
                .join(animal_id.to_string())
                .join(subdir),
            output_root: output_root.to_path_buf(),
            experiment_name: experiment_name.to_string(),
            shared: None,
            frame_cache: None,
        }
    }

    /// A view of this experiment whose shared stages read and write the
    /// recording-level checkpoints of `layout`.
    pub fn shared_view(&self, layout: SharedLayout) -> Self {
        Self {
            shared: Some(layout),
            ..self.clone()
        }
    }

    pub fn shared_layout(&self) -> Option<&SharedLayout> {
        self.shared.as_ref()
    }

    pub fn with_frame_cache(mut self, dir: PathBuf) -> Self {
        self.frame_cache = Some(dir);
        self
    }

    pub fn frame_cache_dir(&self) -> Option<&Path> {
        self.frame_cache.as_deref()
    }

    /// Checkpoint directory of a shared stage: the fingerprinted
    /// `checkpoint/<recording>/<stage>-<fp8>/` on a shared view, the
    /// experiment checkpoint directory in the legacy layout.
    pub fn checkpoint_for(&self, kind: Shared) -> PathBuf {
        self.kind_dir(kind)
    }

    fn kind_dir(&self, kind: Shared) -> PathBuf {
        match &self.shared {
            Some(layout) => layout.dir(&self.output_root, kind),
            None => self.checkpoints_dir(),
        }
    }

    fn ensure_shared_dirs(&self) -> Result<()> {
        if self.shared.is_some() {
            for kind in Shared::ALL {
                std::fs::create_dir_all(self.kind_dir(kind))?;
            }
        }
        Ok(())
    }

    /// The output whose existence marks a shared stage as produced.
    pub fn shared_primary(&self, kind: Shared) -> PathBuf {
        match kind {
            Shared::Probe => self.video_info_toml(),
            Shared::Detect => self.pixel_perimeters_toml(),
            Shared::Pose => self.poses_parquet(),
            Shared::Augment => self.augmented_parquet(),
        }
    }

    /// Every path a shared stage owns, in a fixed order so a shared view and a
    /// legacy layout pair up index by index.
    pub fn shared_artifacts(&self, kind: Shared) -> Vec<PathBuf> {
        match kind {
            Shared::Probe => vec![self.video_info_toml(), self.frame_timing_toml()],
            Shared::Detect => vec![
                self.pixel_perimeters_toml(),
                self.detection_frame_png(),
                self.debug_dir(),
                self.annotation_scale_toml(),
            ],
            Shared::Pose => vec![self.poses_parquet()],
            Shared::Augment => vec![
                self.augmented_parquet(),
                self.perimeters_toml(),
                self.scale_toml(),
            ],
        }
    }

    /// Whether this shared view holds a completed `kind` output.
    pub fn shared_sealed(&self, kind: Shared) -> bool {
        self.shared
            .as_ref()
            .is_some_and(|layout| layout.is_sealed(&self.output_root, kind))
            && self.shared_primary(kind).exists()
    }

    /// Probed video resolution, cached next to the frame timing.
    pub fn video_info_toml(&self) -> PathBuf {
        self.kind_dir(Shared::Probe).join("video_info.toml")
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn experiment_name(&self) -> &str {
        &self.experiment_name
    }

    pub fn output_root(&self) -> &Path {
        &self.output_root
    }

    /// Pixel-space perimeters produced by detect/manual/precomputed sources.
    /// Consumed by pose crop and augment, which both operate before metric
    /// coordinate conversion.
    pub fn pixel_perimeters_toml(&self) -> PathBuf {
        self.kind_dir(Shared::Detect).join("pixel_perimeters.toml")
    }

    /// Meter-space perimeters produced by augment and consumed by downstream
    /// behaviour, summary, and inspection stages.
    pub fn perimeters_toml(&self) -> PathBuf {
        self.kind_dir(Shared::Augment).join("perimeters.toml")
    }

    pub fn poses_parquet(&self) -> PathBuf {
        self.kind_dir(Shared::Pose).join("poses.parquet")
    }

    pub fn augmented_parquet(&self) -> PathBuf {
        match self.shared {
            Some(_) => self.kind_dir(Shared::Augment).join("augmented.parquet"),
            None => self.root.join("augmented.parquet"),
        }
    }

    /// Resolved meters-per-pixel value, persisted during augmentation.
    /// Lives under `checkpoint/<experiment>/` — intermediate handoff
    /// from augment → summary, not a user-facing result.
    pub fn scale_toml(&self) -> PathBuf {
        self.kind_dir(Shared::Augment).join("scale.toml")
    }

    /// Annotation-derived scale calibration, written during manual annotation detection.
    pub fn annotation_scale_toml(&self) -> PathBuf {
        match self.shared {
            Some(_) => self.kind_dir(Shared::Detect).join("annotation_scale.toml"),
            None => self.root.join("annotation_scale.toml"),
        }
    }

    pub fn behaviour_parquet(&self) -> PathBuf {
        self.root.join("behaviour.parquet")
    }

    /// Per-heuristic counts/seconds persisted by the behaviour stage.
    /// Intermediate handoff to the summary stage — lives in checkpoints.
    pub fn behaviour_summaries_toml(&self) -> PathBuf {
        self.checkpoints_dir().join("behaviour_summaries.toml")
    }

    /// Task-specific detailed results (zone dwell, indices, …) persisted by
    /// the behaviour stage. Intermediate handoff to the summary stage.
    pub fn behaviour_result_toml(&self) -> PathBuf {
        self.checkpoints_dir().join("behaviour_result.toml")
    }

    /// Consolidated per-animal result file: all metrics + metadata in one TOML.
    /// Written by the summary stage; read by downstream tooling and the Python
    /// inspection package.
    pub fn result_toml(&self) -> PathBuf {
        self.root.join("result.toml")
    }

    pub fn inspection_toml(&self) -> PathBuf {
        self.root
            .join(format!("{}_inspection.toml", self.experiment_name))
    }

    pub fn evaluated_parquet(&self) -> PathBuf {
        self.root
            .join(format!("{}_evaluated.parquet", self.experiment_name))
    }

    pub fn spreadsheet_workbook(&self) -> PathBuf {
        self.root.join(format!("{}.ods", self.experiment_name))
    }

    pub fn excel_workbook(&self) -> PathBuf {
        self.spreadsheet_workbook()
    }

    /// Frame timing info (auto-detected FPS, VFR timestamps).
    /// Lives under `checkpoint/<experiment>/` — intermediate cache, not a final result.
    pub fn frame_timing_toml(&self) -> PathBuf {
        self.kind_dir(Shared::Probe).join("frame_timing.toml")
    }

    /// Per-experiment checkpoint directory under sibling `checkpoint/<experiment>/`.
    /// Always relative to the top-level output root, even when the experiment
    /// is nested under an animal subdirectory.
    pub fn checkpoints_dir(&self) -> PathBuf {
        checkpoint_root(&self.output_root).join(&self.experiment_name)
    }

    pub fn results_parquet(&self) -> PathBuf {
        self.root.join("results.parquet")
    }

    pub fn figures_dir(&self) -> PathBuf {
        self.root.join("figures")
    }

    pub fn videos_dir(&self) -> PathBuf {
        self.root.join("videos")
    }

    /// Directory for incremental pose inference chunks.
    /// Each batch is written as `batch_NNNN.parquet` during inference.
    /// On completion, chunks are merged into `poses.parquet` and this dir is removed.
    ///
    /// Lives under `checkpoint/<experiment>/pose_chunks/` to keep the experiment dir clean.
    pub fn pose_chunks_dir(&self) -> PathBuf {
        self.kind_dir(Shared::Pose).join("pose_chunks")
    }

    /// Path to the exported detection frame image (debug artifact).
    pub fn detection_frame_png(&self) -> PathBuf {
        self.kind_dir(Shared::Detect).join("detection_frame.png")
    }

    /// Directory for pre-extracted representative video frames (PNG).
    /// Created during the summary stage so the Python inspector can load
    /// frames without decoding from the source MP4.
    pub fn frames_dir(&self) -> PathBuf {
        self.root.join("frames")
    }

    /// Directory for detection debug artifacts (always written).
    pub fn debug_dir(&self) -> PathBuf {
        self.kind_dir(Shared::Detect).join("debug")
    }

    /// Ensure the output directory exists.
    pub fn ensure_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.root)?;
        self.ensure_shared_dirs()?;
        Ok(())
    }

    /// Ensure the checkpoints directory exists. Callers that write to
    /// `scale_toml()`, `behaviour_summaries_toml()`, etc. must invoke this first.
    pub fn ensure_checkpoints_dir(&self) -> Result<()> {
        std::fs::create_dir_all(self.checkpoints_dir())?;
        self.ensure_shared_dirs()?;
        Ok(())
    }

    /// Stage-owned outputs rooted at this experiment directory.
    pub fn stage_artifacts(&self, stage: Stage) -> Option<StageArtifacts> {
        match stage {
            Stage::Detect => {
                let primary_output = self.pixel_perimeters_toml();
                let owned_paths = vec![
                    primary_output.clone(),
                    self.detection_frame_png(),
                    self.debug_dir(),
                    self.annotation_scale_toml(),
                ];
                Some(StageArtifacts {
                    primary_output,
                    owned_paths,
                })
            }
            Stage::Pose => {
                let primary_output = self.poses_parquet();
                let owned_paths = vec![
                    primary_output.clone(),
                    self.pose_chunks_dir(),
                    self.frame_timing_toml(),
                ];
                Some(StageArtifacts {
                    primary_output,
                    owned_paths,
                })
            }
            Stage::Augment => {
                let primary_output = self.augmented_parquet();
                let owned_paths = vec![
                    primary_output.clone(),
                    self.perimeters_toml(),
                    self.scale_toml(),
                ];
                Some(StageArtifacts {
                    primary_output,
                    owned_paths,
                })
            }
            Stage::Behaviour => {
                let primary_output = self.behaviour_parquet();
                let owned_paths = vec![
                    primary_output.clone(),
                    self.behaviour_summaries_toml(),
                    self.behaviour_result_toml(),
                ];
                Some(StageArtifacts {
                    primary_output,
                    owned_paths,
                })
            }
            Stage::Summary => {
                let primary_output = self.result_toml();
                let owned_paths = vec![
                    primary_output.clone(),
                    self.results_parquet(),
                    self.inspection_toml(),
                    self.evaluated_parquet(),
                    self.spreadsheet_workbook(),
                    self.frames_dir(),
                ];
                Some(StageArtifacts {
                    primary_output,
                    owned_paths,
                })
            }
            Stage::Inspect => {
                let primary_output = self.root.join("inspection_receipt.json");
                let owned_paths = vec![
                    primary_output.clone(),
                    self.figures_dir(),
                    self.videos_dir(),
                ];
                Some(StageArtifacts {
                    primary_output,
                    owned_paths,
                })
            }
            Stage::Aggregate => None,
        }
    }

    /// All stage-owned outputs that should be deleted when purging `start`
    /// and every downstream stage for a single experiment.
    pub fn purge_paths_from(&self, start: Stage) -> Vec<PathBuf> {
        let mut paths = BTreeSet::new();
        for stage in Stage::from_onward(start) {
            if let Some(artifacts) = self.stage_artifacts(stage) {
                for path in artifacts.owned_paths {
                    paths.insert(path);
                }
            }
        }
        paths.into_iter().collect()
    }

    /// Check if a stage output is fresh (exists and newer than its input).
    pub fn is_fresh(&self, stage: Stage, input_path: Option<&Path>) -> bool {
        // Aggregate is a project-level batch stage with no per-trial output —
        // freshness is tracked by the run-root workbook, not this directory,
        // so the per-trial check always reports "stale".
        if matches!(stage, Stage::Aggregate) {
            return false;
        }
        if stage == Stage::Inspect {
            let retained = std::fs::read_to_string(self.root.join("inspection_config.toml"))
                .ok()
                .and_then(|s| toml::from_str::<toml::Table>(&s).ok())
                .and_then(|mut t| t.remove("inspect"))
                .and_then(|v| v.try_into::<crate::config::InspectConfig>().ok());
            return retained.is_some_and(|settings| crate::inspection::is_fresh(self, &settings));
        }
        let Some(output) = self
            .stage_artifacts(stage)
            .map(|artifacts| artifacts.primary_output)
        else {
            return false;
        };

        // A shared output is fresh only if it was sealed under this fingerprint.
        if let (Some(layout), Some(kind)) = (&self.shared, Shared::of_stage(stage))
            && !layout.is_sealed(&self.output_root, kind)
        {
            return false;
        }

        if !output.exists() {
            return false;
        }

        let Some(input) = input_path else {
            return true;
        };

        // Compare modification times
        let Ok(output_mtime) = mtime(&output) else {
            return false;
        };
        let Ok(input_mtime) = mtime(input) else {
            return false;
        };

        output_mtime >= input_mtime
    }
}

/// Project-level aggregate spreadsheet output path.
pub fn aggregate_output_path(run_root: &Path, output_filename: &str) -> PathBuf {
    run_root.join(normalize_spreadsheet_filename(output_filename))
}

/// Sidecar state file tracking which effective aggregate config produced the
/// current spreadsheet. Used to invalidate freshness when sheet renames or
/// passthrough declarations change without touching per-trial `result.toml`
/// files.
pub fn aggregate_state_path(run_root: &Path, output_filename: &str) -> PathBuf {
    let normalized = normalize_spreadsheet_filename(output_filename);
    run_root.join(format!(".{normalized}.aggregate.toml"))
}

/// Project-level checkpoint root, kept as a sibling of the analytical output dir.
pub fn checkpoint_root(output_root: &Path) -> PathBuf {
    if output_root.file_name().is_some_and(|name| name == "output") {
        output_root
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join("checkpoint")
    } else {
        output_root.join("checkpoint")
    }
}

fn mtime(path: &Path) -> Result<SystemTime> {
    Ok(std::fs::metadata(path)?.modified()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_experiment_outputs_live_under_data_dir() {
        let exp_dir = ExperimentOutputDir::new(Path::new("output"), "trial_01");

        assert_eq!(exp_dir.root(), Path::new("output/data/trial_01"));
        assert_eq!(
            exp_dir.result_toml(),
            Path::new("output/data/trial_01/result.toml")
        );
        assert_eq!(exp_dir.checkpoints_dir(), Path::new("checkpoint/trial_01"));
    }

    #[test]
    fn animal_grouped_outputs_live_under_data_dir() {
        let exp_dir = ExperimentOutputDir::for_animal(Path::new("output"), 721, "721_seq0_ymaze");

        assert_eq!(exp_dir.root(), Path::new("output/data/721/seq0_ymaze"));
        assert_eq!(
            exp_dir.result_toml(),
            Path::new("output/data/721/seq0_ymaze/result.toml")
        );
        assert_eq!(
            exp_dir.checkpoints_dir(),
            Path::new("checkpoint/721_seq0_ymaze")
        );
    }
}
