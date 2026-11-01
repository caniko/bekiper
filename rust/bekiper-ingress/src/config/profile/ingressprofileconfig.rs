use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::IngressSessionContext;
use bekiper_core::scale::ScaleConfig;
use bekiper_core::types::{DetectionRecipe, TaskType};
use bekiper_core::video::FpsConfig;

use crate::annotation::AnnotationSession;

use super::IngressConfig;
use super::detection::{BackendPriority, DetectionConfig, default_backends};
use super::experiment::ExperimentConfig;
use super::io_helpers::*;
use super::scan::{
    AnimalScanConfig, AnnotationConfig, ScanConfig, default_center_zone_meters, default_extensions,
    parse_animal_metadata_csv, parse_animal_sequence_csv,
};
use super::task_overrides::{TaskParamsOverrides, TaskVariant, expand_experiment_variants};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressProfileConfig {
    pub profile: IngressProfileId,

    /// `videomap`: scan-style video roots with optional annotation roots.
    #[serde(default)]
    pub tasks: Vec<VideoMapTaskConfig>,

    /// `sequencetable`: animal-structured DLC root.
    #[serde(default)]
    pub data_dir: Option<PathBuf>,
    #[serde(default)]
    pub task_type: Option<TaskType>,
    #[serde(default)]
    pub trial_sequence: Vec<DetectionRecipe>,
    #[serde(default)]
    pub animal_sequence_csv: Option<PathBuf>,
    #[serde(default)]
    pub animal_metadata_csv: Option<PathBuf>,
    #[serde(default)]
    pub center_zone_meters: Option<(f64, f64)>,
    #[serde(default)]
    pub arena_meters: Option<(f64, f64)>,
    #[serde(default)]
    pub task_params: Option<TaskParamsOverrides>,
    #[serde(default)]
    pub trial_start_perimeter: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IngressProfileId {
    #[serde(rename = "videomap")]
    VideoMap,
    #[serde(rename = "sequencetable")]
    SequenceTable,
}

impl IngressProfileId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VideoMap => "videomap",
            Self::SequenceTable => "sequencetable",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMapTaskConfig {
    pub video_dir: PathBuf,
    #[serde(default)]
    pub pose_data_dir: Option<PathBuf>,
    pub task_type: TaskType,
    pub meters_per_pixel: ScaleConfig,
    #[serde(default)]
    pub fps: FpsConfig,
    pub recipe: Option<DetectionRecipe>,
    #[serde(default)]
    pub recipe_rules: HashMap<String, DetectionRecipe>,
    #[serde(default = "default_backends")]
    pub backend: BackendPriority,
    #[serde(default)]
    pub arena_calibration: Option<bekiper_detect::config::ArenaCalibrationConfig>,
    #[serde(default = "default_extensions")]
    pub extensions: Vec<String>,
    pub annotations: Option<AnnotationConfig>,
    #[serde(default)]
    pub task_params: Option<TaskParamsOverrides>,
    #[serde(default)]
    pub variants: Vec<TaskVariant>,
    #[serde(default)]
    pub trial_start_perimeter: Option<String>,
}

pub struct ResolvedExperiment {
    pub experiment: ExperimentConfig,
}

pub trait IngressProfile: Send + Sync {
    fn id(&self) -> IngressProfileId;

    fn validate_options(&self, config: &IngressProfileConfig) -> Result<()>;

    fn discover(
        &self,
        config: &IngressProfileConfig,
        project: &IngressConfig,
        base_dir: &Path,
        materialize: bool,
    ) -> Result<Vec<ResolvedExperiment>>;
}

pub struct IngressProfileRegistry;

impl IngressProfileRegistry {
    pub fn get(id: IngressProfileId) -> &'static dyn IngressProfile {
        match id {
            IngressProfileId::VideoMap => &VideoMapProfile,
            IngressProfileId::SequenceTable => &SequenceTableProfile,
        }
    }
}

struct VideoMapProfile;
struct SequenceTableProfile;

impl IngressProfile for VideoMapProfile {
    fn id(&self) -> IngressProfileId {
        IngressProfileId::VideoMap
    }

    fn validate_options(&self, config: &IngressProfileConfig) -> Result<()> {
        if config.tasks.is_empty() {
            return Err(BekiperError::Config(
                "videomap profile requires at least one [[ingress_profile.tasks]] entry".into(),
            ));
        }
        Ok(())
    }

    fn discover(
        &self,
        config: &IngressProfileConfig,
        project: &IngressConfig,
        base_dir: &Path,
        _materialize: bool,
    ) -> Result<Vec<ResolvedExperiment>> {
        self.validate_options(config)?;
        let mut expanded = Vec::new();

        for task in &config.tasks {
            let scan = ScanConfig {
                video_dir: task.video_dir.clone(),
                task_type: task.task_type,
                fps: task.fps,
                meters_per_pixel: task.meters_per_pixel.clone(),
                recipe: task.recipe,
                recipe_rules: task.recipe_rules.clone(),
                backend: task.backend.clone(),
                arena_calibration: task.arena_calibration.clone(),
                extensions: task.extensions.clone(),
                annotations: task.annotations.clone(),
                task_params: task.task_params.clone(),
                variants: task.variants.clone(),
                trial_start_perimeter: task.trial_start_perimeter.clone(),
            };

            let dir = base_dir.join(&scan.video_dir);
            if !dir.exists() {
                return Err(BekiperError::Config(format!(
                    "videomap video_dir does not exist: {}",
                    dir.display()
                )));
            }

            let annotation_dir = scan.annotations.as_ref().map(|a| base_dir.join(&a.dir));
            let metadata_contexts = annotation_dir
                .as_ref()
                .map(|dir| load_imen_contexts(dir))
                .transpose()?
                .unwrap_or_default();

            let mut videos = Vec::new();
            collect_videos(&dir, &scan.extensions, &mut videos)?;
            videos.sort();

            let mut metadata_collisions = Vec::new();
            for video_path in &videos {
                let rel_to_base = video_path
                    .strip_prefix(base_dir)
                    .unwrap_or(video_path)
                    .to_path_buf();
                let rel_to_scan_dir = video_path.strip_prefix(&dir).unwrap_or(video_path);
                let pose_data_file = task
                    .pose_data_dir
                    .as_ref()
                    .map(|pose_data_dir| {
                        resolve_mirrored_pose_data_file(base_dir, pose_data_dir, &rel_to_base)
                    })
                    .transpose()?;
                let rule_recipe = resolve_recipe(&scan, rel_to_scan_dir);
                let mut context = if scan.task_type == TaskType::OpenField {
                    metadata_contexts.context_for_oft_video(video_path)
                } else {
                    metadata_contexts.context_for_video(video_path)
                };
                if let Some(rule_recipe) = rule_recipe
                    && is_recipe_scoped_imen_nort(rule_recipe)
                    && let Some(rule_context) =
                        metadata_contexts.context_for_video_recipe(video_path, rule_recipe)
                    && annotation_dir.as_ref().is_none_or(|dir| {
                        manual_annotation_resolves_with_context(
                            dir,
                            video_path,
                            rule_context.clone(),
                        )
                    })
                {
                    context = Some(rule_context);
                }
                let stage_zero_lookup_collision = is_imen_habituation_context(context.as_ref())
                    || metadata_contexts.has_stage_zero_lookup_candidate(video_path);
                let context_recipe = metadata_recipe_for_context(context.as_ref());
                if let (Some(rule_recipe), Some(Some(context_recipe))) =
                    (rule_recipe, context_recipe)
                    && is_recipe_scoped_imen_nort(rule_recipe)
                    && context_recipe != rule_recipe
                {
                    metadata_collisions.push(format!(
                        "Imen NORT metadata collision for {}: path recipe {} requires stage {}, but selected metadata row {} has stage {}; candidate rows: {}",
                        video_path.display(),
                        rule_recipe.as_str(),
                        expected_imen_stage_for_recipe(rule_recipe).unwrap_or("unknown"),
                        context
                            .as_ref()
                            .and_then(|context| context.source_row)
                            .map(|row| row.to_string())
                            .unwrap_or_else(|| "unknown".into()),
                        context
                            .as_ref()
                            .and_then(|context| context.stage.as_deref())
                            .unwrap_or("unknown"),
                        metadata_contexts
                            .candidate_summary_for_video(video_path)
                            .unwrap_or_else(|| "none".into())
                    ));
                    continue;
                }
                let recipe = if scan.task_type == TaskType::OpenField {
                    // Stage 0 supplies trial identity; OFT owns its calibration recipe.
                    rule_recipe
                } else {
                    match metadata_recipe_for_context(context.as_ref()) {
                        Some(Some(recipe)) => Some(recipe),
                        // Imen's NORT renamed videos use animal-like numbers
                        // (`post.T1...video-10`) that can collide with stage-0
                        // `Test 10` metadata rows. If the path has an explicit
                        // task recipe rule, keep that rule so manual annotation
                        // materialization can use the recorded session/reference.
                        // Drop the colliding context here; otherwise the manual
                        // resolver is forced to look for a stage-0 session.
                        Some(None) | None if stage_zero_lookup_collision => {
                            rule_recipe.and_then(|recipe| {
                                metadata_contexts
                                    .context_for_video_recipe(video_path, recipe)
                                    .filter(|rule_context| {
                                        annotation_dir.as_ref().is_some_and(|dir| {
                                            manual_annotation_resolves_with_context(
                                                dir,
                                                video_path,
                                                rule_context.clone(),
                                            )
                                        })
                                    })
                                    .map(|rule_context| {
                                        context = Some(rule_context);
                                        recipe
                                    })
                            })
                        }
                        Some(None) => None, // coverage: unreachable — a stage-0 context is always a lookup collision
                        None => rule_recipe,
                    }
                };

                let base_experiment = ExperimentConfig {
                    recording_id: None,
                    name: derive_experiment_name(&rel_to_base),
                    task_type: scan.task_type,
                    fps: scan.fps,
                    meters_per_pixel: scan.meters_per_pixel.clone(),
                    data_files: pose_data_file.into_iter().collect(),
                    video: Some(rel_to_base),
                    pose_model: if task.pose_data_dir.is_some() {
                        None
                    } else {
                        project.pose_model.clone()
                    },
                    detection: recipe.map(|r| DetectionConfig {
                        recipe: Some(r),
                        backend: scan.backend.clone(),
                        perimeters: None,
                        pipeline_overrides: None,
                        arena_calibration: scan.arena_calibration.clone(),
                    }),
                    task_params: scan.task_params.clone(),
                    trial_start_perimeter: scan.trial_start_perimeter.clone(),
                    annotation_dir: annotation_dir.clone(),
                    frame_timing: None,
                    video_resolution: None,
                    frame_image: None,
                    timestamps_file: None,
                    animal_id: None,
                    session_context: context,
                };

                let variant_experiments =
                    expand_experiment_variants(base_experiment, &scan.variants);
                expanded.extend(
                    variant_experiments
                        .into_iter()
                        .map(|experiment| ResolvedExperiment { experiment }),
                );
            }

            if !metadata_collisions.is_empty() {
                return Err(BekiperError::Config(format!(
                    "Imen NORT metadata preflight failed with {} recipe/stage collision(s); fix the canonical annotation sessions or metadata.csv before regenerating checkpoints:\n{}",
                    metadata_collisions.len(),
                    metadata_collisions.join("\n")
                )));
            }
        }

        Ok(expanded)
    }
}

impl IngressProfile for SequenceTableProfile {
    fn id(&self) -> IngressProfileId {
        IngressProfileId::SequenceTable
    }

    fn validate_options(&self, config: &IngressProfileConfig) -> Result<()> {
        if config.data_dir.is_none() {
            return Err(BekiperError::Config(
                "sequencetable profile requires data_dir".into(),
            ));
        }
        if config.task_type.is_none() {
            return Err(BekiperError::Config(
                "sequencetable profile requires task_type".into(),
            ));
        }
        if config.trial_sequence.is_empty() {
            return Err(BekiperError::Config(
                "sequencetable profile requires trial_sequence".into(),
            ));
        }
        if config.animal_sequence_csv.is_none() {
            return Err(BekiperError::Config(
                "sequencetable profile requires animal_sequence_csv".into(),
            ));
        }
        Ok(())
    }

    fn discover(
        &self,
        config: &IngressProfileConfig,
        project: &IngressConfig,
        base_dir: &Path,
        materialize: bool,
    ) -> Result<Vec<ResolvedExperiment>> {
        self.validate_options(config)?;
        let scan = AnimalScanConfig {
            data_dir: config.data_dir.clone().unwrap(),
            task_type: config.task_type.unwrap(),
            trial_sequence: config.trial_sequence.clone(),
            animal_sequence_csv: config.animal_sequence_csv.clone().unwrap(),
            animal_metadata_csv: config.animal_metadata_csv.clone(),
            task_params: config.task_params.clone(),
            center_zone_meters: config
                .center_zone_meters
                .unwrap_or_else(default_center_zone_meters),
            arena_meters: config.arena_meters,
            trial_start_perimeter: config.trial_start_perimeter.clone(),
        };

        let mut project_metadata = project.animal_metadata.clone();
        let data_dir = base_dir.join(&scan.data_dir);
        if !data_dir.exists() {
            return Err(BekiperError::Config(format!(
                "sequencetable data_dir does not exist: {}",
                data_dir.display()
            )));
        }

        let seq_rows = parse_animal_sequence_csv(&data_dir.join(&scan.animal_sequence_csv))?;
        if let Some(ref meta_csv) = scan.animal_metadata_csv {
            let meta_path = data_dir.join(meta_csv);
            if meta_path.exists() {
                project_metadata = parse_animal_metadata_csv(&meta_path)?;
            }
        }

        let mut expanded = Vec::new();
        for row in &seq_rows {
            let Some(resolved) = resolve_sequence_experiment(&data_dir, &scan, row, materialize)?
            else {
                continue;
            };
            let mut experiment = resolved.experiment;
            experiment.data_files = experiment
                .data_files
                .into_iter()
                .map(|path| relative_to_base(path, base_dir))
                .collect();
            experiment.video = experiment
                .video
                .map(|path| relative_to_base(path, base_dir));
            experiment.frame_image = experiment
                .frame_image
                .map(|path| relative_to_base(path, base_dir));
            experiment.timestamps_file = experiment
                .timestamps_file
                .map(|path| relative_to_base(path, base_dir));
            if let Some(detection) = &mut experiment.detection {
                detection.perimeters = detection
                    .perimeters
                    .take()
                    .map(|path| relative_to_base(path, base_dir));
            } // coverage: unreachable — sequence experiments always carry a pre_computed detection
            let animal_meta = project_metadata.get(&row.animal);
            experiment.session_context = Some(IngressSessionContext {
                profile: IngressProfileId::SequenceTable.as_str().into(),
                source_file: Some(scan.animal_sequence_csv.display().to_string()),
                animal: Some(row.animal.to_string()),
                sequence: Some(row.sequence.to_string()),
                stage: Some(resolved.recipe.as_str().to_string()),
                sex: animal_meta.map(|m| m.sex.clone()),
                age: animal_meta.map(|m| m.age.clone()),
                genotype: animal_meta.map(|m| m.gene.clone()),
                group: animal_meta.map(|m| m.cohort.clone()),
                ..Default::default()
            });
            expanded.push(ResolvedExperiment { experiment });
        }

        Ok(expanded)
    }
}

fn manual_annotation_resolves_with_context(
    annotation_dir: &Path,
    video_path: &Path,
    context: IngressSessionContext,
) -> bool {
    AnnotationSession::load_with_context(annotation_dir, video_path, Some(context)).is_ok()
}

fn is_recipe_scoped_imen_nort(recipe: DetectionRecipe) -> bool {
    matches!(
        recipe,
        DetectionRecipe::NortTraining | DetectionRecipe::NortNovelty
    )
}

fn expected_imen_stage_for_recipe(recipe: DetectionRecipe) -> Option<&'static str> {
    match recipe {
        DetectionRecipe::NortTraining => Some("1"),
        DetectionRecipe::NortNovelty => Some("2"),
        _ => None, // coverage: unreachable — only NORT recipes reach the collision message
    }
}
