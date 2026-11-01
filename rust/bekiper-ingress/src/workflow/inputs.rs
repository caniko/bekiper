use std::path::{Path, PathBuf};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::{PerimeterSpec, PerimetersFile};
use bekiper_core::types::DetectionBackend;

use crate::annotation::{AnnotationSession, SessionMapping};
use crate::config::{DetectionConfig, ExperimentConfig};
use crate::stage::ExperimentOutputDir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InputArtifactState {
    Materialized,
    Missing,
}

#[derive(Debug, Clone)]
pub(super) struct AnnotationInput {
    pub(super) specs: Vec<PerimeterSpec>,
    pub(super) labels: Vec<String>,
    pub(super) detail: String,
    pub(super) scale_plugin_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub(super) enum PerimeterInputSource {
    Checkpoint { path: PathBuf },
    PreComputed { path: PathBuf, count: usize },
    ManualAnnotation(AnnotationInput),
    ArenaCalibrationCv,
    Automatic { backend: DetectionBackend },
    Missing { reason: String },
}

#[derive(Debug, Clone)]
pub(super) struct PerimeterInputStatus {
    pub(super) artifact_state: InputArtifactState,
    pub(super) artifact_path: PathBuf,
    pub(super) sources: Vec<PerimeterInputSource>,
}

impl PerimeterInputStatus {
    pub(super) fn has_non_auto_source(&self) -> bool {
        self.sources.iter().any(|source| {
            matches!(
                source,
                PerimeterInputSource::Checkpoint { .. }
                    | PerimeterInputSource::PreComputed { .. }
                    | PerimeterInputSource::ManualAnnotation(_)
                    | PerimeterInputSource::ArenaCalibrationCv
            )
        })
    }

    pub(super) fn first_manual_annotation(&self) -> Option<&AnnotationInput> {
        self.sources.iter().find_map(|source| match source {
            PerimeterInputSource::ManualAnnotation(annotation) => Some(annotation),
            _ => None,
        })
    }

    pub(super) fn source_summary(&self) -> String {
        if self.sources.is_empty() {
            return "no perimeter source configured".into();
        }
        self.sources
            .iter()
            .map(|source| match source {
                PerimeterInputSource::Checkpoint { path } => {
                    format!("checkpoint {}", path.display())
                }
                PerimeterInputSource::PreComputed { path, count } => {
                    format!("precomputed {} ({} perimeter(s))", path.display(), count)
                }
                PerimeterInputSource::ManualAnnotation(annotation) => {
                    format!("manual annotations resolve via {}", annotation.detail)
                }
                PerimeterInputSource::ArenaCalibrationCv => {
                    "arena calibration cv source available".into()
                }
                PerimeterInputSource::Automatic { backend } => {
                    format!("automatic backend {}", backend.as_str())
                }
                PerimeterInputSource::Missing { reason } => reason.clone(),
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

#[derive(Debug, Clone)]
pub(super) struct PoseInputStatus {
    pub(super) sources: Vec<String>,
    pub(super) missing: Vec<String>,
}

impl PoseInputStatus {
    pub(super) fn is_available(&self) -> bool {
        self.missing.is_empty() && !self.sources.is_empty()
    }
}

pub(super) struct PipelineInputResolver<'a> {
    output_dir: &'a Path,
}

impl<'a> PipelineInputResolver<'a> {
    pub(super) fn new(output_dir: &'a Path) -> Self {
        Self { output_dir }
    }

    pub(super) fn pose_source(&self, experiment: &ExperimentConfig) -> PoseInputStatus {
        let mut sources = Vec::new();
        let mut missing = Vec::new();

        if !experiment.data_files.is_empty() {
            for data_file in &experiment.data_files {
                if data_file.exists() {
                    sources.push(format!("precomputed pose data {}", data_file.display()));
                } else {
                    missing.push(format!("missing pose data file {}", data_file.display()));
                }
            }
            return PoseInputStatus { sources, missing };
        }

        if experiment.needs_pose_inference() {
            match experiment.video.as_deref() {
                Some(video) if video.exists() => {
                    sources.push(format!("pose inference video {}", video.display()))
                }
                Some(video) => {
                    missing.push(format!("missing pose inference video {}", video.display()))
                }
                None => missing.push("missing pose inference video".into()), // coverage: unreachable — needs_pose_inference requires a video and a model
            }
            match experiment.pose_model.as_ref() {
                Some(model) if model.path.exists() => {
                    sources.push(format!("pose model {}", model.path.display()))
                }
                Some(model) => missing.push(format!("missing pose model {}", model.path.display())),
                None => missing.push("missing pose model".into()), // coverage: unreachable — needs_pose_inference requires a video and a model
            }
        } else {
            missing
                .push("no precomputed pose data and no video+pose_model source configured".into());
        }

        PoseInputStatus { sources, missing }
    }

    pub(super) fn perimeter_source(
        &self,
        experiment: &ExperimentConfig,
        exp_dir: &ExperimentOutputDir,
    ) -> PerimeterInputStatus {
        let artifact_path = exp_dir.pixel_perimeters_toml();
        let artifact_state = if artifact_path.exists() {
            InputArtifactState::Materialized
        } else {
            InputArtifactState::Missing
        };
        let mut sources = Vec::new();

        if artifact_state == InputArtifactState::Materialized {
            sources.push(PerimeterInputSource::Checkpoint {
                path: artifact_path.clone(),
            });
        }

        let Some(detection) = experiment.detection.as_ref() else {
            return PerimeterInputStatus {
                artifact_state,
                artifact_path,
                sources,
            };
        };

        for backend in detection.effective_backends().backends() {
            match backend {
                DetectionBackend::PreComputed => {
                    sources.push(resolve_precomputed_source(detection));
                }
                DetectionBackend::ManualAnnotation => {
                    sources.push(resolve_manual_annotation_source(experiment));
                }
                DetectionBackend::ArenaCalibrationCv => {
                    sources.push(resolve_arena_calibration_source(experiment));
                }
                DetectionBackend::ClassicalCv
                | DetectionBackend::Sam2Auto
                | DetectionBackend::GroundingDinoSam2 => {
                    sources.push(PerimeterInputSource::Automatic { backend: *backend });
                }
            }
        }

        PerimeterInputStatus {
            artifact_state,
            artifact_path,
            sources,
        }
    }

    pub(super) fn exp_dir(&self, experiment: &ExperimentConfig) -> ExperimentOutputDir {
        match experiment.animal_id {
            Some(id) => ExperimentOutputDir::for_animal(self.output_dir, id, &experiment.name),
            None => ExperimentOutputDir::new(self.output_dir, &experiment.name),
        }
    }
}

pub(super) fn materialize_manual_annotation(
    experiment: &ExperimentConfig,
    annotation: &AnnotationInput,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    use bekiper_core::scale::ScaleDerivationInfo;

    exp_dir.ensure_dir()?;
    let dest = exp_dir.pixel_perimeters_toml();
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    } // coverage: unreachable — checkpoint paths always have a parent directory
    let annotation_scale = crate::annotation::load_scale_calibrations(&annotation.scale_plugin_dir)
        .ok()
        .and_then(|calibrations| calibrations.into_iter().next())
        .map(|cal| ScaleDerivationInfo {
            meters_per_pixel_y: None,
            method: format!("annotation_{}", cal.method),
            physical_meters: Some(cal.physical_meters),
            pixel_measurement: Some(
                ((cal.p2.0 - cal.p1.0).powi(2) + (cal.p2.1 - cal.p1.1).powi(2)).sqrt(),
            ),
            arm_pixel_spans: Vec::new(),
            reference_label: None,
            endpoints: Some([[cal.p1.0, cal.p1.1], [cal.p2.0, cal.p2.1]]),
            meters_per_pixel: cal.meters_per_pixel.0,
        });
    let mut perimeters = annotation.specs.clone();
    let current_scale = annotation_scale.clone().or_else(|| {
        experiment
            .meters_per_pixel
            .resolve_with_info(&perimeters)
            .ok()
            .map(|(_, info)| info)
    });
    // A generated arena may define this video's own scale (detected box).
    let generated_scale = crate::analysis::append_generated_arena(
        experiment,
        &mut perimeters,
        current_scale.as_ref(),
    )?;
    let pf = PerimetersFile { perimeters };
    let toml_str = toml::to_string_pretty(&pf)
        .map_err(|e| BekiperError::Other(format!("failed to serialize perimeters: {e}")))?;
    std::fs::write(&dest, toml_str)?;
    if let Some(scale_info) = generated_scale.or(annotation_scale) {
        let scale_str = toml::to_string_pretty(&scale_info)
            .map_err(|e| BekiperError::Other(format!("failed to serialize scale: {e}")))?;
        std::fs::write(exp_dir.annotation_scale_toml(), scale_str)?;
    }
    Ok(())
}

fn resolve_precomputed_source(detection: &DetectionConfig) -> PerimeterInputSource {
    let Some(path) = detection.perimeters.as_deref() else {
        return PerimeterInputSource::Missing {
            reason: "pre_computed backend requires a perimeters path".into(),
        };
    };
    if !path.exists() {
        return PerimeterInputSource::Missing {
            reason: format!("missing precomputed perimeters {}", path.display()),
        };
    }
    match std::fs::read_to_string(path)
        .map_err(BekiperError::from)
        .and_then(|content| {
            toml::from_str::<PerimetersFile>(&content)
                .map_err(|error| BekiperError::Data(format!("invalid TOML: {error}")))
        }) {
        Ok(file) => PerimeterInputSource::PreComputed {
            path: path.to_path_buf(),
            count: file.perimeters.len(),
        },
        Err(error) => PerimeterInputSource::Missing {
            reason: format!("failed to parse {}: {error}", path.display()),
        },
    }
}

fn resolve_arena_calibration_source(experiment: &ExperimentConfig) -> PerimeterInputSource {
    if experiment.task_type != bekiper_core::types::TaskType::OpenField {
        return PerimeterInputSource::Missing {
            reason: "arena_calibration_cv is only valid for open_field tasks".into(),
        };
    }
    let Some(video) = experiment.video.as_deref() else {
        return PerimeterInputSource::Missing {
            reason: "arena_calibration_cv requires a video path".into(),
        };
    };
    if !video.exists() {
        return PerimeterInputSource::Missing {
            reason: format!("missing arena calibration video {}", video.display()),
        };
    }
    match &experiment.meters_per_pixel {
        bekiper_core::scale::ScaleConfig::Derived(
            bekiper_core::scale::ScaleMethod::ArenaSide { meters, .. },
        ) if *meters > 0.0 => PerimeterInputSource::ArenaCalibrationCv,
        _ => PerimeterInputSource::Missing {
            reason: "arena_calibration_cv requires meters_per_pixel = { method = \"arena_side\", meters = ... }".into(),
        },
    }
}

fn resolve_manual_annotation_source(experiment: &ExperimentConfig) -> PerimeterInputSource {
    let Some(annotation_dir) = experiment.annotation_dir.as_deref() else {
        return PerimeterInputSource::Missing {
            reason: "manual_annotation backend requires annotations.dir".into(),
        };
    };
    if !annotation_dir.exists() {
        return PerimeterInputSource::Missing {
            reason: format!("missing annotation dir {}", annotation_dir.display()),
        };
    }
    let Some(video_path) = experiment.video.as_deref() else {
        return PerimeterInputSource::Missing {
            reason: "manual_annotation requires a video path".into(),
        };
    };
    match AnnotationSession::load_with_context(
        annotation_dir,
        video_path,
        experiment.session_context.clone(),
    ) {
        Ok(session) => {
            let detail = annotation_detail(&session);
            let labels = session.resolved_object_labels();
            PerimeterInputSource::ManualAnnotation(AnnotationInput {
                specs: session.resolved_specs.clone(),
                labels,
                detail,
                scale_plugin_dir: annotation_dir.join("plugin_files"),
            })
        }
        Err(error) => PerimeterInputSource::Missing {
            reason: format!("manual annotation resolution failed: {error}"),
        },
    }
}

fn annotation_detail(session: &AnnotationSession) -> String {
    match &session.mapping {
        SessionMapping::Metadata { row_index, .. } => {
            let source = session.perimeter_selection_source().label();
            let row = row_index + 2;
            let target = session
                .target_image_stem()
                .map(|target| format!(", target {target}"))
                .unwrap_or_default();
            format!("metadata.csv row {row} ({source}{target})")
        }
        SessionMapping::VideoMap { row_index, .. } => {
            format!("video_map.csv row {}", row_index + 2)
        }
        SessionMapping::Fallback { perimeter_id } => {
            format!("fallback perimeter {perimeter_id}")
        }
    }
}

#[cfg(test)]
#[path = "inputs/tests.rs"]
mod tests;
