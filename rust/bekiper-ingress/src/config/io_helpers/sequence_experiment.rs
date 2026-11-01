//! One resolver for legacy animal scans and the sequence-table profile.
//! Callers own their path presentation and optional session metadata.
use std::path::Path;

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::PerimetersFile;
use bekiper_core::scale::ScaleConfig;
use bekiper_core::types::{DetectionBackend, DetectionRecipe};
use bekiper_core::video::FpsConfig;

use super::*;
use crate::config::detection::{BackendPriority, DetectionConfig};
use crate::config::experiment::ExperimentConfig;
use crate::config::scan::{AnimalScanConfig, AnimalSequenceRow};

pub(in crate::config) struct SequenceExperiment {
    pub(in crate::config) experiment: ExperimentConfig,
    pub(in crate::config) recipe: DetectionRecipe,
}

pub(in crate::config) fn resolve_sequence_experiment(
    data_dir: &Path,
    scan: &AnimalScanConfig,
    row: &AnimalSequenceRow,
    materialize: bool,
) -> Result<Option<SequenceExperiment>> {
    let animal_dir = data_dir.join(row.animal.to_string());
    if !animal_dir.exists() {
        tracing::warn!(animal = row.animal, "animal directory not found, skipping");
        return Ok(None);
    }
    let recipe = scan
        .trial_sequence
        .get(row.sequence as usize)
        .copied()
        .ok_or_else(|| {
            BekiperError::Config(format!(
                "sequence {} exceeds trial_sequence length {} for animal {}",
                row.sequence,
                scan.trial_sequence.len(),
                row.animal
            ))
        })?;
    let data_file = find_dlc_data_file(&animal_dir, row.animal, row.sequence)?;
    let mpp = load_single_scale_calibration(&animal_dir, &row.mpp_id)?;
    let mut perimeters = load_sequence_perimeters(&animal_dir, row.animal, row.sequence, recipe)?;
    let center = load_center_zone(&animal_dir, &row.center_id, scan.center_zone_meters, mpp)?;
    let arena = match scan.arena_meters {
        Some(meters) => load_arena_boundary(&animal_dir, &row.center_id, meters, mpp)?,
        None => None,
    };
    perimeters.extend(arena);
    perimeters.extend(center);
    let video = load_video_metadata(&animal_dir, row.animal, row.sequence)?;
    let video_path = find_experiment_video(&animal_dir, &video.stem);
    let frame_image = find_frame_image(&animal_dir, row.animal, row.sequence);
    let timestamps_file = find_timestamps_file(&animal_dir, row.animal, row.sequence);
    let name = format!("{}_seq{}_{:?}", row.animal, row.sequence, recipe).to_lowercase();
    let path = animal_dir.join(format!(
        "{}.perimeters-seq{}.toml",
        row.animal, row.sequence
    ));
    // Validation still checks serialization, but never materializes a file.
    let content = toml::to_string_pretty(&PerimetersFile { perimeters })
        .map_err(|error| BekiperError::Data(format!("serialize perimeters: {error}")))?;
    if materialize {
        std::fs::write(&path, content)?;
    }
    Ok(Some(SequenceExperiment {
        recipe,
        experiment: ExperimentConfig {
            recording_id: None,
            name,
            task_type: scan.task_type,
            fps: FpsConfig::Manual(video.fps),
            meters_per_pixel: ScaleConfig::Manual(mpp.0),
            data_files: vec![data_file],
            video: video_path,
            pose_model: None,
            detection: Some(DetectionConfig {
                recipe: Some(recipe),
                backend: BackendPriority(vec![DetectionBackend::PreComputed]),
                perimeters: Some(path),
                pipeline_overrides: None,
                arena_calibration: None,
            }),
            task_params: scan.task_params.clone(),
            trial_start_perimeter: scan.trial_start_perimeter.clone(),
            annotation_dir: None,
            frame_timing: None,
            video_resolution: video.resolution,
            frame_image,
            timestamps_file,
            animal_id: Some(row.animal),
            session_context: None,
        },
    }))
}
