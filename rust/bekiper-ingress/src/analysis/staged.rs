//! Receipt-bound staged scoring. No source decoding, augmentation or pose model
//! loading occurs here; raw-pose diagnostics/sheets remain absent.
use crate::{
    analysis::ExperimentAnalysis,
    config::{AnimalMetadata, ExperimentConfig, PipelineConfig},
    preparation::{PREREQUISITES, ReuseTrial},
};
use bekiper_core::{inspection::PerimetersFile, scale::ScaleDerivationInfo, video::FrameTiming};
use polars::prelude::{DataFrame, ParquetReader, SerReader};

pub fn analyze_trial(
    experiment: &ExperimentConfig,
    pipeline: &PipelineConfig,
    animal: Option<&AnimalMetadata>,
    prepared: &ReuseTrial,
) -> anyhow::Result<ExperimentAnalysis> {
    anyhow::ensure!(
        prepared.analysis_id == experiment.name,
        "staged analysis identity differs"
    );
    anyhow::ensure!(
        prepared.prerequisites.len() == PREREQUISITES.len(),
        "staged prerequisite set differs"
    );
    for key in PREREQUISITES {
        prepared
            .prerequisites
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("missing staged prerequisite: {key}"))?
            .verify()?;
    }
    let read = |key: &str| std::fs::read_to_string(&prepared.prerequisites[key].path);
    let pixel: PerimetersFile = bekiper_core::handoff::from_str(&read("pixel_perimeters")?)?;
    let metric: PerimetersFile = bekiper_core::handoff::from_str(&read("perimeters")?)?;
    let scale: ScaleDerivationInfo = bekiper_core::handoff::from_str(&read("scale")?)?;
    let timing: FrameTiming = bekiper_core::handoff::from_str(&read("frame_timing")?)?;
    let mut experiment = experiment.clone();
    experiment.frame_timing = Some(timing);
    let augmented = ParquetReader::new(std::fs::File::open(
        &prepared.prerequisites["augmented"].path,
    )?) // coverage: unreachable — the file was opened and hashed by verify() above
    .finish()?;
    let analysis = crate::analysis::analyze_augmented_pose_with_pipeline(
        &experiment,
        pipeline,
        animal,
        &pixel.perimeters,
        &scale,
        DataFrame::empty(),
        augmented,
    )?;
    let projected =
        crate::analysis::metric_perimeters(&experiment, pipeline, &pixel.perimeters, &scale)?;
    anyhow::ensure!(
        serde_json::to_value(&projected)? == serde_json::to_value(&metric.perimeters)?,
        "staged metric perimeters differ from prepared pixel/scale projection"
    );
    for file in prepared.prerequisites.values() {
        file.verify()?;
    }
    Ok(analysis)
}
