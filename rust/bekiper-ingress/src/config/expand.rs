use std::path::Path;

use bekiper_core::error::{BekiperError, Result};

use crate::config::aggregate::MissingMetadataPolicy;

use super::IngressConfig;
use super::detection::DetectionConfig;
use super::experiment::ExperimentConfig;
use super::io_helpers::*;
use super::profile::{IngressProfileId, IngressProfileRegistry};
use super::scan::parse_animal_metadata_csv;
use super::task_overrides::expand_experiment_variants;

use crate::stage::Stage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExpansionMode {
    Materialize,
    ValidateOnly,
}

impl IngressConfig {
    /// Expand `[[scan]]` and `[[animal_scan]]` entries into concrete
    /// `ExperimentConfig`s.
    ///
    /// Performs filesystem I/O: walks video directories, reads calibration
    /// CSVs, and may write intermediate `{animal}.perimeters-seq{N}.toml`
    /// files under animal_scan `data_dir`s. Call this only when a pipeline
    /// stage is about to run -- **not** from introspection paths like
    /// `--help` or config validation.
    ///
    /// `base_dir` is the directory that relative paths in the config
    /// resolve against (usually `config_path.parent()`).
    ///
    /// `stages` is the list of stages the caller intends to run. Today
    /// every stage iterates the full `experiments` list, so expansion is
    /// unconditional; the first entry is used purely to attach a stage
    /// name to animal-scan expansion errors for clearer diagnostics.
    pub fn expand(&mut self, base_dir: &Path, stages: &[Stage]) -> Result<()> {
        self.expand_internal(base_dir, stages, ExpansionMode::Materialize)
    }

    /// Read-only expansion used by config audit paths.
    ///
    /// Resolves `scan` / `animal_scan` exactly like [`expand`](Self::expand)
    /// but skips writing intermediate perimeter TOMLs under animal data dirs.
    pub fn expand_for_validation(&mut self, base_dir: &Path, stages: &[Stage]) -> Result<()> {
        self.expand_internal(base_dir, stages, ExpansionMode::ValidateOnly)
    }

    fn expand_internal(
        &mut self,
        base_dir: &Path,
        stages: &[Stage],
        mode: ExpansionMode,
    ) -> Result<()> {
        self.expand_inputs(base_dir, stages, mode)?;
        self.prefilter_experiments_by_metadata(base_dir)?;
        super::profile::validate_imen_oft_metadata_bindings(&self.experiments)?;
        Ok(())
    }

    /// Read-only pre-filter expansion retaining excluded and fail-policy rows.
    /// A blocked row is evidence of a failed policy, never permission to run it.
    pub fn expand_for_population(
        &mut self,
        base_dir: &Path,
    ) -> Result<Vec<super::PopulationDecision>> {
        self.expand_inputs(base_dir, Stage::all(), ExpansionMode::ValidateOnly)?;
        self.metadata_population(base_dir)
    }

    fn expand_inputs(
        &mut self,
        base_dir: &Path,
        stages: &[Stage],
        mode: ExpansionMode,
    ) -> Result<()> {
        self.validate()?;
        self.expand_scans(base_dir)?;

        self.expand_ingress_profiles(base_dir, mode)?;

        if !self.animal_scan.is_empty() {
            self.expand_animal_scans(base_dir, mode).map_err(|e| {
                let hint = stages
                    .first()
                    .map(|s| format!("stage `{}`", s.name()))
                    .unwrap_or_else(|| "the selected stages".to_string());
                BekiperError::Config(format!("{hint} requires [[animal_scan]] expansion: {e}"))
            })?;
        }
        self.apply_pose_label_task_defaults();
        self.resolve_whisker_lengths(base_dir)?;
        Ok(())
    }

    /// Pre-flight filter: when `[aggregate.metadata]` is configured, drop or
    /// fail-fast on experiments whose parsed animal ID is absent from the
    /// metadata workbook. Mirrors the policy applied at aggregation time so
    /// excluded animals never reach the analysis stages in the first place.
    ///
    /// Policy mapping:
    /// - [`MissingMetadataPolicy::Fail`] — abort with the full list of missing IDs.
    /// - [`MissingMetadataPolicy::ExcludeMissing`] — drop matching experiments,
    ///   log a single summary line.
    /// - [`MissingMetadataPolicy::AllowBlanks`] — no-op.
    fn prefilter_experiments_by_metadata(&mut self, base_dir: &Path) -> Result<()> {
        let Some(metadata_cfg) = self.aggregate.metadata.as_ref() else {
            return Ok(());
        };
        if matches!(
            metadata_cfg.missing_policy,
            MissingMetadataPolicy::AllowBlanks
        ) {
            return Ok(());
        }

        let decisions = self.metadata_population(base_dir)?;
        let missing_pairs: Vec<_> = decisions
            .iter()
            .filter(|decision| decision.disposition != super::PopulationDisposition::Admitted)
            .map(|decision| (decision.analysis_id.clone(), decision.animal_id))
            .collect();
        let total_before = self.experiments.len();

        if missing_pairs.is_empty() {
            return Ok(());
        }

        match metadata_cfg.missing_policy {
            MissingMetadataPolicy::Fail => {
                let mut ids: Vec<String> = missing_pairs
                    .iter()
                    .map(|(name, id)| match id {
                        Some(id) => format!("{id} ({name})"),
                        None => format!("<unparseable> ({name})"),
                    })
                    .collect();
                ids.sort();
                Err(BekiperError::Data(format!(
                    "{}: {} experiment(s) lack matching animal metadata: {}",
                    base_dir.join(&metadata_cfg.path).display(),
                    missing_pairs.len(),
                    ids.join(", ")
                )))
            }
            MissingMetadataPolicy::ExcludeMissing => {
                let mut decisions = decisions.iter();
                self.experiments.retain(|_| {
                    decisions.next().is_some_and(|decision| {
                        decision.disposition == super::PopulationDisposition::Admitted
                    })
                });
                let mut ids: Vec<u32> = missing_pairs.iter().filter_map(|(_, id)| *id).collect();
                ids.sort_unstable();
                ids.dedup();
                let unparseable = missing_pairs.iter().filter(|(_, id)| id.is_none()).count();
                tracing::info!(
                    metadata_workbook = %base_dir.join(&metadata_cfg.path).display(),
                    excluded_experiments = missing_pairs.len(),
                    kept_experiments = total_before - missing_pairs.len(),
                    excluded_animal_ids = ?ids,
                    unparseable_experiments = unparseable,
                    "pre-flight: dropped experiments missing from animal metadata"
                );
                Ok(())
            }
            MissingMetadataPolicy::AllowBlanks => unreachable!(), // coverage: unreachable — AllowBlanks returns before the population scan
        }
    }

    fn apply_pose_label_task_defaults(&mut self) {
        let model = self.pipeline.pose_label_model;
        for experiment in &mut self.experiments {
            match experiment.task_params.as_mut() {
                Some(overrides) => overrides.fill_pose_label_defaults(model),
                None => {
                    let defaults =
                        super::task_overrides::TaskParamsOverrides::with_pose_label_defaults(model);
                    if !matches!(
                        model,
                        super::pipeline::PoseLabelModel::SuperanimalTopviewmouse
                    ) {
                        experiment.task_params = Some(defaults);
                    }
                }
            }
        }
    }

    fn expand_ingress_profiles(&mut self, base_dir: &Path, mode: ExpansionMode) -> Result<()> {
        if self.ingress_profile.is_empty() {
            return Ok(());
        }

        for profile_config in &self.ingress_profile {
            if profile_config.profile == IngressProfileId::SequenceTable
                && let (Some(data_dir), Some(meta_csv)) = (
                    &profile_config.data_dir,
                    &profile_config.animal_metadata_csv,
                )
            {
                let meta_path = base_dir.join(data_dir).join(meta_csv);
                if meta_path.exists() {
                    self.animal_metadata = parse_animal_metadata_csv(&meta_path)?;
                }
            }
        }

        let mut expanded = Vec::new();
        for profile_config in &self.ingress_profile {
            let profile = IngressProfileRegistry::get(profile_config.profile);
            let resolved = profile.discover(
                profile_config,
                self,
                base_dir,
                matches!(mode, ExpansionMode::Materialize),
            )?;
            expanded.extend(resolved.into_iter().map(|r| r.experiment));
        }

        expanded.append(&mut self.experiments);
        self.experiments = expanded;
        Ok(())
    }

    /// Walk each `[[scan]]` directory, create `ExperimentConfig` entries, and
    /// prepend them to the explicit `experiments` list.
    fn expand_scans(&mut self, base_dir: &Path) -> Result<()> {
        if self.scan.is_empty() {
            return Ok(());
        }

        let mut expanded = Vec::new();

        for scan in &self.scan {
            let dir = base_dir.join(&scan.video_dir);
            if !dir.exists() {
                return Err(BekiperError::Config(format!(
                    "scan video_dir does not exist: {}",
                    dir.display()
                )));
            }

            let mut videos: Vec<std::path::PathBuf> = Vec::new();
            collect_videos(&dir, &scan.extensions, &mut videos)?;
            videos.sort();

            tracing::info!(
                dir = %scan.video_dir.display(),
                count = videos.len(),
                "scan discovered videos"
            );

            for video_path in &videos {
                let rel_to_base = video_path
                    .strip_prefix(base_dir)
                    .unwrap_or(video_path)
                    .to_path_buf();

                let rel_to_scan_dir = video_path.strip_prefix(&dir).unwrap_or(video_path);

                let name = derive_experiment_name(&rel_to_base);
                let recipe = resolve_recipe(scan, rel_to_scan_dir);

                let annotation_dir = scan.annotations.as_ref().map(|a| base_dir.join(&a.dir));

                let base_experiment = ExperimentConfig {
                    recording_id: None,
                    name,
                    task_type: scan.task_type,
                    fps: scan.fps,
                    meters_per_pixel: scan.meters_per_pixel.clone(),
                    data_files: Vec::new(),
                    video: Some(rel_to_base),
                    pose_model: self.pose_model.clone(),
                    detection: recipe.map(|r| DetectionConfig {
                        recipe: Some(r),
                        backend: scan.backend.clone(),
                        perimeters: None,
                        pipeline_overrides: None,
                        arena_calibration: scan.arena_calibration.clone(),
                    }),
                    task_params: scan.task_params.clone(),
                    trial_start_perimeter: scan.trial_start_perimeter.clone(),
                    annotation_dir,
                    frame_timing: None,
                    video_resolution: None,
                    frame_image: None,
                    timestamps_file: None,
                    animal_id: None,
                    session_context: None,
                };
                expanded.extend(expand_experiment_variants(base_experiment, &scan.variants));
            }
        }

        // Scans go first, then explicit experiments (which can override/extend)
        expanded.append(&mut self.experiments);
        self.experiments = expanded;

        // Apply shared pose_model to any experiment that doesn't have one
        if let Some(ref shared) = self.pose_model {
            for exp in &mut self.experiments {
                if exp.pose_model.is_none() && exp.video.is_some() {
                    exp.pose_model = Some(shared.clone());
                }
            }
        }

        Ok(())
    }

    /// Expand `[[animal_scan]]` entries into `ExperimentConfig`s.
    ///
    /// Each row in `animal_sequence_csv` maps (animal, sequence) -> calibration
    /// file IDs. For each row we resolve the DLC data file, perimeters, scale
    /// calibration, center point, and FPS from the animal's directory.
    fn expand_animal_scans(&mut self, base_dir: &Path, mode: ExpansionMode) -> Result<()> {
        if self.animal_scan.is_empty() {
            return Ok(()); // coverage: unreachable — the only caller checks for animal scans
        }

        let mut expanded = Vec::new();

        for scan in &self.animal_scan {
            let data_dir = base_dir.join(&scan.data_dir);
            if !data_dir.exists() {
                return Err(BekiperError::Config(format!(
                    "animal_scan data_dir does not exist: {}",
                    data_dir.display()
                )));
            }

            // Parse the animal sequence CSV
            let seq_csv_path = data_dir.join(&scan.animal_sequence_csv);
            let seq_rows = super::scan::parse_animal_sequence_csv(&seq_csv_path)?;

            // Parse optional animal metadata CSV
            if let Some(ref meta_csv) = scan.animal_metadata_csv {
                let meta_path = data_dir.join(meta_csv);
                if meta_path.exists() {
                    self.animal_metadata = parse_animal_metadata_csv(&meta_path)?;
                    tracing::info!(count = self.animal_metadata.len(), "loaded animal metadata");
                }
            }

            tracing::info!(
                dir = %scan.data_dir.display(),
                rows = seq_rows.len(),
                "expanding animal scan"
            );

            for row in &seq_rows {
                if let Some(experiment) = resolve_sequence_experiment(
                    &data_dir,
                    scan,
                    row,
                    matches!(mode, ExpansionMode::Materialize),
                )? {
                    expanded.push(experiment.experiment);
                }
            }
        }

        // Append animal scan experiments after any existing experiments
        self.experiments.extend(expanded);
        Ok(())
    }
}

#[cfg(test)]
#[path = "expand/tests.rs"]
mod tests;
