//! Whisker-fan plumbing: env overlay on base config layers and per-experiment
//! age-table resolution (the pure arithmetic lives in `bekiper-feature`).
use std::collections::HashMap;
use std::path::Path;

use bekiper_core::error::{BekiperError, Result};
use bekiper_feature::heuristic::solo::whisker_config::{
    AgeTableSpec, MetersSpec, WhiskerEnv, WhiskerLengthSpec, resolve_age_length,
};

use super::{IngressConfig, TaskParamsOverrides};
use crate::aggregate::metadata::{
    AnimalAges, animal_id_from_experiment, parse_animal_id, read_animal_ages,
};
use crate::aggregate::session_timepoint;

impl IngressConfig {
    /// Overlay env/constructor whisker knobs on experiments and scan/profile
    /// templates. Variants are left alone so a `nose_only` or fixed-reach
    /// variant keeps its explicit setting.
    pub fn apply_whisker_env(&mut self, env: WhiskerEnv) {
        if env.is_empty() {
            return;
        }
        let apply = |params: &mut Option<TaskParamsOverrides>| {
            let whisker = params
                .get_or_insert_with(Default::default)
                .exploration
                .get_or_insert_with(Default::default)
                .whisker
                .get_or_insert_with(Default::default);
            env.apply(whisker);
        };
        for experiment in &mut self.experiments {
            apply(&mut experiment.task_params);
        }
        for scan in &mut self.scan {
            apply(&mut scan.task_params);
        }
        for scan in &mut self.animal_scan {
            apply(&mut scan.task_params);
        }
        for profile in &mut self.ingress_profile {
            apply(&mut profile.task_params);
            for task in &mut profile.tasks {
                apply(&mut task.task_params);
            }
        }
    }

    /// Replace each enabled `age_table` length with the experiment's resolved
    /// `{ meters }`, recording `age_days` and the workbook source. A missing
    /// timepoint, or a missing animal or age under a non-constant table, is a
    /// hard error naming the experiment.
    pub(crate) fn resolve_whisker_lengths(&mut self, base_dir: &Path) -> Result<()> {
        let mut workbooks: HashMap<(String, String, String, String, String), AnimalAges> =
            HashMap::new();
        for experiment in &mut self.experiments {
            let session = experiment
                .session_context
                .as_ref()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|e| BekiperError::Config(e.to_string()))?
                .unwrap_or_default();
            let name = experiment.name.clone();
            let Some(whisker) = experiment
                .task_params
                .as_mut()
                .and_then(|p| p.exploration.as_mut())
                .and_then(|e| e.whisker.as_mut())
            else {
                continue;
            };
            let Some(WhiskerLengthSpec::AgeTable(spec)) = whisker.length.clone() else {
                continue;
            };
            if !whisker.is_enabled() {
                continue;
            }
            let ages = workbook_ages(&mut workbooks, base_dir, &spec)?;
            let animal = session
                .get("animal")
                .and_then(|v| v.as_str())
                .and_then(parse_animal_id)
                .or_else(|| animal_id_from_experiment(&name));
            let resolved = resolve_age_length(
                &spec,
                &name,
                // Recording metadata first; otherwise the `NORT (pre|post)` video
                // folder, which names the experiment (as in `audit.rs`).
                session_timepoint(&session).or_else(|| {
                    ["pre", "post"]
                        .into_iter()
                        .find(|t| name.contains(&format!("_{t}_")))
                }),
                animal.and_then(|id| ages.get(&id).copied()),
                // Recording metadata age (days); months-valued metadata would only warn.
                session
                    .get("age")
                    .and_then(|v| v.as_str())
                    .and_then(|v| v.trim().parse().ok()),
            )
            .map_err(BekiperError::Config)?;
            if let Some(warning) = &resolved.warning {
                tracing::warn!(%warning, "whisker age mismatch");
            }
            whisker.length = Some(WhiskerLengthSpec::Meters(MetersSpec {
                meters: resolved.length_m,
            }));
            whisker.age_table = Some(spec);
            whisker.age_days = resolved.age_days;
            whisker.length_source = Some(resolved.source);
        }
        Ok(())
    }
}

fn workbook_ages<'a>(
    cache: &'a mut HashMap<(String, String, String, String, String), AnimalAges>,
    base_dir: &Path,
    spec: &AgeTableSpec,
) -> Result<&'a AnimalAges> {
    let key = (
        spec.source.clone(),
        spec.sheet.clone(),
        spec.animal_column.clone(),
        spec.pre_age_column.clone(),
        spec.post_age_column.clone(),
    );
    if !cache.contains_key(&key) {
        let ages = read_animal_ages(
            &base_dir.join(&spec.source),
            &spec.sheet,
            &spec.animal_column,
            &spec.pre_age_column,
            &spec.post_age_column,
        )?;
        cache.insert(key.clone(), ages);
    }
    Ok(&cache[&key])
}
