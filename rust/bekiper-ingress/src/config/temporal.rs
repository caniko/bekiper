use bekiper_core::error::{BekiperError, Result};
use bekiper_feature::tolerance::{DwellOverrides, TemporalPolicy};

use super::{IngressConfig, TaskParamsOverrides};

impl IngressConfig {
    pub(super) fn validate_ymaze_temporal_policies(&self) -> Result<()> {
        for params in self
            .experiments
            .iter()
            .filter_map(|e| e.task_params.as_ref())
            .chain(
                self.animal_scan
                    .iter()
                    .filter_map(|s| s.task_params.as_ref()),
            )
        {
            validate_policy(params)?;
        }
        for scan in &self.scan {
            validate_variant_policies(scan.task_params.as_ref(), &scan.variants)?;
        }
        for profile in &self.ingress_profile {
            match profile.profile {
                super::profile::IngressProfileId::VideoMap => {
                    // VideoMap discovery uses task-local params, then variants.
                    for task in &profile.tasks {
                        validate_variant_policies(task.task_params.as_ref(), &task.variants)?;
                    }
                }
                super::profile::IngressProfileId::SequenceTable => {
                    if let Some(params) = &profile.task_params {
                        validate_policy(params)?;
                    }
                }
            }
        }
        Ok(())
    }
    /// Overlay every template and variant before expansion, so constructor/CLI
    /// values beat variant TOML too. Also used by native annotation previews.
    pub fn apply_ymaze_dwell_overrides(&mut self, overrides: DwellOverrides) -> Result<()> {
        if overrides.is_empty() {
            return Ok(());
        }
        overrides
            .apply(Default::default())
            .validate()
            .map_err(BekiperError::Config)?;
        let apply = |params: &mut TaskParamsOverrides, inherited: TemporalPolicy| {
            let ymaze = params.ymaze.get_or_insert_with(Default::default);
            let policy = overrides.apply(ymaze.temporal_policy.unwrap_or(inherited));
            ymaze.temporal_policy = Some(policy);
            policy
        };
        for experiment in &mut self.experiments {
            apply(
                experiment.task_params.get_or_insert_with(Default::default),
                TemporalPolicy::ymaze_default(),
            );
        }
        for scan in &mut self.scan {
            let inherited = apply(
                scan.task_params.get_or_insert_with(Default::default),
                TemporalPolicy::ymaze_default(),
            );
            for variant in &mut scan.variants {
                apply(&mut variant.task_params, inherited);
            }
        }
        for scan in &mut self.animal_scan {
            apply(
                scan.task_params.get_or_insert_with(Default::default),
                TemporalPolicy::ymaze_default(),
            );
        }
        for profile in &mut self.ingress_profile {
            let inherited = apply(
                profile.task_params.get_or_insert_with(Default::default),
                TemporalPolicy::ymaze_default(),
            );
            for task in &mut profile.tasks {
                let inherited = apply(
                    task.task_params.get_or_insert_with(Default::default),
                    inherited,
                );
                for variant in &mut task.variants {
                    apply(&mut variant.task_params, inherited);
                }
            }
        }
        self.post_analysis.ymaze_dwell_overrides =
            self.post_analysis.ymaze_dwell_overrides.merged(overrides);
        Ok(())
    }
}

fn validate_variant_policies(
    base: Option<&TaskParamsOverrides>,
    variants: &[super::TaskVariant],
) -> Result<()> {
    let base = base.cloned().unwrap_or_default();
    if variants.is_empty() {
        validate_policy(&base)
    } else {
        // As in expand_experiment_variants, a template is not itself executed
        // when variants exist. Validate each effective field-by-field merge.
        for variant in variants {
            validate_policy(&base.merged(&variant.task_params))?;
        }
        Ok(())
    }
}

fn validate_policy(params: &TaskParamsOverrides) -> Result<()> {
    if params
        .oft_window_seconds
        .is_some_and(|v| !v.is_finite() || v < 0.0)
    {
        return Err(BekiperError::Config(
            "OFT window seconds must be finite and nonnegative".into(),
        ));
    }
    if let Some(ymaze) = &params.ymaze
        && let Some(policy) = ymaze.temporal_policy
    {
        let threshold = ymaze
            .proximal_threshold
            .unwrap_or(bekiper_behaviour::radial_arm::DEFAULT_PROXIMAL_THRESHOLD);
        // Only the archived radial geometry forces uninterrupted membership.
        let radial = ymaze.threshold_geometry.unwrap_or_default()
            == bekiper_behaviour::radial_arm::YMazeThresholdGeometry::ImenRadialVertexSpanV1;
        if radial
            && ymaze
                .principal_thresholds
                .as_deref()
                .unwrap_or(&bekiper_behaviour::radial_arm::DEFAULT_PRINCIPAL_THRESHOLDS)
                .contains(&threshold)
        {
            policy.validate_uninterrupted()
        } else {
            policy.validate()
        }
        .map_err(BekiperError::Config)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "temporal/tests.rs"]
mod tests;
