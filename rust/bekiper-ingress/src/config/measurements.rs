//! Scientific measurement overlays, applied to templates and variants before expansion.
use super::{IngressConfig, TaskParamsOverrides};
use bekiper_core::error::{BekiperError, Result};

#[derive(Debug, Clone, Copy, Default)]
pub struct MeasurementOverrides {
    pub oft_window_seconds: Option<f64>,
    pub nort_strict_boundaries: Option<bool>,
}

impl MeasurementOverrides {
    pub fn with_env(self, lookup: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let seconds = match self.oft_window_seconds {
            Some(v) => Some(v),
            None => lookup("BEKIPER_OFT_WINDOW_SECONDS")
                .map(|v| {
                    v.parse::<f64>().map_err(|_| {
                        BekiperError::Config("BEKIPER_OFT_WINDOW_SECONDS must be a number".into())
                    })
                })
                .transpose()?,
        };
        if seconds.is_some_and(|v| !v.is_finite() || v < 0.0) {
            return Err(BekiperError::Config(
                "OFT window seconds must be finite and nonnegative".into(),
            ));
        }
        let strict = match self.nort_strict_boundaries {
            Some(v) => Some(v),
            None => lookup("BEKIPER_NORT_STRICT_BOUNDARIES")
                .map(|v| match v.as_str() {
                    "true" | "1" => Ok(true),
                    "false" | "0" => Ok(false),
                    _ => Err(BekiperError::Config(
                        "BEKIPER_NORT_STRICT_BOUNDARIES must be true or false".into(),
                    )),
                })
                .transpose()?,
        };
        Ok(Self {
            oft_window_seconds: seconds,
            nort_strict_boundaries: strict,
        })
    }
}

impl IngressConfig {
    pub fn apply_measurement_overrides(&mut self, options: MeasurementOverrides) {
        let apply = |params: &mut Option<TaskParamsOverrides>| {
            let params = params.get_or_insert_with(Default::default);
            if let Some(value) = options.oft_window_seconds {
                params.oft_window_seconds = Some(value);
            }
            if let Some(value) = options.nort_strict_boundaries {
                params
                    .exploration
                    .get_or_insert_with(Default::default)
                    .strict_boundaries = Some(value);
            }
        };
        if options.oft_window_seconds.is_none() && options.nort_strict_boundaries.is_none() {
            return;
        }
        for experiment in &mut self.experiments {
            apply(&mut experiment.task_params);
        }
        for scan in &mut self.scan {
            apply(&mut scan.task_params);
            for variant in &mut scan.variants {
                let mut p = Some(variant.task_params.clone());
                apply(&mut p);
                variant.task_params = p.unwrap();
            }
        }
        for scan in &mut self.animal_scan {
            apply(&mut scan.task_params);
        }
        for profile in &mut self.ingress_profile {
            apply(&mut profile.task_params);
            for task in &mut profile.tasks {
                apply(&mut task.task_params);
                for variant in &mut task.variants {
                    let mut p = Some(variant.task_params.clone());
                    apply(&mut p);
                    variant.task_params = p.unwrap();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Owner decision 2026-10-28: OFT and NORT share the 8 cm band / 24 cm center.
    #[test]
    fn imen_oft_and_nort_share_eight_cm_band() {
        use bekiper_core::types::TaskType;
        use bekiper_detect::config::ArenaCenterRule;
        let config: IngressConfig = toml::from_str(include_str!(
            "../../../../examples/neo/imen/bekiper_config.toml"
        ))
        .unwrap();
        let arena = 0.40;
        let mut checked = 0;
        for task in &config.ingress_profile[0].tasks {
            if !matches!(
                task.task_type,
                TaskType::OpenField | TaskType::ObjectRecognition
            ) {
                continue;
            }
            let rule = task
                .arena_calibration
                .as_ref()
                .unwrap()
                .effective_center_rule();
            let (w, h) = match rule {
                ArenaCenterRule::VideoRelative {
                    width_fraction: Some(w),
                    height_fraction: Some(h),
                } => (arena * w, arena * h),
                ArenaCenterRule::FrameCentered {
                    center_width_meters,
                    center_height_meters,
                    ..
                } => (center_width_meters, center_height_meters),
                other => panic!("unexpected center rule {other:?}"), // coverage: unreachable — test-only guard; the Imen config uses only the two rules matched above
            };
            for side in [w, h] {
                assert!(((arena - side) / 2.0 - 0.08).abs() < 1e-9, "{side}");
            }
            let params = task.task_params.as_ref().unwrap();
            if task.task_type == TaskType::OpenField {
                let fraction = params.peripheral_band_fraction;
                assert!((fraction.unwrap() * arena - 0.08).abs() < 1e-9);
            } else {
                // Owner decision 2026-10-28: merged bouts >= 0.25 s.
                let exploration = params.exploration.as_ref().unwrap();
                assert_eq!(exploration.min_bout_seconds, Some(0.25));
            }
            checked += 1;
        }
        assert_eq!(checked, 2);
    }

    #[test]
    fn constructor_measurements_win_over_invalid_env_and_variant_file_values() {
        let explicit = MeasurementOverrides {
            oft_window_seconds: Some(240.0),
            nort_strict_boundaries: Some(false),
        };
        let resolved = explicit.with_env(|_| Some("invalid".into())).unwrap();
        let mut config: IngressConfig = toml::from_str(include_str!(
            "../../../../examples/neo/imen/bekiper_config.toml"
        ))
        .unwrap();
        config.apply_measurement_overrides(resolved);
        for task in &config.ingress_profile[0].tasks {
            let params = task.task_params.as_ref().unwrap();
            assert_eq!(params.oft_window_seconds, Some(240.0));
            assert_eq!(
                params.exploration.as_ref().unwrap().strict_boundaries,
                Some(false)
            );
            for variant in &task.variants {
                let params = params.merged(&variant.task_params);
                assert_eq!(params.oft_window_seconds, Some(240.0));
            }
        }
        assert!(
            MeasurementOverrides::default()
                .with_env(|_| Some("invalid".into()))
                .is_err()
        );
        let resolved = MeasurementOverrides::default()
            .with_env(|k| {
                std::collections::HashMap::from([
                    ("BEKIPER_OFT_WINDOW_SECONDS", "300"),
                    ("BEKIPER_NORT_STRICT_BOUNDARIES", "true"),
                ])
                .get(k)
                .map(|v| v.to_string())
            })
            .unwrap();
        assert_eq!(resolved.oft_window_seconds, Some(300.0));
        assert_eq!(resolved.nort_strict_boundaries, Some(true));
    }
}
