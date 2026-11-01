use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use bekiper_core::types::{DetectionBackend, DetectionRecipe};

/// Detection source — either a recipe name (auto-detect) or a pre-computed file.
///
/// The `backend` field controls which detection method(s) to try. It accepts
/// either a single backend or a priority list (tried in order, first success wins):
///
/// ```toml
/// # Single backend
/// backend = "classical_cv"
///
/// # Priority list — try classical CV first, fall back to SAM2
/// backend = ["classical_cv", "sam2_auto"]
/// ```
///
/// Defaults to `["classical_cv"]` when omitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Recipe for auto-detection.
    pub recipe: Option<DetectionRecipe>,
    /// Detection backend(s) in priority order.
    #[serde(default = "default_backends")]
    pub backend: BackendPriority,
    /// Path to a pre-computed perimeters.toml file.
    pub perimeters: Option<PathBuf>,
    /// Optional overrides for the detection pipeline parameters.
    /// When set, these values override the recipe defaults.
    ///
    /// ```toml
    /// [detection.pipeline_overrides]
    /// threshold.floor_percentile = 80.0
    /// mouse_removal.dilate_frac = 0.01
    /// area_filter.min_frac = 0.01
    /// ```
    #[serde(default)]
    pub pipeline_overrides: Option<DetectionPipelineOverrides>,
    /// Optional deterministic OFT arena calibration settings.
    #[serde(default)]
    pub arena_calibration: Option<bekiper_detect::config::ArenaCalibrationConfig>,
}

impl DetectionConfig {
    /// Effective backend list: prepends `PreComputed` when a perimeters file is set.
    pub fn effective_backends(&self) -> BackendPriority {
        if self.perimeters.is_some() {
            let mut backends = vec![DetectionBackend::PreComputed];
            backends.extend(self.backend.backends().iter().copied());
            BackendPriority(backends)
        } else {
            self.backend.clone()
        }
    }
}

/// Optional overrides for detection pipeline parameters.
/// Any field set here overrides the recipe default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DetectionPipelineOverrides {
    pub expected_count: Option<usize>,
    pub threshold: Option<bekiper_detect::config::ThresholdConfig>,
    pub mouse_removal: Option<bekiper_detect::config::MouseRemovalConfig>,
    pub area_filter: Option<bekiper_detect::config::AreaFilterConfig>,
}

impl DetectionPipelineOverrides {
    /// Apply overrides to a pipeline config, returning the modified config.
    pub fn apply(
        &self,
        mut config: bekiper_detect::config::DetectionPipelineConfig,
    ) -> bekiper_detect::config::DetectionPipelineConfig {
        if let Some(count) = self.expected_count {
            config.expected_count = count;
        }
        if let Some(ref threshold) = self.threshold {
            config.threshold = threshold.clone();
        }
        if let Some(ref mouse_removal) = self.mouse_removal {
            config.mouse_removal = mouse_removal.clone();
        }
        if let Some(ref area_filter) = self.area_filter {
            config.area_filter = area_filter.clone();
        }
        config
    }
}

pub(crate) fn default_backends() -> BackendPriority {
    BackendPriority(vec![DetectionBackend::ClassicalCv])
}

/// One or more detection backends in priority order.
///
/// Deserializes from either a single string (`"classical_cv"`) or an array
/// (`["classical_cv", "sam2_auto"]`).
#[derive(Debug, Clone, Serialize)]
pub struct BackendPriority(pub Vec<DetectionBackend>);

impl BackendPriority {
    pub fn backends(&self) -> &[DetectionBackend] {
        &self.0
    }
}

impl Default for BackendPriority {
    fn default() -> Self {
        default_backends()
    }
}

impl<'de> Deserialize<'de> for BackendPriority {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de;

        struct BackendPriorityVisitor;

        impl<'de> de::Visitor<'de> for BackendPriorityVisitor {
            type Value = BackendPriority;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a detection backend string or array of backend strings")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
                let backend: DetectionBackend = v.parse().map_err(de::Error::custom)?;
                Ok(BackendPriority(vec![backend]))
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut backends = Vec::new();
                while let Some(s) = seq.next_element::<String>()? {
                    let backend: DetectionBackend = s.parse().map_err(de::Error::custom)?;
                    backends.push(backend);
                }
                if backends.is_empty() {
                    return Err(de::Error::custom("backend list must not be empty"));
                }
                Ok(BackendPriority(backends))
            }
        }

        deserializer.deserialize_any(BackendPriorityVisitor)
    }
}
