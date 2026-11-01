use serde::{Deserialize, Serialize};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::types::MidpointGroup;

use crate::PoseBackend;
use crate::skeleton::Skeleton;

/// Keypoint prediction for a single body part in a single frame.
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
}

/// How to normalize pixel values before feeding into the model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Normalization {
    /// Scale `[0,255]` to `[0,1]`.
    ZeroOne,
    /// ImageNet-style: (pixel/255 - mean) / std per channel.
    ImageNet { mean: [f32; 3], std: [f32; 3] },
}

/// How the model encodes keypoint predictions in its output tensor.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OutputLayout {
    /// Shape [1, K, 3] — each keypoint has (x, y, confidence).
    /// Used by DLC SuperAnimal, Lightning Pose.
    KeypointsFlat,

    /// Shape [1, K, H, W] — heatmaps per keypoint, argmax for location.
    /// Used by some SLEAP models.
    Heatmaps,
}

/// Alternate labels that a model or dataset may use for the same anatomical
/// keypoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelAlias {
    pub canonical: String,
    pub aliases: Vec<String>,
}

/// Compile-time pose model definition.
///
/// Built-in models implement this trait so labels and skeleton topology cannot
/// drift apart: producing a runnable [`ModelSpec`] requires both.
pub trait PoseModelDefinition {
    fn backend(&self) -> PoseBackend;
    fn labels(&self) -> Vec<String>;
    fn input_height(&self) -> u32;
    fn input_width(&self) -> u32;
    fn input_channels(&self) -> u32;
    fn channels_first(&self) -> bool;
    fn normalization(&self) -> Normalization;
    fn output_layout(&self) -> OutputLayout;
    fn skeleton(&self) -> Skeleton;

    fn midpoint_groups(&self) -> Vec<MidpointGroup> {
        Vec::new()
    }

    fn label_aliases(&self) -> Vec<LabelAlias> {
        Vec::new()
    }

    fn to_model_spec(&self) -> Result<ModelSpec> {
        let spec = ModelSpec {
            backend: self.backend(),
            labels: self.labels(),
            input_height: self.input_height(),
            input_width: self.input_width(),
            input_channels: self.input_channels(),
            channels_first: self.channels_first(),
            normalization: self.normalization(),
            output_layout: self.output_layout(),
            skeleton: self.skeleton(),
            midpoint_groups: self.midpoint_groups(),
            label_aliases: self.label_aliases(),
        };
        spec.validate()?;
        Ok(spec)
    }
}

pub struct DlcSuperAnimalDefinition;
pub struct SleapSingleInstanceDefinition;
pub struct LightningPoseDefinition;

include!("spec/dlc_definition.rs");

include!("spec/sleap_definition.rs");

include!("spec/lightning_definition.rs");

/// Full specification for running an ONNX pose model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Which backend produced this model.
    pub backend: PoseBackend,

    /// Ordered list of keypoint names matching the model's output indices.
    pub labels: Vec<String>,

    /// Model input height in pixels.
    pub input_height: u32,

    /// Model input width in pixels.
    pub input_width: u32,

    /// Number of input channels (typically 3 for RGB, 1 for grayscale).
    pub input_channels: u32,

    /// Whether the model expects NCHW (true) or NHWC (false) layout.
    pub channels_first: bool,

    /// Pixel normalization strategy.
    pub normalization: Normalization,

    /// How the model outputs keypoint predictions.
    pub output_layout: OutputLayout,

    /// Model-owned anatomical topology. Required for all model specs,
    /// including custom specs.
    pub skeleton: Skeleton,

    /// Model-owned derived midpoint labels required by the skeleton topology.
    #[serde(default)]
    pub midpoint_groups: Vec<MidpointGroup>,

    /// Optional alternate label names for anatomical keypoints.
    #[serde(default)]
    pub label_aliases: Vec<LabelAlias>,
}

impl ModelSpec {
    /// Default spec for DLC SuperAnimal top-view mouse models.
    pub fn dlc_super_animal() -> Self {
        DlcSuperAnimalDefinition
            .to_model_spec()
            .expect("built-in DLC SuperAnimal spec must validate")
    }

    /// Default spec for SLEAP single-instance models.
    pub fn sleap_single_instance() -> Self {
        SleapSingleInstanceDefinition
            .to_model_spec()
            .expect("built-in SLEAP spec must validate")
    }

    /// Default spec for Lightning Pose models.
    pub fn lightning_pose() -> Self {
        LightningPoseDefinition
            .to_model_spec()
            .expect("built-in Lightning Pose spec must validate")
    }

    pub fn validate(&self) -> Result<()> {
        if self.labels.is_empty() {
            return Err(BekiperError::Config(
                "model spec must define at least one label".into(),
            ));
        }
        if self.input_height == 0 || self.input_width == 0 || self.input_channels == 0 {
            return Err(BekiperError::Config(
                "model spec input dimensions and channels must be > 0".into(),
            ));
        }
        let mut seen_labels = std::collections::HashSet::new();
        for label in &self.labels {
            if label.is_empty() {
                return Err(BekiperError::Config(
                    "model spec labels cannot contain empty strings".into(),
                ));
            }
            if !seen_labels.insert(label.as_str()) {
                return Err(BekiperError::Config(format!(
                    "model spec labels contain duplicate label '{label}'"
                )));
            }
        }

        let mut expanded_labels = self.validate_midpoint_groups()?;
        expanded_labels.extend(self.labels.iter().cloned());
        expanded_labels.sort();
        expanded_labels.dedup();
        self.skeleton.validate_against_labels(&expanded_labels)?;

        for alias in &self.label_aliases {
            if alias.canonical.is_empty() {
                return Err(BekiperError::Config(
                    "model spec label alias canonical name cannot be empty".into(),
                ));
            }
            if !self.labels.iter().any(|label| label == &alias.canonical) {
                return Err(BekiperError::Config(format!(
                    "model spec label alias canonical '{}' is not in labels",
                    alias.canonical
                )));
            }
            if alias.aliases.is_empty() {
                return Err(BekiperError::Config(format!(
                    "model spec label alias for '{}' must include at least one alias",
                    alias.canonical
                )));
            }
            for alias_label in &alias.aliases {
                if alias_label.is_empty() {
                    return Err(BekiperError::Config(format!(
                        "model spec label alias for '{}' contains an empty alias",
                        alias.canonical
                    )));
                }
            }
        }

        Ok(())
    }

    fn validate_midpoint_groups(&self) -> Result<Vec<String>> {
        let raw_labels: std::collections::HashSet<&str> =
            self.labels.iter().map(String::as_str).collect();
        let mut available = raw_labels.clone();
        let mut outputs = std::collections::HashSet::new();
        let mut derived_labels = Vec::new();

        for group in &self.midpoint_groups {
            if group.output_label.is_empty() {
                return Err(BekiperError::Config(
                    "model spec midpoint output_label cannot be empty".into(),
                ));
            }
            if raw_labels.contains(group.output_label.as_str()) {
                return Err(BekiperError::Config(format!(
                    "model spec midpoint '{}' duplicates a raw model label",
                    group.output_label
                )));
            }
            if !outputs.insert(group.output_label.as_str()) {
                return Err(BekiperError::Config(format!(
                    "model spec midpoint '{}' is defined more than once",
                    group.output_label
                )));
            }
            if group.source_labels.is_empty() {
                return Err(BekiperError::Config(format!(
                    "model spec midpoint '{}' must define at least one source label",
                    group.output_label
                )));
            }
            let mut source_seen = std::collections::HashSet::new();
            let mut missing = Vec::new();
            for source in &group.source_labels {
                if source.is_empty() {
                    return Err(BekiperError::Config(format!(
                        "model spec midpoint '{}' contains an empty source label",
                        group.output_label
                    )));
                }
                if !source_seen.insert(source.as_str()) {
                    return Err(BekiperError::Config(format!(
                        "model spec midpoint '{}' contains duplicate source label '{}'",
                        group.output_label, source
                    )));
                }
                if !available.contains(source.as_str()) {
                    missing.push(source.clone());
                }
            }
            if !missing.is_empty() {
                return Err(BekiperError::Config(format!(
                    "model spec midpoint '{}' references unavailable source label(s): {}",
                    group.output_label,
                    missing.join(", ")
                )));
            }

            available.insert(group.output_label.as_str());
            derived_labels.push(group.output_label.clone());
        }

        Ok(derived_labels)
    }
}
