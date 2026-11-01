use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::scale::ScaleDerivationInfo;
use crate::types::Label;
use crate::video::VideoMetadata;

use super::*;

/// Shared categorical identities for figures and native videos. FNV-1a gives
/// arbitrary labels a stable fallback without depending on panel ordering.
pub fn inspection_identity_color(label: &str) -> &'static str {
    const PALETTE: [&str; 8] = [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#332288",
    ];
    if let [
        b'P',
        p @ b'0'..=b'1',
        b' ',
        b'G',
        g @ b'0'..=b'1',
        b' ',
        b'F' | b'O',
        o @ b'0'..=b'1',
    ] = label.as_bytes()
    {
        return PALETTE
            [usize::from(p - b'0') + 2 * usize::from(g - b'0') + 4 * usize::from(o - b'0')];
    }
    let index = match label {
        "nose" | "familiar" => 0,
        "left_ear" | "novel" => 1,
        "right_ear" => 2,
        "center_ear" => 3,
        "tail" | "base_tail" | "tail_base" => 4,
        "body_center" => 5,
        "Missing" => return "#777777",
        "Unassigned" => return "#dddddd",
        "Overlapping targets" => return "#332288",
        _ => label
            .strip_prefix("object_")
            .filter(|n| {
                !n.is_empty()
                    && n.bytes().all(|b| b.is_ascii_digit())
                    && n.bytes().any(|b| b != b'0')
            })
            .map(|n| {
                let remainder = n
                    .bytes()
                    .fold(0_usize, |r, b| (10 * r + usize::from(b - b'0')) % 8);
                (remainder + 7) % 8
            })
            .unwrap_or_else(|| {
                (label
                    .as_bytes()
                    .iter()
                    .fold(0xcbf29ce484222325_u64, |h, b| {
                        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
                    })
                    % 8) as usize
            }),
    };
    PALETTE[index]
}

#[cfg(test)]
mod identity_color_tests {
    use super::inspection_identity_color;

    #[test]
    fn decimal_object_identities_are_ascii_and_unbounded() {
        for (label, expected) in [
            ("object_2", "#E69F00"),
            ("object_0002", "#E69F00"),
            ("object_18446744073709551616", "#332288"),
            ("object_²", "#332288"),
            ("object_+2", "#009E73"),
        ] {
            assert_eq!(inspection_identity_color(label), expected, "{label}");
        }
    }
}

macro_rules! common_field {
    ($feature:expr, $field:ident) => {
        match $feature {
            FeatureInspection::StateDistribution { $field, .. }
            | FeatureInspection::Perimeter { $field, .. }
            | FeatureInspection::Zone { $field, .. }
            | FeatureInspection::Quadrant { $field, .. }
            | FeatureInspection::Heuristic { $field, .. }
            | FeatureInspection::HeuristicCloud { $field, .. }
            | FeatureInspection::HeuristicDecomposition { $field, .. }
            | FeatureInspection::Angle { $field, .. }
            | FeatureInspection::Trajectory { $field, .. }
            | FeatureInspection::ScaleDerivation { $field, .. }
            | FeatureInspection::TimeSeries { $field, .. }
            | FeatureInspection::RayCast { $field, .. }
            | FeatureInspection::Proximity { $field, .. }
            | FeatureInspection::NortDiscrimination { $field, .. }
            | FeatureInspection::ArmSequence { $field, .. }
            | FeatureInspection::YMazeArmSections { $field, .. }
            | FeatureInspection::CenterPeriphery { $field, .. }
            | FeatureInspection::DetectionFields { $field, .. } => $field,
        }
    };
}

impl FeatureInspection {
    /// Identifier used by the Python CLI as the output figure stem.
    pub fn label(&self) -> &str {
        common_field!(self, label)
    }

    /// Preferred video-frame index for this feature's overlay, if one was
    /// nominated at ingest. When `None`, Python falls back to its internal
    /// frame picker.
    pub fn representative_frame_index(&self) -> Option<u64> {
        *common_field!(self, representative_frame_index)
    }

    /// Short discriminator string used in the figure filename.
    pub fn kind(&self) -> &'static str {
        match self {
            FeatureInspection::StateDistribution { .. } => "state_distribution",
            FeatureInspection::Perimeter { .. } => "perimeter",
            FeatureInspection::Zone { .. } => "zone",
            FeatureInspection::Quadrant { .. } => "quadrant",
            FeatureInspection::Heuristic { .. } => "heuristic",
            FeatureInspection::HeuristicCloud { .. } => "heuristic_cloud",
            FeatureInspection::HeuristicDecomposition { .. } => "heuristic_decomposition",
            FeatureInspection::Angle { .. } => "angle",
            FeatureInspection::Trajectory { .. } => "trajectory",
            FeatureInspection::ScaleDerivation { .. } => "scale_derivation",
            FeatureInspection::TimeSeries { .. } => "time_series",
            FeatureInspection::RayCast { .. } => "ray_cast",
            FeatureInspection::Proximity { .. } => "proximity",
            FeatureInspection::NortDiscrimination { .. } => "object_discrimination",
            FeatureInspection::ArmSequence { .. } => "arm_sequence",
            FeatureInspection::YMazeArmSections { .. } => "ymaze_arm_sections",
            FeatureInspection::CenterPeriphery { .. } => "center_periphery",
            FeatureInspection::DetectionFields { .. } => "detection_fields",
        }
    }

    /// Optional color override for this feature's rendering.
    pub fn color(&self) -> Option<&str> {
        common_field!(self, color).as_deref()
    }

    /// Optional linewidth override for this feature's rendering.
    pub fn linewidth(&self) -> Option<f64> {
        *common_field!(self, linewidth)
    }

    /// Story grouping key for multi-panel figures, if pre-computed.
    pub fn story_key(&self) -> Option<&str> {
        common_field!(self, story_key).as_deref()
    }

    /// Set the story grouping key (used by post-processing passes).
    pub fn set_story_key(&mut self, key: String) {
        *common_field!(self, story_key) = Some(key);
    }

    /// Set the representative frame index (used by post-processing passes
    /// that compute the best frame from the evaluated DataFrame).
    pub fn set_representative_frame_index(&mut self, idx: u64) {
        *common_field!(self, representative_frame_index) = Some(idx);
    }

    /// Return the result column name for boolean-column–backed features.
    pub fn result_column(&self) -> Option<&str> {
        match self {
            FeatureInspection::Zone { result_column, .. }
            | FeatureInspection::Heuristic { result_column, .. }
            | FeatureInspection::HeuristicCloud { result_column, .. }
            | FeatureInspection::HeuristicDecomposition { result_column, .. }
            | FeatureInspection::Proximity { result_column, .. } => Some(result_column),
            _ => None,
        }
    }

    /// Return the coordinate labels for features that carry them.
    pub fn coordinate_labels(&self) -> Option<&[String]> {
        match self {
            FeatureInspection::Zone {
                coordinate_labels, ..
            }
            | FeatureInspection::Quadrant {
                coordinate_labels, ..
            }
            | FeatureInspection::Heuristic {
                coordinate_labels, ..
            }
            | FeatureInspection::HeuristicCloud {
                coordinate_labels, ..
            }
            | FeatureInspection::HeuristicDecomposition {
                coordinate_labels, ..
            }
            | FeatureInspection::Trajectory {
                coordinate_labels, ..
            }
            | FeatureInspection::Proximity {
                coordinate_labels, ..
            }
            | FeatureInspection::NortDiscrimination {
                coordinate_labels, ..
            }
            | FeatureInspection::ArmSequence {
                coordinate_labels, ..
            }
            | FeatureInspection::YMazeArmSections {
                coordinate_labels, ..
            }
            | FeatureInspection::CenterPeriphery {
                coordinate_labels, ..
            } => Some(coordinate_labels),
            FeatureInspection::Angle { .. } => None, // Angle has separate from/to labels, not a Vec
            FeatureInspection::RayCast { .. } => None, // origin_label only
            _ => None,
        }
    }

    /// Optional title override for this feature's figure heading.
    pub fn title_override(&self) -> Option<&str> {
        common_field!(self, title_override).as_deref()
    }
}

/// Column mapping for a tracked label's coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinateColumnSpec {
    pub x: String,
    pub y: String,
}

/// Full inspection manifest — everything the Python inspection package needs
/// to recreate plots from the Rust analysis output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionManifest {
    pub video: VideoMetadata,
    pub video_path: Option<PathBuf>,
    /// Optional source-study context supplied by an ingress profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_context: Option<IngressSessionContext>,
    /// Path to a single arena screenshot image (e.g. `0.frame-721.jpeg`).
    /// Used as the background for inspection figures when no video is available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_image: Option<PathBuf>,
    pub labels: Vec<Label>,
    pub label_colors: HashMap<Label, String>,
    pub coordinate_columns: HashMap<Label, CoordinateColumnSpec>,
    pub perimeters: Vec<PerimeterSpec>,
    pub heuristics: Vec<HeuristicMeta>,
    pub settings: InspectionSettings,
    /// How meters_per_pixel was derived; present when scale was auto-detected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_derivation: Option<ScaleDerivationInfo>,
    /// Auto-generated quadrant grids (NxN). Empty for tasks that do not run
    /// quadrant analysis. Defaulted for backwards-compatible deserialization
    /// of pre-feature_inspections manifests.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quadrants: Vec<QuadrantSpec>,
    /// Per-feature inspection figure declarations. Every behavioural element
    /// (perimeter, zone, quadrant, heuristic, angle, trajectory) emits one
    /// entry here so the Python inspector can render a screenshot-overlay
    /// figure for each. Defaulted for backwards-compatibility.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feature_inspections: Vec<FeatureInspection>,

    /// Perimeters pre-converted to video-pixel space via [`ToVideoPixels`].
    /// Python renderers use these directly on frames — only a uniform
    /// `frame_scale` ratio is needed if the displayed frame differs from
    /// `video.resolution`. Empty for old manifests (Python falls back to
    /// metric perimeters + manual conversion).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pixel_perimeters: Vec<PerimeterSpec>,
    /// Quadrant grids pre-converted to video-pixel space.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pixel_quadrants: Vec<QuadrantSpec>,
}

/// Runtime settings relevant to inspection/visualization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InspectionSettings {
    /// Resolved analysis-space transform, inverted for overlays on source images.
    #[serde(default)]
    pub invert_y: bool,
    pub minimum_seconds_tolerance: f64,
    pub maximum_seconds_distraction: f64,
    pub meters_per_pixel: f64,
}

/// Wrapper for serializing a `Vec<PerimeterSpec>` as a TOML file.
/// TOML requires a root table, so bare arrays must be wrapped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerimetersFile {
    pub perimeters: Vec<PerimeterSpec>,
}

#[cfg(test)]
mod detection_fields_tests {
    use super::FeatureInspection;

    #[test]
    fn detection_fields_wire_shape_round_trips_with_optional_fan() {
        let raw = r#"kind = "detection_fields"
label = "detection_fields"
nose_label = "nose"
head_label = "center_ear"
cone_half_angle_deg = 45.0
nose_reach_m = 0.03
contact_distance_m = 0.005
description = "d""#;
        let feature: FeatureInspection = toml::from_str(raw).unwrap();
        assert_eq!(feature.kind(), "detection_fields");
        assert!(matches!(
            &feature,
            FeatureInspection::DetectionFields { whisker_length_m: None, object_labels, .. }
                if object_labels.is_empty()
        ));
        let back: FeatureInspection = toml::from_str(&toml::to_string(&feature).unwrap()).unwrap();
        assert_eq!(back.label(), "detection_fields");
    }
}
