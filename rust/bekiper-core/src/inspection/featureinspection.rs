use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::*;

/// One arm's scored threshold segment (metres, perimeter frame).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmThresholdLine {
    pub arm: String,
    pub p1: (f64, f64),
    pub p2: (f64, f64),
}

// Every variant has the same optional presentation contract. Expand whole
// enum items so the public struct variants and serde wire shape stay identical.
macro_rules! feature_inspections {
    ($( $(#[$metadata:meta])* $variant:ident { $( $(#[$field_metadata:meta])* $field:ident : $type:ty,)* } ),* $(,)?) => {
        /// Per-feature inspection record. Every behavioural computation that lands in
        /// `AnalysisResult` emits one of these so the Python inspector can render a
        /// screenshot-overlay figure for it. The variant carries the minimal metadata
        /// needed for `bekiper-inspect` to look up the geometry/columns and pick a
        /// representative frame.
        ///
        /// The `label` field is the figure identifier (used as the output filename
        /// stem); the other fields point at perimeters / coordinate columns / result
        /// columns by name so the Python side never has to duplicate this information.
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(tag = "kind")]
        pub enum FeatureInspection {
            $(
                $(#[$metadata])*
                $variant {
                    $( $(#[$field_metadata])* $field: $type,)*
                    /// Preferred overlay frame, when nominated by a producer.
                    #[serde(default, skip_serializing_if = "Option::is_none")]
                    representative_frame_index: Option<u64>,
                    #[serde(default, skip_serializing_if = "Option::is_none")]
                    color: Option<String>,
                    #[serde(default, skip_serializing_if = "Option::is_none")]
                    linewidth: Option<f64>,
                    #[serde(default, skip_serializing_if = "Option::is_none")]
                    title_override: Option<String>,
                    /// Precomputed grouping key; None keeps this feature ungrouped.
                    #[serde(default, skip_serializing_if = "Option::is_none")]
                    story_key: Option<String>,
                },
            )*
        }
    };
}

feature_inspections! {
    /// Disjoint descriptive states computed by the native producer. These
    /// durations are whole-session quantities, not conditional preference.
    #[serde(rename = "state_distribution")]
    StateDistribution {
        label: String,
        category_labels: Vec<String>,
        frame_counts: Vec<u64>,
        seconds: Vec<f64>,
        percentages: Vec<f64>,
        denominator_frames: u64,
        denominator_seconds: f64,
        source_columns: Vec<String>,
        basis: String,
        style: String,
        description: String,
    },
    /// Bare perimeter overlay — no boolean state, just draw the geometry on
    /// the chosen frame.
    #[serde(rename = "perimeter")]
    Perimeter {
        label: String,
        perimeter_label: String,
        description: String,
    },
    /// Zone confinement: draw a perimeter and color the trajectory by an
    /// in/out boolean column.
    #[serde(rename = "zone")]
    Zone {
        label: String,
        perimeter_label: String,
        result_column: String,
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        description: String,
    },
    /// Quadrant grid: re-draw the NxN grid on the frame, fill cells by
    /// per-cell time fraction. Pre-computed occupancy grid can be provided.
    #[serde(rename = "quadrant")]
    Quadrant {
        label: String,
        quadrant_label: String,
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        description: String,
        /// Pre-computed occupancy grid (row-major, values 0.0–1.0 normalized
        /// by max). When present, Python uses this directly instead of
        /// computing from the parquet.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        occupancy_grid: Option<Vec<Vec<f64>>>,
    },
    /// Heuristic result: trajectory in pixel coords, colored by the boolean
    /// result column. Optionally overlays an associated perimeter.
    #[serde(rename = "heuristic")]
    Heuristic {
        label: String,
        result_column: String,
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        perimeter_label: Option<String>,
        description: String,
    },
    /// Heuristic result drawn as a point cloud rather than a linked
    /// trajectory. Intended for temporal/non-spatial behaviours such as
    /// freezing where connecting active samples with lines is misleading.
    #[serde(rename = "heuristic_cloud")]
    HeuristicCloud {
        label: String,
        result_column: String,
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        perimeter_label: Option<String>,
        description: String,
    },
    /// Standalone multi-panel breakdown of the boolean gates that compose
    /// a heuristic.
    #[serde(rename = "heuristic_decomposition")]
    HeuristicDecomposition {
        label: String,
        heuristic_name: String,
        result_column: String,
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        perimeter_label: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        components: Vec<HeuristicDecompositionComponent>,
        #[serde(default, skip_serializing_if = "is_false")]
        show_final_result: bool,
        description: String,
    },
    /// Body-axis or heading angle: at the chosen frame, draw a vector from
    /// `from_coordinate_label` to `to_coordinate_label` and annotate the
    /// resulting heading in degrees.
    #[serde(rename = "angle")]
    Angle {
        label: String,
        from_coordinate_label: String,
        to_coordinate_label: String,
        description: String,
    },
    /// Bare trajectory of one tracked label.
    #[serde(rename = "trajectory")]
    Trajectory {
        label: String,
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        description: String,
    },
    /// Scale derivation: how meters-per-pixel was derived. Rendered as
    /// an overlay showing the reference perimeter and dimension
    /// annotations on the first video frame.
    #[serde(rename = "scale_derivation")]
    ScaleDerivation {
        label: String,
        description: String,
    },
    /// Dual-panel figure: video frame + trajectory on the left, scalar
    /// line plot over frames on the right.
    #[serde(rename = "time_series")]
    TimeSeries {
        label: String,
        /// Column in the evaluation parquet that holds the scalar values to plot.
        value_column: String,
        /// Physical unit string for the Y axis label (e.g. "m/s", "m", "m²").
        unit: String,
        description: String,
    },
    /// Olfactory/sensing cone: draws a wedge (origin vertex + angle +
    /// length) at the representative frame.
    #[serde(rename = "ray_cast")]
    RayCast {
        label: String,
        /// Coordinate label for the ray origin (e.g. "snout", "nose").
        origin_label: String,
        /// Column in the parquet holding the heading angle in radians.
        angle_column: String,
        /// Ray length in meters.
        length_meters: f64,
        /// Full cone width in degrees (the cone spans ±half this angle around the heading).
        cone_width_deg: f64,
        description: String,
    },
    /// Proximity buffer: draws the primary perimeter plus an expanded
    /// buffer ring and colors the trajectory by distance to the perimeter.
    #[serde(rename = "proximity")]
    Proximity {
        label: String,
        perimeter_label: String,
        /// Distance buffer in meters around the perimeter.
        buffer_meters: f64,
        /// Boolean column in the parquet: True when inside the buffer zone.
        result_column: String,
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        description: String,
    },
    /// Object discrimination figure: highlights object perimeters on the arena
    /// and overlays per-object exploration time bars + discrimination indices.
    #[serde(rename = "object_discrimination")]
    NortDiscrimination {
        label: String,
        /// Stage-aware object-recognition protocol (`nort_training`,
        /// `nort_novelty`, `oul_training`, `oul_update`, `oul_test`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        protocol: Option<String>,
        /// Perimeter labels of the objects (e.g. ["novel", "familiar"]).
        object_labels: Vec<String>,
        /// Per-object exploration time in seconds, same order as `object_labels`.
        exploration_times: Vec<f64>,
        /// Discrimination Index: (T_N − T_F) / (T_N + T_F). Range [-1, 1].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        discrimination_index: Option<f64>,
        /// Recognition Index: T_N / (T_N + T_F). Range [0, 1].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        recognition_index: Option<f64>,
        /// Qualia decomposition: per-sensory-channel per-object statistics.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        qualia_channels: Option<Vec<QualiaChannelSpec>>,
        /// Number of distinct observation bouts (object switches) after
        /// temporal tolerance filtering.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        alternation_count: Option<u32>,
        /// Merged heuristic results: key = merge name, value = per-object seconds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        merged_channels: Option<HashMap<String, Vec<f64>>>,
        /// Coordinate labels for trajectory overlay.
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        description: String,
    },
    /// Arm entry sequence figure for Y-maze or radial-arm maze tasks.
    /// Renders numbered arm-entry annotations along the trajectory.
    #[serde(rename = "arm_sequence")]
    ArmSequence {
        label: String,
        /// Ordered arm-entry labels (e.g. ["A", "B", "C", "A", ...]).
        arm_sequence: Vec<String>,
        /// Coordinate labels for trajectory overlay.
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        /// Spontaneous alternation percentage (Y-maze); None for radial-arm.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        alternation_pct: Option<f64>,
        description: String,
    },
    /// Y-maze arm sectioning figure: draws the center/proximal boundary
    /// along each arm.
    #[serde(rename = "ymaze_arm_sections")]
    YMazeArmSections {
        label: String,
        /// Radial-maze perimeter label.
        perimeter_label: String,
        /// Fraction of arm length from center for the center/proximal boundary.
        proximal_threshold: f64,
        /// Scored p% cross-line per arm, in metres (arm-start cross-line
        /// geometry). Empty for archived radial results, which fall back to the
        /// centroid-projected chord.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        threshold_lines: Vec<ArmThresholdLine>,
        /// Coordinate labels for trajectory overlay.
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        description: String,
    },
    /// Open Field center-vs-periphery composite: occupancy heatmap with
    /// center/periphery zones and thigmotaxis score.
    #[serde(rename = "center_periphery")]
    CenterPeriphery {
        label: String,
        /// Perimeter label for the center zone.
        center_perimeter_label: String,
        /// Perimeter label for the arena boundary.
        arena_perimeter_label: String,
        /// Coordinate labels for trajectory / heatmap.
        #[serde(deserialize_with = "deserialize_labels")]
        coordinate_labels: Vec<String>,
        /// Percentage of trial time in center.
        center_time_pct: f64,
        /// Percentage of trial time in periphery.
        periphery_time_pct: f64,
        description: String,
    },
    /// NORT object-observation detection fields mapped onto the animal: the
    /// nose cone (apex at the nose, axis `head_label`→`nose_label`), its reach
    /// and contact circle, and, when `whisker_*` are set, the two lateral
    /// whisker fans at the apex `head + apex_fraction * (nose - head)`.
    /// Per-object hit columns (parallel to `object_labels`) highlight what
    /// fires on the chosen frame; all are optional diagnostics.
    #[serde(rename = "detection_fields")]
    DetectionFields {
        label: String,
        nose_label: String,
        head_label: String,
        /// Half-angle of the nose cone (degrees) around the head axis.
        cone_half_angle_deg: f64,
        /// Nose reach (metres) and contact radius (metres).
        nose_reach_m: f64,
        contact_distance_m: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        whisker_apex_fraction: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        whisker_half_angle_deg: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        whisker_length_m: Option<f64>,
        /// Object perimeter labels to outline.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        object_labels: Vec<String>,
        /// Per-object boolean columns: nose within reach, nose angle/contact
        /// gate, and whisker-fan hit (each parallel to `object_labels`).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        nose_near_columns: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        nose_gate_columns: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        whisker_hit_columns: Vec<String>,
        description: String,
    },
}
