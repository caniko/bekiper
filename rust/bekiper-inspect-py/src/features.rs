//! PyO3 wrappers for `FeatureInspection` (16 variants) and helper structs.
//!
//! The Python API mirrors the prior dataclass-per-variant layout:
//! a `kind` class constant identifies the variant, all fields are exposed
//! as read-only Python attributes via `get_all`. Construction matches the
//! prior dataclass keyword API (kw-only with sensible defaults).

#![allow(clippy::too_many_arguments)]
#![allow(unused_parens)] // `(Vec::new())` defaults must be a single macro tt

use std::collections::HashMap;

use crate::perimeter::PyPerimeterSpec;
use bekiper_core::inspection::{
    FeatureInspection, HeuristicDecompositionComponent, QualiaChannelSpec,
};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Helper structs used inside FeatureInspection variants
// ---------------------------------------------------------------------------

include!("features/helpers.rs");

#[derive(Clone, Default)]
struct Common {
    representative_frame_index: Option<u64>,
    color: Option<String>,
    linewidth: Option<f64>,
    title_override: Option<String>,
    story_key: Option<String>,
}

impl Common {
    fn from_rust(feature: &FeatureInspection) -> Self {
        Self {
            representative_frame_index: feature.representative_frame_index(),
            color: feature.color().map(str::to_owned),
            linewidth: feature.linewidth(),
            title_override: feature.title_override().map(str::to_owned),
            story_key: feature.story_key().map(str::to_owned),
        }
    }
}

// Macro to apply common getters and the `kind` classattr.
macro_rules! common_methods {
    ($cls:ident, $kind:literal) => {
        #[pymethods]
        impl $cls {
            #[classattr]
            #[allow(non_upper_case_globals)]
            const kind: &'static str = $kind;

            #[getter]
            fn representative_frame_index(&self) -> Option<u64> {
                self.common.representative_frame_index
            }
            #[getter]
            fn color(&self) -> Option<String> {
                self.common.color.clone()
            }
            #[getter]
            fn linewidth(&self) -> Option<f64> {
                self.common.linewidth
            }
            #[getter]
            fn title_override(&self) -> Option<String> {
                self.common.title_override.clone()
            }
            #[getter]
            fn story_key(&self) -> Option<String> {
                self.common.story_key.clone()
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------

// Declare each Python class once; getters and constructor signatures share
// the same field inventory, including the existing optional defaults.
macro_rules! feature_class {
    ($class:ident, $name:literal, $kind:literal;
        required { $($required:ident : $required_type:ty,)* }
        optional { $($optional:ident : $optional_type:ty = $default:tt,)* }
    ) => {
        #[pyclass(name = $name, module = "bekiper_rs")]
        #[derive(Clone)]
        pub struct $class {
            $(#[pyo3(get)] pub $required: $required_type,)*
            $(#[pyo3(get)] pub $optional: $optional_type,)*
            common: Common,
        }
        common_methods!($class, $kind);
        #[pymethods]
        impl $class {
            #[new]
            #[pyo3(signature = (
                $($required,)* $($optional=$default,)*
                representative_frame_index=None, color=None, linewidth=None,
                title_override=None, story_key=None,
            ))]
            fn new(
                $($required: $required_type,)* $($optional: $optional_type,)*
                representative_frame_index: Option<u64>, color: Option<String>,
                linewidth: Option<f64>, title_override: Option<String>, story_key: Option<String>,
            ) -> Self {
                Self { $($required,)* $($optional,)*
                    common: Common { representative_frame_index, color, linewidth, title_override, story_key },
                }
            }
        }
    };
}

feature_class! { PyFeaturePerimeter, "FeaturePerimeter", "perimeter";
    required {
        label: String,
        perimeter_label: String,
        description: String,
    }
    optional {
    }
}

feature_class! { PyFeatureStateDistribution, "FeatureStateDistribution", "state_distribution";
    required {
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
    }
    optional {}
}

feature_class! { PyFeatureZone, "FeatureZone", "zone";
    required {
        label: String,
        perimeter_label: String,
        result_column: String,
        coordinate_labels: Vec<String>,
        description: String,
    }
    optional {
    }
}

feature_class! { PyFeatureQuadrant, "FeatureQuadrant", "quadrant";
    required {
        label: String,
        quadrant_label: String,
        coordinate_labels: Vec<String>,
        description: String,
    }
    optional {
        occupancy_grid: Option<Vec<Vec<f64>>> = None,
    }
}

feature_class! { PyFeatureHeuristic, "FeatureHeuristic", "heuristic";
    required {
        label: String,
        result_column: String,
        coordinate_labels: Vec<String>,
        description: String,
    }
    optional {
        perimeter_label: Option<String> = None,
    }
}

feature_class! { PyFeatureHeuristicCloud, "FeatureHeuristicCloud", "heuristic_cloud";
    required {
        label: String,
        result_column: String,
        coordinate_labels: Vec<String>,
        description: String,
    }
    optional {
        perimeter_label: Option<String> = None,
    }
}

feature_class! { PyFeatureHeuristicDecomposition, "FeatureHeuristicDecomposition", "heuristic_decomposition";
    required {
        label: String,
        heuristic_name: String,
        result_column: String,
        coordinate_labels: Vec<String>,
        components: Vec<PyHeuristicDecompositionComponent>,
        description: String,
    }
    optional {
        perimeter_label: Option<String> = None,
        show_final_result: bool = false,
    }
}

feature_class! { PyFeatureAngle, "FeatureAngle", "angle";
    required {
        label: String,
        from_coordinate_label: String,
        to_coordinate_label: String,
        description: String,
    }
    optional {
    }
}

feature_class! { PyFeatureTrajectory, "FeatureTrajectory", "trajectory";
    required {
        label: String,
        coordinate_labels: Vec<String>,
        description: String,
    }
    optional {
    }
}

feature_class! { PyFeatureScaleDerivation, "FeatureScaleDerivation", "scale_derivation";
    required {
        label: String,
        description: String,
    }
    optional {
    }
}

feature_class! { PyFeatureRayCast, "FeatureRayCast", "ray_cast";
    required {
        label: String,
        origin_label: String,
        angle_column: String,
        length_meters: f64,
        cone_width_deg: f64,
        description: String,
    }
    optional {
    }
}

feature_class! { PyFeatureDetectionFields, "FeatureDetectionFields", "detection_fields";
    required {
        label: String,
        nose_label: String,
        head_label: String,
        cone_half_angle_deg: f64,
        nose_reach_m: f64,
        contact_distance_m: f64,
        description: String,
    }
    optional {
        whisker_apex_fraction: Option<f64> = None,
        whisker_half_angle_deg: Option<f64> = None,
        whisker_length_m: Option<f64> = None,
        object_labels: Vec<String> = (Vec::new()),
        nose_near_columns: Vec<String> = (Vec::new()),
        nose_gate_columns: Vec<String> = (Vec::new()),
        whisker_hit_columns: Vec<String> = (Vec::new()),
    }
}

feature_class! { PyFeatureTimeSeries, "FeatureTimeSeries", "time_series";
    required {
        label: String,
        value_column: String,
        unit: String,
        description: String,
    }
    optional {
    }
}

feature_class! { PyFeatureProximity, "FeatureProximity", "proximity";
    required {
        label: String,
        perimeter_label: String,
        buffer_meters: f64,
        result_column: String,
        coordinate_labels: Vec<String>,
        description: String,
    }
    optional {
    }
}

feature_class! { PyFeatureNortDiscrimination, "FeatureNortDiscrimination", "object_discrimination";
    required {
        label: String,
        object_labels: Vec<String>,
        exploration_times: Vec<f64>,
        coordinate_labels: Vec<String>,
        description: String,
    }
    optional {
        protocol: Option<String> = None,
        discrimination_index: Option<f64> = None,
        recognition_index: Option<f64> = None,
        qualia_channels: Option<Vec<PyQualiaChannelSpec>> = None,
        alternation_count: Option<u32> = None,
        merged_channels: Option<HashMap<String, Vec<f64>>> = None,
    }
}

feature_class! { PyFeatureArmSequence, "FeatureArmSequence", "arm_sequence";
    required {
        label: String,
        arm_sequence: Vec<String>,
        coordinate_labels: Vec<String>,
        description: String,
    }
    optional {
        alternation_pct: Option<f64> = None,
    }
}

feature_class! { PyFeatureYMazeArmSections, "FeatureYMazeArmSections", "ymaze_arm_sections";
    required {
        label: String,
        perimeter_label: String,
        proximal_threshold: f64,
        coordinate_labels: Vec<String>,
        description: String,
    }
    optional {
        threshold_lines: Vec<(String, (f64, f64), (f64, f64))> = (Vec::new()),
    }
}

feature_class! { PyFeatureCenterPeriphery, "FeatureCenterPeriphery", "center_periphery";
    required {
        label: String,
        center_perimeter_label: String,
        arena_perimeter_label: String,
        coordinate_labels: Vec<String>,
        center_time_pct: f64,
        periphery_time_pct: f64,
        description: String,
    }
    optional {
    }
}

// ---------------------------------------------------------------------------
// Conversion: Rust enum -> Python object (heterogenous, returned as PyObject)
// ---------------------------------------------------------------------------

pub fn feature_to_py(py: Python<'_>, f: &FeatureInspection) -> PyResult<Py<pyo3::PyAny>> {
    macro_rules! into_py {
        ($val:expr) => {
            $val.into_pyobject(py)?.unbind().into()
        };
    }
    let common = Common::from_rust(f);
    Ok(match f {
        FeatureInspection::StateDistribution {
            label,
            category_labels,
            frame_counts,
            seconds,
            percentages,
            denominator_frames,
            denominator_seconds,
            source_columns,
            basis,
            style,
            description,
            ..
        } => into_py!(PyFeatureStateDistribution {
            label: label.clone(),
            category_labels: category_labels.clone(),
            frame_counts: frame_counts.clone(),
            seconds: seconds.clone(),
            percentages: percentages.clone(),
            denominator_frames: *denominator_frames,
            denominator_seconds: *denominator_seconds,
            source_columns: source_columns.clone(),
            basis: basis.clone(),
            style: style.clone(),
            description: description.clone(),
            common,
        }),
        FeatureInspection::Perimeter {
            label,
            perimeter_label,
            description,
            ..
        } => into_py!(PyFeaturePerimeter {
            label: label.clone(),
            perimeter_label: perimeter_label.clone(),
            description: description.clone(),
            common,
        }),
        FeatureInspection::Zone {
            label,
            perimeter_label,
            result_column,
            coordinate_labels,
            description,
            ..
        } => into_py!(PyFeatureZone {
            label: label.clone(),
            perimeter_label: perimeter_label.clone(),
            result_column: result_column.clone(),
            coordinate_labels: coordinate_labels.clone(),
            description: description.clone(),
            common,
        }),
        FeatureInspection::Quadrant {
            label,
            quadrant_label,
            coordinate_labels,
            description,
            occupancy_grid,
            ..
        } => into_py!(PyFeatureQuadrant {
            label: label.clone(),
            quadrant_label: quadrant_label.clone(),
            coordinate_labels: coordinate_labels.clone(),
            description: description.clone(),
            occupancy_grid: occupancy_grid.clone(),
            common,
        }),
        FeatureInspection::Heuristic {
            label,
            result_column,
            coordinate_labels,
            perimeter_label,
            description,
            ..
        } => into_py!(PyFeatureHeuristic {
            label: label.clone(),
            result_column: result_column.clone(),
            coordinate_labels: coordinate_labels.clone(),
            description: description.clone(),
            perimeter_label: perimeter_label.clone(),
            common,
        }),
        FeatureInspection::HeuristicCloud {
            label,
            result_column,
            coordinate_labels,
            perimeter_label,
            description,
            ..
        } => into_py!(PyFeatureHeuristicCloud {
            label: label.clone(),
            result_column: result_column.clone(),
            coordinate_labels: coordinate_labels.clone(),
            description: description.clone(),
            perimeter_label: perimeter_label.clone(),
            common,
        }),
        FeatureInspection::HeuristicDecomposition {
            label,
            heuristic_name,
            result_column,
            coordinate_labels,
            perimeter_label,
            components,
            show_final_result,
            description,
            ..
        } => into_py!(PyFeatureHeuristicDecomposition {
            label: label.clone(),
            heuristic_name: heuristic_name.clone(),
            result_column: result_column.clone(),
            coordinate_labels: coordinate_labels.clone(),
            components: components
                .iter()
                .map(|c| PyHeuristicDecompositionComponent::from_rust(py, c))
                .collect::<PyResult<Vec<_>>>()?,
            description: description.clone(),
            perimeter_label: perimeter_label.clone(),
            show_final_result: *show_final_result,
            common,
        }),
        FeatureInspection::Angle {
            label,
            from_coordinate_label,
            to_coordinate_label,
            description,
            ..
        } => into_py!(PyFeatureAngle {
            label: label.clone(),
            from_coordinate_label: from_coordinate_label.clone(),
            to_coordinate_label: to_coordinate_label.clone(),
            description: description.clone(),
            common,
        }),
        FeatureInspection::Trajectory {
            label,
            coordinate_labels,
            description,
            ..
        } => into_py!(PyFeatureTrajectory {
            label: label.clone(),
            coordinate_labels: coordinate_labels.clone(),
            description: description.clone(),
            common,
        }),
        FeatureInspection::ScaleDerivation {
            label, description, ..
        } => into_py!(PyFeatureScaleDerivation {
            label: label.clone(),
            description: description.clone(),
            common,
        }),
        FeatureInspection::TimeSeries {
            label,
            value_column,
            unit,
            description,
            ..
        } => into_py!(PyFeatureTimeSeries {
            label: label.clone(),
            value_column: value_column.clone(),
            unit: unit.clone(),
            description: description.clone(),
            common,
        }),
        FeatureInspection::RayCast {
            label,
            origin_label,
            angle_column,
            length_meters,
            cone_width_deg,
            description,
            ..
        } => into_py!(PyFeatureRayCast {
            label: label.clone(),
            origin_label: origin_label.clone(),
            angle_column: angle_column.clone(),
            length_meters: *length_meters,
            cone_width_deg: *cone_width_deg,
            description: description.clone(),
            common,
        }),
        FeatureInspection::DetectionFields {
            label,
            nose_label,
            head_label,
            cone_half_angle_deg,
            nose_reach_m,
            contact_distance_m,
            whisker_apex_fraction,
            whisker_half_angle_deg,
            whisker_length_m,
            object_labels,
            nose_near_columns,
            nose_gate_columns,
            whisker_hit_columns,
            description,
            ..
        } => into_py!(PyFeatureDetectionFields {
            label: label.clone(),
            nose_label: nose_label.clone(),
            head_label: head_label.clone(),
            cone_half_angle_deg: *cone_half_angle_deg,
            nose_reach_m: *nose_reach_m,
            contact_distance_m: *contact_distance_m,
            whisker_apex_fraction: *whisker_apex_fraction,
            whisker_half_angle_deg: *whisker_half_angle_deg,
            whisker_length_m: *whisker_length_m,
            object_labels: object_labels.clone(),
            nose_near_columns: nose_near_columns.clone(),
            nose_gate_columns: nose_gate_columns.clone(),
            whisker_hit_columns: whisker_hit_columns.clone(),
            description: description.clone(),
            common,
        }),
        FeatureInspection::Proximity {
            label,
            perimeter_label,
            buffer_meters,
            result_column,
            coordinate_labels,
            description,
            ..
        } => into_py!(PyFeatureProximity {
            label: label.clone(),
            perimeter_label: perimeter_label.clone(),
            buffer_meters: *buffer_meters,
            result_column: result_column.clone(),
            coordinate_labels: coordinate_labels.clone(),
            description: description.clone(),
            common,
        }),
        FeatureInspection::NortDiscrimination {
            label,
            protocol,
            object_labels,
            exploration_times,
            discrimination_index,
            recognition_index,
            qualia_channels,
            alternation_count,
            merged_channels,
            coordinate_labels,
            description,
            ..
        } => into_py!(PyFeatureNortDiscrimination {
            label: label.clone(),
            object_labels: object_labels.clone(),
            exploration_times: exploration_times.clone(),
            coordinate_labels: coordinate_labels.clone(),
            description: description.clone(),
            protocol: protocol.clone(),
            discrimination_index: *discrimination_index,
            recognition_index: *recognition_index,
            qualia_channels: qualia_channels
                .as_ref()
                .map(|qs| qs.iter().map(PyQualiaChannelSpec::from_rust).collect()),
            alternation_count: *alternation_count,
            merged_channels: merged_channels.clone(),
            common,
        }),
        FeatureInspection::ArmSequence {
            label,
            arm_sequence,
            coordinate_labels,
            alternation_pct,
            description,
            ..
        } => into_py!(PyFeatureArmSequence {
            label: label.clone(),
            arm_sequence: arm_sequence.clone(),
            coordinate_labels: coordinate_labels.clone(),
            description: description.clone(),
            alternation_pct: *alternation_pct,
            common,
        }),
        FeatureInspection::YMazeArmSections {
            label,
            perimeter_label,
            proximal_threshold,
            threshold_lines,
            coordinate_labels,
            description,
            ..
        } => into_py!(PyFeatureYMazeArmSections {
            label: label.clone(),
            perimeter_label: perimeter_label.clone(),
            proximal_threshold: *proximal_threshold,
            coordinate_labels: coordinate_labels.clone(),
            description: description.clone(),
            threshold_lines: threshold_lines
                .iter()
                .map(|l| (l.arm.clone(), l.p1, l.p2))
                .collect(),
            common,
        }),
        FeatureInspection::CenterPeriphery {
            label,
            center_perimeter_label,
            arena_perimeter_label,
            coordinate_labels,
            center_time_pct,
            periphery_time_pct,
            description,
            ..
        } => into_py!(PyFeatureCenterPeriphery {
            label: label.clone(),
            center_perimeter_label: center_perimeter_label.clone(),
            arena_perimeter_label: arena_perimeter_label.clone(),
            coordinate_labels: coordinate_labels.clone(),
            center_time_pct: *center_time_pct,
            periphery_time_pct: *periphery_time_pct,
            description: description.clone(),
            common,
        }),
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHeuristicDecompositionComponent>()?;
    m.add_class::<PyQualiaChannelSpec>()?;
    m.add_class::<PyFeaturePerimeter>()?;
    m.add_class::<PyFeatureZone>()?;
    m.add_class::<PyFeatureQuadrant>()?;
    m.add_class::<PyFeatureHeuristic>()?;
    m.add_class::<PyFeatureHeuristicCloud>()?;
    m.add_class::<PyFeatureHeuristicDecomposition>()?;
    m.add_class::<PyFeatureStateDistribution>()?;
    m.add_class::<PyFeatureAngle>()?;
    m.add_class::<PyFeatureTrajectory>()?;
    m.add_class::<PyFeatureScaleDerivation>()?;
    m.add_class::<PyFeatureRayCast>()?;
    m.add_class::<PyFeatureDetectionFields>()?;
    m.add_class::<PyFeatureTimeSeries>()?;
    m.add_class::<PyFeatureProximity>()?;
    m.add_class::<PyFeatureNortDiscrimination>()?;
    m.add_class::<PyFeatureArmSequence>()?;
    m.add_class::<PyFeatureYMazeArmSections>()?;
    m.add_class::<PyFeatureCenterPeriphery>()?;
    Ok(())
}
