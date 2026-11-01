use serde::{Deserialize, Deserializer, Serialize};

use crate::types::Label;

// ---------------------------------------------------------------------------
// Coordinate-space enforcement via trait
// ---------------------------------------------------------------------------

/// Trait for inspection geometry that lives in metric space and must be
/// converted to video-pixel space before rendering on a frame.
///
/// Every type that carries spatial coordinates drawn on inspection figures
/// **must** implement this trait. The manifest builder calls
/// [`ToVideoPixels::to_video_pixels`] on all geometry before serialization, which means:
///
/// 1. Adding a new geometry type that skips this trait is a compile error
///    at the call site in `summary.rs` (the generic helper requires the bound).
/// 2. Python receives pixel-space geometry and never has to convert — the
///    inconsistent `as_pixels` / `meters_per_pixel` / `frame_shape` dance
///    is eliminated.
///
/// The conversion is: `pixel_value = metric_value / meters_per_pixel`.
/// Any additional frame-scaling (when the displayed frame differs from
/// `VideoMetadata.resolution`) is a uniform ratio applied by Python at
/// render time — it does not depend on the geometry type and cannot be
/// forgotten per-element.
pub trait ToVideoPixels {
    /// Return a copy with all spatial coordinates converted from metric
    /// space to video-pixel space (matching `VideoMetadata.resolution`).
    fn to_video_pixels(&self, meters_per_pixel: f64) -> Self;
}

// ---------------------------------------------------------------------------
// PerimeterSpec
// ---------------------------------------------------------------------------

/// Geometry specification for a perimeter, serializable to TOML for Python inspection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape")]
pub enum PerimeterSpec {
    #[serde(rename = "circle")]
    Circle {
        label: Label,
        center_x: f64,
        center_y: f64,
        radius: f64,
    },
    #[serde(rename = "rectangle")]
    Rectangle {
        label: Label,
        center_x: f64,
        center_y: f64,
        width: f64,
        height: f64,
    },
    #[serde(rename = "polygon")]
    Polygon {
        label: Label,
        vertices: Vec<(f64, f64)>,
    },
    #[serde(rename = "radial_maze")]
    RadialMaze {
        label: Label,
        center_vertices: Vec<(f64, f64)>,
        arms: Vec<Vec<(f64, f64)>>,
    },
    #[serde(rename = "triangle")]
    Triangle {
        label: Label,
        vertices: [(f64, f64); 3],
    },
}

impl PerimeterSpec {
    /// Reflect the Y axis while retaining the exact native shape and dimensions.
    pub fn reflect_y(&self, height: f64) -> Self {
        let mut reflected = self.clone();
        let reflect = |vertices: &mut [(f64, f64)]| {
            for (_, y) in vertices {
                *y = height - *y;
            }
        };
        match &mut reflected {
            Self::Circle { center_y, .. } | Self::Rectangle { center_y, .. } => {
                *center_y = height - *center_y
            }
            Self::Polygon { vertices, .. } => reflect(vertices),
            Self::Triangle { vertices, .. } => reflect(vertices),
            Self::RadialMaze {
                center_vertices,
                arms,
                ..
            } => {
                reflect(center_vertices);
                for vertices in arms {
                    reflect(vertices);
                }
            }
        }
        reflected
    }
    /// Convert to four polygon corners or an axis-aligned bounding rectangle.
    pub fn convert_polygon_rectangle(&self) -> Option<Self> {
        match self {
            Self::Rectangle {
                label,
                center_x: x,
                center_y: y,
                width: w,
                height: h,
            } if [*x, *y, *w, *h].iter().all(|v| v.is_finite()) && *w > 0.0 && *h > 0.0 => {
                let vertices = vec![
                    (x - w / 2.0, y - h / 2.0),
                    (x + w / 2.0, y - h / 2.0),
                    (x + w / 2.0, y + h / 2.0),
                    (x - w / 2.0, y + h / 2.0),
                ];
                vertices
                    .iter()
                    .all(|(x, y)| x.is_finite() && y.is_finite())
                    .then(|| Self::Polygon {
                        label: label.clone(),
                        vertices,
                    })
            }
            Self::Polygon { label, vertices } if vertices.len() >= 3 => {
                if !vertices.iter().all(|(x, y)| x.is_finite() && y.is_finite()) {
                    return None;
                }
                let (mut min_x, mut min_y) = vertices[0];
                let (mut max_x, mut max_y) = vertices[0];
                let mut area = 0.0;
                for (i, &(x, y)) in vertices.iter().enumerate() {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                    let (nx, ny) = vertices[(i + 1) % vertices.len()];
                    area += x * ny - nx * y;
                }
                let width = max_x - min_x;
                let height = max_y - min_y;
                (area.is_finite()
                    && area != 0.0
                    && width.is_finite()
                    && height.is_finite()
                    && width > 0.0
                    && height > 0.0)
                    .then(|| Self::Rectangle {
                        label: label.clone(),
                        center_x: min_x + width / 2.0,
                        center_y: min_y + height / 2.0,
                        width,
                        height,
                    })
            }
            _ => None,
        }
    }

    pub fn label(&self) -> &Label {
        match self {
            PerimeterSpec::Circle { label, .. }
            | PerimeterSpec::Rectangle { label, .. }
            | PerimeterSpec::Polygon { label, .. }
            | PerimeterSpec::RadialMaze { label, .. }
            | PerimeterSpec::Triangle { label, .. } => label,
        }
    }

    /// Scale all coordinates by the given factor (typically meters_per_pixel).
    pub fn scale(&self, factor: f64) -> Self {
        self.scale_xy(factor, factor)
    }

    /// Scale x and y by separate factors (non-square source pixels).
    pub fn scale_xy(&self, factor_x: f64, factor_y: f64) -> Self {
        let (fx, fy) = (factor_x, factor_y);
        match self {
            PerimeterSpec::Circle {
                label,
                center_x,
                center_y,
                radius,
            } => PerimeterSpec::Circle {
                label: label.clone(),
                center_x: center_x * fx,
                center_y: center_y * fy,
                // ponytail: a circle stays a circle under the geometric-mean
                // factor (~1.5% radial error at 3% anisotropy); emit an
                // ellipse/polygon if anisotropic circular objects matter.
                radius: radius * (fx * fy).sqrt(),
            },
            PerimeterSpec::Rectangle {
                label,
                center_x,
                center_y,
                width,
                height,
            } => PerimeterSpec::Rectangle {
                label: label.clone(),
                center_x: center_x * fx,
                center_y: center_y * fy,
                width: width * fx,
                height: height * fy,
            },
            PerimeterSpec::Polygon { label, vertices } => PerimeterSpec::Polygon {
                label: label.clone(),
                vertices: vertices.iter().map(|(x, y)| (x * fx, y * fy)).collect(),
            },
            PerimeterSpec::RadialMaze {
                label,
                center_vertices,
                arms,
            } => PerimeterSpec::RadialMaze {
                label: label.clone(),
                center_vertices: center_vertices
                    .iter()
                    .map(|(x, y)| (x * fx, y * fy))
                    .collect(),
                arms: arms
                    .iter()
                    .map(|arm| arm.iter().map(|(x, y)| (x * fx, y * fy)).collect())
                    .collect(),
            },
            PerimeterSpec::Triangle { label, vertices } => PerimeterSpec::Triangle {
                label: label.clone(),
                vertices: [
                    (vertices[0].0 * fx, vertices[0].1 * fy),
                    (vertices[1].0 * fx, vertices[1].1 * fy),
                    (vertices[2].0 * fx, vertices[2].1 * fy),
                ],
            },
        }
    }
}

impl ToVideoPixels for PerimeterSpec {
    fn to_video_pixels(&self, meters_per_pixel: f64) -> Self {
        if meters_per_pixel > 0.0 {
            self.scale(1.0 / meters_per_pixel)
        } else {
            self.clone()
        }
    }
}

/// Metadata about a heuristic result column in the evaluation Parquet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeuristicMeta {
    pub name: String,
    pub result_column: String,
    pub true_frames: u64,
    pub total_frames: u64,
    pub seconds: f64,
    /// Pre-computed contiguous True spans as `(start_frame, end_frame)`
    /// pairs where `end_frame` is exclusive. Used by the Python inspector
    /// for timeline rendering without re-reading the parquet.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub truth_spans: Vec<(u64, u64)>,
}

/// Per-sensory-channel observation data for qualia decomposition in NORT figures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualiaChannelSpec {
    /// Heuristic type: `"body_proximity"`, `"olfaction"`, or `"whisker_interaction"`.
    pub heuristic_type: String,
    /// Per-object exploration seconds (ordered by `object_labels`).
    pub per_object_seconds: Vec<f64>,
    /// Per-object bout counts (ordered by `object_labels`).
    pub per_object_bouts: Vec<u32>,
    /// Per-object relative bias: `100 * obj_seconds / channel_total`.
    pub relative_bias: Vec<f64>,
}

/// One boolean gate that participates in a heuristic decomposition figure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeuristicDecompositionComponent {
    /// Short panel label rendered above the component figure.
    pub label: String,
    /// Boolean column that drives this component panel.
    pub result_column: String,
    /// Coordinate labels used for trajectory overlay.
    #[serde(default, deserialize_with = "deserialize_labels")]
    pub coordinate_labels: Vec<String>,
    /// Optional perimeter to draw for spatial context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perimeter_label: Option<String>,
    /// The evaluated proximity distance; renderers never infer it from names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub buffer_meters: Option<f64>,
    /// Explicit origin/direction/target bindings for full-trial ray diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ray_origin_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ray_direction_x_column: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ray_direction_y_column: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ray_target_x_column: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ray_target_y_column: Option<String>,
    /// Optional native scalar diagnostic shown with its explicit unit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_column: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_unit: Option<String>,
    /// Exact metric-space geometry used by the heuristic and buffer gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluated_perimeter: Option<super::PerimeterSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub buffer_perimeter: Option<super::PerimeterSpec>,
}

impl HeuristicDecompositionComponent {
    pub fn with_geometry<S: crate::shape::Expandable>(mut self, shape: &S, label: &str) -> Self {
        use crate::shape::Shape;
        self.evaluated_perimeter = shape.inspection_perimeter(label);
        self.buffer_perimeter = self
            .buffer_meters
            .and_then(|distance| shape.expand(distance).inspection_perimeter(label));
        self
    }
    pub fn with_values(mut self, column: String, unit: &str) -> Self {
        self.value_column = Some(column);
        self.value_unit = Some(unit.to_string());
        self
    }

    pub fn with_buffer(mut self, meters: f64) -> Self {
        self.buffer_meters = Some(meters);
        self
    }

    pub fn with_ray(
        mut self,
        ray: &crate::types::RaySpec,
        target_x: String,
        target_y: String,
    ) -> Self {
        self.ray_origin_label = Some(
            ray.origin
                .x
                .strip_suffix("_x")
                .unwrap_or(&ray.origin.x)
                .to_string(),
        );
        self.ray_direction_x_column = Some(ray.direction.x.clone());
        self.ray_direction_y_column = Some(ray.direction.y.clone());
        self.ray_target_x_column = Some(target_x);
        self.ray_target_y_column = Some(target_y);
        self
    }
}

/// Explicit decomposition metadata for a heuristic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeuristicDecompositionSpec {
    /// User-facing heuristic name (matches `HeuristicSummary.name`).
    pub heuristic_name: String,
    /// Final boolean result column for the heuristic decomposition figure.
    pub result_column: String,
    /// Coordinate labels for the final-result panel and frame picking.
    #[serde(default, deserialize_with = "deserialize_labels")]
    pub coordinate_labels: Vec<String>,
    /// Optional perimeter to draw on the final-result panel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perimeter_label: Option<String>,
    /// Ordered component panels rendered ahead of the final result.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<HeuristicDecompositionComponent>,
    /// Whether the final result differs meaningfully from the last component
    /// and should therefore get its own trailing panel.
    #[serde(default, skip_serializing_if = "is_false")]
    pub show_final_result: bool,
}

pub(super) fn is_false(value: &bool) -> bool {
    !*value
}

/// Auto-generated NxN grid for quadrant analysis. Carries the bounding box and
/// shape so the Python inspector can re-draw the grid on a video frame and
/// color cells by per-cell occupancy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadrantSpec {
    pub label: String,
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
    pub rows: u32,
    pub cols: u32,
}

impl ToVideoPixels for QuadrantSpec {
    fn to_video_pixels(&self, meters_per_pixel: f64) -> Self {
        if meters_per_pixel > 0.0 {
            let s = 1.0 / meters_per_pixel;
            QuadrantSpec {
                label: self.label.clone(),
                x_min: self.x_min * s,
                y_min: self.y_min * s,
                x_max: self.x_max * s,
                y_max: self.y_max * s,
                rows: self.rows,
                cols: self.cols,
            }
        } else {
            self.clone()
        }
    }
}

/// Deserialize a `Vec<String>` that accepts either a single string (wraps in
/// a one-element vec) or a list of strings. This allows backwards-compatible
/// reading of old manifests that used `coordinate_label = "nose"` alongside
/// the new `coordinate_labels = ["nose", "tail_base"]`.
pub(super) fn deserialize_labels<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        Single(String),
        Multiple(Vec<String>),
    }
    match StringOrVec::deserialize(deserializer)? {
        StringOrVec::Single(s) => Ok(vec![s]),
        StringOrVec::Multiple(v) => Ok(v),
    }
}
