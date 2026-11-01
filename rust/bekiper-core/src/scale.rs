use serde::{Deserialize, Serialize};

use crate::error::{BekiperError, Result};
use crate::inspection::PerimeterSpec;
use crate::types::MetersPerPixel;

/// How to obtain the pixel-to-meter conversion factor.
///
/// In TOML config this is specified as `meters_per_pixel`:
///
/// ```toml
/// # Manual — user measured or knows the value:
/// meters_per_pixel = 0.001914
///
/// # Derived from the detected arena side (NORT / OFT):
/// meters_per_pixel = { method = "arena_side", meters = 0.40 }
///
/// # Derived from Y-maze arm length:
/// meters_per_pixel = { method = "arm_length", meters = 0.35 }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScaleConfig {
    /// A manually specified meters-per-pixel value.
    Manual(f64),
    /// Derive from detected perimeters using a known physical dimension.
    Derived(ScaleMethod),
}

/// Method for deriving meters-per-pixel from detected perimeters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
pub enum ScaleMethod {
    /// Measure the shorter side of the detected arena perimeter (label "arena")
    /// and divide the known physical side length by the pixel measurement.
    ///
    /// Suitable for square arenas (NORT 40×40 cm, OFT 40×40 cm).
    #[serde(rename = "arena_side")]
    ArenaSide {
        /// Physical side length in meters (e.g. 0.40 for 40 cm).
        meters: f64,
        /// The arena is square, so its detected width and height each span
        /// `meters`: non-square pixels get separate horizontal and vertical
        /// scales. Default false keeps the short side (rectangular arenas).
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        square: bool,
    },
    /// Measure the longest span of each arm polygon in a RadialMaze perimeter
    /// and divide the known physical arm length by the mean pixel measurement.
    ///
    /// Suitable for Y-maze (arm length 35 cm).
    #[serde(rename = "arm_length")]
    ArmLength {
        /// Physical arm length in meters (e.g. 0.35 for 35 cm).
        meters: f64,
    },
}

/// Metadata about how meters-per-pixel was derived, for inspection figures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleDerivationInfo {
    /// Derivation method name: "manual", "arena_side", or "arm_length".
    pub method: String,
    /// The known physical dimension in meters (e.g. 0.40 for a 40 cm arena side).
    /// `None` for manual.
    pub physical_meters: Option<f64>,
    /// The measured pixel dimension used to compute the scale.
    /// `None` for manual.
    pub pixel_measurement: Option<f64>,
    /// Per-arm pixel spans (only for arm_length method).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arm_pixel_spans: Vec<f64>,
    /// The label of the perimeter used for derivation. `None` for manual.
    pub reference_label: Option<String>,
    /// The two pixel-space endpoints of the measured span, when applicable
    /// (e.g. annotation_diagonal). `None` for methods where the span is
    /// implicit in the perimeter geometry (arena_side, arm_length, manual).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoints: Option<[[f64; 2]; 2]>,
    /// The resulting meters-per-pixel value (horizontal axis when
    /// `meters_per_pixel_y` is set).
    pub meters_per_pixel: f64,
    /// Vertical meters-per-pixel when the source pixels are not square
    /// (e.g. a detected 40 cm box that is 204 x 210 px). `None` = isotropic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meters_per_pixel_y: Option<f64>,
}

impl ScaleDerivationInfo {
    /// Vertical metres/pixel; equals `meters_per_pixel` for square pixels.
    pub fn meters_per_pixel_y(&self) -> f64 {
        self.meters_per_pixel_y.unwrap_or(self.meters_per_pixel)
    }

    /// Display aspect correction `mpp_y / mpp_x` (1 for square pixels).
    pub fn y_aspect(&self) -> f64 {
        self.meters_per_pixel_y() / self.meters_per_pixel
    }
}

/// Trait for deriving the pixel-to-meter scale factor from detected
/// perimeters whose real-world dimensions are known.
pub trait ScaleDerivation {
    /// Derive meters-per-pixel from pixel-space perimeters.
    fn derive_scale(&self, perimeters: &[PerimeterSpec]) -> Result<MetersPerPixel>;
}

impl ScaleConfig {
    /// Resolve to a concrete meters-per-pixel value.
    ///
    /// For `Manual`, returns the value directly.
    /// For `Derived`, computes it from the detected perimeters.
    pub fn resolve(&self, perimeters: &[PerimeterSpec]) -> Result<MetersPerPixel> {
        match self {
            ScaleConfig::Manual(mpp) => Ok(MetersPerPixel(*mpp)),
            ScaleConfig::Derived(method) => method.derive_scale(perimeters),
        }
    }

    /// Resolve to a concrete meters-per-pixel value with derivation metadata
    /// for inspection figures.
    pub fn resolve_with_info(
        &self,
        perimeters: &[PerimeterSpec],
    ) -> Result<(MetersPerPixel, ScaleDerivationInfo)> {
        match self {
            ScaleConfig::Manual(mpp) => {
                let info = ScaleDerivationInfo {
                    meters_per_pixel_y: None,
                    method: "manual".into(),
                    physical_meters: None,
                    pixel_measurement: None,
                    arm_pixel_spans: Vec::new(),
                    reference_label: None,
                    endpoints: None,
                    meters_per_pixel: *mpp,
                };
                Ok((MetersPerPixel(*mpp), info))
            }
            ScaleConfig::Derived(method) => method.derive_scale_with_info(perimeters),
        }
    }
}

// ---------------------------------------------------------------------------
// ScaleDerivation implementations
// ---------------------------------------------------------------------------

impl ScaleDerivation for ScaleMethod {
    fn derive_scale(&self, perimeters: &[PerimeterSpec]) -> Result<MetersPerPixel> {
        match self {
            ScaleMethod::ArenaSide { meters, square } => ArenaSideDeriver {
                meters: *meters,
                square: *square,
            }
            .derive_scale(perimeters),
            ScaleMethod::ArmLength { meters } => {
                ArmLengthDeriver { meters: *meters }.derive_scale(perimeters)
            }
        }
    }
}

impl ScaleMethod {
    fn derive_scale_with_info(
        &self,
        perimeters: &[PerimeterSpec],
    ) -> Result<(MetersPerPixel, ScaleDerivationInfo)> {
        match self {
            ScaleMethod::ArenaSide { meters, square } => ArenaSideDeriver {
                meters: *meters,
                square: *square,
            }
            .derive_with_info(perimeters),
            ScaleMethod::ArmLength { meters } => {
                ArmLengthDeriver { meters: *meters }.derive_with_info(perimeters)
            }
        }
    }
}

/// Derives scale from the shorter side of a detected arena perimeter.
struct ArenaSideDeriver {
    meters: f64,
    square: bool,
}

impl ArenaSideDeriver {
    /// Find the arena perimeter and return ((width, height) in pixels, label).
    fn find_arena_extent(&self, perimeters: &[PerimeterSpec]) -> Result<((f64, f64), String)> {
        let (extent, label) = perimeters
            .iter()
            .find_map(|p| match p {
                PerimeterSpec::Rectangle {
                    label,
                    width,
                    height,
                    ..
                } if label == "arena" => Some(((*width, *height), label.clone())),
                PerimeterSpec::Polygon { label, vertices } if label == "arena" => {
                    let xs: Vec<f64> = vertices.iter().map(|(x, _)| *x).collect();
                    let ys: Vec<f64> = vertices.iter().map(|(_, y)| *y).collect();
                    let w = xs.iter().cloned().reduce(f64::max)?
                        - xs.iter().cloned().reduce(f64::min)?;
                    let h = ys.iter().cloned().reduce(f64::max)?
                        - ys.iter().cloned().reduce(f64::min)?;
                    Some(((w, h), label.clone()))
                }
                PerimeterSpec::Circle { label, radius, .. } if label == "arena" => {
                    Some(((2.0 * radius, 2.0 * radius), label.clone()))
                }
                _ => None,
            })
            .ok_or_else(|| {
                BekiperError::Config(
                    "arena_side scale method requires a perimeter labeled 'arena'".into(),
                )
            })?;

        if extent.0.min(extent.1) <= 0.0 {
            return Err(BekiperError::Config(
                "arena perimeter has zero or negative pixel extent".into(),
            ));
        }

        Ok((extent, label))
    }

    /// (horizontal, vertical) metres/pixel and the pixel span used for x.
    /// A square arena spans `meters` on both axes; otherwise the short side
    /// sets one isotropic scale.
    fn axes(&self, (width, height): (f64, f64)) -> (f64, Option<f64>, f64) {
        if self.square && width != height {
            (self.meters / width, Some(self.meters / height), width)
        } else {
            let side = width.min(height);
            (self.meters / side, None, side)
        }
    }

    fn derive_with_info(
        &self,
        perimeters: &[PerimeterSpec],
    ) -> Result<(MetersPerPixel, ScaleDerivationInfo)> {
        let (extent, label) = self.find_arena_extent(perimeters)?;
        let (mpp, meters_per_pixel_y, side_pixels) = self.axes(extent);
        let info = ScaleDerivationInfo {
            meters_per_pixel_y,
            method: "arena_side".into(),
            physical_meters: Some(self.meters),
            pixel_measurement: Some(side_pixels),
            arm_pixel_spans: Vec::new(),
            reference_label: Some(label),
            endpoints: None,
            meters_per_pixel: mpp,
        };
        Ok((MetersPerPixel(mpp), info))
    }
}

impl ScaleDerivation for ArenaSideDeriver {
    fn derive_scale(&self, perimeters: &[PerimeterSpec]) -> Result<MetersPerPixel> {
        let (extent, _) = self.find_arena_extent(perimeters)?;
        Ok(MetersPerPixel(self.axes(extent).0))
    }
}

/// Derives scale from Y-maze / radial arm polygon spans.
struct ArmLengthDeriver {
    meters: f64,
}

impl ArmLengthDeriver {
    /// Find the radial maze and return (arm_spans, mean_span, label).
    fn find_arm_spans(&self, perimeters: &[PerimeterSpec]) -> Result<(Vec<f64>, f64, String)> {
        let (arms, label) = perimeters
            .iter()
            .find_map(|p| match p {
                PerimeterSpec::RadialMaze { arms, label, .. } => Some((arms, label.clone())),
                _ => None,
            })
            .ok_or_else(|| {
                BekiperError::Config(
                    "arm_length scale method requires a RadialMaze perimeter".into(),
                )
            })?;

        if arms.is_empty() {
            return Err(BekiperError::Config("RadialMaze has no arms".into()));
        }

        let arm_lengths: Vec<f64> = arms.iter().map(|arm| arm_length_span(arm)).collect();
        let mean_length = arm_lengths.iter().sum::<f64>() / arm_lengths.len() as f64;

        if mean_length <= 0.0 {
            return Err(BekiperError::Config(
                "RadialMaze arms have zero pixel extent".into(),
            ));
        }

        Ok((arm_lengths, mean_length, label))
    }

    fn derive_with_info(
        &self,
        perimeters: &[PerimeterSpec],
    ) -> Result<(MetersPerPixel, ScaleDerivationInfo)> {
        let (arm_spans, mean_length, label) = self.find_arm_spans(perimeters)?;
        let mpp = self.meters / mean_length;
        let info = ScaleDerivationInfo {
            meters_per_pixel_y: None,
            method: "arm_length".into(),
            physical_meters: Some(self.meters),
            pixel_measurement: Some(mean_length),
            arm_pixel_spans: arm_spans,
            reference_label: Some(label),
            endpoints: None,
            meters_per_pixel: mpp,
        };
        Ok((MetersPerPixel(mpp), info))
    }
}

impl ScaleDerivation for ArmLengthDeriver {
    fn derive_scale(&self, perimeters: &[PerimeterSpec]) -> Result<MetersPerPixel> {
        let (_, mean_length, _) = self.find_arm_spans(perimeters)?;
        Ok(MetersPerPixel(self.meters / mean_length))
    }
}

/// Arm length from an arm polygon.
///
/// Y-maze annotations are quadrilateral arm areas. The physical arm length is
/// the longer distance between opposite edge midpoints, not the polygon
/// diagonal. Two-point legacy arm lines fall back to point-to-point distance.
fn arm_length_span(vertices: &[(f64, f64)]) -> f64 {
    match vertices.len() {
        0 | 1 => 0.0,
        2 => distance(vertices[0], vertices[1]),
        4 => {
            let edge_midpoint =
                |a: (f64, f64), b: (f64, f64)| ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
            let span_a = distance(
                edge_midpoint(vertices[0], vertices[1]),
                edge_midpoint(vertices[2], vertices[3]),
            );
            let span_b = distance(
                edge_midpoint(vertices[1], vertices[2]),
                edge_midpoint(vertices[3], vertices[0]),
            );
            span_a.max(span_b)
        }
        _ => polygon_max_span(vertices),
    }
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt()
}

/// Maximum distance between any two vertices of a polygon (diameter).
fn polygon_max_span(vertices: &[(f64, f64)]) -> f64 {
    let mut max_dist = 0.0f64;
    for (i, &(x1, y1)) in vertices.iter().enumerate() {
        for &(x2, y2) in &vertices[i + 1..] {
            let d = distance((x1, y1), (x2, y2));
            max_dist = max_dist.max(d);
        }
    }
    max_dist
}

#[cfg(test)]
#[path = "scale/tests.rs"]
mod tests;

#[cfg(test)]
mod per_axis_tests {
    use super::ScaleDerivationInfo;
    use crate::inspection::PerimeterSpec;

    #[test]
    fn vertical_scale_is_optional_and_square_pixels_serialize_unchanged() {
        let mut info = ScaleDerivationInfo {
            method: "arena_box_detected".into(),
            physical_meters: Some(0.40),
            pixel_measurement: Some(204.0),
            arm_pixel_spans: Vec::new(),
            reference_label: Some("arena".into()),
            endpoints: None,
            meters_per_pixel: 0.40 / 204.0,
            meters_per_pixel_y: None,
        };
        let square = toml::to_string(&info).unwrap();
        assert!(!square.contains("meters_per_pixel_y"));
        assert_eq!(info.y_aspect(), 1.0);
        info.meters_per_pixel_y = Some(0.40 / 210.0);
        assert!((info.y_aspect() - 204.0 / 210.0).abs() < 1e-12);
        let restored: ScaleDerivationInfo =
            toml::from_str(&toml::to_string(&info).unwrap()).unwrap();
        assert_eq!(restored.meters_per_pixel_y, info.meters_per_pixel_y);
        // Old files without the field still load as square pixels.
        let old: ScaleDerivationInfo = toml::from_str(&square).unwrap();
        assert_eq!(old.meters_per_pixel_y(), old.meters_per_pixel);
    }

    #[test]
    fn perimeters_scale_each_axis_separately() {
        let rect = PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 112.0,
            center_y: 112.0,
            width: 204.0,
            height: 210.0,
        };
        let scaled = rect.scale_xy(0.40 / 204.0, 0.40 / 210.0);
        assert!(
            matches!(
                scaled,
                PerimeterSpec::Rectangle { width, height, center_y, .. }
                    if (width - 0.40).abs() < 1e-12
                        && (height - 0.40).abs() < 1e-12
                        && (center_y - 112.0 * 0.40 / 210.0).abs() < 1e-12
            ),
            "{scaled:?}"
        );
    }
}
