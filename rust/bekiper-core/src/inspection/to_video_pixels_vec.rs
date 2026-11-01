use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::scale::ScaleDerivationInfo;
use crate::types::{Label, TaskType};
use crate::video::VideoMetadata;

use super::*;

/// Convert a slice of metric-space geometry to video-pixel space.
/// The trait bound ensures every new geometry type that forgets to
/// implement `ToVideoPixels` causes a compile error here.
pub fn to_video_pixels_vec<T: ToVideoPixels>(items: &[T], meters_per_pixel: f64) -> Vec<T> {
    items
        .iter()
        .map(|item| item.to_video_pixels(meters_per_pixel))
        .collect()
}

impl InspectionManifest {
    /// Vertical/horizontal metres-per-pixel ratio of the source (1 for square
    /// pixels). Metric geometry divided by `settings.meters_per_pixel` lands in
    /// square-pixel space; displays divide their vertical scale by this.
    pub fn y_aspect(&self) -> f64 {
        self.scale_derivation
            .as_ref()
            .map_or(1.0, ScaleDerivationInfo::y_aspect)
    }

    /// Source height in metres, for Y inversion.
    pub fn metric_source_height(&self) -> f64 {
        self.video.resolution.1 as f64 * self.settings.meters_per_pixel * self.y_aspect()
    }

    /// Bindings that need an inverse Y transform when drawing on source images.
    /// Positions reflect about source height; vector components/headings negate.
    pub fn overlay_y_bindings(&self) -> (Vec<String>, Vec<String>) {
        let mut positions = self
            .coordinate_columns
            .values()
            .map(|c| c.y.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let mut vectors = std::collections::BTreeSet::new();
        for feature in &self.feature_inspections {
            match feature {
                FeatureInspection::HeuristicDecomposition { components, .. } => {
                    for component in components {
                        positions.extend(component.ray_target_y_column.clone());
                        vectors.extend(component.ray_direction_y_column.clone());
                    }
                }
                FeatureInspection::RayCast { angle_column, .. } => {
                    vectors.insert(angle_column.clone());
                }
                _ => {}
            }
        }
        (
            positions.into_iter().collect(),
            vectors.into_iter().collect(),
        )
    }

    /// A display-only inverse transform. Recorded analysis geometry, masks and
    /// scalars stay in the persisted manifest/table; both renderers use this copy.
    pub fn source_overlay_geometry(&self) -> Self {
        let mut display = self.clone();
        if !self.settings.invert_y {
            return display;
        }
        let pixel_height = self.video.resolution.1 as f64;
        let metric_height = self.metric_source_height();
        display.perimeters = self
            .perimeters
            .iter()
            .map(|p| p.reflect_y(metric_height))
            .collect();
        display.pixel_perimeters = self
            .pixel_perimeters
            .iter()
            .map(|p| p.reflect_y(pixel_height))
            .collect();
        let reflect_quadrants = |quadrants: &mut Vec<QuadrantSpec>, height: f64| {
            for q in quadrants {
                (q.y_min, q.y_max) = (height - q.y_max, height - q.y_min);
            }
        };
        reflect_quadrants(&mut display.quadrants, metric_height);
        reflect_quadrants(&mut display.pixel_quadrants, pixel_height);
        for feature in &mut display.feature_inspections {
            match feature {
                FeatureInspection::HeuristicDecomposition { components, .. } => {
                    for component in components {
                        component.evaluated_perimeter = component
                            .evaluated_perimeter
                            .as_ref()
                            .map(|p| p.reflect_y(metric_height));
                        component.buffer_perimeter = component
                            .buffer_perimeter
                            .as_ref()
                            .map(|p| p.reflect_y(metric_height));
                    }
                }
                FeatureInspection::Quadrant {
                    occupancy_grid: Some(grid),
                    ..
                } => grid.reverse(),
                _ => {}
            }
        }
        display
    }
    /// Populate `pixel_perimeters` and `pixel_quadrants` from their
    /// metric-space counterparts. Called once after the manifest is built,
    /// before serialization.
    pub fn compute_pixel_geometry(&mut self) {
        let mpp = self.settings.meters_per_pixel;
        self.pixel_perimeters = to_video_pixels_vec(&self.perimeters, mpp);
        self.pixel_quadrants = to_video_pixels_vec(&self.quadrants, mpp);
    }

    /// Write the manifest as TOML to the given path.
    pub fn write_toml(&self, path: &Path) -> Result<()> {
        let toml_str = toml::to_string_pretty(self)
            .map_err(|e| crate::error::BekiperError::Other(e.to_string()))?;
        std::fs::write(path, toml_str)?;
        Ok(())
    }

    /// Read a manifest from a TOML file.
    pub fn from_toml(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content).map_err(|e| crate::error::BekiperError::Config(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// ExperimentResult — consolidated per-animal result file (`result.toml`)
// ---------------------------------------------------------------------------

/// High-level experiment identification written to `result.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentInfo {
    pub name: String,
    pub task_type: TaskType,
    pub fps: f64,
    pub meters_per_pixel: f64,
}

/// Animal metadata sourced from `metadata-animal.csv` (via `animal_scan`).
///
/// Lives in `bekiper-core` because `ExperimentResult` embeds it and the
/// serialized TOML schema must be stable across crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimalMetadata {
    pub animal: u32,
    pub cohort: String,
    pub gene: String,
    pub sex: String,
    pub age: String,
}

/// Source-study context preserved by ingress profiles.
///
/// This is intentionally generic: each study/profile can populate the fields
/// it knows without forcing every dataset into the same hierarchy.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngressSessionContext {
    pub profile: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_row: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trial: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_part: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apparatus: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub treatment: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub age: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub genotype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_a: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_b: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fecal_boli_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comments: Option<String>,
}

/// Consolidated per-animal result: all metrics + metadata in one TOML file.
///
/// Replaces the previous fan-out of `summary.toml`, `<experiment>_inspection.toml`,
/// `behaviour_summaries.toml`, `behaviour_result.toml`, and `scale.toml`.
/// The Parquet files (`augmented.parquet`, `behaviour.parquet`,
/// `<experiment>_evaluated.parquet`) remain the bulk frame-level data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub experiment: ExperimentInfo,
    pub video: VideoMetadata,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_path: Option<PathBuf>,
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
    /// Task-specific detailed results (zone dwell, discrimination indices, …).
    /// Free-form TOML table so each task can emit its own schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detailed: Option<toml::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animal_metadata: Option<AnimalMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_context: Option<IngressSessionContext>,
    /// Auto-generated quadrant grids; mirrors `InspectionManifest::quadrants`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quadrants: Vec<QuadrantSpec>,
    /// Per-feature inspection figure declarations; mirrors
    /// `InspectionManifest::feature_inspections`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feature_inspections: Vec<FeatureInspection>,
}

impl ExperimentResult {
    /// Write the consolidated result as TOML to the given path.
    pub fn write_toml(&self, path: &Path) -> Result<()> {
        let toml_str = toml::to_string_pretty(self)
            .map_err(|e| crate::error::BekiperError::Other(e.to_string()))?;
        std::fs::write(path, toml_str)?;
        Ok(())
    }

    /// Read a consolidated result from a TOML file.
    pub fn from_toml(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content).map_err(|e| crate::error::BekiperError::Config(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// serde_json::Value ↔ TOML conversion helpers
// ---------------------------------------------------------------------------

/// Convert a `serde_json::Value` to a TOML string, skipping null values.
pub fn json_value_to_toml_string(val: &serde_json::Value) -> Result<String> {
    let toml_val = json_to_toml_value(val)?;
    toml::to_string_pretty(&toml_val).map_err(|e| crate::error::BekiperError::Other(e.to_string()))
}

/// Parse a TOML string into a `serde_json::Value`.
pub fn toml_str_to_json_value(s: &str) -> Result<serde_json::Value> {
    let toml_val: toml::Value =
        toml::from_str(s).map_err(|e| crate::error::BekiperError::Config(e.to_string()))?;
    Ok(toml_to_json_value(&toml_val))
}

fn json_to_toml_value(val: &serde_json::Value) -> Result<toml::Value> {
    match val {
        serde_json::Value::Null => Ok(toml::Value::String(String::new())),
        serde_json::Value::Bool(b) => Ok(toml::Value::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(toml::Value::Integer(i))
            } else {
                Ok(toml::Value::Float(n.as_f64().unwrap_or(0.0)))
            }
        }
        serde_json::Value::String(s) => Ok(toml::Value::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let vals: Result<Vec<_>> = arr.iter().map(json_to_toml_value).collect();
            Ok(toml::Value::Array(vals?))
        }
        serde_json::Value::Object(map) => {
            let mut table = toml::map::Map::new();
            for (k, v) in map {
                if !v.is_null() {
                    table.insert(k.clone(), json_to_toml_value(v)?);
                }
            }
            Ok(toml::Value::Table(table))
        }
    }
}

fn toml_to_json_value(val: &toml::Value) -> serde_json::Value {
    match val {
        toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
        toml::Value::Integer(i) => serde_json::json!(*i),
        toml::Value::Float(f) => serde_json::json!(*f),
        toml::Value::String(s) => serde_json::Value::String(s.clone()),
        toml::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(toml_to_json_value).collect())
        }
        toml::Value::Table(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), toml_to_json_value(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
    }
}
