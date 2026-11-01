//! TOML loader for the inspection manifest.
//!
//! Returns a `LoadedManifest` PyO3 object holding everything the Python
//! `InspectionManifest` carries except the polars `df`, which the Python
//! wrapper attaches after this loader returns.

use std::path::{Path, PathBuf};

use bekiper_core::inspection::InspectionManifest as RustManifest;
use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::features::feature_to_py;
use crate::perimeter::PyPerimeterSpec;
use crate::simple::{
    PyCoordinateColumnSpec, PyHeuristicMeta, PyIngressSessionContext, PyInspectionSettings,
    PyQuadrantSpec, PyScaleDerivation, PyVideoMeta,
};

/// Resolve a TOML-relative path against the TOML file's location, walking
/// up parent directories to find the first match. Mirrors the Python
/// `_resolve_optional_path` helper.
fn resolve_optional_path(raw: Option<&str>, toml_path: &Path) -> Option<String> {
    let raw = raw?;
    if raw.is_empty() {
        return None;
    }
    let p = Path::new(raw);
    if p.is_absolute() {
        return Some(p.to_string_lossy().into_owned());
    }
    if let Some(parent) = toml_path.parent() {
        let mut current: Option<&Path> = Some(parent);
        while let Some(dir) = current {
            let candidate = dir.join(p);
            if candidate.exists() {
                return Some(
                    candidate
                        .canonicalize()
                        .unwrap_or(candidate)
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            current = dir.parent();
        }
    }
    Some(raw.to_string())
}

#[pyclass(name = "LoadedManifest", module = "bekiper_rs")]
pub struct PyLoadedManifest {
    #[pyo3(get)]
    pub video: PyVideoMeta,
    #[pyo3(get)]
    pub frame_image: Option<String>,
    #[pyo3(get)]
    pub labels: Vec<String>,
    #[pyo3(get)]
    pub label_colors: std::collections::HashMap<String, String>,
    pub coordinate_columns_inner: std::collections::HashMap<String, PyCoordinateColumnSpec>,
    #[pyo3(get)]
    pub perimeters: Vec<PyPerimeterSpec>,
    #[pyo3(get)]
    pub heuristics: Vec<PyHeuristicMeta>,
    #[pyo3(get)]
    pub settings: PyInspectionSettings,
    #[pyo3(get)]
    pub scale_derivation: Option<PyScaleDerivation>,
    #[pyo3(get)]
    pub quadrants: Vec<PyQuadrantSpec>,
    pub feature_inspections_inner: Vec<Py<pyo3::PyAny>>,
    #[pyo3(get)]
    pub session_context: Option<PyIngressSessionContext>,
    #[pyo3(get)]
    pub pixel_perimeters: Vec<PyPerimeterSpec>,
    #[pyo3(get)]
    pub pixel_quadrants: Vec<PyQuadrantSpec>,
    #[pyo3(get)]
    pub overlay_y_columns: Vec<String>,
    #[pyo3(get)]
    pub overlay_vector_y_columns: Vec<String>,
}

#[pymethods]
impl PyLoadedManifest {
    #[getter]
    fn coordinate_columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.coordinate_columns_inner {
            d.set_item(k, v.clone().into_pyobject(py)?)?;
        }
        Ok(d)
    }

    #[getter]
    fn feature_inspections(&self, py: Python<'_>) -> Vec<Py<pyo3::PyAny>> {
        self.feature_inspections_inner
            .iter()
            .map(|o| o.clone_ref(py))
            .collect()
    }
}

/// Load an inspection manifest TOML file. Returns a `LoadedManifest`
/// containing every typed field; the caller (Python) attaches the polars
/// DataFrame.
#[pyfunction]
fn load_inspection_toml(py: Python<'_>, toml_path: PathBuf) -> PyResult<PyLoadedManifest> {
    let content = std::fs::read_to_string(&toml_path).map_err(|e| {
        PyFileNotFoundError::new_err(format!("failed to read {}: {e}", toml_path.display()))
    })?;
    let mut raw: toml::Value = toml::from_str(&content)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid TOML: {e}")))?;
    // Backfill `settings` defaults so older inspection outputs that omit
    // the field — or omit individual sub-fields — remain loadable. Mirrors
    // the previous Python loader's `DEFAULT_*` fallbacks.
    if let toml::Value::Table(t) = &mut raw {
        let settings = t
            .entry("settings".to_string())
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
        if let toml::Value::Table(s) = settings {
            s.entry("minimum_seconds_tolerance".to_string())
                .or_insert(toml::Value::Float(0.5));
            s.entry("maximum_seconds_distraction".to_string())
                .or_insert(toml::Value::Float(1.0 / 3.0));
            s.entry("meters_per_pixel".to_string())
                .or_insert(toml::Value::Float(0.001));
        }
    }
    let manifest: RustManifest = raw
        .clone()
        .try_into()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid manifest: {e}")))?;

    let video_path = manifest
        .video_path
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .or_else(|| {
            raw.get("video_path")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });
    let video_path_resolved = resolve_optional_path(video_path.as_deref(), &toml_path);

    let frame_image = manifest
        .frame_image
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .or_else(|| {
            raw.get("frame_image")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });
    let frame_image_resolved = resolve_optional_path(frame_image.as_deref(), &toml_path);

    let (overlay_y_columns, overlay_vector_y_columns) = manifest.overlay_y_bindings();
    let manifest = manifest.source_overlay_geometry();
    let perimeters = manifest
        .perimeters
        .iter()
        .map(|p| PyPerimeterSpec::from_rust(py, p))
        .collect::<PyResult<Vec<_>>>()?;
    let pixel_perimeters = manifest
        .pixel_perimeters
        .iter()
        .map(|p| PyPerimeterSpec::from_rust(py, p))
        .collect::<PyResult<Vec<_>>>()?;
    let heuristics = manifest
        .heuristics
        .iter()
        .map(PyHeuristicMeta::from_rust)
        .collect();
    let quadrants = manifest
        .quadrants
        .iter()
        .map(PyQuadrantSpec::from_rust)
        .collect();
    let pixel_quadrants = manifest
        .pixel_quadrants
        .iter()
        .map(PyQuadrantSpec::from_rust)
        .collect();
    let coordinate_columns_inner = manifest
        .coordinate_columns
        .iter()
        .map(|(k, v)| (k.clone(), PyCoordinateColumnSpec::from_rust(v)))
        .collect();
    let label_colors = manifest.label_colors.clone();
    let scale_derivation = manifest
        .scale_derivation
        .as_ref()
        .map(PyScaleDerivation::from_rust);
    let session_context = manifest
        .session_context
        .as_ref()
        .map(PyIngressSessionContext::from_rust);
    let feature_inspections_inner = manifest
        .feature_inspections
        .iter()
        .map(|f| feature_to_py(py, f))
        .collect::<PyResult<Vec<_>>>()?;

    Ok(PyLoadedManifest {
        video: PyVideoMeta::from_rust(&manifest.video, video_path_resolved),
        frame_image: frame_image_resolved,
        labels: manifest.labels.clone(),
        label_colors,
        coordinate_columns_inner,
        perimeters,
        heuristics,
        settings: PyInspectionSettings::from_rust(&manifest.settings),
        scale_derivation,
        quadrants,
        feature_inspections_inner,
        session_context,
        pixel_perimeters,
        pixel_quadrants,
        overlay_y_columns,
        overlay_vector_y_columns,
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLoadedManifest>()?;
    m.add_function(wrap_pyfunction!(load_inspection_toml, m)?)?;
    Ok(())
}
