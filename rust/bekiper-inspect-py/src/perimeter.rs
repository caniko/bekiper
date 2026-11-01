//! PyO3 wrapper for `PerimeterSpec`.
//!
//! The Python API surface intentionally matches the previous dataclass
//! (`label`, `shape`, `params: dict`) so existing plotting code that reads
//! `p.shape == "circle"` and `p.params["radius"]` continues to work
//! unchanged. Internally the value is constructed from the Rust enum via
//! [`from_rust`], which is the single source of truth for the variant
//! schemas.

use bekiper_core::inspection::PerimeterSpec;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Extract a sequence of `(x, y)` pairs that may be either tuples or lists
/// in the Python value (TOML always loads vertices as lists, but tests
/// pass tuples too).
fn extract_xy_list(value: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<(f64, f64)>> {
    let mut out = Vec::new();
    for item in value.try_iter()? {
        let item = item?;
        let pair: Vec<f64> = item.extract()?;
        if pair.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "perimeter vertex must be a 2-element sequence",
            ));
        }
        out.push((pair[0], pair[1]));
    }
    Ok(out)
}

#[pyclass(name = "PerimeterSpec", module = "bekiper_rs")]
pub struct PyPerimeterSpec {
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub shape: String,
    /// `dict[str, Any]` mirroring the per-shape fields:
    /// - `circle`: `center_x`, `center_y`, `radius`
    /// - `rectangle`: `center_x`, `center_y`, `width`, `height`
    /// - `polygon`: `vertices` (`list[tuple[float, float]]`)
    /// - `radial_maze`: `center_vertices`, `arms`
    /// - `triangle`: `vertices`
    pub params: Py<PyDict>,
}

impl Clone for PyPerimeterSpec {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            label: self.label.clone(),
            shape: self.shape.clone(),
            params: self.params.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyPerimeterSpec {
    #[new]
    #[pyo3(signature = (label, shape, params=None))]
    fn new(
        py: Python<'_>,
        label: String,
        shape: String,
        params: Option<Py<PyDict>>,
    ) -> PyResult<Self> {
        let params = match params {
            Some(p) => p,
            None => PyDict::new(py).unbind(),
        };
        Ok(Self {
            label,
            shape,
            params,
        })
    }

    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.params.bind(py).clone()
    }

    /// `(center_x, center_y)` for circle/rectangle, or polygon centroid;
    /// `None` when the shape carries no obvious center.
    #[getter]
    fn center(&self, py: Python<'_>) -> PyResult<Option<(f64, f64)>> {
        let p = self.params.bind(py);
        if let (Ok(Some(cx)), Ok(Some(cy))) = (p.get_item("center_x"), p.get_item("center_y")) {
            return Ok(Some((cx.extract::<f64>()?, cy.extract::<f64>()?)));
        }
        if let Ok(Some(verts)) = p.get_item("vertices") {
            let pts = extract_xy_list(&verts)?;
            if pts.is_empty() {
                return Ok(None);
            }
            let n = pts.len() as f64;
            let (sx, sy) = pts
                .iter()
                .fold((0.0, 0.0), |(ax, ay), (x, y)| (ax + x, ay + y));
            return Ok(Some((sx / n, sy / n)));
        }
        Ok(None)
    }

    /// `vertices` as a numpy array of shape `(N, 2)` for polygon/triangle;
    /// `None` for shapes without an explicit vertex list. Mirrors the prior
    /// Python implementation that wrapped `params["vertices"]` in
    /// `np.array(...)`.
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, pyo3::PyAny>>> {
        let p = self.params.bind(py);
        match p.get_item("vertices")? {
            Some(v) => {
                let np = py.import("numpy")?;
                Ok(Some(np.call_method1("array", (v,))?))
            }
            None => Ok(None),
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "PerimeterSpec(label={:?}, shape={:?}, params=<{} keys>)",
            self.label,
            self.shape,
            self.params.bind(py).len()
        )
    }
}

impl PyPerimeterSpec {
    /// Build a Python-facing `PerimeterSpec` from the Rust enum.
    pub fn from_rust(py: Python<'_>, p: &PerimeterSpec) -> PyResult<Self> {
        let params = PyDict::new(py);
        let (label, shape) = match p {
            PerimeterSpec::Circle {
                label,
                center_x,
                center_y,
                radius,
            } => {
                params.set_item("center_x", *center_x)?;
                params.set_item("center_y", *center_y)?;
                params.set_item("radius", *radius)?;
                (label.clone(), "circle".to_string())
            }
            PerimeterSpec::Rectangle {
                label,
                center_x,
                center_y,
                width,
                height,
            } => {
                params.set_item("center_x", *center_x)?;
                params.set_item("center_y", *center_y)?;
                params.set_item("width", *width)?;
                params.set_item("height", *height)?;
                (label.clone(), "rectangle".to_string())
            }
            PerimeterSpec::Polygon { label, vertices } => {
                let verts = PyList::new(py, vertices.iter().map(|(x, y)| (*x, *y)))?;
                params.set_item("vertices", verts)?;
                (label.clone(), "polygon".to_string())
            }
            PerimeterSpec::RadialMaze {
                label,
                center_vertices,
                arms,
            } => {
                let cv = PyList::new(py, center_vertices.iter().map(|(x, y)| (*x, *y)))?;
                params.set_item("center_vertices", cv)?;
                let arms_py = PyList::empty(py);
                for arm in arms {
                    let a = PyList::new(py, arm.iter().map(|(x, y)| (*x, *y)))?;
                    arms_py.append(a)?;
                }
                params.set_item("arms", arms_py)?;
                (label.clone(), "radial_maze".to_string())
            }
            PerimeterSpec::Triangle { label, vertices } => {
                let verts = PyList::new(py, vertices.iter().map(|(x, y)| (*x, *y)))?;
                params.set_item("vertices", verts)?;
                (label.clone(), "triangle".to_string())
            }
        };
        Ok(Self {
            label,
            shape,
            params: params.unbind(),
        })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPerimeterSpec>()?;
    Ok(())
}
