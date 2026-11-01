//! PyO3 wrapper exposing the built-in `bekiper_core::experiment` descriptors.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Built-in experiment descriptors as a list of plain dicts (keys mirror the TOML).
#[pyfunction]
fn experiment_descriptors<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let json = serde_json::to_string(bekiper_core::experiment::descriptors())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.import("json")?.call_method1("loads", (json,))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(experiment_descriptors, m)?)
}
