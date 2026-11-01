//! PyO3 wrapper for `bekiper_core::env_check::check_env`.

use std::path::Path;

use bekiper_core::env_check::{EnvCheckReport, check_env};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyMapping};

#[pyclass(name = "EnvCheckReport", module = "bekiper_rs")]
#[derive(Clone)]
pub struct PyEnvCheckReport {
    #[pyo3(get, set)]
    pub hard_issues: Vec<String>,
    #[pyo3(get, set)]
    pub soft_missing: Vec<String>,
}

#[pymethods]
impl PyEnvCheckReport {
    #[new]
    #[pyo3(signature = (hard_issues=None, soft_missing=None))]
    fn new(hard_issues: Option<Vec<String>>, soft_missing: Option<Vec<String>>) -> Self {
        Self {
            hard_issues: hard_issues.unwrap_or_default(),
            soft_missing: soft_missing.unwrap_or_default(),
        }
    }

    fn is_clean(&self) -> bool {
        self.hard_issues.is_empty() && self.soft_missing.is_empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "EnvCheckReport(hard_issues={:?}, soft_missing={:?})",
            self.hard_issues, self.soft_missing
        )
    }
}

impl From<EnvCheckReport> for PyEnvCheckReport {
    fn from(r: EnvCheckReport) -> Self {
        Self {
            hard_issues: r.hard_issues,
            soft_missing: r.soft_missing.into_iter().map(String::from).collect(),
        }
    }
}

/// Run the environment check.
///
/// Both arguments mirror the legacy Python signature so existing tests
/// and call sites work without modification:
///
/// - ``env`` — mapping of env var → value. Defaults to ``os.environ``.
/// - ``path_exists`` — predicate for whether ``ORT_DYLIB_PATH`` resolves
///   to a real file. Defaults to ``Path.exists``.
#[pyfunction]
#[pyo3(signature = (env=None, path_exists=None))]
fn check_env_py(
    py: Python<'_>,
    env: Option<Bound<'_, PyAny>>,
    path_exists: Option<Bound<'_, PyAny>>,
) -> PyResult<PyEnvCheckReport> {
    // Materialize the lookup into a Rust HashMap so we can pass closures
    // freely without re-entering Python for each variable.
    let mut env_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let env_obj = match env {
        Some(e) => e,
        None => py.import("os")?.getattr("environ")?,
    };
    let mapping = env_obj.downcast::<PyMapping>()?;
    let items_any = mapping.items()?;
    let items = items_any.downcast::<PyList>()?;
    for entry in items.iter() {
        let (k, v): (String, String) = entry.extract()?;
        env_map.insert(k, v);
    }

    let lookup = |key: &str| env_map.get(key).cloned();

    // The Python `path_exists` callback either takes a `pathlib.Path` (its
    // historical signature) or, when omitted, defaults to `Path.exists`
    // applied to a pathlib.Path.
    let pathlib_path = py.import("pathlib")?.getattr("Path")?;
    let path_exists_callback: Box<dyn Fn(&Path) -> bool> = match path_exists {
        Some(cb) => Box::new(move |p: &Path| -> bool {
            let s = p.to_string_lossy();
            let py_path = match pathlib_path.call1((s.as_ref(),)) {
                Ok(v) => v,
                Err(_) => return false,
            };
            cb.call1((py_path,))
                .and_then(|v| v.is_truthy())
                .unwrap_or(false)
        }),
        None => Box::new(|p: &Path| p.exists()),
    };

    let report = check_env(lookup, |p| path_exists_callback(p));
    Ok(report.into())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnvCheckReport>()?;
    m.add_function(wrap_pyfunction!(check_env_py, m)?)?;
    // Expose under the legacy Python name `check_env`.
    let f = m.getattr("check_env_py")?;
    m.add("check_env", f)?;
    Ok(())
}
