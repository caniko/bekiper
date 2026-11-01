//! PyO3 bindings exposing bekiper-core inspection types to Python.
//!
//! The `bekiper-rs` Python package is the single source of truth for these
//! types — the Python `bekiper_inspect.manifest` package re-exports them and
//! delegates TOML parsing to the Rust loader here.

use pyo3::prelude::*;

mod env_check;
mod experiment;
mod features;
mod loader;
mod perimeter;
mod simple;

#[pymodule]
fn _bekiper_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    simple::register(m)?;
    perimeter::register(m)?;
    features::register(m)?;
    loader::register(m)?;
    env_check::register(m)?;
    experiment::register(m)?;
    themekit::python::register(m)?;
    Ok(())
}
