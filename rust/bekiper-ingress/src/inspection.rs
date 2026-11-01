//! Shared trial presentation boundary for staged and sealed-source execution.
use crate::{
    config::{ExperimentConfig, InspectConfig},
    stage::ExperimentOutputDir,
};
use bekiper_core::error::{BekiperError, Result};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn data(error: impl std::fmt::Display) -> BekiperError {
    BekiperError::Data(error.to_string())
}
fn check_cancel(cancelled: &dyn Fn() -> bool) -> Result<()> {
    if cancelled() {
        Err(data("inspection cancelled"))
    } else {
        Ok(())
    }
}

include!("inspection/bindings.rs");

include!("inspection/capabilities.rs");

include!("inspection/source_context.rs");

include!("inspection/inputs.rs");

include!("inspection/receipts.rs");

include!("inspection/render.rs");
