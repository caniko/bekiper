pub mod cache;
pub mod deeplabcut;
pub mod detect;
pub mod infer;
pub mod io;
pub mod lightning;
pub mod pipeline;
pub mod skeleton;
pub mod sleap;
pub mod steps;
#[cfg(feature = "video")]
pub mod video;

use std::fmt;
use std::path::Path;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::error::Result;

/// Supported pose estimation backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PoseBackend {
    DeepLabCut,
    Sleap,
    LightningPose,
}

impl fmt::Display for PoseBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeepLabCut => f.write_str("DeepLabCut"),
            Self::Sleap => f.write_str("SLEAP"),
            Self::LightningPose => f.write_str("LightningPose"),
        }
    }
}

/// Unified interface for reading pose estimation output.
///
/// All backends produce the same flat column layout:
///   `{label}_x`, `{label}_y`, `{label}_likelihood`
///
/// plus a `frame` column (integer index).
pub trait PoseReader: Send + Sync {
    fn backend(&self) -> PoseBackend;

    /// Read pose data from a file into a LazyFrame with normalized columns.
    fn read(&self, path: &Path) -> Result<LazyFrame>;

    /// The body part labels this reader produces.
    fn labels(&self) -> &[String];
}
