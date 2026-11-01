use std::io::BufRead;
use std::path::Path;

use bekiper_core::error::{BekiperError, Result};

use crate::PoseBackend;
use crate::deeplabcut::DeepLabCutReader;
use crate::lightning::LightningPoseReader;
use crate::sleap::SleapCsvReader;

/// Auto-detect the pose estimation backend from a file's header structure.
///
/// Detection heuristics:
/// - **DeepLabCut HDF5**: file extension `.h5` / `.hdf5` (binary format,
///   no text header to peek; only DLC writes h5 in this project)
/// - **DeepLabCut CSV**: First row starts with "scorer" (multi-level header)
/// - **SLEAP**: Header contains columns matching `*.x`, `*.y`, `*.score`
/// - **Lightning Pose**: Header contains columns matching `*_x`, `*_y`, `*_likelihood`
pub fn detect_backend(path: &Path) -> Result<PoseBackend> {
    // HDF5 is binary; reading it via BufReader::lines() crashes with
    // "stream did not contain valid UTF-8". Short-circuit on extension —
    // bekiper-pose only writes/reads DLC-flavoured h5 files.
    if let Some(ext) = path.extension().and_then(|e| e.to_str())
        && (ext.eq_ignore_ascii_case("h5") || ext.eq_ignore_ascii_case("hdf5"))
    {
        return Ok(PoseBackend::DeepLabCut);
    }

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines();

    let first_line = lines
        .next()
        .ok_or_else(|| BekiperError::Data("File is empty".into()))??;

    let first_cell = first_line.split(',').next().unwrap_or("").trim();

    // DLC: first row starts with "scorer"
    if first_cell.eq_ignore_ascii_case("scorer") {
        return Ok(PoseBackend::DeepLabCut);
    }

    // Check header columns for SLEAP vs Lightning Pose patterns
    let cols: Vec<&str> = first_line.split(',').map(|c| c.trim()).collect();

    let has_dot_score = cols.iter().any(|c| c.ends_with(".score"));
    let has_dot_x = cols.iter().any(|c| c.ends_with(".x"));

    if has_dot_x && has_dot_score {
        return Ok(PoseBackend::Sleap);
    }

    let has_underscore_likelihood = cols.iter().any(|c| c.ends_with("_likelihood"));
    let has_underscore_x = cols.iter().any(|c| c.ends_with("_x"));

    if has_underscore_x && has_underscore_likelihood {
        return Ok(PoseBackend::LightningPose);
    }

    // Fallback: SLEAP without scores (just .x/.y)
    if has_dot_x {
        return Ok(PoseBackend::Sleap);
    }

    Err(BekiperError::Data(format!(
        "Could not detect pose backend from header: {first_line}"
    )))
}

/// Auto-detect backend and construct the appropriate reader.
pub fn open(path: &Path) -> Result<Box<dyn crate::PoseReader>> {
    let backend = detect_backend(path)?;
    match backend {
        PoseBackend::DeepLabCut => Ok(Box::new(DeepLabCutReader::from_file(path)?)),
        PoseBackend::Sleap => Ok(Box::new(SleapCsvReader::from_file(path)?)),
        PoseBackend::LightningPose => Ok(Box::new(LightningPoseReader::from_file(path)?)),
    }
}
