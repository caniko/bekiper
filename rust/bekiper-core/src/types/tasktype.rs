use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A named label for a tracked body part (e.g. "nose", "left_ear", "tail_base").
///
/// Type alias — body-part names originate from external CSV headers and ONNX model
/// specs (27+ variants for DLC SuperAnimal alone), so a closed enum is impractical.
pub type Label = String;

// ---------------------------------------------------------------------------
// TaskType — finite set of behavioural experiment protocols
// ---------------------------------------------------------------------------

/// Behavioural task type. Determines which analysis pipeline and heuristics
/// are applied to the experiment data.
///
/// Each variant maps to a specific scientific protocol with standardised
/// measurements (discrimination index, alternation %, center time %, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Novel Object Recognition Test (NORT) / Objects in Updating Locations (OUL).
    ObjectRecognition,
    /// Y-maze spontaneous alternation / radial arm maze.
    YMaze,
    /// Open Field Test (OFT).
    OpenField,
    /// Cheeseboard / reward tracing task.
    Cheeseboard,
}

include!("tasktype/task_type.rs");

/// Which auto-detection recipe to use for identifying objects/zones in the
/// video frame. Each variant corresponds to a specific experimental setup
/// and expected object configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetectionRecipe {
    /// NORT familiarisation phase: two identical objects.
    NortTraining,
    /// NORT test phase: one familiar + one novel object.
    NortNovelty,
    /// OUL familiarisation: two identical objects.
    OulTraining,
    /// OUL update phase: one object moved to new location.
    OulUpdate,
    /// OUL test phase: one familiar + one novel object.
    OulTest,
    /// Open Field Test: rectangular arena, no objects.
    Oft,
    /// Y-maze / radial arm maze.
    YMaze,
}

include!("tasktype/detection_recipe.rs");

/// Stage-specific object-recognition protocol.
///
/// This separates task semantics from the legacy Python class hierarchy:
/// all variants still run through the same Rust object-recognition engine,
/// but detailed metrics can distinguish training vs novelty vs OUL stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectRecognitionProtocol {
    NortTraining,
    NortNovelty,
    OulTraining,
    OulUpdate,
    OulTest,
}

include!("tasktype/object_recognition.rs");

/// Detection backend for identifying objects/zones in video frames.
///
/// When multiple backends are specified in a priority list, the system tries
/// each in order, falling back to the next on failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetectionBackend {
    /// Pre-computed perimeters.toml file — skip all detection entirely.
    /// The file is copied directly to the output directory.
    PreComputed,
    /// Pre-annotated perimeters from a manual annotation directory.
    /// Resolves perimeters from metadata CSV + plugin files (rectangle CSV,
    /// COCO JSON, radial-maze dirs). No video frame analysis needed.
    ManualAnnotation,
    /// Pure Rust classical CV pipeline (threshold, flood fill, morphology).
    /// Fast, no GPU required, works well for high-contrast dark objects.
    ClassicalCv,
    /// Deterministic Open Field arena calibration from static box/floor edges.
    /// This is bounded calibration, not object detection.
    ArenaCalibrationCv,
    /// SAM2 automatic mask generator (Python subprocess).
    /// Good general-purpose segmentation, requires GPU.
    Sam2Auto,
    /// GroundingDINO + SAM2 text-prompted detection (Python subprocess).
    /// Best for ambiguous scenes where text prompts help disambiguate.
    GroundingDinoSam2,
}

include!("tasktype/backend.rs");

/// Pre-defined pose estimation model configurations.
///
/// Each preset maps to a specific ONNX model architecture with known
/// input dimensions, normalization, and output format.
/// Use `Custom(PathBuf)` for user-provided spec JSON files.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum PoseModelPreset {
    /// DLC SuperAnimal top-view mouse (HRNet-W32, 27 kp, heatmaps).
    DlcSuperAnimal,
    /// SLEAP single-instance (13 kp, heatmaps).
    SleapSingleInstance,
    /// Lightning Pose (27 kp, flat keypoints).
    LightningPose,
    /// User-provided JSON spec file.
    Custom(PathBuf),
}

include!("tasktype/pose_model.rs");
