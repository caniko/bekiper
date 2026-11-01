use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::AnimalMetadata;
use bekiper_core::scale::ScaleConfig;
use bekiper_core::types::{DetectionRecipe, TaskType};
use bekiper_core::video::FpsConfig;

use super::detection::{BackendPriority, default_backends};
use super::task_overrides::{TaskParamsOverrides, TaskVariant};

/// A directory scan rule: discovers videos and expands them into experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanConfig {
    /// Directory to walk (relative to the config file). All video files found
    /// recursively will become experiments.
    pub video_dir: PathBuf,
    pub task_type: TaskType,
    /// FPS: a number for manual override, or omit to auto-detect from video.
    #[serde(default)]
    pub fps: FpsConfig,
    /// Pixel-to-meter scale: either a manual value or a derivation method.
    ///
    /// ```toml
    /// meters_per_pixel = 0.001914
    /// meters_per_pixel = { method = "arena_side", meters = 0.40 }
    /// meters_per_pixel = { method = "arm_length", meters = 0.35 }
    /// ```
    pub meters_per_pixel: ScaleConfig,
    /// Default recipe applied to every discovered video.
    pub recipe: Option<DetectionRecipe>,
    /// Glob-pattern -> recipe overrides. First match wins.
    /// Example: `{ "*/T1/*" = "nort_training", "*/T2/*" = "nort_novelty" }`
    #[serde(default)]
    pub recipe_rules: HashMap<String, DetectionRecipe>,
    /// Detection backend(s) in priority order. Applied to all discovered videos.
    #[serde(default = "default_backends")]
    pub backend: BackendPriority,
    #[serde(default)]
    pub arena_calibration: Option<bekiper_detect::config::ArenaCalibrationConfig>,
    /// File extension filter. Defaults to `["mp4"]`.
    #[serde(default = "default_extensions")]
    pub extensions: Vec<String>,
    /// Manual annotation directory for pre-annotated perimeters.
    ///
    /// ```toml
    /// annotations = { dir = "annotations/nort" }
    /// ```
    pub annotations: Option<AnnotationConfig>,
    /// Task-specific parameter overrides (shared across all discovered videos).
    ///
    /// ```toml
    /// [scan.task_params]
    /// freezing_displacement_threshold = 0.005
    /// peripheral_band_fraction = 0.25
    /// ```
    #[serde(default)]
    pub task_params: Option<TaskParamsOverrides>,
    /// Named task-parameter variants. Phase 2 lands the schema only; expansion
    /// keeps the existing single-output behavior until variant expansion is
    /// wired separately.
    #[serde(default)]
    pub variants: Vec<TaskVariant>,
    /// Optional perimeter label that defines the trial start area.
    ///
    /// Cheeseboard uses this to resolve the start-area exit event from a
    /// specific perimeter rather than falling back to enclosure entry.
    #[serde(default)]
    pub trial_start_perimeter: Option<String>,
}

/// Manual annotation source for pre-annotated perimeters.
///
/// Points to a directory containing:
/// - `metadata.csv` mapping video numbers to perimeter IDs
/// - `plugin_files/` with perimeter annotations (CSV bboxes, COCO JSON, arm-lines)
///
/// ```toml
/// annotations = { dir = "annotations/nort" }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationConfig {
    /// Annotation directory (relative to the config file).
    pub dir: PathBuf,
}

pub(crate) fn default_extensions() -> Vec<String> {
    vec!["mp4".into()]
}

// ── Animal-structured data scan ────────────────────────────────────────────

/// Scan configuration for animal-structured directories with pre-computed
/// DLC pose data and pre-annotated perimeters.
///
/// ```toml
/// [[animal_scan]]
/// data_dir = "dlc-data"
/// task_type = "object_recognition"
/// trial_sequence = ["oul_training", "oul_training", "oul_training", "oul_update", "oul_test"]
/// animal_sequence_csv = "metadata-animal_sequence.csv"
/// animal_metadata_csv = "metadata-animal.csv"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimalScanConfig {
    /// Root directory containing animal subdirectories and metadata CSVs.
    pub data_dir: PathBuf,
    pub task_type: TaskType,
    /// Maps sequence index -> detection recipe. `trial_sequence[seq]` determines
    /// which OUL/NORT phase each video belongs to.
    pub trial_sequence: Vec<DetectionRecipe>,
    /// CSV mapping (Animal, Sequence) -> calibration file IDs.
    /// Columns: `Animal, Sequence, MetersPerPixel, Center, Enclosure`
    pub animal_sequence_csv: PathBuf,
    /// Optional CSV with animal metadata (Cohort, Gene, Sex, Age).
    pub animal_metadata_csv: Option<PathBuf>,
    /// Task-specific parameter overrides (shared across all animals).
    #[serde(default)]
    pub task_params: Option<TaskParamsOverrides>,
    /// Center zone dimensions in meters (width, height). Default: 0.35 x 0.35.
    ///
    /// Creates a perimeter labeled `"center_zone"` for OFT center/periphery
    /// analysis. This is **not** the arena boundary — see `arena_meters`.
    #[serde(default = "default_center_zone_meters")]
    pub center_zone_meters: (f64, f64),
    /// Full arena boundary dimensions in meters (width, height).
    ///
    /// When set, creates a perimeter labeled `"arena"` representing the
    /// physical arena floor. The `ConfinementNuller` clips tracking data to
    /// this boundary. When absent, confinement nulling is skipped for the
    /// animal-scan path.
    #[serde(default)]
    pub arena_meters: Option<(f64, f64)>,
    /// Optional perimeter label that defines the trial start area.
    ///
    /// This is threaded through to each expanded experiment and is primarily
    /// used by Cheeseboard/reward-tracing.
    #[serde(default)]
    pub trial_start_perimeter: Option<String>,
}

pub(crate) fn default_center_zone_meters() -> (f64, f64) {
    (0.35, 0.35)
}

/// A row from `metadata-animal_sequence.csv`.
#[derive(Debug, Clone)]
pub struct AnimalSequenceRow {
    pub animal: u32,
    pub sequence: u32,
    /// ID suffix for the `meters_per_pixel-diagonal-0.707-{id}.csv` file.
    pub mpp_id: String,
    /// ID suffix for the `center-{id}.csv` file.
    pub center_id: String,
    /// ID suffix for the enclosure file (may be empty).
    pub enclosure_id: String,
}

/// Parse `metadata-animal_sequence.csv` into rows.
///
/// Columns: `Animal, Sequence, MetersPerPixel, Center, Enclosure`
pub fn parse_animal_sequence_csv(path: &Path) -> Result<Vec<AnimalSequenceRow>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
        if record.len() < 4 {
            continue;
        }
        let animal: u32 = record[0]
            .trim()
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: Animal: {e}", path.display())))?;
        let sequence: u32 = record[1]
            .trim()
            .parse()
            .map_err(|e| BekiperError::Data(format!("{}: Sequence: {e}", path.display())))?;
        let mpp_id = record[2].trim().to_string();
        let center_id = record[3].trim().to_string();
        let enclosure_id = record
            .get(4)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        rows.push(AnimalSequenceRow {
            animal,
            sequence,
            mpp_id,
            center_id,
            enclosure_id,
        });
    }
    Ok(rows)
}

/// Parse `metadata-animal.csv` into a map keyed by animal ID.
///
/// Columns: `Animal, Cohort, Gene, Sex, Age` (trailing empty columns ignored).
pub fn parse_animal_metadata_csv(path: &Path) -> Result<HashMap<u32, AnimalMetadata>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

    let mut map = HashMap::new();
    for record in reader.records() {
        let record = record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
        if record.len() < 5 {
            continue;
        }
        let animal: u32 = match record[0].trim().parse() {
            Ok(v) => v,
            Err(_) => continue, // skip non-numeric rows
        };
        map.insert(
            animal,
            AnimalMetadata {
                animal,
                cohort: record[1].trim().to_string(),
                gene: record[2].trim().to_string(),
                sex: record[3].trim().to_string(),
                age: record[4].trim().to_string(),
            },
        );
    }
    Ok(map)
}
