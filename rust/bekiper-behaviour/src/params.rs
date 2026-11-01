//! Shared parameters for all behavioural tasks (FPS, coordinate columns,
//! thresholds).

use serde::{Deserialize, Serialize};

use bekiper_core::types::CoordinateColumns;

// ── Shared task-parameter defaults ────────────────────────────────────────
// All values are overridable at runtime via `TaskParamsOverrides` in
// `bekiper-ingress/src/config.rs`. Freezing defaults live in
// `bekiper-feature` next to the algorithm and are re-exported here.

pub use bekiper_feature::heuristic::solo::freezing::{
    DEFAULT_DISPLACEMENT_THRESHOLD as DEFAULT_FREEZING_DISPLACEMENT_THRESHOLD,
    DEFAULT_WINDOW_SECONDS as DEFAULT_FREEZING_WINDOW_SECONDS,
};

/// Minimum dwell time (s) for a zone entry to be counted. Filters out
/// brief boundary crossings.
pub use bekiper_feature::tolerance::DEFAULT_MIN_DWELL_SECONDS;
pub use bekiper_feature::zone::DEFAULT_OFT_WINDOW_SECONDS;
pub use bekiper_feature::zone::DEFAULT_VIDEO_CENTER_FRACTION;

/// Peripheral band width as a fraction of the shorter arena side. Default
/// 0.2 means 1/5 of the arena side forms the outer band.
pub const DEFAULT_PERIPHERAL_BAND_FRACTION: f64 = 0.2;

/// Minimum OFT freezing bout duration (s) counted in output totals.
pub const DEFAULT_OFT_FREEZING_MIN_BOUT_S: f64 = 1.0;
/// Minimum OFT grooming bout duration (s) counted in output totals.
pub const DEFAULT_OFT_GROOMING_MIN_BOUT_S: f64 = 1.0;
/// Minimum OFT rearing bout duration (s) counted in output totals.
pub const DEFAULT_OFT_REARING_MIN_BOUT_S: f64 = 0.5;

/// Default column name for the body centroid X coordinate.
pub const DEFAULT_CENTROID_X: &str = "mouse_center_x";
/// Default column name for the body centroid Y coordinate.
pub const DEFAULT_CENTROID_Y: &str = "mouse_center_y";
/// Default column name for the nose X coordinate.
pub const DEFAULT_NOSE_X: &str = "nose_x";
/// Default column name for the nose Y coordinate.
pub const DEFAULT_NOSE_Y: &str = "nose_y";
/// Default column name for the head-base X coordinate.
pub const DEFAULT_HEAD_X: &str = "center_ear_x";
/// Default column name for the head-base Y coordinate.
pub const DEFAULT_HEAD_Y: &str = "center_ear_y";
/// Default column name for the tail-base X coordinate.
pub const DEFAULT_TAIL_X: &str = "tail_base_x";
/// Default column name for the tail-base Y coordinate.
pub const DEFAULT_TAIL_Y: &str = "tail_base_y";

/// Shared parameters common to all behavioural tasks.
///
/// These defaults match the standard rodent behavioural neuroscience protocols.
/// All values are overridable through the TOML configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskParams {
    /// Frames per second of the video.
    pub fps: f64,
    /// Window duration (seconds) for the rolling displacement freezing check.
    pub freezing_window_seconds: f64,
    /// Maximum cumulative displacement (meters) within the freezing window
    /// for a frame to be classified as frozen.
    pub freezing_displacement_threshold: f64,
    /// Minimum dwell time (seconds) for a zone entry to count.
    pub min_dwell_seconds: f64,
    /// OFT seconds from first valid detection; zero reproduces full-video scoring.
    #[serde(default = "default_oft_window_seconds")]
    pub oft_window_seconds: f64,
    /// Body centroid coordinate columns (for zone confinement, distance, speed).
    pub centroid_coords: CoordinateColumns,
    /// Nose coordinate columns (for exploration proximity).
    pub nose_coords: CoordinateColumns,
    /// Head base coordinate columns (for freezing speed, head direction).
    pub head_coords: CoordinateColumns,
    /// Tail base coordinate columns (for dual-body-part zone checks).
    pub tail_coords: CoordinateColumns,
    /// Peripheral band as a fraction of the shorter arena side.
    /// E.g. 0.2 means 1/5 of arena side is peripheral zone.
    pub peripheral_band_fraction: f64,
}

impl TaskParams {
    /// Create with the given FPS and standard defaults.
    ///
    /// # Panics
    /// Panics if `fps` is not positive (must be > 0).
    pub fn new(fps: f64) -> Self {
        assert!(fps > 0.0, "fps must be positive, got {fps}");
        Self {
            fps,
            freezing_window_seconds: DEFAULT_FREEZING_WINDOW_SECONDS,
            freezing_displacement_threshold: DEFAULT_FREEZING_DISPLACEMENT_THRESHOLD,
            min_dwell_seconds: DEFAULT_MIN_DWELL_SECONDS,
            oft_window_seconds: DEFAULT_OFT_WINDOW_SECONDS,
            centroid_coords: CoordinateColumns::new(DEFAULT_CENTROID_X, DEFAULT_CENTROID_Y),
            nose_coords: CoordinateColumns::new(DEFAULT_NOSE_X, DEFAULT_NOSE_Y),
            head_coords: CoordinateColumns::new(DEFAULT_HEAD_X, DEFAULT_HEAD_Y),
            tail_coords: CoordinateColumns::new(DEFAULT_TAIL_X, DEFAULT_TAIL_Y),
            peripheral_band_fraction: DEFAULT_PERIPHERAL_BAND_FRACTION,
        }
    }

    /// Config-keyed view of the shared knobs (everything but `fps`).
    pub fn effective_parameters(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).unwrap_or_default();
        if let Some(map) = value.as_object_mut() {
            map.remove("fps");
        }
        value
    }

    /// Minimum dwell frames, derived from fps and min_dwell_seconds.
    pub fn min_dwell_frames(&self) -> u32 {
        bekiper_math::discrete::legacy_frame_threshold(self.min_dwell_seconds, self.fps)
    }
}

fn default_oft_window_seconds() -> f64 {
    DEFAULT_OFT_WINDOW_SECONDS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_dwell_frames_default() {
        let params = TaskParams::new(30.0);
        // 30 fps * 1.0s default = 30 frames
        assert_eq!(params.min_dwell_frames(), 30);
    }

    #[test]
    fn min_dwell_frames_custom() {
        let mut params = TaskParams::new(60.0);
        params.min_dwell_seconds = 0.5;
        // 60 fps * 0.5s = 30 frames
        assert_eq!(params.min_dwell_frames(), 30);
    }

    #[test]
    fn min_dwell_frames_fractional() {
        let mut params = TaskParams::new(25.0);
        params.min_dwell_seconds = 1.5;
        // 25 * 1.5 = 37.5 → truncated to 37
        assert_eq!(params.min_dwell_frames(), 37);
    }
}
