use serde::{Deserialize, Serialize};

use crate::types::MetersPerPixel;

/// Metadata about a video recording session.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// Frames per second of the recording.
    pub fps: f64,

    /// Total number of frames.
    pub total_frames: u64,

    /// Video resolution (width, height) in pixels.
    pub resolution: (u32, u32),

    /// Conversion factor from pixels to meters.
    pub meters_per_pixel: MetersPerPixel,
}

impl VideoMetadata {
    /// Convert a frame count to seconds.
    pub fn frames_to_seconds(&self, frames: u64) -> f64 {
        frames as f64 / self.fps
    }

    /// Convert seconds to frame count.
    pub fn seconds_to_frames(&self, seconds: f64) -> u64 {
        (seconds * self.fps).round() as u64
    }
}

/// How to obtain the frames-per-second value.
///
/// In TOML config:
/// ```toml
/// # Manual — user knows the FPS:
/// fps = 15.0
///
/// # Omitted — auto-detected from the video file (default):
/// # fps is not set
/// ```
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FpsConfig {
    /// A manually specified FPS value.
    Manual(f64),
    /// Auto-detect from the video file's metadata / timestamps.
    #[default]
    FromVideo,
}

/// Frame timing information extracted from a video.
///
/// For CFR (constant frame rate) videos, only `fps` is needed.
/// For VFR (variable frame rate) videos, per-frame timestamps are stored
/// and used for accurate time calculations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameTiming {
    /// Average/nominal FPS (always set, even for VFR).
    pub fps: f64,
    /// Per-frame timestamps in seconds (from PTS). Empty for CFR.
    /// When present, `timestamps[i]` is the presentation time of frame `i`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub timestamps: Vec<f64>,
    /// Whether the video was detected as VFR.
    pub is_vfr: bool,
}

impl FrameTiming {
    /// Create CFR timing from a constant frame rate.
    pub fn cfr(fps: f64) -> Self {
        Self {
            fps,
            timestamps: Vec::new(),
            is_vfr: false,
        }
    }

    /// Create timing from per-frame timestamps.
    ///
    /// Automatically detects VFR by checking if inter-frame intervals vary
    /// by more than 10% of the median interval.
    pub fn from_timestamps(timestamps: Vec<f64>) -> Self {
        if timestamps.len() < 2 {
            return Self {
                fps: 0.0,
                timestamps,
                is_vfr: false,
            };
        }

        let mut dts: Vec<f64> = timestamps.windows(2).map(|w| w[1] - w[0]).collect();
        let fps = if dts.is_empty() {
            0.0
        } else {
            let total_duration = match (timestamps.first(), timestamps.last()) {
                (Some(first), Some(last)) => last - first,
                _ => 0.0,
            };
            if total_duration > 0.0 {
                (timestamps.len() - 1) as f64 / total_duration
            } else {
                0.0
            }
        };

        // Detect VFR: check coefficient of variation of inter-frame intervals
        dts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_dt = dts[dts.len() / 2];
        let is_vfr = if median_dt > 0.0 {
            // Check if >5% of frames deviate more than 10% from median
            let outliers = dts
                .iter()
                .filter(|&&dt| (dt - median_dt).abs() / median_dt > 0.10)
                .count();
            outliers as f64 / dts.len() as f64 > 0.05
        } else {
            false
        };

        Self {
            fps,
            timestamps,
            is_vfr,
        }
    }

    /// Get the time delta between frame `i` and frame `i-1`.
    /// For CFR, returns `1/fps`. For VFR, returns actual interval.
    /// Returns 0 for frame 0 (no previous frame).
    pub fn dt(&self, frame_index: usize) -> f64 {
        if frame_index == 0 {
            return 0.0;
        }
        if self.is_vfr && frame_index < self.timestamps.len() {
            self.timestamps[frame_index] - self.timestamps[frame_index - 1]
        } else {
            1.0 / self.fps
        }
    }

    /// Convert a frame count to seconds.
    /// For CFR: `frames / fps`.
    /// For VFR: uses actual timestamps if available.
    pub fn frames_to_seconds(&self, start_frame: usize, end_frame: usize) -> f64 {
        if self.is_vfr && start_frame < self.timestamps.len() && end_frame < self.timestamps.len() {
            self.timestamps[end_frame] - self.timestamps[start_frame]
        } else {
            (end_frame - start_frame) as f64 / self.fps
        }
    }

    /// Total number of frames with timing info.
    pub fn frame_count(&self) -> usize {
        if self.timestamps.is_empty() {
            0
        } else {
            self.timestamps.len()
        }
    }

    /// Convert a count of true frames (within a range) to seconds.
    /// For CFR, simply divides by fps.
    /// For VFR, sums actual dt values for marked frames.
    pub fn bool_frames_to_seconds(&self, mask: &[bool]) -> f64 {
        if !self.is_vfr || self.timestamps.len() < 2 {
            let count = mask.iter().filter(|&&b| b).count();
            return count as f64 / self.fps;
        }

        let mut total = 0.0;
        for (i, &is_true) in mask.iter().enumerate() {
            if is_true {
                total += self.dt(i);
            }
        }
        total
    }

    /// Seconds from trial start to a given frame index.
    pub fn frame_to_elapsed(&self, frame_index: usize) -> f64 {
        if self.is_vfr && frame_index < self.timestamps.len() && !self.timestamps.is_empty() {
            self.timestamps[frame_index] - self.timestamps[0]
        } else {
            frame_index as f64 / self.fps
        }
    }

    /// Find the frame index closest to a given elapsed time (seconds from start).
    pub fn elapsed_to_frame(&self, elapsed: f64) -> usize {
        if self.is_vfr && !self.timestamps.is_empty() {
            let target = self.timestamps[0] + elapsed;
            match self
                .timestamps
                .binary_search_by(|t| t.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => i,
                Err(i) => i.min(self.timestamps.len() - 1),
            }
        } else {
            (elapsed * self.fps).round() as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfr_timing() {
        let t = FrameTiming::cfr(30.0);
        assert!(!t.is_vfr);
        assert!((t.dt(0) - 0.0).abs() < 1e-10);
        assert!((t.dt(1) - 1.0 / 30.0).abs() < 1e-10);
        assert!((t.frames_to_seconds(0, 30) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cfr_from_constant_timestamps() {
        let ts: Vec<f64> = (0..100).map(|i| i as f64 / 30.0).collect();
        let t = FrameTiming::from_timestamps(ts);
        assert!(!t.is_vfr);
        assert!((t.fps - 30.0).abs() < 0.1);
    }

    #[test]
    fn vfr_detection() {
        // Create timestamps with variable intervals
        let mut ts = Vec::new();
        let mut t = 0.0;
        for i in 0..100 {
            ts.push(t);
            // Alternate between 1/30 and 1/15 second intervals
            t += if i % 2 == 0 { 1.0 / 30.0 } else { 1.0 / 15.0 };
        }
        let timing = FrameTiming::from_timestamps(ts.clone());
        assert!(timing.is_vfr);

        // Check dt uses actual intervals
        let dt0 = timing.dt(1);
        let dt1 = timing.dt(2);
        assert!((dt0 - 1.0 / 30.0).abs() < 1e-10);
        assert!((dt1 - 1.0 / 15.0).abs() < 1e-10);
    }

    #[test]
    fn elapsed_to_frame_roundtrip() {
        let t = FrameTiming::cfr(30.0);
        assert_eq!(t.elapsed_to_frame(1.0), 30);
        assert!((t.frame_to_elapsed(30) - 1.0).abs() < 1e-10);
    }
}
