use polars::prelude::*;

use crate::pipeline::PipelineStep;

/// Crop the trial temporally by dropping frames from the start and/or end.
///
/// This handles the common case where recordings are longer than the intended
/// trial duration and need trimming. All values in seconds; converted to
/// frames using the provided FPS.
pub struct TemporalCrop {
    /// Seconds to crop from the start of the recording.
    pub crop_start_seconds: f64,
    /// Seconds to crop from the end of the recording.
    pub crop_end_seconds: f64,
    /// Target trial length in seconds. If set (> 0), the trial is sliced to
    /// this duration after applying start/end crops. Excess frames are removed
    /// from the end.
    pub target_length_seconds: f64,
    /// Frames per second, for converting seconds to frame counts.
    pub fps: f64,
}

impl PipelineStep for TemporalCrop {
    fn name(&self) -> &str {
        "temporal_crop"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let df = lf.collect().expect("temporal_crop: collect failed");
        let n = df.height();

        let start_frames = (self.crop_start_seconds * self.fps).round() as usize;
        let end_frames = (self.crop_end_seconds * self.fps).round() as usize;

        let skip = start_frames.min(n);
        let take = n.saturating_sub(skip).saturating_sub(end_frames);

        if take == 0 {
            return df.lazy();
        }

        let mut cropped = df.slice(skip as i64, take);

        // Apply target length if configured
        if self.target_length_seconds > 0.0 {
            let target_frames = (self.target_length_seconds * self.fps).round() as usize;
            if target_frames < cropped.height() {
                cropped = cropped.slice(0, target_frames);
            }
        }

        cropped.lazy()
    }
}
