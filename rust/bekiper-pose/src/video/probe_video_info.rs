use super::*;
use std::path::Path;

use bekiper_core::error::{BekiperError, Result};
use image::DynamicImage;

/// Probe video metadata without decoding any frames.
///
/// Returns fps, resolution, total frames, and codec info.
pub fn probe_video_info(path: &Path) -> Result<VideoInfo> {
    ffmpeg_next::init().map_err(|e| BekiperError::Data(format!("ffmpeg init: {e}")))?;

    let input = ffmpeg_next::format::input(path)
        .map_err(|e| BekiperError::Data(format!("cannot open video {}: {e}", path.display())))?;

    let stream = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| BekiperError::Data("no video stream found".into()))?;

    let time_base = stream.time_base();
    let fps = f64::from(stream.avg_frame_rate());
    let total_frames = derive_total_frames(
        stream.frames(),
        stream.duration(),
        (time_base.numerator(), time_base.denominator()),
        fps,
    );
    let params = stream.parameters();

    let ctx = ffmpeg_next::codec::Context::from_parameters(params)
        .map_err(|e| BekiperError::Data(format!("codec context: {e}")))?;
    let decoder = ctx
        .decoder()
        .video()
        .map_err(|e| BekiperError::Data(format!("video decoder: {e}")))?;

    let codec_name = decoder
        .codec()
        .map(|c| c.name().to_string())
        .unwrap_or_else(|| "unknown".into());

    Ok(VideoInfo {
        width: decoder.width(),
        height: decoder.height(),
        output_width: decoder.width(),
        output_height: decoder.height(),
        total_frames,
        fps,
        codec: codec_name,
        hw_accel: false,
        time_base: (time_base.numerator(), time_base.denominator()),
    })
}

pub(super) fn derive_total_frames(
    reported_frames: i64,
    duration: i64,
    time_base: (i32, i32),
    fps: f64,
) -> u64 {
    if reported_frames > 0 {
        return reported_frames as u64;
    }

    let (time_base_num, time_base_den) = time_base;
    if duration <= 0 || time_base_num <= 0 || time_base_den <= 0 || !fps.is_finite() || fps <= 0.0 {
        return 0;
    }

    let seconds = duration as f64 * time_base_num as f64 / time_base_den as f64;
    let frames = (seconds * fps).round();
    if frames.is_finite() && frames > 0.0 {
        frames as u64
    } else {
        0
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[test]
    fn video_total_frames_prefers_reported_frame_count() {
        assert_eq!(derive_total_frames(123, 4000, (1, 1000), 25.0), 123);
    }

    #[test]
    fn video_total_frames_falls_back_to_duration() {
        assert_eq!(derive_total_frames(0, 4000, (1, 1000), 25.0), 100);
    }

    #[test]
    fn video_total_frames_returns_zero_without_valid_metadata() {
        assert_eq!(derive_total_frames(0, 0, (1, 1000), 25.0), 0);
        assert_eq!(derive_total_frames(0, 4000, (0, 1000), 25.0), 0);
        assert_eq!(derive_total_frames(0, 4000, (1, 1000), 0.0), 0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn vulkan_decoder_skips_h264_on_local_rocm() {
        assert!(!should_try_vulkan_decoder(ffmpeg_next::codec::Id::H264));
        assert!(should_try_vulkan_decoder(ffmpeg_next::codec::Id::HEVC));
    }
}

/// Extract all frame presentation timestamps from a video without decoding.
///
/// This does a fast demux pass, reading only packet PTS values from the video
/// stream. No pixel decoding occurs, so it's much faster than full decode.
///
/// Returns a `FrameTiming` with per-frame timestamps and VFR detection.
pub fn extract_frame_timestamps(path: &Path) -> Result<bekiper_core::video::FrameTiming> {
    ffmpeg_next::init().map_err(|e| BekiperError::Data(format!("ffmpeg init: {e}")))?;

    let mut input = ffmpeg_next::format::input(path)
        .map_err(|e| BekiperError::Data(format!("cannot open video {}: {e}", path.display())))?;

    let stream = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| BekiperError::Data("no video stream found".into()))?;

    let stream_index = stream.index();
    let time_base = stream.time_base();
    let tb_num = time_base.numerator() as f64;
    let tb_den = time_base.denominator() as f64;

    let mut timestamps = Vec::new();

    for (stream, packet) in input.packets() {
        if stream.index() == stream_index
            && let Some(pts) = packet.pts()
        {
            timestamps.push(pts as f64 * tb_num / tb_den);
        }
    }

    // Sort by timestamp (packets should already be in order, but be safe)
    timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    tracing::info!(
        n_frames = timestamps.len(),
        path = %path.display(),
        "extracted frame timestamps"
    );

    Ok(bekiper_core::video::FrameTiming::from_timestamps(
        timestamps,
    ))
}

impl Iterator for VideoDecoder {
    type Item = Result<(u64, DynamicImage)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_frame() {
            Ok(Some(frame)) => Some(Ok(frame)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
