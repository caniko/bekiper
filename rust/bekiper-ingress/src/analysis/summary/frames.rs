/// Extract representative frames nominated by feature inspections and save
/// them as PNGs to `{output}/frames/frame_XXXXXX.png`. The Python inspector
/// can then load these directly instead of decoding from the source MP4.
#[cfg(feature = "pose-video")]
pub(crate) fn extract_representative_frames(
    video_path: &Path,
    feature_inspections: &[bekiper_core::inspection::FeatureInspection],
    evaluated_df: Option<&polars::prelude::DataFrame>,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    // Nominations are evaluated-row positions; PNG names use source-video keys.
    let source_frames = evaluated_df
        .map(super::frame_keys::source_frame_indices)
        .transpose()?;
    let mut indices: Vec<u64> = feature_inspections
        .iter()
        .filter_map(|fi| fi.representative_frame_index())
        .map(|row| match &source_frames {
            Some(frames) => frames.get(row as usize).copied().ok_or_else(|| {
                BekiperError::Data(format!(
                    "Representative row {row} is outside the evaluated table"
                ))
            }),
            None => Ok(row),
        })
        .collect::<Result<_>>()?;
    indices.sort_unstable();
    indices.dedup();

    if indices.is_empty() {
        tracing::debug!("no representative frames to extract");
        return Ok(());
    }

    let frames_dir = exp_dir.frames_dir();
    std::fs::create_dir_all(&frames_dir)?;

    tracing::info!(
        n_frames = indices.len(),
        output = %frames_dir.display(),
        "extracting representative frames"
    );

    // Variants of one recording share extracted frames through a cache keyed by
    // source frame; only frames missing from it are decoded.
    let cache = exp_dir.frame_cache_dir();
    let png_name = |idx: u64| format!("frame_{idx:06}.png");
    let missing: Vec<u64> = match cache {
        Some(cache) => indices
            .iter()
            .copied()
            .filter(|idx| !cache.join(png_name(*idx)).exists())
            .collect(),
        None => indices.clone(),
    };
    let target_dir = cache.unwrap_or(&frames_dir);
    std::fs::create_dir_all(target_dir)?;
    if missing.is_empty() {
        return link_cached_frames(&indices, target_dir, &frames_dir);
    }
    let mut decoder = VideoDecoder::open(video_path)?;

    // Frame keys are presentation-order ordinals, not rounded PTS * average
    // FPS. A single sequential pass also handles irregular clocks and the
    // delayed B-frames at EOF without relabelling keyframe seeks.
    for idx in missing {
        let frame = loop {
            let frame = decoder.next_frame_timed()?.ok_or_else(|| {
                BekiperError::Data(format!("Cannot decode representative source frame {idx}"))
            })?;
            if frame.index == idx {
                break frame;
            }
            if frame.index > idx {
                let reason = "was skipped by the decoder"; // coverage: unreachable — decoded ordinals are consecutive and indices sorted
                let message = format!("Representative source frame {idx} {reason}"); // coverage: unreachable — see above
                return Err(BekiperError::Data(message)); // coverage: unreachable — see above
            }
        };
        let png_path = target_dir.join(png_name(idx));
        // Sibling variants may extract concurrently: write a private file, then
        // rename it into place (identical content, atomic replace).
        static PARTIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = PARTIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let partial = target_dir.join(format!("partial_{idx:06}_{unique}.png"));
        frame
            .image
            .save(&partial)
            .map_err(|e| BekiperError::Data(format!("save frame PNG: {e}")))?;
        std::fs::rename(&partial, &png_path)?;
        tracing::debug!(frame = idx, path = %png_path.display(), "extracted frame");
    }

    link_cached_frames(&indices, target_dir, &frames_dir)
}

/// Link frames from the shared cache into a variant's `frames/` directory.
/// Without a cache the frames were already written in place.
#[cfg(feature = "pose-video")]
fn link_cached_frames(indices: &[u64], source: &Path, frames_dir: &Path) -> Result<()> {
    if source == frames_dir {
        return Ok(());
    }
    for idx in indices {
        let name = format!("frame_{idx:06}.png");
        crate::stage_share::link_frame(&source.join(&name), &frames_dir.join(&name))?;
    }
    Ok(())
}
