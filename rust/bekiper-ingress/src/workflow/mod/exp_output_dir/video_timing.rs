/// Populate per-experiment video metadata needed by downstream stages.
///
/// Every experiment with a video gets its resolution filled when missing so
/// `pipeline.invert_y` can resolve a source frame height regardless of how
/// FPS was configured. Experiments with `fps = FromVideo` additionally probe
/// frame timestamps and cache them in `frame_timing.toml`.
#[cfg(test)]
fn probe_video_timing_with<FRes, FTiming>(
    scheduler: &ParallelScheduler,
    experiments: &mut [ExperimentConfig],
    base_dir: &Path,
    output_dir: &Path,
    resolve_resolution: &FRes,
    extract_timing: &FTiming,
) -> Result<()>
where
    FRes: Fn(&Path) -> Result<(u32, u32)> + Sync,
    FTiming: Fn(&Path) -> Result<FrameTiming> + Sync,
{
    let legacy = experiments
        .iter()
        .map(|experiment| exp_output_dir(output_dir, experiment))
        .collect();
    probe_video_timing_shared(
        scheduler,
        experiments,
        base_dir,
        &SharedRun::legacy_only(legacy),
        resolve_resolution,
        extract_timing,
    )
}

/// Same as above; variants of one recording probe their video once and link
/// the cached result into their own checkpoint directory.
fn probe_video_timing_shared<FRes, FTiming>(
    scheduler: &ParallelScheduler,
    experiments: &mut [ExperimentConfig],
    base_dir: &Path,
    shared: &SharedRun,
    resolve_resolution: &FRes,
    extract_timing: &FTiming,
) -> Result<()>
where
    FRes: Fn(&Path) -> Result<(u32, u32)> + Sync,
    FTiming: Fn(&Path) -> Result<FrameTiming> + Sync,
{
    let mut items: Vec<(usize, &mut ExperimentConfig)> =
        experiments.iter_mut().enumerate().collect();
    scheduler.try_for_each_mut(&mut items, |(i, experiment)| -> Result<()> {
        let i = *i;
        if experiment.video.is_none() {
            experiment.video_resolution = crate::analysis::resolve_source_resolution(experiment)?;
            return Ok(());
        }

        let video_path = experiment
            .video
            .as_ref()
            .map(|video| resolve_runtime_path(base_dir, video));
        let Some(video_path) = video_path else {
            return Ok(());
        };
        let from_video = matches!(experiment.fps, FpsConfig::FromVideo);

        // Variants of one recording probe once: the first computes under the
        // probe lock, siblings read the sealed result and link it.
        if shared.is_shared(i) {
            shared.execute(
                i,
                Shared::Probe,
                false,
                |view| view.shared_sealed(Shared::Probe),
                |view| {
                    let (width, height) = resolve_resolution(&video_path)?;
                    write_toml(&view.video_info_toml(), &VideoInfo { width, height })?;
                    if from_video {
                        let timing = extract_timing(&video_path)?;
                        tracing::info!(
                            experiment = %experiment.name,
                            fps = timing.fps,
                            is_vfr = timing.is_vfr,
                            n_frames = timing.frame_count(),
                            "auto-detected video timing"
                        );
                        write_toml(&view.frame_timing_toml(), &timing)?;
                    }
                    Ok(())
                },
            )?;
            let dir = shared.legacy(i);
            if experiment.video_resolution.is_none() {
                let info: VideoInfo =
                    toml::from_str(&std::fs::read_to_string(dir.video_info_toml())?)
                        .map_err(|e| BekiperError::Data(format!("invalid video_info.toml: {e}")))?;
                experiment.video_resolution = Some((info.width, info.height));
            }
            if from_video {
                let timing: FrameTiming = toml::from_str(&std::fs::read_to_string(
                    dir.frame_timing_toml(),
                )?)
                .map_err(|e| BekiperError::Data(format!("invalid frame_timing.toml: {e}")))?;
                experiment.frame_timing = Some(timing);
            }
            return Ok(());
        }

        if experiment.video_resolution.is_none() {
            let (width, height) = resolve_resolution(&video_path)?;
            experiment.video_resolution = Some((width, height));
        }

        if !from_video {
            return Ok(());
        }

        // Check for cached timing first
        let timing_path = shared.legacy(i).frame_timing_toml();
        if timing_path.exists() {
            let content = std::fs::read_to_string(&timing_path)?;
            if let Ok(timing) = toml::from_str::<FrameTiming>(&content) {
                tracing::debug!(
                    experiment = %experiment.name,
                    fps = timing.fps,
                    "loaded cached frame timing"
                );
                experiment.frame_timing = Some(timing);
                return Ok(());
            }
        }

        let timing = extract_timing(&video_path)?;
        tracing::info!(
            experiment = %experiment.name,
            fps = timing.fps,
            is_vfr = timing.is_vfr,
            n_frames = timing.frame_count(),
            "auto-detected video timing"
        );

        // Cache in checkpoints dir (not the experiment result dir)
        write_toml(&timing_path, &timing)?;

        experiment.frame_timing = Some(timing);
        Ok(())
    })
}

#[derive(serde::Serialize, serde::Deserialize)]
struct VideoInfo {
    width: u32,
    height: u32,
}

fn write_toml<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = toml::to_string_pretty(value).map_err(|e| BekiperError::Other(e.to_string()))?;
    std::fs::write(path, text)?;
    Ok(())
}
