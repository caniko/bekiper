struct VideoProbe {
    path: std::path::PathBuf,
    detection_frame: u64,
}

fn probe_experiment_video(
    experiment: &ExperimentConfig,
    base_dir: &Path,
    summary: &mut DetectionDryRunSummary,
) -> Option<VideoProbe> {
    let Some(video) = experiment.video.as_deref() else {
        summary.push(
            DryRunSeverity::Error,
            "path/video",
            &experiment.name,
            "detection requires a video path",
        );
        return None;
    };
    let video_path = resolve_runtime_path(base_dir, video);
    if !video_path.exists() {
        summary.push(
            DryRunSeverity::Error,
            "path/video",
            &experiment.name,
            format!("missing video {}", video_path.display()),
        );
        return None;
    }
    match VideoDecoder::open(&video_path) {
        Ok(decoder) => {
            let info = decoder.info();
            summary.push(
                DryRunSeverity::Ok,
                "path/video",
                &experiment.name,
                format!(
                    "{}: {}x{}, {} frame(s)",
                    video_path.display(),
                    info.width,
                    info.height,
                    info.total_frames
                ),
            );
            Some(VideoProbe {
                path: video_path,
                detection_frame: info.total_frames / 2,
            })
        }
        Err(error) => {
            summary.push(
                DryRunSeverity::Error,
                "path/video",
                &experiment.name,
                format!("cannot open video {}: {error}", video_path.display()),
            );
            None
        }
    }
}
