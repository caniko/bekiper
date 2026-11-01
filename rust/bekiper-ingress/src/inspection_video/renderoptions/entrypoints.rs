pub fn render_from_directory(dir: &Path, opts: &RenderOptions) -> Result<Vec<PathBuf>> {
    let theme = VideoTheme::load(opts.config_path.as_deref(), opts.ffmpeg_preset.as_deref());
    render_from_directory_with_theme(dir, opts, &theme)
}

pub fn render_from_directory_with_theme(
    dir: &Path,
    opts: &RenderOptions,
    theme: &VideoTheme,
) -> Result<Vec<PathBuf>> {
    render_from_directory_with_cancel(dir, opts, theme, &|| false)
}

pub fn render_from_directory_with_cancel(
    dir: &Path,
    opts: &RenderOptions,
    theme: &VideoTheme,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<PathBuf>> {
    validate_options(opts)?;
    let bundle = LoadedInspection::load(dir, opts.source_video.as_deref())?;
    let jobs = build_jobs(&bundle, opts)?;
    if jobs.is_empty() {
        return Err(BekiperError::Config(
            "no renderable inspection videos selected".into(),
        ));
    }

    let mut written = Vec::with_capacity(jobs.len());
    for job in jobs {
        tracing::info!(output = %job.output_path.display(), target = %job.display_name(), "rendering inspection video");
        render_job(&bundle, theme, opts, &job, cancelled)?;
        written.push(job.output_path);
    }
    Ok(written)
}
