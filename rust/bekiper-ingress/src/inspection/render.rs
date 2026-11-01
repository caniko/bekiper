pub fn is_fresh(dir: &ExperimentOutputDir, settings: &InspectConfig) -> bool {
    let verify = || -> Result<()> {
        let receipt: Value =
            serde_json::from_slice(&std::fs::read(dir.root().join("inspection_receipt.json"))?)
                .map_err(data)?;
        if receipt["status"] != "complete"
            || receipt["settings"] != serde_json::to_value(settings).map_err(data)?
            || receipt["inputs"] != json!(inputs(dir)?)
        {
            return Err(data("inspection receipt is incomplete or stale"));
        }
        verify_files(&receipt["artifacts"])?;
        let rendering: Value = serde_json::from_slice(&std::fs::read(
            dir.figures_dir().join("render_receipt.json"),
        )?)
        .map_err(data)?;
        validate_render_receipt(dir, &rendering)?;
        Ok(())
    };
    verify().is_ok()
}

pub fn render(
    dir: &ExperimentOutputDir,
    experiment: &ExperimentConfig,
    settings: &InspectConfig,
    cancelled: &dyn Fn() -> bool,
) -> Result<()> {
    if !settings.enabled {
        return Ok(());
    }
    check_cancel(cancelled)?;
    let receipt_path = dir.root().join("inspection_receipt.json");
    let mut receipt =
        json!({"schema_version":1,"status":"running","settings":settings,"artifacts":[]});
    write_json(&receipt_path, &receipt)?;
    let result = (|| -> Result<()> {
        let before = inputs(dir)?;
        receipt["inputs"] = json!(before);
        let config = dir.root().join("inspection_config.toml");
        let python = std::env::var_os("VIRTUAL_ENV")
            .map(PathBuf::from)
            .map(|p| p.join("bin/python"))
            .filter(|p| p.is_file())
            .unwrap_or_else(|| "python".into());
        let status = bekiper_post_analysis::pipeline::run_child(
            Command::new(python)
                .args(["-m", "bekiper_cli", "inspect", "plot"])
                .arg(dir.root())
                .arg("--config")
                .arg(&config)
                .arg("--output")
                .arg(dir.figures_dir())
                .env("MPLBACKEND", "Agg"),
            &dir.root().join("inspection.stdout.log"),
            &dir.root().join("inspection.stderr.log"),
            cancelled,
        )
        .map_err(data)?;
        if !status.success() {
            return Err(data(format!(
                "inspection renderer failed for {}: {}",
                experiment.name,
                std::fs::read_to_string(dir.root().join("inspection.stderr.log"))?
            )));
        }
        check_cancel(cancelled)?;
        let python_receipt = dir.figures_dir().join("render_receipt.json");
        let rendering: Value =
            serde_json::from_slice(&std::fs::read(&python_receipt)?).map_err(data)?;
        let mut artifacts = validate_render_receipt(dir, &rendering)?;
        receipt["effective_theme"] = rendering["effective_theme"].clone();
        receipt["selected_outputs"] = rendering["inventory"].clone();
        artifacts.push(fingerprint(&python_receipt)?);
        if settings.video_trials.contains(&experiment.name) {
            #[cfg(feature = "pose-video")]
            {
                use crate::inspection_video::{
                    RenderOptions, VideoTheme, render_from_directory_with_cancel,
                };
                let theme = VideoTheme::load_with_env(Some(&config), None, |_| None);
                receipt["effective_video_settings"] = serde_json::to_value(&theme).map_err(data)?;
                let mut video_outputs = Vec::new();
                for path in render_from_directory_with_cancel(
                    dir.root(),
                    &RenderOptions {
                        output: Some(dir.videos_dir().join("overview.mp4")),
                        config_path: Some(config),
                        ..Default::default()
                    },
                    &theme,
                    cancelled,
                )? {
                    artifacts.push(fingerprint(&path)?);
                    let metadata_path = path.with_extension("video.json");
                    let metadata: Value =
                        serde_json::from_slice(&std::fs::read(&metadata_path)?).map_err(data)?;
                    video_outputs.push(metadata);
                    artifacts.push(fingerprint(&metadata_path)?);
                }
                receipt["video_outputs"] = json!(video_outputs);
            }
            #[cfg(not(feature = "pose-video"))]
            return Err(data(
                "selected inspection videos require pose-video support",
            ));
        }
        check_cancel(cancelled)?;
        if before != inputs(dir)? {
            return Err(data("inspection inputs changed while rendering"));
        }
        receipt["artifacts"] = json!(artifacts);
        Ok(())
    })();
    match result {
        Ok(()) => {
            receipt["status"] = "complete".into();
            write_json(&receipt_path, &receipt)
        }
        Err(error) => {
            receipt["status"] = if cancelled() { "cancelled" } else { "failed" }.into();
            receipt["error"] = error.to_string().into();
            write_json(&receipt_path, &receipt)?;
            Err(error)
        }
    }
}
