impl Default for VideoTheme {
    fn default() -> Self {
        Self {
            ffmpeg_preset: DEFAULT_FFMPEG_PRESET.into(),
            video_point_color: DEFAULT_VIDEO_POINT_COLOR.into(),
            video_true_color: DEFAULT_VIDEO_TRUE_COLOR.into(),
            video_false_color: DEFAULT_VIDEO_FALSE_COLOR.into(),
            video_perimeter_color: DEFAULT_VIDEO_PERIMETER_COLOR.into(),
            linewidth: DEFAULT_LINEWIDTH,
            quiver_alpha: DEFAULT_QUIVER_ALPHA,
        }
    }
}

impl VideoTheme {
    pub fn load(config_path: Option<&Path>, ffmpeg_preset_override: Option<&str>) -> Self {
        Self::load_with_env(config_path, ffmpeg_preset_override, |key| {
            std::env::var(key).ok()
        })
    }

    pub(crate) fn load_with_env<F>(
        config_path: Option<&Path>,
        ffmpeg_preset_override: Option<&str>,
        env_lookup: F,
    ) -> Self
    where
        F: Fn(&str) -> Option<String>,
    {
        let mut theme = Self::default();

        if let Some(path) = config_path
            && path.is_file()
        {
            match fs::read_to_string(path) {
                Ok(content) => match toml::from_str::<toml::Value>(&content) {
                    Ok(value) => {
                        if let Some(plot) = value
                            .get("inspect")
                            .and_then(|v| v.get("plot"))
                            .and_then(|v| v.as_table())
                        {
                            apply_theme_table(&mut theme, plot);
                        }
                    }
                    Err(err) => {
                        tracing::warn!(path = %path.display(), error = %err, "invalid theme TOML");
                    }
                },
                Err(err) => {
                    tracing::warn!(path = %path.display(), error = %err, "failed to read theme TOML");
                }
            }
        }

        apply_env_overrides(&mut theme, env_lookup);

        if let Some(override_preset) = ffmpeg_preset_override {
            theme.ffmpeg_preset = override_preset.to_string();
        }

        theme
    }
}
