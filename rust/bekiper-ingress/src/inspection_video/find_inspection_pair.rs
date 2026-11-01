fn find_inspection_pair(dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let mut tomls = Vec::new();
    let mut parquets = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
            if name.ends_with("_inspection.toml") {
                tomls.push(path);
            } else if name.ends_with("_evaluated.parquet") {
                parquets.push(path);
            }
        }
    }

    tomls.sort();
    parquets.sort();

    let toml_path = tomls.into_iter().next().ok_or_else(|| {
        BekiperError::Data(format!("No *_inspection.toml found in {}", dir.display()))
    })?;
    let parquet_path = parquets.into_iter().next().ok_or_else(|| {
        BekiperError::Data(format!("No *_evaluated.parquet found in {}", dir.display()))
    })?;

    Ok((toml_path, parquet_path))
}

fn sanitize_for_filename(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn map_codec(codec: &str) -> &str {
    match codec {
        "h264" => "libx264",
        other => other,
    }
}

fn encoder_supports_preset(codec: &str) -> bool {
    matches!(
        codec,
        "libx264"
            | "libx265"
            | "h264_nvenc"
            | "hevc_nvenc"
            | "av1_nvenc"
            | "h264_amf"
            | "hevc_amf"
            | "av1_amf"
    )
}

fn encoder_supports_crf(codec: &str) -> bool {
    matches!(
        codec,
        "libx264" | "libx265" | "libsvtav1" | "libaom-av1" | "libvpx-vp9"
    )
}

/// Largest CRF the encoder accepts (x264/x265 stop at 51, the rest at 63).
fn crf_max(codec: &str) -> u8 {
    if matches!(codec, "libx264" | "libx265") {
        51
    } else {
        63
    }
}

/// `BEKIPER_VIDEO_CRF` through `lookup`, applied only to CRF codecs so a
/// globally exported value never breaks encoders without a CRF (nvenc, amf).
pub fn crf_from_env(codec: &str, lookup: impl Fn(&str) -> Option<String>) -> Result<Option<u8>> {
    if !encoder_supports_crf(map_codec(codec)) {
        return Ok(None);
    }
    lookup("BEKIPER_VIDEO_CRF")
        .map(|value| {
            value.trim().parse().map_err(|_| {
                BekiperError::Config(format!("BEKIPER_VIDEO_CRF must be 0..=255, got '{value}'"))
            })
        })
        .transpose()
}

fn apply_theme_table(theme: &mut VideoTheme, table: &toml::map::Map<String, toml::Value>) {
    if let Some(value) = table.get("ffmpeg_preset").and_then(|value| value.as_str()) {
        theme.ffmpeg_preset = value.to_string();
    }
    if let Some(value) = table
        .get("video_point_color")
        .and_then(|value| value.as_str())
    {
        theme.video_point_color = value.to_string();
    }
    if let Some(value) = table
        .get("video_true_color")
        .and_then(|value| value.as_str())
    {
        theme.video_true_color = value.to_string();
    }
    if let Some(value) = table
        .get("video_false_color")
        .and_then(|value| value.as_str())
    {
        theme.video_false_color = value.to_string();
    }
    if let Some(value) = table
        .get("video_perimeter_color")
        .and_then(|value| value.as_str())
    {
        theme.video_perimeter_color = value.to_string();
    }
    if let Some(value) = table.get("linewidth").and_then(|value| value.as_float()) {
        theme.linewidth = value;
    }
    if let Some(value) = table.get("quiver_alpha").and_then(|value| value.as_float()) {
        theme.quiver_alpha = value as f32;
    }
}

fn apply_env_overrides<F>(theme: &mut VideoTheme, env_lookup: F)
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(value) = env_lookup("BEKIPER_PLOT_FFMPEG_PRESET") {
        theme.ffmpeg_preset = value;
    }
    if let Some(value) = env_lookup("BEKIPER_PLOT_VIDEO_POINT_COLOR") {
        theme.video_point_color = value;
    }
    if let Some(value) = env_lookup("BEKIPER_PLOT_VIDEO_TRUE_COLOR") {
        theme.video_true_color = value;
    }
    if let Some(value) = env_lookup("BEKIPER_PLOT_VIDEO_FALSE_COLOR") {
        theme.video_false_color = value;
    }
    if let Some(value) = env_lookup("BEKIPER_PLOT_VIDEO_PERIMETER_COLOR") {
        theme.video_perimeter_color = value;
    }
    if let Some(value) =
        env_lookup("BEKIPER_PLOT_LINEWIDTH").and_then(|value| value.parse::<f64>().ok())
    {
        theme.linewidth = value;
    }
    if let Some(value) =
        env_lookup("BEKIPER_PLOT_QUIVER_ALPHA").and_then(|value| value.parse::<f32>().ok())
    {
        theme.quiver_alpha = value;
    }
}

#[cfg(test)]
#[path = "find_inspection_pair/tests.rs"]
mod tests;
