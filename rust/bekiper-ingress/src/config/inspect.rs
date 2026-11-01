use serde::{Deserialize, Serialize};

/// Captured presentation settings. Python owns the plot-theme schema; retaining
/// its TOML table here preserves profiles/routing across sealed snapshots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InspectConfig {
    #[serde(default = "enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub plot: toml::Table,
    /// Exact recorded analysis IDs selected for full-source overview videos.
    #[serde(default)]
    pub video_trials: Vec<String>,
}

fn enabled() -> bool {
    true
}
impl Default for InspectConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            plot: Default::default(),
            video_trials: Vec::new(),
        }
    }
}

impl InspectConfig {
    /// Reserve a child runtime plus raster surfaces for multi-panel stories.
    /// Workers render one figure at a time and retain their lease until exit.
    pub fn workspace_bytes(&self) -> bekiper_core::error::Result<u64> {
        if !self.enabled {
            return Ok(0);
        }
        fn max_dpi(table: &toml::Table) -> u64 {
            table
                .iter()
                .map(|(key, value)| match value {
                    toml::Value::Integer(v) if key == "min_dpi" && *v > 0 => *v as u64,
                    toml::Value::Table(t) => max_dpi(t),
                    _ => 0,
                })
                .max()
                .unwrap_or(0)
        }
        let configured = max_dpi(&self.plot);
        let dpi = if configured == 0 { 300 } else { configured };
        // Neo/Imen's largest native multi-object stories reached 4.2 GiB RSS
        // at 300 DPI. Reserve 6 GiB for surfaces/artists plus the child runtime.
        dpi.checked_mul(dpi)
            .and_then(|v| v.checked_mul(6 * 1024 * 1024 * 1024))
            .map(|v| v.div_ceil(300 * 300))
            .and_then(|v| v.checked_add(256 * 1024 * 1024))
            .ok_or_else(|| {
                bekiper_core::error::BekiperError::Config("inspection workspace overflow".into())
            })
    }
    pub fn apply_env_overrides_from(&mut self, lookup: impl Fn(&str) -> Option<String>) {
        if let Some(disabled) = lookup("BEKIPER_INSPECT_DISABLED") {
            match disabled.as_str() {
                "1" | "true" => self.enabled = false,
                "0" | "false" => self.enabled = true,
                _ => {}
            }
        }
        if let Some(trials) = lookup("BEKIPER_INSPECT_VIDEO_TRIALS") {
            self.video_trials = trials
                .split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| t.trim().to_owned())
                .collect();
        }
        for field in [
            "ffmpeg_preset",
            "video_point_color",
            "video_true_color",
            "video_false_color",
            "video_perimeter_color",
        ] {
            if let Some(value) = lookup(&format!("BEKIPER_PLOT_{}", field.to_ascii_uppercase())) {
                self.plot.insert(field.into(), value.into());
            }
        }
        for field in ["linewidth", "quiver_alpha"] {
            if let Some(value) = lookup(&format!("BEKIPER_PLOT_{}", field.to_ascii_uppercase()))
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|v| v.is_finite())
            {
                self.plot.insert(field.into(), value.into());
            }
        }
        // Capture visual environment overlays once. Replays do not consult the
        // renderer's live environment. Values retain their TOML types.
        for field in [
            "true_colormap",
            "false_colormap",
            "palette",
            "valid_scatter_color",
            "invalid_scatter_color",
            "trajectory_gradient_colormap",
            "trajectory_render_mode",
            "density_colormap",
            "density_scale",
            "bar_primary_color",
            "bar_secondary_color",
            "hint_text_color",
            "quiver_single_cmap",
            "color_mode",
        ] {
            if let Some(value) = lookup(&format!("BEKIPER_PLOT_{}", field.to_ascii_uppercase())) {
                self.plot.insert(field.into(), value.into());
            }
        }
        for (field, alias) in [("min_dpi", "DPI"), ("fig_format", "FORMAT")] {
            if let Some(value) = lookup(&format!("BEKIPER_PLOT_{alias}"))
                .or_else(|| lookup(&format!("BEKIPER_PLOT_{}", field.to_ascii_uppercase())))
            {
                if field == "min_dpi" {
                    if let Ok(value) = value.parse::<i64>()
                        && value > 0
                    {
                        self.plot.insert(field.into(), value.into());
                    }
                } else {
                    self.plot.insert(field.into(), value.into());
                }
            }
        }
        for field in [
            "density_bins",
            "velocity_field_bins",
            "small_multiples_n",
            "quiver_stride",
            "quiver_binary_cmaps",
            "trajectory_gradient",
            "show_velocity_field",
            "fontsize",
            "title_fontsize",
            "annotation_fontsize",
            "bbox_alpha",
            "trajectory_jump_mad_factor",
            "density_alpha",
            "min_speed_filter",
            "timeline_alpha",
            "summary_figsize",
            "timeline_figsize",
            "coordinates_figsize",
            "state_color_columns",
        ] {
            if let Some(value) = lookup(&format!("BEKIPER_PLOT_{}", field.to_ascii_uppercase())) {
                let table = toml::from_str::<toml::Table>(&format!("value = {value}"));
                if let Ok(mut table) = table
                    && let Some(value) = table.remove("value")
                {
                    self.plot.insert(field.into(), value);
                } else if field == "quiver_binary_cmaps" {
                    // Same comma-separated form as `--quiver-binary-cmaps`.
                    let names = value.split(',').map(|name| name.trim().into()).collect();
                    self.plot.insert(field.into(), toml::Value::Array(names));
                }
            }
        }
    }

    pub fn validate_video_selection<'a>(
        &self,
        names: impl Iterator<Item = &'a str>,
    ) -> bekiper_core::error::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let names = names.collect::<std::collections::BTreeSet<_>>();
        for requested in &self.video_trials {
            if !names.contains(requested.as_str()) {
                return Err(bekiper_core::error::BekiperError::Config(format!(
                    "selected inspection video has no analysis in this run: {requested}"
                )));
            }
        }
        Ok(())
    }
}
