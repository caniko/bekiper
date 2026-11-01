impl PostAnalysisConfig {
    /// Overlay process environment variables on top of this config.
    pub fn apply_env_overrides(&mut self) {
        self.apply_env_overrides_from(|k| std::env::var(k).ok());
    }

    /// Overlay environment-variable settings using the supplied lookup closure.
    /// Presentation values are logged and ignored when unparseable. Primary
    /// selection/worker errors remain invalid until final constructor validation.
    ///
    /// Supported keys:
    /// - `BEKIPER_YMAZE_PRIMARY_THRESHOLDS` — comma-separated 25/45/65; empty disables fan-out.
    /// - `BEKIPER_PRIMARY_ANALYSIS_WORKERS` — positive concurrent-context cap.
    /// - `BEKIPER_POST_ANALYSIS_DISABLED` — when present and truthy
    ///   (`1`/`true`/`yes`/`on`, case-insensitive), sets `enabled = false`.
    /// - `BEKIPER_POST_ANALYSIS_PLAN_PATH` — sets `plan_path`.
    /// - `BEKIPER_POST_ANALYSIS_RUN_ID` — sets `run_id`.
    /// - `BEKIPER_POST_ANALYSIS_FIGURES_ENABLED` — sets `figures.enabled`.
    /// - `BEKIPER_POST_ANALYSIS_FIGURES_SELECTION` — comma-separated names.
    /// - `BEKIPER_POST_ANALYSIS_FIGURES_DPI` — sets `figures.dpi`.
    /// - `BEKIPER_POST_ANALYSIS_FIGURES_FORMATS` — comma-separated list for `figures.formats`.
    /// - `BEKIPER_POST_ANALYSIS_REPORT_ENABLED` — sets `report.enabled`.
    /// - `BEKIPER_POST_ANALYSIS_REPORT_TITLE` — sets `report.title`.
    /// - `BEKIPER_POST_ANALYSIS_REPORT_COMPILER` — sets `report.compiler`.
    pub fn apply_env_overrides_from<F>(&mut self, lookup: F)
    where
        F: Fn(&str) -> Option<String>,
    {
        if let Some(raw) = lookup("BEKIPER_YMAZE_PRIMARY_THRESHOLDS") {
            self.ymaze_primary_thresholds = Some(if raw.trim().is_empty() {
                Vec::new()
            } else {
                raw.split(',')
                    .map(|s| s.trim().parse::<u8>().unwrap_or(0))
                    .collect()
            });
        }
        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_RENDER_WORKERS") {
            self.render_workers = Some(raw.trim().parse::<usize>().unwrap_or(0));
        }
        if let Some(raw) = lookup("BEKIPER_PRIMARY_ANALYSIS_WORKERS") {
            self.primary_workers = raw.parse::<usize>().unwrap_or(0);
        }
        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_DISABLED") {
            let trimmed = raw.trim().to_ascii_lowercase();
            match trimmed.as_str() {
                "1" | "true" | "yes" | "on" => {
                    tracing::info!("BEKIPER_POST_ANALYSIS_DISABLED override applied");
                    self.enabled = false;
                }
                "0" | "false" | "no" | "off" => {
                    tracing::info!("BEKIPER_POST_ANALYSIS_DISABLED=false override applied");
                    self.enabled = true;
                }
                "" => tracing::warn!("BEKIPER_POST_ANALYSIS_DISABLED is empty, ignoring"),
                _ => tracing::warn!(
                    value = %raw,
                    "BEKIPER_POST_ANALYSIS_DISABLED is not a boolean flag, ignoring"
                ),
            }
        }

        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_PLAN_PATH") {
            if raw.trim().is_empty() {
                tracing::warn!("BEKIPER_POST_ANALYSIS_PLAN_PATH is empty, ignoring");
            } else {
                tracing::info!(plan_path = %raw, "BEKIPER_POST_ANALYSIS_PLAN_PATH override applied");
                self.plan_path = Some(PathBuf::from(raw.trim()));
            }
        }

        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_RUN_ID") {
            if raw.trim().is_empty() {
                tracing::warn!("BEKIPER_POST_ANALYSIS_RUN_ID is empty, ignoring");
            } else {
                tracing::info!(run_id = %raw, "BEKIPER_POST_ANALYSIS_RUN_ID override applied");
                self.run_id = Some(raw.trim().to_string());
            }
        }

        // Figures overrides
        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_FIGURES_ENABLED") {
            let trimmed = raw.trim().to_ascii_lowercase();
            match trimmed.as_str() {
                "1" | "true" | "yes" | "on" => {
                    tracing::info!("BEKIPER_POST_ANALYSIS_FIGURES_ENABLED override applied");
                    self.figures.enabled = true;
                }
                "0" | "false" | "no" | "off" => {
                    tracing::info!("BEKIPER_POST_ANALYSIS_FIGURES_ENABLED=false override applied");
                    self.figures.enabled = false;
                }
                "" => tracing::warn!("BEKIPER_POST_ANALYSIS_FIGURES_ENABLED is empty, ignoring"),
                _ => tracing::warn!(
                    value = %raw,
                    "BEKIPER_POST_ANALYSIS_FIGURES_ENABLED is not a boolean flag, ignoring"
                ),
            }
        }

        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_FIGURES_SELECTION") {
            self.figures.selection = Some(
                raw.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect(),
            );
        }

        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_FIGURES_DPI") {
            if let Ok(dpi) = raw.trim().parse::<u32>() {
                tracing::info!(
                    dpi = dpi,
                    "BEKIPER_POST_ANALYSIS_FIGURES_DPI override applied"
                );
                self.figures.dpi = dpi;
            } else {
                tracing::warn!(value = %raw, "BEKIPER_POST_ANALYSIS_FIGURES_DPI is not a valid u32, ignoring");
            }
        }

        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_FIGURES_FORMATS") {
            let formats: Vec<String> = raw
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if formats.is_empty() {
                tracing::warn!(
                    "BEKIPER_POST_ANALYSIS_FIGURES_FORMATS parsed to empty list, ignoring"
                );
            } else {
                tracing::info!(formats = ?formats, "BEKIPER_POST_ANALYSIS_FIGURES_FORMATS override applied");
                self.figures.formats = formats;
            }
        }

        // Report overrides
        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_REPORT_ENABLED") {
            let trimmed = raw.trim().to_ascii_lowercase();
            match trimmed.as_str() {
                "1" | "true" | "yes" | "on" => {
                    tracing::info!("BEKIPER_POST_ANALYSIS_REPORT_ENABLED override applied");
                    self.report.enabled = true;
                }
                "0" | "false" | "no" | "off" => {
                    tracing::info!("BEKIPER_POST_ANALYSIS_REPORT_ENABLED=false override applied");
                    self.report.enabled = false;
                }
                "" => tracing::warn!("BEKIPER_POST_ANALYSIS_REPORT_ENABLED is empty, ignoring"),
                _ => tracing::warn!(
                    value = %raw,
                    "BEKIPER_POST_ANALYSIS_REPORT_ENABLED is not a boolean flag, ignoring"
                ),
            }
        }

        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_REPORT_TITLE") {
            if raw.trim().is_empty() {
                tracing::warn!("BEKIPER_POST_ANALYSIS_REPORT_TITLE is empty, ignoring");
            } else {
                tracing::info!(title = %raw, "BEKIPER_POST_ANALYSIS_REPORT_TITLE override applied");
                self.report.title = raw.trim().to_string();
            }
        }

        if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_REPORT_COMPILER") {
            if raw.trim().is_empty() {
                tracing::warn!("BEKIPER_POST_ANALYSIS_REPORT_COMPILER is empty, ignoring");
            } else {
                tracing::info!(compiler = %raw, "BEKIPER_POST_ANALYSIS_REPORT_COMPILER override applied");
                self.report.compiler = raw.trim().to_string();
            }
        }
    }
}
