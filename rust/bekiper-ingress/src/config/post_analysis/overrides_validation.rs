impl PostAnalysisConfig {
    pub fn apply_overrides(&mut self, options: &PostAnalysisOverrides) {
        if let Some(values) = &options.ymaze_primary_thresholds {
            self.ymaze_primary_thresholds = Some(values.clone());
        }
        if let Some(value) = &options.contexts {
            self.contexts = Some(value.clone());
        }
        if let Some(value) = options.render_workers {
            self.render_workers = Some(value);
        }
        if let Some(value) = options.primary_workers {
            self.primary_workers = value;
        }
        if let Some(value) = options.enabled {
            self.enabled = value;
        }
        if let Some(value) = &options.plan_path {
            self.plan_path = Some(value.clone());
        }
        if let Some(value) = &options.run_id {
            self.run_id = Some(value.clone());
        }
        if let Some(value) = options.figures_enabled {
            self.figures.enabled = value;
        }
        if let Some(value) = &options.figure_selection {
            self.figures.selection = Some(value.clone());
        }
        if let Some(value) = &options.figure_formats {
            self.figures.formats = value.clone();
        }
        if let Some(value) = options.figure_dpi {
            self.figures.dpi = value;
        }
        if let Some(value) = options.report_enabled {
            self.report.enabled = value;
        }
        if let Some(value) = &options.report_title {
            self.report.title = value.clone();
        }
        if let Some(value) = &options.report_compiler {
            self.report.compiler = value.clone();
        }
    }

    pub fn validate(&self) -> bekiper_core::error::Result<()> {
        let thresholds = self.ymaze_primary_thresholds.as_deref().unwrap_or(&[]);
        let unique: std::collections::BTreeSet<_> = thresholds.iter().collect();
        // 0% is a principal context only under the cross-line geometry; an
        // unresolved plan defers that check to post-analysis.
        let zero_ok = self.frozen_plan.as_ref().is_none_or(|plan| {
            plan["assays"]["ymaze"]["threshold_geometry"] == "imen_arm_start_crossline_v1"
        });
        if self.render_workers == Some(0)
            || self.contexts.as_ref().is_some_and(|c| {
                self.ymaze_primary_thresholds
                    .as_ref()
                    .is_some_and(|t| !t.is_empty())
                    || c.iter()
                        .map(|s| (&s.assay, &s.variant))
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        != c.len()
            })
        {
            return Err(bekiper_core::error::BekiperError::Config(
                "post-analysis render_workers must be positive; contexts must be distinct and replace the thresholds".into(),
            ));
        }
        if self.primary_workers == 0
            || unique.len() != thresholds.len()
            || thresholds
                .iter()
                .any(|t| ![25, 45, 65].contains(t) && !(zero_ok && *t == 0))
        {
            return Err(bekiper_core::error::BekiperError::Config(
                "primary contexts require distinct 25/45/65 thresholds and positive workers".into(),
            ));
        }
        use bekiper_core::error::BekiperError;
        self.ymaze_dwell_overrides
            .apply(Default::default())
            .validate()
            .map_err(BekiperError::Config)?;
        if !self.enabled {
            return Ok(());
        }
        let invalid = if self
            .frozen_plan
            .as_ref()
            .is_some_and(|plan| !plan.is_object())
        {
            Some("frozen_plan must be a JSON object")
        } else if self.run_id.as_ref().is_some_and(|id| !valid_run_id(id)) {
            Some("run_id must be a nonempty ASCII slug (letters, digits, '-' or '_')")
        } else if self.figures.enabled
            && (self.figures.dpi == 0
                || self.figures.formats.is_empty()
                || self
                    .figures
                    .formats
                    .iter()
                    .any(|format| !["png", "svg", "pdf"].contains(&format.as_str())))
        {
            Some("figures need positive DPI and png/svg/pdf formats")
        } else if self.report.enabled
            && !["latexmk", "lualatex"].contains(&self.report.compiler.as_str())
        {
            Some("report.compiler must be latexmk or lualatex")
        } else if self.plan_path.is_none()
            && self.frozen_plan.is_none()
            && self.figures.selection.as_ref().is_some_and(|selection| {
                selection.iter().any(|name| {
                    ![
                        "figure_oft",
                        "figure_ymaze",
                        "figure_nort",
                        "figure_oft_distributions",
                        "figure_ymaze_distributions",
                        "figure_nort_distributions",
                        "diagnostic_nort_states",
                    ]
                    .contains(&name.as_str())
                })
            })
        {
            Some("selected post-analysis figure requires a registered plan")
        } else {
            None
        };
        if let Some(message) = invalid {
            return Err(BekiperError::Config(format!("post-analysis: {message}")));
        }
        Ok(())
    }
}
