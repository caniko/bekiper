impl PostAnalysisConfig {
    /// Rendering is serial inside a cohort publication; the caller retains this
    /// workspace alongside the native results through child exit and exports.
    pub fn workspace_bytes(&self) -> bekiper_core::error::Result<u64> {
        self.render_workspace_bytes(self.frozen_plan.as_ref())
    }

    /// Upper bound for concurrently active contexts, including a captured plan
    /// before asset restoration. The project setting owns the worker cap.
    fn primary_concurrency(&self, plan: Option<&serde_json::Value>) -> usize {
        let count = if let Some(contexts) = self.contexts.as_ref().filter(|c| !c.is_empty()) {
            contexts.len()
        } else if let Some(thresholds) = &self.ymaze_primary_thresholds {
            thresholds.len().max(1)
        } else {
            plan.and_then(|p| {
                ["contexts", "ymaze_primary_thresholds"]
                    .iter()
                    .filter_map(|key| p["execution"][*key].as_array())
                    .find(|v| !v.is_empty())
            })
            .map_or(1, |v| v.len().max(1))
        };
        self.primary_workers.min(count)
    }

    /// Scored records/plotting data are private to each active primary context.
    /// Source input records remain shared and are already held by the caller.
    pub fn cohort_workspace_bytes(
        &self,
        retained_bytes: u64,
        plan: Option<&serde_json::Value>,
    ) -> bekiper_core::error::Result<u64> {
        let workers = if self.enabled {
            self.primary_concurrency(plan.or(self.frozen_plan.as_ref()))
        } else {
            1
        };
        // The original scored endpoints and three private HCAR1 statistical /
        // plotting views coexist; endpoint decoding and scoring remain shared.
        let policy_views = if self.enabled
            && plan
                .or(self.frozen_plan.as_ref())
                .is_some_and(|p| p["statistics"].get("hcar1_policy_plan").is_some())
        {
            4
        } else {
            1
        };
        retained_bytes
            .checked_mul(policy_views)
            .and_then(|bytes| bytes.checked_add(16 * 1024 * 1024))
            .and_then(|bytes| bytes.checked_mul(workers as u64))
            .and_then(|bytes| {
                bytes.checked_add(
                    self.render_workspace_bytes(plan.or(self.frozen_plan.as_ref()))
                        .ok()?,
                )
            })
            .ok_or_else(|| {
                bekiper_core::error::BekiperError::Config("cohort workspace overflow".into())
            })
    }

    fn render_workspace_bytes(
        &self,
        plan: Option<&serde_json::Value>,
    ) -> bekiper_core::error::Result<u64> {
        if !self.enabled {
            return Ok(0);
        }
        let settings = super::InspectConfig {
            enabled: self.figures.enabled || self.report.enabled,
            plot: toml::Table::from_iter([(
                "min_dpi".into(),
                toml::Value::Integer(i64::from(self.figures.dpi)),
            )]),
            video_trials: Vec::new(),
        };
        settings
            .workspace_bytes()?
            .checked_mul(self.primary_concurrency(plan) as u64)
            .ok_or_else(|| {
                bekiper_core::error::BekiperError::Config(
                    "primary export workspace overflow".into(),
                )
            })
    }
}
