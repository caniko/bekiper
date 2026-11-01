impl PostAnalysisConfig {
    /// Resolve `plan_path` relative to the config file's parent directory.
    /// Returns `None` if `plan_path` is not set.
    pub fn resolved_plan_path(&self, config_parent: &Path) -> Option<PathBuf> {
        self.plan_path.as_ref().map(|p| {
            if p.is_absolute() {
                p.clone()
            } else {
                config_parent.join(p)
            }
        })
    }

    /// Returns the effective frozen plan: prefer `frozen_plan` (embedded),
    /// fall back to loading from `resolved_plan_path`, otherwise `None`.
    pub fn effective_frozen_plan(
        &self,
        config_parent: &Path,
    ) -> bekiper_core::error::Result<Option<serde_json::Value>> {
        if let Some(ref plan) = self.frozen_plan {
            return Ok(Some(plan.clone()));
        }
        if let Some(path) = self.resolved_plan_path(config_parent) {
            let bytes = std::fs::read(&path)?;
            let plan = if path
                .extension()
                .is_some_and(|extension| extension == "toml")
            {
                bekiper_core::handoff::from_str(std::str::from_utf8(&bytes).map_err(|error| {
                    bekiper_core::error::BekiperError::Config(error.to_string())
                })?)
            } else {
                serde_json::from_slice(&bytes)
                    .map_err(|error| bekiper_core::error::BekiperError::Config(error.to_string()))
            }
            .map_err(|error| {
                bekiper_core::error::BekiperError::Config(format!(
                    "invalid post-analysis plan {}: {error}",
                    path.display()
                ))
            })?;
            return Ok(Some(plan));
        }
        Ok(None)
    }

    /// Returns the effective frozen fingerprints: prefer `frozen_fingerprints`,
    /// fall back to loading from plan's `source_fingerprints` field if present.
    pub fn effective_frozen_fingerprints(
        &self,
        config_parent: &Path,
    ) -> bekiper_core::error::Result<Vec<serde_json::Value>> {
        if !self.frozen_fingerprints.is_empty() {
            return Ok(self.frozen_fingerprints.clone());
        }
        if let Some(plan) = self.effective_frozen_plan(config_parent)?
            && let Some(fingerprints) = plan.get("source_fingerprints").and_then(|v| v.as_array())
        {
            return Ok(fingerprints.clone());
        }
        Ok(Vec::new())
    }
}
