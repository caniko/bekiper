use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Reviewer package settings (`[review]`), read by `bekiper review package`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReviewConfig {
    /// Parallel composite renderers. CLI `--workers` and `BEKIPER_REVIEW_WORKERS`
    /// win; unset means `min(available cores, 8)`.
    #[serde(default)]
    pub workers: Option<usize>,
    /// Criteria endpoints shown first per assay (`oft`, `nort`, `ymaze`); every
    /// other recorded endpoint goes to the appendix. An absent list means the descriptor's default criteria; an empty list means no criteria list.
    #[serde(default)]
    pub primary_endpoints: BTreeMap<String, Vec<String>>,
    /// Markdown file (relative to the config) whose dated headings form the
    /// Parameters page decision log. File-only.
    #[serde(default)]
    pub requirements: Option<std::path::PathBuf>,
    /// Markdown file (relative to the config) rendered as the Exploratory
    /// research-extensions page; `Figure: <id>` lines place the figures. File-only.
    #[serde(default)]
    pub extensions: Option<std::path::PathBuf>,
    /// Overview page title; default `Reviewer package`. File-only.
    #[serde(default)]
    pub title: Option<String>,
    /// Per-experiment presentation overrides (`[review.presentation.<experiment>]`,
    /// keyed by assay key or descriptor id): `title`, `short_name`, `steps`,
    /// `principal`, `endpoints`, `hide`. `primary_endpoints` fills `endpoints`
    /// where unset. File-only.
    #[serde(default)]
    pub presentation: bekiper_core::experiment::PresentationOverrides,
}

impl ReviewConfig {
    /// Presentation overrides with `primary_endpoints` folded in as the endpoint order.
    pub fn presentation_overrides(&self) -> bekiper_core::experiment::PresentationOverrides {
        let mut overrides = self.presentation.clone();
        for (experiment, endpoints) in &self.primary_endpoints {
            // Fold into the entry the resolver reads: assay key first, then descriptor id.
            let key = bekiper_core::experiment::descriptor(experiment)
                .into_iter()
                .flat_map(|d| [d.key.clone(), d.id.clone()])
                .find(|k| overrides.experiments.contains_key(k))
                .unwrap_or_else(|| experiment.clone());
            let entry = overrides.experiments.entry(key).or_default();
            if entry.endpoints.is_none() {
                entry.endpoints = Some(endpoints.clone());
            }
        }
        overrides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_section_parses_and_defaults_to_empty() {
        #[derive(Deserialize)]
        struct File {
            #[serde(default)]
            review: ReviewConfig,
        }
        let parsed: File = toml::from_str(
            "[review]\nworkers = 6\nextensions = \"research_extensions.md\"\n[review.primary_endpoints]\nnort = [\"DI\", \"pct_novel\"]\n",
        )
        .unwrap();
        assert_eq!(parsed.review.workers, Some(6));
        assert_eq!(
            parsed.review.extensions,
            Some(std::path::PathBuf::from("research_extensions.md"))
        );
        assert_eq!(parsed.review.primary_endpoints["nort"], ["DI", "pct_novel"]);
        let empty: File = toml::from_str("").unwrap();
        assert_eq!(empty.review, ReviewConfig::default());
    }

    #[test]
    fn presentation_overrides_fold_primary_endpoints() {
        let review: ReviewConfig = toml::from_str(
            "title = \"Neo/Imen reviewer package\"\n[primary_endpoints]\nnort = [\"DI\"]\noft = [\"center_time_pct\"]\n[presentation.nort]\nendpoints = [\"pct_novel\"]\nsteps = { t1 = \"Familiarisation\" }\n",
        )
        .unwrap();
        let overrides = review.presentation_overrides();
        assert_eq!(
            overrides.experiments["nort"].endpoints.as_deref(),
            Some(&["pct_novel".to_string()][..])
        );
        assert_eq!(
            overrides.experiments["oft"].endpoints.as_deref(),
            Some(&["center_time_pct".to_string()][..])
        );
        assert_eq!(overrides.experiments["nort"].steps["t1"], "Familiarisation");
        assert!(toml::from_str::<ReviewConfig>("[presentation.nort]\ncolour = 1\n").is_err());
    }

    #[test]
    fn primary_endpoints_fold_into_the_entry_the_resolver_reads() {
        let review: ReviewConfig = toml::from_str(
            "[primary_endpoints]\nnort = [\"DI\"]\n[presentation.nort]\ntitle = \"T\"\n",
        )
        .unwrap();
        let o = review.presentation_overrides();
        assert_eq!(o.experiments.len(), 1);
        assert_eq!(
            o.experiments["nort"].endpoints.as_deref(),
            Some(&["DI".to_string()][..])
        );
        let review: ReviewConfig = toml::from_str(
            "[primary_endpoints]\nnort = [\"DI\"]\n[presentation.object_recognition]\ntitle = \"T\"\n",
        )
        .unwrap();
        let o = review.presentation_overrides();
        assert_eq!(o.experiments.len(), 1);
        assert!(o.experiments["object_recognition"].endpoints.is_some());
    }
}
