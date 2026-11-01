//! Metadata admission decisions shared by execution and read-only population audits.
use super::{IngressConfig, MissingMetadataPolicy};
use crate::aggregate::metadata::{animal_id_from_experiment, load_aggregate_metadata};
use bekiper_core::error::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PopulationDisposition {
    Admitted,
    Excluded,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PopulationDecision {
    pub analysis_id: String,
    pub animal_id: Option<u32>,
    pub disposition: PopulationDisposition,
    pub reason: Option<String>,
}

impl IngressConfig {
    pub(super) fn metadata_population(&self, base_dir: &Path) -> Result<Vec<PopulationDecision>> {
        let policy = self
            .aggregate
            .metadata
            .as_ref()
            .map(|cfg| cfg.missing_policy);
        // Preserve AllowBlanks' execution semantics: it does not require loading
        // a workbook at prefilter time. The ledger separately identifies its bytes.
        let metadata = self
            .aggregate
            .metadata
            .as_ref()
            .filter(|cfg| cfg.missing_policy != MissingMetadataPolicy::AllowBlanks)
            .map(|cfg| load_aggregate_metadata(base_dir, cfg))
            .transpose()?;
        Ok(self
            .experiments
            .iter()
            .map(|experiment| {
                let animal_id = animal_id_from_experiment(&experiment.name);
                let missing = metadata
                    .as_ref()
                    .is_some_and(|table| animal_id.is_none_or(|id| !table.rows.contains_key(&id)));
                PopulationDecision {
                    analysis_id: experiment.name.clone(),
                    animal_id,
                    disposition: match (missing, policy) {
                        (true, Some(MissingMetadataPolicy::ExcludeMissing)) => {
                            PopulationDisposition::Excluded
                        }
                        (true, Some(MissingMetadataPolicy::Fail)) => PopulationDisposition::Blocked,
                        _ => PopulationDisposition::Admitted,
                    },
                    reason: missing.then(|| {
                        if animal_id.is_some() {
                            "missing_animal_metadata".into()
                        } else {
                            "unparseable_animal_identity".into()
                        }
                    }),
                }
            })
            .collect())
    }
}
