//! Read-only pre-filter population, with byte identities for the admission inputs.
use super::*;
use crate::config::{MissingMetadataPolicy, PopulationDecision, PopulationDisposition};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PopulationFile {
    pub path: PathBuf,
    pub sha256: String,
}

impl PopulationFile {
    fn capture(path: PathBuf) -> Result<Self> {
        let sha256 = format!("{:x}", Sha256::digest(std::fs::read(&path)?));
        Ok(Self { path, sha256 })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PopulationRow {
    #[serde(flatten)]
    pub decision: PopulationDecision,
    /// Deduplicates variants by video path, or by pose-source paths if video-free.
    /// This is an inventory key, not an authoritative registry recording binding.
    pub recording_key: String,
    pub task_type: TaskType,
    pub video: Option<PathBuf>,
    pub pose_sources: Vec<PathBuf>,
    #[serde(default)]
    pub session_context: serde_json::Value,
    #[serde(default)]
    pub task_params: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PopulationLedger {
    pub schema_version: u32,
    pub config: PopulationFile,
    pub metadata_workbook: Option<PopulationFile>,
    pub metadata_policy: Option<MissingMetadataPolicy>,
    pub data_root: PathBuf,
    pub pre_filter_analyses: usize,
    pub admitted_analyses: usize,
    pub excluded_analyses: usize,
    pub blocked_analyses: usize,
    pub pre_filter_recordings: usize,
    pub admitted_recordings: usize,
    pub rows: Vec<PopulationRow>,
    pub fingerprint: String,
}

impl PopulationLedger {
    fn digest(&self) -> Result<String> {
        let mut value =
            serde_json::to_value(self).map_err(|e| BekiperError::Data(e.to_string()))?;
        value
            .as_object_mut()
            .ok_or_else(|| BekiperError::Data("population must be an object".into()))?
            .remove("fingerprint");
        let bytes = if self.schema_version == 1 {
            serde_json::to_vec(&value).map_err(|e| BekiperError::Data(e.to_string()))?
        } else {
            bekiper_core::handoff::to_string(&value)?.into_bytes()
        };
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn validate(&self) -> Result<()> {
        let analyses: BTreeSet<_> = self.rows.iter().map(|r| &r.decision.analysis_id).collect();
        let recordings: BTreeSet<_> = self.rows.iter().map(|r| &r.recording_key).collect();
        let admitted: Vec<_> = self
            .rows
            .iter()
            .filter(|r| r.decision.disposition == PopulationDisposition::Admitted)
            .collect();
        let admitted_recordings: BTreeSet<_> = admitted.iter().map(|r| &r.recording_key).collect();
        if !matches!(self.schema_version, 1 | 2)
            || self.fingerprint != self.digest()?
            || analyses.len() != self.rows.len()
            || self.pre_filter_analyses != self.rows.len()
            || self.admitted_analyses != admitted.len()
            || self.excluded_analyses
                != self
                    .rows
                    .iter()
                    .filter(|r| r.decision.disposition == PopulationDisposition::Excluded)
                    .count()
            || self.blocked_analyses
                != self
                    .rows
                    .iter()
                    .filter(|r| r.decision.disposition == PopulationDisposition::Blocked)
                    .count()
            || self.pre_filter_recordings != recordings.len()
            || self.admitted_recordings != admitted_recordings.len()
        {
            return Err(BekiperError::Data("inconsistent population ledger".into()));
        }
        Ok(())
    }
}

/// Expand without materializing artifacts or dropping excluded analyses. Large
/// video/pose files are named, not hashed; this is not a restorable source capture.
pub fn population_ledger_at(path: &Path, base: &Path) -> Result<PopulationLedger> {
    let base = if base.as_os_str().is_empty() {
        Path::new(".")
    } else {
        base
    };
    let base = base.canonicalize()?;
    let config_identity = PopulationFile::capture(path.canonicalize()?)?;
    let mut config = IngressConfig::from_file(path)?;
    let metadata_workbook = config
        .aggregate
        .metadata
        .as_ref()
        .map(|cfg| PopulationFile::capture(base.join(&cfg.path).canonicalize()?))
        .transpose()?;
    let decisions = config.expand_for_population(&base)?;
    let mut rows = Vec::with_capacity(decisions.len());
    for (experiment, decision) in config.experiments.iter().zip(decisions) {
        let video = experiment.video.as_ref().map(|path| base.join(path));
        let pose_sources: Vec<_> = experiment
            .data_files
            .iter()
            .map(|path| base.join(path))
            .collect();
        let recording_paths = video
            .clone()
            .map(|p| vec![p])
            .unwrap_or_else(|| pose_sources.clone());
        if recording_paths.is_empty() {
            return Err(BekiperError::Data(format!(
                "{} has no recording source",
                experiment.name
            )));
        }
        let recording_key = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&recording_paths)
                    .map_err(|e| BekiperError::Data(e.to_string()))?
            )
        );
        rows.push(PopulationRow {
            decision,
            recording_key,
            task_type: experiment.task_type,
            video,
            pose_sources,
            session_context: serde_json::to_value(&experiment.session_context)
                .map_err(|e| BekiperError::Data(e.to_string()))?,
            task_params: serde_json::to_value(&experiment.task_params)
                .map_err(|e| BekiperError::Data(e.to_string()))?,
        });
    }
    for identity in std::iter::once(&config_identity).chain(metadata_workbook.iter()) {
        if PopulationFile::capture(identity.path.clone())? != *identity {
            return Err(BekiperError::Data(
                "population inputs changed during expansion".into(),
            ));
        }
    }
    let recordings: BTreeSet<_> = rows.iter().map(|r| &r.recording_key).collect();
    let admitted_recordings: BTreeSet<_> = rows
        .iter()
        .filter(|r| r.decision.disposition == PopulationDisposition::Admitted)
        .map(|r| &r.recording_key)
        .collect();
    let mut ledger = PopulationLedger {
        schema_version: 2,
        config: config_identity,
        metadata_workbook,
        metadata_policy: config.aggregate.metadata.map(|cfg| cfg.missing_policy),
        data_root: base,
        pre_filter_analyses: rows.len(),
        admitted_analyses: rows
            .iter()
            .filter(|r| r.decision.disposition == PopulationDisposition::Admitted)
            .count(),
        excluded_analyses: rows
            .iter()
            .filter(|r| r.decision.disposition == PopulationDisposition::Excluded)
            .count(),
        blocked_analyses: rows
            .iter()
            .filter(|r| r.decision.disposition == PopulationDisposition::Blocked)
            .count(),
        pre_filter_recordings: recordings.len(),
        admitted_recordings: admitted_recordings.len(),
        rows,
        fingerprint: String::new(),
    };
    ledger.fingerprint = ledger.digest()?;
    ledger.validate()?;
    Ok(ledger)
}
