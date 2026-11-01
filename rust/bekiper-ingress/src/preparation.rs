//! Shared preparation inputs and producer receipts. Both staged workflows and
//! registry planning depend on ingress; neither owns a parallel contract.
use crate::config::{ExperimentConfig, GpuConfig, PipelineConfig};
use bekiper_core::{
    error::{BekiperError, Result},
    inspection::PerimeterSpec,
    scale::ScaleDerivationInfo,
    video::FrameTiming,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
};

pub const IMPLEMENTATION_VERSION: &str = env!("BEKIPER_PREPARATION_VERSION");
pub const PREREQUISITES: [&str; 5] = [
    "augmented",
    "pixel_perimeters",
    "perimeters",
    "scale",
    "frame_timing",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub path: PathBuf,
    pub length: u64,
    pub modified_ns: u128,
    pub device: u64,
    pub inode: u64,
}

impl SourceIdentity {
    pub fn capture(path: &Path) -> anyhow::Result<Self> {
        use std::os::unix::fs::MetadataExt;
        let path = path.canonicalize()?;
        let metadata = std::fs::metadata(&path)?;
        anyhow::ensure!(
            metadata.is_file(),
            "source is not a file: {}",
            path.display()
        );
        std::fs::File::open(&path)?;
        Ok(Self {
            path,
            length: metadata.len(),
            modified_ns: metadata
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| BekiperError::Data(e.to_string()))?
                .as_nanos(),
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }

    pub fn verify(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            Self::capture(&self.path)? == *self,
            "source changed since snapshot: {}",
            self.path.display()
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HashedFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

impl HashedFile {
    pub fn capture(path: &Path) -> anyhow::Result<Self> {
        let before = SourceIdentity::capture(path)?;
        let mut file = std::fs::File::open(&before.path)?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hash.update(&buffer[..read]);
        }
        before.verify()?;
        Ok(Self {
            path: before.path,
            bytes: before.length,
            sha256: format!("{:x}", hash.finalize()),
        })
    }

    pub fn verify(&self) -> anyhow::Result<()> {
        if self.bytes == 0 || Self::capture(&self.path)? != *self {
            anyhow::bail!("reuse evidence bytes differ: {}", self.path.display());
        }
        Ok(())
    }
}

/// Captured preparation-only values; hashing never consults the filesystem or
/// caller's working directory. The original producer provides these inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparationInputs {
    pub schema_version: u32,
    pub implementation_version: String,
    pub pipeline: PipelineConfig,
    pub gpu: Option<GpuConfig>,
    pub experiment: ExperimentConfig,
    pub frame_timing: Option<FrameTiming>,
    pub video_resolution: Option<(u32, u32)>,
    pub timestamps_file: Option<PathBuf>,
    pub perimeters: Vec<PerimeterSpec>,
    pub scale: Option<ScaleDerivationInfo>,
    pub sources: Vec<SourceIdentity>,
    pub absent_sidecars: Vec<PathBuf>,
    #[serde(default)]
    pub model_sha256: BTreeMap<PathBuf, String>,
}

impl PreparationInputs {
    pub fn fingerprint(&self) -> Result<String> {
        if !matches!(self.schema_version, 1 | 2) || self.implementation_version.is_empty() {
            return Err(BekiperError::Data(
                "unsupported preparation schema or empty implementation identity".into(),
            ));
        }
        let mut experiment = serde_json::to_value(&self.experiment)
            .map_err(|e| BekiperError::Data(e.to_string()))?;
        let object = experiment
            .as_object_mut()
            .ok_or_else(|| BekiperError::Data("experiment must be an object".into()))?;
        object.remove("name");
        object.remove("task_params");
        let mut sources = self.sources.clone();
        sources.sort_by(|a, b| a.path.cmp(&b.path));
        let mut absent = self.absent_sidecars.clone();
        absent.sort();
        let models = self
            .experiment
            .pose_model
            .as_ref()
            .map(|model| -> Result<_> {
                let mut paths = vec![&model.path];
                if let bekiper_core::types::PoseModelPreset::Custom(path) = &model.spec {
                    paths.push(path);
                }
                paths
                    .into_iter()
                    .map(|path| {
                        self.sources
                            .iter()
                            .find(|s| &s.path == path)
                            .cloned()
                            .ok_or_else(|| {
                                BekiperError::Data("model has no captured identity".into())
                            })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let mut value = serde_json::json!({"pipeline": self.pipeline, "experiment": experiment,
            "frame_timing": self.frame_timing, "video_resolution": self.video_resolution,
            "timestamps_file": self.timestamps_file, "perimeters": self.perimeters,
            "scale": self.scale, "sources": sources, "absent_sidecars": absent, "model_sources": models});
        let bytes = if self.schema_version == 1 {
            value["source_version"] = self.implementation_version.clone().into();
            serde_json::to_vec(&value).map_err(|e| BekiperError::Data(e.to_string()))?
        } else {
            value["implementation_version"] = self.implementation_version.clone().into();
            value["model_sha256"] = serde_json::to_value(&self.model_sha256)
                .map_err(|e| BekiperError::Data(e.to_string()))?;
            value["gpu"] =
                serde_json::to_value(&self.gpu).map_err(|e| BekiperError::Data(e.to_string()))?;
            bekiper_core::handoff::to_string(&value)?.into_bytes()
        };
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReuseTrial {
    pub analysis_id: String,
    pub preparation_fingerprint: String,
    pub prerequisites: BTreeMap<String, HashedFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requalification {
    pub target_snapshot_fingerprint: String,
    pub qualification_source_version: String,
    pub preparation_version: String,
    pub producer_snapshot: HashedFile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReuseReceipt {
    pub schema_version: u32,
    pub dataset_id: String,
    pub producer_run_id: String,
    pub producer_source_version: String,
    pub completed: bool,
    pub completion_evidence: HashedFile,
    pub trials: Vec<ReuseTrial>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requalification: Option<Requalification>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerCompletion {
    pub schema_version: u32,
    pub phase: String,
    pub dataset_id: String,
    pub producer_run_id: String,
    pub producer_source_version: String,
    pub preparation_version: String,
    pub producer_snapshot: HashedFile,
    pub upstream_trials: Vec<ReuseTrial>,
}

/// Fresh copies retain the original producer and artifact identities. Copying
/// does not turn this target into a new preparation producer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StagedTrial {
    pub producer_run_id: String,
    pub producer_source_version: String,
    pub original: ReuseTrial,
    pub copied: ReuseTrial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StagedPreparation {
    pub schema_version: u32,
    pub dataset_id: String,
    pub target_snapshot_fingerprint: String,
    pub target_source_version: String,
    pub preparation_version: String,
    pub root: PathBuf,
    pub output_root: PathBuf,
    pub original_snapshot: HashedFile,
    pub retained_snapshot: HashedFile,
    pub rebased_config: HashedFile,
    pub reuse_receipts: Vec<HashedFile>,
    pub trials: Vec<StagedTrial>,
    pub run_admitted: bool,
}

/// The same layout is used by direct producers and the staged consumer.
pub fn artifact_paths(dir: &crate::stage::ExperimentOutputDir) -> BTreeMap<String, PathBuf> {
    [
        ("augmented", dir.augmented_parquet()),
        ("pixel_perimeters", dir.pixel_perimeters_toml()),
        ("perimeters", dir.perimeters_toml()),
        ("scale", dir.scale_toml()),
        ("frame_timing", dir.frame_timing_toml()),
    ]
    .into_iter()
    .map(|(key, path)| (key.into(), path))
    .collect()
}
