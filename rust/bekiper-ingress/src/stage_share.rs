//! Shared stage work for parameter variants of one recording.
//!
//! Variants expand to `{base}__{variant}` experiments that differ only in
//! `task_params`. The probe, detect, pose and augment stages do not read most
//! of those parameters, so each stage declares the inputs it reads and hashes
//! them into a fingerprint. Variants with equal fingerprints share one
//! execution and one checkpoint at
//! `checkpoint/<recording>/<stage>-<fp8>/`, sealed by a `fingerprint.toml`
//! next to the output. Each variant keeps the legacy result and checkpoint
//! layout: shared outputs are hard-linked into it and a `shared.toml` marker
//! names the checkpoints it consumed. A directory without the marker is a
//! legacy layout and keeps working unchanged.
//!
//! See `docs/src/pipeline/multi-parameter-runs.md` sections 2 and 3.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use bekiper_core::error::{BekiperError, Result};

use crate::config::{ExperimentConfig, IngressConfig};
use crate::stage::{ExperimentOutputDir, Stage, checkpoint_root};

/// Record written next to a shared stage output once the stage succeeded.
pub const FINGERPRINT_FILE: &str = "fingerprint.toml";
/// Marker in a variant's result directory naming the checkpoints it consumed.
pub const MARKER_FILE: &str = "shared.toml";

/// A unit of work that variants of one recording can share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Shared {
    /// Video resolution and frame timing probe.
    Probe,
    Detect,
    Pose,
    Augment,
}

impl Shared {
    pub const ALL: [Shared; 4] = [Shared::Probe, Shared::Detect, Shared::Pose, Shared::Augment];

    pub fn name(self) -> &'static str {
        match self {
            Shared::Probe => "probe",
            Shared::Detect => "detect",
            Shared::Pose => "pose",
            Shared::Augment => "augment",
        }
    }

    fn index(self) -> usize {
        self as usize
    }

    pub fn of_stage(stage: Stage) -> Option<Shared> {
        match stage {
            Stage::Detect => Some(Shared::Detect),
            Stage::Pose => Some(Shared::Pose),
            Stage::Augment => Some(Shared::Augment),
            _ => None,
        }
    }
}

/// `task_params` keys the augment stage reads (snout direction step,
/// `workflow/augment/redetectedkeypoint/plan.rs`). Every other `task_params`
/// key is read by behaviour only and never changes a stage fingerprint.
pub const AUGMENT_TASK_PARAM_KEYS: &[&str] = &["nose_coords", "head_coords"];

/// Experiment keys left out of every stage fingerprint: the variant identity
/// and the behaviour-only parameters. Everything else an experiment carries is
/// a declared input of each stage, so a newly added experiment field defaults
/// to invalidating checkpoints instead of silently sharing stale ones.
const NON_STAGE_EXPERIMENT_KEYS: &[&str] = &["name", "task_params"];

fn to_value<T: Serialize>(value: &T) -> Result<Value> {
    serde_json::to_value(value)
        .map_err(|e| BekiperError::Other(format!("stage fingerprint input: {e}")))
}

/// Canonical JSON: sorted keys at every depth, numbers in serde_json's
/// shortest round-trip form.
fn canonical(value: &Value) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (i, key) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String(key.clone()).to_string());
                    out.push(':');
                    write(&map[key], out);
                }
                out.push('}');
            }
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            other => out.push_str(&other.to_string()),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

/// Hash of one stage's declared inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageFingerprint {
    pub hex: String,
    /// Canonical JSON of the declared inputs, recorded for diagnosis.
    pub inputs: String,
}

impl StageFingerprint {
    fn new(kind: Shared, inputs: &Value) -> Self {
        let inputs = canonical(inputs);
        let mut hasher = Sha256::new();
        hasher.update(kind.name().as_bytes());
        hasher.update(b"\n");
        hasher.update(inputs.as_bytes());
        let hex = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Self { hex, inputs }
    }

    pub fn short(&self) -> &str {
        &self.hex[..8]
    }
}

/// Fingerprints of every shared stage of one experiment. Later stages hash
/// the fingerprints of the stages they consume, so a change propagates down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordingFingerprints([StageFingerprint; 4]);

impl RecordingFingerprints {
    pub fn new(experiment: &ExperimentConfig, config: &IngressConfig) -> Result<Self> {
        let Value::Object(mut inputs) = to_value(experiment)? else {
            return Err(BekiperError::Other(
                "experiment did not serialize to a table".into(),
            ));
        };
        for key in NON_STAGE_EXPERIMENT_KEYS {
            inputs.remove(*key);
        }
        // `serde(skip)` runtime fields that stages read.
        inputs.insert(
            "annotation_dir".into(),
            to_value(&experiment.annotation_dir)?,
        );
        inputs.insert("frame_image".into(), to_value(&experiment.frame_image)?);
        inputs.insert(
            "timestamps_file".into(),
            to_value(&experiment.timestamps_file)?,
        );
        inputs.insert("animal_id".into(), to_value(&experiment.animal_id)?);
        inputs.insert(
            "session_context".into(),
            to_value(&experiment.session_context)?,
        );
        let experiment_inputs = Value::Object(inputs);

        let probe = StageFingerprint::new(
            Shared::Probe,
            &serde_json::json!({
                "video": to_value(&experiment.video)?,
                "fps": to_value(&experiment.fps)?,
            }),
        );
        let detect = StageFingerprint::new(
            Shared::Detect,
            &serde_json::json!({ "experiment": experiment_inputs }),
        );
        let pose = StageFingerprint::new(
            Shared::Pose,
            &serde_json::json!({
                "experiment": experiment_inputs,
                "gpu": to_value(&config.gpu)?,
                "pose_model": to_value(&config.pose_model)?,
                "detect": detect.hex,
            }),
        );
        let task_params = to_value(&experiment.task_params)?;
        let augment_task_params: BTreeMap<&str, Value> = AUGMENT_TASK_PARAM_KEYS
            .iter()
            .map(|key| (*key, task_params.get(*key).cloned().unwrap_or(Value::Null)))
            .collect();
        let augment = StageFingerprint::new(
            Shared::Augment,
            &serde_json::json!({
                "experiment": experiment_inputs,
                "pipeline": to_value(&config.pipeline)?,
                "gpu": to_value(&config.gpu)?,
                "pose_model": to_value(&config.pose_model)?,
                "pose": pose.hex,
                "probe": probe.hex,
                "task_params": to_value(&augment_task_params)?,
            }),
        );
        Ok(Self([probe, detect, pose, augment]))
    }

    pub fn get(&self, kind: Shared) -> &StageFingerprint {
        &self.0[kind.index()]
    }
}

/// Where one experiment's shared stage outputs live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedLayout {
    recording: String,
    fingerprints: RecordingFingerprints,
}

#[derive(Serialize, Deserialize)]
struct FingerprintRecord {
    stage: String,
    fingerprint: String,
    inputs: String,
    /// Adopted from a legacy per-experiment checkpoint whose parameters were
    /// never recorded.
    #[serde(default)]
    adopted: bool,
}

impl SharedLayout {
    pub fn new(recording: &str, fingerprints: RecordingFingerprints) -> Self {
        Self {
            recording: recording.to_string(),
            fingerprints,
        }
    }

    pub fn recording(&self) -> &str {
        &self.recording
    }

    pub fn fingerprint(&self, kind: Shared) -> &StageFingerprint {
        self.fingerprints.get(kind)
    }

    /// `checkpoint/<recording>/<stage>-<fp8>/`.
    pub fn dir(&self, output_root: &Path, kind: Shared) -> PathBuf {
        checkpoint_root(output_root)
            .join(&self.recording)
            .join(format!(
                "{}-{}",
                kind.name(),
                self.fingerprint(kind).short()
            ))
    }

    /// Whether the stage completed under exactly this fingerprint.
    pub fn is_sealed(&self, output_root: &Path, kind: Shared) -> bool {
        std::fs::read_to_string(self.dir(output_root, kind).join(FINGERPRINT_FILE))
            .ok()
            .and_then(|text| toml::from_str::<FingerprintRecord>(&text).ok())
            .is_some_and(|record| record.fingerprint == self.fingerprint(kind).hex)
    }

    pub fn seal(&self, output_root: &Path, kind: Shared, adopted: bool) -> Result<()> {
        let dir = self.dir(output_root, kind);
        std::fs::create_dir_all(&dir)?;
        let fingerprint = self.fingerprint(kind);
        let text = toml::to_string_pretty(&FingerprintRecord {
            stage: kind.name().into(),
            fingerprint: fingerprint.hex.clone(),
            inputs: fingerprint.inputs.clone(),
            adopted,
        })
        .map_err(|e| BekiperError::Other(e.to_string()))?;
        std::fs::write(dir.join(FINGERPRINT_FILE), text)?;
        Ok(())
    }

    /// Forget the seal before re-running so a failed run is never "fresh".
    pub fn unseal(&self, output_root: &Path, kind: Shared) {
        let _ = std::fs::remove_file(self.dir(output_root, kind).join(FINGERPRINT_FILE));
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SharedMarker {
    pub recording: String,
    /// Consumed checkpoints by stage name.
    #[serde(default)]
    pub consumed: BTreeMap<String, ConsumedStage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConsumedStage {
    pub fingerprint: String,
    /// Checkpoint directory relative to the checkpoint root.
    pub checkpoint: String,
}

pub fn read_marker(result_dir: &Path) -> Option<SharedMarker> {
    toml::from_str(&std::fs::read_to_string(result_dir.join(MARKER_FILE)).ok()?).ok()
}

fn remove_path(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(path)?,
        Ok(_) => std::fs::remove_file(path)?,
        Err(_) => {}
    }
    Ok(())
}

/// Make `dst` a hard-linked (or, across filesystems, copied) image of `src`.
/// A missing `src` removes `dst`, so a stale artifact never outlives its source.
fn link_tree(src: &Path, dst: &Path) -> Result<()> {
    remove_path(dst)?;
    if src.is_dir() {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            link_tree(&entry.path(), &dst.join(entry.file_name()))?;
        }
    } else if src.exists() {
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if std::fs::hard_link(src, dst).is_err() {
            std::fs::copy(src, dst)?;
        }
    }
    Ok(())
}

/// Adopt a legacy per-experiment checkpoint into the shared store so existing
/// roots are not recomputed. The legacy parameters were never recorded, so the
/// adopted output is exactly as trustworthy as it was before.
pub fn adopt_legacy(
    shared: &ExperimentOutputDir,
    legacy_members: &[&ExperimentOutputDir],
    kind: Shared,
) -> Result<bool> {
    let Some(layout) = shared.shared_layout() else {
        return Ok(false);
    };
    if shared.shared_primary(kind).exists() {
        return Ok(false);
    }
    for member in legacy_members {
        // A directory that already consumed this stage from the shared store holds links, not a legacy checkpoint.
        let consumed =
            read_marker(member.root()).is_some_and(|m| m.consumed.contains_key(kind.name()));
        if consumed || !member.shared_primary(kind).exists() {
            continue;
        }
        for (src, dst) in member
            .shared_artifacts(kind)
            .iter()
            .zip(shared.shared_artifacts(kind))
        {
            link_tree(src, &dst)?;
        }
        layout.seal(shared.output_root(), kind, true)?;
        tracing::info!(
            stage = kind.name(),
            recording = layout.recording(),
            adopted_from = member.experiment_name(),
            "adopted legacy checkpoint into the shared store"
        );
        return Ok(true);
    }
    Ok(false)
}

/// Link the shared outputs of `kinds` into a variant's legacy layout and
/// update its `shared.toml`. If a consumed stage changed fingerprint since the
/// marker was written, the variant's behaviour and later outputs are stale and
/// are removed. Kinds whose shared output is absent are left untouched.
pub fn materialize(
    shared: &ExperimentOutputDir,
    member: &ExperimentOutputDir,
    kinds: &[Shared],
) -> Result<()> {
    let Some(layout) = shared.shared_layout() else {
        return Ok(());
    };
    let mut marker = read_marker(member.root()).unwrap_or_default();
    marker.recording = layout.recording().to_string();
    let mut stale_downstream = false;
    for &kind in kinds {
        if !shared.shared_primary(kind).exists() {
            continue;
        }
        for (src, dst) in shared
            .shared_artifacts(kind)
            .iter()
            .zip(member.shared_artifacts(kind))
        {
            link_tree(src, &dst)?;
        }
        let fingerprint = layout.fingerprint(kind).hex.clone();
        let previous = marker.consumed.insert(
            kind.name().into(),
            ConsumedStage {
                fingerprint: fingerprint.clone(),
                checkpoint: format!(
                    "{}/{}-{}",
                    layout.recording(),
                    kind.name(),
                    layout.fingerprint(kind).short()
                ),
            },
        );
        stale_downstream |= previous.is_some_and(|p| p.fingerprint != fingerprint);
    }
    member.ensure_dir()?;
    if stale_downstream {
        tracing::info!(
            experiment = member.experiment_name(),
            "shared upstream changed; removing stale behaviour outputs"
        );
        for path in member.purge_paths_from(Stage::Behaviour) {
            remove_path(&path)?;
        }
    }
    let text = toml::to_string_pretty(&marker).map_err(|e| BekiperError::Other(e.to_string()))?;
    std::fs::write(member.root().join(MARKER_FILE), text)?;
    Ok(())
}

/// Delete every shared checkpoint directory of a recording for the stages
/// that `purge_stage_outputs` clears (any fingerprint).
pub fn purge_shared(output_root: &Path, recording: &str, start: Stage) -> Result<usize> {
    let base = checkpoint_root(output_root).join(recording);
    let Ok(entries) = std::fs::read_dir(&base) else {
        return Ok(0);
    };
    let mut removed = 0;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let purge = Shared::ALL.iter().any(|kind| {
            name.starts_with(&format!("{}-", kind.name()))
                && match kind {
                    Shared::Probe => start <= Stage::Pose,
                    other => Shared::of_stage_rev(*other).is_some_and(|stage| stage >= start),
                }
        });
        if purge {
            remove_path(&entry.path())?;
            removed += 1;
        }
    }
    Ok(removed)
}

impl Shared {
    fn of_stage_rev(self) -> Option<Stage> {
        match self {
            Shared::Probe => None,
            Shared::Detect => Some(Stage::Detect),
            Shared::Pose => Some(Stage::Pose),
            Shared::Augment => Some(Stage::Augment),
        }
    }
}

/// Shared-stage execution for one run: the fingerprinted layout of every
/// variant experiment and one lock per `(recording, stage, fingerprint)`.
///
/// The first variant to reach a shared stage computes it while its siblings
/// block on the lock and then reuse the sealed output. Each variant then links
/// the output into its own legacy layout. Experiments without a
/// `recording_id` are not shared and keep the legacy flow.
pub struct SharedRun {
    layouts: Vec<Option<SharedLayout>>,
    legacy: Vec<ExperimentOutputDir>,
    siblings: BTreeMap<String, Vec<usize>>,
    // The flag records that this run already executed the stage (for `--force`).
    locks: Mutex<BTreeMap<(String, Shared, String), Arc<Mutex<bool>>>>,
}

impl SharedRun {
    pub fn new(config: &IngressConfig, legacy: Vec<ExperimentOutputDir>) -> Result<Self> {
        let mut layouts = Vec::with_capacity(legacy.len());
        let mut siblings: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, experiment) in config.experiments.iter().enumerate() {
            layouts.push(match &experiment.recording_id {
                Some(recording) => {
                    siblings.entry(recording.clone()).or_default().push(i);
                    Some(SharedLayout::new(
                        recording,
                        RecordingFingerprints::new(experiment, config)?,
                    ))
                }
                None => None,
            });
        }
        Ok(Self {
            layouts,
            legacy,
            siblings,
            locks: Default::default(),
        })
    }

    /// No sharing: every experiment keeps the legacy layout.
    pub fn legacy_only(legacy: Vec<ExperimentOutputDir>) -> Self {
        Self {
            layouts: vec![None; legacy.len()],
            legacy,
            siblings: BTreeMap::new(),
            locks: Default::default(),
        }
    }

    /// The legacy-layout directory of experiment `i`.
    pub fn legacy(&self, i: usize) -> &ExperimentOutputDir {
        &self.legacy[i]
    }

    pub fn is_shared(&self, i: usize) -> bool {
        self.layouts[i].is_some()
    }

    /// The legacy-layout directory of experiment `i` (plus the shared
    /// representative-frame cache when it has one).
    pub fn legacy_dir(&self, i: usize, experiment: &ExperimentConfig) -> ExperimentOutputDir {
        let dir = self.legacy[i].clone();
        match (
            &self.layouts[i],
            frame_cache_dir(dir.output_root(), experiment),
        ) {
            (Some(_), Some(cache)) => dir.with_frame_cache(cache),
            _ => dir,
        }
    }

    pub fn view(&self, i: usize) -> Option<ExperimentOutputDir> {
        self.layouts[i]
            .clone()
            .map(|layout| self.legacy[i].shared_view(layout))
    }

    /// Run shared stage `kind` for experiment `i` unless a sealed output with
    /// this fingerprint exists. `run` receives the directory to write into: the
    /// shared view, or the legacy layout outside a recording group. Afterwards
    /// the shared outputs are linked into the legacy layout of `i`.
    pub fn execute(
        &self,
        i: usize,
        kind: Shared,
        force: bool,
        is_fresh: impl FnOnce(&ExperimentOutputDir) -> bool,
        run: impl FnOnce(&ExperimentOutputDir) -> Result<()>,
    ) -> Result<()> {
        let Some(view) = self.view(i) else {
            let dir = &self.legacy[i];
            if force || !is_fresh(dir) {
                dir.ensure_dir()?;
                run(dir)?;
            }
            return Ok(());
        };
        let layout = self.layouts[i].as_ref().expect("shared view has a layout");
        let lock = {
            let mut locks = self.locks.lock().expect("shared lock map poisoned");
            locks
                .entry((
                    layout.recording().to_string(),
                    kind,
                    layout.fingerprint(kind).hex.clone(),
                ))
                .or_default()
                .clone()
        };
        let mut ran = lock.lock().expect("shared stage lock poisoned");
        // Only a sibling with the same fingerprint can have produced the output.
        let siblings: Vec<&ExperimentOutputDir> = self.siblings[layout.recording()]
            .iter()
            .filter(|&&j| {
                self.layouts[j]
                    .as_ref()
                    .is_some_and(|other| other.fingerprint(kind) == layout.fingerprint(kind))
            })
            .map(|&j| &self.legacy[j])
            .collect();
        adopt_legacy(&view, &siblings, kind)?;
        let stale = if force { !*ran } else { !is_fresh(&view) };
        if stale {
            *ran = true;
            view.ensure_dir()?;
            layout.unseal(view.output_root(), kind);
            run(&view)?;
            if view.shared_primary(kind).exists() {
                layout.seal(view.output_root(), kind, false)?;
            }
        } else {
            tracing::debug!(
                stage = kind.name(),
                recording = layout.recording(),
                "reusing shared output"
            );
        }
        drop(ran);
        materialize(&view, &self.legacy[i], &[kind])
    }

    /// Remove shared checkpoints for the purged stages of every recording.
    pub fn purge(&self, output_root: &Path, start: Stage) -> Result<()> {
        for recording in self.siblings.keys() {
            purge_shared(output_root, recording, start)?;
        }
        Ok(())
    }
}

/// Representative-frame PNGs shared by the variants of a recording, keyed by
/// the source video's identity so a replaced video never reuses frames.
pub(crate) fn frame_cache_dir(
    output_root: &Path,
    experiment: &ExperimentConfig,
) -> Option<PathBuf> {
    let recording = experiment.recording_id.as_deref()?;
    let video = experiment.video.as_ref()?;
    let meta = std::fs::metadata(video).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let mut hasher = Sha256::new();
    hasher.update(video.to_string_lossy().as_bytes());
    hasher.update(meta.len().to_le_bytes());
    hasher.update(mtime.to_le_bytes());
    let key: String = hasher.finalize()[..4]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Some(
        checkpoint_root(output_root)
            .join(recording)
            .join(format!("frames-{key}")),
    )
}

/// Hard-link a shared frame into a variant's `frames/` directory.
pub(crate) fn link_frame(src: &Path, dst: &Path) -> Result<()> {
    link_tree(src, dst)
}

#[cfg(test)]
mod tests;
