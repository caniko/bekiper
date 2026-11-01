//! Inspection videos for the reviewer package: one per recording (its principal
//! context, Y-maze: the 45 % variant), rendered from the finished run (no
//! re-analysis), keyed for resume and embedded on animal pages. The step's
//! descriptor picks the kind: object exploration gets the qualia video
//! (`--qualia`: detection fields and per-object channel timeline), the others
//! the scored overview (`--scored`).
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Mutex,
};

use bekiper_core::experiment::VideoKind;

use super::{
    Cancel, PackageArgs,
    index::{self, Recording},
    sha256_file,
};

/// Bump when the scored video layout or encoding settings change.
pub(crate) const VIDEO_VERSION: &str = "3";

/// Review embeds trade some sharpness for size: CRF and qualia upscale
/// (overridden by `BEKIPER_VIDEO_CRF` / `BEKIPER_VIDEO_UPSCALE`).
const REVIEW_CRF: &str = "30";
const REVIEW_UPSCALE: &str = "2";

/// Effective `(crf, upscale)` for review renders.
fn encoding() -> (String, String) {
    let var = |name, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
    (
        var("BEKIPER_VIDEO_CRF", REVIEW_CRF),
        var("BEKIPER_VIDEO_UPSCALE", REVIEW_UPSCALE),
    )
}

/// `OLD=NEW` prefix rebase for recorded source-video paths that moved.
pub(crate) fn rebase(recorded: &str, rule: Option<&str>) -> Result<PathBuf> {
    let Some(rule) = rule else {
        return Ok(PathBuf::from(recorded));
    };
    let (old, new) = rule
        .split_once('=')
        .context("--video-source must be OLD_PREFIX=NEW_PREFIX")?;
    Ok(match recorded.strip_prefix(old) {
        Some(rest) => PathBuf::from(format!("{new}{rest}")),
        None => PathBuf::from(recorded),
    })
}

/// Renderer flag and output stem of a step's video kind.
fn kind(video: VideoKind) -> (&'static str, &'static str) {
    match video {
        VideoKind::Qualia => ("--qualia", "qualia"),
        VideoKind::Scored => ("--scored", "scored"),
    }
}

fn key(trial_dir: &Path, source: &Path, mode: themekit::Mode) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(VIDEO_VERSION);
    hasher.update(mode.as_str());
    let (crf, upscale) = encoding();
    hasher.update(format!("crf={crf};upscale={upscale}"));
    for path in index::inputs(trial_dir)? {
        hasher.update(sha256_file(&path)?);
    }
    let meta =
        fs::metadata(source).with_context(|| format!("source video {}", source.display()))?;
    hasher.update(source.to_string_lossy().as_bytes());
    hasher.update(meta.len().to_le_bytes());
    if let Ok(modified) = meta.modified()
        && let Ok(since) = modified.duration_since(std::time::UNIX_EPOCH)
    {
        hasher.update(since.as_secs().to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn folder(recording: &Recording) -> String {
    recording
        .animal
        .as_deref()
        .map_or("unassigned".to_string(), index::slug)
}

struct Job {
    index: usize,
    trial_dir: PathBuf,
    source: PathBuf,
    key: String,
    out: PathBuf,
    relative: String,
    variant: Option<String>,
    video: VideoKind,
}

/// Render missing or stale videos in parallel and attach every video's sidecar
/// (or its error) to its recording. Returns the number rendered now.
pub(crate) fn render(
    args: &PackageArgs,
    recordings: &mut [Recording],
    mode: themekit::Mode,
    workers: usize,
    cancel: &Cancel,
) -> Result<usize> {
    // Videos of earlier runs (other recordings or animals) must not be sealed with this one.
    let root = args.output.join("videos");
    let folders: Vec<String> = recordings.iter().map(folder).collect();
    super::animals::prune_stale(&root, folders.iter().map(String::as_str));
    for dir in &folders {
        super::animals::prune_stale(
            &root.join(dir),
            recordings
                .iter()
                .filter(|r| folder(r) == *dir)
                .map(|r| r.id.as_str()),
        );
    }
    let mut jobs = Vec::new();
    for (i, recording) in recordings.iter_mut().enumerate() {
        let trial = recording.principal();
        let source = rebase(&trial.video, args.video_source.as_deref())?;
        if !source.is_file() {
            recording.video =
                Some(json!({"error": format!("source video missing: {}", source.display())}));
            continue;
        }
        let folder = folder(recording);
        let relative = format!("videos/{folder}/{}", recording.id);
        let out = args.output.join(&relative);
        let key = key(&trial.dir, &source, mode)?;
        let sidecar: Option<Value> = fs::read(out.with_extension("json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        let fresh = sidecar.as_ref().is_some_and(|s| {
            s["key"] == json!(key)
                && sha256_file(&out.with_extension("mp4")).ok().as_deref()
                    == s["mp4_sha256"].as_str()
        });
        if fresh && args.preview.is_none() {
            recording.video = sidecar;
            continue;
        }
        let variant = (recording.trials.len() > 1).then(|| trial.variant_label());
        jobs.push(Job {
            index: i,
            trial_dir: trial.dir.clone(),
            source,
            key,
            out,
            relative,
            variant,
            video: trial
                .placement
                .as_ref()
                .map_or_else(VideoKind::default, |p| p.video),
        });
    }
    let rendered = jobs.len();
    let results: Mutex<Vec<(usize, Value)>> = Mutex::new(Vec::new());
    let queue = Mutex::new(jobs.into_iter());
    let logs = args.output.join("logs/videos");
    fs::create_dir_all(&logs)?;
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1) {
            scope.spawn(|| {
                loop {
                    let job = queue.lock().expect("video queue").next();
                    let Some(job) = job else { break };
                    if cancel.cancelled() {
                        break;
                    }
                    let value = render_one(args, &job, mode, &logs, cancel)
                        .unwrap_or_else(|error| json!({"error": format!("{error:#}")}));
                    results
                        .lock()
                        .expect("video results")
                        .push((job.index, value));
                }
            });
        }
    });
    if cancel.cancelled() {
        bail!("review packaging cancelled");
    }
    for (index, value) in results.into_inner().expect("video results") {
        recordings[index].video = Some(value);
    }
    Ok(rendered)
}

fn render_one(
    args: &PackageArgs,
    job: &Job,
    mode: themekit::Mode,
    logs: &Path,
    cancel: &Cancel,
) -> Result<Value> {
    let name = job.relative.replace('/', "__");
    let staging = logs.join(&name);
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)?;
    let (flag, stem) = kind(job.video);
    let (crf, upscale) = encoding();
    let staged = staging.join(format!("{stem}.inspection.mp4"));
    let mut command = Command::new(args.video_renderer.as_deref().context("video renderer")?);
    command
        .arg("inspect")
        .arg("video")
        .arg(&job.trial_dir)
        .arg(flag)
        .arg("--crf")
        .arg(&crf)
        .arg("--source-video")
        .arg(&job.source)
        .arg("--output")
        .arg(&staged)
        .env("BEKIPER_PLOT_COLOR_MODE", mode.as_str());
    if flag == "--qualia" {
        command.arg("--upscale").arg(&upscale);
    }
    let config = job.trial_dir.join("inspection_config.toml");
    if config.is_file() {
        command.arg("--config").arg(&config);
    }
    let status = bekiper_post_analysis::pipeline::run_child(
        &mut command,
        &staging.join("stdout.log"),
        &staging.join("stderr.log"),
        &|| cancel.cancelled(),
    )?;
    if !status.success() {
        bail!("video renderer exited {status}; see logs/videos/{name}/stderr.log");
    }
    let receipt: Value = serde_json::from_slice(&fs::read(
        staging.join(format!("{stem}.inspection.video.json")),
    )?)
    .context("video receipt")?;
    if let Some(parent) = job.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(&staged, job.out.with_extension("mp4"))?;
    let poster = job.out.with_extension("poster.png");
    let has_poster = fs::rename(
        staging.join(format!("{stem}.inspection.poster.png")),
        &poster,
    )
    .is_ok();
    let (width, height) = (
        receipt["resolution"][0].clone(),
        receipt["resolution"][1].clone(),
    );
    let sidecar = json!({
        "key": job.key,
        "video_version": VIDEO_VERSION,
        "path": format!("{}.mp4", job.relative),
        "poster": has_poster.then(|| format!("{}.poster.png", job.relative)),
        "width": width,
        "height": height,
        "variant": job.variant,
        "rows": receipt["scored_rows"],
        "frames": receipt["frames_written"],
        "fps": receipt["encoded_fps"],
        "source_video": receipt["source_video"],
        "mp4_sha256": sha256_file(&job.out.with_extension("mp4"))?,
    });
    fs::write(
        job.out.with_extension("json"),
        serde_json::to_vec_pretty(&sidecar)?,
    )?;
    let _ = fs::remove_dir_all(&staging);
    Ok(sidecar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebase_replaces_only_the_recorded_prefix() {
        assert_eq!(
            rebase("/old/root/OFT/a.mp4", Some("/old/root=/new")).unwrap(),
            PathBuf::from("/new/OFT/a.mp4")
        );
        assert_eq!(
            rebase("/elsewhere/a.mp4", Some("/old/root=/new")).unwrap(),
            PathBuf::from("/elsewhere/a.mp4")
        );
        assert_eq!(rebase("/a.mp4", None).unwrap(), PathBuf::from("/a.mp4"));
        assert!(rebase("/a.mp4", Some("no-equals")).is_err());
    }

    #[test]
    fn nort_recordings_get_the_qualia_video() {
        let video = |key: &str| bekiper_core::experiment::descriptor(key).unwrap().steps[0].video;
        assert_eq!(kind(video("nort")), ("--qualia", "qualia"));
        assert_eq!(kind(video("oft")), ("--scored", "scored"));
        assert_eq!(kind(video("ymaze")), ("--scored", "scored"));
    }
}
