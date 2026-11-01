//! Reviewer package: one mega-composite WebP per trial, readable cohort pages,
//! an offline HTML index and a checksummed zip delivery.
//!
//! Rust discovers trials, joins the recorded post-analysis metrics, schedules
//! renderer workers, writes pages and seals the package; the packaged Python
//! plot backend draws figures. Nothing is rescored.
use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Subcommand};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

mod animals;
mod extensions;
mod html;
mod index;
mod parameters;
mod seal;
#[cfg(test)]
mod tests;
mod videos;

use index::Recording;

/// Must match `COMPOSITE_VERSION` in `bekiper_inspect/plot/composite.py`.
const COMPOSITE_VERSION: &str = "3";

#[derive(Args)]
pub struct ReviewArgs {
    #[command(subcommand)]
    command: ReviewCommand,
}

#[derive(Subcommand)]
enum ReviewCommand {
    /// Build (or resume) a reviewer package from a completed analysis run.
    Package(Box<PackageArgs>),
    /// Verify a package directory or its zip against SHA256SUMS.
    Verify { path: PathBuf },
}

#[derive(Args, Clone)]
pub struct PackageArgs {
    /// Completed analysis output root containing `data/<trial>` and `data/_post_analysis`.
    #[arg(long)]
    pub run: PathBuf,
    /// Package directory; reusable only for the same run (resume).
    #[arg(long)]
    pub output: PathBuf,
    /// Project configuration (`[review]`, task parameters shown to reviewers).
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Registered analysis plan; supplies group names (G1-G8 and factors).
    #[arg(long)]
    pub plan: Option<PathBuf>,
    /// Parallel renderers (default `[review].workers`, else min(cores, 8)).
    #[arg(long, env = "BEKIPER_REVIEW_WORKERS")]
    pub workers: Option<usize>,
    /// Restrict to these trial IDs (directory names under `data/`).
    #[arg(long)]
    pub trial: Vec<String>,
    /// Also write `<output>.zip` and `<output>.zip.sha256`.
    #[arg(long)]
    pub zip: bool,
    /// Preview one experiment by its assay key (`oft`, `nort`, `ymaze`, ...): a
    /// representative sample (min/median/max per session plus edge cases)
    /// re-rendered on every run, with that experiment's cohort figures over the
    /// full population. Reuses the finished run's results.
    #[arg(long)]
    pub preview: Option<String>,
    /// Preview picks per session stratum (3 = min, median, max).
    #[arg(long, default_value_t = 3)]
    pub per_stratum: usize,
    /// Colour mode of every figure and page: dark (night mode, the default) or
    /// light. Falls back to `[inspect.plot] color_mode` in `--config`.
    #[arg(long, env = "BEKIPER_PLOT_COLOR_MODE", value_parser = ["dark", "light"])]
    pub theme: Option<String>,
    /// Also render one scored inspection video per recording (Y-maze: the 45 %
    /// variant) and embed it on the animal pages; needs the source videos.
    #[arg(long)]
    pub videos: bool,
    /// Rebase recorded source-video paths that moved: `OLD_PREFIX=NEW_PREFIX`.
    #[arg(long, requires = "videos")]
    pub video_source: Option<String>,
    /// Plot backend executable.
    #[arg(long, default_value = "bekiper-inspect", hide = true)]
    pub renderer: PathBuf,
    /// Video renderer executable (default: this `bekiper` binary).
    #[arg(long, hide = true)]
    pub video_renderer: Option<PathBuf>,
    /// Override a `--config` key, e.g. `review.presentation.nort.title="NOR"` or
    /// `review.primary_endpoints.oft=["center_time_pct"]`. Repeatable; beats
    /// `BEKIPER_SET` (`;`-separated) and the TOML. Unknown keys fail.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub set: Vec<String>,
}

pub fn run(args: ReviewArgs) -> Result<()> {
    match args.command {
        ReviewCommand::Package(args) => {
            let summary = package(&args)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            Ok(())
        }
        ReviewCommand::Verify { path } => {
            let count = if path.extension().is_some_and(|e| e == "zip") {
                seal::verify_zip(&path)?
            } else {
                seal::verify_dir(&path)?
            };
            println!("Verified {count} files");
            Ok(())
        }
    }
}

/// `[review]` and the resolved colour mode (flag/env, `[inspect.plot] color_mode`, dark),
/// after the generic `BEKIPER_SET` / `--set` layer.
fn load_review_config(
    path: Option<&Path>,
    theme: Option<&str>,
    set: &[String],
) -> Result<(bekiper_ingress::config::ReviewConfig, themekit::Mode)> {
    load_review_config_from(path, theme, set, &|key| std::env::var(key).ok())
}

/// `BEKIPER_SET` pairs inside the review scope (`review.*`, `inspect.plot.color_mode`).
fn review_env_pairs(lookup: &impl Fn(&str) -> Option<String>) -> Vec<String> {
    lookup("BEKIPER_SET")
        .unwrap_or_default()
        .split(';')
        .map(str::trim)
        .filter(|pair| {
            let key = pair.split_once('=').map_or(*pair, |(k, _)| k).trim();
            key.starts_with("review.") || key == "inspect.plot.color_mode"
        })
        .map(String::from)
        .collect()
}

fn load_review_config_from(
    path: Option<&Path>,
    theme: Option<&str>,
    set: &[String],
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<(bekiper_ingress::config::ReviewConfig, themekit::Mode)> {
    let mut root = toml::Value::Table(match path {
        Some(path) => toml::from_str(&fs::read_to_string(path)?)?,
        None => toml::Table::new(),
    });
    // Run-scope `BEKIPER_SET` keys belong to `bekiper run`; explicit `--set` keys are validated.
    let scoped = |key: &str| {
        if key == "BEKIPER_SET" {
            Some(review_env_pairs(lookup).join(";"))
        } else {
            lookup(key)
        }
    };
    let record = bekiper_ingress::config::apply_set_overrides(&mut root, set, &scoped)?;
    let toml::Value::Table(mut file) = root else {
        bail!("config root is not a table");
    };
    let configured = file
        .get("inspect")
        .and_then(|v| v.get("plot"))
        .and_then(|v| v.get("color_mode"))
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let mode = themekit::Mode::resolve(theme, None, configured.as_deref())?;
    let review = file
        .remove("review")
        .map(|review| review.try_into())
        .transpose()
        .with_context(|| {
            format!(
                "[review] in {}",
                path.map(Path::display)
                    .map(|p| p.to_string())
                    .unwrap_or_default()
            )
        })?
        .unwrap_or_default();
    // `review package` reads only `[review]` and `inspect.plot.color_mode`; any
    // other override key is unknown here.
    bekiper_ingress::config::ensure_overrides_retained(
        &json!({"review": review, "inspect": {"plot": {"color_mode": configured}}}),
        &record,
    )?;
    Ok((review, mode))
}

pub(crate) fn package(args: &PackageArgs) -> Result<Value> {
    let data = args.run.join("data");
    ensure!(
        data.is_dir(),
        "{} has no data/ directory",
        args.run.display()
    );
    let out = &args.output;
    let state_path = out.join("review_state.json");
    let run_root = fs::canonicalize(&args.run)?;
    if out.exists() {
        let state: Value = fs::read(&state_path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .context("output exists but is not a reviewer package from this tool")?;
        ensure!(
            state["run"] == json!(run_root),
            "{} belongs to another run ({})",
            out.display(),
            state["run"]
        );
        ensure!(
            state["preview"] == json!(args.preview),
            "{} is a {} package; use a separate output for {}",
            out.display(),
            state["preview"]
                .as_str()
                .map_or("full".into(), |a| format!("{a} preview")),
            args.preview
                .as_deref()
                .map_or("the full build".into(), |a| format!("the {a} preview")),
        );
    }
    fs::create_dir_all(out.join("logs"))?;
    fs::write(
        &state_path,
        serde_json::to_vec_pretty(&json!({"run": run_root, "preview": args.preview}))?,
    )?;

    let (review, mode) =
        load_review_config(args.config.as_deref(), args.theme.as_deref(), &args.set)?;
    let workers = args
        .workers
        .or(review.workers)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get().min(8))
                .unwrap_or(1)
        })
        .max(1);
    let groups = index::group_names(args.plan.as_deref())?;
    let contexts = index::contexts(&data)?;
    let metrics = index::metrics(&contexts)?;
    let trials = index::discover(&data, &args.trial)?;
    ensure!(
        !trials.is_empty(),
        "no inspected trials under {}",
        data.display()
    );
    let mut recordings = index::recordings(trials, &metrics);
    index::ensure_unique_slugs(
        "animal",
        recordings.iter().filter_map(|r| r.animal.as_deref()),
    )?;
    let presentation = index::present(&mut recordings, &metrics, &review, args.videos)?;
    let mut preview = None;
    if let Some(assay) = &args.preview {
        ensure!(
            index::experiment(&presentation, assay).is_some(),
            "--preview {assay}: no such experiment in this run (have: {})",
            presentation
                .experiments
                .iter()
                .map(|e| e.key.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let criteria = index::criteria(&presentation, assay);
        let population = recordings.iter().filter(|r| r.assay == *assay).count();
        let failed_before =
            |r: &Recording| out.join(&r.figure).with_extension("failed.json").is_file();
        recordings = index::preview_sample(
            recordings,
            assay,
            args.per_stratum,
            &criteria,
            &failed_before,
        );
        ensure!(!recordings.is_empty(), "no {assay} recordings to preview");
        preview = Some(format!(
            "Preview: {assay}, {} of {population} recordings, sampled by min/median/max of the first criteria \
endpoint per session ({}) plus edge cases. Cohort figures cover all recordings.",
            recordings.len(),
            criteria
                .first()
                .map_or("no criteria configured", String::as_str)
        ));
        // Every preview is a fresh round: drop the previous round's composites.
        let _ = fs::remove_dir_all(out.join("figures/trials"));
    }
    let cancel = Cancel::new()?;

    let rendered = render_composites(
        args,
        &mut recordings,
        &groups,
        &presentation,
        mode,
        workers,
        &cancel,
    )?;
    let animal_figures = animals::render(
        args,
        &recordings,
        &metrics,
        &groups,
        &presentation,
        mode,
        workers,
        &cancel,
    )?;
    let videos_rendered = if args.videos {
        let mut video_args = args.clone();
        if video_args.video_renderer.is_none() {
            video_args.video_renderer = Some(std::env::current_exe()?);
        }
        videos::render(&video_args, &mut recordings, mode, workers, &cancel)?
    } else {
        0
    };
    let cohort = render_cohort(args, &contexts, &groups, &presentation, mode, &cancel)?;
    extensions::render(args, &recordings, &groups, mode, &cancel)?;
    copy_data(args, &contexts)?;
    let site = html::Site {
        review: &review,
        presentation: &presentation,
        groups: &groups,
        preview: preview.as_deref(),
        mode,
        animal_figures: &animal_figures,
    };
    let pages = html::write_site(out, &recordings, &cohort, &site, args)?;
    let provenance = json!({
        "schema_version": 1,
        "run": run_root,
        "config": args.config,
        "plan": args.plan,
        "bekiper_version": env!("CARGO_PKG_VERSION"),
        "composite_version": COMPOSITE_VERSION,
        "animal_summary_version": animals::ANIMAL_SUMMARY_VERSION,
        "color_mode": mode.as_str(),
        "overrides": {"env": review_env_pairs(&|k| std::env::var(k).ok()), "cli": args.set},
        "review": serde_json::to_value(&review)?,
        "contexts": contexts.iter().map(|c| c.file_name().map(|n| n.to_string_lossy().into_owned())).collect::<Vec<_>>(),
        "recordings": recordings.len(),
        "trials": recordings.iter().map(|r| r.trials.len()).sum::<usize>(),
        "rendered_now": rendered,
        "videos": args.videos,
        "video_version": videos::VIDEO_VERSION,
        "videos_rendered_now": videos_rendered,
        "preview": args.preview,
        "pages": pages,
    });
    fs::create_dir_all(out.join("provenance"))?;
    // The resolved experiments, steps, contexts and blocks the pages follow.
    fs::write(
        out.join("provenance/presentation.json"),
        serde_json::to_vec_pretty(&presentation)?,
    )?;
    fs::write(
        out.join("provenance/provenance.json"),
        serde_json::to_vec_pretty(&provenance)?,
    )?;
    let files = seal::write_sums(out)?;
    let video_failures = recordings
        .iter()
        .filter(|r| r.video.as_ref().is_some_and(|v| v.get("error").is_some()))
        .count();
    let mut summary = json!({"output": out, "files": files, "recordings": recordings.len(),
        "rendered_now": rendered, "failures": recordings.iter().filter(|r| r.failed.is_some()).count(),
        "videos_rendered_now": videos_rendered, "video_failures": video_failures});
    if args.zip {
        let zip = seal::write_zip(out)?;
        summary["zip"] = json!(zip);
        summary["zip_files"] = json!(seal::verify_zip(&zip)?);
    }
    Ok(summary)
}

pub(crate) struct Cancel {
    flag: Arc<AtomicBool>,
}

impl Cancel {
    fn new() -> Result<Self> {
        let flag = Arc::new(AtomicBool::new(false));
        #[cfg(unix)]
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            signal_hook::flag::register(signal, Arc::clone(&flag))?;
        }
        Ok(Self { flag })
    }

    fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut fs::File::open(path)?, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Input key: manifests + evaluated tables + header + layout version + colour mode.
fn recording_key(recording: &Recording, header: &Value, mode: themekit::Mode) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(COMPOSITE_VERSION);
    hasher.update(mode.as_str());
    hasher.update(serde_json::to_vec(header)?);
    for trial in &recording.trials {
        for path in index::inputs(&trial.dir)? {
            hasher.update(path.file_name().unwrap_or_default().as_encoded_bytes());
            hasher.update(sha256_file(&path)?);
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn render_composites(
    args: &PackageArgs,
    recordings: &mut [Recording],
    groups: &BTreeMap<String, String>,
    presentation: &bekiper_core::experiment::ResolvedPresentation,
    mode: themekit::Mode,
    workers: usize,
    cancel: &Cancel,
) -> Result<usize> {
    let mut jobs: Vec<(usize, Value)> = Vec::new();
    for (i, recording) in recordings.iter_mut().enumerate() {
        let criteria = index::criteria(presentation, &recording.assay);
        let header = index::header(recording, groups, &criteria);
        let key = recording_key(recording, &header, mode)?;
        let out = args.output.join(&recording.figure);
        let sidecar: Option<Value> = fs::read(out.with_extension("json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        let webp = out.with_extension("webp");
        let fresh = sidecar.as_ref().is_some_and(|s| {
            s["key"] == json!(key)
                && webp.is_file()
                && sha256_file(&webp).ok().as_deref() == s["webp_sha256"].as_str()
        });
        if fresh && args.preview.is_none() {
            continue;
        }
        let variants: Vec<Value> = recording
            .trials
            .iter()
            .map(|t| json!({"name": t.variant_label(), "dir": t.dir}))
            .collect();
        jobs.push((
            i,
            json!({"key": key, "out": out, "header": header, "variants": variants, "theme": mode.as_str()}),
        ));
    }
    // Balance shards by rough cost (descriptor `render_cost` × scoring variants).
    jobs.sort_by_key(|(i, _)| std::cmp::Reverse(recordings[*i].cost()));
    let rendered = jobs.len();
    run_job_shards(
        args,
        "composite",
        jobs.into_iter().map(|(_, job)| job).collect(),
        workers,
        cancel,
    )?;
    if cancel.cancelled() {
        bail!("review packaging cancelled");
    }
    for recording in recordings.iter_mut() {
        let out = args.output.join(&recording.figure);
        let failed = out.with_extension("failed.json");
        recording.failed = fs::read(&failed)
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .map(|v| v["error"].as_str().unwrap_or("render failed").to_string());
        recording.sidecar = fs::read(out.with_extension("json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        if recording.failed.is_none() && recording.sidecar.is_none() {
            recording.failed = Some("renderer produced no output".into());
        }
    }
    Ok(rendered)
}

/// Run `<renderer> <subcommand> --jobs <shard>.json` over up to `workers`
/// round-robin shards in parallel; a failed job leaves its own `.failed.json`.
pub(crate) fn run_job_shards(
    args: &PackageArgs,
    subcommand: &str,
    jobs: Vec<Value>,
    workers: usize,
    cancel: &Cancel,
) -> Result<()> {
    let mut shards: Vec<Vec<Value>> = vec![Vec::new(); workers.min(jobs.len()).max(1)];
    let count = shards.len();
    for (n, job) in jobs.into_iter().enumerate() {
        shards[n % count].push(job);
    }
    let shard_dir = &args.output.join("logs");
    std::thread::scope(|scope| -> Result<()> {
        let handles: Vec<_> = shards
            .iter()
            .enumerate()
            .filter(|(_, shard)| !shard.is_empty())
            .map(|(n, shard)| {
                let path = shard_dir.join(format!("{subcommand}_{n:02}.jobs.json"));
                let renderer = &args.renderer;
                scope.spawn(move || -> Result<()> {
                    fs::write(&path, serde_json::to_vec(shard)?)?;
                    let mut command = Command::new(renderer);
                    command.arg(subcommand).arg("--jobs").arg(&path);
                    // A nonzero exit means some jobs failed; each left a .failed.json.
                    bekiper_post_analysis::pipeline::run_child(
                        &mut command,
                        &shard_dir.join(format!("{subcommand}_{n:02}.stdout.log")),
                        &shard_dir.join(format!("{subcommand}_{n:02}.stderr.log")),
                        &|| cancel.cancelled(),
                    )?;
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("renderer thread panicked"))??;
        }
        Ok(())
    })?;
    if cancel.cancelled() {
        bail!("review packaging cancelled");
    }
    Ok(())
}

/// Whether an experiment is scored per threshold context (Y-maze); the others
/// are shown once, from the principal cohort context.
fn has_contexts(assay: &str) -> bool {
    bekiper_core::experiment::descriptor(assay)
        .is_some_and(|d| d.contexts.as_ref().is_some_and(|c| c.varies()))
}

/// Cohort figures per post-analysis context; experiments without threshold
/// contexts only from the first (principal) one, which the renderer decides from
/// the descriptors. A preview re-renders only its assay on every run.
fn render_cohort(
    args: &PackageArgs,
    contexts: &[PathBuf],
    groups: &BTreeMap<String, String>,
    presentation: &bekiper_core::experiment::ResolvedPresentation,
    mode: themekit::Mode,
    cancel: &Cancel,
) -> Result<Vec<Value>> {
    let work = args.output.join("logs");
    let groups_path = work.join("groups.json");
    let primary_path = work.join("primary_endpoints.json");
    fs::write(&groups_path, serde_json::to_vec(groups)?)?;
    let primary: BTreeMap<&str, &Vec<String>> = presentation
        .experiments
        .iter()
        .filter(|e| !e.endpoints.is_empty())
        .map(|e| (e.key.as_str(), &e.endpoints))
        .collect();
    fs::write(&primary_path, serde_json::to_vec(&primary)?)?;
    let mut indexes = Vec::new();
    for (n, context) in contexts.iter().enumerate() {
        let name = context
            .file_name()
            .context("context name")?
            .to_string_lossy()
            .into_owned();
        let target = args.output.join("figures/cohort").join(&name);
        let index_path = target.join("cohort_index.json");
        let preview = args.preview.as_deref();
        if n > 0 && preview.is_some_and(|a| !has_contexts(a)) {
            continue;
        }
        // A preview always redraws; a full build redraws when the colour mode changed.
        let cached_mode = fs::read(&index_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .map(|index| index["color_mode"].clone());
        // The endpoint lists the figures were drawn with, kept beside the index.
        let primary_stamp = target.join(".primary_endpoints.json");
        let stale_endpoints = fs::read(&primary_stamp).ok() != Some(fs::read(&primary_path)?);
        if preview.is_some()
            || cached_mode.is_some_and(|m| m != json!(mode.as_str()))
            || (index_path.is_file() && stale_endpoints)
        {
            let _ = fs::remove_dir_all(&target);
        }
        if !index_path.is_file() {
            let mut command = Command::new(&args.renderer);
            command
                .arg("review-figures")
                .arg("--context")
                .arg(context)
                .arg("--output")
                .arg(&target)
                .arg("--groups")
                .arg(&groups_path)
                .arg("--primary")
                .arg(&primary_path)
                .arg("--theme")
                .arg(mode.as_str());
            if let Some(assay) = preview {
                command.arg("--assay").arg(assay);
            }
            let status = bekiper_post_analysis::pipeline::run_child(
                &mut command,
                &work.join(format!("cohort_{name}.stdout.log")),
                &work.join(format!("cohort_{name}.stderr.log")),
                &|| cancel.cancelled(),
            )?;
            ensure!(
                status.success(),
                "cohort figures failed for {name}; see logs/"
            );
        }
        fs::write(&primary_stamp, fs::read(&primary_path)?)?;
        let mut index: Value = serde_json::from_slice(&fs::read(&index_path)?)?;
        index["path"] = json!(format!("figures/cohort/{name}"));
        indexes.push(index);
    }
    Ok(indexes)
}

/// Workbook, configuration and the human-readable cohort files (no machine JSON).
fn copy_data(args: &PackageArgs, contexts: &[PathBuf]) -> Result<()> {
    let data = args.output.join("data");
    fs::create_dir_all(&data)?;
    for entry in fs::read_dir(&args.run)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "ods") {
            link_or_copy(&path, &data.join(path.file_name().context("ods name")?))?;
        }
    }
    if let Some(config) = &args.config {
        link_or_copy(config, &data.join("bekiper_config.toml"))?;
    }
    if let Some(plan) = &args.plan {
        link_or_copy(plan, &data.join("post_analysis_plan.toml"))?;
    }
    let provenance = args.output.join("provenance");
    fs::create_dir_all(&provenance)?;
    if let Some(root) = contexts.first().and_then(|c| c.parent()) {
        let manifest = root.join("run_manifest.json");
        if manifest.is_file() {
            link_or_copy(&manifest, &provenance.join("cohort_run_manifest.json"))?;
        }
    }
    for (n, context) in contexts.iter().enumerate() {
        if n > 0 && args.preview.as_deref().is_some_and(|p| !has_contexts(p)) {
            continue;
        }
        let name = context
            .file_name()
            .context("context")?
            .to_string_lossy()
            .into_owned();
        let target = data.join("cohort").join(&name);
        fs::create_dir_all(&target)?;
        for file in [
            "README.md",
            "report.pdf",
            "report.md",
            "metrics.toml",
            "parameters.toml",
        ] {
            if context.join(file).is_file() {
                link_or_copy(&context.join(file), &target.join(file))?;
            }
        }
        let manifest = context.join("artifact_manifest.json");
        if manifest.is_file() {
            link_or_copy(
                &manifest,
                &provenance.join(format!("artifact_manifest_{name}.json")),
            )?;
        }
        // Vector originals: every assay once, then only the figures of experiments
        // with threshold contexts, which differ per context.
        for assay in bekiper_core::experiment::post_analysis_descriptors()
            .filter_map(|d| d.post_analysis.as_ref().map(|p| p.assay.as_str()))
        {
            if (n > 0 && !has_contexts(assay))
                || args.preview.as_deref().is_some_and(|p| p != assay)
            {
                continue;
            }
            let source = context.join("figures").join(assay);
            let Ok(entries) = fs::read_dir(&source) else {
                continue;
            };
            let dest = target.join("figures").join(assay);
            fs::create_dir_all(&dest)?;
            for entry in entries {
                let path = entry?.path();
                // PDF vector originals only; the SVGs are the same figures again.
                if path.extension().is_some_and(|e| e == "pdf") {
                    link_or_copy(&path, &dest.join(path.file_name().context("figure")?))?;
                }
            }
        }
    }
    Ok(())
}

fn link_or_copy(from: &Path, to: &Path) -> Result<()> {
    if to.exists() {
        if fs::metadata(from)?.len() == fs::metadata(to)?.len() {
            return Ok(());
        }
        fs::remove_file(to)?;
    }
    if fs::hard_link(from, to).is_err() {
        fs::copy(from, to).with_context(|| format!("copy {}", from.display()))?;
    }
    Ok(())
}
