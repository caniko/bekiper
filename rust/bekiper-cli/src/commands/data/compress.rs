use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Stdio;

use anyhow::{Context, Result};
use clap::Args;

use bekiper_core::identity::DEFAULT_CONFIG_FILENAME;
use bekiper_ingress::config::IngressConfig;

#[derive(Args)]
pub struct CompressDataArgs {
    /// Path to the project configuration file. Used to resolve the output dir when --run-root is omitted.
    #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
    config: PathBuf,

    /// Existing run root containing the data directory. Defaults to the configured output dir.
    #[arg(long)]
    run_root: Option<PathBuf>,

    /// Archive path to write. Defaults to `<run-root>/data.tar.zst`.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Keep `<run-root>/data` after the archive is written and validated.
    #[arg(long, default_value_t = false)]
    keep_source: bool,

    /// Replace an existing archive at the output path.
    #[arg(long, default_value_t = false)]
    force: bool,
}

pub fn run(args: CompressDataArgs) -> Result<()> {
    let run_root = resolve_run_root(&args.config, args.run_root.as_deref())?;
    let archive_path = args
        .output
        .clone()
        .unwrap_or_else(|| run_root.join("data.tar.zst"));

    let result = compress_data_dir(CompressDataOptions {
        run_root,
        archive_path,
        keep_source: args.keep_source,
        force: args.force,
    })?;

    println!(
        "Compressed {} to {} with zstd level {}, long window {}, {} thread(s).",
        result.source_dir.display(),
        result.archive_path.display(),
        result.strategy.level,
        result.strategy.long_window_log,
        result.strategy.threads
    );
    if result.removed_source {
        println!("Removed uncompressed data directory after archive validation.");
    }

    Ok(())
}

pub(crate) struct CompressDataOptions {
    pub(crate) run_root: PathBuf,
    pub(crate) archive_path: PathBuf,
    pub(crate) keep_source: bool,
    pub(crate) force: bool,
}

pub(crate) struct CompressDataResult {
    pub(crate) source_dir: PathBuf,
    pub(crate) archive_path: PathBuf,
    pub(crate) strategy: CompressionStrategy,
    pub(crate) removed_source: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompressionStrategy {
    pub(crate) level: u8,
    pub(crate) long_window_log: u8,
    pub(crate) threads: usize,
}

pub(crate) fn compress_data_dir(opts: CompressDataOptions) -> Result<CompressDataResult> {
    ensure_command_available("tar")?;
    ensure_command_available("zstd")?;

    let source_dir = opts.run_root.join("data");
    if !source_dir.is_dir() {
        anyhow::bail!(
            "missing data directory {}; run analysis first or extract an existing data.tar.zst archive",
            source_dir.display()
        );
    }

    let _lease = bekiper_post_analysis::pipeline::ProjectDataLease::archive(&opts.run_root)?;

    if opts.archive_path.exists() && !opts.force {
        anyhow::bail!(
            "archive {} already exists; pass --force to replace it",
            opts.archive_path.display()
        );
    }

    if let Some(parent) = opts.archive_path.parent() {
        std::fs::create_dir_all(if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        })
        .with_context(|| {
            format!(
                "failed to create archive parent directory {}",
                parent.display()
            )
        })?;
    }

    let archive_parent = opts
        .archive_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let archive_name = opts
        .archive_path
        .file_name()
        .context("archive needs a filename")?;
    let archive_path = archive_parent.join(archive_name);
    let source_dir = source_dir.canonicalize()?;
    anyhow::ensure!(
        !archive_path.starts_with(&source_dir),
        "archive output must be outside the data directory to survive source cleanup"
    );
    let artifact_snapshots = bekiper_post_analysis::pipeline::verify_project_exports(&source_dir)?;

    let strategy = CompressionStrategy::balanced_for_host();
    let tmp_path = temporary_archive_path(&archive_path);
    let tmp_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp_path)
        .with_context(|| {
            format!(
                "temporary archive is already claimed: {}",
                tmp_path.display()
            )
        })?;
    let _temporary = TemporaryArchive(tmp_path.clone());

    let zstd_program = format!(
        "zstd -T{} -{} --long={}",
        strategy.threads, strategy.level, strategy.long_window_log
    );
    let status = Command::new("tar")
        .arg("-C")
        .arg(&opts.run_root)
        .arg("--sort=name")
        .arg("--use-compress-program")
        .arg(&zstd_program)
        .arg("-cf")
        .arg("-")
        .arg("data")
        .env("LC_ALL", "C")
        .stdout(tmp_file.try_clone()?)
        .status()
        .context("failed to spawn tar for data compression")?;
    if !status.success() {
        anyhow::bail!("tar/zstd compression failed with status {status}");
    }

    let status = Command::new("zstd")
        .arg("-t")
        .arg(&tmp_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("failed to validate compressed archive with zstd -t")?;
    if !status.success() {
        anyhow::bail!("compressed archive validation failed with status {status}");
    }

    tmp_file.sync_all()?;
    bekiper_post_analysis::pipeline::verify_data_archive(&tmp_path, &artifact_snapshots)?;
    let current = bekiper_post_analysis::pipeline::verify_project_exports(&source_dir)?;
    anyhow::ensure!(
        current == artifact_snapshots,
        "post-analysis artifacts changed during compression; source retained"
    );

    if opts.force {
        std::fs::rename(&tmp_path, &archive_path)
    } else {
        // A hard link is an atomic, no-replace publication on the same volume.
        std::fs::hard_link(&tmp_path, &archive_path)
    }
    .with_context(|| {
        format!(
            "failed to move temporary archive {} to {}",
            tmp_path.display(),
            archive_path.display()
        )
    })?;
    std::fs::File::open(&archive_parent)?.sync_all()?;

    let removed_source = !opts.keep_source;
    if removed_source {
        std::fs::remove_dir_all(&source_dir).with_context(|| {
            format!(
                "archive was written, but failed to remove source directory {}",
                source_dir.display()
            )
        })?;
    }

    Ok(CompressDataResult {
        source_dir,
        archive_path,
        strategy,
        removed_source,
    })
}

struct TemporaryArchive(PathBuf);

impl Drop for TemporaryArchive {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn resolve_run_root(config_path: &Path, run_root: Option<&Path>) -> Result<PathBuf> {
    if let Some(run_root) = run_root {
        return Ok(run_root.to_path_buf());
    }
    let config = IngressConfig::from_file(config_path)?;
    let base_dir = config_path.parent().unwrap_or(Path::new("."));
    Ok(base_dir.join(
        config
            .project
            .output_dir
            .as_deref()
            .unwrap_or(Path::new("output")),
    ))
}

fn ensure_command_available(command: &str) -> Result<()> {
    let status = Command::new(command)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("required command `{command}` is not available on PATH"))?;
    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("required command `{command}` failed its --version check")
    }
}

fn temporary_archive_path(archive_path: &Path) -> PathBuf {
    let file_name = archive_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("data.tar.zst");
    archive_path.with_file_name(format!("{file_name}.tmp-{}", std::process::id()))
}

impl CompressionStrategy {
    pub(crate) fn balanced_for_host() -> Self {
        let total_ram_gib = total_memory_bytes()
            .map(|bytes| (bytes / 1024 / 1024 / 1024).max(1) as usize)
            .unwrap_or(16);
        let cpus = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        Self::for_host(cpus, total_ram_gib)
    }

    fn for_host(cpus: usize, total_ram_gib: usize) -> Self {
        let memory_limited_threads = (total_ram_gib / 2).clamp(1, 16);
        Self {
            level: 10,
            long_window_log: 27,
            threads: cpus.clamp(1, memory_limited_threads),
        }
    }
}

fn total_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        let Some(rest) = line.strip_prefix("MemTotal:") else {
            continue;
        };
        let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        return Some(kib * 1024);
    }
    None
}

#[cfg(test)]
#[path = "compress/tests.rs"]
mod tests;
