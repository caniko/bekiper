use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Component, Path, PathBuf},
};

/// Shared exporters and exclusive archive cleanup coordinate for the lifetime
/// of their filesystem work. The persistent lock file stays outside `data` so
/// deleting an archived source cannot split the lock into two different inodes.
pub struct ProjectDataLease {
    _file: File,
}

impl ProjectDataLease {
    pub fn export(run_root: &Path) -> Result<Self> {
        Self::acquire(run_root, false)
    }

    pub fn archive(run_root: &Path) -> Result<Self> {
        Self::acquire(run_root, true)
    }

    #[cfg(target_os = "linux")]
    fn acquire(run_root: &Path, exclusive: bool) -> Result<Self> {
        use std::os::fd::AsRawFd;
        unsafe extern "C" {
            fn flock(fd: std::ffi::c_int, operation: std::ffi::c_int) -> std::ffi::c_int;
        }
        fs::create_dir_all(run_root)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(run_root.join(".post-analysis-data.lock"))?;
        // LOCK_SH=1, LOCK_EX=2, LOCK_NB=4. Closing the last descriptor releases
        // the kernel lock, including on unexpected process exit; Drop also
        // unlocks explicitly so fork/exec inheritance cannot extend this lease.
        let operation = if exclusive { 2 } else { 1 } | 4;
        // SAFETY: flock receives a live owned file descriptor and valid flags.
        if unsafe { flock(file.as_raw_fd(), operation) } != 0 {
            return Err(std::io::Error::last_os_error())
                .context("project data is in use by an exporter or archiver");
        }
        Ok(Self { _file: file })
    }

    #[cfg(not(target_os = "linux"))]
    fn acquire(_run_root: &Path, _exclusive: bool) -> Result<Self> {
        anyhow::bail!(
            "coordinated project export/archive publication requires Linux filesystem locks"
        )
    }
}

#[cfg(target_os = "linux")]
impl Drop for ProjectDataLease {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        unsafe extern "C" {
            fn flock(fd: std::ffi::c_int, operation: std::ffi::c_int) -> std::ffi::c_int;
        }
        // LOCK_UN=8 releases this open file description even if a concurrent
        // fork still holds an inherited descriptor before close-on-exec.
        // SAFETY: the lease owns a live descriptor and LOCK_UN is a valid flag.
        let _ = unsafe { flock(self._file.as_raw_fd(), 8) };
    }
}

/// Cooperating publishers claim a destination before doing any native work.
/// A failed process leaves a visible claim rather than silently replacing data.
pub struct Publication {
    destination: PathBuf,
    claim_path: PathBuf,
    _claim: File,
    staging: Option<tempfile::TempDir>,
}

impl Publication {
    pub fn claim(destination: &Path) -> Result<Self> {
        let name = destination
            .file_name()
            .and_then(|name| name.to_str())
            .context("output needs a directory name")?;
        ensure!(
            !matches!(name, "." | "..") && !name.is_empty(),
            "invalid output directory name"
        );
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let parent = parent.canonicalize()?;
        let destination = parent.join(name);
        let claim_path = parent.join(format!(".{name}.claim"));
        let mut claim = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&claim_path)
            .context("post-analysis output is already claimed; use a fresh output directory")?;
        let acquired = (|| -> Result<tempfile::TempDir> {
            ensure!(
                !destination.try_exists()?,
                "use a fresh output directory; published runs are immutable"
            );
            writeln!(claim, "pid={}", std::process::id())?;
            claim.sync_all()?;
            Ok(tempfile::Builder::new()
                .prefix(&format!(".{name}.staging-"))
                .tempdir_in(&parent)?)
        })();
        match acquired {
            Ok(staging) => Ok(Self {
                destination,
                claim_path,
                _claim: claim,
                staging: Some(staging),
            }),
            Err(error) => {
                let _ = fs::remove_file(claim_path);
                Err(error)
            }
        }
    }

    pub fn directory(&self) -> &Path {
        self.staging.as_ref().expect("live publication").path()
    }

    pub fn fail(&mut self, stage: &str, error: &anyhow::Error) -> Result<PathBuf> {
        let directory = self.directory().to_path_buf();
        // A final manifest exists only after every stage and package check.
        if directory.join("run_manifest.json").try_exists()? {
            fs::remove_file(directory.join("run_manifest.json"))?;
        }
        let message = redact_error(&format!("{error:#}"), &directory);
        write_json(
            &directory.join("failure_manifest.json"),
            &json!({
                "schema_version":"2.0", "status": if stage == "cancelled" { "cancelled" } else { "error" },
                "failed_stage":stage,"error":message,"published":false
            }),
        )?;
        let retained = self.staging.take().expect("live publication").keep();
        Ok(retained)
    }

    pub fn publish(&mut self) -> Result<()> {
        let directory = self.directory();
        verify_artifacts(directory)?;
        ensure!(
            !self.destination.try_exists()?,
            "publication destination appeared while processing"
        );
        File::open(directory)?.sync_all()?;
        let parent = File::open(self.destination.parent().context("publication parent")?)?;
        parent.sync_all()?;
        rename_exclusive(directory, &self.destination)?;
        self.staging.take();
        // The commit has happened. A directory-sync error cannot turn valid,
        // visible artifacts into a failed numerical execution.
        let _ = parent.sync_all();
        Ok(())
    }
}

/// Linux's no-replace rename makes the missing-destination check atomic, even
/// for a non-cooperating writer creating an empty directory during publication.
#[cfg(target_os = "linux")]
fn rename_exclusive(source: &Path, destination: &Path) -> Result<()> {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    unsafe extern "C" {
        fn renameat2(
            old_dir: i32,
            old_path: *const std::ffi::c_char,
            new_dir: i32,
            new_path: *const std::ffi::c_char,
            flags: u32,
        ) -> i32;
    }
    let source = CString::new(source.as_os_str().as_bytes())?;
    let destination = CString::new(destination.as_os_str().as_bytes())?;
    // SAFETY: both NUL-terminated buffers remain alive throughout the call.
    // AT_FDCWD = -100; RENAME_NOREPLACE = 1. No descriptor or pointer is retained.
    let status = unsafe { renameat2(-100, source.as_ptr(), -100, destination.as_ptr(), 1) };
    if status != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn rename_exclusive(_: &Path, _: &Path) -> Result<()> {
    anyhow::bail!("atomic no-replace publication is supported on Linux")
}

impl Drop for Publication {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.claim_path);
    }
}

fn redact_error(message: &str, directory: &Path) -> String {
    message.replace(&directory.display().to_string(), "<staging>")
}

pub fn write_json(path: &Path, value: &(impl serde::Serialize + ?Sized)) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)?;
    file.into_inner()
        .map_err(|error| error.into_error())?
        .sync_all()?;
    Ok(())
}

pub fn artifact_inventory(directory: &Path) -> Result<Vec<Value>> {
    let mut paths = Vec::new();
    collect_files(directory, directory, &mut paths)?;
    paths.sort();
    paths
        .into_par_iter()
        .filter(|path| path != "artifact_manifest.json")
        .map(|path| {
            let file = directory.join(&path);
            let fingerprint = file_fingerprint(&file)?;
            Ok(json!({"path":path,"bytes":fingerprint["bytes"],"sha256":fingerprint["sha256"]}))
        })
        .collect()
}

pub fn file_fingerprint(path: &Path) -> Result<Value> {
    let mut file = File::open(path)?;
    Ok(
        crate::fingerprint::reader(&mut file, "artifact size overflow")?
            .named(path.file_name().unwrap_or_default().to_string_lossy()),
    )
}

fn collect_files(root: &Path, directory: &Path, paths: &mut Vec<String>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = entry.file_type()?;
        ensure!(!metadata.is_symlink(), "package contains a symbolic link");
        let path = entry.path();
        if metadata.is_dir() {
            collect_files(root, &path, paths)?;
        } else {
            ensure!(metadata.is_file(), "package contains a nonregular artifact");
            let relative = path
                .strip_prefix(root)?
                .to_str()
                .context("artifact filename is not UTF-8")?
                .replace('\\', "/");
            ensure!(safe_relative_path(&relative), "unsafe artifact member");
            paths.push(relative);
        }
    }
    Ok(())
}

pub fn safe_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && Path::new(path)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

pub fn write_artifact_manifest(directory: &Path) -> Result<()> {
    write_json(
        &directory.join("artifact_manifest.json"),
        &json!({"schema_version":"1.0","files":artifact_inventory(directory)?}),
    )
}

/// Verify exact membership as well as hashes. Stale, unselected files fail just
/// as a missing or altered selected figure does.
pub fn verify_artifacts(directory: &Path) -> Result<Value> {
    let manifest: Value =
        serde_json::from_slice(&fs::read(directory.join("artifact_manifest.json"))?)?;
    ensure!(
        manifest["schema_version"] == "1.0",
        "unsupported artifact manifest version"
    );
    let files = manifest["files"]
        .as_array()
        .context("artifact manifest requires files")?;
    let mut seen = BTreeSet::new();
    for item in files {
        let path = item["path"]
            .as_str()
            .context("artifact member needs a path")?;
        ensure!(
            safe_relative_path(path) && path != "artifact_manifest.json" && seen.insert(path),
            "unsafe or duplicate artifact member"
        );
        ensure!(
            item["sha256"].as_str().is_some_and(
                |hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            ),
            "artifact member requires SHA-256"
        );
        ensure!(
            item["bytes"].as_u64().is_some(),
            "artifact member requires byte count"
        );
    }
    let current = artifact_inventory(directory)?;
    ensure!(
        current.len() == files.len(),
        "artifact membership differs from manifest"
    );
    let current: BTreeMap<_, _> = current
        .iter()
        .map(|item| (item["path"].as_str(), item))
        .collect();
    for expected in files {
        let actual = *current
            .get(&expected["path"].as_str())
            .context("manifest artifact missing")?;
        ensure!(
            actual == expected,
            "artifact fingerprint mismatch: {}",
            expected["path"]
        );
    }
    Ok(manifest)
}

#[cfg(test)]
#[path = "publication/tests.rs"]
mod tests;
