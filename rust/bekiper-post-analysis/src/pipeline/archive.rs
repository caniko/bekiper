//! Verify selected post-analysis package members before deleting data sources.
//! Tar members are streamed to the verifier; no archive paths are extracted.
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::Path,
    process::{Command, Stdio},
};

use super::{publication::verify_artifacts, safe_relative_path};

#[derive(Debug, PartialEq)]
pub struct ProjectArtifactSnapshot {
    pub run: String,
    pub manifest: Value,
    pub manifest_fingerprint: Value,
}

/// Active claims, incomplete staging directories and failed runs cannot be
/// treated as a complete export when an archive command removes its source.
pub fn verify_project_exports(data: &Path) -> Result<Vec<ProjectArtifactSnapshot>> {
    let root = data.join("_post_analysis");
    if !root.try_exists()? {
        return Ok(Vec::new());
    }
    ensure!(
        fs::symlink_metadata(&root)?.file_type().is_dir(),
        "post-analysis export root must be a real directory"
    );
    let mut snapshots = Vec::new();
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("export run name is not UTF-8"))?;
        ensure!(
            entry.file_type()?.is_dir() && safe_relative_path(&name) && !name.starts_with('.'),
            "post-analysis exports contain an active, incomplete or unsafe member: {name}"
        );
        let manifest = verify_artifacts(&entry.path())
            .with_context(|| format!("verifying post-analysis run {name}"))?;
        let run: Value =
            serde_json::from_slice(&fs::read(entry.path().join("run_manifest.json"))?)?;
        ensure!(
            run["schema_version"] == "2.0"
                && matches!(
                    run["status"].as_str(),
                    Some("complete" | "partial" | "blocked")
                ),
            "post-analysis run {name} has no completed execution manifest"
        );
        let fingerprint =
            super::publication::file_fingerprint(&entry.path().join("artifact_manifest.json"))?;
        snapshots.push(ProjectArtifactSnapshot {
            run: name,
            manifest,
            manifest_fingerprint: fingerprint,
        });
    }
    snapshots.sort_by(|a, b| a.run.cmp(&b.run));
    Ok(snapshots)
}

/// Compare actual archived bytes with the manifests verified before creation.
pub fn verify_data_archive(archive: &Path, snapshots: &[ProjectArtifactSnapshot]) -> Result<()> {
    let mut selected = BTreeMap::new();
    for snapshot in snapshots {
        ensure!(
            safe_relative_path(&snapshot.run) && !snapshot.run.contains('/'),
            "unsafe archived run name"
        );
        let prefix = format!("data/_post_analysis/{}", snapshot.run);
        let manifest_member = format!("{prefix}/artifact_manifest.json");
        ensure!(
            selected
                .insert(manifest_member, &snapshot.manifest_fingerprint)
                .is_none(),
            "duplicate archived run"
        );
        for artifact in snapshot.manifest["files"]
            .as_array()
            .context("archive snapshot needs files")?
        {
            let relative = artifact["path"].as_str().context("artifact path")?;
            ensure!(
                safe_relative_path(relative),
                "unsafe archived artifact path"
            );
            ensure!(
                selected
                    .insert(format!("{prefix}/{relative}"), artifact)
                    .is_none(),
                "duplicate archived artifact"
            );
        }
    }
    if selected.is_empty() {
        return Ok(());
    }
    let mut selected: Vec<_> = selected.into_iter().collect();
    selected.sort_by(|a, b| a.0.split('/').cmp(b.0.split('/')));
    // Compression uses tar --sort=name with the C locale. Requested members
    // therefore appear in component-wise lexical (depth-first) order in tar's
    // concatenated stream. Full-path sorting disagrees for prefix siblings.
    // One decoder pass avoids rereading a large pose archive per figure.
    let mut child = Command::new("tar")
        .args(["--use-compress-program=zstd -d", "-xOf"])
        .arg(archive)
        .arg("--")
        .args(selected.iter().map(|(member, _)| member))
        .env("LC_ALL", "C")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to start archive artifact verifier")?;
    let read = (|| -> Result<()> {
        let mut stream = child
            .stdout
            .take()
            .context("archive verifier needs stdout")?;
        let mut buffer = [0u8; 65536];
        for (member, expected) in &selected {
            let bytes = expected["bytes"]
                .as_u64()
                .context("archive artifact needs byte count")?;
            let fingerprint = crate::fingerprint::reader(
                &mut stream.by_ref().take(bytes),
                "archive artifact size overflow",
            )?; // coverage: unreachable — verifier pipe reads cannot be made to fail; truncation is checked by byte count
            ensure!(
                fingerprint.bytes == bytes,
                "archive artifact is missing or truncated: {member}"
            );
            ensure!(
                expected["sha256"] == fingerprint.sha256,
                "archive artifact fingerprint mismatch: {member}"
            );
        }
        ensure!(
            stream.read(&mut buffer)? == 0,
            "archive contains duplicate or oversized selected artifacts"
        );
        Ok(())
    })();
    if read.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    read?;
    ensure!(status.success(), "archive artifact is absent or unreadable");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::publication::{write_artifact_manifest, write_json};
    use super::*;
    use serde_json::json;

    fn package(parent: &Path) -> std::path::PathBuf {
        let data = parent.join("data");
        let run = data.join("_post_analysis/cohort-v1");
        fs::create_dir_all(&run).unwrap();
        write_json(
            &run.join("run_manifest.json"),
            &json!({"schema_version":"2.0","status":"partial"}),
        )
        .unwrap();
        fs::write(run.join("selected.svg"), "<svg>selected</svg>").unwrap();
        fs::write(run.join("report.pdf"), "%PDF fixture").unwrap();
        write_artifact_manifest(&run).unwrap();
        data
    }

    fn archive(parent: &Path, file: &Path) {
        let status = Command::new("tar")
            .arg("-C")
            .arg(parent)
            .args(["--sort=name", "--use-compress-program=zstd -T1 -1", "-cf"])
            .arg(file)
            .arg("data")
            .env("LC_ALL", "C")
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn real_archive_members_are_verified_before_cleanup() {
        let root = tempfile::tempdir().unwrap();
        let data = package(root.path());
        let snapshots = verify_project_exports(&data).unwrap();
        let file = root.path().join("data.tar.zst");
        archive(root.path(), &file);
        verify_data_archive(&file, &snapshots).unwrap();
        fs::write(
            data.join("_post_analysis/cohort-v1/selected.svg"),
            "changed",
        )
        .unwrap();
        assert!(verify_project_exports(&data).is_err());
        archive(root.path(), &file);
        assert!(verify_data_archive(&file, &snapshots).is_err());
        assert!(data.is_dir());
    }

    #[test]
    fn incomplete_runs_prevent_destructive_archiving() {
        let root = tempfile::tempdir().unwrap();
        let data = package(root.path());
        fs::write(data.join("_post_analysis/.cohort-v2.claim"), "pid=1").unwrap();
        assert!(verify_project_exports(&data).is_err());
        fs::remove_file(data.join("_post_analysis/.cohort-v2.claim")).unwrap();
        fs::remove_file(data.join("_post_analysis/cohort-v1/run_manifest.json")).unwrap();
        assert!(verify_project_exports(&data).is_err());
    }

    #[test]
    fn bounded_hashing_preserves_empty_chunked_and_truncated_member_boundaries() {
        let root = tempfile::tempdir().unwrap();
        let data = package(root.path());
        let run = data.join("_post_analysis/cohort-v1");
        fs::write(run.join("empty.bin"), []).unwrap();
        let large: Vec<_> = (0..131_083).map(|index| (index % 251) as u8).collect();
        fs::write(run.join("chunked.bin"), large).unwrap();
        fs::write(run.join("zz-last.bin"), [7; 256]).unwrap();
        write_artifact_manifest(&run).unwrap();
        let snapshots = verify_project_exports(&data).unwrap();
        let file = root.path().join("data.tar.zst");
        archive(root.path(), &file);
        verify_data_archive(&file, &snapshots).unwrap();
        fs::write(run.join("zz-last.bin"), [7; 32]).unwrap();
        archive(root.path(), &file);
        let error = verify_data_archive(&file, &snapshots).unwrap_err();
        assert!(
            error.to_string().contains("missing or truncated"),
            "{error:#}"
        );
        assert!(data.is_dir());
    }

    #[test]
    fn archive_verification_follows_recursive_order_for_prefix_siblings() {
        let root = tempfile::tempdir().unwrap();
        let data = package(root.path());
        let first = data.join("_post_analysis/cohort-v1");
        let second = data.join("_post_analysis/cohort-v1-extra");
        fs::create_dir_all(first.join("nested")).unwrap();
        fs::write(first.join("nested/a.json"), "nested artifact").unwrap();
        fs::write(first.join("nested.json"), "prefix sibling artifact").unwrap();
        fs::create_dir_all(&second).unwrap();
        write_json(
            &second.join("run_manifest.json"),
            &json!({"schema_version":"2.0","status":"complete"}),
        )
        .unwrap();
        fs::write(second.join("figure.pdf"), "%PDF other fixture").unwrap();
        write_artifact_manifest(&first).unwrap();
        write_artifact_manifest(&second).unwrap();
        let snapshots = verify_project_exports(&data).unwrap();
        let file = root.path().join("data.tar.zst");
        archive(root.path(), &file);
        verify_data_archive(&file, &snapshots).unwrap();
    }
}
