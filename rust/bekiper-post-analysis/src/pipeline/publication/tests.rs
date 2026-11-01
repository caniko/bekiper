use super::*;

#[test]
fn project_data_leases_exclude_archive_cleanup_during_exports() {
    let root = tempfile::tempdir().unwrap();
    let first = ProjectDataLease::export(root.path()).unwrap();
    let second = ProjectDataLease::export(root.path()).unwrap();
    assert!(ProjectDataLease::archive(root.path()).is_err());
    drop(first);
    assert!(ProjectDataLease::archive(root.path()).is_err());
    drop(second);
    let archive = ProjectDataLease::archive(root.path()).unwrap();
    assert!(ProjectDataLease::export(root.path()).is_err());
    assert!(ProjectDataLease::archive(root.path()).is_err());
    drop(archive);
    ProjectDataLease::export(root.path()).unwrap();
}

#[test]
fn inherited_descriptor_does_not_extend_a_finished_parent_lease() {
    let root = tempfile::tempdir().unwrap();
    let lease = ProjectDataLease::export(root.path()).unwrap();
    // A concurrently spawning child inherits the open file description
    // between fork and exec, even though Rust sets close-on-exec.
    let inherited = lease._file.try_clone().unwrap();
    assert!(ProjectDataLease::archive(root.path()).is_err());
    drop(lease);
    let archive = ProjectDataLease::archive(root.path()).unwrap();
    assert!(ProjectDataLease::export(root.path()).is_err());
    drop(inherited);
    assert!(ProjectDataLease::export(root.path()).is_err());
    drop(archive);
    ProjectDataLease::export(root.path()).unwrap();
}

#[test]
fn concurrent_claims_and_published_outputs_cannot_be_reused() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("run");
    let mut publication = Publication::claim(&output).unwrap();
    assert!(Publication::claim(&output).is_err());
    write_json(&publication.directory().join("metrics.json"), &json!([])).unwrap();
    write_artifact_manifest(publication.directory()).unwrap();
    publication.publish().unwrap();
    assert!(output.join("metrics.json").is_file());
    assert!(Publication::claim(&output).is_err());
}

#[test]
fn racing_publishers_have_exactly_one_owner() {
    use std::sync::{Arc, Barrier, mpsc};
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("race");
    let start = Arc::new(Barrier::new(3));
    let finish = Arc::new(Barrier::new(3));
    let (sender, receiver) = mpsc::channel();
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let (output, start, finish, sender) = (
                output.clone(),
                start.clone(),
                finish.clone(),
                sender.clone(),
            );
            std::thread::spawn(move || {
                start.wait();
                let claim = Publication::claim(&output);
                sender.send(claim.is_ok()).unwrap();
                finish.wait();
                drop(claim);
            })
        })
        .collect();
    start.wait();
    assert_eq!(
        receiver.iter().take(2).filter(|claimed| *claimed).count(),
        1
    );
    finish.wait();
    for worker in workers {
        worker.join().unwrap();
    }
    assert!(!output.exists());
}

#[test]
fn destination_created_after_claim_is_preserved() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("run");
    let mut publication = Publication::claim(&output).unwrap();
    write_json(&publication.directory().join("metrics.json"), &json!([])).unwrap();
    write_artifact_manifest(publication.directory()).unwrap();
    fs::create_dir(&output).unwrap();
    assert!(publication.publish().is_err());
    assert!(output.read_dir().unwrap().next().is_none());
    let failure = publication
        .fail("publication", &anyhow::anyhow!("destination exists"))
        .unwrap();
    assert!(failure.join("failure_manifest.json").is_file());
}

#[test]
fn failure_retains_evidence_without_publishing_completed_manifest() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("run");
    let mut publication = Publication::claim(&output).unwrap();
    write_json(
        &publication.directory().join("run_manifest.json"),
        &json!({"status":"complete"}),
    )
    .unwrap();
    let failed = publication
        .fail("renderer", &anyhow::anyhow!("renderer failed"))
        .unwrap();
    drop(publication);
    assert!(!output.exists());
    assert!(!failed.join("run_manifest.json").exists());
    assert!(failed.join("failure_manifest.json").is_file());
    assert!(Publication::claim(&output).is_ok());
}

#[test]
fn artifact_manifest_rejects_tampering_stale_files_and_symlinks() {
    let parent = tempfile::tempdir().unwrap();
    write_json(&parent.path().join("figure.json"), &json!([1])).unwrap();
    write_artifact_manifest(parent.path()).unwrap();
    verify_artifacts(parent.path()).unwrap();
    fs::write(parent.path().join("figure.json"), "[2]\n").unwrap();
    assert!(verify_artifacts(parent.path()).is_err());
    write_artifact_manifest(parent.path()).unwrap();
    fs::write(parent.path().join("stale.svg"), "stale").unwrap();
    assert!(verify_artifacts(parent.path()).is_err());
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("figure.json", parent.path().join("linked")).unwrap();
        assert!(artifact_inventory(parent.path()).is_err());
    }
}

#[test]
fn failure_evidence_cannot_be_retained_once_staging_is_gone() {
    let parent = tempfile::tempdir().unwrap();
    let mut publication = Publication::claim(&parent.path().join("run")).unwrap();
    fs::remove_dir_all(publication.directory()).unwrap();
    assert!(
        publication
            .fail("renderer", &anyhow::anyhow!("renderer failed"))
            .is_err()
    );
}

#[test]
fn exclusive_rename_reports_the_operating_system_error() {
    let parent = tempfile::tempdir().unwrap();
    let error =
        rename_exclusive(&parent.path().join("absent"), &parent.path().join("run")).unwrap_err();
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::NotFound
    );
}
