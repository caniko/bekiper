use super::*;
use serde_json::json;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("bekiper-compress-{}-{stamp}", std::process::id()));
        let run = root.join("data/_post_analysis/cohort-v1");
        std::fs::create_dir_all(&run).unwrap();
        bekiper_post_analysis::pipeline::write_json(
            &run.join("run_manifest.json"),
            &json!({"schema_version":"2.0","status":"partial"}),
        )
        .unwrap();
        std::fs::create_dir_all(run.join("figures/oft")).unwrap();
        std::fs::write(
            run.join("figures/oft/figure_oft.svg"),
            "<svg>selected figure</svg>",
        )
        .unwrap();
        std::fs::write(run.join("metrics.toml"), "schema_version = \"1.0\"\n").unwrap();
        std::fs::write(run.join("parameters.toml"), "schema_version = \"1.0\"\n").unwrap();
        std::fs::write(run.join("report.pdf"), "%PDF selected report").unwrap();
        bekiper_post_analysis::pipeline::write_artifact_manifest(&run).unwrap();
        Self(root)
    }

    fn options(&self) -> CompressDataOptions {
        CompressDataOptions {
            run_root: self.0.clone(),
            archive_path: self.0.join("data.tar.zst"),
            keep_source: false,
            force: false,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn selected_cohort_package_is_verified_and_survives_source_cleanup() {
    let fixture = Fixture::new();
    let before =
        bekiper_post_analysis::pipeline::verify_project_exports(&fixture.0.join("data")).unwrap();
    let result = compress_data_dir(fixture.options()).unwrap();
    assert!(result.removed_source);
    assert!(!fixture.0.join("data").exists());
    bekiper_post_analysis::pipeline::verify_data_archive(&result.archive_path, &before).unwrap();
    let list = Command::new("tar")
        .args(["--use-compress-program=zstd -d", "-tf"])
        .arg(result.archive_path)
        .output()
        .unwrap();
    assert!(list.status.success());
    let members = String::from_utf8(list.stdout).unwrap();
    assert!(members.contains("data/_post_analysis/cohort-v1/figures/oft/figure_oft.svg"));
    assert!(members.contains("data/_post_analysis/cohort-v1/metrics.toml"));
    assert!(members.contains("data/_post_analysis/cohort-v1/parameters.toml"));
    assert!(members.contains("data/_post_analysis/cohort-v1/report.pdf"));
    assert!(!members.contains("stale.png"));
}

#[test]
fn failed_artifact_validation_preserves_source_and_previous_archive() {
    let fixture = Fixture::new();
    let mut options = fixture.options();
    options.force = true;
    std::fs::write(&options.archive_path, "previous archive").unwrap();
    std::fs::write(
        fixture
            .0
            .join("data/_post_analysis/cohort-v1/figures/oft/figure_oft.svg"),
        "tampered",
    )
    .unwrap();
    assert!(compress_data_dir(options).is_err());
    assert_eq!(
        std::fs::read_to_string(fixture.0.join("data.tar.zst")).unwrap(),
        "previous archive"
    );
    assert!(fixture.0.join("data").is_dir());
}

#[test]
fn archive_under_source_is_rejected_before_cleanup() {
    let fixture = Fixture::new();
    let mut options = fixture.options();
    options.archive_path = fixture.0.join("data/archive.tar.zst");
    assert!(compress_data_dir(options).is_err());
    assert!(fixture.0.join("data").is_dir());
}

#[test]
fn active_cohort_export_preserves_source_and_prior_archive() {
    let fixture = Fixture::new();
    let _export = bekiper_post_analysis::pipeline::ProjectDataLease::export(&fixture.0).unwrap();
    let mut options = fixture.options();
    options.force = true;
    std::fs::write(&options.archive_path, "prior archive").unwrap();
    assert!(compress_data_dir(options).is_err());
    assert!(fixture.0.join("data").is_dir());
    assert_eq!(
        std::fs::read_to_string(fixture.0.join("data.tar.zst")).unwrap(),
        "prior archive"
    );
}

#[test]
fn compression_strategy_balances_cpu_and_memory() {
    assert_eq!(
        CompressionStrategy::for_host(32, 64),
        CompressionStrategy {
            level: 10,
            long_window_log: 27,
            threads: 16
        }
    );
    assert_eq!(CompressionStrategy::for_host(32, 16).threads, 8);
    assert_eq!(CompressionStrategy::for_host(2, 64).threads, 2);
    assert_eq!(CompressionStrategy::for_host(0, 1).threads, 1);
}
