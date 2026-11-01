use clap::Parser;

/// Mirrors the CLI enum for testing parse behavior.
#[derive(Parser, Debug)]
#[command(name = "bekiper")]
pub(crate) enum TestCli {
    Run(TestRunArgs),
    Data(TestDataArgs),
    Annotate(TestAnnotateArgs),
    Inspect(TestInspectArgs),
    Check(TestCheckArgs),
    Doctor(TestDoctorArgs),
}

// ── run ─────────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
pub(crate) struct TestRunArgs {
    #[arg(short, long, default_value = "bekiper_config.toml")]
    pub(crate) config: std::path::PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) no_inspect: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) no_aggregate: bool,
    #[arg(long)]
    pub(crate) aggregate_output: Option<String>,
    #[arg(long)]
    pub(crate) from: Option<String>,
    #[arg(long)]
    pub(crate) only: Option<String>,
    #[arg(long, default_value_t = false)]
    pub(crate) force: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) refresh_augment: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) no_detect: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) no_auto_detect: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) only_first_animal: bool,
    #[arg(long)]
    pub(crate) animal: Option<usize>,
    #[arg(long, default_value_t = false)]
    pub(crate) invert_y: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) no_invert_y: bool,
    #[arg(long)]
    pub(crate) num_threads: Option<usize>,
    #[arg(long, default_value_t = false)]
    pub(crate) memory_scheduler: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) no_memory_scheduler: bool,
    #[arg(long)]
    pub(crate) max_ram_usage: Option<f64>,
    #[arg(long, default_value_t = false)]
    pub(crate) compress_data: bool,
}

// ── data ────────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
pub(crate) struct TestDataArgs {
    #[command(subcommand)]
    pub(crate) command: TestDataCommand,
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum TestDataCommand {
    Aggregate(TestAggregateArgs),
    Compress(TestCompressArgs),
    Convert(TestConvertArgs),
    MergeTimestamps(TestMergeTimestampsArgs),
    PurgeCache(TestPurgeCacheArgs),
    PurgeStaleVariants(TestPurgeStaleVariantsArgs),
}

#[derive(clap::Args, Debug)]
pub(crate) struct TestAggregateArgs {
    #[arg(short, long, default_value = "bekiper_config.toml")]
    pub(crate) config: std::path::PathBuf,
    #[arg(long)]
    pub(crate) run_root: Option<std::path::PathBuf>,
    #[arg(long)]
    pub(crate) output: Option<String>,
}

#[derive(clap::Args, Debug)]
pub(crate) struct TestCompressArgs {
    #[arg(short, long, default_value = "bekiper_config.toml")]
    pub(crate) config: std::path::PathBuf,
    #[arg(long)]
    pub(crate) run_root: Option<std::path::PathBuf>,
    #[arg(long)]
    pub(crate) output: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = false)]
    pub(crate) keep_source: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) force: bool,
}

#[derive(clap::Args, Debug)]
pub(crate) struct TestConvertArgs {
    pub(crate) path: std::path::PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) force: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) python_fallback: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) invert_y: bool,
    #[arg(long)]
    pub(crate) frame_height_px: Option<u32>,
}

#[derive(clap::Args, Debug)]
pub(crate) struct TestMergeTimestampsArgs {
    pub(crate) dataset_dir: std::path::PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) force: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) delete_source: bool,
    #[arg(long, default_value = ".")]
    pub(crate) delimiter: String,
}

#[derive(clap::Args, Debug)]
pub(crate) struct TestPurgeCacheArgs {
    #[arg(short, long, default_value = "bekiper_config.toml")]
    pub(crate) config: std::path::PathBuf,
    #[arg(long)]
    pub(crate) from: Option<String>,
    #[arg(long, default_value_t = false)]
    pub(crate) all: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run: bool,
}

#[derive(clap::Args, Debug)]
pub(crate) struct TestPurgeStaleVariantsArgs {
    #[arg(short, long, default_value = "bekiper_config.toml")]
    pub(crate) config: std::path::PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run: bool,
}

// ── annotate ────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
pub(crate) struct TestAnnotateArgs {
    #[command(subcommand)]
    pub(crate) command: TestAnnotateCommand,
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum TestAnnotateCommand {
    Gui {
        config: std::path::PathBuf,
    },
    Sync {
        annotation_dir: std::path::PathBuf,
        #[arg(long, conflicts_with = "export_only")]
        import_only: bool,
        #[arg(long, conflicts_with = "import_only")]
        export_only: bool,
        #[arg(long)]
        dry_run: bool,
    },
    Status {
        annotation_dir: std::path::PathBuf,
    },
}

// ── inspect ─────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
pub(crate) struct TestInspectArgs {
    #[command(subcommand)]
    pub(crate) command: TestInspectCommand,
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum TestInspectCommand {
    Plot {
        #[arg(default_value = "inspection")]
        dir: std::path::PathBuf,
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
        #[arg(short, long, default_value = ".svgz")]
        format: String,
    },
    Video {
        #[arg(default_value = "inspection")]
        dir: std::path::PathBuf,
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
        #[arg(long)]
        output_dir: Option<std::path::PathBuf>,
        #[arg(long, default_value = "h264")]
        codec: String,
        #[arg(long)]
        heuristic: Option<String>,
        #[arg(long)]
        feature: Option<String>,
        #[arg(long, default_value_t = false)]
        all_features: bool,
        #[arg(long)]
        config: Option<std::path::PathBuf>,
        #[arg(long)]
        ffmpeg_preset: Option<String>,
    },
    Info {
        #[arg(default_value = "inspection")]
        dir: std::path::PathBuf,
    },
    CheckAnnotation {
        #[arg(short, long)]
        path: std::path::PathBuf,
    },
    Overlay {
        videos: Vec<std::path::PathBuf>,
        #[arg(short, long)]
        annotations: std::path::PathBuf,
        #[arg(short, long, default_value = "output/annotation-inspects")]
        output_dir: std::path::PathBuf,
    },
}

// ── check ───────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
pub(crate) struct TestCheckArgs {
    #[command(subcommand)]
    pub(crate) command: TestCheckCommand,
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum TestCheckCommand {
    Config {
        #[arg(short, long, default_value = "bekiper_config.toml")]
        config: std::path::PathBuf,
    },
    Readiness {
        #[arg(short, long, default_value = "bekiper_config.toml")]
        config: std::path::PathBuf,
        #[arg(long, default_value = "human")]
        format: String,
    },
    PoseQuality {
        #[arg(short, long, default_value = "bekiper_config.toml")]
        config: std::path::PathBuf,
        #[arg(long, default_value = "human")]
        format: String,
    },
    Likelihood {
        #[arg(short, long, default_value = "bekiper_config.toml")]
        config: std::path::PathBuf,
        #[arg(long, default_value = "human")]
        format: String,
        #[arg(long)]
        all_labels: bool,
    },
    External {
        #[arg(short, long, default_value = "bekiper_config.toml")]
        config: std::path::PathBuf,
        #[arg(long, default_value = "human")]
        format: String,
    },
}

// ── doctor ──────────────────────────────────────────────────────────────────

#[derive(clap::Args, Debug)]
pub(crate) struct TestDoctorArgs {
    #[arg(short, long)]
    pub(crate) config: Option<std::path::PathBuf>,
    #[arg(long, default_value = "human")]
    pub(crate) format: String,
}

// ── tests ───────────────────────────────────────────────────────────────────

#[test]
fn parse_annotate_sync_dry_run() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "annotate",
        "sync",
        "/data/annotations/nort",
        "--dry-run",
    ])
    .unwrap();
    let TestCli::Annotate(args) = cli else {
        panic!("expected Annotate");
    };
    let TestAnnotateCommand::Sync {
        annotation_dir,
        import_only,
        export_only,
        dry_run,
    } = args.command
    else {
        panic!("expected annotate sync");
    };
    assert_eq!(annotation_dir.to_str().unwrap(), "/data/annotations/nort");
    assert!(!import_only);
    assert!(!export_only);
    assert!(dry_run);
}

#[test]
fn parse_annotate_status() {
    let cli = TestCli::try_parse_from(["bekiper", "annotate", "status", "/data/annotations/nort"])
        .unwrap();
    assert!(matches!(
        cli,
        TestCli::Annotate(TestAnnotateArgs {
            command: TestAnnotateCommand::Status { .. }
        })
    ));
}

#[test]
fn parse_annotate_gui() {
    let cli =
        TestCli::try_parse_from(["bekiper", "annotate", "gui", "bekiper_config.toml"]).unwrap();
    let TestCli::Annotate(args) = cli else {
        panic!("expected Annotate");
    };
    let TestAnnotateCommand::Gui { config } = args.command else {
        panic!("expected annotate gui");
    };
    assert_eq!(config.to_str().unwrap(), "bekiper_config.toml");
}

#[test]
fn parse_data_aggregate_defaults() {
    let cli = TestCli::try_parse_from(["bekiper", "data", "aggregate"]).unwrap();
    let TestCli::Data(args) = cli else {
        panic!("expected Data");
    };
    let TestDataCommand::Aggregate(a) = args.command else {
        panic!("expected aggregate");
    };
    assert_eq!(a.config.to_str().unwrap(), "bekiper_config.toml");
    assert!(a.run_root.is_none());
    assert!(a.output.is_none());
}

#[test]
fn parse_run_defaults() {
    let cli = TestCli::try_parse_from(["bekiper", "run"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert_eq!(args.config.to_str().unwrap(), "bekiper_config.toml");
    assert!(!args.no_inspect);
    assert!(!args.no_aggregate);
    assert!(args.aggregate_output.is_none());
    assert!(args.from.is_none());
    assert!(args.only.is_none());
    assert!(!args.force);
    assert!(!args.refresh_augment);
    assert!(!args.dry_run);
    assert!(!args.no_detect);
    assert!(!args.no_auto_detect);
    assert!(!args.only_first_animal);
    assert!(args.animal.is_none());
    assert!(!args.invert_y);
    assert!(!args.no_invert_y);
    assert!(args.num_threads.is_none());
    assert!(!args.memory_scheduler);
    assert!(!args.no_memory_scheduler);
    assert!(args.max_ram_usage.is_none());
    assert!(!args.compress_data);
}

#[test]
fn parse_run_refresh_augment() {
    let cli = TestCli::try_parse_from(["bekiper", "run", "--refresh-augment"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert!(args.refresh_augment);
}

#[test]
fn parse_run_with_config() {
    let cli = TestCli::try_parse_from(["bekiper", "run", "-c", "my_config.toml"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert_eq!(args.config.to_str().unwrap(), "my_config.toml");
}

#[test]
fn parse_run_no_inspect() {
    let cli = TestCli::try_parse_from(["bekiper", "run", "--no-inspect"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert!(args.no_inspect);
}

#[test]
fn parse_run_from_stage() {
    let cli = TestCli::try_parse_from(["bekiper", "run", "--from", "augment"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert_eq!(args.from.as_deref(), Some("augment"));
}

#[test]
fn parse_run_only_stage() {
    let cli = TestCli::try_parse_from(["bekiper", "run", "--only", "pose"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert_eq!(args.only.as_deref(), Some("pose"));
}

#[test]
fn parse_run_force() {
    let cli = TestCli::try_parse_from(["bekiper", "run", "--force"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert!(args.force);
}

#[test]
fn parse_run_no_detect() {
    let cli = TestCli::try_parse_from(["bekiper", "run", "--no-detect"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert!(args.no_detect);
}

#[test]
fn parse_run_no_auto_detect() {
    let cli = TestCli::try_parse_from(["bekiper", "run", "--no-auto-detect"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert!(args.no_auto_detect);
}

#[test]
fn parse_run_aggregate_and_invert_overrides() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "run",
        "--no-aggregate",
        "--aggregate-output",
        "renamed.xlsx",
        "--invert-y",
        "--animal",
        "2",
    ])
    .unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert!(args.no_aggregate);
    assert_eq!(args.aggregate_output.as_deref(), Some("renamed.xlsx"));
    assert!(args.invert_y);
    assert_eq!(args.animal, Some(2));
}

#[test]
fn parse_run_runtime_scheduler_overrides() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "run",
        "--num-threads",
        "3",
        "--memory-scheduler",
        "--max-ram-usage",
        "0.7",
    ])
    .unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert_eq!(args.num_threads, Some(3));
    assert!(args.memory_scheduler);
    assert_eq!(args.max_ram_usage, Some(0.7));
}

#[test]
fn parse_run_runtime_scheduler_disable_override() {
    let cli = TestCli::try_parse_from(["bekiper", "run", "--no-memory-scheduler"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert!(args.no_memory_scheduler);
    assert!(!args.memory_scheduler);
}

#[test]
fn parse_run_compress_data() {
    let cli = TestCli::try_parse_from(["bekiper", "run", "--compress-data"]).unwrap();
    let TestCli::Run(args) = cli else {
        panic!("expected Run");
    };
    assert!(args.compress_data);
}

#[test]
fn parse_data_compress_direct() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "data",
        "compress",
        "--run-root",
        "examples/neo/imen/output",
        "--keep-source",
        "--force",
    ])
    .unwrap();
    let TestCli::Data(args) = cli else {
        panic!("expected Data");
    };
    let TestDataCommand::Compress(a) = args.command else {
        panic!("expected compress");
    };
    assert_eq!(
        a.run_root.unwrap().to_str().unwrap(),
        "examples/neo/imen/output"
    );
    assert!(a.keep_source);
    assert!(a.force);
}

#[test]
fn parse_check_config_defaults() {
    let cli = TestCli::try_parse_from(["bekiper", "check", "config"]).unwrap();
    let TestCli::Check(args) = cli else {
        panic!("expected Check");
    };
    let TestCheckCommand::Config { config } = args.command else {
        panic!("expected config");
    };
    assert_eq!(config.to_str().unwrap(), "bekiper_config.toml");
}

#[test]
fn parse_check_readiness_json() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "check",
        "readiness",
        "-c",
        "examples/neo/imen/bekiper_config.toml",
        "--format",
        "json",
    ])
    .unwrap();
    let TestCli::Check(args) = cli else {
        panic!("expected Check");
    };
    let TestCheckCommand::Readiness { config, format } = args.command else {
        panic!("expected readiness");
    };
    assert_eq!(
        config.to_str().unwrap(),
        "examples/neo/imen/bekiper_config.toml"
    );
    assert_eq!(format, "json");
}

#[test]
fn parse_check_external_defaults() {
    let cli = TestCli::try_parse_from(["bekiper", "check", "external"]).unwrap();
    let TestCli::Check(args) = cli else {
        panic!("expected Check");
    };
    let TestCheckCommand::External { config, format } = args.command else {
        panic!("expected external");
    };
    assert_eq!(config.to_str().unwrap(), "bekiper_config.toml");
    assert_eq!(format, "human");
}

#[test]
fn parse_check_pose_quality_json() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "check",
        "pose-quality",
        "-c",
        "examples/neo/imen/bekiper_config.toml",
        "--format",
        "json",
    ])
    .unwrap();
    let TestCli::Check(args) = cli else {
        panic!("expected Check");
    };
    let TestCheckCommand::PoseQuality { config, format } = args.command else {
        panic!("expected pose-quality");
    };
    assert_eq!(
        config.to_str().unwrap(),
        "examples/neo/imen/bekiper_config.toml"
    );
    assert_eq!(format, "json");
}

#[test]
fn parse_check_likelihood_json() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "check",
        "likelihood",
        "-c",
        "examples/neo/imen/bekiper_config.toml",
        "--format",
        "json",
    ])
    .unwrap();
    let TestCli::Check(args) = cli else {
        panic!("expected Check");
    };
    let TestCheckCommand::Likelihood {
        config,
        format,
        all_labels,
    } = args.command
    else {
        panic!("expected likelihood");
    };
    assert_eq!(
        config.to_str().unwrap(),
        "examples/neo/imen/bekiper_config.toml"
    );
    assert_eq!(format, "json");
    assert!(!all_labels);
}

#[test]
fn parse_check_likelihood_all_labels() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "check",
        "likelihood",
        "-c",
        "examples/neo/imen/bekiper_config.toml",
        "--all-labels",
    ])
    .unwrap();
    let TestCli::Check(args) = cli else {
        panic!("expected Check");
    };
    let TestCheckCommand::Likelihood { all_labels, .. } = args.command else {
        panic!("expected likelihood");
    };
    assert!(all_labels);
}

#[test]
fn parse_doctor_with_config() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "doctor",
        "-c",
        "examples/neo/imen/bekiper_config.toml",
    ])
    .unwrap();
    let TestCli::Doctor(args) = cli else {
        panic!("expected Doctor");
    };
    assert_eq!(
        args.config.unwrap().to_str().unwrap(),
        "examples/neo/imen/bekiper_config.toml"
    );
    assert_eq!(args.format, "human");
}

#[test]
fn parse_data_convert_with_fallback_and_invert_y() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "data",
        "convert",
        "poses",
        "--python-fallback",
        "--invert-y",
        "--frame-height-px",
        "480",
    ])
    .unwrap();
    let TestCli::Data(args) = cli else {
        panic!("expected Data");
    };
    let TestDataCommand::Convert(a) = args.command else {
        panic!("expected convert");
    };
    assert_eq!(a.path.to_str().unwrap(), "poses");
    assert!(a.python_fallback);
    assert!(a.invert_y);
    assert_eq!(a.frame_height_px, Some(480));
}

#[test]
fn parse_data_merge_timestamps_defaults() {
    let cli =
        TestCli::try_parse_from(["bekiper", "data", "merge-timestamps", "/data/dataset"]).unwrap();
    let TestCli::Data(args) = cli else {
        panic!("expected Data");
    };
    let TestDataCommand::MergeTimestamps(a) = args.command else {
        panic!("expected merge-timestamps");
    };
    assert_eq!(a.dataset_dir.to_str().unwrap(), "/data/dataset");
    assert!(!a.force);
    assert!(!a.delete_source);
    assert_eq!(a.delimiter, ".");
}

#[test]
fn parse_data_merge_timestamps_with_flags() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "data",
        "merge-timestamps",
        "/data/dataset",
        "--force",
        "--delete-source",
        "--delimiter",
        "_",
    ])
    .unwrap();
    let TestCli::Data(args) = cli else {
        panic!("expected Data");
    };
    let TestDataCommand::MergeTimestamps(a) = args.command else {
        panic!("expected merge-timestamps");
    };
    assert_eq!(a.dataset_dir.to_str().unwrap(), "/data/dataset");
    assert!(a.force);
    assert!(a.delete_source);
    assert_eq!(a.delimiter, "_");
}

#[test]
fn parse_data_purge_cache_from_stage() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "data",
        "purge-cache",
        "--from",
        "summary",
        "--dry-run",
    ])
    .unwrap();
    let TestCli::Data(args) = cli else {
        panic!("expected Data");
    };
    let TestDataCommand::PurgeCache(a) = args.command else {
        panic!("expected purge-cache");
    };
    assert_eq!(a.from.as_deref(), Some("summary"));
    assert!(!a.all);
    assert!(a.dry_run);
}

#[test]
fn parse_invalid_command_fails() {
    let result = TestCli::try_parse_from(["bekiper", "nonexistent"]);
    assert!(result.is_err());
}

#[test]
fn parse_inspect_without_subcommand_fails() {
    let result = TestCli::try_parse_from(["bekiper", "inspect"]);
    assert!(result.is_err());
}

#[test]
fn parse_data_without_subcommand_fails() {
    let result = TestCli::try_parse_from(["bekiper", "data"]);
    assert!(result.is_err());
}

#[test]
fn parse_inspect_plot_defaults() {
    let cli = TestCli::try_parse_from(["bekiper", "inspect", "plot"]).unwrap();
    let TestCli::Inspect(args) = cli else {
        panic!("expected Inspect");
    };
    let TestInspectCommand::Plot {
        dir,
        output,
        format,
    } = args.command
    else {
        panic!("expected Plot");
    };
    assert_eq!(dir.to_str().unwrap(), "inspection");
    assert!(output.is_none());
    assert_eq!(format, ".svgz");
}

#[test]
fn parse_inspect_overlay() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "inspect",
        "overlay",
        "video1.mp4",
        "video2.mp4",
        "-a",
        "annotations/nort",
    ])
    .unwrap();
    let TestCli::Inspect(args) = cli else {
        panic!("expected Inspect");
    };
    let TestInspectCommand::Overlay {
        videos,
        annotations,
        output_dir,
    } = args.command
    else {
        panic!("expected Overlay");
    };
    assert_eq!(videos.len(), 2);
    assert_eq!(videos[0].to_str().unwrap(), "video1.mp4");
    assert_eq!(annotations.to_str().unwrap(), "annotations/nort");
    assert_eq!(output_dir.to_str().unwrap(), "output/annotation-inspects");
}
