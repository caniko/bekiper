use std::collections::BTreeSet;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use clap::Args;

use bekiper_core::identity::DEFAULT_CONFIG_FILENAME;
use bekiper_ingress::config::IngressConfig;
use bekiper_ingress::stage::{
    ExperimentOutputDir, Stage, aggregate_output_path, aggregate_state_path,
};

#[derive(Args)]
pub struct PurgeCacheArgs {
    /// Path to the project configuration file.
    #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
    config: PathBuf,

    /// Purge this stage and every downstream stage.
    #[arg(long)]
    from: Option<String>,

    /// Purge all *non-detection* stage-owned outputs (pose, augment,
    /// behaviour, summary, inspect, aggregate). Detection outputs are
    /// preserved. To also purge detection, additionally pass
    /// `--include-detection`.
    #[arg(long, default_value_t = false)]
    all: bool,

    /// Print what would be removed without deleting anything.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Explicitly authorise deletion of detection-stage outputs
    /// (`pixel_perimeters.toml`, arena-calibration debug, annotation scale,
    /// detection frame). Required to purge detection because rebuilding it
    /// is the most expensive part of the pipeline (manual annotation, SAM2,
    /// arena calibration). Without this flag, `--from detect` is rejected
    /// and `--all` skips detection-owned paths.
    #[arg(long, default_value_t = false)]
    include_detection: bool,

    /// Skip the interactive "yes/no" confirmation that gates detection
    /// deletion. Combine with `--include-detection` for non-interactive
    /// use (CI, scripts). Has no effect when `--include-detection` is not
    /// set.
    #[arg(long, default_value_t = false)]
    yes: bool,
}

pub fn run(args: PurgeCacheArgs) -> anyhow::Result<()> {
    run_with_confirm(args, prompt_yes_no_stdin)
}

fn run_with_confirm(
    args: PurgeCacheArgs,
    confirm: impl FnOnce(&str) -> bool,
) -> anyhow::Result<()> {
    let start_stage = match (args.all, args.from.as_deref()) {
        (true, None) => {
            // `--all` starts at the first non-detection stage by default.
            // Detection is only included when --include-detection is set.
            if args.include_detection {
                Stage::Detect
            } else {
                Stage::Pose
            }
        }
        (false, Some(stage)) => Stage::from_name(stage)?,
        (true, Some(_)) => anyhow::bail!("--all and --from are mutually exclusive"),
        (false, None) => anyhow::bail!("pass either --from <stage> or --all"),
    };

    let purges_detection = matches!(start_stage, Stage::Detect);

    if purges_detection && !args.include_detection {
        anyhow::bail!(
            "refusing to purge detection-stage outputs without --include-detection.\n\
             Detection is the most expensive stage to rebuild (manual annotation, SAM2, \
             arena calibration). Pass --include-detection to authorise deletion. \
             To skip detection while purging everything else, use --from pose."
        );
    }

    if purges_detection && !args.dry_run && !args.yes && !confirm(detection_confirmation_prompt()) {
        println!("Aborted: detection-stage outputs were not deleted.");
        return Ok(());
    }

    let mut config = IngressConfig::from_file(&args.config)?;
    let base_dir = args.config.parent().unwrap_or(Path::new("."));
    config.expand_for_validation(base_dir, Stage::all())?;
    let run_root = base_dir.join(
        config
            .project
            .output_dir
            .as_deref()
            .unwrap_or(Path::new("output")),
    );

    let mut paths = BTreeSet::new();
    for experiment in &config.experiments {
        let exp_dir = if let Some(animal_id) = experiment.animal_id {
            ExperimentOutputDir::for_animal(&run_root, animal_id, &experiment.name)
        } else {
            ExperimentOutputDir::new(&run_root, &experiment.name)
        };
        for path in exp_dir.purge_paths_from(start_stage) {
            paths.insert(path);
        }
    }
    if Stage::from_onward(start_stage).contains(&Stage::Aggregate) {
        paths.insert(aggregate_output_path(
            &run_root,
            &config.aggregate.output_filename,
        ));
        paths.insert(aggregate_state_path(
            &run_root,
            &config.aggregate.output_filename,
        ));
    }

    let existing: Vec<PathBuf> = paths.into_iter().filter(|path| path.exists()).collect();
    if args.dry_run {
        for path in &existing {
            println!("{}", path.display());
        }
        println!("Would remove {} path(s).", existing.len());
        return Ok(());
    }

    for path in &existing {
        remove_path(path)?;
    }

    println!("Removed {} path(s).", existing.len());
    Ok(())
}

fn detection_confirmation_prompt() -> &'static str {
    "About to delete detection-stage outputs (pixel_perimeters.toml, arena-calibration\n\
     debug, annotation scale, detection frame). These are the most expensive artifacts\n\
     to rebuild — manual annotation work, SAM2 masks, and arena calibration will all\n\
     have to run again.\n\nProceed? [y/N]: "
}

fn prompt_yes_no_stdin(prompt: &str) -> bool {
    if !std::io::stdin().is_terminal() {
        eprintln!(
            "refusing to delete detection outputs: stdin is not a TTY and no --yes flag was passed"
        );
        return false;
    }
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let mut buf = String::new();
    if std::io::stdin().read_line(&mut buf).is_err() {
        return false;
    }
    let answer = buf.trim().to_ascii_lowercase();
    matches!(answer.as_str(), "y" | "yes")
}

fn remove_path(path: &Path) -> anyhow::Result<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)?;
    } else if path.exists() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, b"test").unwrap();
    }

    fn write_dir_marker(path: &Path) {
        std::fs::create_dir_all(path).unwrap();
        std::fs::write(path.join(".keep"), b"test").unwrap();
    }

    fn setup_project(tag: &str) -> (PathBuf, PathBuf, ExperimentOutputDir) {
        let dir = std::env::temp_dir().join(format!("bekiper_cli_purge_cache_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let config_path = dir.join("bekiper_config.toml");
        std::fs::write(
            &config_path,
            r#"
            [project]
            project_name = "purge"
            data_directory = "."
            output_dir = "output"

            [[experiments]]
            name = "trial_01"
            data_files = ["trial.csv"]
            task_type = "open_field"
            fps = 30.0
            meters_per_pixel = 0.001
            "#,
        )
        .unwrap();
        std::fs::write(dir.join("trial.csv"), b"frame,nose_x,nose_y\n").unwrap();

        let run_root = dir.join("output");
        let exp_dir = ExperimentOutputDir::new(&run_root, "trial_01");
        assert_eq!(exp_dir.root(), run_root.join("data/trial_01"));
        assert_eq!(exp_dir.checkpoints_dir(), dir.join("checkpoint/trial_01"));
        write_file(&exp_dir.pixel_perimeters_toml());
        write_file(&exp_dir.perimeters_toml());
        write_file(&exp_dir.poses_parquet());
        write_dir_marker(&exp_dir.pose_chunks_dir());
        write_file(&exp_dir.scale_toml());
        write_file(&exp_dir.augmented_parquet());
        write_file(&exp_dir.behaviour_parquet());
        write_file(&exp_dir.behaviour_summaries_toml());
        write_file(&exp_dir.behaviour_result_toml());
        write_file(&exp_dir.result_toml());
        write_file(&exp_dir.inspection_toml());
        write_file(&exp_dir.evaluated_parquet());
        write_file(&exp_dir.spreadsheet_workbook());
        write_dir_marker(&exp_dir.frames_dir());
        write_dir_marker(&exp_dir.figures_dir());
        write_dir_marker(&exp_dir.videos_dir());
        write_file(&aggregate_output_path(&run_root, "results.xlsx"));
        write_file(&aggregate_state_path(&run_root, "results.xlsx"));

        (dir, config_path, exp_dir)
    }

    fn always_yes(_: &str) -> bool {
        true
    }

    fn always_no(_: &str) -> bool {
        false
    }

    #[test]
    fn purge_cache_dry_run_keeps_files() {
        let (dir, config_path, exp_dir) = setup_project("dry_run");

        run_with_confirm(
            PurgeCacheArgs {
                config: config_path,
                from: Some("summary".into()),
                all: false,
                dry_run: true,
                include_detection: false,
                yes: false,
            },
            always_yes,
        )
        .unwrap();

        assert!(exp_dir.result_toml().exists());
        assert!(exp_dir.figures_dir().exists());
        assert!(aggregate_output_path(&dir.join("output"), "results.xlsx").exists());
        assert!(aggregate_state_path(&dir.join("output"), "results.xlsx").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn purge_cache_from_summary_removes_downstream_only() {
        let (dir, config_path, exp_dir) = setup_project("summary");

        run_with_confirm(
            PurgeCacheArgs {
                config: config_path,
                from: Some("summary".into()),
                all: false,
                dry_run: false,
                include_detection: false,
                yes: false,
            },
            always_yes,
        )
        .unwrap();

        assert!(exp_dir.behaviour_parquet().exists());
        assert!(!exp_dir.result_toml().exists());
        assert!(!exp_dir.inspection_toml().exists());
        assert!(!exp_dir.evaluated_parquet().exists());
        assert!(!exp_dir.frames_dir().exists());
        assert!(!exp_dir.figures_dir().exists());
        assert!(!exp_dir.videos_dir().exists());
        assert!(!aggregate_output_path(&dir.join("output"), "results.xlsx").exists());
        assert!(!aggregate_state_path(&dir.join("output"), "results.xlsx").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn purge_cache_all_preserves_detection_outputs_by_default() {
        let (dir, config_path, exp_dir) = setup_project("all_default");

        run_with_confirm(
            PurgeCacheArgs {
                config: config_path,
                from: None,
                all: true,
                dry_run: false,
                include_detection: false,
                yes: false,
            },
            always_yes,
        )
        .unwrap();

        // Detection-stage outputs survive a default --all purge.
        assert!(exp_dir.pixel_perimeters_toml().exists());
        // Pose stage and onward are gone.
        assert!(!exp_dir.poses_parquet().exists());
        assert!(!exp_dir.pose_chunks_dir().exists());
        assert!(!exp_dir.augmented_parquet().exists());
        assert!(!exp_dir.behaviour_parquet().exists());
        assert!(!aggregate_output_path(&dir.join("output"), "results.xlsx").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn purge_cache_all_with_include_detection_removes_detection() {
        let (dir, config_path, exp_dir) = setup_project("all_include_detection");

        run_with_confirm(
            PurgeCacheArgs {
                config: config_path,
                from: None,
                all: true,
                dry_run: false,
                include_detection: true,
                yes: true, // skip the interactive prompt in tests
            },
            always_yes,
        )
        .unwrap();

        assert!(!exp_dir.pixel_perimeters_toml().exists());
        assert!(!exp_dir.perimeters_toml().exists());
        assert!(!exp_dir.poses_parquet().exists());
        assert!(!exp_dir.pose_chunks_dir().exists());
        assert!(!exp_dir.behaviour_parquet().exists());
        assert!(!aggregate_output_path(&dir.join("output"), "results.xlsx").exists());
        assert!(!aggregate_state_path(&dir.join("output"), "results.xlsx").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn purge_cache_from_detect_without_flag_errors() {
        let (dir, config_path, exp_dir) = setup_project("from_detect_no_flag");

        let err = run_with_confirm(
            PurgeCacheArgs {
                config: config_path,
                from: Some("detect".into()),
                all: false,
                dry_run: false,
                include_detection: false,
                yes: false,
            },
            always_yes,
        )
        .unwrap_err();
        assert!(err.to_string().contains("--include-detection"));

        // Nothing was deleted.
        assert!(exp_dir.pixel_perimeters_toml().exists());
        assert!(exp_dir.behaviour_parquet().exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn purge_cache_include_detection_aborts_on_no() {
        let (dir, config_path, exp_dir) = setup_project("include_detect_aborts");

        run_with_confirm(
            PurgeCacheArgs {
                config: config_path,
                from: Some("detect".into()),
                all: false,
                dry_run: false,
                include_detection: true,
                yes: false,
            },
            always_no,
        )
        .unwrap();

        // Confirmation declined → nothing deleted.
        assert!(exp_dir.pixel_perimeters_toml().exists());
        assert!(exp_dir.poses_parquet().exists());
        assert!(exp_dir.behaviour_parquet().exists());

        std::fs::remove_dir_all(&dir).ok();
    }
}
