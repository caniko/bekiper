use std::path::PathBuf;

use clap::{ArgAction, Args};

use bekiper_core::identity::DEFAULT_CONFIG_FILENAME;
use bekiper_ingress::config::IngressConfig;
use bekiper_ingress::stage::Stage;
use bekiper_ingress::workflow::{
    RunOptions, dry_run_detect, run_project_with_post_analysis_overrides,
};

use crate::commands::data::compress::{CompressDataOptions, compress_data_dir};

#[path = "run/post_analysis.rs"]
mod post_analysis;

#[cfg(test)]
#[path = "run/tests.rs"]
mod tests;

#[derive(Args)]
pub struct RunArgs {
    /// OFT seconds from first valid centroid (default 300); zero selects the full video.
    #[arg(long)]
    oft_window_seconds: Option<f64>,
    /// Use strict NORT gap/speed upper bounds; false reproduces inclusive historical bounds.
    #[arg(long, action = ArgAction::Set)]
    nort_strict_boundaries: Option<bool>,
    /// Minimum same-arm dual-landmark dwell (seconds); default 1, 0 disables it.
    #[arg(long)]
    ymaze_dwell_seconds: Option<f64>,
    /// Minimum duration comparison: at_least (>=) or greater_than (>).
    #[arg(long)]
    ymaze_dwell_comparison: Option<bekiper_feature::tolerance::DurationComparison>,
    /// Override any config key. `<task>.<path>` is relative to that task's
    /// `task_params` (`nort.exploration.whisker.apex_fraction=0.9`,
    /// `ymaze.ymaze.principal_thresholds=[0.25,0.45]`); `<task>@<variant>.<path>`
    /// targets one variant (`ymaze@arm0_dwell1s.ymaze.temporal_policy.max_gap_seconds=0.5`);
    /// any other path is a root key (`aggregate.output_filename=x.ods`).
    /// Repeatable; beats `BEKIPER_SET` (`;`-separated) and the TOML. Unknown keys fail.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    set: Vec<String>,
    #[command(flatten)]
    post_analysis: post_analysis::PostAnalysisArgs,

    /// Path to the project configuration file.
    #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
    config: PathBuf,

    /// Skip the inspect stage (no figure generation).
    #[arg(long, default_value_t = false)]
    no_inspect: bool,
    /// Recorded analysis ID to render as a full-source overview video. Repeatable.
    #[arg(long = "inspection-video-trial")]
    inspection_video_trials: Vec<String>,

    /// Skip the aggregate stage.
    #[arg(long, default_value_t = false)]
    no_aggregate: bool,

    /// Override the aggregate spreadsheet filename. Legacy `.xlsx` values are
    /// accepted and normalized to `.ods`.
    #[arg(long)]
    aggregate_output: Option<String>,

    /// Re-run from this stage onward (detect, pose, augment, behaviour, summary, inspect, aggregate).
    #[arg(long)]
    from: Option<String>,

    /// Run only this single stage.
    #[arg(long)]
    only: Option<String>,

    /// Force re-run all stages, ignoring checkpoints.
    #[arg(long, default_value_t = false)]
    force: bool,

    /// Delete augment and downstream outputs before running from augment.
    ///
    /// Use this after changing interpolation, smoothing, skeleton repair, or
    /// coordinate-transform logic. Detection and pose artifacts are preserved.
    #[arg(long, default_value_t = false)]
    refresh_augment: bool,

    /// Print the detection plan for each experiment without running anything.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Do not run detection; fail fast if required pose data or perimeter artifacts are missing.
    #[arg(long, default_value_t = false)]
    no_detect: bool,

    /// Do not run automatic detection; allow only precomputed/manual perimeter sources.
    #[arg(long, default_value_t = false)]
    no_auto_detect: bool,

    /// Only process experiments belonging to the first animal.
    #[arg(long, default_value_t = false)]
    only_first_animal: bool,

    /// Only process experiments belonging to the animal at this index (0-based).
    #[arg(long)]
    animal: Option<usize>,

    /// Invert Y coordinates during augmentation.
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_invert_y")]
    invert_y: bool,

    /// Explicitly disable Y-axis inversion during augmentation.
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "invert_y")]
    no_invert_y: bool,

    /// Hard cap on experiment-level parallel workers. Omit to use all cores.
    #[arg(long, value_parser = parse_positive_num_threads)]
    num_threads: Option<usize>,

    /// Enable host-RAM-aware admission control for parallel experiment work.
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_memory_scheduler")]
    memory_scheduler: bool,

    /// Disable host-RAM-aware admission control for parallel experiment work.
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "memory_scheduler")]
    no_memory_scheduler: bool,

    /// Maximum whole-machine RAM usage fraction before new parallel work is throttled.
    #[arg(long, value_parser = parse_max_ram_usage)]
    max_ram_usage: Option<f64>,

    /// Archive output/data to output/data.tar.zst after a successful analysis run.
    #[arg(long, default_value_t = false)]
    compress_data: bool,
}

pub fn run(args: RunArgs) -> anyhow::Result<()> {
    let post_analysis = args.post_analysis.overrides();
    if args.dry_run {
        dry_run_detect(&args.config)?;
        return Ok(());
    }

    let mut from_stage = args.from.as_deref().map(Stage::from_name).transpose()?;

    let only_stage = args.only.as_deref().map(Stage::from_name).transpose()?;
    if args.refresh_augment {
        if only_stage.is_some() {
            anyhow::bail!("--refresh-augment cannot be combined with --only");
        }
        if let Some(stage) = from_stage
            && stage > Stage::Augment
        {
            anyhow::bail!("--refresh-augment cannot be combined with --from {stage}");
        }
        from_stage = Some(Stage::Augment);
    }

    let animal_index = if args.only_first_animal {
        Some(0)
    } else {
        args.animal
    };

    let invert_y = if args.invert_y {
        Some(true)
    } else if args.no_invert_y {
        Some(false)
    } else {
        None
    };

    let memory_scheduler_enabled = if args.memory_scheduler {
        Some(true)
    } else if args.no_memory_scheduler {
        Some(false)
    } else {
        None
    };

    let opts = RunOptions {
        measurements: bekiper_ingress::config::MeasurementOverrides {
            oft_window_seconds: args.oft_window_seconds,
            nort_strict_boundaries: args.nort_strict_boundaries,
        },
        ymaze_dwell: bekiper_feature::tolerance::DwellOverrides {
            seconds: args.ymaze_dwell_seconds,
            comparison: args.ymaze_dwell_comparison,
        },
        set: args.set.clone(),
        from_stage,
        only_stage,
        no_inspect: args.no_inspect,
        inspection_video_trials: (!args.inspection_video_trials.is_empty())
            .then_some(args.inspection_video_trials),
        no_aggregate: args.no_aggregate,
        aggregate_output: args.aggregate_output,
        invert_y,
        force: args.force,
        refresh_augment: args.refresh_augment,
        no_detect: args.no_detect,
        no_auto_detect: args.no_auto_detect,
        animal_index,
        num_threads: args.num_threads,
        memory_scheduler_enabled,
        max_ram_usage_fraction: args.max_ram_usage,
    };

    run_project_with_post_analysis_overrides(&args.config, &opts, &post_analysis)?;

    if args.compress_data {
        let run_root =
            configured_run_root(&args.config, &post_analysis, opts.ymaze_dwell, &opts.set)?;
        let archive_path = run_root.join("data.tar.zst");
        compress_data_dir(CompressDataOptions {
            run_root,
            archive_path,
            keep_source: false,
            force: false,
        })?;
    }

    println!("Analysis complete.");
    Ok(())
}

fn configured_run_root(
    config_path: &std::path::Path,
    post_analysis: &bekiper_ingress::config::PostAnalysisOverrides,
    dwell: bekiper_feature::tolerance::DwellOverrides,
    set: &[String],
) -> anyhow::Result<PathBuf> {
    let config = IngressConfig::from_file_with_set_overrides_from(
        config_path,
        post_analysis,
        dwell,
        Default::default(),
        set,
        |key| std::env::var(key).ok(),
    )?;
    let base_dir = config_path.parent().unwrap_or(std::path::Path::new("."));
    Ok(base_dir.join(
        config
            .project
            .output_dir
            .as_deref()
            .unwrap_or(std::path::Path::new("output")),
    ))
}

fn parse_positive_num_threads(raw: &str) -> std::result::Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|e| format!("invalid thread count '{raw}': {e}"))?;
    if value == 0 {
        return Err("thread count must be > 0".into());
    }
    Ok(value)
}

fn parse_max_ram_usage(raw: &str) -> std::result::Result<f64, String> {
    let value = raw
        .parse::<f64>()
        .map_err(|e| format!("invalid RAM usage fraction '{raw}': {e}"))?;
    if !(0.0 < value && value <= 1.0) {
        return Err("RAM usage fraction must be in the interval (0.0, 1.0]".into());
    }
    Ok(value)
}
