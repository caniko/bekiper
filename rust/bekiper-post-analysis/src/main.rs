use bekiper_post_analysis::RunOptions;
use clap::Parser;
use std::path::PathBuf;

/// Native OFT/Y-maze/NORT post-analysis using registered scientific decisions.
#[derive(Debug, Parser)]
#[command(
    version,
    about,
    subcommand_negates_reqs = true,
    args_conflicts_with_subcommands = true
)]
struct Args {
    #[command(subcommand)]
    maintenance: Option<bekiper_post_analysis::maintenance::MaintenanceCommand>,
    /// Minimum same-arm dual-landmark dwell (seconds); default 1, 0 disables it.
    #[arg(long)]
    ymaze_dwell_seconds: Option<f64>,
    /// Duration comparison: at_least (>=, default) or greater_than (>).
    #[arg(long)]
    ymaze_dwell_comparison: Option<bekiper_feature::tolerance::DurationComparison>,
    #[arg(long, required = true)]
    config: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long = "request")]
    requested_outputs: Vec<String>,
    #[arg(long = "sensitivity-threshold", value_parser = parse_threshold)]
    sensitivity_thresholds: Vec<u8>,
    /// Disable supplementary sensitivity panels without changing D03.
    #[arg(long, conflicts_with = "sensitivity_thresholds")]
    no_sensitivity: bool,
    /// Independent primary export contexts, comma-separated or repeatable.
    #[arg(long = "primary-threshold", value_delimiter = ',', value_parser = parse_threshold)]
    primary_thresholds: Option<Vec<u8>>,
    /// Bound concurrent primary contexts; default 1.
    #[arg(long, value_parser = clap::value_parser!(usize))]
    primary_workers: Option<usize>,
    /// Emit the complete run manifest for scripting and compatibility clients.
    #[arg(long)]
    json: bool,
}

fn parse_threshold(value: &str) -> Result<u8, String> {
    match value.parse::<u8>() {
        Ok(value @ (25 | 45 | 65)) => Ok(value),
        _ => Err("sensitivity threshold must be 25, 45 or 65".into()),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if let Some(command) = args.maintenance {
        println!("{}", serde_json::to_string_pretty(&command.run()?)?);
        return Ok(());
    }
    let config = args
        .config
        .ok_or_else(|| anyhow::anyhow!("run requires --config"))?;
    let options = RunOptions {
        primary_thresholds: args.primary_thresholds,
        contexts: None,
        render_workers: None,
        primary_workers: args.primary_workers,
        ymaze_dwell: bekiper_feature::tolerance::DwellOverrides {
            seconds: args.ymaze_dwell_seconds,
            comparison: args.ymaze_dwell_comparison,
        },
        output: args.output,
        requested_outputs: (!args.requested_outputs.is_empty()).then_some(args.requested_outputs),
        sensitivity_thresholds: if args.no_sensitivity {
            Some(Vec::new())
        } else {
            (!args.sensitivity_thresholds.is_empty()).then_some(args.sensitivity_thresholds)
        },
    };
    let manifest = bekiper_post_analysis::pipeline::run(&config, &options)?;
    if args.json {
        println!("{}", manifest);
    } else {
        println!(
            "{}",
            serde_json::json!({"status": manifest["status"], "run_mode": manifest["run_mode"]})
        );
    }
    Ok(())
}
