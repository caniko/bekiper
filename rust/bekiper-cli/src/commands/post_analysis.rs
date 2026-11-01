use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct PostAnalysisArgs {
    #[command(subcommand)]
    pub command: PostAnalysisCommand,
}

#[derive(Subcommand)]
pub enum PostAnalysisCommand {
    #[command(flatten)]
    Maintenance(bekiper_post_analysis::maintenance::MaintenanceCommand),
    /// Analyse exported session measurements using registered scientific choices.
    Run {
        #[arg(long = "primary-threshold", value_delimiter = ',')]
        primary_thresholds: Option<Vec<u8>>,
        #[arg(long)]
        primary_workers: Option<usize>,
        /// Minimum same-arm dual-landmark dwell (seconds); default 1, 0 disables it.
        #[arg(long)]
        ymaze_dwell_seconds: Option<f64>,
        /// Duration comparison: at_least (>=, default) or greater_than (>).
        #[arg(long)]
        ymaze_dwell_comparison: Option<bekiper_feature::tolerance::DurationComparison>,
        /// Analysis JSON configuration file.
        #[arg(long)]
        config: PathBuf,
        /// Fresh output directory; overrides environment and configuration.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Requested output group; repeat this option to select several groups.
        #[arg(long)]
        request: Vec<String>,
        /// Y-maze sensitivity threshold (25/45/65); repeat this option.
        #[arg(long = "sensitivity-threshold")]
        sensitivity_threshold: Vec<u8>,
        /// Disable supplementary sensitivity panels without changing D03.
        #[arg(long, conflicts_with = "sensitivity_threshold")]
        no_sensitivity: bool,
    },
}

pub fn run(args: PostAnalysisArgs) -> anyhow::Result<()> {
    match args.command {
        PostAnalysisCommand::Maintenance(command) => {
            println!("{}", serde_json::to_string_pretty(&command.run()?)?);
            Ok(())
        }
        PostAnalysisCommand::Run {
            primary_thresholds,
            primary_workers,
            ymaze_dwell_seconds,
            ymaze_dwell_comparison,
            config,
            output,
            request,
            sensitivity_threshold,
            no_sensitivity,
        } => {
            let options = bekiper_post_analysis::RunOptions {
                primary_thresholds,
                contexts: None,
                render_workers: None,
                primary_workers,
                ymaze_dwell: bekiper_feature::tolerance::DwellOverrides {
                    seconds: ymaze_dwell_seconds,
                    comparison: ymaze_dwell_comparison,
                },
                output,
                requested_outputs: (!request.is_empty()).then_some(request),
                sensitivity_thresholds: if no_sensitivity {
                    Some(Vec::new())
                } else {
                    (!sensitivity_threshold.is_empty()).then_some(sensitivity_threshold)
                },
            };
            let manifest = bekiper_post_analysis::pipeline::run(&config, &options)?;
            println!(
                "Post-analysis status: {} ({})",
                manifest["status"], manifest["run_mode"]
            );
            Ok(())
        }
    }
}
