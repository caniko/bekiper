mod commands;
mod env_check;
pub(crate) mod python;
pub(crate) mod report;

use clap::Parser;

/// Bekiper — automatic analysis of kinematic data from behavioral experiments.
#[derive(Parser)]
#[command(name = "bekiper", version, about)]
enum Cli {
    /// Create a new project directory with a config template.
    Init(commands::init::InitArgs),

    /// Diagnose the local runtime environment and project analysis prerequisites.
    Doctor(commands::doctor::DoctorArgs),

    /// Run the full analysis pipeline (detect → pose → augment → behaviour → summary → inspect → aggregate).
    ///
    /// Each stage checkpoints its output to disk. Re-running skips stages
    /// whose outputs are already fresh. Use --from or --only to control
    /// which stages run. Use --force to re-run everything.
    #[cfg(feature = "full")]
    Run(commands::run::RunArgs),

    /// Data preparation and housekeeping.
    Data(commands::data::DataArgs),

    /// Annotation GUI, server/client, and canonical-store ops.
    #[cfg(any(
        feature = "gui",
        feature = "annotate-server",
        feature = "annotate-client"
    ))]
    Annotate(commands::annotate::AnnotateArgs),

    /// Inspection and visualization commands.
    Inspect(commands::inspect::InspectArgs),
    /// Reviewer package: one composite per trial, cohort pages, sealed zip.
    Review(commands::review::ReviewArgs),

    /// Reproducible behavioural post-analysis from exported measurements.
    PostAnalysis(commands::post_analysis::PostAnalysisArgs),

    /// Read-only project validation commands.
    Check(commands::check::CheckArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = match Cli::parse() {
        #[cfg(any(
            feature = "gui",
            feature = "annotate-server",
            feature = "annotate-client"
        ))]
        Cli::Annotate(args) => return commands::annotate::run(args),
        cli => cli,
    };

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Diagnose missing ROCm / ONNX Runtime environment early so users
    // running outside the nix devshell get an actionable warning instead
    // of a cryptic crash deep inside `ort` or `hip`.
    if !matches!(&cli, Cli::PostAnalysis(_)) {
        env_check::warn_if_env_incomplete();
    }

    match cli {
        Cli::Init(args) => commands::init::run(args),
        Cli::Doctor(args) => commands::doctor::run(args),
        #[cfg(feature = "full")]
        Cli::Run(args) => commands::run::run(args),
        Cli::Data(args) => commands::data::run(args),
        #[cfg(any(
            feature = "gui",
            feature = "annotate-server",
            feature = "annotate-client"
        ))]
        Cli::Annotate(args) => commands::annotate::run(args),
        Cli::Inspect(args) => commands::inspect::run(args),
        Cli::Review(args) => commands::review::run(args),
        Cli::PostAnalysis(args) => commands::post_analysis::run(args),
        Cli::Check(args) => commands::check::run(args),
    }
}

#[cfg(test)]
mod post_analysis_cli_tests {
    use super::{Cli, commands};
    use clap::Parser;

    #[test]
    fn native_cli_accepts_population_bound_trial_export() {
        assert!(
            Cli::try_parse_from([
                "bekiper",
                "inspect",
                "export",
                "completed-run",
                "--config",
                "captured.toml",
                "--output",
                "new-exports",
                "--population",
                "population.toml",
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "bekiper",
                "inspect",
                "export",
                "run",
                "--config",
                "config.toml",
                "--output",
                "fresh",
                "--population",
                "population.toml",
                "--trial",
                "only-one",
            ])
            .is_err()
        );
    }

    #[test]
    fn native_cli_accepts_post_analysis_output_requests() {
        let parsed = Cli::try_parse_from([
            "bekiper",
            "post-analysis",
            "run",
            "--config",
            "analysis config.json",
            "--output",
            "fresh results",
            "--request",
            "metrics",
            "--request",
            "models",
        ]);
        assert!(
            parsed.is_ok(),
            "native post-analysis must parse: {}",
            parsed.as_ref().err().unwrap()
        );
        let Cli::PostAnalysis(args) = parsed.unwrap() else {
            panic!("wrong command");
        };
        let commands::post_analysis::PostAnalysisCommand::Run {
            config,
            output,
            request,
            ..
        } = args.command
        else {
            panic!("wrong post-analysis command");
        };
        assert_eq!(config, std::path::PathBuf::from("analysis config.json"));
        assert_eq!(output, Some(std::path::PathBuf::from("fresh results")));
        assert_eq!(request, ["metrics", "models"]);
    }

    #[cfg(feature = "full")]
    #[test]
    fn inspect_video_parses_scored_source_and_crf_flags() {
        let parsed = Cli::try_parse_from([
            "bekiper",
            "inspect",
            "video",
            "trial",
            "--scored",
            "--source-video",
            "rebased/video.mp4",
            "--crf",
            "20",
        ])
        .unwrap();
        let Cli::Inspect(args) = parsed else {
            panic!("wrong command");
        };
        let commands::inspect::InspectCommand::Video {
            source_video,
            crf,
            scored,
            ..
        } = args.command
        else {
            panic!("wrong inspect command");
        };
        assert!(scored);
        assert_eq!(crf, Some(20));
        assert_eq!(
            source_video,
            Some(std::path::PathBuf::from("rebased/video.mp4"))
        );
        assert!(
            Cli::try_parse_from(["bekiper", "inspect", "video", "trial", "--crf", "300"]).is_err()
        );
    }

    #[test]
    fn native_cli_supports_disabling_sensitivity_without_changing_science() {
        assert!(
            Cli::try_parse_from([
                "bekiper",
                "post-analysis",
                "run",
                "--config",
                "analysis.json",
                "--no-sensitivity"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "bekiper",
                "post-analysis",
                "run",
                "--config",
                "analysis.json",
                "--no-sensitivity",
                "--sensitivity-threshold",
                "25"
            ])
            .is_err()
        );
    }
}
