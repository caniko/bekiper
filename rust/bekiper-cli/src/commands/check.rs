use std::path::PathBuf;

use clap::{Args, Subcommand};

use bekiper_core::identity::DEFAULT_CONFIG_FILENAME;

use crate::report::{ReportFormat, print_likelihood_report, print_report};
mod qualification;

#[derive(Args)]
pub struct CheckArgs {
    #[command(subcommand)]
    pub command: CheckCommand,
}

#[derive(Subcommand)]
pub enum CheckCommand {
    /// Run fixed qualification gates and retain source-bound receipts on an exact clean revision.
    Qualify {
        #[arg(long)]
        checkout: PathBuf,
        #[arg(long)]
        revision: String,
        #[arg(long, value_enum)]
        profile: qualification::QualificationProfile,
        #[arg(long)]
        output: PathBuf,
    },
    /// Reconcile a native pre-filter ledger with original workbook/ODS sources into a fresh handoff.
    ReconcilePopulation {
        #[arg(long)]
        population: PathBuf,
        #[arg(long)]
        workbook: PathBuf,
        #[arg(long, default_value = "Updated_Mice-overview")]
        sheet: String,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        ymaze_ods: Option<PathBuf>,
        /// Original OFT/NORT geometry-binding CSV, kept separate from workbook biology.
        #[arg(long)]
        metadata_csv: Option<PathBuf>,
        /// TOML exclusion authorization pinned to the population fingerprint and workbook SHA-256.
        #[arg(long)]
        authorization: Option<PathBuf>,
        /// New output directory; source files and existing handoffs are immutable.
        #[arg(long)]
        output: PathBuf,
    },
    /// Inventory every analysis before metadata filtering, including exclusions (read-only).
    Population {
        #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
        config: PathBuf,
        /// Relative-path base used by a registry registration; defaults to the config directory.
        #[arg(long)]
        data_root: Option<PathBuf>,
    },
    /// Verify every manual-annotation experiment resolves to annotation data.
    Annotations {
        /// Path to the project configuration file.
        #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
        config: PathBuf,
    },

    /// Parse and validate a project config without running the pipeline.
    Config {
        /// Path to the project configuration file.
        #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
        config: PathBuf,
    },

    /// Run the full non-mutating readiness gate before analysis.
    Readiness {
        /// Path to the project configuration file.
        #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
        config: PathBuf,

        /// Output format.
        #[arg(long, value_enum, default_value_t = ReportFormat::Human)]
        format: ReportFormat,
    },

    /// Inspect pose artifacts and report trajectory quality issues without rewriting data.
    PoseQuality {
        /// Path to the project configuration file.
        #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
        config: PathBuf,

        /// Output format.
        #[arg(long, value_enum, default_value_t = ReportFormat::Human)]
        format: ReportFormat,
    },

    /// Compact ranked report for pose likelihood scoring across a dataset.
    Likelihood {
        /// Path to the project configuration file.
        #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
        config: PathBuf,

        /// Output format.
        #[arg(long, value_enum, default_value_t = ReportFormat::Human)]
        format: ReportFormat,

        /// Include every pose label instead of the task-critical analysis labels.
        #[arg(long)]
        all_labels: bool,
    },

    /// Run optional third-party validation integrations when configured.
    External {
        /// Path to the project configuration file.
        #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
        config: PathBuf,

        /// Output format.
        #[arg(long, value_enum, default_value_t = ReportFormat::Human)]
        format: ReportFormat,
    },
}

pub fn run(args: CheckArgs) -> anyhow::Result<()> {
    match args.command {
        CheckCommand::Qualify {
            checkout,
            revision,
            profile,
            output,
        } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&qualification::qualify(
                    &checkout, &revision, profile, &output
                )?)?
            );
            Ok(())
        }
        CheckCommand::ReconcilePopulation {
            population,
            workbook,
            sheet,
            plan,
            ymaze_ods,
            metadata_csv,
            authorization,
            output,
        } => {
            use bekiper_ingress::audit::reconciliation::{
                ReconciliationInputs, export_reconciliation, reconcile_population,
            };
            let report = reconcile_population(&ReconciliationInputs {
                population: &population,
                workbook: &workbook,
                workbook_sheet: &sheet,
                plan: &plan,
                ymaze_ods: ymaze_ods.as_deref(),
                metadata_csv: metadata_csv.as_deref(),
                authorization: authorization.as_deref(),
            })?;
            let destination = export_reconciliation(&report, &output)?;
            println!(
                "{}",
                serde_json::json!({"output":destination,"recordings":report.recordings.len(),
                "authorized_excluded_recordings":report.authorized_excluded_recordings,
                "remaining_recordings":report.remaining_recordings,"policy_candidates":report.policy_candidates,
                "scientific_readiness":report.scientific_readiness,"analysis_admitted":false})
            );
            Ok(())
        }
        CheckCommand::Population { config, data_root } => {
            let base = data_root
                .as_deref()
                .unwrap_or_else(|| config.parent().unwrap_or(std::path::Path::new(".")));
            let ledger = bekiper_ingress::audit::population_ledger_at(&config, base)?;
            println!("{}", bekiper_core::handoff::to_string(&ledger)?);
            anyhow::ensure!(
                ledger.blocked_analyses == 0,
                "metadata policy blocks {} analysis entries",
                ledger.blocked_analyses
            );
            Ok(())
        }
        CheckCommand::Annotations { config } => {
            let summary = bekiper_ingress::audit::check_manual_annotation_coverage(&config)?;
            println!(
                "Manual annotations: {}/{} assigned",
                summary.assigned_sessions, summary.total_manual_sessions
            );

            if summary.is_complete() {
                return Ok(());
            }

            for failure in &summary.missing_sessions {
                println!();
                println!("Missing: {}", failure.experiment);
                if let Some(video_path) = &failure.video_path {
                    println!("  video: {}", video_path.display());
                }
                if let Some(annotation_dir) = &failure.annotation_dir {
                    println!("  annotation_dir: {}", annotation_dir.display());
                }
                println!("  error: {}", failure.error);
            }

            anyhow::bail!(
                "{} manual-annotation session(s) are missing or unresolved",
                summary.missing_sessions.len()
            );
        }
        CheckCommand::Config { config } => {
            let summary = bekiper_ingress::audit::validate_config_file(&config)?;
            println!(
                "Config OK: {} experiment(s), {} ingress profile(s), {} scan rule(s), {} animal_scan rule(s)",
                summary.experiments,
                summary.ingress_profile_entries,
                summary.scan_entries,
                summary.animal_scan_entries
            );
            Ok(())
        }
        CheckCommand::Readiness { config, format } => {
            let report = bekiper_ingress::audit::check_readiness(&config);
            print_report(&report, format)?;
            if report.has_errors() {
                anyhow::bail!("readiness check found {} error(s)", report.counts().errors);
            }
            Ok(())
        }
        CheckCommand::PoseQuality { config, format } => {
            let report = bekiper_ingress::audit::check_pose_quality(&config);
            print_report(&report, format)?;
            if report.has_errors() {
                anyhow::bail!(
                    "pose-quality check found {} error(s)",
                    report.counts().errors
                );
            }
            Ok(())
        }
        CheckCommand::Likelihood {
            config,
            format,
            all_labels,
        } => {
            let report = bekiper_ingress::audit::check_likelihood(&config, all_labels);
            print_likelihood_report(&report, format)?;
            if report.has_errors() {
                anyhow::bail!("likelihood check found {} error(s)", report.counts().errors);
            }
            Ok(())
        }
        CheckCommand::External { config, format } => {
            let report = bekiper_ingress::audit::check_external(&config);
            print_report(&report, format)?;
            if report.has_errors() {
                anyhow::bail!("external check found {} error(s)", report.counts().errors);
            }
            Ok(())
        }
    }
}
