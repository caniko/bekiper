//! Shared standalone/project CLI contracts for native qualification surfaces.
use crate::Record;
use clap::Subcommand;
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
pub enum MaintenanceCommand {
    /// Verify published artifacts, independent contexts, policy inference and render metadata.
    Verify {
        #[arg(long)]
        input: PathBuf,
        /// Match embedded engine and source-lock digests to this checkout.
        #[arg(long)]
        source: Option<PathBuf>,
    },
    /// Check dependency-scoped registrations without inference or run admission.
    VerifyPlan {
        #[arg(long)]
        config: PathBuf,
    },
    /// Retain PDF text bounds and representative page images for visual review.
    ReviewPdfs {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Prepare explicitly synthetic A/B/C fixtures from a plan and numerical oracle.
    PreparePolicyFixture {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

impl MaintenanceCommand {
    pub fn run(self) -> anyhow::Result<Record> {
        match self {
            Self::Verify { input, source } => {
                crate::verification::verify_run_at_source(&input, source.as_deref())
            }
            Self::VerifyPlan { config } => crate::verification::verify_plan(&config),
            Self::ReviewPdfs { input, output } => crate::verification::review_pdfs(&input, &output),
            Self::PreparePolicyFixture {
                plan,
                oracle,
                output,
            } => crate::fixtures::prepare_hcar1(&plan, &oracle, &output),
        }
    }
}
