use std::path::PathBuf;

use clap::Args;

use crate::report::{ReportFormat, print_report};

#[derive(Args)]
pub struct DoctorArgs {
    /// Path to the project configuration file.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = ReportFormat::Human)]
    format: ReportFormat,
}

pub fn run(args: DoctorArgs) -> anyhow::Result<()> {
    let report = bekiper_ingress::audit::doctor_report(args.config.as_deref());
    print_report(&report, args.format)?;
    if report.has_errors() {
        anyhow::bail!("doctor found {} error(s)", report.counts().errors);
    }
    Ok(())
}
