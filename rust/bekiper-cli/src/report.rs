use clap::ValueEnum;

use bekiper_ingress::audit::{
    AuditReport, AuditSeverity, LikelihoodAggregateRow, LikelihoodAuditReport, LikelihoodPairRow,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ReportFormat {
    Human,
    Json,
    Toml,
}

impl std::fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Human => f.write_str("human"),
            Self::Json => f.write_str("json"),
            Self::Toml => f.write_str("toml"),
        }
    }
}

pub fn print_report(report: &AuditReport, format: ReportFormat) -> anyhow::Result<()> {
    match format {
        ReportFormat::Toml => println!("{}", bekiper_core::handoff::to_string(report)?),
        ReportFormat::Json => {
            println!("{}", serde_json::to_string_pretty(report)?);
        }
        ReportFormat::Human => print_human_report(report),
    }
    Ok(())
}

pub fn print_likelihood_report(
    report: &LikelihoodAuditReport,
    format: ReportFormat,
) -> anyhow::Result<()> {
    match format {
        ReportFormat::Toml => println!("{}", bekiper_core::handoff::to_string(report)?),
        ReportFormat::Json => {
            println!("{}", serde_json::to_string_pretty(report)?);
        }
        ReportFormat::Human => print_human_likelihood_report(report),
    }
    Ok(())
}

fn print_human_likelihood_report(report: &LikelihoodAuditReport) {
    let counts = report.counts();
    let summary = &report.summary;
    println!(
        "{}: {}/{} experiment(s), {} frame(s), {} scored label stream(s), {:.1}% low-likelihood at threshold {:.3}; {} missing pose file(s)",
        report.name,
        summary.experiments_inspected,
        summary.experiments_expected,
        summary.total_frames,
        summary.labels_scored,
        summary.low_likelihood_fraction() * 100.0,
        summary.likelihood_threshold,
        summary.experiments_missing_poses
    );
    if summary.all_labels {
        println!("Mode: all pose labels (--all-labels).");
    } else {
        println!(
            "Mode: focused analysis labels for {}: {}. Use --all-labels for the full pose model report.",
            summary.pose_label_model.as_str(),
            summary.focused_labels.join(", ")
        );
    }

    if summary.labels_skipped_missing_likelihood > 0 {
        println!(
            "Skipped {} label stream(s) with missing likelihood columns.",
            summary.labels_skipped_missing_likelihood
        );
    }
    if summary.augmented_missing_count > 0 {
        println!(
            "Note: {} experiment(s) are missing augmented.parquet; likelihood scoring only uses poses.parquet.",
            summary.augmented_missing_count
        );
    }

    print_pair_rows(
        "Worst experiment/label likelihood streams",
        &report.worst_pairs,
    );
    print_aggregate_rows("Worst labels across experiments", &report.worst_labels);
    print_aggregate_rows("Worst experiments across labels", &report.worst_experiments);
    if !report.task_phase_summary.is_empty() {
        print_aggregate_rows("Task/phase summary", &report.task_phase_summary);
    }

    if counts.errors > 0 {
        println!();
        println!("Errors:");
        for diagnostic in report
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == AuditSeverity::Error)
        {
            let subject = diagnostic
                .experiment
                .as_deref()
                .or_else(|| diagnostic.artifact_path.as_ref().and_then(|p| p.to_str()))
                .unwrap_or("-");
            println!(
                "[ERROR] {} {}: {}",
                diagnostic.category, subject, diagnostic.message
            );
            if let Some(cause) = &diagnostic.cause {
                println!("  cause: {cause}");
            }
            if let Some(fix_hint) = &diagnostic.fix_hint {
                println!("  fix: {fix_hint}");
            }
            if let Some(command) = &diagnostic.validation_command {
                println!("  validate: {command}");
            }
        }
    }
}

fn print_pair_rows(title: &str, rows: &[LikelihoodPairRow]) {
    println!();
    println!("{title}:");
    if rows.is_empty() {
        println!("  none");
        return;
    }
    for (idx, row) in rows.iter().enumerate() {
        println!(
            "  {:>2}. {:<44} {:<18} {:>6.1}% ({}/{}) mean={}",
            idx + 1,
            truncate(&row.experiment, 44),
            truncate(&row.label, 18),
            row.low_likelihood_fraction * 100.0,
            row.low_likelihood_frames,
            row.frames,
            format_optional(row.mean_likelihood)
        );
    }
}

fn print_aggregate_rows(title: &str, rows: &[LikelihoodAggregateRow]) {
    println!();
    println!("{title}:");
    if rows.is_empty() {
        println!("  none");
        return;
    }
    for (idx, row) in rows.iter().enumerate() {
        println!(
            "  {:>2}. {:<44} {:>6.1}% ({}/{}) mean={} experiments={} labels={}",
            idx + 1,
            truncate(&row.key, 44),
            row.low_likelihood_fraction * 100.0,
            row.low_likelihood_frames,
            row.frames,
            format_optional(row.mean_likelihood),
            row.experiments,
            row.labels
        );
    }
}

fn format_optional(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".into())
}

fn truncate(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{}...", truncated.trim_end())
    } else {
        truncated
    }
}

fn print_human_report(report: &AuditReport) {
    let counts = report.counts();
    println!(
        "{}: {} ok, {} skipped, {} warning(s), {} error(s)",
        report.name, counts.ok, counts.skipped, counts.warnings, counts.errors
    );

    let mut by_category = std::collections::BTreeMap::<&str, (usize, usize, usize, usize)>::new();
    for diagnostic in &report.diagnostics {
        let counts = by_category.entry(&diagnostic.category).or_default();
        match diagnostic.severity {
            AuditSeverity::Ok => counts.0 += 1,
            AuditSeverity::Skipped => counts.1 += 1,
            AuditSeverity::Warning => counts.2 += 1,
            AuditSeverity::Error => counts.3 += 1,
        }
    }

    if !by_category.is_empty() {
        println!();
        println!("Checks by category:");
        for (category, (ok, skipped, warnings, errors)) in by_category {
            println!(
                "  {category}: {ok} ok, {skipped} skipped, {warnings} warning(s), {errors} error(s)"
            );
        }
    }

    let blocking = report
        .diagnostics
        .iter()
        .filter(|diagnostic| {
            matches!(
                diagnostic.severity,
                AuditSeverity::Warning | AuditSeverity::Error
            )
        })
        .collect::<Vec<_>>();
    if blocking.is_empty() {
        println!();
        println!("No warnings or errors.");
        return;
    }

    println!();
    println!("Warnings and errors:");
    for diagnostic in blocking {
        let subject = diagnostic
            .experiment
            .as_deref()
            .or_else(|| diagnostic.artifact_path.as_ref().and_then(|p| p.to_str()))
            .unwrap_or("-");
        println!(
            "[{}] {} {}: {}",
            diagnostic.severity.label(),
            diagnostic.category,
            subject,
            diagnostic.message
        );
        if let Some(cause) = &diagnostic.cause {
            println!("  cause: {cause}");
        }
        if let Some(fix_hint) = &diagnostic.fix_hint {
            println!("  fix: {fix_hint}");
        }
        if let Some(command) = &diagnostic.validation_command {
            println!("  validate: {command}");
        }
    }
}
