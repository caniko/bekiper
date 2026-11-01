//! Audit readiness responsibilities.
use super::*;

pub fn check_readiness(path: &Path) -> AuditReport {
    let mut report = AuditReport::new("readiness", Some(path.to_path_buf()));
    let validation_command = format!("bekiper check readiness -c {}", path.display());
    match population_ledger_at(path, path.parent().unwrap_or(Path::new("."))) {
        Ok(population) => report.population = Some(population),
        Err(error) => report.push(
            AuditDiagnostic::new(
                AuditSeverity::Error,
                "population",
                "could not inventory the pre-filter population",
            )
            .cause(error.to_string()),
        ),
    }

    match validate_config_file(path) {
        Ok(summary) => report.push(
            AuditDiagnostic::new(
                AuditSeverity::Ok,
                "config",
                format!(
                    "config expands to {} experiment(s), {} ingress profile(s), {} scan rule(s), {} animal_scan rule(s)",
                    summary.experiments,
                    summary.ingress_profile_entries,
                    summary.scan_entries,
                    summary.animal_scan_entries
                ),
            )
            .artifact_path(path)
            .validation_command(&validation_command),
        ),
        Err(error) => report.push(
            AuditDiagnostic::new(AuditSeverity::Error, "config", "config validation failed")
                .artifact_path(path)
                .cause(error.to_string())
                .fix_hint("fix the referenced config path, model, data, annotation, or perimeter artifact")
                .validation_command(format!("bekiper check config -c {}", path.display())),
        ),
    }

    match check_manual_annotation_coverage(path) {
        Ok(summary) if summary.is_complete() => report.push(
            AuditDiagnostic::new(
                AuditSeverity::Ok,
                "annotation",
                format!(
                    "manual annotations assigned for {}/{} analysis session(s)",
                    summary.assigned_sessions, summary.total_manual_sessions
                ),
            )
            .validation_command(format!("bekiper check annotations -c {}", path.display())),
        ),
        Ok(summary) => {
            report.push(
                AuditDiagnostic::new(
                    AuditSeverity::Error,
                    "annotation",
                    format!(
                        "{} manual-annotation session(s) are missing or unresolved",
                        summary.missing_sessions.len()
                    ),
                )
                .cause("one or more manual_annotation experiments could not resolve annotation geometry")
                .fix_hint("repair or import the source annotation data for the listed session(s)")
                .validation_command(format!("bekiper check annotations -c {}", path.display())),
            );
            for failure in summary.missing_sessions {
                let mut diagnostic =
                    AuditDiagnostic::new(AuditSeverity::Error, "annotation", failure.error)
                        .experiment(failure.experiment)
                        .fix_hint(
                            "repair the annotation source or metadata mapping for this experiment",
                        )
                        .validation_command(format!(
                            "bekiper check annotations -c {}",
                            path.display()
                        ));
                if let Some(video_path) = failure.video_path {
                    diagnostic = diagnostic.artifact_path(video_path);
                } else if let Some(annotation_dir) = failure.annotation_dir {
                    diagnostic = diagnostic.artifact_path(annotation_dir);
                }
                report.push(diagnostic);
            }
        }
        Err(error) => report.push(
            AuditDiagnostic::new(
                AuditSeverity::Error,
                "annotation",
                "manual annotation coverage check failed",
            )
            .cause(error.to_string())
            .fix_hint("fix config expansion or annotation store loading before running analysis")
            .validation_command(format!("bekiper check annotations -c {}", path.display())),
        ),
    }

    #[cfg(feature = "pose-gpu")]
    {
        match crate::workflow::analyze_detection_dry_run(path) {
            Ok(summary) => report.extend(summary.into_diagnostics(path)),
            Err(error) => report.push(
                AuditDiagnostic::new(
                    AuditSeverity::Error,
                    "detection",
                    "detection dry run could not be evaluated",
                )
                .cause(error.to_string())
                .fix_hint(
                    "fix config expansion and detection prerequisites before running analysis",
                )
                .validation_command(format!("bekiper run -c {} --dry-run", path.display())),
            ),
        }
    }
    #[cfg(not(feature = "pose-gpu"))]
    report.push(
        AuditDiagnostic::new(
            AuditSeverity::Skipped,
            "detection",
            "detection dry run requires the pose-gpu feature",
        )
        .fix_hint("build with default features for full readiness checks")
        .validation_command("cargo check -p bekiper-cli --features full".to_string()),
    );

    report.extend(doctor_diagnostics(Some(path)));
    report
}

pub(super) fn path_readiness_diagnostic(
    category: &str,
    path: &Path,
    cause: impl Into<String>,
    validation_command: impl Into<String>,
) -> AuditDiagnostic {
    if path.exists() {
        AuditDiagnostic::new(
            AuditSeverity::Ok,
            category,
            format!("{} exists", path.display()),
        )
        .artifact_path(path)
        .validation_command(validation_command)
    } else {
        AuditDiagnostic::new(
            AuditSeverity::Error,
            category,
            format!("missing {}", path.display()),
        )
        .artifact_path(path)
        .cause(cause)
        .fix_hint("restore or regenerate this required artifact")
        .validation_command(validation_command)
    }
}
