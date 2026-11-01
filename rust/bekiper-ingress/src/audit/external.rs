//! Audit external responsibilities.
use super::*;

pub fn check_external(path: &Path) -> AuditReport {
    let mut report = AuditReport::new("external", Some(path.to_path_buf()));
    let base_dir = path.parent().unwrap_or(Path::new("."));

    run_great_expectations_check(base_dir, &mut report);
    run_soda_check(base_dir, &mut report);
    run_dvc_check(base_dir, &mut report);
    run_deepchecks_check(base_dir, &mut report);
    run_env_doctor_check(&mut report);

    report
}

pub(super) fn run_great_expectations_check(base_dir: &Path, report: &mut AuditReport) {
    let gx_dir = base_dir.join("gx");
    let legacy = base_dir.join("great_expectations");
    if !gx_dir.exists() && !legacy.exists() {
        report.push(AuditDiagnostic::new(
            AuditSeverity::Skipped,
            "external/great_expectations",
            "Great Expectations not configured",
        ));
        return;
    }
    run_external_command(
        Command::new("great_expectations")
            .args(["checkpoint", "list"])
            .current_dir(base_dir),
        AuditDiagnostic::new(
            AuditSeverity::Ok,
            "external/great_expectations",
            "Great Expectations CLI can list checkpoints",
        ),
        AuditDiagnostic::new(
            AuditSeverity::Warning,
            "external/great_expectations",
            "Great Expectations is configured but checkpoint listing failed",
        )
        .fix_hint("run the project Great Expectations checkpoint command directly"),
        AuditDiagnostic::new(
            AuditSeverity::Warning,
            "external/great_expectations",
            "Great Expectations is configured but the CLI is not available",
        )
        .fix_hint(
            "install Great Expectations in the project environment or remove the stale config",
        ),
        report,
    );
}

pub(super) fn run_soda_check(base_dir: &Path, report: &mut AuditReport) {
    let config = base_dir.join("configuration.yml");
    let checks = base_dir.join("checks.yml");
    if !config.exists() || !checks.exists() {
        report.push(AuditDiagnostic::new(
            AuditSeverity::Skipped,
            "external/soda",
            "Soda Core not configured",
        ));
        return;
    }
    run_external_command(
        Command::new("soda")
            .args(["scan", "-c"])
            .arg(&config)
            .arg(&checks)
            .current_dir(base_dir),
        AuditDiagnostic::new(AuditSeverity::Ok, "external/soda", "Soda Core scan passed"),
        AuditDiagnostic::new(
            AuditSeverity::Error,
            "external/soda",
            "Soda Core scan failed",
        )
        .fix_hint("fix the failing Soda checks or upstream data contract")
        .validation_command(format!(
            "soda scan -c {} {}",
            config.display(),
            checks.display()
        )),
        AuditDiagnostic::new(
            AuditSeverity::Warning,
            "external/soda",
            "Soda Core is configured but the CLI is not available",
        )
        .fix_hint("install Soda Core in the project environment or remove the stale config"),
        report,
    );
}

pub(super) fn run_dvc_check(base_dir: &Path, report: &mut AuditReport) {
    if !base_dir.join("dvc.yaml").exists() && !base_dir.join(".dvc").exists() {
        report.push(AuditDiagnostic::new(
            AuditSeverity::Skipped,
            "external/dvc",
            "DVC not configured",
        ));
        return;
    }
    run_external_command(
        Command::new("dvc").arg("status").current_dir(base_dir),
        AuditDiagnostic::new(AuditSeverity::Ok, "external/dvc", "DVC status completed"),
        AuditDiagnostic::new(
            AuditSeverity::Warning,
            "external/dvc",
            "DVC status reported issues",
        )
        .fix_hint("inspect DVC-tracked data/model provenance")
        .validation_command("dvc status"),
        AuditDiagnostic::new(
            AuditSeverity::Warning,
            "external/dvc",
            "DVC is configured but the CLI is not available",
        )
        .fix_hint("install DVC or remove stale DVC metadata"),
        report,
    );
}

pub(super) fn run_deepchecks_check(base_dir: &Path, report: &mut AuditReport) {
    let script = base_dir.join("deepchecks.py");
    if !script.exists() {
        report.push(AuditDiagnostic::new(
            AuditSeverity::Skipped,
            "external/deepchecks",
            "Deepchecks suite not configured",
        ));
        return;
    }
    run_external_command(
        Command::new("python3").arg(&script).current_dir(base_dir),
        AuditDiagnostic::new(
            AuditSeverity::Ok,
            "external/deepchecks",
            "Deepchecks suite passed",
        ),
        AuditDiagnostic::new(
            AuditSeverity::Error,
            "external/deepchecks",
            "Deepchecks suite failed",
        )
        .fix_hint("fix the failing Deepchecks model/data validation suite")
        .validation_command(format!("python3 {}", script.display())),
        AuditDiagnostic::new(
            AuditSeverity::Warning,
            "external/deepchecks",
            "Deepchecks suite is configured but python3 could not run it",
        ),
        report,
    );
}

pub(super) fn run_env_doctor_check(report: &mut AuditReport) {
    run_external_command(
        Command::new("env-doctor").arg("check"),
        AuditDiagnostic::new(
            AuditSeverity::Ok,
            "external/env-doctor",
            "Env-Doctor check passed",
        ),
        AuditDiagnostic::new(
            AuditSeverity::Warning,
            "external/env-doctor",
            "Env-Doctor reported issues",
        )
        .fix_hint("inspect Env-Doctor's GPU/Python environment recommendations")
        .validation_command("env-doctor check"),
        AuditDiagnostic::new(
            AuditSeverity::Skipped,
            "external/env-doctor",
            "Env-Doctor not installed",
        ),
        report,
    );
}

pub(super) fn run_external_command(
    command: &mut Command,
    success: AuditDiagnostic,
    failure: AuditDiagnostic,
    unavailable: AuditDiagnostic,
    report: &mut AuditReport,
) {
    report.push(external_command_result(
        command.output(),
        success,
        failure,
        unavailable,
    ));
}

pub(super) fn external_command_result(
    output: std::io::Result<std::process::Output>,
    success: AuditDiagnostic,
    failure: AuditDiagnostic,
    unavailable: AuditDiagnostic,
) -> AuditDiagnostic {
    match output {
        Ok(output) if output.status.success() => success,
        Ok(output) => failure.cause(stderr_tail(&String::from_utf8_lossy(&output.stderr))),
        Err(error) if unavailable.severity != AuditSeverity::Skipped => {
            unavailable.cause(error.to_string())
        }
        Err(_) => unavailable,
    }
}

pub(super) fn stderr_tail(stderr: &str) -> String {
    let tail = stderr
        .lines()
        .rev()
        .take(20)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    if tail.is_empty() {
        "(no stderr)".into()
    } else {
        tail
    }
}
