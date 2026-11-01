#[test]
fn external_checks_skip_when_no_optional_tools_are_configured() {
    let dir = reset_fixture_root("bekiper_external_skipped");
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = write_config(
        &dir,
        r#"
        [project]
        project_name = "external"
        data_directory = "."
        "#,
    );

    let report = check_external(&config_path);
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.category == "external/dvc" && diagnostic.severity == AuditSeverity::Skipped
    }));
    assert!(report.diagnostics.iter().any(|diagnostic| {
        diagnostic.category == "external/great_expectations"
            && diagnostic.severity == AuditSeverity::Skipped
    }));

    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(unix)]
#[test]
fn external_exit_and_spawn_results_preserve_severity_and_policy_fields() {
    use std::os::unix::process::ExitStatusExt;
    let success = AuditDiagnostic::new(AuditSeverity::Ok, "tool", "passed");
    let failure = AuditDiagnostic::new(AuditSeverity::Error, "tool", "failed")
        .fix_hint("repair input")
        .validation_command("tool scan");
    let unavailable = AuditDiagnostic::new(AuditSeverity::Warning, "tool", "unavailable");
    let output = |status| std::process::Output {
        status: std::process::ExitStatus::from_raw(status),
        stdout: b"descriptive stdout".to_vec(),
        stderr: b"last diagnostic\n".to_vec(),
    };
    let passed = external_command_result(
        Ok(output(0)),
        success.clone(),
        failure.clone(),
        unavailable.clone(),
    );
    assert_eq!(passed.severity, AuditSeverity::Ok);
    assert!(passed.cause.is_none());
    let failed = external_command_result(
        Ok(output(256)),
        success.clone(),
        failure.clone(),
        unavailable.clone(),
    );
    assert_eq!(failed.severity, AuditSeverity::Error);
    assert_eq!(failed.cause.as_deref(), Some("last diagnostic"));
    assert_eq!(failed.fix_hint, failure.fix_hint);
    assert_eq!(failed.validation_command, failure.validation_command);
    let missing = || {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing executable",
        ))
    };
    let warning = external_command_result(missing(), success.clone(), failure.clone(), unavailable);
    assert_eq!(warning.severity, AuditSeverity::Warning);
    assert_eq!(warning.cause.as_deref(), Some("missing executable"));
    let skipped = external_command_result(
        missing(),
        success,
        failure,
        AuditDiagnostic::new(AuditSeverity::Skipped, "tool", "not installed"),
    );
    assert_eq!(skipped.severity, AuditSeverity::Skipped);
    assert!(skipped.cause.is_none());
}
