impl DetectionDryRunSummary {
    fn push(
        &mut self,
        severity: DryRunSeverity,
        category: impl Into<String>,
        experiment: impl Into<String>,
        message: impl Into<String>,
    ) {
        match severity {
            DryRunSeverity::Ok => self.ok += 1,
            DryRunSeverity::Skipped => {}
            DryRunSeverity::Warning => self.warnings += 1,
            DryRunSeverity::Error => self.errors += 1,
        }
        self.diagnostics.push(DryRunDiagnostic {
            severity,
            category: category.into(),
            experiment: experiment.into(),
            message: message.into(),
        });
    }

    pub fn print(&self) {
        println!(
            "Detection dry run: {} experiment(s), {} skipped, {} ok, {} warning(s), {} error(s)",
            self.total_experiments, self.skipped_experiments, self.ok, self.warnings, self.errors
        );

        let mut category_counts: BTreeMap<&str, (usize, usize, usize, usize)> = BTreeMap::new();
        for diagnostic in &self.diagnostics {
            let counts = category_counts
                .entry(diagnostic.category.as_str())
                .or_default();
            match diagnostic.severity {
                DryRunSeverity::Ok => counts.0 += 1,
                DryRunSeverity::Skipped => counts.1 += 1,
                DryRunSeverity::Warning => counts.2 += 1,
                DryRunSeverity::Error => counts.3 += 1,
            }
        }

        if !category_counts.is_empty() {
            println!();
            println!("Checks by category:");
            for (category, (ok, skipped, warnings, errors)) in category_counts {
                println!("  {category}: {ok} ok, {warnings} warning(s), {errors} error(s)");
                if skipped > 0 {
                    println!("    skipped: {skipped}");
                }
            }
        }

        let mut grouped: BTreeMap<&str, Vec<&DryRunDiagnostic>> = BTreeMap::new();
        for diagnostic in &self.diagnostics {
            if matches!(
                diagnostic.severity,
                DryRunSeverity::Ok | DryRunSeverity::Skipped
            ) {
                continue;
            }
            grouped
                .entry(diagnostic.category.as_str())
                .or_default()
                .push(diagnostic);
        }

        if grouped.is_empty() {
            println!();
            println!("No warnings or errors.");
            return;
        }

        println!();
        println!("Warnings and errors:");
        for (category, diagnostics) in grouped {
            println!();
            println!("== {category} ==");
            for diagnostic in diagnostics {
                println!(
                    "[{}] {}: {}",
                    diagnostic.severity.label(),
                    diagnostic.experiment,
                    diagnostic.message
                );
            }
        }
    }

    pub fn into_diagnostics(self, config_path: &Path) -> Vec<AuditDiagnostic> {
        let validation_command = format!("bekiper run -c {} --dry-run", config_path.display());
        let mut diagnostics = Vec::new();
        diagnostics.push(
            AuditDiagnostic::new(
                if self.errors > 0 {
                    AuditSeverity::Error
                } else if self.warnings > 0 {
                    AuditSeverity::Warning
                } else {
                    AuditSeverity::Ok
                },
                "detection",
                format!(
                    "detection dry run: {} experiment(s), {} skipped, {} ok, {} warning(s), {} error(s)",
                    self.total_experiments,
                    self.skipped_experiments,
                    self.ok,
                    self.warnings,
                    self.errors
                ),
            )
            .validation_command(&validation_command),
        );
        diagnostics.extend(self.diagnostics.into_iter().filter_map(|diagnostic| {
            if matches!(
                diagnostic.severity,
                DryRunSeverity::Ok | DryRunSeverity::Skipped
            ) {
                return None;
            }
            Some(
                AuditDiagnostic::new(
                    match diagnostic.severity {
                        DryRunSeverity::Ok => AuditSeverity::Ok,
                        DryRunSeverity::Skipped => AuditSeverity::Skipped,
                        DryRunSeverity::Warning => AuditSeverity::Warning,
                        DryRunSeverity::Error => AuditSeverity::Error,
                    },
                    diagnostic.category,
                    diagnostic.message,
                )
                .experiment(diagnostic.experiment)
                .fix_hint("fix the listed detection prerequisite before running analysis")
                .validation_command(&validation_command),
            )
        }));
        diagnostics
    }
}
