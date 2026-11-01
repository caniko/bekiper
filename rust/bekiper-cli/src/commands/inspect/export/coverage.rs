fn coverage(path: &Path, trials: &[Trial]) -> Result<Value> {
    let ledger: PopulationLedger = if path.extension().is_some_and(|e| e == "toml") {
        bekiper_core::handoff::from_str(&fs::read_to_string(path)?)?
    } else {
        serde_json::from_value(inputs::load_json(path)?)?
    };
    ledger.validate()?;
    ensure!(
        ledger.blocked_analyses == 0,
        "population contains blocked analyses"
    );
    let actual: BTreeSet<_> = trials.iter().map(|t| t.analysis_id.as_str()).collect();
    let allowed: BTreeSet<_> = ledger
        .rows
        .iter()
        .map(|r| r.decision.analysis_id.as_str())
        .collect();
    ensure!(
        actual.is_subset(&allowed),
        "export contains analyses outside population"
    );
    let mut rows = Vec::new();
    for row in &ledger.rows {
        let rendered = actual.contains(row.decision.analysis_id.as_str());
        if row.decision.disposition == PopulationDisposition::Admitted {
            ensure!(
                rendered,
                "missing admitted inspection: {}",
                row.decision.analysis_id
            );
        } else {
            ensure!(
                row.decision
                    .reason
                    .as_ref()
                    .is_some_and(|r| !r.trim().is_empty()),
                "excluded population row requires a reason"
            );
        }
        rows.push(json!({"analysis_id":row.decision.analysis_id,"recording_key":row.recording_key,
            "animal_id":row.decision.animal_id,"task_type":row.task_type,"disposition":row.decision.disposition,
            "reason":row.decision.reason,"trial_exported":rendered}));
    }
    ensure!(!actual.is_empty(), "population has no rendered analyses");
    Ok(
        json!({"status":"complete_admitted_coverage","population_fingerprint":ledger.fingerprint,
        "recordings":ledger.pre_filter_recordings,"analyses":ledger.pre_filter_analyses,
        "admitted_analyses":ledger.admitted_analyses,"excluded_analyses":ledger.excluded_analyses,"rows":rows}),
    )
}
