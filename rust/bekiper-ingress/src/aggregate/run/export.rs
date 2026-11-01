/// Walk `run_root/data` for `{experiment}/result.toml` files, build one sheet per
/// recipe via [`build_sheet_dataframes`], and write the consolidated spreadsheet
/// to `run_root/{cfg.aggregate.output_filename}`.
///
/// If `run_root/data` does not exist, the scan falls back to the legacy
/// `run_root/{experiment}/result.toml` layout.
pub fn run_aggregate(config: &IngressConfig, run_root: &Path) -> Result<()> {
    let cfg = &config.aggregate;
    if !cfg.enabled && !config.post_analysis.enabled {
        tracing::info!("aggregate stage disabled, skipping");
        return Ok(());
    }

    // Walk up to two levels deep to support both flat
    // (`{root}/data/{experiment}/result.toml`) and animal-grouped
    // (`{root}/data/{animal}/{sequence}/result.toml`) layouts. If the data
    // root is absent, fall back to the historical run-root layout.
    let result_scan_root = aggregate_result_scan_root(run_root);
    let mut trials: Vec<TrialResult> = Vec::new();
    collect_trial_results(&result_scan_root, &mut trials, 0, 2);

    export_aggregate_trials(config, run_root, trials)
}

/// Export directly from computed trial results, without parsing intermediate
/// result files. The same scientific validation and metadata rules apply.
pub fn export_aggregate_trials(
    config: &IngressConfig,
    run_root: &Path,
    trials: Vec<TrialResult>,
) -> Result<()> {
    export_aggregate_trials_with_cancel(config, run_root, trials, &|| false)
}

/// The direct-source engine retains its admitted byte lease until native
/// scoring, rendering, compilation and publication have all returned.
pub fn export_aggregate_trials_with_cancel(
    config: &IngressConfig,
    run_root: &Path,
    trials: Vec<TrialResult>,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<()> {
    if cancelled() {
        return Err(BekiperError::Data("cohort export cancelled".into()));
    }
    // Keep metadata-excluded rows in the native inclusion ledger. Workbook
    // filtering below must not erase the cohort's registered exclusions.
    super::post_analysis::export_post_analysis_trials_with_cancel(
        config, run_root, &trials, cancelled,
    )?;
    if cancelled() {
        return Err(BekiperError::Data("cohort export cancelled".into()));
    }
    export_aggregate_workbook(config, run_root, trials)
}

fn export_aggregate_workbook(
    config: &IngressConfig,
    run_root: &Path,
    mut trials: Vec<TrialResult>,
) -> Result<()> {
    let cfg = &config.aggregate;
    if !cfg.enabled {
        return Ok(());
    }

    if trials.is_empty() {
        tracing::warn!(
            run_root = %run_root.display(),
            "no result.toml files found; nothing to aggregate"
        );
        return Ok(());
    }

    // Filesystem enumeration and concurrent source completion have different
    // orders. Normalize once for data sheets and diagnostic sheets alike.
    trials.sort_by(|a, b| a.experiment.cmp(&b.experiment));

    let discovered_trial_count = trials.len();
    let excluded_experiment_names: HashSet<&str> = config
        .experiments
        .iter()
        .filter(|experiment| crate::analysis_exclusion::is_analysis_excluded_experiment(experiment))
        .map(|experiment| experiment.name.as_str())
        .collect();
    let mut skipped_excluded = Vec::new();
    trials.retain(|trial| {
        let excluded = crate::analysis_exclusion::is_analysis_excluded_result(
            &trial.experiment,
            trial.task_type,
            &trial.session_context,
            excluded_experiment_names.contains(trial.experiment.as_str()),
        );
        if excluded {
            skipped_excluded.push(trial.experiment.clone());
        }
        !excluded
    });
    if !skipped_excluded.is_empty() {
        tracing::info!(
            count = skipped_excluded.len(),
            experiments = %skipped_excluded.join(", "),
            "skipping analysis-excluded result.toml files during aggregation; remove stale result.toml files or rerun summary after exclusion to clean the output tree"
        );
    }
    if trials.is_empty() {
        return Err(BekiperError::Config(format!(
            "aggregate skipped all {discovered_trial_count} result.toml file(s) as analysis-excluded; no workbook was written"
        )));
    }

    let metadata_exclusions = apply_aggregate_metadata(config, run_root, &mut trials)?;
    if trials.is_empty() {
        return Err(BekiperError::Config(
            "aggregate metadata excluded every result.toml row; no data sheets were written".into(),
        ));
    }

    validate_nort_aggregate_consistency(run_root, &trials)?;
    validate_ymaze_aggregate_source_summaries(run_root, &trials)?;
    warn_missing_oft_fecal_boli_source(&trials);

    let trial_count = trials.len();
    let diagnostics = build_nort_exploration_diagnostics_dataframe(&trials)?;
    let parameters = build_parameters_dataframe(&trials)?;
    let sheets = build_sheet_dataframes(trials, cfg)?;
    let sheet_count = sheets.len();
    if sheets.is_empty() {
        return Err(BekiperError::Config(format!(
            "aggregate produced no sheets from {trial_count} result.toml file(s); check aggregate sheet-key extraction"
        )));
    }

    let mut workbook = SpreadsheetWorkbook::new();
    for (sheet_name, df, _) in &sheets {
        workbook.push_sheet(SpreadsheetSheet::from_dataframe(
            sheet_name.clone(),
            df.clone(),
        ));
    }
    if let Some(diagnostics) = diagnostics {
        workbook.push_sheet(SpreadsheetSheet::from_dataframe(
            NORT_EXPLORATION_DIAGNOSTICS_SHEET,
            diagnostics,
        ));
    }
    if let Some(parameters) = parameters {
        workbook.push_sheet(SpreadsheetSheet::from_dataframe("parameters", parameters));
    }
    if !metadata_exclusions.is_empty() {
        workbook.push_sheet(metadata_exclusions_sheet(&metadata_exclusions));
    }
    if let Some(overview) = build_animal_overview_dataframe(&sheets)? {
        workbook.push_sheet(SpreadsheetSheet::from_dataframe(
            "animal_overview",
            overview,
        ));
    }

    let out_path = aggregate_output_path_for_config(config, run_root);
    workbook.save_ods(&out_path)?;
    let state_path = aggregate_state_path_for_config(config, run_root);
    let state = AggregateFreshnessState::from_config(config, run_root);
    let state_toml =
        toml::to_string_pretty(&state).map_err(|e| BekiperError::Other(e.to_string()))?;
    std::fs::write(&state_path, state_toml)?;

    tracing::info!(
        trials = trial_count,
        sheets = sheet_count,
        output = %out_path.display(),
        "aggregated trials into consolidated workbook"
    );
    Ok(())
}
