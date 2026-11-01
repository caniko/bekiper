pub(super) fn aggregate_metadata_base_dir(
    config: &IngressConfig,
    run_root: &Path,
) -> std::path::PathBuf {
    if config.project.data_directory.is_absolute() {
        config.project.data_directory.clone()
    } else {
        run_root
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&config.project.data_directory)
    }
}

fn aggregate_metadata_modified_unix_nanos(config: &IngressConfig, run_root: &Path) -> Option<u128> {
    let metadata = config.aggregate.metadata.as_ref()?;
    let base_dir = aggregate_metadata_base_dir(config, run_root);
    let path = if metadata.path.is_absolute() {
        metadata.path.clone()
    } else {
        base_dir.join(&metadata.path)
    };
    std::fs::metadata(path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
}

fn apply_aggregate_metadata(
    config: &IngressConfig,
    run_root: &Path,
    trials: &mut Vec<TrialResult>,
) -> Result<Vec<MetadataExclusion>> {
    let Some(metadata_cfg) = config.aggregate.metadata.as_ref() else {
        return Ok(Vec::new());
    };
    let base_dir = aggregate_metadata_base_dir(config, run_root);
    let metadata = match load_aggregate_metadata(&base_dir, metadata_cfg) {
        Ok(metadata) => metadata,
        Err(err)
            if metadata_cfg.missing_policy == crate::config::MissingMetadataPolicy::AllowBlanks =>
        {
            tracing::warn!(
                error = %err,
                "aggregate metadata source unavailable; continuing without metadata enrichment"
            );
            return Ok(Vec::new());
        }
        Err(err) => return Err(err),
    };
    enrich_trials_with_metadata(trials, &metadata, metadata_cfg.missing_policy)
}

fn metadata_exclusions_sheet(exclusions: &[MetadataExclusion]) -> SpreadsheetSheet {
    let mut rows = vec![vec![
        SpreadsheetCell::string("animal"),
        SpreadsheetCell::string("reason"),
    ]];
    for exclusion in exclusions {
        rows.push(vec![
            SpreadsheetCell::Int(exclusion.animal as i64),
            SpreadsheetCell::string(&exclusion.reason),
        ]);
    }
    SpreadsheetSheet::from_rows("metadata_exclusions", rows, 1)
}
