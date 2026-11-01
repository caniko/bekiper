fn purge_stage_outputs(
    config: &IngressConfig,
    output_dir: &Path,
    start_stage: Stage,
    reason: &str,
) -> Result<()> {
    let mut removed = 0usize;
    for experiment in &config.experiments {
        let exp_dir = exp_output_dir(output_dir, experiment);
        for path in exp_dir.purge_paths_from(start_stage) {
            if remove_existing_path(&path)? {
                removed += 1;
            }
        }
    }
    if Stage::from_onward(start_stage).contains(&Stage::Aggregate) {
        for path in [
            aggregate_output_path(output_dir, &config.aggregate.output_filename),
            aggregate_state_path(output_dir, &config.aggregate.output_filename),
        ] {
            if remove_existing_path(&path)? {
                removed += 1;
            }
        }
    }

    tracing::info!(start_stage = %start_stage, removed, reason, "purged stage outputs");
    Ok(())
}

fn remove_existing_path(path: &Path) -> Result<bool> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)?;
        Ok(true)
    } else if path.exists() {
        std::fs::remove_file(path)?;
        Ok(true)
    } else {
        Ok(false)
    }
}
