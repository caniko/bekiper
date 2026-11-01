fn verify_trial_renderer(output: &Path, record: &Value) -> Result<()> {
    let relative = record["renderer_receipt"]
        .as_str()
        .context("renderer receipt path")?;
    ensure!(
        pipeline::safe_relative_path(relative),
        "unsafe renderer receipt path"
    );
    let path = output.join(relative);
    let name = record["trial"].as_str().context("trial name")?;
    let stems = record["plot_stems"].as_array().context("declared plots")?;
    let plots = stems
        .iter()
        .map(|s| {
            Ok(output
                .join("trials")
                .join(name)
                .join("figures")
                .join(format!("{}.png", s.as_str().context("plot stem")?)))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        verify_renderer_receipt(&path, &plots, record["dpi"].as_u64().context("figure DPI")?)?
            == record["renderer_identity"],
        "renderer receipt changed after export"
    );
    Ok(())
}

/// Verify retained sources, population, command logs and decoded figures without rendering.
pub fn verify(output: &Path) -> Result<Value> {
    let output = output.canonicalize()?;
    let receipt = inputs::load_json(&output.join("plot_manifest.json"))?;
    ensure!(
        receipt["schema_version"] == 2
            && receipt["scope"] == "trial_inspections"
            && receipt["status"] == "completed",
        "trial export has no completed native receipt"
    );
    let root = Path::new(receipt["run_root"].as_str().context("run root")?);
    let config = Path::new(receipt["config_source"].as_str().context("config source")?);
    ensure!(
        inputs::fingerprint(config)? == receipt["config"]
            && inputs::fingerprint(&output.join("bekiper_config.toml"))?["sha256"]
                == receipt["config"]["sha256"],
        "plot config differs from retained receipt"
    );
    let records = receipt["trials"].as_array().context("trial records")?;
    let selected: Vec<_> = if receipt["coverage_scope"] == "selected_trials_only" {
        records
            .iter()
            .map(|r| {
                r["trial"]
                    .as_str()
                    .context("selected trial")
                    .map(str::to_owned)
            })
            .collect::<Result<_>>()?
    } else {
        Vec::new()
    };
    let trials = discover(&root.join("data"), &selected)?;
    ensure!(
        trials.len() == records.len(),
        "trial receipt population differs from run"
    );
    for (trial, record) in trials.iter().zip(records) {
        ensure!(
            record["trial"] == trial.name
                && record["analysis_id"] == trial.analysis_id
                && record["manifest"] == json!(trial.manifest)
                && record["evaluated_table"] == json!(trial.table)
                && record["manifest_identity"] == trial.manifest_identity
                && record["evaluated_identity"] == trial.table_identity,
            "inspection input or identity differs from receipt: {}",
            trial.name
        );
        verify_trial_renderer(&output, record)?;
    }
    match receipt["coverage_scope"].as_str() {
        Some("pre_filter_population") => {
            let path = Path::new(
                receipt["population_source"]
                    .as_str()
                    .context("population source")?,
            );
            ensure!(
                inputs::fingerprint(path)? == receipt["population_identity"]
                    && coverage(path, &trials)? == receipt["coverage"],
                "population differs from retained receipt"
            );
        }
        Some("discovered_trials_only" | "selected_trials_only") => {
            ensure!(
                receipt["coverage"].is_null() && receipt["population_source"].is_null(),
                "mislabelled export coverage"
            );
        }
        _ => anyhow::bail!("unsupported trial coverage scope"),
    }
    let commands = receipt["commands"].as_array().context("command logs")?;
    ensure!(
        commands.len() >= trials.len() * 2,
        "missing render command evidence"
    );
    for (index, command) in commands.iter().enumerate() {
        ensure!(
            command["status"] == "passed" && command["exit_status"] == 0,
            "failed render command"
        );
        for stream in ["stdout", "stderr"] {
            ensure!(
                inputs::fingerprint(&output.join(format!("logs/{index:06}.{stream}.log")))?
                    == command[stream],
                "render log differs from receipt"
            );
        }
    }
    let files = receipt["files"].as_array().context("export files")?;
    let mut seen = BTreeSet::new();
    let mut stems: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for file in files {
        let relative = file["path"].as_str().context("artifact path")?;
        ensure!(
            pipeline::safe_relative_path(relative) && seen.insert(relative),
            "unsafe or duplicate export artifact"
        );
        let path = output.join(relative);
        ensure!(
            path.canonicalize()?.starts_with(&output)
                && !fs::symlink_metadata(&path)?.file_type().is_symlink()
                && fs::metadata(&path)?.len() > 0
                && inputs::fingerprint(&path)? == file["identity"],
            "artifact differs from receipt: {relative}"
        );
        if path.extension().is_some_and(|e| e == "png") {
            let record = records
                .iter()
                .find(|r| {
                    relative.starts_with(&format!(
                        "trials/{}/figures/",
                        r["trial"].as_str().unwrap_or("")
                    ))
                })
                .context("unbound trial figure")?;
            let dpi = record["dpi"].as_u64().context("captured figure DPI")?;
            ensure!(
                read_toml(&output.join("bekiper_config.toml"))?["inspect"]["plot"]["min_dpi"]
                    == dpi,
                "figure DPI differs from config"
            );
            bekiper_post_analysis::verification::verify_png(&path, dpi)?;
            stems
                .entry(record["trial"].as_str().context("trial name")?.to_owned())
                .or_default()
                .insert(
                    path.file_stem()
                        .and_then(|n| n.to_str())
                        .context("plot stem")?
                        .to_owned(),
                );
        } else {
            ensure!(
                path.extension().is_some_and(|e| e == "mp4")
                    && records.iter().any(|r| relative.starts_with(&format!(
                        "trials/{}/videos/",
                        r["trial"].as_str().unwrap_or("")
                    ))),
                "unbound export artifact"
            );
        }
    }
    for record in records {
        let name = record["trial"].as_str().context("trial name")?;
        let expected: BTreeSet<_> = record["plot_stems"]
            .as_array()
            .context("declared plot stems")?
            .iter()
            .map(|s| s.as_str().context("plot stem").map(str::to_owned))
            .collect::<Result<_>>()?;
        ensure!(
            !expected.is_empty() && stems.get(name) == Some(&expected),
            "missing declared trial figures: {name}"
        );
    }
    ensure!(
        inputs::load_json(&output.join("plot_manifest.json"))? == receipt,
        "trial receipt changed during verification"
    );
    unchanged(&trials)?;
    ensure!(
        inputs::fingerprint(config)? == receipt["config"]
            && inputs::fingerprint(&output.join("bekiper_config.toml"))?["sha256"]
                == receipt["config"]["sha256"],
        "plot config changed during verification"
    );
    if let Some(source) = receipt["population_source"].as_str() {
        ensure!(
            inputs::fingerprint(Path::new(source))? == receipt["population_identity"],
            "population changed during verification"
        );
    }
    Ok(
        json!({"schema_version":2,"artifact_status":"verified","scope":"trial_inspections",
        "coverage_scope":receipt["coverage_scope"],"coverage":receipt["coverage"],"trials":trials.len(),"artifacts":files.len(),
        "driver":receipt["driver"],"analysis_admitted":false,"scientific_readiness":"unverified"}),
    )
}
