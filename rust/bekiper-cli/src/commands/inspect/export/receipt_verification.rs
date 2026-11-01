fn save(output: &Path, receipt: &Value) -> Result<()> {
    let pending = output.join("plot_manifest.pending.json");
    pipeline::write_json(&pending, receipt)?;
    fs::rename(pending, output.join("plot_manifest.json"))?;
    Ok(())
}

fn unchanged(trials: &[Trial]) -> Result<()> {
    for trial in trials {
        ensure!(
            inputs::fingerprint(&trial.manifest)? == trial.manifest_identity
                && inputs::fingerprint(&trial.table)? == trial.table_identity,
            "inspection inputs changed: {}",
            trial.name
        );
    }
    Ok(())
}

fn verify_renderer_receipt(path: &Path, plots: &[PathBuf], dpi: u64) -> Result<Value> {
    let receipt = inputs::load_json(path)?;
    ensure!(
        receipt["status"] == "complete"
            && receipt["effective_theme"]["min_dpi"] == dpi
            && receipt["effective_theme"]["fig_format"] == ".png",
        "renderer receipt lacks captured presentation"
    );
    let sources = receipt["inputs"]
        .as_array()
        .context("renderer source inventory")?;
    ensure!(
        !sources.is_empty(),
        "renderer has no retained source identities"
    );
    let artifacts = receipt["artifacts"]
        .as_array()
        .context("renderer artifact inventory")?;
    let mut actual = BTreeSet::new();
    for item in sources.iter().chain(artifacts) {
        let source = Path::new(item["path"].as_str().context("renderer evidence path")?);
        let identity = inputs::fingerprint(source)?;
        ensure!(
            identity["bytes"] == item["bytes"] && identity["sha256"] == item["sha256"],
            "renderer source/artifact bytes changed: {}",
            source.display()
        );
    }
    for artifact in artifacts {
        let path = Path::new(
            artifact["path"]
                .as_str()
                .context("renderer artifact path")?,
        )
        .canonicalize()?;
        ensure!(actual.insert(path), "duplicate renderer artifact");
    }
    ensure!(
        actual == plots.iter().cloned().collect(),
        "renderer inventory differs from declared figures"
    );
    inputs::fingerprint(path)
}
