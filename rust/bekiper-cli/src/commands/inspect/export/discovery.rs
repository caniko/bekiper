fn read_toml(path: &Path) -> Result<Value> {
    Ok(bekiper_core::handoff::from_str(&fs::read_to_string(path)?)?)
}

fn walk(path: &Path, manifests: &mut BTreeMap<PathBuf, PathBuf>) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        ensure!(
            !kind.is_symlink(),
            "symlink in inspection tree: {}",
            entry.path().display()
        );
        if kind.is_dir() {
            walk(&entry.path(), manifests)?;
        } else if entry
            .file_name()
            .to_str()
            .is_some_and(|n| n.ends_with("_inspection.toml"))
        {
            ensure!(
                manifests.insert(path.to_owned(), entry.path()).is_none(),
                "ambiguous inspection manifests: {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn discover(data: &Path, selected: &[String]) -> Result<Vec<Trial>> {
    let mut manifests = BTreeMap::new();
    walk(data, &mut manifests)?;
    ensure!(
        !manifests.is_empty(),
        "no inspection manifests under run/data"
    );
    let requested: BTreeSet<_> = selected.iter().collect();
    ensure!(
        requested.len() == selected.len(),
        "duplicate trial selection"
    );
    let mut found = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let mut trials = Vec::new();
    for (dir, manifest) in manifests {
        let name = dir
            .strip_prefix(data)?
            .to_str()
            .context("non-UTF-8 trial path")?
            .to_owned();
        found.insert(name.clone());
        if !requested.is_empty() && !requested.contains(&name) {
            continue;
        }
        let analysis_id = manifest
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix("_inspection.toml"))
            .context("inspection identity")?
            .to_owned();
        ensure!(
            !analysis_id.is_empty() && ids.insert(analysis_id.clone()),
            "duplicate/empty analysis identity"
        );
        let payload = read_toml(&manifest)?;
        if let Some(recorded) = payload["experiment"]["name"].as_str() {
            ensure!(
                recorded == analysis_id,
                "manifest experiment identity differs from filename"
            );
        }
        let table = dir.join(format!("{analysis_id}_evaluated.parquet"));
        let tables: Vec<_> = fs::read_dir(&dir)?
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.ends_with("_evaluated.parquet"))
            })
            .map(|e| e.path())
            .collect();
        ensure!(
            tables == [table.clone()] && fs::metadata(&table)?.len() > 0,
            "missing, empty or ambiguous evaluated table: {name}"
        );
        trials.push(Trial {
            name,
            analysis_id,
            manifest_identity: inputs::fingerprint(&manifest)?,
            table_identity: inputs::fingerprint(&table)?,
            manifest,
            table,
        });
    }
    ensure!(
        requested.iter().all(|n| found.contains(*n)),
        "unknown trial selection"
    );
    Ok(trials)
}
