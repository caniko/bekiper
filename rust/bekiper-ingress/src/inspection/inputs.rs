fn write_json(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension("json.pending");
    std::fs::write(&temporary, serde_json::to_vec_pretty(value).map_err(data)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn fingerprint(path: &Path) -> Result<Value> {
    let mut value = bekiper_post_analysis::pipeline::publication_fingerprint(path).map_err(data)?;
    value["path"] = path.to_string_lossy().into_owned().into();
    Ok(value)
}

fn inputs(dir: &ExperimentOutputDir) -> Result<Vec<Value>> {
    let mut paths = vec![
        dir.inspection_toml(),
        dir.evaluated_parquet(),
        dir.root().join("inspection_config.toml"),
        dir.root().join("trial_context.json"),
    ];
    if dir.frames_dir().exists() {
        for entry in std::fs::read_dir(dir.frames_dir())? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "png") {
                paths.push(path);
            }
        }
    }
    paths.sort();
    let mut inventory = paths
        .iter()
        .map(|p| fingerprint(p))
        .collect::<Result<Vec<_>>>()?;
    let manifest = bekiper_core::inspection::InspectionManifest::from_toml(&dir.inspection_toml())?;
    for path in [manifest.video_path, manifest.frame_image]
        .into_iter()
        .flatten()
    {
        let path = if path.is_absolute() {
            path
        } else {
            dir.root().join(path)
        };
        inventory.push(if path.is_file() {
            fingerprint(&path)?
        } else {
            json!({"path":path,"status":"unavailable","reason":"source_media_not_retained"})
        });
    }
    Ok(inventory)
}
