fn verify_files(files: &Value) -> Result<()> {
    for file in files
        .as_array()
        .ok_or_else(|| data("inspection receipt requires file inventory"))?
    {
        let path = file["path"]
            .as_str()
            .ok_or_else(|| data("inspection receipt missing path"))?;
        let actual = fingerprint(Path::new(path))?;
        if actual["sha256"] != file["sha256"]
            || actual["bytes"] != file["bytes"]
            || actual["bytes"] == 0
        {
            return Err(data(format!(
                "inspection artifact changed or empty: {path}"
            )));
        }
    }
    Ok(())
}

fn validate_render_receipt(dir: &ExperimentOutputDir, receipt: &Value) -> Result<Vec<Value>> {
    use std::collections::BTreeSet;
    if receipt["status"] != "complete" {
        return Err(data("renderer has no completed receipt"));
    }
    verify_files(&receipt["inputs"])?;
    verify_files(&receipt["artifacts"])?;
    let inventory = receipt["inventory"]
        .as_array()
        .ok_or_else(|| data("renderer missing output inventory"))?;
    let format = receipt["effective_theme"]["fig_format"]
        .as_str()
        .ok_or_else(|| data("renderer missing effective format"))?;
    let mut expected = BTreeSet::new();
    let mut declared = BTreeSet::new();
    for job in inventory {
        let stem = job["stem"]
            .as_str()
            .ok_or_else(|| data("render job missing stem"))?;
        if job["selected"] == true {
            expected.insert(
                dir.figures_dir()
                    .join(format!("{stem}{format}"))
                    .canonicalize()?,
            );
        } else if job["reason"].is_null() {
            return Err(data("unselected render job has no reason"));
        }
        for feature in job["features"].as_array().into_iter().flatten() {
            // Per-object and all-object stories intentionally share bindings.
            // Coverage is a set; artifact identity remains unique below.
            declared.insert(
                feature
                    .as_str()
                    .ok_or_else(|| data("renderer feature identity"))?
                    .to_owned(),
            );
        }
    }
    let manifest = bekiper_core::inspection::InspectionManifest::from_toml(&dir.inspection_toml())?;
    if declared
        != manifest
            .feature_inspections
            .iter()
            .map(|f| f.label().to_owned())
            .collect()
    {
        return Err(data("renderer inventory does not cover declared features"));
    }
    let artifacts = receipt["artifacts"]
        .as_array()
        .ok_or_else(|| data("renderer inventory"))?
        .clone();
    let actual = artifacts
        .iter()
        .map(|f| {
            f["path"]
                .as_str()
                .ok_or_else(|| data("renderer artifact path"))
                .and_then(|p| std::fs::canonicalize(p).map_err(Into::into))
        })
        .collect::<Result<BTreeSet<_>>>()?;
    if actual != expected || actual.len() != artifacts.len() {
        return Err(data("renderer artifacts do not match selected outputs"));
    }
    Ok(artifacts)
}
