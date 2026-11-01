//! Source fingerprints and portable effective configuration.
use super::*;

pub(super) fn reference_fingerprints(
    config: &Record,
    config_path: &Path,
    directory: &Path,
) -> Result<(Record, Record)> {
    let reference = config["source"]["reference"]
        .as_str()
        .unwrap_or("reference/BEHAVIORAL_RECIPE.md");
    let recipe = config_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(reference);
    if recipe.is_file() {
        let amendment = recipe
            .parent()
            .context("recipe parent")?
            .join("YMAZE_AMENDMENT.json");
        ensure!(
            amendment.is_file(),
            "explicit recipe reference requires YMAZE_AMENDMENT.json beside it"
        );
        std::fs::copy(&recipe, directory.join("BEHAVIORAL_RECIPE.md"))?;
        std::fs::copy(&amendment, directory.join("YMAZE_AMENDMENT.json"))?;
        Ok((
            fingerprint(&directory.join("BEHAVIORAL_RECIPE.md"))?,
            fingerprint(&directory.join("YMAZE_AMENDMENT.json"))?,
        ))
    } else {
        ensure!(
            reference == "reference/BEHAVIORAL_RECIPE.md",
            "configured source reference is missing"
        );
        std::fs::write(directory.join("BEHAVIORAL_RECIPE.md"), crate::SOURCE_RECIPE)?;
        std::fs::write(
            directory.join("YMAZE_AMENDMENT.json"),
            crate::YMAZE_AMENDMENT,
        )?;
        Ok((
            crate::source::embedded_fingerprint("BEHAVIORAL_RECIPE.md", crate::SOURCE_RECIPE),
            crate::source::embedded_fingerprint("YMAZE_AMENDMENT.json", crate::YMAZE_AMENDMENT),
        ))
    }
}

pub(super) fn render_bundle_with_cancel(
    directory: &Path,
    bundle: &str,
    workers: Option<u64>,
    cancelled: &dyn Fn() -> bool,
) -> Result<Record> {
    let mut command = Command::new("python");
    command
        .args(["-m", "bekiper_cli.post_analysis.render_bundle", "--bundle"])
        .arg(directory.join(bundle))
        .arg("--output")
        .arg(directory)
        .env("MPLBACKEND", "Agg");
    if let Some(workers) = workers {
        command.arg("--workers").arg(workers.to_string());
    }
    let status = run_child(
        &mut command,
        &directory.join("renderer.stdout.log"),
        &directory.join("renderer.stderr.log"),
        cancelled,
    )?;
    ensure!(
        status.success(),
        "plotting-bundle renderer failed: {}",
        std::fs::read_to_string(directory.join("renderer.stderr.log"))?
    );
    Ok(
        serde_json::from_slice(&std::fs::read(directory.join("renderer.stdout.log"))?)
            .unwrap_or_else(|_| json!({"status":"rendered","runtime_versions":"not_reported"})),
    )
}

pub(super) fn sanitize_config(config: &mut Record, fingerprints: &[Record]) {
    if let Some(inputs) = config["inputs"].as_object_mut() {
        for (key, value) in inputs.iter_mut() {
            if let Some(role) = key.strip_suffix("_path") {
                *value = json!(
                    fingerprints
                        .iter()
                        .filter(|item| item["role"] == role)
                        .collect::<Vec<_>>()
                );
            }
        }
        if let Some(evidence) = inputs
            .get_mut("upstream_rule_provenance")
            .and_then(Value::as_object_mut)
        {
            for item in evidence.values_mut() {
                if let Some(files) = item["evidence_files"].as_array_mut() {
                    for file in files {
                        if let Some(path) = file["path"].as_str() {
                            file["path"] = Path::new(path)
                                .file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("evidence")
                                .into();
                        }
                    }
                }
            }
        }
    }
    config["outputs"]["directory"] = "published_run".into();
    if config["source"].is_object() {
        config["source"]["reference"] = "BEHAVIORAL_RECIPE.md".into();
    }
    if let Some(object) = config.as_object_mut() {
        object.remove("_native_metadata");
        object.remove("_native_sources");
    }
}
