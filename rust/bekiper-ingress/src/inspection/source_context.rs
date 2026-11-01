#[cfg(feature = "pose-video")]
pub(crate) fn source_overlay_dataframe(
    manifest: &bekiper_core::inspection::InspectionManifest,
    df: polars::prelude::DataFrame,
) -> Result<polars::prelude::DataFrame> {
    use polars::prelude::*;
    if !manifest.settings.invert_y {
        return Ok(df);
    }
    let height = manifest.metric_source_height();
    if !height.is_finite() || height <= 0. {
        return Err(data(
            "inspection Y transform needs source resolution and scale",
        ));
    }
    let (positions, vectors) = manifest.overlay_y_bindings();
    let expressions = positions
        .iter()
        .map(|name| (lit(height) - col(name)).alias(name))
        .chain(vectors.iter().map(|name| (-col(name)).alias(name)))
        .collect::<Vec<_>>();
    Ok(df.lazy().with_columns(expressions).collect()?)
}

pub fn capture_context(
    dir: &ExperimentOutputDir,
    experiment: &ExperimentConfig,
    settings: &InspectConfig,
) -> Result<()> {
    dir.ensure_dir()?;
    let table = toml::Value::try_from(settings).map_err(data)?;
    let content = toml::to_string_pretty(&toml::Table::from_iter([("inspect".into(), table)]))
        .map_err(data)?;
    std::fs::write(dir.root().join("inspection_config.toml"), content)?;
    let context = json!({"schema_version":1,"experiment":experiment,
        "annotation_dir":experiment.annotation_dir,"frame_timing":experiment.frame_timing,
        "video_resolution":experiment.video_resolution,"frame_image":experiment.frame_image,
        "timestamps_file":experiment.timestamps_file,"animal_id":experiment.animal_id,
        "session_context":experiment.session_context,"inspection":settings});
    write_json(&dir.root().join("trial_context.json"), &context)
}
