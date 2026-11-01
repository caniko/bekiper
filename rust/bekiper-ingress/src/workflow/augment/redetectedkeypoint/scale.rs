#[cfg(feature = "pose-gpu")]
fn resolve_scale(
    experiment: &ExperimentConfig,
    pixel_perimeters: &[PerimeterSpec],
    exp_dir: &ExperimentOutputDir,
) -> Result<(MetersPerPixel, bekiper_core::scale::ScaleDerivationInfo)> {
    let load_scale_toml = |path: std::path::PathBuf| {
        std::fs::read_to_string(&path).ok().and_then(|content| {
            toml::from_str::<bekiper_core::scale::ScaleDerivationInfo>(&content).ok()
        })
    };

    let (mpp, scale_info) = if let Some(info) = load_scale_toml(exp_dir.scale_toml()) {
        let mpp = MetersPerPixel(info.meters_per_pixel);
        tracing::info!(
            meters_per_pixel = mpp.0,
            method = %info.method,
            "reusing previously derived scale checkpoint"
        );
        (mpp, info)
    } else if let Some(info) = load_scale_toml(exp_dir.annotation_scale_toml()) {
        let mpp = MetersPerPixel(info.meters_per_pixel);
        tracing::info!(
            meters_per_pixel = mpp.0,
            method = %info.method,
            "using annotation-derived scale"
        );
        (mpp, info)
    } else {
        experiment
            .meters_per_pixel
            .resolve_with_info(pixel_perimeters)?
    };

    tracing::info!(
        meters_per_pixel = mpp.0,
        method = %scale_info.method,
        "resolved scale"
    );

    let scale_str =
        toml::to_string_pretty(&scale_info).map_err(|e| BekiperError::Other(e.to_string()))?;
    exp_dir.ensure_checkpoints_dir()?;
    std::fs::write(exp_dir.scale_toml(), scale_str)?;

    if !pixel_perimeters.is_empty() {
        let meter_perimeters: Vec<PerimeterSpec> = pixel_perimeters
            .iter()
            .map(|perimeter| perimeter.scale_xy(mpp.0, scale_info.meters_per_pixel_y()))
            .collect();
        let pf = PerimetersFile {
            perimeters: meter_perimeters,
        };
        let toml_str =
            toml::to_string_pretty(&pf).map_err(|e| BekiperError::Other(e.to_string()))?;
        std::fs::write(exp_dir.perimeters_toml(), toml_str)?;
        tracing::info!("scaled perimeters to meter-space");
    }

    Ok((mpp, scale_info))
}
