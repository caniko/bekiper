#[cfg(feature = "pose-gpu")]
pub(crate) fn run_augment(
    experiment: &ExperimentConfig,
    config: &IngressConfig,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    let mut local_model = None;
    run_augment_with_model(experiment, config, exp_dir, &mut local_model)
}

#[cfg(feature = "pose-gpu")]
pub(crate) fn run_augment_with_model(
    experiment: &ExperimentConfig,
    config: &IngressConfig,
    exp_dir: &ExperimentOutputDir,
    shared_model: &mut Option<OnnxPoseModel>,
) -> Result<()> {
    let poses_path = exp_dir.poses_parquet();
    if !poses_path.exists() {
        return Err(BekiperError::Data(format!(
            "poses.parquet not found at {} — run pose stage first",
            poses_path.display()
        )));
    }

    let pixel_perimeters = load_pixel_perimeters(exp_dir)?;
    let (mpp, scale_info) = resolve_scale(experiment, &pixel_perimeters, exp_dir)?;
    let mpp_y = scale_info.meters_per_pixel_y();
    if config.pipeline.invert_y {
        let height = resolve_source_frame_height_px(experiment, &config.pipeline)?.unwrap_or(0)
            as f64
            * mpp_y;
        let perimeters = PerimetersFile {
            perimeters: pixel_perimeters
                .iter()
                .map(|p| p.scale_xy(mpp.0, mpp_y).reflect_y(height))
                .collect(),
        };
        std::fs::write(
            exp_dir.perimeters_toml(),
            toml::to_string_pretty(&perimeters).map_err(|e| BekiperError::Other(e.to_string()))?,
        )?;
    }

    let labels = discover_labels(experiment, &poses_path)?;
    let (pre_builder, post_pipeline, skeleton_plan) = augmentation_plan(
        experiment,
        &config.pipeline,
        &labels,
        &pixel_perimeters,
        mpp,
        scale_info.meters_per_pixel_y,
    )?;
    let raw_lf = attach_source_clock(scan_file(&poses_path)?, experiment)?;
    let output_path = exp_dir.augmented_parquet();
    let augmented = apply_augmentation(
        raw_lf,
        pre_builder,
        post_pipeline,
        skeleton_plan,
        |fitted, requests, repaired_df, final_eval, redetected_rows| {
            redetect_requested_keypoints(
                experiment,
                config,
                exp_dir,
                shared_model,
                fitted,
                requests,
                repaired_df,
                final_eval,
                redetected_rows,
            )
        },
    )?;
    let mut df = augmented.collect()?;
    bekiper_pose::io::write_parquet(&mut df, &output_path)?;
    tracing::info!(output = %output_path.display(), experiment = %experiment.name, n_labels = labels.len(), "augmentation complete");
    Ok(())
}

type SkeletonPlan = Option<(
    crate::config::SkeletonRepairConfig,
    bekiper_pose::skeleton::Skeleton,
)>;
