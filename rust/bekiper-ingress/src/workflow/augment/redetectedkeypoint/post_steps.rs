fn build_post_augment_pipeline(
    x_cols: &[String],
    y_cols: &[String],
    pipeline_cfg: &crate::config::PipelineConfig,
    meters_per_pixel: f64,
    meters_per_pixel_y: Option<f64>,
    source_frame_height_px: Option<u32>,
    skeleton: Option<bekiper_pose::skeleton::Skeleton>,
) -> AugmentedBuilder {
    let builder = AugmentedBuilder::new().add_step(HighVelocityFilter {
        max_velocity: pipeline_cfg.max_velocity,
        x_columns: x_cols.to_vec(),
        y_columns: y_cols.to_vec(),
        max_iterations: pipeline_cfg.high_velocity_iterations,
        sigma_threshold: pipeline_cfg.displacement_sigma_threshold,
    });

    let kalman_columns = || {
        x_cols
            .iter()
            .zip(y_cols.iter())
            .map(|(x_col, y_col)| {
                let label = x_col
                    .strip_suffix("_x")
                    .unwrap_or(x_col.as_str())
                    .to_string();
                KalmanSmootherColumns {
                    likelihood_col: Some(format!("{label}_likelihood")),
                    label,
                    x_col: x_col.clone(),
                    y_col: y_col.clone(),
                }
            })
            .collect()
    };

    let builder = if let Some(skeleton) = skeleton {
        builder.add_step(SkeletonKinematicSmoother {
            skeleton,
            columns: kalman_columns(),
            process_noise: pipeline_cfg.kalman_process_noise,
            measurement_noise: pipeline_cfg.kalman_measurement_noise,
            min_likelihood: pipeline_cfg.kalman_min_likelihood,
            max_gap_frames: pipeline_cfg.kalman_max_gap_frames,
            constraint_iterations: 2,
            constraint_strength: 0.75,
        })
    } else {
        match pipeline_cfg.smoothing_method {
            SmoothingMethod::Median => builder.add_step(MedianSmoother {
                window_size: pipeline_cfg.smoothing_window,
                columns: x_cols.iter().chain(y_cols.iter()).cloned().collect(),
            }),
            SmoothingMethod::Kalman => builder.add_step(ConfidenceKalmanSmoother {
                columns: kalman_columns(),
                process_noise: pipeline_cfg.kalman_process_noise,
                measurement_noise: pipeline_cfg.kalman_measurement_noise,
                min_likelihood: pipeline_cfg.kalman_min_likelihood,
                max_gap_frames: pipeline_cfg.kalman_max_gap_frames,
            }),
        }
    };

    builder.add_step(CoordinateTransform {
        meters_per_pixel,
        meters_per_pixel_y,
        invert_y: pipeline_cfg.invert_y,
        y_height: source_frame_height_px.unwrap_or(0) as f64,
        x_columns: x_cols.to_vec(),
        y_columns: y_cols.to_vec(),
    })
}

fn resolve_source_frame_height_px(
    experiment: &ExperimentConfig,
    pipeline_cfg: &crate::config::PipelineConfig,
) -> Result<Option<u32>> {
    if !pipeline_cfg.invert_y {
        return Ok(None);
    }

    if let Some((_, height)) = crate::analysis::resolve_source_resolution(experiment)? {
        return Ok(Some(height));
    }

    Err(BekiperError::Config(format!(
        "pipeline.invert_y=true but no source frame height is available for experiment '{}'",
        experiment.name
    )))
}
