fn augmentation_plan(
    experiment: &ExperimentConfig,
    pipeline_cfg: &crate::config::PipelineConfig,
    labels: &[String],
    pixel_perimeters: &[PerimeterSpec],
    mpp: MetersPerPixel,
    meters_per_pixel_y: Option<f64>,
) -> Result<(AugmentedBuilder, AugmentedBuilder, SkeletonPlan)> {
    if labels.is_empty() {
        return Err(BekiperError::Data("pose input has no labels".into()));
    }
    let (mut x_cols, mut y_cols, lik_cols) = label_columns(labels);
    let coord_cols: Vec<String> = x_cols.iter().chain(y_cols.iter()).cloned().collect();

    let mut pre_builder = AugmentedBuilder::new();

    let needs_crop = pipeline_cfg.crop_start_seconds > 0.0
        || pipeline_cfg.crop_end_seconds > 0.0
        || pipeline_cfg.target_trial_length_seconds > 0.0;
    if needs_crop {
        let fps = experiment.effective_fps().unwrap_or(30.0);
        pre_builder = pre_builder.add_step(TemporalCrop {
            crop_start_seconds: pipeline_cfg.crop_start_seconds,
            crop_end_seconds: pipeline_cfg.crop_end_seconds,
            target_length_seconds: pipeline_cfg.target_trial_length_seconds,
            fps,
        });
    }

    pre_builder = pre_builder.add_step(LikelihoodFilter::new(
        pipeline_cfg.likelihood_threshold,
        lik_cols.clone(),
        coord_cols.clone(),
    ));

    if pipeline_cfg.likelihood_trim {
        pre_builder = pre_builder.add_step(LikelihoodTrimmer {
            threshold: pipeline_cfg.likelihood_threshold,
            likelihood_columns: lik_cols.clone(),
        });
    }

    if let Some(arena_shape) = find_arena_shape(pixel_perimeters) {
        let reference = bekiper_core::types::CoordinateColumns::new(
            format!("{}_x", labels[0]),
            format!("{}_y", labels[0]),
        );
        pre_builder = pre_builder.add_step(ConfinementNuller {
            perimeter: bekiper_perimeter::perimeter::Perimeter::new("arena", arena_shape),
            reference,
            columns_to_null: coord_cols,
        });
        tracing::debug!("added enclosure confinement nuller");
    }

    let model_midpoint_groups = pipeline_cfg
        .pose_label_model
        .midpoint_groups_for_task(experiment.task_type);
    let mut available_labels = labels.to_vec();
    let applicable_groups = resolve_midpoint_groups(
        labels,
        &model_midpoint_groups,
        &pipeline_cfg.midpoint_groups,
    )?;

    if !applicable_groups.is_empty() {
        pre_builder = pre_builder.add_step(MidpointComputer {
            groups: applicable_groups.clone(),
        });

        for group in &applicable_groups {
            let label = group.output_label.clone();
            x_cols.push(format!("{label}_x"));
            y_cols.push(format!("{label}_y"));
            available_labels.push(label);
        }
    }

    let (candidate_skeleton_cfg, explicit_skeleton) = pipeline_cfg
        .skeleton
        .clone()
        .map(|cfg| (cfg, true))
        .unwrap_or_else(|| {
            (
                pipeline_cfg.model_skeleton_repair_config(experiment.task_type),
                false,
            )
        });

    let skeleton_plan = {
        let skeleton = candidate_skeleton_cfg.preset.resolve();
        let missing = skeleton.missing_labels(available_labels.iter().map(String::as_str));
        if !missing.is_empty() {
            if explicit_skeleton {
                return Err(BekiperError::Config(format!(
                    "pipeline.skeleton preset '{}' requires missing labels after midpoint expansion: {}",
                    skeleton.name,
                    missing.join(", ")
                )));
            }
            tracing::warn!(
                model = pipeline_cfg.pose_label_model.as_str(),
                preset = %skeleton.name,
                missing = %missing.join(", "),
                "model skeleton preset skipped because required labels are unavailable after midpoint expansion"
            );
            None
        } else {
            Some((candidate_skeleton_cfg, skeleton))
        }
    };

    let source_frame_height_px = resolve_source_frame_height_px(experiment, pipeline_cfg)?;
    let mut post_pipeline = build_post_augment_pipeline(
        &x_cols,
        &y_cols,
        pipeline_cfg,
        mpp.0,
        meters_per_pixel_y,
        source_frame_height_px,
        skeleton_plan.as_ref().map(|(_, skeleton)| skeleton.clone()),
    );
    // Directions must use the same smoothed metric-space positions and axis
    // transform as their origins. This applies to recorded and virtual heads.
    let has = |label: &str| available_labels.iter().any(|candidate| candidate == label);
    if has("center_ear") && has("nose") {
        let mut step = SnoutDirectionStep::default();
        if let Some(params) = &experiment.task_params {
            let known = |c: &bekiper_core::types::CoordinateColumns| {
                x_cols.contains(&c.x) && y_cols.contains(&c.y)
            };
            for (configured, slot, name) in [
                (&params.nose_coords, &mut step.nose, "nose_coords"),
                (&params.head_coords, &mut step.head, "head_coords"),
            ] {
                if let Some(columns) = configured {
                    if !known(columns) {
                        return Err(BekiperError::Config(format!(
                            "task_params.{name} ({}, {}) are not augmented position columns; \
                             snout_dx/snout_dy cannot be derived from them",
                            columns.x, columns.y
                        )));
                    }
                    *slot = columns.clone();
                }
            }
        }
        post_pipeline = post_pipeline.add_step(step);
    }
    if has("center_ear") && has("left_ear") && has("right_ear") {
        post_pipeline = post_pipeline.add_step(EarDirectionStep);
    }
    Ok((pre_builder, post_pipeline, skeleton_plan))
}
