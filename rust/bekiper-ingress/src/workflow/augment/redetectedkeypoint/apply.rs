fn apply_augmentation<F>(
    raw_lf: polars::prelude::LazyFrame,
    pre_builder: AugmentedBuilder,
    post_pipeline: AugmentedBuilder,
    skeleton_plan: SkeletonPlan,
    mut redetect: F,
) -> Result<polars::prelude::LazyFrame>
where
    F: FnMut(
        &FittedSkeletonRepair,
        &[FrameSkeletonViolations],
        &mut DataFrame,
        &mut SkeletonViolationEvaluation,
        &mut BTreeSet<usize>,
    ) -> Result<()>,
{
    if let Some((skeleton_cfg, skeleton)) = skeleton_plan {
        let prepared_df = pre_builder.build_and_collect(raw_lf)?;
        let repair_step = SkeletonRepairStep {
            skeleton,
            relative_tolerance: skeleton_cfg.relative_tolerance,
            mad_multiplier: skeleton_cfg.mad_multiplier,
            min_baseline_frames: skeleton_cfg.min_baseline_frames,
        };
        let fitted = repair_step.fit(&prepared_df)?;
        let initial_eval = fitted.evaluate(&prepared_df)?;
        let requests = initial_eval.flagged_rows(&prepared_df)?;
        let mut final_eval = initial_eval.clone();
        let mut redetected_rows = BTreeSet::new();
        let mut repaired_df = fitted.apply(prepared_df, &initial_eval, &BTreeSet::new())?;

        if matches!(
            skeleton_cfg.repair,
            SkeletonRepairMode::RedetectThenInterpolate
        ) && !requests.is_empty()
        {
            redetect(
                &fitted,
                &requests,
                &mut repaired_df,
                &mut final_eval,
                &mut redetected_rows,
            )?;
        }

        let repaired_df = fitted.apply(repaired_df, &final_eval, &redetected_rows)?;
        Ok(post_pipeline.build(repaired_df.lazy()))
    } else {
        let prepared_lf = pre_builder.build(raw_lf);
        Ok(post_pipeline.build(prepared_lf))
    }
}

pub(crate) fn augment_pose(
    raw: polars::prelude::LazyFrame,
    experiment: &ExperimentConfig,
    pipeline: &crate::config::PipelineConfig,
    labels: &[String],
    perimeters: &[PerimeterSpec],
    mpp: MetersPerPixel,
    meters_per_pixel_y: Option<f64>,
) -> Result<polars::prelude::LazyFrame> {
    let (pre, post, skeleton) = augmentation_plan(
        experiment,
        pipeline,
        labels,
        perimeters,
        mpp,
        meters_per_pixel_y,
    )?;
    apply_augmentation(
        attach_source_clock(raw, experiment)?,
        pre,
        post,
        skeleton,
        |_, _, _, _, _| {
            if experiment.video.is_some() && experiment.pose_model.is_some() {
                return Err(BekiperError::Config("direct precomputed analysis cannot perform ONNX re-detection; use the checkpoint GPU workflow for this configuration".into()));
            }
            Ok(())
        },
    )
}
