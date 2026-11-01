#[cfg(feature = "pose-gpu")]
fn ensure_redetect_model<'a>(
    experiment: &ExperimentConfig,
    config: &IngressConfig,
    shared_model: &'a mut Option<OnnxPoseModel>,
) -> Result<Option<&'a mut OnnxPoseModel>> {
    if experiment.video.is_none() {
        return Ok(None);
    }

    let Some(pose_cfg) = experiment.pose_model.as_ref() else {
        return Ok(None);
    };

    if shared_model.is_none() {
        let spec = crate::workflow::pose::resolve_model_spec(&pose_cfg.spec)?;
        *shared_model = Some(OnnxPoseModel::load_with_options(
            &pose_cfg.path,
            spec,
            config.gpu.to_pose_options(),
        )?);
    }

    Ok(shared_model.as_mut())
}

/// Re-detect flagged keypoints when the experiment supplies a video and model.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "pose-gpu")]
fn redetect_requested_keypoints(
    experiment: &ExperimentConfig,
    config: &IngressConfig,
    exp_dir: &ExperimentOutputDir,
    shared_model: &mut Option<OnnxPoseModel>,
    fitted: &FittedSkeletonRepair,
    requests: &[FrameSkeletonViolations],
    repaired_df: &mut DataFrame,
    final_eval: &mut SkeletonViolationEvaluation,
    redetected_rows: &mut BTreeSet<usize>,
) -> Result<()> {
    if let Some(model) = ensure_redetect_model(experiment, config, shared_model)? {
        apply_redetected_candidates(
            experiment,
            exp_dir,
            model,
            config.pipeline.likelihood_threshold,
            fitted,
            requests,
            repaired_df,
            final_eval,
            redetected_rows,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "pose-gpu")]
fn apply_redetected_candidates(
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
    model: &mut OnnxPoseModel,
    likelihood_threshold: f64,
    fitted: &FittedSkeletonRepair,
    requests: &[FrameSkeletonViolations],
    repaired_df: &mut DataFrame,
    final_eval: &mut SkeletonViolationEvaluation,
    redetected_rows: &mut BTreeSet<usize>,
) -> Result<()> {
    let Some(video_path) = experiment.video.as_ref() else {
        return Ok(());
    };

    let crop_region = crate::workflow::pose::compute_annotation_crop(experiment, exp_dir, model)?;
    let mut decoder = VideoDecoder::open(video_path)?;

    apply_redetected_candidates_with(
        fitted,
        requests,
        likelihood_threshold,
        repaired_df,
        final_eval,
        redetected_rows,
        |frame| match redetect_frame(&mut decoder, model, crop_region, frame) {
            Ok(prediction) => Ok(prediction),
            Err(error) => {
                tracing::warn!(frame, error = %error, "frame re-detection failed");
                Ok(None)
            }
        },
    )
}
