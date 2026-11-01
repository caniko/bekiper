pub(crate) fn inspection_state_payload(manifest: &InspectionManifest) -> Result<serde_json::Value> {
    serde_json::to_value(
        manifest
            .feature_inspections
            .iter()
            .filter(|f| f.kind() == "state_distribution")
            .collect::<Vec<_>>(),
    )
    .map_err(|e| BekiperError::Data(format!("inspection state payload: {e}")))
}

/// The staged and source-stream adapters share the complete inspection/result
/// boundary. Runtime identity and resolved timing must survive both adapters.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_trial(
    experiment: &ExperimentConfig,
    animal: Option<&crate::config::AnimalMetadata>,
    result: &mut AnalysisResult,
    detailed: Option<&TaskDetail>,
    perimeters: &[PerimeterSpec],
    quadrants: &[bekiper_core::inspection::QuadrantSpec],
    scale: Option<ScaleDerivationInfo>,
    raw: Option<&DataFrame>,
    augmented: Option<&DataFrame>,
    pipeline: &crate::config::PipelineConfig,
) -> Result<(InspectionManifest, crate::aggregate::TrialResult)> {
    use polars::prelude::*;
    let df = result
        .evaluated_df
        .as_mut()
        .ok_or_else(|| BekiperError::Data("analysis returned no evaluated frame".into()))?;
    let (timing, fps) =
        super::behaviour::resolve_behaviour_timing_and_fps(experiment, &df.clone().lazy())?;
    let mut resolved = experiment.clone();
    resolved.video_resolution = super::resolve_source_resolution(experiment)?;
    resolved.fps = bekiper_core::video::FpsConfig::Manual(fps);
    resolved.frame_timing = Some(timing);
    let labels = df
        .get_column_names()
        .iter()
        .filter_map(|n| n.strip_suffix("_x").map(str::to_string))
        .filter(|label| !label.starts_with("__bk_"))
        .collect::<Vec<_>>();
    let mpp = scale
        .as_ref()
        .map(|s| s.meters_per_pixel)
        .or(match experiment.meters_per_pixel {
            bekiper_core::scale::ScaleConfig::Manual(value) => Some(value),
            _ => None,
        })
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or_else(|| BekiperError::Data("inspection requires resolved positive scale".into()))?;
    let mpp_y = scale
        .as_ref()
        .map_or(mpp, |s| s.meters_per_pixel_y.unwrap_or(mpp));
    // The scorer's own scale (fallback/reuse markers included) and resolved task knobs.
    let effective = scale.clone().and_then(|scale| {
        let task = super::behaviour::build_task(&resolved, perimeters, fps)
            .inspect_err(|e| {
                tracing::warn!(experiment = %experiment.name, error = %e, "effective_parameters omitted: task build failed");
            })
            .ok()?
            .effective_parameters()?;
        Some(super::effective_parameters::effective_parameters(
            experiment, &task, &scale,
        ))
    });
    if effective.is_none() && scale.is_none() {
        tracing::warn!(experiment = %experiment.name, "effective_parameters omitted: no resolved scale");
    }
    let mut manifest = build_inspection_manifest(
        &resolved,
        &labels,
        &result.summaries,
        &result.decompositions,
        perimeters,
        quadrants,
        df.height() as u64,
        mpp,
        scale,
        detailed,
        Some(df),
    );
    for (stage, source, factor) in [
        ("raw_pose", raw, (mpp, mpp_y)),
        ("augmented", augmented, (1.0, 1.0)),
    ] {
        if let Some(source) = source {
            retain_tracking_stage(df, source, &labels, stage, factor, &mut manifest)?;
            if stage == "raw_pose" && pipeline.invert_y {
                let height = resolved
                    .video_resolution
                    .map(|(_, h)| h as f64 * mpp_y)
                    .ok_or_else(|| {
                        BekiperError::Config(
                            "raw tracking Y inversion needs source resolution".into(),
                        )
                    })?;
                for binding in manifest
                    .coordinate_columns
                    .iter()
                    .filter(|(name, _)| name.starts_with("__bk_stage__raw_pose__"))
                    .map(|(_, columns)| columns.y.clone())
                    .collect::<Vec<_>>()
                {
                    let transformed = df
                        .clone()
                        .lazy()
                        .select([(lit(height) - col(&binding)).alias(&binding)])
                        .collect()?;
                    df.with_column(transformed.column(&binding)?.clone())?;
                }
            }
        }
    }
    emit_tracking_inspections(&labels, &mut manifest);
    manifest.settings.invert_y = pipeline.invert_y;
    // Decompositions may use an evaluated/eroded perimeter whose label is not
    // part of the input arena. Retain that exact native geometry for parent
    // features as well as component panels; never recreate it from its name.
    for component in result
        .decompositions
        .iter()
        .flat_map(|spec| &spec.components)
    {
        for perimeter in [&component.evaluated_perimeter, &component.buffer_perimeter]
            .into_iter()
            .flatten()
        {
            if !manifest
                .perimeters
                .iter()
                .any(|p| p.label() == perimeter.label())
            {
                manifest.perimeters.push(perimeter.clone());
            }
        }
    }
    manifest.compute_pixel_geometry();
    // An OFT task may derive the center from the configured arena. Bind the
    // task-specific composite to that evaluated geometry when no canonical
    // center outline was supplied. Explicit NORT outlines remain authoritative.
    if experiment.task_type == TaskType::OpenField
        && !manifest
            .perimeters
            .iter()
            .any(|p| p.label() == "center_zone")
    {
        for feature in &mut manifest.feature_inspections {
            if let bekiper_core::inspection::FeatureInspection::CenterPeriphery {
                center_perimeter_label,
                ..
            } = feature
            {
                *center_perimeter_label = "in_center".into();
            }
        }
    }
    crate::inspection::validate_bindings(&manifest, df)?;
    crate::inspection::validate_capabilities(&manifest, &result.decompositions, detailed)?;
    let mut trial = crate::aggregate::TrialResult {
        experiment: experiment.name.clone(),
        task_type: experiment.task_type,
        analysis_role: variant_analysis_role(experiment).or_else(|| {
            crate::analysis_exclusion::aggregate_role_for_experiment(experiment).map(str::to_string)
        }),
        fps,
        meters_per_pixel: mpp,
        animal_metadata: animal
            .map(serde_json::to_value)
            .transpose()
            .map_err(|e| BekiperError::Other(e.to_string()))?
            .unwrap_or_default(),
        session_context: serde_json::to_value(&experiment.session_context)
            .map_err(|e| BekiperError::Other(e.to_string()))?,
        summaries: result
            .summaries
            .iter()
            .map(serde_json::to_value)
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| BekiperError::Other(e.to_string()))?,
        detailed: detailed.map(TaskDetail::as_json_value).unwrap_or_default(),
    };
    if trial.detailed.is_object() {
        trial.detailed["analysis_timing"] = serde_json::json!({"fps":fps,"is_vfr":resolved.frame_timing.as_ref().is_some_and(|t| t.is_vfr),
            "clock_column":if df.column("elapsed_s").is_ok() {Some("elapsed_s")} else {None},
            "rows":df.height(),"source_frame_column":if df.column("frame").is_ok() {Some("frame")} else {None}});
        trial.detailed["analysis_coordinate_transform"] = serde_json::json!({"meters_per_pixel":mpp,"invert_y":pipeline.invert_y,
            "source_resolution":resolved.video_resolution});
        if let Some(effective) = effective {
            trial.detailed["effective_parameters"] = effective;
        }
        trial.detailed["native_state_distributions"] = inspection_state_payload(&manifest)?;
    }
    Ok((manifest, trial))
}

// ---------------------------------------------------------------------------
// Frame pre-extraction for the Python inspector
// ---------------------------------------------------------------------------
