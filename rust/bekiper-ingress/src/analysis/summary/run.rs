#[cfg(feature = "pose-gpu")]
pub(crate) fn run_summary(
    experiment: &ExperimentConfig,
    config: &IngressConfig,
    exp_dir: &ExperimentOutputDir,
) -> Result<()> {
    if crate::analysis_exclusion::is_analysis_excluded_experiment(experiment) {
        tracing::info!(
            experiment = %experiment.name,
            "skipping summary for analysis-excluded experiment"
        );
        return Ok(());
    }

    let summaries_path = exp_dir.behaviour_summaries_toml();

    let content = std::fs::read_to_string(&summaries_path)?;
    #[derive(serde::Deserialize)]
    struct SummariesFile {
        summaries: Vec<HeuristicSummary>,
        #[serde(default)]
        decompositions: Vec<HeuristicDecompositionSpec>,
        #[serde(default)]
        quadrants: Vec<bekiper_core::inspection::QuadrantSpec>,
    }
    let sf: SummariesFile = toml::from_str(&content)
        .map_err(|e| BekiperError::Config(format!("invalid summaries TOML: {e}")))?;
    let summaries = sf.summaries;
    let decompositions = sf.decompositions;
    let quadrants = sf.quadrants;

    // Load task-specific detailed result if available
    let detailed_result: Option<TaskDetail> = {
        let result_path = exp_dir.behaviour_result_toml();
        if result_path.exists() {
            let content = std::fs::read_to_string(&result_path)?;
            Some(
                toml_str_to_json_value(&content)
                    .map_err(|e| {
                        BekiperError::Config(format!("invalid behaviour_result.toml: {e}"))
                    })
                    .and_then(|value| TaskDetail::from_json_value(experiment.task_type, value))?,
            )
        } else {
            None
        }
    };

    let scale_info = load_scale_info(exp_dir).ok();
    validate_ymaze_source_summary(experiment, exp_dir, detailed_result.as_ref())?;
    let animal_metadata = experiment
        .animal_id
        .or_else(|| experiment.name.split('_').next()?.parse::<u32>().ok())
        .and_then(|id| config.animal_metadata.get(&id));
    let summary_path = exp_dir.result_toml();
    crate::inspection::capture_context(exp_dir, experiment, &config.inspect)?;
    let summary;

    // Export inspection artifacts (*_inspection.toml + *_evaluated.parquet)
    // so that bekiper-inspect can generate figures.
    let behaviour_path = exp_dir.behaviour_parquet();
    if behaviour_path.exists() {
        let perimeters = load_perimeters(exp_dir)?;

        // Read the behaviour parquet to get total_frames for the manifest
        let df = polars::prelude::LazyFrame::scan_parquet(
            behaviour_path.to_str().unwrap().into(),
            Default::default(),
        )
        .map_err(|e| BekiperError::Data(format!("scan behaviour parquet: {e}")))?
        .collect()
        .map_err(|e| BekiperError::Data(format!("collect behaviour parquet: {e}")))?;

        let load = |path: &Path| -> Result<Option<DataFrame>> {
            if path.exists() {
                Ok(Some(bekiper_pose::io::scan_file(path)?.collect()?))
            } else {
                Ok(None)
            }
        };
        let raw = load(&exp_dir.poses_parquet())?;
        let augmented = load(&exp_dir.augmented_parquet())?;
        let mut result = AnalysisResult {
            task_type: experiment.task_type,
            summaries: summaries.clone(),
            decompositions: decompositions.clone(),
            evaluated_df: Some(df),
        };
        let (manifest, mut trial) = assemble_trial(
            experiment,
            animal_metadata,
            &mut result,
            detailed_result.as_ref(),
            &perimeters,
            &quadrants,
            scale_info,
            raw.as_ref(),
            augmented.as_ref(),
            &config.pipeline,
        )?;
        if let Some(parameters) = trial.detailed.get_mut("effective_parameters") {
            super::effective_parameters::apply_override_sources(
                parameters,
                trial.task_type,
                &trial.experiment,
                &config.override_record,
            );
        }
        summary = serde_json::to_value(trial).map_err(|e| BekiperError::Other(e.to_string()))?;

        export_inspection(exp_dir.root(), &experiment.name, &result, &manifest)?;
        tracing::info!(experiment = %experiment.name, "exported inspection artifacts");

        // Pre-extract representative frames so the Python inspector can load
        // PNGs instead of decoding from the source video.
        if let Some(ref video_path) = experiment.video
            && video_path.exists()
        {
            extract_representative_frames(
                video_path,
                &manifest.feature_inspections,
                result.evaluated_df.as_ref(),
                exp_dir,
            )?;
        }
    } else {
        return Err(BekiperError::Data(
            "summary requires evaluated behaviour data".into(),
        ));
    }

    let toml_str =
        json_value_to_toml_string(&summary).map_err(|e| BekiperError::Other(e.to_string()))?;
    std::fs::write(&summary_path, toml_str)?;
    tracing::info!(output = %summary_path.display(), n_heuristics = summaries.len(), "summary complete");

    // Export multi-sheet OpenDocument spreadsheet
    crate::export::export_spreadsheet_with_parameters(
        exp_dir,
        &experiment.name,
        &summaries,
        detailed_result.as_ref(),
        animal_metadata,
        summary["detailed"].get("effective_parameters"),
    )?;

    Ok(())
}
