fn run(
    config: &IngressConfig,
    run_root: &Path,
    trials: &[TrialResult],
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Record> {
    let settings = &config.post_analysis;
    if settings.plan_path.is_none() && settings.frozen_plan.is_none() {
        return descriptive::run(config, run_root, trials, cancelled);
    }
    // Ordinary workflow loading makes plan_path absolute. A sealed registry
    // supplies an embedded plan with absolute restored asset paths instead.
    let parent = run_root.parent().unwrap_or(Path::new(".")).to_path_buf();
    let plan_path = settings
        .resolved_plan_path(&parent)
        .unwrap_or_else(|| parent.join("captured-post-analysis-plan.json"));
    let mut plan = if let Some(plan) = &settings.frozen_plan {
        plan.clone()
    } else {
        // Resolve and fingerprint the same opened plan bytes. Project overlays
        // below own presentation settings, so standalone environment options
        // must not leak into this constructor path.
        native(inputs::resolved_config(
            &plan_path,
            &RunOptions {
                ymaze_dwell: settings.ymaze_dwell_overrides,
                ..Default::default()
            },
            &|_| None,
        ))?
    };
    require(plan.is_object(), "post-analysis plan must be a JSON object")?;
    require(
        !cancelled(),
        "post-analysis cancelled before input validation",
    )?;
    let mut fingerprints = settings.frozen_fingerprints.clone();
    let plan_base = plan_path.parent().unwrap_or(Path::new("."));
    // This path consumes fresh native measurements. Retain configured file
    // identities in the frozen plan, but never substitute those measurements.
    let mut metadata_plan = plan.clone();
    for assay in ["oft", "ymaze", "nort"] {
        metadata_plan["inputs"][format!("{assay}_path")] = Value::Null;
    }
    let (metadata, descriptors) = native(inputs::load_inputs(&metadata_plan, plan_base))?;
    fingerprints.extend(descriptors);
    let mut enriched_trials = trials.to_vec();
    if let Some(settings) = &config.aggregate.metadata {
        let base = super::run::aggregate_metadata_base_dir(config, run_root);
        let path = base.join(&settings.path);
        let before = native(inputs::fingerprint(&path));
        match super::metadata::load_aggregate_metadata(&base, settings) {
            Ok(workbook) => {
                let mut descriptor = before?;
                require(
                    descriptor == native(inputs::fingerprint(&path))?,
                    "animal metadata changed during cohort capture",
                )?; // coverage: unreachable — needs the workbook to change between two reads
                descriptor["role"] = "aggregate_animal_metadata".into();
                descriptor["sheet"] = settings.sheet.clone().into();
                fingerprints.push(descriptor);
                // Preserve all registered exclusions in the cohort ledger. Only
                // the workbook exporter applies ExcludeMissing as a row filter.
                let policy =
                    if settings.missing_policy == crate::config::MissingMetadataPolicy::Fail {
                        settings.missing_policy
                    } else {
                        crate::config::MissingMetadataPolicy::AllowBlanks
                    };
                super::metadata::enrich_trials_with_metadata(
                    &mut enriched_trials,
                    &workbook,
                    policy,
                )?;
            }
            Err(error)
                if settings.missing_policy == crate::config::MissingMetadataPolicy::AllowBlanks =>
            {
                tracing::warn!(%error, "cohort animal metadata unavailable; using registered plan biology");
            }
            Err(error) => return Err(error),
        }
    }
    let rows = attach_native_trials(&metadata, &enriched_trials, &plan)?;
    for trial in trials {
        if trial.task_type == TaskType::Cheeseboard {
            continue;
        }
        let bytes = native(serde_json::to_string(trial))?;
        let mut descriptor =
            bekiper_post_analysis::source::embedded_fingerprint("native_trial.json", &bytes);
        descriptor["role"] = "native_trial_result".into();
        descriptor["experiment"] = trial.experiment.clone().into();
        fingerprints.push(descriptor);
    }
    if !plan["outputs"].is_object() {
        plan["outputs"] = json!({});
    }
    if plan["outputs"]["requested_outputs"].is_null() {
        plan["outputs"]["requested_outputs"] =
            json!(native(inputs::requested_output_ids(&plan, &rows))?);
    }
    apply_project_outputs(&mut plan, settings)?;
    let run_id = settings.run_id.clone().unwrap_or_else(|| {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("cohort-{timestamp}-{}", std::process::id())
    });
    let output = post_analysis_output_path(run_root, &run_id)?;
    let output = if output.is_absolute() {
        output
    } else {
        std::env::current_dir()?.join(output)
    };
    let options = RunOptions {
        primary_thresholds: settings.ymaze_primary_thresholds.clone(),
        contexts: settings.contexts.clone(),
        render_workers: settings.render_workers,
        primary_workers: Some(
            settings.primary_workers.min(
                config
                    .project
                    .runtime
                    .num_threads
                    .unwrap_or(settings.primary_workers),
            ),
        ),
        // File plans were resolved above before validation. Captured plans
        // cross the same resolver at the native-record execution boundary.
        ymaze_dwell: if settings.frozen_plan.is_some() {
            settings.ymaze_dwell_overrides
        } else {
            Default::default()
        },
        output: Some(output),
        ..Default::default()
    };
    // An embedded plan must never cause a live plan-path reread for provenance.
    // Its sealed plan fingerprint is already supplied by restoration.
    require(
        settings.frozen_plan.is_none()
            || fingerprints
                .iter()
                .any(|record| record["role"] == "post_analysis_plan"),
        "captured plan requires its sealed fingerprint",
    )?;
    native(bekiper_post_analysis::pipeline::run_records_with_cancel(
        plan,
        rows,
        fingerprints,
        &plan_path,
        &options,
        cancelled,
    ))
}
