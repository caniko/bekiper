/// Run the full checkpointed pipeline for all experiments.
pub fn run_project(config_path: &Path, opts: &RunOptions) -> Result<()> {
    run_project_with_post_analysis_overrides(
        config_path,
        opts,
        &crate::config::PostAnalysisOverrides::default(),
    )
}

/// Run with constructor-level post-analysis presentation and plan overrides.
pub fn run_project_with_post_analysis_overrides(
    config_path: &Path,
    opts: &RunOptions,
    post_analysis: &crate::config::PostAnalysisOverrides,
) -> Result<()> {
    let mut config = IngressConfig::from_file_with_set_overrides_from(
        config_path,
        post_analysis,
        opts.ymaze_dwell,
        opts.measurements,
        &opts.set,
        |key| std::env::var(key).ok(),
    )?;
    apply_run_option_overrides(&mut config, opts);
    let base_dir = config_path.parent().unwrap_or(Path::new("."));
    let output_dir = base_dir.join(
        config
            .project
            .output_dir
            .as_deref()
            .unwrap_or(Path::new("output")),
    );
    if output_dir
        .parent()
        .is_some_and(|parent| parent.join("preparation.toml").exists())
    {
        return Err(BekiperError::Config(
            "receipt-bound staged roots require bekiper-analysis run-staged; file-existence freshness cannot admit them".into(),
        ));
    }
    let scheduler = ParallelScheduler::new(&config.project.runtime)?;

    tracing::info!(
        num_threads = scheduler.num_threads(),
        memory_scheduler_enabled = scheduler.memory_scheduler_enabled(),
        max_ram_usage_fraction = scheduler.max_ram_usage_fraction(),
        memory_source = scheduler.memory_source_name(),
        "configured analyze scheduler"
    );

    let mut stages = resolve_stages(opts);
    if opts.no_detect && opts.only_stage == Some(Stage::Detect) {
        return Err(BekiperError::Config(
            "--no-detect cannot be combined with --only detect".into(),
        ));
    }
    if opts.no_detect {
        stages.retain(|stage| *stage != Stage::Detect);
    }

    // Check the required cohort plan before expansion or any native work. The
    // aggregate exporter still owns resolution, fingerprinting and validation.
    if stages.contains(&Stage::Aggregate) && !opts.no_aggregate && config.post_analysis.enabled {
        let plan = config
            .post_analysis
            .effective_frozen_plan(base_dir)
            .map_err(|error| {
                BekiperError::Config(format!(
                    "cannot read registered post-analysis plan: {error}"
                ))
            })?;
        if plan.as_ref().is_some_and(|plan| !plan.is_object()) {
            return Err(BekiperError::Config(
                "enabled post-analysis requires a registered TOML table or historical JSON object plan".into(),
            ));
        }
    }

    // Expand scan / animal_scan entries now that we know which stages will
    // run. This is what actually reads calibration CSVs and may write
    // intermediate `.perimeters-seqN.toml` files under the data dir, so it
    // must happen at pipeline entry, not during config parsing.
    config.expand(base_dir, &stages)?;
    normalize_runtime_paths(&mut config, base_dir);

    // Filter to a single animal when --only-first-animal or --animal N is used.
    if let Some(animal_idx) = opts.animal_index {
        filter_to_animal(&mut config.experiments, animal_idx);
    }

    filter_analysis_excluded_experiments(&mut config.experiments, &stages);
    if stages.contains(&Stage::Inspect) && !opts.no_inspect {
        config
            .inspect
            .validate_video_selection(config.experiments.iter().map(|e| e.name.as_str()))?;
    }

    run_experiments(
        &mut config,
        opts,
        &stages,
        base_dir,
        &output_dir,
        config_path,
        &scheduler,
        &RealRunner,
    )
}

/// Execute `stages` for every expanded experiment.
///
/// Variants of one recording share probe, detect, pose and augment work: the
/// first variant to reach a stage computes it under a `(recording, stage,
/// fingerprint)` lock and its siblings reuse the sealed output
/// (`stage_share`). Behaviour, summary and inspection stay per experiment.
#[allow(clippy::too_many_arguments)]
fn run_experiments(
    config: &mut IngressConfig,
    opts: &RunOptions,
    stages: &[Stage],
    base_dir: &Path,
    output_dir: &Path,
    config_path: &Path,
    scheduler: &ParallelScheduler,
    runner: &dyn StageRunner,
) -> Result<()> {
    let legacy_dirs: Vec<ExperimentOutputDir> = config
        .experiments
        .iter()
        .map(|experiment| exp_output_dir(output_dir, experiment))
        .collect();
    let shared = SharedRun::new(config, legacy_dirs)?;

    if opts.refresh_augment {
        purge_stage_outputs(
            config,
            output_dir,
            Stage::Augment,
            "refreshing augment outputs",
        )?;
        shared.purge(output_dir, Stage::Augment)?;
    }

    if opts.no_detect {
        validate_no_detect_inputs(config, output_dir, stages)?;
    }
    if opts.no_auto_detect {
        validate_no_auto_detect_inputs(config, output_dir, stages, opts.force)?;
    }

    // Probe video files to populate frame_timing for experiments that need it.
    // Skip probing when only running stages that don't need timing (e.g. pose-only).
    let needs_timing = stages.iter().any(|s| {
        matches!(
            s,
            Stage::Augment | Stage::Behaviour | Stage::Summary | Stage::Inspect
        )
    });
    if needs_timing {
        probe_video_timing_shared(
            scheduler,
            &mut config.experiments,
            base_dir,
            &shared,
            &|video| runner.video_resolution(video),
            &|video| runner.frame_timestamps(video),
        )?;
    }
    let config = &*config;
    let experiment_indices: Vec<usize> = (0..config.experiments.len()).collect();

    let run_detect = stages.contains(&Stage::Detect);

    // Phase 1: Run CPU-only detection (classical CV) in parallel for all experiments.
    // Detection failures are logged but don't abort the batch — GPU fallback follows.
    if run_detect {
        let cpu_failures = std::sync::Mutex::new(Vec::new());
        scheduler.for_each(&experiment_indices, |&i| {
            let experiment = &config.experiments[i];
            let result = shared.execute(
                i,
                Shared::Detect,
                opts.force,
                |dir| is_stage_fresh(Stage::Detect, dir, experiment, config),
                |dir| {
                    tracing::info!(stage = "detect", experiment = %experiment.name, "running (cpu backends)");
                    if let Err(e) = runner.detect(experiment, dir, false, opts.no_auto_detect) {
                        tracing::warn!(experiment = %experiment.name, error = %e, "cpu detection failed");
                        if opts.no_auto_detect
                            && let Ok(mut failures) = cpu_failures.lock()
                        {
                            failures.push(format!("{}: {e}", experiment.name));
                        }
                    }
                    Ok(())
                },
            );
            if let Err(e) = result {
                tracing::warn!(experiment = %experiment.name, error = %e, "shared detect bookkeeping failed");
            }
        });
        let cpu_failures = cpu_failures
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if opts.no_auto_detect && !cpu_failures.is_empty() {
            return Err(BekiperError::Data(format!(
                "--no-auto-detect failed with {} non-automatic detection error(s):\n{}",
                cpu_failures.len(),
                cpu_failures
                    .iter()
                    .take(20)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n")
            )));
        }

        // Phase 2: Run GPU detection sequentially for experiments that still need it
        // (i.e. CPU backends didn't produce a result).
        // Failures are logged per-experiment; the batch continues.
        let mut gpu_failures: Vec<String> = Vec::new();
        for (i, experiment) in config.experiments.iter().enumerate() {
            if shared.legacy(i).pixel_perimeters_toml().exists() {
                continue;
            }
            let result = shared.execute(
                i,
                Shared::Detect,
                opts.force,
                |dir| is_stage_fresh(Stage::Detect, dir, experiment, config),
                |dir| {
                    tracing::info!(stage = "detect", experiment = %experiment.name, "running (gpu backends)");
                    runner.detect(experiment, dir, true, opts.no_auto_detect)
                },
            );
            if let Err(e) = result {
                tracing::error!(experiment = %experiment.name, error = %e, "all detection backends failed");
                gpu_failures.push(experiment.name.clone());
            }
        }
        if !gpu_failures.is_empty() {
            tracing::warn!(
                count = gpu_failures.len(),
                "detection failed for {} experiment(s) — downstream stages will be skipped for these",
                gpu_failures.len(),
            );
        }
    }

    // Phase 3: Run remaining stages.
    // Pose runs sequentially with a shared model (single GPU), then CPU stages run in parallel.
    let remaining_stages: Vec<_> = stages
        .iter()
        .filter(|s| **s != Stage::Detect && **s != Stage::Aggregate)
        .copied()
        .collect();

    if !remaining_stages.is_empty() {
        validate_detect_outputs_before_downstream(config, output_dir, &remaining_stages)?;

        let execution_plan = plan_stage_execution(&remaining_stages, config);

        if !execution_plan.sequential_shared_model.is_empty() {
            let pose_cfg = config
                .experiments
                .iter()
                .find_map(|e| e.pose_model.as_ref());

            let mut shared_model = if let Some(cfg) = pose_cfg {
                tracing::info!(model = %cfg.path.display(), spec = %cfg.spec, "loading pose model...");
                let spec = resolve_model_spec(&cfg.spec)?;
                let model = OnnxPoseModel::load_with_options(
                    &cfg.path,
                    spec,
                    config.gpu.to_pose_options(),
                )?;
                tracing::info!(uses_gpu = model.uses_gpu(), "pose model loaded");
                Some(model)
            } else {
                None
            };

            let total = config.experiments.len();
            for (i, experiment) in config.experiments.iter().enumerate() {
                for stage in &execution_plan.sequential_shared_model {
                    let Some(kind) = Shared::of_stage(*stage) else {
                        return Err(BekiperError::Config(format!(
                            "stage {} cannot run in the shared-model scheduler",
                            stage.name()
                        )));
                    };
                    shared.execute(
                        i,
                        kind,
                        opts.force,
                        |dir| is_stage_fresh(*stage, dir, experiment, config),
                        |dir| {
                            dir.ensure_dir()?;
                            tracing::info!(
                                stage = stage.name(),
                                experiment = %experiment.name,
                                progress = format!("{}/{}", i + 1, total),
                                "running"
                            );
                            match stage {
                                Stage::Pose => runner.pose(experiment, dir, &mut shared_model),
                                _ => runner.augment(experiment, config, dir, &mut shared_model),
                            }
                        },
                    )?;
                }
            }
        }

        if !execution_plan.parallel_cpu.is_empty() {
            scheduler.try_for_each(&experiment_indices, |&i| -> Result<()> {
                let experiment = &config.experiments[i];
                let exp_dir = shared.legacy_dir(i, experiment);
                for stage in &execution_plan.parallel_cpu {
                    if *stage == Stage::Inspect {
                        continue;
                    }
                    let run = |dir: &ExperimentOutputDir| -> Result<()> {
                        dir.ensure_dir()?;
                        tracing::info!(stage = stage.name(), experiment = %experiment.name, "running");
                        match stage {
                            Stage::Augment => runner.augment(experiment, config, dir, &mut None),
                            _ => run_stage(*stage, experiment, config, dir, config_path),
                        }
                    };
                    if let Some(kind) = Shared::of_stage(*stage) {
                        shared.execute(
                            i,
                            kind,
                            opts.force,
                            |dir| is_stage_fresh(*stage, dir, experiment, config),
                            run,
                        )?;
                    } else if opts.force || !is_stage_fresh(*stage, &exp_dir, experiment, config) {
                        run(&exp_dir)?;
                    } else {
                        tracing::info!(stage = stage.name(), experiment = %experiment.name, "skipping (fresh)");
                    }
                }
                Ok(())
            })?;
        }
        // Raster/vector stories have a substantially larger child workspace
        // than a native CPU trial. Drain native workers, then admit one renderer
        // at a time rather than multiplying Matplotlib/FFmpeg by Rayon threads.
        if execution_plan.parallel_cpu.contains(&Stage::Inspect)
            && !opts.no_inspect
            && config.inspect.enabled
        {
            for (i, experiment) in config.experiments.iter().enumerate() {
                let exp_dir = shared.legacy_dir(i, experiment);
                if opts.force || !is_stage_fresh(Stage::Inspect, &exp_dir, experiment, config) {
                    crate::parallel::wait_for_renderer_workspace(
                        config.inspect.workspace_bytes()?,
                        &config.project.runtime,
                    )?;
                    run_inspect(experiment, config, &exp_dir)?;
                }
            }
        }
    }

    if stages.contains(&Stage::Aggregate) {
        let should_run =
            opts.force || config.post_analysis.enabled || !is_aggregate_fresh(config, output_dir);
        if should_run {
            tracing::info!(stage = "aggregate", run_root = %output_dir.display(), "running");
            crate::parallel::wait_for_renderer_workspace(
                config.post_analysis.workspace_bytes()?,
                &config.project.runtime,
            )?;
            run_aggregate(config, output_dir)?;
        } else {
            tracing::info!(stage = "aggregate", run_root = %output_dir.display(), "skipping (fresh)");
        }
    }

    Ok(())
}

/// The heavy per-stage operations, injectable so tests can count executions.
trait StageRunner: Sync {
    fn video_resolution(&self, video: &Path) -> Result<(u32, u32)>;
    fn frame_timestamps(&self, video: &Path) -> Result<FrameTiming>;
    fn detect(
        &self,
        experiment: &ExperimentConfig,
        dir: &ExperimentOutputDir,
        allow_gpu: bool,
        no_auto_detect: bool,
    ) -> Result<()>;
    fn pose(
        &self,
        experiment: &ExperimentConfig,
        dir: &ExperimentOutputDir,
        model: &mut Option<OnnxPoseModel>,
    ) -> Result<()>;
    fn augment(
        &self,
        experiment: &ExperimentConfig,
        config: &IngressConfig,
        dir: &ExperimentOutputDir,
        model: &mut Option<OnnxPoseModel>,
    ) -> Result<()>;
}

struct RealRunner;

impl StageRunner for RealRunner {
    fn video_resolution(&self, video: &Path) -> Result<(u32, u32)> {
        let info = bekiper_pose::video::probe_video_info(video)?;
        Ok((info.width, info.height))
    }

    fn frame_timestamps(&self, video: &Path) -> Result<FrameTiming> {
        bekiper_pose::video::extract_frame_timestamps(video)
    }

    fn detect(
        &self,
        experiment: &ExperimentConfig,
        dir: &ExperimentOutputDir,
        allow_gpu: bool,
        no_auto_detect: bool,
    ) -> Result<()> {
        run_detect_filtered(experiment, dir, allow_gpu, no_auto_detect)
    }

    fn pose(
        &self,
        experiment: &ExperimentConfig,
        dir: &ExperimentOutputDir,
        model: &mut Option<OnnxPoseModel>,
    ) -> Result<()> {
        run_pose_with_model_gpu(experiment, dir, model)
    }

    fn augment(
        &self,
        experiment: &ExperimentConfig,
        config: &IngressConfig,
        dir: &ExperimentOutputDir,
        model: &mut Option<OnnxPoseModel>,
    ) -> Result<()> {
        run_augment_with_model(experiment, config, dir, model)
    }
}
