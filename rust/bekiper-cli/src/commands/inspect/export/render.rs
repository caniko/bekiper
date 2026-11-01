fn render_with(args: ExportArgs, renderer: &std::ffi::OsStr) -> Result<Value> {
    #[cfg(not(feature = "full"))]
    ensure!(
        args.video_trial.is_empty(),
        "video trial exports require a full-featured bekiper executable"
    );
    let root = args.run_root.canonicalize()?;
    let _source_lease = pipeline::ProjectDataLease::export(&root)?;
    let prepared = prepare_export(&args, &root)?;
    let data = &prepared.data;
    let config = &prepared.config;
    let config_identity = &prepared.config_identity;
    let dpi = prepared.dpi;
    let trials = &prepared.trials;
    let output = &prepared.output;
    fs::create_dir(output)
        .context("trial export requires a fresh output directory outside run/data")?;
    let mut receipt = json!({"schema_version":2,"scope":"trial_inspections","status":"rendering","run_root":root,
        "config_source":config,"config":config_identity,"population_source":prepared.population_path,"population_identity":prepared.population_identity,
        "coverage":prepared.population,"coverage_scope":if args.population.is_some() { "pre_filter_population" } else if args.trial.is_empty() { "discovered_trials_only" } else { "selected_trials_only" },
        "driver":inputs::fingerprint(&std::env::current_exe()?)?,"scientific_readiness":"unverified","analysis_admitted":false,
        "commands":[],"trials":[],"files":[]});
    save(output, &receipt)?;
    let result = (|| -> Result<()> {
        fs::create_dir(output.join("logs"))?;
        let captured = output.join("bekiper_config.toml");
        fs::copy(config, &captured)?;
        ensure!(
            inputs::fingerprint(&captured)?["sha256"] == config_identity["sha256"],
            "plot config changed during copy"
        );
        let cancellation = Cancellation::new()?;
        let cancelled = || cancellation.flag.load(Ordering::Relaxed);
        for trial in trials {
            ensure!(!cancelled(), "trial export cancelled");
            let destination = output.join("trials").join(&trial.name);
            let figures = destination.join("figures");
            fs::create_dir_all(&figures)?;
            let plot = |dry_run: bool| {
                let mut command = Command::new(renderer);
                command
                    .arg("plot")
                    .arg(data.join(&trial.name))
                    .arg("--config")
                    .arg(&captured)
                    .arg("--output")
                    .arg(&figures)
                    .arg("--dpi")
                    .arg(dpi.to_string())
                    .args(["--format", ".png"]);
                // Replay the captured presentation rather than later ambient plot overrides.
                for (key, _) in std::env::vars_os()
                    .filter(|(key, _)| key.to_string_lossy().starts_with("BEKIPER_PLOT_"))
                {
                    command.env_remove(key);
                }
                if dry_run {
                    command.arg("--dry-run");
                }
                command
            };
            let expected =
                planned_stems(&execute(&mut plot(true), output, &mut receipt, &cancelled)?);
            ensure!(
                !expected.is_empty(),
                "renderer declared no trial plots: {}",
                trial.name
            );
            let stdout = execute(&mut plot(false), output, &mut receipt, &cancelled)?;
            let mut paths = verify_figures(&expected, &stdout, &figures, dpi)?;
            let renderer_receipt = figures.join("render_receipt.json");
            let renderer_identity = verify_renderer_receipt(&renderer_receipt, &paths, dpi)?;
            if prepared.videos.contains(&trial.name) {
                let features = read_toml(&trial.manifest)?;
                let video_dir = destination.join("videos");
                fs::create_dir(&video_dir)?;
                let mut selectors = vec![None];
                for family in ["olfaction", "whisker_interaction"] {
                    if features["feature_inspections"].as_array().is_some_and(|f| {
                        f.iter()
                            .any(|v| v["label"] == format!("multiobject_{family}"))
                    }) {
                        selectors.push(Some(family));
                    }
                }
                for selector in selectors {
                    let path = video_dir
                        .join(format!("{}.inspection.mp4", selector.unwrap_or("overview")));
                    let mut command = Command::new(std::env::current_exe()?);
                    command
                        .args(["inspect", "video"])
                        .arg(trial.manifest.parent().context("trial parent")?)
                        .arg("--config")
                        .arg(&captured)
                        .arg("--output")
                        .arg(&path);
                    if let Some(selector) = selector {
                        command.args(["--heuristic", selector]);
                    }
                    execute(&mut command, output, &mut receipt, &cancelled)?;
                    ensure!(
                        fs::metadata(&path)?.len() > 0,
                        "missing or empty video output"
                    );
                    paths.push(path);
                }
            }
            unchanged(std::slice::from_ref(trial))?;
            receipt["trials"].as_array_mut().context("trial receipt")?.push(json!({"trial":trial.name,"analysis_id":trial.analysis_id,
                "manifest":trial.manifest,"manifest_identity":trial.manifest_identity,"evaluated_table":trial.table,"evaluated_identity":trial.table_identity,"plot_stems":expected,"dpi":dpi,
                "renderer_receipt":renderer_receipt.strip_prefix(output)?,"renderer_identity":renderer_identity}));
            for path in paths {
                receipt["files"].as_array_mut().context("export files")?.push(json!({"path":path.strip_prefix(output)?,"identity":inputs::fingerprint(&path)?}));
            }
            save(output, &receipt)?;
        }
        prepared.verify_inputs(&args.trial, &captured, &receipt)?;
        ensure!(!cancelled(), "trial export cancelled");
        Ok(())
    })();
    receipt["status"] = if result.is_ok() {
        "completed"
    } else {
        "failed"
    }
    .into();
    if let Err(error) = &result {
        receipt["error"] = format!("{error:#}").into();
    }
    save(output, &receipt)?;
    result?;
    println!(
        "Verified {} trial exports: {}",
        trials.len(),
        output.join("plot_manifest.json").display()
    );
    Ok(receipt)
}
