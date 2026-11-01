struct PreparedExport {
    data: PathBuf,
    config: PathBuf,
    config_identity: Value,
    dpi: u64,
    trials: Vec<Trial>,
    videos: BTreeSet<String>,
    population_path: Option<PathBuf>,
    population_identity: Option<Value>,
    population: Option<Value>,
    output: PathBuf,
}

fn prepare_export(args: &ExportArgs, root: &Path) -> Result<PreparedExport> {
    let data = root.join("data").canonicalize()?;
    ensure!(data.starts_with(root), "data directory escapes run root");
    let config = args.config.canonicalize()?;
    let config_identity = inputs::fingerprint(&config)?;
    let settings = read_toml(&config)?;
    let dpi = settings["inspect"]["plot"]["min_dpi"]
        .as_u64()
        .filter(|v| *v > 0)
        .context("captured inspect.plot.min_dpi is required")?;
    ensure!(
        matches!(
            settings["inspect"]["plot"]["fig_format"].as_str(),
            Some(".png" | "png")
        ),
        "captured inspect.plot.fig_format must be PNG"
    );
    let trials = discover(&data, &args.trial)?;
    let videos: BTreeSet<_> = args.video_trial.iter().cloned().collect();
    ensure!(
        videos.len() == args.video_trial.len()
            && videos.iter().all(|n| trials.iter().any(|t| &t.name == n)),
        "video trials must be unique and selected"
    );
    let population_path = args
        .population
        .as_ref()
        .map(|path| path.canonicalize())
        .transpose()?;
    let population_identity = population_path
        .as_ref()
        .map(|p| inputs::fingerprint(p))
        .transpose()?;
    let population = population_path
        .as_ref()
        .map(|p| coverage(p, &trials))
        .transpose()?;
    let parent = args
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let output = parent.join(args.output.file_name().context("export destination")?);
    ensure!(
        !root.starts_with(&output) && !output.starts_with(&data) && !config.starts_with(&output),
        "export destination overlaps source inputs"
    );
    if let Some(path) = &population_path {
        ensure!(
            !path.canonicalize()?.starts_with(&output),
            "export destination overlaps population"
        );
    }
    Ok(PreparedExport {
        data,
        config,
        config_identity,
        dpi,
        trials,
        videos,
        population_path,
        population_identity,
        population,
        output,
    })
}

impl PreparedExport {
    fn verify_inputs(&self, selected: &[String], captured: &Path, receipt: &Value) -> Result<()> {
        unchanged(&self.trials)?;
        for record in receipt["trials"].as_array().context("rendered trials")? {
            verify_trial_renderer(&self.output, record)?;
        }
        let after = discover(&self.data, selected)?;
        ensure!(
            after
                .iter()
                .map(|t| (&t.name, &t.analysis_id))
                .collect::<Vec<_>>()
                == self
                    .trials
                    .iter()
                    .map(|t| (&t.name, &t.analysis_id))
                    .collect::<Vec<_>>(),
            "inspection population changed during export"
        );
        ensure!(
            inputs::fingerprint(&self.config)? == self.config_identity
                && inputs::fingerprint(captured)?["sha256"] == self.config_identity["sha256"],
            "captured plot config changed during export"
        );
        if let Some(path) = &self.population_path {
            ensure!(
                Some(inputs::fingerprint(path)?) == self.population_identity
                    && Some(coverage(path, &self.trials)?) == self.population,
                "population changed during export"
            );
        }
        Ok(())
    }
}
