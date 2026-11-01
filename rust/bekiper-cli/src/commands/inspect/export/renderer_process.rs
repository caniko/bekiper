struct Cancellation {
    flag: Arc<AtomicBool>,
    #[cfg(unix)]
    signals: Vec<signal_hook::SigId>,
}

impl Cancellation {
    fn new() -> Result<Self> {
        let result = Self {
            flag: Arc::new(AtomicBool::new(false)),
            #[cfg(unix)]
            signals: Vec::new(),
        };
        #[cfg(unix)]
        let mut result = result;
        #[cfg(unix)]
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            result.signals.push(signal_hook::flag::register(
                signal,
                Arc::clone(&result.flag),
            )?);
        }
        Ok(result)
    }
}

impl Drop for Cancellation {
    fn drop(&mut self) {
        #[cfg(unix)]
        for signal in self.signals.drain(..) {
            signal_hook::low_level::unregister(signal);
        }
    }
}

fn execute(
    command: &mut Command,
    output: &Path,
    receipt: &mut Value,
    cancelled: &dyn Fn() -> bool,
) -> Result<String> {
    let index = receipt["commands"]
        .as_array()
        .context("export commands")?
        .len();
    let stdout = output.join(format!("logs/{index:06}.stdout.log"));
    let stderr = output.join(format!("logs/{index:06}.stderr.log"));
    receipt["commands"].as_array_mut().context("export commands")?.push(json!({
        "program":command.get_program().to_string_lossy(),
        "args":command.get_args().map(|a| a.to_string_lossy()).collect::<Vec<_>>(),"status":"running"}));
    save(output, receipt)?;
    let result = pipeline::run_child(command, &stdout, &stderr, cancelled);
    let entry = receipt["commands"]
        .as_array_mut()
        .context("export commands")?
        .last_mut()
        .context("export command")?;
    for (name, path) in [("stdout", &stdout), ("stderr", &stderr)] {
        if path.is_file() {
            entry[name] = inputs::fingerprint(path)?;
        }
    }
    entry["status"] = if result.as_ref().is_ok_and(|s| s.success()) {
        "passed"
    } else {
        "failed"
    }
    .into();
    if let Ok(status) = &result {
        entry["exit_status"] = json!(status.code());
    }
    save(output, receipt)?;
    ensure!(
        result?.success(),
        "renderer failed; see {}",
        stderr.display()
    );
    let mut text = String::new();
    fs::File::open(stdout)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_string(&mut text)?;
    ensure!(text.len() <= 16 * 1024 * 1024, "oversized renderer output");
    Ok(text)
}
