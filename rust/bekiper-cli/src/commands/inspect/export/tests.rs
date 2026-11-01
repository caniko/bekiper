use super::*;

fn trial(root: &Path, name: &str, id: &str) {
    let dir = root.join("data").join(name);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join(format!("{id}_inspection.toml")),
        format!("[experiment]\nname = '{id}'\n"),
    )
    .unwrap();
    fs::write(dir.join(format!("{id}_evaluated.parquet")), b"table").unwrap();
}

#[test]
fn discovery_preserves_animal_paths_and_rejects_missing_or_ambiguous_inputs() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("data")).unwrap();
    assert!(discover(&dir.path().join("data"), &[]).is_err());
    trial(dir.path(), "721/training", "721_training");
    let found = discover(&dir.path().join("data"), &[]).unwrap();
    assert_eq!(found[0].name, "721/training");
    assert!(discover(&dir.path().join("data"), &["721/missing".into()]).is_err());
    fs::write(
        found[0]
            .manifest
            .parent()
            .unwrap()
            .join("other_inspection.toml"),
        "",
    )
    .unwrap();
    assert!(discover(&dir.path().join("data"), &[]).is_err());
}

#[test]
fn filename_cannot_relabel_recorded_experiment() {
    let dir = tempfile::tempdir().unwrap();
    trial(dir.path(), "one", "one");
    fs::write(
        dir.path().join("data/one/one_inspection.toml"),
        "[experiment]\nname = 'another'\n",
    )
    .unwrap();
    assert!(discover(&dir.path().join("data"), &[]).is_err());
}

#[cfg(unix)]
#[test]
fn discovery_does_not_follow_symlinks_outside_source_tree() {
    let dir = tempfile::tempdir().unwrap();
    trial(dir.path(), "one", "one");
    std::os::unix::fs::symlink(dir.path().join("data/one"), dir.path().join("data/foreign"))
        .unwrap();
    assert!(discover(&dir.path().join("data"), &[]).is_err());
}

fn png(path: &Path, dpi: u32) {
    let mut encoder = png::Encoder::new(fs::File::create(path).unwrap(), 1, 1);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_pixel_dims(Some(png::PixelDimensions {
        xppu: (f64::from(dpi) / 0.0254).round() as u32,
        yppu: (f64::from(dpi) / 0.0254).round() as u32,
        unit: png::Unit::Meter,
    }));
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&[0, 0, 0])
        .unwrap();
}

#[test]
fn zero_exit_or_stale_file_cannot_hide_omitted_plot_and_dpi_is_decoded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("summary.png");
    let expected = BTreeSet::from(["summary".into()]);
    png(&path, 300);
    assert!(verify_figures(&expected, "All figures written", dir.path(), 300).is_err());
    let saved = format!("Saved: {}\n", path.display());
    assert_eq!(
        verify_figures(&expected, &saved, dir.path(), 300).unwrap(),
        std::slice::from_ref(&path)
    );
    assert!(verify_figures(&expected, &saved, dir.path(), 150).is_err());
    fs::write(&path, b"not a decoded PNG").unwrap();
    assert!(verify_figures(&expected, &saved, dir.path(), 300).is_err());
}

fn population(path: &Path, rows: Value) {
    let admitted = rows
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["disposition"] == "admitted")
        .count();
    let blocked = rows
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["disposition"] == "blocked")
        .count();
    let count = rows.as_array().unwrap().len();
    let mut ledger = json!({"schema_version":2,"config":{"path":"synthetic.toml","sha256":"fixture"},
        "data_root":"synthetic","pre_filter_analyses":count,"admitted_analyses":admitted,
        "excluded_analyses":count-admitted-blocked,"blocked_analyses":blocked,
        "pre_filter_recordings":count,"admitted_recordings":admitted,"rows":rows});
    let digest_input = path.with_extension("digest.toml");
    fs::write(
        &digest_input,
        bekiper_core::handoff::to_string(&ledger).unwrap(),
    )
    .unwrap();
    ledger["fingerprint"] = inputs::fingerprint(&digest_input).unwrap()["sha256"].clone();
    fs::write(path, bekiper_core::handoff::to_string(&ledger).unwrap()).unwrap();
}

#[test]
fn population_gate_keeps_exclusions_but_rejects_missing_foreign_and_blocked_trials() {
    let dir = tempfile::tempdir().unwrap();
    trial(dir.path(), "one", "one");
    let found = discover(&dir.path().join("data"), &[]).unwrap();
    let path = dir.path().join("population.toml");
    let row = |id: &str, disposition: &str| {
        json!({"analysis_id":id,"recording_key":id,"task_type":"open_field",
        "pose_sources":[],"disposition":disposition,"reason":"explicit synthetic test exclusion"})
    };
    population(
        &path,
        json!([row("one", "admitted"), row("excluded", "excluded")]),
    );
    let receipt = coverage(&path, &found).unwrap();
    assert_eq!(receipt["recordings"], 2);
    assert_eq!(receipt["rows"][1]["trial_exported"], false);
    population(
        &path,
        json!([row("one", "admitted"), row("missing", "admitted")]),
    );
    assert!(
        coverage(&path, &found)
            .unwrap_err()
            .to_string()
            .contains("missing admitted")
    );
    population(&path, json!([row("foreign", "admitted")]));
    assert!(coverage(&path, &found).is_err());
    population(&path, json!([row("one", "blocked")]));
    assert!(coverage(&path, &found).is_err());
}

#[cfg(unix)]
fn renderer(dir: &Path, mutate: bool, omit: bool) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let image = dir.join("source.png");
    png(&image, 300);
    let path = dir.join("renderer");
    let identity = inputs::fingerprint(&image).unwrap();
    let renderer_receipt = json!({"status":"complete","effective_theme":{"min_dpi":300,"fig_format":".png"},
        "inputs":[{"path":image,"bytes":identity["bytes"],"sha256":identity["sha256"]}],
        "artifacts":[{"path":"$output/summary.png","bytes":identity["bytes"],"sha256":identity["sha256"]}]});
    let script = format!(
        "#!/bin/sh\ntrial=$2\nfor arg do\n  if [ \"$arg\" = --dry-run ]; then echo '  summary'; exit 0; fi\ndone\nwhile [ \"$1\" != --output ]; do shift; done\noutput=$2\n{}\n{}\n",
        if omit {
            String::new()
        } else {
            format!(
                "cp '{}' \"$output/summary.png\"\necho \"Saved: $output/summary.png\"\ncat > \"$output/render_receipt.json\" <<EOF\n{}\nEOF",
                image.display(),
                renderer_receipt
            )
        },
        if mutate {
            "printf changed > \"$trial/one_evaluated.parquet\""
        } else {
            ""
        }
    );
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn args(root: &Path) -> ExportArgs {
    let config = root.join("config.toml");
    fs::write(
        &config,
        "[inspect.plot]\nfig_format = '.png'\nmin_dpi = 300\n",
    )
    .unwrap();
    ExportArgs {
        run_root: root.to_owned(),
        config,
        output: root.join("exports"),
        population: None,
        trial: Vec::new(),
        video_trial: Vec::new(),
    }
}

#[cfg(not(feature = "full"))]
#[test]
fn minimal_export_rejects_video_before_creating_output() {
    let dir = tempfile::tempdir().unwrap();
    trial(dir.path(), "one", "one");
    let mut options = args(dir.path());
    options.video_trial = vec!["one".into()];
    let error = render_with(options, "unused-renderer".as_ref()).unwrap_err();
    assert!(error.to_string().contains("full-featured"));
    assert!(!dir.path().join("exports").exists());
}

#[cfg(feature = "full")]
#[test]
fn duplicate_or_unselected_video_trials_fail_before_creating_output() {
    for videos in [vec!["one", "one"], vec!["missing"]] {
        let dir = tempfile::tempdir().unwrap();
        trial(dir.path(), "one", "one");
        let mut options = args(dir.path());
        options.video_trial = videos.into_iter().map(str::to_owned).collect();
        let error = render_with(options, "unused-renderer".as_ref()).unwrap_err();
        assert!(error.to_string().contains("unique and selected"));
        assert!(!dir.path().join("exports").exists());
    }
}

#[cfg(unix)]
#[test]
fn native_export_retains_config_and_inputs_and_refuses_occupied_output() {
    let dir = tempfile::tempdir().unwrap();
    trial(dir.path(), "721/training", "one");
    let renderer = renderer(dir.path(), false, false);
    let before =
        inputs::fingerprint(&dir.path().join("data/721/training/one_evaluated.parquet")).unwrap();
    let receipt = render_with(args(dir.path()), renderer.as_os_str()).unwrap();
    assert_eq!(receipt["status"], "completed");
    assert_eq!(receipt["trials"][0]["trial"], "721/training");
    assert_eq!(receipt["coverage_scope"], "discovered_trials_only");
    assert_eq!(receipt["files"].as_array().unwrap().len(), 1);
    assert_eq!(
        verify(&dir.path().join("exports")).unwrap()["artifact_status"],
        "verified"
    );
    assert_eq!(
        before,
        inputs::fingerprint(&dir.path().join("data/721/training/one_evaluated.parquet")).unwrap()
    );
    assert_eq!(
        fs::read(dir.path().join("exports/bekiper_config.toml")).unwrap(),
        fs::read(dir.path().join("config.toml")).unwrap()
    );
    assert!(render_with(args(dir.path()), renderer.as_os_str()).is_err());
    fs::write(
        dir.path()
            .join("exports/trials/721/training/figures/summary.png"),
        b"tampered",
    )
    .unwrap();
    assert!(verify(&dir.path().join("exports")).is_err());
}

#[cfg(unix)]
#[test]
fn omitted_plot_or_mutated_source_retains_failed_receipt() {
    for (mutate, omit) in [(false, true), (true, false)] {
        let dir = tempfile::tempdir().unwrap();
        trial(dir.path(), "one", "one");
        let renderer = renderer(dir.path(), mutate, omit);
        assert!(render_with(args(dir.path()), renderer.as_os_str()).is_err());
        let receipt = inputs::load_json(&dir.path().join("exports/plot_manifest.json")).unwrap();
        assert_eq!(receipt["status"], "failed");
        assert!(receipt["error"].as_str().unwrap().contains(if mutate {
            "inputs changed"
        } else {
            "not rendered"
        }));
    }
}

#[cfg(unix)]
#[test]
fn population_bound_export_verifies_exclusions_and_rejects_changed_cached_frame() {
    let dir = tempfile::tempdir().unwrap();
    trial(dir.path(), "one", "one");
    let renderer = renderer(dir.path(), false, false);
    let pop = dir.path().join("population.toml");
    population(
        &pop,
        json!([
            {"analysis_id":"one","recording_key":"one","task_type":"open_field","pose_sources":[],"disposition":"admitted"},
            {"analysis_id":"excluded","recording_key":"excluded","task_type":"open_field","pose_sources":[],"disposition":"excluded","reason":"authorized synthetic exclusion"}
        ]),
    );
    let mut options = args(dir.path());
    options.population = Some(pop);
    render_with(options, renderer.as_os_str()).unwrap();
    let receipt = verify(&dir.path().join("exports")).unwrap();
    assert_eq!(receipt["coverage_scope"], "pre_filter_population");
    assert_eq!(receipt["coverage"]["recordings"], 2);
    assert_eq!(receipt["coverage"]["excluded_analyses"], 1);
    fs::write(dir.path().join("source.png"), b"changed cached frame").unwrap();
    assert!(
        verify(&dir.path().join("exports"))
            .unwrap_err()
            .to_string()
            .contains("renderer source/artifact bytes changed")
    );
}

#[cfg(target_os = "linux")]
#[test]
fn archive_cleanup_excludes_trial_export_before_rendering() {
    let dir = tempfile::tempdir().unwrap();
    trial(dir.path(), "one", "one");
    let renderer = renderer(dir.path(), false, false);
    let _archive = pipeline::ProjectDataLease::archive(dir.path()).unwrap();
    assert!(
        render_with(args(dir.path()), renderer.as_os_str())
            .unwrap_err()
            .to_string()
            .contains("project data is in use")
    );
    assert!(!dir.path().join("exports").exists());
}

#[cfg(target_os = "linux")]
#[test]
fn cancelled_command_is_reaped_and_records_failure_while_export_lease_is_held() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("logs")).unwrap();
    let lease = pipeline::ProjectDataLease::export(dir.path()).unwrap();
    let mut receipt = json!({"commands": []});
    let stdout = dir.path().join("logs/000000.stdout.log");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let result = execute(
        Command::new("sh").args(["-c", "echo $$; exec sleep 60"]),
        dir.path(),
        &mut receipt,
        &|| {
            assert!(pipeline::ProjectDataLease::archive(dir.path()).is_err());
            assert!(
                std::time::Instant::now() < deadline,
                "renderer did not report its PID"
            );
            fs::read_to_string(&stdout).is_ok_and(|pid| !pid.trim().is_empty())
        },
    );
    assert!(result.unwrap_err().to_string().contains("cancelled"));
    let pid = fs::read_to_string(stdout).unwrap();
    assert!(!Path::new(&format!("/proc/{}", pid.trim())).exists());
    let retained = inputs::load_json(&dir.path().join("plot_manifest.json")).unwrap();
    assert_eq!(retained, receipt);
    assert_eq!(retained["commands"][0]["status"], "failed");
    for stream in ["stdout", "stderr"] {
        assert_eq!(
            retained["commands"][0][stream],
            inputs::fingerprint(&dir.path().join(format!("logs/000000.{stream}.log"))).unwrap()
        );
    }
    drop(lease);
    assert!(pipeline::ProjectDataLease::archive(dir.path()).is_ok());
}
