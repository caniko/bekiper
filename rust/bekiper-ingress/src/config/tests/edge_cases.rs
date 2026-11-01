fn edge_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bekiper_config_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn project_config(body: &str) -> IngressConfig {
    toml::from_str(&format!(
        "[project]\nproject_name = \"edge\"\ndata_directory = \".\"\n{body}"
    ))
    .unwrap()
}

#[test]
fn detection_overrides_replace_each_recipe_field() {
    use bekiper_detect::config::{AreaFilterConfig, MouseRemovalConfig, ThresholdConfig};
    let overrides = DetectionPipelineOverrides {
        expected_count: Some(3),
        threshold: Some(ThresholdConfig::default()),
        mouse_removal: Some(MouseRemovalConfig::default()),
        area_filter: Some(AreaFilterConfig::default()),
    };
    let base = bekiper_detect::recipe::recipe_to_config(DetectionRecipe::NortTraining);
    let applied = overrides.apply(base.clone());
    assert_eq!(applied.expected_count, 3);
    assert_eq!(applied.threshold, overrides.threshold.clone().unwrap());
    assert_eq!(
        applied.mouse_removal,
        overrides.mouse_removal.clone().unwrap()
    );
    assert_eq!(applied.area_filter, overrides.area_filter.clone().unwrap());
    let unchanged = DetectionPipelineOverrides::default().apply(base.clone());
    assert_eq!(unchanged.expected_count, base.expected_count);
    assert_eq!(unchanged.threshold, base.threshold);
    for (body, message) in [
        ("backend = 5", "a detection backend string or array"),
        ("backend = []", "backend list must not be empty"),
    ] {
        let error = toml::from_str::<DetectionConfig>(body)
            .unwrap_err()
            .to_string();
        assert!(error.contains(message), "{body}: {error}");
    }
}

#[test]
fn animal_csvs_skip_short_and_non_numeric_rows_and_report_bad_bytes() {
    let dir = edge_dir("animal_csvs");
    let sequence = dir.join("sequence.csv");
    std::fs::write(&sequence, "Animal,Sequence,Mpp,Center\n1,2\n3,0,m,c\n").unwrap();
    let rows = parse_animal_sequence_csv(&sequence).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].enclosure_id, "");
    std::fs::write(&sequence, b"Animal,Sequence,Mpp,Center\n1,0,\xff,c\n").unwrap();
    assert!(parse_animal_sequence_csv(&sequence).is_err());
    assert!(parse_animal_sequence_csv(&dir.join("missing.csv")).is_err());

    let metadata = dir.join("metadata.csv");
    std::fs::write(
        &metadata,
        "Animal,Cohort,Gene,Sex,Age\n4, c1 , wt , f , 12w \nabc,c,g,s,a\n5,c,g\n",
    )
    .unwrap();
    let parsed = parse_animal_metadata_csv(&metadata).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[&4].cohort, "c1");
    assert_eq!(parsed[&4].age, "12w");
    std::fs::write(&metadata, b"Animal,Cohort,Gene,Sex,Age\n4,\xff,g,s,a\n").unwrap();
    assert!(parse_animal_metadata_csv(&metadata).is_err());
    assert!(parse_animal_metadata_csv(&dir.join("missing.csv")).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn inspect_settings_capture_typed_environment_overlays() {
    let env: HashMap<&str, &str> = HashMap::from([
        ("BEKIPER_INSPECT_DISABLED", "true"),
        ("BEKIPER_INSPECT_VIDEO_TRIALS", " a, ,b "),
        ("BEKIPER_PLOT_FFMPEG_PRESET", "fast"),
        ("BEKIPER_PLOT_LINEWIDTH", "2.5"),
        ("BEKIPER_PLOT_QUIVER_ALPHA", "nan"),
        ("BEKIPER_PLOT_PALETTE", "muted"),
        ("BEKIPER_PLOT_COLOR_MODE", "light"),
        ("BEKIPER_PLOT_DPI", "150"),
        ("BEKIPER_PLOT_FIG_FORMAT", ".svg"),
        ("BEKIPER_PLOT_DENSITY_BINS", "64"),
        ("BEKIPER_PLOT_SUMMARY_FIGSIZE", "[4, 3]"),
        ("BEKIPER_PLOT_FONTSIZE", "not toml ["),
        ("BEKIPER_PLOT_QUIVER_BINARY_CMAPS", "Reds, Blues"),
    ]);
    let mut inspect = InspectConfig::default();
    inspect.apply_env_overrides_from(env_lookup(&env));
    assert!(!inspect.enabled);
    assert_eq!(inspect.video_trials, ["a", "b"]);
    assert_eq!(inspect.plot["ffmpeg_preset"].as_str(), Some("fast"));
    assert_eq!(inspect.plot["linewidth"].as_float(), Some(2.5));
    assert!(
        !inspect.plot.contains_key("quiver_alpha"),
        "non-finite floats are ignored"
    );
    assert_eq!(inspect.plot["palette"].as_str(), Some("muted"));
    assert_eq!(inspect.plot["color_mode"].as_str(), Some("light"));
    assert_eq!(inspect.plot["min_dpi"].as_integer(), Some(150));
    assert_eq!(inspect.plot["fig_format"].as_str(), Some(".svg"));
    assert_eq!(inspect.plot["density_bins"].as_integer(), Some(64));
    assert_eq!(inspect.plot["summary_figsize"].as_array().unwrap().len(), 2);
    assert_eq!(
        inspect.plot["quiver_binary_cmaps"].as_array().unwrap(),
        &[toml::Value::from("Reds"), toml::Value::from("Blues")]
    );
    assert!(
        !inspect.plot.contains_key("fontsize"),
        "invalid TOML values are ignored"
    );
    assert_eq!(
        inspect.workspace_bytes().unwrap(),
        0,
        "disabled rendering reserves nothing"
    );
    inspect.validate_video_selection(["x"].into_iter()).unwrap();

    let mut enabled = InspectConfig::default();
    let env = HashMap::from([
        ("BEKIPER_INSPECT_DISABLED", "0"),
        ("BEKIPER_PLOT_DPI", "-1"),
        ("BEKIPER_INSPECT_VIDEO_TRIALS", "a"),
    ]);
    enabled.enabled = false;
    enabled.apply_env_overrides_from(env_lookup(&env));
    assert!(enabled.enabled);
    assert!(
        !enabled.plot.contains_key("min_dpi"),
        "nonpositive DPI is ignored"
    );
    enabled
        .validate_video_selection(["a", "b"].into_iter())
        .unwrap();
    assert!(
        enabled
            .validate_video_selection(["b"].into_iter())
            .unwrap_err()
            .to_string()
            .contains("no analysis in this run: a")
    );
    let mut unknown = InspectConfig::default();
    unknown.apply_env_overrides_from(|key| {
        (key == "BEKIPER_INSPECT_DISABLED").then(|| "maybe".into())
    });
    assert!(unknown.enabled);

    // Nested profile DPIs count; an absurd DPI overflows the reservation.
    let mut nested = InspectConfig {
        plot: toml::from_str("[profiles.print]\nmin_dpi = 600\nlabel = 'x'").unwrap(),
        ..Default::default()
    };
    assert!(
        nested.workspace_bytes().unwrap() > InspectConfig::default().workspace_bytes().unwrap()
    );
    nested.plot = toml::from_str(&format!("min_dpi = {}", i64::MAX)).unwrap();
    assert!(
        nested
            .workspace_bytes()
            .unwrap_err()
            .to_string()
            .contains("overflow")
    );
}

#[test]
fn measurement_overrides_reach_every_template_and_reject_invalid_values() {
    for (key, value, message) in [
        ("BEKIPER_OFT_WINDOW_SECONDS", "-1", "finite and nonnegative"),
        (
            "BEKIPER_OFT_WINDOW_SECONDS",
            "NaN",
            "finite and nonnegative",
        ),
        (
            "BEKIPER_NORT_STRICT_BOUNDARIES",
            "maybe",
            "must be true or false",
        ),
    ] {
        let error = MeasurementOverrides::default()
            .with_env(|k| (k == key).then(|| value.to_string()))
            .unwrap_err()
            .to_string();
        assert!(error.contains(message), "{key}={value}: {error}");
    }
    let resolved = MeasurementOverrides::default()
        .with_env(|k| (k == "BEKIPER_NORT_STRICT_BOUNDARIES").then(|| "0".to_string()))
        .unwrap();
    assert_eq!(resolved.nort_strict_boundaries, Some(false));
    let mut config = project_config(
        r#"
        [[experiments]]
        name = "explicit"
        task_type = "open_field"
        meters_per_pixel = 0.001
        [[scan]]
        video_dir = "videos"
        task_type = "open_field"
        meters_per_pixel = 0.001
        [[scan.variants]]
        name = "variant"
        [[animal_scan]]
        data_dir = "dlc"
        task_type = "open_field"
        trial_sequence = ["oft"]
        animal_sequence_csv = "sequence.csv"
        "#,
    );
    let mut untouched = config.clone();
    untouched.apply_measurement_overrides(MeasurementOverrides::default());
    assert!(untouched.experiments[0].task_params.is_none());
    config.apply_measurement_overrides(MeasurementOverrides {
        oft_window_seconds: Some(120.0),
        nort_strict_boundaries: None,
    });
    let window = |params: &Option<TaskParamsOverrides>| params.as_ref().unwrap().oft_window_seconds;
    assert_eq!(window(&config.experiments[0].task_params), Some(120.0));
    assert_eq!(window(&config.scan[0].task_params), Some(120.0));
    assert_eq!(
        config.scan[0].variants[0].task_params.oft_window_seconds,
        Some(120.0)
    );
    assert_eq!(window(&config.animal_scan[0].task_params), Some(120.0));
}

#[test]
fn temporal_policies_validate_sequence_profiles_and_reach_animal_scans() {
    let mut config = project_config(
        r#"
        [[animal_scan]]
        data_dir = "dlc"
        task_type = "y_maze"
        trial_sequence = ["ymaze"]
        animal_sequence_csv = "sequence.csv"
        [[ingress_profile]]
        profile = "sequencetable"
        [ingress_profile.task_params]
        oft_window_seconds = -1.0
        "#,
    );
    assert!(
        config
            .validate_ymaze_temporal_policies()
            .unwrap_err()
            .to_string()
            .contains("finite and nonnegative")
    );
    config.ingress_profile[0].task_params = None;
    config.validate_ymaze_temporal_policies().unwrap();
    config
        .apply_ymaze_dwell_overrides(bekiper_feature::tolerance::DwellOverrides {
            seconds: Some(2.0),
            ..Default::default()
        })
        .unwrap();
    let policy = config.animal_scan[0]
        .task_params
        .as_ref()
        .and_then(|p| p.ymaze.as_ref())
        .and_then(|y| y.temporal_policy)
        .unwrap();
    assert_eq!(policy.min_duration_seconds, 2.0);
}

#[test]
fn fps_from_video_needs_probed_timing() {
    let mut experiment: ExperimentConfig =
        toml::from_str("name = 'probe'\ntask_type = 'open_field'\nmeters_per_pixel = 0.001")
            .unwrap();
    assert!(matches!(experiment.fps, FpsConfig::FromVideo));
    assert!(
        experiment
            .effective_fps()
            .unwrap_err()
            .to_string()
            .contains("no video has been probed")
    );
    experiment.frame_timing = Some(bekiper_core::video::FrameTiming::cfr(25.0));
    assert_eq!(experiment.effective_fps().unwrap(), 25.0);
}

#[test]
fn pose_label_models_describe_their_labels_and_parse_environment_flags() {
    use super::pipeline::PoseLabelModel;
    assert_eq!(
        PoseLabelModel::ImenDlc.required_source_labels_for_task(TaskType::Cheeseboard),
        ["nose", "left_ear", "right_ear"]
    );
    assert!(
        PoseLabelModel::SuperanimalTopviewmouse
            .derived_labels()
            .is_empty()
    );
    assert_eq!(
        PoseLabelModel::ImenDlc
            .midpoint_groups_for_task(TaskType::Cheeseboard)
            .len(),
        1
    );
    let labels = vec!["nose".to_string()];
    assert!(
        !PoseLabelModel::ImenDlc
            .missing_source_labels(&labels)
            .is_empty()
    );
    assert_eq!(PoseLabelModel::infer_from_labels(&labels), None);
    let mut pipeline = PipelineConfig::default();
    assert!(pipeline.has_model_skeleton_repair());
    let env = HashMap::from([
        ("BEKIPER_PIPELINE_POSE_LABEL_MODEL", "unknown"),
        ("BEKIPER_PIPELINE_INVERT_Y", "off"),
    ]);
    pipeline.invert_y = true;
    crate::traced(|| pipeline.apply_env_overrides_from(env_lookup(&env)));
    assert!(!pipeline.invert_y);
    assert_eq!(
        pipeline.pose_label_model,
        PipelineConfig::default().pose_label_model
    );
}

#[test]
fn process_environment_overlays_match_an_explicit_lookup() {
    let mut process = project_config("");
    let mut explicit = process.clone();
    process.apply_env_overrides();
    explicit.apply_env_overrides_from(|k| std::env::var(k).ok());
    assert_eq!(process.aggregate, explicit.aggregate);
    assert_eq!(process.post_analysis, explicit.post_analysis);
    assert_eq!(process.inspect, explicit.inspect);
    let mut aggregate = AggregateConfig::default();
    aggregate.apply_env_overrides();
    assert_eq!(aggregate, explicit.aggregate);
    let mut post = PostAnalysisConfig::default();
    post.apply_env_overrides();
    assert_eq!(post, explicit.post_analysis);
    let mut pipeline = PipelineConfig::default();
    pipeline.apply_env_overrides();
    assert_eq!(pipeline.invert_y, explicit.pipeline.invert_y);
    assert_eq!(normalize_spreadsheet_filename("  "), "results.ods");
}

#[test]
fn post_analysis_environment_logs_and_ignores_invalid_presentation_values() {
    let mut post = PostAnalysisConfig::default();
    let empty = HashMap::from([
        ("BEKIPER_POST_ANALYSIS_DISABLED", ""),
        ("BEKIPER_POST_ANALYSIS_PLAN_PATH", " "),
        ("BEKIPER_POST_ANALYSIS_RUN_ID", " "),
        ("BEKIPER_POST_ANALYSIS_FIGURES_ENABLED", ""),
        ("BEKIPER_POST_ANALYSIS_FIGURES_DPI", "x"),
        ("BEKIPER_POST_ANALYSIS_FIGURES_FORMATS", " , "),
        ("BEKIPER_POST_ANALYSIS_REPORT_ENABLED", ""),
        ("BEKIPER_POST_ANALYSIS_REPORT_TITLE", " "),
        ("BEKIPER_POST_ANALYSIS_REPORT_COMPILER", " "),
    ]);
    let before = post.clone();
    crate::traced(|| post.apply_env_overrides_from(env_lookup(&empty)));
    assert_eq!(post, before);
    let invalid = HashMap::from([
        ("BEKIPER_POST_ANALYSIS_DISABLED", "maybe"),
        ("BEKIPER_POST_ANALYSIS_FIGURES_ENABLED", "maybe"),
        ("BEKIPER_POST_ANALYSIS_REPORT_ENABLED", "maybe"),
    ]);
    crate::traced(|| post.apply_env_overrides_from(env_lookup(&invalid)));
    assert_eq!(post, before);
    let enabled = HashMap::from([
        ("BEKIPER_POST_ANALYSIS_FIGURES_ENABLED", "yes"),
        ("BEKIPER_POST_ANALYSIS_REPORT_ENABLED", "on"),
    ]);
    post.figures.enabled = false;
    post.report.enabled = false;
    crate::traced(|| post.apply_env_overrides_from(env_lookup(&enabled)));
    assert!(post.figures.enabled && post.report.enabled);
}

#[test]
fn post_analysis_constructor_overrides_and_validation_messages() {
    let mut post = PostAnalysisConfig::default();
    post.apply_overrides(&PostAnalysisOverrides {
        figure_selection: Some(vec!["figure_oft".into()]),
        figure_formats: Some(vec!["svg".into()]),
        report_enabled: Some(true),
        report_title: Some("Cohort".into()),
        report_compiler: Some("lualatex".into()),
        ..Default::default()
    });
    assert_eq!(
        post.figures.selection.as_deref(),
        Some(&["figure_oft".to_string()][..])
    );
    assert_eq!(post.figures.formats, ["svg"]);
    assert!(post.report.enabled);
    assert_eq!(post.report.title, "Cohort");
    assert_eq!(post.report.compiler, "lualatex");
    let invalid = |change: &dyn Fn(&mut PostAnalysisConfig)| {
        let mut post = PostAnalysisConfig {
            enabled: true,
            ..Default::default()
        };
        change(&mut post);
        post.validate().unwrap_err().to_string()
    };
    assert!(invalid(&|p| p.frozen_plan = Some(serde_json::json!([1]))).contains("JSON object"));
    assert!(
        invalid(&|p| {
            p.figures.enabled = true;
            p.figures.formats = vec!["gif".into()];
        })
        .contains("png/svg/pdf")
    );
    assert!(
        invalid(&|p| {
            p.report.enabled = true;
            p.report.compiler = "pdflatex".into();
        })
        .contains("latexmk or lualatex")
    );
    assert!(
        invalid(&|p| p.figures.selection = Some(vec!["figure_custom".into()]))
            .contains("requires a registered plan")
    );
}

#[test]
fn frozen_plans_load_from_toml_or_json_and_expose_their_fingerprints() {
    let dir = edge_dir("frozen_plan");
    let mut post = PostAnalysisConfig::default();
    assert_eq!(post.effective_frozen_plan(&dir).unwrap(), None);
    assert!(post.effective_frozen_fingerprints(&dir).unwrap().is_empty());
    std::fs::write(
        dir.join("plan.json"),
        r#"{"source_fingerprints": [{"role": "x"}]}"#,
    )
    .unwrap();
    post.plan_path = Some("plan.json".into());
    assert_eq!(
        post.effective_frozen_fingerprints(&dir).unwrap(),
        [serde_json::json!({"role": "x"})]
    );
    std::fs::write(dir.join("plan.toml"), "title = 'plan'\n").unwrap();
    post.plan_path = Some(dir.join("plan.toml"));
    assert_eq!(
        post.effective_frozen_plan(&dir).unwrap(),
        Some(serde_json::json!({"title": "plan"}))
    );
    assert!(post.effective_frozen_fingerprints(&dir).unwrap().is_empty());
    std::fs::write(dir.join("binary.toml"), b"\xff").unwrap();
    post.plan_path = Some("binary.toml".into());
    assert!(post.effective_frozen_plan(&dir).is_err());
    for (name, bytes) in [("bad.json", &b"{"[..]), ("bad.toml", &b"= 1"[..])] {
        std::fs::write(dir.join(name), bytes).unwrap();
        post.plan_path = Some(name.into());
        assert!(
            post.effective_frozen_plan(&dir)
                .unwrap_err()
                .to_string()
                .contains("invalid post-analysis plan")
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn post_analysis_workspace_counts_plan_contexts_and_overflow() {
    let mut post = PostAnalysisConfig {
        enabled: true,
        primary_workers: usize::MAX,
        ..Default::default()
    };
    post.figures.enabled = true;
    let small = post.workspace_bytes().unwrap();
    post.frozen_plan =
        Some(serde_json::json!({"execution": {"ymaze_primary_thresholds": [25, 45]}}));
    assert_eq!(post.workspace_bytes().unwrap(), 2 * small);
    post.figures.dpi = 50_000;
    post.frozen_plan = Some(serde_json::json!({
        "execution": {"ymaze_primary_thresholds": vec![25; 200_000]}
    }));
    assert!(
        post.workspace_bytes()
            .unwrap_err()
            .to_string()
            .contains("primary export workspace overflow")
    );
}

#[test]
fn task_overrides_apply_every_field_and_merge_nested_heuristics() {
    let overrides: TaskParamsOverrides = toml::from_str(
        r#"
        freezing_window_seconds = 2.0
        min_dwell_seconds = 0.5
        oft_window_seconds = 300.0
        centroid_coords = { x = "c_x", y = "c_y" }
        nose_coords = { x = "n_x", y = "n_y" }
        head_coords = { x = "h_x", y = "h_y" }
        tail_coords = { x = "t_x", y = "t_y" }
        [rearing]
        max_speed = 0.02
        [grooming]
        max_speed = 0.01
        "#,
    )
    .unwrap();
    let params = overrides.apply(bekiper_behaviour::TaskParams::new(30.0));
    assert_eq!(params.freezing_window_seconds, 2.0);
    assert_eq!(params.min_dwell_seconds, 0.5);
    assert_eq!(params.oft_window_seconds, 300.0);
    assert_eq!(params.centroid_coords.x, "c_x");
    assert_eq!(params.nose_coords.x, "n_x");
    assert_eq!(params.head_coords.x, "h_x");
    assert_eq!(params.tail_coords.x, "t_x");
    let variant: TaskParamsOverrides =
        toml::from_str("[rearing]\nmax_body_length = 0.06\n[grooming]\nmin_body_length = 0.03")
            .unwrap();
    let merged = overrides.merged(&variant);
    let rearing = merged.rearing.unwrap();
    assert_eq!(
        (rearing.max_speed, rearing.max_body_length),
        (Some(0.02), Some(0.06))
    );
    let grooming = merged.grooming.unwrap();
    assert_eq!(
        (grooming.max_speed, grooming.min_body_length),
        (Some(0.01), Some(0.03))
    );
    // A template-only nested table is inherited by a variant that omits it.
    let inherited = overrides.merged(&TaskParamsOverrides::default());
    assert_eq!(inherited.rearing.unwrap().max_speed, Some(0.02));
}

/// DLC animal directory: two sequences for 1035 plus metadata CSVs.
fn write_animal_dataset(root: &Path) {
    let data = root.join("dlc");
    let animal = data.join("1035");
    std::fs::create_dir_all(&animal).unwrap();
    std::fs::write(
        data.join("metadata-animal_sequence.csv"),
        "Animal,Sequence,MetersPerPixel,Center,Enclosure\n1035,0,1035_0,1035_0,enc\n1035,1,1035_0,1035_0\n999,0,999_0,999_0,\n1,2\n",
    )
    .unwrap();
    std::fs::write(
        data.join("metadata-animal.csv"),
        "Animal,Cohort,Gene,Sex,Age\n1035,c1,wt,m,12w\n",
    )
    .unwrap();
    for (name, contents) in [
        ("0.coordinates-1035_0.csv", ""),
        ("1.coordinates-1035_1.parquet", ""),
        (
            "meters_per_pixel-diagonal-0.707-1035_0.csv",
            "mpp,0,0,100,100,img.png,200,200\n",
        ),
        ("center-1035_0.csv", "center,50,60\n"),
        ("0.perimeter-circle-1035.csv", "0,10,10,15,10\n"),
        (
            "0.video_metadata-1035.yaml",
            "name: 0.video-1035_0\naverage_fps: 25\nresolution:\n  - 640\n  - 480\nnotes: x\n",
        ),
        (
            "1.video-1035_1-metadata.yaml",
            "source: camera\naverage_fps: 30\nresolution:\n  - 640\n",
        ),
        ("0.video-1035_0.mp4", ""),
        ("0.frame-1035.jpeg", ""),
        ("0.timestamps-1035_0.csv", ""),
    ] {
        std::fs::write(animal.join(name), contents).unwrap();
    }
}

const SEQUENCE_FIELDS: &str = r#"
data_dir = "dlc"
task_type = "open_field"
trial_sequence = ["oft", "oft"]
animal_sequence_csv = "metadata-animal_sequence.csv"
animal_metadata_csv = "metadata-animal.csv"
arena_meters = [0.5, 0.5]
"#;

#[test]
fn sequence_table_profile_resolves_animal_directories() {
    let root = edge_dir("sequence_table");
    write_animal_dataset(&root);
    let mut config = project_config(&format!(
        "[[ingress_profile]]\nprofile = \"sequencetable\"\n{SEQUENCE_FIELDS}"
    ));
    crate::traced(|| config.expand_for_validation(&root, &[])).unwrap();
    assert_eq!(config.animal_metadata[&1035].sex, "m");
    let names: Vec<_> = config.experiments.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["1035_seq0_oft", "1035_seq1_oft"]);
    let first = &config.experiments[0];
    assert_eq!(
        first.data_files,
        [PathBuf::from("dlc/1035/0.coordinates-1035_0.csv")]
    );
    assert_eq!(
        first.video.as_deref(),
        Some(Path::new("dlc/1035/0.video-1035_0.mp4"))
    );
    assert!(first.frame_image.is_some() && first.timestamps_file.is_some());
    assert!(matches!(first.fps, FpsConfig::Manual(fps) if fps == 25.0));
    assert_eq!(first.video_resolution, Some((640, 480)));
    let context = first.session_context.as_ref().unwrap();
    assert_eq!(
        (context.profile.as_str(), context.sex.as_deref()),
        ("sequencetable", Some("m"))
    );
    assert_eq!(
        first.detection.as_ref().unwrap().perimeters.as_deref(),
        Some(Path::new("dlc/1035/1035.perimeters-seq0.toml"))
    );
    let second = &config.experiments[1];
    assert!(matches!(second.fps, FpsConfig::Manual(fps) if fps == 30.0));
    assert_eq!(second.video_resolution, None);
    assert!(second.video.is_none() && second.frame_image.is_none());

    // Validation never writes perimeter files; materializing expansion does.
    let perimeters = root.join("dlc/1035/1035.perimeters-seq0.toml");
    assert!(!perimeters.exists());
    let mut materialized = project_config(&format!(
        "[[ingress_profile]]\nprofile = \"sequencetable\"\n{SEQUENCE_FIELDS}"
    ));
    materialized.expand(&root, &[]).unwrap();
    let written: bekiper_core::inspection::PerimetersFile =
        toml::from_str(&std::fs::read_to_string(&perimeters).unwrap()).unwrap();
    let labels: Vec<_> = written
        .perimeters
        .iter()
        .map(|p| p.label().to_string())
        .collect();
    assert_eq!(labels, ["object_0", "arena", "center_zone"]);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn animal_scans_expand_like_sequence_tables() {
    let root = edge_dir("animal_scan");
    write_animal_dataset(&root);
    let mut config = project_config(&format!("[[animal_scan]]\n{SEQUENCE_FIELDS}"));
    crate::traced(|| config.expand_for_validation(&root, &[crate::stage::Stage::Augment])).unwrap();
    assert_eq!(config.experiments.len(), 2);
    assert_eq!(config.animal_metadata[&1035].cohort, "c1");

    let mut missing = project_config(&format!(
        "[[animal_scan]]\n{}",
        SEQUENCE_FIELDS.replace("\"dlc\"", "\"absent\"")
    ));
    let error = missing
        .expand_for_validation(&root, &[crate::stage::Stage::Augment])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("stage `augment` requires [[animal_scan]] expansion"),
        "{error}"
    );
    assert!(
        error.contains("animal_scan data_dir does not exist"),
        "{error}"
    );
    let mut short = project_config(&format!(
        "[[animal_scan]]\n{}",
        SEQUENCE_FIELDS.replace("[\"oft\", \"oft\"]", "[\"oft\"]")
    ));
    let error = short
        .expand_for_validation(&root, &[])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("sequence 1 exceeds trial_sequence length 1"),
        "{error}"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn sequence_table_profile_requires_its_options_and_data_dir() {
    let root = edge_dir("sequence_options");
    for (missing, message) in [
        ("data_dir", "requires data_dir"),
        ("task_type", "requires task_type"),
        ("trial_sequence", "requires trial_sequence"),
        ("animal_sequence_csv", "requires animal_sequence_csv"),
    ] {
        let fields = SEQUENCE_FIELDS
            .lines()
            .filter(|line| !line.starts_with(missing))
            .collect::<Vec<_>>()
            .join("\n");
        let mut config = project_config(&format!(
            "[[ingress_profile]]\nprofile = \"sequencetable\"\n{fields}"
        ));
        let error = config
            .expand_for_validation(&root, &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains(message), "{missing}: {error}");
    }
    let mut config = project_config(&format!(
        "[[ingress_profile]]\nprofile = \"sequencetable\"\n{SEQUENCE_FIELDS}"
    ));
    let error = config
        .expand_for_validation(&root, &[])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("sequencetable data_dir does not exist"),
        "{error}"
    );
    let mut videomap = project_config("[[ingress_profile]]\nprofile = \"videomap\"\n");
    let error = videomap
        .expand_for_validation(&root, &[])
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires at least one"), "{error}");
    assert_eq!(IngressProfileId::VideoMap.as_str(), "videomap");
    let mut absent = project_config(
        "[[ingress_profile]]\nprofile = \"videomap\"\n[[ingress_profile.tasks]]\nvideo_dir = \"absent\"\ntask_type = \"open_field\"\nmeters_per_pixel = 0.001\n",
    );
    let error = absent
        .expand_for_validation(&root, &[])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("videomap video_dir does not exist"),
        "{error}"
    );
    let mut scan = project_config(
        "[[scan]]\nvideo_dir = \"absent\"\ntask_type = \"open_field\"\nmeters_per_pixel = 0.001\n",
    );
    let error = scan
        .expand_for_validation(&root, &[])
        .unwrap_err()
        .to_string();
    assert!(error.contains("scan video_dir does not exist"), "{error}");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn animal_directory_helpers_report_missing_or_malformed_calibration() {
    use super::io_helpers::*;
    let root = edge_dir("animal_helpers");
    let animal = root.join("7");
    std::fs::create_dir_all(&animal).unwrap();
    assert!(
        find_dlc_data_file(&animal, 7, 0)
            .unwrap_err()
            .to_string()
            .contains("no DLC data file found")
    );
    std::fs::write(
        animal.join("meters_per_pixel-diagonal-x-7_0.csv"),
        "mpp,0,0,1,1\n",
    )
    .unwrap();
    assert!(
        load_single_scale_calibration(&animal, "7_0")
            .unwrap_err()
            .to_string()
            .contains("cannot parse physical meters")
    );
    std::fs::remove_file(animal.join("meters_per_pixel-diagonal-x-7_0.csv")).unwrap();
    std::fs::write(
        animal.join("meters_per_pixel-diagonal-1.0-7_1.csv"),
        "mpp,5,5,5,5\nshort\n",
    )
    .unwrap();
    // A zero-length calibration line is not a scale; another animal's id falls back here.
    assert!(
        load_single_scale_calibration(&animal, "7_1")
            .unwrap_err()
            .to_string()
            .contains("no scale calibration found for mpp_id '7_1'")
    );
    std::fs::write(
        animal.join("meters_per_pixel-diagonal-1.0-8_0.csv"),
        "mpp,0,0,0,10\n",
    )
    .unwrap();
    assert_eq!(
        load_single_scale_calibration(&animal, "8_0").unwrap().0,
        0.1
    );
    std::fs::write(
        animal.join("meters_per_pixel-diagonal-1.0-7_2.csv"),
        "mpp,1,2\n",
    )
    .unwrap();
    std::fs::write(animal.join("meters_per_pixel-diagonal-1.0-7_3.csv"), "").unwrap();
    for id in ["7_2", "7_3"] {
        assert!(load_single_scale_calibration(&animal, id).is_err(), "{id}");
    }
    std::fs::write(animal.join("center-8_0.csv"), "center,1,2\n").unwrap();
    let mpp = bekiper_core::types::MetersPerPixel(0.01);
    assert!(
        load_center_zone(&animal, "8_0", (0.1, 0.1), mpp)
            .unwrap()
            .is_some()
    );
    assert!(
        load_arena_boundary(&animal, "9_0", (0.4, 0.4), mpp)
            .unwrap()
            .is_none()
    );
    assert!(
        load_sequence_perimeters(&animal, 7, 3, DetectionRecipe::Oft)
            .unwrap()
            .is_empty()
    );
    std::fs::write(animal.join("3.perimeter-notes.txt"), "").unwrap();
    assert!(
        load_sequence_perimeters(&animal, 7, 3, DetectionRecipe::Oft)
            .unwrap()
            .is_empty()
    );
    let meta = load_video_metadata(&animal, 7, 3).unwrap();
    assert_eq!((meta.fps, meta.stem.as_str()), (30.0, "3.video-7"));
    std::fs::write(
        animal.join("3.video_metadata-7.yaml"),
        "average_fps: fast\n",
    )
    .unwrap();
    assert!(load_video_metadata(&animal, 7, 3).is_err());
    assert!(!segment_matches("pre*", "post"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn metadata_prefilter_reports_unparseable_names_and_logs_exclusions() {
    let root = edge_dir("prefilter");
    crate::write_test_xlsx(&root.join("animals.xlsx"), &[&["Animal ID"], &["1"]]);
    let config = |policy: &str, names: &[&str]| {
        let experiments = names
            .iter()
            .map(|name| {
                format!("[[experiments]]\nname = \"{name}\"\ntask_type = \"open_field\"\nmeters_per_pixel = 0.001\n")
            })
            .collect::<String>();
        project_config(&format!(
            "[aggregate.metadata]\npath = \"animals.xlsx\"\nmissing_policy = \"{policy}\"\n{experiments}"
        ))
    };
    let mut kept = config("exclude_missing", &["1_seq0_oft"]);
    kept.expand_for_validation(&root, &[]).unwrap();
    assert_eq!(kept.experiments.len(), 1);
    let mut dropped = config(
        "exclude_missing",
        &["1_seq0_oft", "2_seq0_oft", "descriptive"],
    );
    crate::traced(|| dropped.expand_for_validation(&root, &[])).unwrap();
    assert_eq!(dropped.experiments.len(), 1);
    let mut failing = config("fail", &["descriptive"]);
    let error = failing
        .expand_for_validation(&root, &[])
        .unwrap_err()
        .to_string();
    assert!(error.contains("<unparseable> (descriptive)"), "{error}");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn scans_share_the_project_pose_model_and_tolerate_missing_metadata_csvs() {
    let root = edge_dir("scan_pose_model");
    std::fs::create_dir_all(root.join("videos")).unwrap();
    std::fs::write(root.join("videos/trial.mp4"), "").unwrap();
    let mut config = project_config(
        "[pose_model]\npath = \"model.onnx\"\nspec = \"spec.json\"\n[[experiments]]\nname = \"explicit\"\ntask_type = \"open_field\"\nmeters_per_pixel = 0.001\nvideo = \"videos/trial.mp4\"\n[[scan]]\nvideo_dir = \"videos\"\ntask_type = \"open_field\"\nmeters_per_pixel = 0.001\n",
    );
    crate::traced(|| config.expand_for_validation(&root, &[])).unwrap();
    assert!(config.experiments.iter().all(|e| e.pose_model.is_some()));

    write_animal_dataset(&root);
    std::fs::remove_file(root.join("dlc/metadata-animal.csv")).unwrap();
    let mut scan = project_config(&format!("[[animal_scan]]\n{SEQUENCE_FIELDS}"));
    scan.expand_for_validation(&root, &[]).unwrap();
    assert!(scan.animal_metadata.is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

fn nort_fixture(dir: &Path, videos: &[&str], metadata: &str) {
    for video in videos {
        let path = dir.join(video);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"video").unwrap();
        let pose = dir
            .join("deeplabcut/poses")
            .join(Path::new(video).with_extension("h5"));
        std::fs::create_dir_all(pose.parent().unwrap()).unwrap();
        std::fs::write(pose, b"h5").unwrap();
    }
    std::fs::create_dir_all(dir.join("annotations/nort")).unwrap();
    std::fs::write(dir.join("annotations/nort/metadata.csv"), metadata).unwrap();
}

fn imen_nort_rules_config() -> IngressConfig {
    project_config(
        r#"
        [[ingress_profile]]
        profile = "videomap"

        [[ingress_profile.tasks]]
        video_dir = "NORT videos"
        pose_data_dir = "deeplabcut/poses"
        task_type = "object_recognition"
        meters_per_pixel = { method = "arena_side", meters = 0.40 }
        backend = ["manual_annotation"]
        annotations = { dir = "annotations/nort" }

        [ingress_profile.tasks.recipe_rules]
        "**/T1/**" = "nort_training"
        "**/T2/**" = "nort_novelty"
        "#,
    )
}

#[test]
fn imen_nort_preflight_reports_every_recipe_stage_collision() {
    let dir = edge_dir("nort_collisions");
    nort_fixture(
        &dir,
        &[
            "NORT videos/NORT (pre)/T1/Config B/pre.T1.configB.video-83.mp4",
            "NORT videos/NORT (pre)/T2/Config B/pre.T2.configB.video-84.mp4",
        ],
        concat!(
            "Phase,PhasePart,TrialNumber,Stage,Animal,Video_file_name\n",
            "B,1,83,2,56,Test 83.szv\n",
            "B,1,84,1,57,Test 84.szv\n",
        ),
    );
    let error = imen_nort_rules_config()
        .expand_for_validation(&dir, &[])
        .unwrap_err()
        .to_string();
    assert!(error.contains("2 recipe/stage collision(s)"), "{error}");
    assert!(
        error.contains("path recipe nort_training requires stage 1"),
        "{error}"
    );
    assert!(
        error.contains("path recipe nort_novelty requires stage 2"),
        "{error}"
    );
    assert!(
        error.contains("candidate rows: row 2 phase_part 1 stage 2 animal 56"),
        "{error}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn imen_nort_rule_recovers_a_phaseless_stage_row_over_a_stage_zero_match() {
    let dir = edge_dir("nort_phaseless");
    nort_fixture(
        &dir,
        &["NORT videos/NORT (pre)/T1/Config B/pre.T1.configB.video-83.mp4"],
        concat!(
            "Phase,PhasePart,TrialNumber,Stage,Animal,Video_file_name\n",
            ",1,83,1,83,Test 83.szv\n",
            "B,1,83,0,83,Test 83.szv\n",
        ),
    );
    write_canonical_test_session(&dir, 83, 1, 1, "training_83", &["familiar", "variable"]);
    let mut config = imen_nort_rules_config();
    config.expand_for_validation(&dir, &[]).unwrap();
    let experiment = &config.experiments[0];
    assert_eq!(
        experiment.detection.as_ref().and_then(|d| d.recipe),
        Some(DetectionRecipe::NortTraining)
    );
    let context = experiment.session_context.as_ref().unwrap();
    assert_eq!(
        (context.stage.as_deref(), context.phase.as_deref()),
        (Some("1"), None)
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sequence_table_metadata_csv_is_optional() {
    let root = edge_dir("sequence_metadata_optional");
    write_animal_dataset(&root);
    std::fs::remove_file(root.join("dlc/metadata-animal.csv")).unwrap();
    for fields in [
        SEQUENCE_FIELDS.to_string(),
        SEQUENCE_FIELDS.replace("animal_metadata_csv = \"metadata-animal.csv\"\n", ""),
    ] {
        let mut config = project_config(&format!(
            "[[ingress_profile]]\nprofile = \"sequencetable\"\n{fields}"
        ));
        config.expand_for_validation(&root, &[]).unwrap();
        assert!(
            config.experiments[0]
                .session_context
                .as_ref()
                .unwrap()
                .sex
                .is_none()
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn whisker_env_reaches_scans_and_animal_scans() {
    use bekiper_feature::heuristic::solo::whisker_config::WhiskerEnv;
    let mut config = project_config(
        r#"
        [[scan]]
        video_dir = "videos"
        task_type = "object_recognition"
        meters_per_pixel = 0.001
        [[animal_scan]]
        data_dir = "dlc"
        task_type = "object_recognition"
        trial_sequence = ["nort"]
        animal_sequence_csv = "sequence.csv"
        "#,
    );
    config.apply_whisker_env(WhiskerEnv::default());
    assert!(config.scan[0].task_params.is_none());
    config.apply_whisker_env(WhiskerEnv {
        apex_fraction: Some(0.8),
        ..Default::default()
    });
    let apex = |params: &Option<TaskParamsOverrides>| {
        params
            .as_ref()
            .unwrap()
            .exploration
            .as_ref()
            .unwrap()
            .whisker
            .as_ref()
            .unwrap()
            .apex_fraction
    };
    assert_eq!(apex(&config.scan[0].task_params), Some(0.8));
    assert_eq!(apex(&config.animal_scan[0].task_params), Some(0.8));
}

#[test]
fn whisker_age_tables_resolve_per_experiment_sharing_one_workbook_read() {
    use bekiper_feature::heuristic::solo::whisker_config::{WhiskerEnv, WhiskerLengthSpec};
    let imen = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/neo/imen");
    let experiment = |name: &str, age: &str| {
        format!(
            r#"
            [[experiments]]
            name = "{name}"
            task_type = "object_recognition"
            meters_per_pixel = 0.001
            [experiments.session_context]
            animal = "1"
            phase_part = "2"
            age = "{age}"
            [experiments.task_params.exploration.whisker]
            enabled = true
            [experiments.task_params.exploration.whisker.length]
            method = "age_table"
            source = "mice_overview.xlsx"
            table = [[0.0, 0.0117], [400.0, 0.0117]]
            "#
        )
    };
    let mut config = project_config(&format!(
        "{}{}",
        experiment("1_post_a", "5"),
        experiment("1_post_b", "10000")
    ));
    config.apply_whisker_env(WhiskerEnv {
        apex_fraction: Some(0.8),
        ..Default::default()
    });
    config.resolve_whisker_lengths(&imen).unwrap();
    for experiment in &config.experiments {
        let whisker = experiment
            .task_params
            .as_ref()
            .and_then(|p| p.exploration.as_ref())
            .and_then(|e| e.whisker.as_ref())
            .unwrap();
        assert!(matches!(whisker.length, Some(WhiskerLengthSpec::Meters(_))));
        assert!(whisker.age_table.is_some() && whisker.length_source.is_some());
    }
}
