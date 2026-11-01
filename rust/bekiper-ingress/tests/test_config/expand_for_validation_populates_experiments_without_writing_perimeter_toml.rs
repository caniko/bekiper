#[test]
fn expand_for_validation_populates_experiments_without_writing_perimeter_toml() {
    let root = fresh_scan_root("validation_no_writes");
    let data_dir = root.join("dlc-data");
    let animal_dir = data_dir.join("100");
    std::fs::create_dir_all(&animal_dir).unwrap();

    std::fs::write(
        data_dir.join("metadata-animal_sequence.csv"),
        "Animal,Sequence,MetersPerPixel,Center,Enclosure\n100,0,100_0,100_0,\n",
    )
    .unwrap();
    std::fs::write(
        animal_dir.join("0.coordinates-100_0DLC_resnet50_test.csv"),
        "",
    )
    .unwrap();
    std::fs::write(
        animal_dir.join("meters_per_pixel-diagonal-0.5-100_0.csv"),
        "label,0,0,100,0,img,640,480\n",
    )
    .unwrap();

    let config_path = write_scan_config(&root, "dlc-data");
    let mut config = IngressConfig::from_file(&config_path).unwrap();
    let perimeter_toml = animal_dir.join("100.perimeters-seq0.toml");

    config.expand_for_validation(&root, Stage::all()).unwrap();
    assert_eq!(config.experiments.len(), 1);
    assert!(
        !perimeter_toml.exists(),
        "expand_for_validation() must not materialize {}",
        perimeter_toml.display()
    );

    std::fs::remove_dir_all(&root).ok();
}

#[cfg(feature = "pose-gpu")]
#[test]
fn detection_dry_run_does_not_materialize_animal_perimeters() {
    let root = fresh_scan_root("dry_run_no_writes");
    let data_dir = root.join("dlc-data");
    let animal_dir = data_dir.join("100");
    std::fs::create_dir_all(&animal_dir).unwrap();
    std::fs::write(
        data_dir.join("metadata-animal_sequence.csv"),
        "Animal,Sequence,MetersPerPixel,Center,Enclosure\n100,0,100_0,100_0,\n",
    )
    .unwrap();
    std::fs::write(
        animal_dir.join("0.coordinates-100_0DLC_resnet50_test.csv"),
        "",
    )
    .unwrap();
    std::fs::write(
        animal_dir.join("meters_per_pixel-diagonal-0.5-100_0.csv"),
        "label,0,0,100,0,img,640,480\n",
    )
    .unwrap();
    let config_path = write_scan_config(&root, "dlc-data");
    let perimeter_path = animal_dir.join("100.perimeters-seq0.toml");
    let summary = bekiper_ingress::workflow::analyze_detection_dry_run(&config_path).unwrap();
    assert_eq!(summary.total_experiments, 1);
    assert!(
        !perimeter_path.exists(),
        "dry-run wrote {}",
        perimeter_path.display()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn experiment_with_detection_backend() {
    let toml_str = r#"
        [project]
        project_name = "detect_test"
        data_directory = "/data"

        [[experiments]]
        name = "nort1"
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001
        video = "/data/video.mp4"

        [experiments.detection]
        recipe = "nort_training"
        backend = ["classical_cv", "sam2_auto"]
    "#;
    let config: IngressConfig = toml::from_str(toml_str).unwrap();
    let det = config.experiments[0].detection.as_ref().unwrap();
    assert_eq!(
        det.backend.backends(),
        &[DetectionBackend::ClassicalCv, DetectionBackend::Sam2Auto]
    );
}

// ---------------------------------------------------------------------------
// AggregateConfig — defaults, TOML round-trip, env overlay.
//
// Mirrors the `gpu_env_overlay_*` tests in src/config.rs. Per CLAUDE.md, all
// env-overlay tests use a HashMap-backed closure and NEVER mutate the
// process environment.
// ---------------------------------------------------------------------------

#[test]
fn aggregate_config_defaults() {
    let cfg = AggregateConfig::default();
    assert!(cfg.enabled, "aggregate stage must default to enabled");
    assert_eq!(cfg.output_filename, "results.ods");
    assert!(cfg.recipe_to_sheet.is_empty());
    assert!(cfg.aggregation.is_empty());
}

#[test]
fn aggregate_config_absent_section_uses_defaults() {
    // An IngressConfig with no [aggregate] block should still deserialize
    // cleanly, with AggregateConfig::default() plugged in.
    let toml_str = r#"
        [project]
        project_name = "agg_defaults"
        data_directory = "/data"

        [[experiments]]
        name = "exp1"
        data_files = ["/data/exp1.csv"]
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001
    "#;
    let config: IngressConfig = toml::from_str(toml_str).unwrap();
    assert!(config.aggregate.enabled);
    assert_eq!(config.aggregate.output_filename, "results.ods");
    assert!(config.aggregate.recipe_to_sheet.is_empty());
    assert!(config.aggregate.aggregation.is_empty());
}

#[test]
fn aggregate_config_parses_recipe_to_sheet_and_rules() {
    // Rename three recipes to training/update/test in config order. Aggregate
    // output is per-trial; the only supported explicit rule is passthrough.
    let toml_str = r#"
        [project]
        project_name = "agg_dejana"
        data_directory = "/data"

        [aggregate]
        enabled = true
        output_filename = "dejana_oul_results.xlsx"

        [aggregate.recipe_to_sheet]
        oultraining = "training"
        oulupdate   = "update"
        oultest     = "test"

        [aggregate.aggregation]
        training = "passthrough"
        update   = "passthrough"
        test     = "passthrough"
    "#;
    let config: IngressConfig = toml::from_str(toml_str).unwrap();
    let agg = &config.aggregate;
    assert!(agg.enabled);
    assert_eq!(agg.output_filename, "dejana_oul_results.xlsx");
    assert_eq!(
        agg.recipe_to_sheet.get("oultraining").map(String::as_str),
        Some("training")
    );
    assert_eq!(
        agg.recipe_to_sheet.get("oultest").map(String::as_str),
        Some("test")
    );
    assert_eq!(
        agg.recipe_to_sheet.keys().cloned().collect::<Vec<_>>(),
        vec![
            "oultraining".to_string(),
            "oulupdate".to_string(),
            "oultest".to_string()
        ]
    );
    assert_eq!(
        agg.aggregation.get("training"),
        Some(&AggregationRule::Passthrough)
    );
    assert_eq!(
        agg.aggregation.get("test"),
        Some(&AggregationRule::Passthrough)
    );
}

#[test]
fn aggregate_config_rejects_mean_by_animal_rule() {
    let toml_str = r#"
        [project]
        project_name = "agg_invalid"
        data_directory = "/data"

        [aggregate.aggregation]
        training = "mean_by_animal"
    "#;
    let err = toml::from_str::<IngressConfig>(toml_str).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("mean_by_animal"),
        "parse error should mention invalid rule, got: {message}"
    );
}

#[test]
fn aggregate_config_rejects_mean_by_animal_qc_rule() {
    let toml_str = r#"
        [project]
        project_name = "agg_invalid"
        data_directory = "/data"

        [aggregate.aggregation]
        training = "mean_by_animal_qc"
    "#;
    let err = toml::from_str::<IngressConfig>(toml_str).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("mean_by_animal_qc"),
        "parse error should mention invalid rule, got: {message}"
    );
}

#[test]
fn aggregate_config_parses_metadata_source() {
    let toml_str = r#"
        [project]
        project_name = "imen"
        data_directory = "."

        [aggregate]
        output_filename = "imen_results.ods"

        [aggregate.metadata]
        path = "animal_metadata.xlsx"
        sheet = "Updated_Mice-overview"
        missing_policy = "allow_blanks"
    "#;
    let config: IngressConfig = toml::from_str(toml_str).unwrap();
    let metadata = config.aggregate.metadata.as_ref().unwrap();
    assert_eq!(
        metadata.path,
        std::path::PathBuf::from("animal_metadata.xlsx")
    );
    assert_eq!(metadata.sheet, "Updated_Mice-overview");
    assert_eq!(metadata.missing_policy, MissingMetadataPolicy::AllowBlanks);
}

#[test]
fn aggregate_env_overlay_disabled_flag_sets_enabled_false() {
    let mut cfg = AggregateConfig::default();
    let env = HashMap::from([("BEKIPER_AGGREGATE_DISABLED", "1")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!(!cfg.enabled);
}

#[test]
fn aggregate_env_overlay_disabled_true_sets_enabled_false() {
    let mut cfg = AggregateConfig::default();
    let env = HashMap::from([("BEKIPER_AGGREGATE_DISABLED", "true")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!(!cfg.enabled);
}

#[test]
fn aggregate_env_overlay_disabled_yes_case_insensitive() {
    let mut cfg = AggregateConfig::default();
    let env = HashMap::from([("BEKIPER_AGGREGATE_DISABLED", "YES")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!(!cfg.enabled);
}

#[test]
fn aggregate_env_overlay_disabled_false_re_enables() {
    // File said enabled=false; env flips it back on.
    let mut cfg = AggregateConfig {
        enabled: false,
        ..AggregateConfig::default()
    };
    let env = HashMap::from([("BEKIPER_AGGREGATE_DISABLED", "false")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!(cfg.enabled);
}

#[test]
fn aggregate_env_overlay_disabled_nonsense_ignored() {
    let mut cfg = AggregateConfig::default();
    let env = HashMap::from([("BEKIPER_AGGREGATE_DISABLED", "maybe")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    // Unparseable value leaves enabled at its prior state (default true).
    assert!(cfg.enabled);
}

#[test]
fn aggregate_env_overlay_applies_output_filename() {
    let mut cfg = AggregateConfig::default();
    let env = HashMap::from([("BEKIPER_AGGREGATE_OUTPUT", "override.xlsx")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert_eq!(cfg.output_filename, "override.ods");
}

#[test]
fn aggregate_env_overlay_rejects_empty_output_filename() {
    let mut cfg = AggregateConfig {
        output_filename: "file.xlsx".to_string(),
        ..AggregateConfig::default()
    };
    let env = HashMap::from([("BEKIPER_AGGREGATE_OUTPUT", "   ")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    // Whitespace-only value is rejected; prior filename preserved.
    assert_eq!(cfg.output_filename, "file.xlsx");
}

#[test]
fn aggregate_env_overlay_beats_file_values() {
    // Simulates: file sets enabled=true / output_filename="file.xlsx";
    // env overrides both.
    let mut cfg = AggregateConfig {
        enabled: true,
        output_filename: "file.xlsx".to_string(),
        recipe_to_sheet: Default::default(),
        aggregation: Default::default(),
        metadata: None,
    };
    let env = HashMap::from([
        ("BEKIPER_AGGREGATE_DISABLED", "1"),
        ("BEKIPER_AGGREGATE_OUTPUT", "env_override.xlsx"),
    ]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!(!cfg.enabled);
    assert_eq!(cfg.output_filename, "env_override.ods");
}

#[test]
fn aggregate_env_overlay_empty_env_preserves_config() {
    let mut cfg = AggregateConfig {
        enabled: false,
        output_filename: "custom.xlsx".to_string(),
        recipe_to_sheet: Default::default(),
        aggregation: Default::default(),
        metadata: None,
    };
    let env: HashMap<&'static str, &'static str> = HashMap::new();
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!(!cfg.enabled);
    assert_eq!(cfg.output_filename, "custom.xlsx");
}

#[test]
fn aggregate_env_overlay_partial_keys_only_touches_specified() {
    let mut cfg = AggregateConfig {
        enabled: true,
        output_filename: "kept.xlsx".to_string(),
        recipe_to_sheet: Default::default(),
        aggregation: Default::default(),
        metadata: None,
    };
    // Only override the disabled flag — filename stays.
    let env = HashMap::from([("BEKIPER_AGGREGATE_DISABLED", "1")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!(!cfg.enabled);
    assert_eq!(cfg.output_filename, "kept.xlsx");
}

#[test]
fn aggregate_env_overlay_does_not_touch_recipe_or_aggregation_maps() {
    // Env overlay has no knobs for the nested maps; they must round-trip
    // unchanged even when other keys are set.
    let mut recipe_to_sheet = IndexMap::new();
    recipe_to_sheet.insert("oultraining".to_string(), "training".to_string());
    let mut aggregation = IndexMap::new();
    aggregation.insert("training".to_string(), AggregationRule::Passthrough);

    let mut cfg = AggregateConfig {
        enabled: true,
        output_filename: "x.xlsx".to_string(),
        recipe_to_sheet: recipe_to_sheet.clone(),
        aggregation: aggregation.clone(),
        metadata: None,
    };
    let env = HashMap::from([
        ("BEKIPER_AGGREGATE_DISABLED", "1"),
        ("BEKIPER_AGGREGATE_OUTPUT", "new.xlsx"),
    ]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert_eq!(cfg.recipe_to_sheet, recipe_to_sheet);
    assert_eq!(cfg.aggregation, aggregation);
    assert_eq!(cfg.output_filename, "new.ods");
}
