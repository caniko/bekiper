#[test]
fn glob_star_matches_single_segment() {
    assert!(io_helpers::glob_matches(
        "*/T1/*",
        "NORT (post)/T1/Config A"
    ));
    assert!(!io_helpers::glob_matches(
        "*/T1/*",
        "NORT (post)/T2/Config A"
    ));
}

#[test]
fn glob_double_star() {
    assert!(io_helpers::glob_matches("**/T1/**", "a/b/T1/c/d"));
    assert!(io_helpers::glob_matches("**/*.mp4", "deep/nested/file.mp4"));
}

#[test]
fn derive_name_nort() {
    let p = Path::new("NORT videos/NORT (post)/T1/Config A/post.T1.configA.video-10.mp4");
    let name = io_helpers::derive_experiment_name(p);
    assert_eq!(
        name,
        "nort_videos_nort_post_t1_config_a_post_t1_configa_video_10"
    );
}

#[test]
fn derive_name_oft() {
    let p = Path::new("OFT videos/OFT (pre)/pre.video-42.mp4");
    let name = io_helpers::derive_experiment_name(p);
    assert_eq!(name, "oft_videos_oft_pre_pre_video_42");
}

#[test]
fn segment_matches_wildcard() {
    assert!(io_helpers::segment_matches("T*", "T1"));
    assert!(io_helpers::segment_matches("Config*", "Config A"));
    assert!(io_helpers::segment_matches("*", "anything"));
    assert!(io_helpers::segment_matches("NORT*", "NORT (post)"));
}

#[test]
fn resolve_recipe_rules_override_default() {
    let scan = ScanConfig {
        video_dir: PathBuf::from("videos"),
        task_type: TaskType::ObjectRecognition,
        fps: FpsConfig::Manual(15.0),
        meters_per_pixel: ScaleConfig::Manual(0.001),
        recipe: Some(DetectionRecipe::NortTraining),
        recipe_rules: HashMap::from([("**/T2/**".into(), DetectionRecipe::NortNovelty)]),
        backend: BackendPriority::default(),
        arena_calibration: None,
        extensions: scan::default_extensions(),
        annotations: None,
        task_params: None,
        variants: Vec::new(),
        trial_start_perimeter: None,
    };

    let t1 = Path::new("NORT (post)/T1/Config A/video.mp4");
    assert_eq!(
        io_helpers::resolve_recipe(&scan, t1),
        Some(DetectionRecipe::NortTraining)
    );

    let t2 = Path::new("NORT (post)/T2/Config B/video.mp4");
    assert_eq!(
        io_helpers::resolve_recipe(&scan, t2),
        Some(DetectionRecipe::NortNovelty)
    );
}

#[test]
fn profile_registry_returns_in_tree_profiles() {
    assert_eq!(
        IngressProfileRegistry::get(IngressProfileId::VideoMap).id(),
        IngressProfileId::VideoMap
    );
    assert_eq!(
        IngressProfileRegistry::get(IngressProfileId::SequenceTable).id(),
        IngressProfileId::SequenceTable
    );
}

// ── GpuConfig env-overlay tests ─────────────────────────────────
//
// All tests use `apply_env_overrides_from` with an injected
// HashMap-backed closure instead of mutating the process
// environment. This keeps the tests thread-safe and avoids the
// `unsafe` std::env::set_var now required in recent Rust editions.
