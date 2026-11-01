use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bekiper_core::scale::ScaleConfig;
use bekiper_core::types::{DetectionBackend, PoseModelPreset, TaskType};
use bekiper_core::video::FpsConfig;
use bekiper_ingress::config::{
    AggregateConfig, AggregationRule, DetectionConfig, ExperimentConfig, IngressConfig,
    MissingMetadataPolicy, PoseModelConfig,
};
use bekiper_ingress::stage::Stage;
use indexmap::IndexMap;

/// Build a `Fn(&str) -> Option<String>` closure backed by a `HashMap`.
///
/// Mirrors the helper used by the `gpu_env_overlay_*` tests in
/// `src/config.rs` — per CLAUDE.md, tests must **never** mutate the process
/// environment (`std::env::set_var` is `unsafe` and thread-unsafe under
/// cargo test's parallel runner).
fn env_lookup<'a>(
    map: &'a HashMap<&'static str, &'static str>,
) -> impl Fn(&str) -> Option<String> + 'a {
    move |k| map.get(k).map(|v| (*v).to_string())
}

#[test]
fn ingress_config_from_toml_string() {
    let toml_str = r#"
        [project]
        project_name = "test_project"
        data_directory = "/data"

        [[experiments]]
        name = "trial1"
        data_files = ["/data/trial1.csv"]
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001

        [[experiments]]
        name = "trial2"
        data_files = ["/data/trial2.csv", "/data/trial2b.csv"]
        task_type = "y_maze"
        fps = 25.0
        meters_per_pixel = 0.002
    "#;

    let config: IngressConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.project.project_name, "test_project");
    assert_eq!(config.experiments.len(), 2);
    assert_eq!(config.experiments[0].name, "trial1");
    assert_eq!(config.experiments[0].task_type, TaskType::ObjectRecognition);
    assert!(
        matches!(config.experiments[0].fps, FpsConfig::Manual(fps) if (fps - 30.0).abs() < f64::EPSILON)
    );
    assert_eq!(config.experiments[1].data_files.len(), 2);
}

#[test]
fn experiment_config_serde_roundtrip() {
    let exp = ExperimentConfig {
        recording_id: None,
        name: "test".into(),
        data_files: vec!["/a.csv".into()],
        task_type: TaskType::Cheeseboard,
        fps: FpsConfig::Manual(30.0),
        meters_per_pixel: ScaleConfig::Manual(0.001),
        video: None,
        pose_model: None,
        detection: None,
        task_params: None,
        trial_start_perimeter: None,
        annotation_dir: None,
        frame_timing: None,
        video_resolution: None,
        frame_image: None,
        timestamps_file: None,
        animal_id: None,
        session_context: None,
    };
    let toml_str = toml::to_string(&exp).unwrap();
    let deser: ExperimentConfig = toml::from_str(&toml_str).unwrap();
    assert_eq!(deser.name, "test");
    assert_eq!(deser.task_type, TaskType::Cheeseboard);
}

#[test]
fn ingress_config_from_file_nonexistent() {
    let result = IngressConfig::from_file(std::path::Path::new("/nonexistent/config.toml"));
    assert!(result.is_err());
}

#[test]
fn ingress_config_from_file_roundtrip() {
    let toml_str = r#"
        [project]
        project_name = "file_test"
        data_directory = "/data"

        [[experiments]]
        name = "exp1"
        data_files = ["/data/exp1.csv"]
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001
    "#;

    let tmp = std::env::temp_dir().join("bekiper_test_ingress_config.toml");
    std::fs::write(&tmp, toml_str).unwrap();

    let config = IngressConfig::from_file(&tmp).unwrap();
    assert_eq!(config.project.project_name, "file_test");
    assert_eq!(config.experiments.len(), 1);

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn pose_model_config_resolves_relative_assets_against_base_dir() {
    let base_dir = Path::new("/tmp/bekiper_pose_model_relative");
    let cfg = PoseModelConfig {
        path: PathBuf::from("models/model.onnx"),
        spec: PoseModelPreset::Custom(PathBuf::from("models/model.json")),
        batch_size: 16,
    };

    let resolved = cfg.resolved_against(base_dir);

    assert_eq!(resolved.path, base_dir.join("models/model.onnx"));
    assert_eq!(
        resolved.spec,
        PoseModelPreset::Custom(base_dir.join("models/model.json"))
    );
    assert_eq!(resolved.batch_size, 16);
}

#[test]
fn pose_model_config_preserves_absolute_path_and_named_preset() {
    let base_dir = Path::new("/tmp/bekiper_pose_model_absolute");
    let absolute_model_path = std::env::temp_dir().join("bekiper_absolute_model.onnx");
    let cfg = PoseModelConfig {
        path: absolute_model_path.clone(),
        spec: PoseModelPreset::DlcSuperAnimal,
        batch_size: 8,
    };

    let resolved = cfg.resolved_against(base_dir);

    assert_eq!(resolved.path, absolute_model_path);
    assert_eq!(resolved.spec, PoseModelPreset::DlcSuperAnimal);
    assert_eq!(resolved.batch_size, 8);
}

#[test]
fn detection_config_single_backend() {
    let toml_str = r#"
        recipe = "nort_training"
        backend = "classical_cv"
    "#;
    let config: DetectionConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.backend.backends(), &[DetectionBackend::ClassicalCv]);
}

#[test]
fn detection_config_backend_priority_list() {
    let toml_str = r#"
        recipe = "nort_training"
        backend = ["classical_cv", "sam2_auto", "grounding_dino_sam2"]
    "#;
    let config: DetectionConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(
        config.backend.backends(),
        &[
            DetectionBackend::ClassicalCv,
            DetectionBackend::Sam2Auto,
            DetectionBackend::GroundingDinoSam2,
        ]
    );
}

#[test]
fn detection_config_defaults_to_classical_cv() {
    let toml_str = r#"
        recipe = "nort_training"
    "#;
    let config: DetectionConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.backend.backends(), &[DetectionBackend::ClassicalCv]);
}

#[test]
fn detection_config_backend_shorthand_aliases() {
    let toml_str = r#"
        recipe = "oft"
        backend = ["cv", "sam2", "gdino"]
    "#;
    let config: DetectionConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(
        config.backend.backends(),
        &[
            DetectionBackend::ClassicalCv,
            DetectionBackend::Sam2Auto,
            DetectionBackend::GroundingDinoSam2,
        ]
    );
}

#[test]
fn ingress_env_overlay_applies_runtime_settings() {
    let mut config: IngressConfig = toml::from_str(
        r#"
        [project]
        project_name = "runtime_env"
        data_directory = "."

        [project.runtime]
        num_threads = 2
        memory_scheduler_enabled = false
        max_ram_usage_fraction = 0.6
        "#,
    )
    .unwrap();

    let env = HashMap::from([
        ("BEKIPER_RUNTIME_NUM_THREADS", "5"),
        ("BEKIPER_RUNTIME_MEMORY_SCHEDULER", "true"),
        ("BEKIPER_RUNTIME_MAX_RAM_USAGE", "0.85"),
    ]);
    config.apply_env_overrides_from(env_lookup(&env));

    assert_eq!(config.project.runtime.num_threads, Some(5));
    assert!(config.project.runtime.memory_scheduler_enabled);
    assert!((config.project.runtime.max_ram_usage_fraction - 0.85).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Lazy animal_scan expansion: from_file() parses; expand() does I/O.
// ---------------------------------------------------------------------------

/// Allocate a unique temp directory for an animal_scan fixture.
fn fresh_scan_root(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let root = std::env::temp_dir().join(format!("bekiper_animal_scan_{tag}_{pid}_{n}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn write_scan_config(root: &std::path::Path, dlc_dir_name: &str) -> std::path::PathBuf {
    let config = format!(
        r#"
        [project]
        project_name = "lazy_scan_test"
        data_directory = "."

        [[animal_scan]]
        data_dir = "{dlc_dir_name}"
        task_type = "object_recognition"
        trial_sequence = ["oul_training"]
        animal_sequence_csv = "metadata-animal_sequence.csv"
        "#
    );
    let path = root.join("bekiper_config.toml");
    std::fs::write(&path, config).unwrap();
    path
}

#[test]
fn from_file_with_missing_animal_scan_csv_succeeds() {
    let root = fresh_scan_root("missing_csv_from_file");
    std::fs::create_dir_all(root.join("dlc-data")).unwrap();
    let config_path = write_scan_config(&root, "dlc-data");

    // Pure parse — should succeed even though metadata-animal_sequence.csv
    // is absent, because expansion is deferred to expand().
    let config = IngressConfig::from_file(&config_path).expect("from_file must be pure parse");
    assert!(
        config.experiments.is_empty(),
        "experiments should be empty before expand()"
    );
    assert_eq!(config.animal_scan.len(), 1);

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn expand_with_missing_animal_scan_csv_returns_stage_error() {
    let root = fresh_scan_root("missing_csv_expand");
    std::fs::create_dir_all(root.join("dlc-data")).unwrap();
    let config_path = write_scan_config(&root, "dlc-data");

    let mut config = IngressConfig::from_file(&config_path).unwrap();
    let err = config
        .expand(&root, &[Stage::Augment])
        .expect_err("expand must fail when the manifest CSV is missing");

    let msg = err.to_string();
    assert!(
        msg.contains("stage `augment`"),
        "error should name the triggering stage, got: {msg}"
    );
    assert!(
        msg.contains("metadata-animal_sequence.csv"),
        "error should mention the missing file, got: {msg}"
    );

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn expand_without_animal_scan_succeeds_without_csv() {
    // Config with only explicit [[experiments]] — no animal_scan at all.
    // expand() must be a no-op, regardless of which stage is passed.
    let root = fresh_scan_root("explicit_experiments");
    let config_path = root.join("bekiper_config.toml");
    let toml_str = r#"
        [project]
        project_name = "explicit_test"
        data_directory = "/data"

        [[experiments]]
        name = "exp1"
        data_files = ["/data/exp1.csv"]
        task_type = "object_recognition"
        fps = 30.0
        meters_per_pixel = 0.001
    "#;
    std::fs::write(&config_path, toml_str).unwrap();

    let mut config = IngressConfig::from_file(&config_path).unwrap();
    assert_eq!(config.experiments.len(), 1);
    config
        .expand(&root, &[Stage::Augment])
        .expect("expand should succeed with no animal_scan entries");
    assert_eq!(config.experiments.len(), 1);
    assert_eq!(config.experiments[0].name, "exp1");

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn from_file_does_not_write_perimeter_toml_files() {
    // Regression test: before the lazy-expansion refactor, just loading a
    // config with [[animal_scan]] would WRITE `{animal}.perimeters-seqN.toml`
    // files back into the source data directory. After the refactor, those
    // writes must happen only in expand(), never in from_file().
    let root = fresh_scan_root("no_writes_on_load");
    let data_dir = root.join("dlc-data");
    let animal_dir = data_dir.join("100");
    std::fs::create_dir_all(&animal_dir).unwrap();

    // Manifest CSV: one row, sequence 0, mpp_id/center_id reference
    // calibration files we'll create below.
    std::fs::write(
        data_dir.join("metadata-animal_sequence.csv"),
        "Animal,Sequence,MetersPerPixel,Center,Enclosure\n100,0,100_0,100_0,\n",
    )
    .unwrap();

    // DLC coordinate file — contents aren't parsed at expand time,
    // only the filename is looked up.
    std::fs::write(
        animal_dir.join("0.coordinates-100_0DLC_resnet50_test.csv"),
        "",
    )
    .unwrap();

    // Scale calibration: pixel distance (0,0)→(100,0) = 100 px, physical
    // 0.5 m → 0.005 m/px.
    std::fs::write(
        animal_dir.join("meters_per_pixel-diagonal-0.5-100_0.csv"),
        "label,0,0,100,0,img,640,480\n",
    )
    .unwrap();
    // No perimeter, center, or video metadata files — the expand loop
    // treats those as optional (warn and default).

    let config_path = write_scan_config(&root, "dlc-data");

    // Load only. No expand(). Nothing should be written.
    let mut config = IngressConfig::from_file(&config_path).unwrap();
    let perimeter_toml = animal_dir.join("100.perimeters-seq0.toml");
    assert!(
        !perimeter_toml.exists(),
        "from_file() must not write {} as a side effect",
        perimeter_toml.display()
    );
    assert!(
        config.experiments.is_empty(),
        "experiments must not be populated before expand()"
    );

    // Now expand — the file should appear, proving the side effect moved.
    config.expand(&root, &[Stage::Augment]).unwrap();
    assert!(
        perimeter_toml.exists(),
        "expand() must write {}",
        perimeter_toml.display()
    );
    assert_eq!(
        config.experiments.len(),
        1,
        "expand() should populate exactly one experiment from the single manifest row"
    );

    std::fs::remove_dir_all(&root).ok();
}
