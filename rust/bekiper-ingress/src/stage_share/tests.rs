use super::*;
use crate::config::TaskParamsOverrides;
use serde_json::json;

fn config() -> IngressConfig {
    toml::from_str(
        r#"
        [project]
        project_name = "share"
        data_directory = "."
        "#,
    )
    .unwrap()
}

fn experiment() -> ExperimentConfig {
    toml::from_str(
        r#"
        name = "rec__a"
        task_type = "y_maze"
        fps = 30.0
        meters_per_pixel = 0.001
        video = "rec.mp4"
        "#,
    )
    .unwrap()
}

fn fingerprints(experiment: &ExperimentConfig, config: &IngressConfig) -> RecordingFingerprints {
    RecordingFingerprints::new(experiment, config).unwrap()
}

fn changed(before: &RecordingFingerprints, after: &RecordingFingerprints) -> Vec<&'static str> {
    Shared::ALL
        .iter()
        .filter(|kind| before.get(**kind) != after.get(**kind))
        .map(|kind| kind.name())
        .collect()
}

/// A sample value for every `task_params` key. A new field must be added here
/// and either declared in `AUGMENT_TASK_PARAM_KEYS` or proven behaviour-only.
fn task_param_samples() -> BTreeMap<&'static str, Value> {
    let coords = json!({ "x": "sample_x", "y": "sample_y" });
    BTreeMap::from([
        ("freezing_window_seconds", json!(1.5)),
        ("freezing_displacement_threshold", json!(0.01)),
        ("min_dwell_seconds", json!(2.0)),
        ("oft_window_seconds", json!(60.0)),
        ("peripheral_band_fraction", json!(0.3)),
        ("centroid_coords", coords.clone()),
        ("nose_coords", coords.clone()),
        ("head_coords", coords.clone()),
        ("tail_coords", coords),
        ("exploration", json!({ "nose_max_distance": 0.05 })),
        ("ymaze", json!({ "proximal_threshold": 0.5 })),
        ("rearing", json!({ "max_speed": 0.5 })),
        ("grooming", json!({ "max_speed": 0.5 })),
    ])
}

#[test]
fn every_task_param_is_either_declared_or_behaviour_only() {
    let all_keys: Vec<String> = match serde_json::to_value(TaskParamsOverrides::default()).unwrap()
    {
        Value::Object(map) => map.keys().cloned().collect(),
        other => panic!("task_params is not a table: {other}"),
    };
    let samples = task_param_samples();
    for key in &all_keys {
        assert!(
            samples.contains_key(key.as_str()),
            "task_params.{key} has no drift sample; declare it in AUGMENT_TASK_PARAM_KEYS or add a behaviour-only sample"
        );
    }
    assert_eq!(all_keys.len(), samples.len(), "stale sample key");

    let config = config();
    let base = experiment();
    let baseline = fingerprints(&base, &config);
    for key in all_keys {
        let mut perturbed = base.clone();
        perturbed.task_params =
            Some(serde_json::from_value(json!({ key.clone(): samples[key.as_str()] })).unwrap());
        let expected: &[&str] = if AUGMENT_TASK_PARAM_KEYS.contains(&key.as_str()) {
            &["augment"]
        } else {
            &[]
        };
        assert_eq!(
            changed(&baseline, &fingerprints(&perturbed, &config)),
            expected,
            "task_params.{key}"
        );
    }
}

#[test]
fn fingerprints_follow_what_each_stage_reads() {
    let config = config();
    let base = experiment();
    let baseline = fingerprints(&base, &config);

    // The variant name is identity only.
    let mut renamed = base.clone();
    renamed.name = "rec__b".into();
    assert!(changed(&baseline, &fingerprints(&renamed, &config)).is_empty());

    // A different video reaches every stage.
    let mut video = base.clone();
    video.video = Some("other.mp4".into());
    assert_eq!(
        changed(&baseline, &fingerprints(&video, &config)),
        ["probe", "detect", "pose", "augment"]
    );

    // Pipeline parameters are read by augment only.
    let mut pipeline = config.clone();
    pipeline.pipeline.invert_y = !pipeline.pipeline.invert_y;
    assert_eq!(
        changed(&baseline, &fingerprints(&base, &pipeline)),
        ["augment"]
    );
}

#[test]
fn fingerprint_is_canonical() {
    let a = json!({ "b": 1.5, "a": [1, { "z": 0, "y": null }] });
    let b: Value = serde_json::from_str(r#"{"a":[1,{"y":null,"z":0}],"b":1.5}"#).unwrap();
    assert_eq!(canonical(&a), canonical(&b));
    assert_eq!(canonical(&a), r#"{"a":[1,{"y":null,"z":0}],"b":1.5}"#);
}

fn tmp(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("bekiper-stage-share-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn seal_is_bound_to_the_fingerprint() {
    let root = tmp("seal").join("output");
    let config = config();
    let layout = SharedLayout::new("rec", fingerprints(&experiment(), &config));
    assert!(!layout.is_sealed(&root, Shared::Pose));
    layout.seal(&root, Shared::Pose, false).unwrap();
    assert!(layout.is_sealed(&root, Shared::Pose));

    // Same directory name prefix but another fingerprint is not sealed.
    let mut changed_config = config.clone();
    changed_config.gpu = toml::from_str("device_id = 7").unwrap();
    let other = SharedLayout::new("rec", fingerprints(&experiment(), &changed_config));
    assert_ne!(
        layout.dir(&root, Shared::Pose),
        other.dir(&root, Shared::Pose)
    );
    assert!(!other.is_sealed(&root, Shared::Pose));
    layout.unseal(&root, Shared::Pose);
    assert!(!layout.is_sealed(&root, Shared::Pose));
}

#[test]
fn materialize_links_outputs_and_invalidates_stale_behaviour() {
    let root = tmp("materialize").join("output");
    let config = config();
    let layout = SharedLayout::new("rec", fingerprints(&experiment(), &config));
    let view_a = ExperimentOutputDir::new(&root, "rec__a").shared_view(layout.clone());
    view_a.ensure_dir().unwrap();
    std::fs::write(view_a.poses_parquet(), b"poses-v1").unwrap();
    layout.seal(&root, Shared::Pose, false).unwrap();

    let legacy_b = ExperimentOutputDir::new(&root, "rec__b");
    let view_b = legacy_b.shared_view(layout);
    materialize(&view_b, &legacy_b, &[Shared::Pose]).unwrap();
    assert_eq!(
        std::fs::read(legacy_b.poses_parquet()).unwrap(),
        b"poses-v1"
    );
    assert_eq!(
        read_marker(legacy_b.root()).unwrap().consumed["pose"].fingerprint,
        view_b
            .shared_layout()
            .unwrap()
            .fingerprint(Shared::Pose)
            .hex
    );

    // Upstream parameters change: the new checkpoint replaces the link and the
    // variant's behaviour output (computed from the old one) is removed.
    std::fs::write(legacy_b.behaviour_parquet(), b"behaviour-v1").unwrap();
    let mut changed_config = config.clone();
    changed_config.gpu = toml::from_str("device_id = 7").unwrap();
    let layout2 = SharedLayout::new("rec", fingerprints(&experiment(), &changed_config));
    let view2 = legacy_b.shared_view(layout2);
    view2.ensure_dir().unwrap();
    std::fs::write(view2.poses_parquet(), b"poses-v2").unwrap();
    materialize(&view2, &legacy_b, &[Shared::Pose]).unwrap();
    assert_eq!(
        std::fs::read(legacy_b.poses_parquet()).unwrap(),
        b"poses-v2"
    );
    assert!(!legacy_b.behaviour_parquet().exists());
}

#[test]
fn legacy_checkpoint_is_adopted_but_marked_layouts_are_not() {
    let root = tmp("adopt").join("output");
    let layout = SharedLayout::new("rec", fingerprints(&experiment(), &config()));
    let legacy = ExperimentOutputDir::new(&root, "rec__a");
    legacy.ensure_dir().unwrap();
    legacy.ensure_checkpoints_dir().unwrap();
    std::fs::write(legacy.poses_parquet(), b"legacy-poses").unwrap();

    let view = ExperimentOutputDir::new(&root, "rec__b").shared_view(layout.clone());
    assert!(adopt_legacy(&view, &[&legacy], Shared::Pose).unwrap());
    assert_eq!(
        std::fs::read(view.poses_parquet()).unwrap(),
        b"legacy-poses"
    );
    assert!(layout.is_sealed(&root, Shared::Pose));

    // A directory that already consumes shared outputs is not legacy.
    let consumer = ExperimentOutputDir::new(&root, "rec__c");
    consumer.ensure_dir().unwrap();
    std::fs::write(
        consumer.root().join(MARKER_FILE),
        "recording = \"rec\"\n[consumed.pose]\nfingerprint = \"x\"\ncheckpoint = \"y\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(consumer.checkpoints_dir()).unwrap();
    std::fs::write(consumer.poses_parquet(), b"linked").unwrap();
    let other = ExperimentOutputDir::new(&root, "rec__d").shared_view(SharedLayout::new(
        "other",
        fingerprints(&experiment(), &config()),
    ));
    assert!(!adopt_legacy(&other, &[&consumer], Shared::Pose).unwrap());
}
