use super::*;
use bekiper_core::{
    inspection::{InspectionManifest, PerimetersFile},
    scale::ScaleConfig,
};
use polars::prelude::*;

#[test]
fn cropped_nullable_inverted_recorded_clock_trials_match_staged_analysis() {
    for task in ["y_maze", "open_field", "object_recognition"] {
        let dir =
            std::env::temp_dir().join(format!("bekiper-alignment-{task}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut config: IngressConfig = toml::from_str(&format!(
            r#"
            [project]
            project_name = "alignment"
            data_directory = "."
            [pipeline]
            crop_start_seconds = 1.0
            crop_end_seconds = 1.0
            likelihood_trim = true
            smoothing_window = 1
            smoothing_method = "kalman"
            kalman_max_gap_frames = 0
            max_velocity = 10000.0
            invert_y = true
            [[experiments]]
            name = "13_recording__compare_0pct_dwell0s"
            task_type = "{task}"
            fps = 2.0
            meters_per_pixel = 0.001
            [experiments.task_params.ymaze]
            start_mode = "video_start"
            proximal_threshold = 0.0
            zone_max_gap_seconds = 0.0
            pose_gap_fill_seconds = 0.0
            arm_membership_hysteresis = false
            [experiments.task_params.ymaze.temporal_policy]
            min_duration_seconds = 0.0
            [experiments.task_params.exploration]
            window_mode = "configured_full_trial"
            speed_filter_mode = "disabled"
        "#
        ))
        .unwrap();
        let experiment = &mut config.experiments[0];
        let source_image = dir.join("source.png");
        image::RgbImage::new(400, 240).save(&source_image).unwrap();
        experiment.frame_image = Some(source_image);
        let experiment = experiment.clone();
        let geometry = if task == "y_maze" {
            vec![PerimeterSpec::RadialMaze {
                label: "maze".into(),
                center_vertices: vec![(160., 60.), (200., 60.), (200., 100.), (160., 100.)],
                arms: vec![
                    vec![(200., 60.), (340., 60.), (340., 100.), (200., 100.)],
                    vec![(160., 10.), (200., 10.), (200., 60.), (160., 60.)],
                    vec![(20., 60.), (160., 60.), (160., 100.), (20., 100.)],
                ],
            }]
        } else if task == "object_recognition" {
            vec![
                PerimeterSpec::Rectangle {
                    label: "arena".into(),
                    center_x: 200.,
                    center_y: 80.,
                    width: 380.,
                    height: 140.,
                },
                PerimeterSpec::Circle {
                    label: "object_1".into(),
                    center_x: 310.,
                    center_y: 100.,
                    radius: 10.,
                },
                PerimeterSpec::Circle {
                    label: "object_2".into(),
                    center_x: 100.,
                    center_y: 80.,
                    radius: 10.,
                },
            ]
        } else {
            vec![PerimeterSpec::Rectangle {
                label: "arena".into(),
                center_x: 200.,
                center_y: 80.,
                width: 380.,
                height: 140.,
            }]
        };
        let (_, scale) = ScaleConfig::Manual(0.001)
            .resolve_with_info(&geometry)
            .unwrap();
        let labels = [
            "nose",
            "center_ear",
            "tail_base",
            "mouse_center",
            "left_ear",
            "right_ear",
        ]
        .map(str::to_owned);
        let mut columns = vec![
            Column::new("frame".into(), (0..20_u64).collect::<Vec<_>>()),
            Column::new(
                "elapsed_s".into(),
                (0..20)
                    .map(|i| i as f64 * 0.5 + (i / 5) as f64 * 0.05)
                    .collect::<Vec<_>>(),
            ),
        ];
        for label in &labels {
            columns.extend([
                Column::new(
                    format!("{label}_x").into(),
                    vec![Some(if label == "nose" { 300. } else { 280. }); 20],
                ),
                Column::new(
                    format!("{label}_y").into(),
                    (0..20)
                        .map(|i| {
                            if i == 9 {
                                None
                            } else {
                                Some(if label == "nose" { 90. } else { 80. })
                            }
                        })
                        .collect::<Vec<_>>(),
                ),
                Column::new(
                    format!("{label}_likelihood").into(),
                    (0..20)
                        .map(|i| if i <= 2 || i >= 17 { 0. } else { 1. })
                        .collect::<Vec<_>>(),
                ),
            ]);
        }
        let mut raw = DataFrame::new(20, columns).unwrap();
        let direct = analyze_loaded_pose(
            &experiment,
            &config.pipeline,
            None,
            &geometry,
            &scale,
            raw.clone(),
            &labels,
        )
        .unwrap();
        let evaluated = direct.result.evaluated_df.as_ref().unwrap();
        assert_eq!(
            evaluated
                .column("frame")
                .unwrap()
                .u64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            (3..17).collect::<Vec<_>>()
        );
        assert!(
            evaluated
                .column("center_ear_y")
                .unwrap()
                .f64()
                .unwrap()
                .get(6)
                .is_none()
        );
        assert!(
            (evaluated
                .column("__bk_stage__raw_pose__nose_y")
                .unwrap()
                .f64()
                .unwrap()
                .get(0)
                .unwrap()
                - 0.15)
                .abs()
                < 1e-12
        );
        let display =
            crate::inspection::source_overlay_dataframe(&direct.manifest, evaluated.clone())
                .unwrap();
        assert!(
            (display
                .column("nose_y")
                .unwrap()
                .f64()
                .unwrap()
                .get(0)
                .unwrap()
                - 0.09)
                .abs()
                < 1e-12
        );
        assert!(
            (evaluated
                .column("snout_dy")
                .unwrap()
                .f64()
                .unwrap()
                .get(0)
                .unwrap()
                + 0.01)
                .abs()
                < 1e-12
        );
        if task == "y_maze" {
            assert!(
                evaluated
                    .column("in_arm_A")
                    .unwrap()
                    .bool()
                    .unwrap()
                    .get(0)
                    .unwrap()
            );
        }
        let output = ExperimentOutputDir::new(&dir, &experiment.name);
        output.ensure_dir().unwrap();
        output.ensure_checkpoints_dir().unwrap();
        std::fs::write(
            output.pixel_perimeters_toml(),
            toml::to_string_pretty(&PerimetersFile {
                perimeters: geometry,
            })
            .unwrap(),
        )
        .unwrap();
        bekiper_pose::io::write_parquet(&mut raw, &output.poses_parquet()).unwrap();
        augment::run_augment(&experiment, &config, &output).unwrap();
        behaviour::run_behaviour(&experiment, &output).unwrap();
        summary::run_summary(&experiment, &config, &output).unwrap();
        let staged_df = bekiper_pose::io::scan_file(&output.evaluated_parquet())
            .unwrap()
            .collect()
            .unwrap();
        assert!(evaluated.equals_missing(&staged_df));
        let staged_manifest = InspectionManifest::from_toml(&output.inspection_toml()).unwrap();
        assert_eq!(
            serde_json::to_value(&direct.manifest).unwrap(),
            serde_json::to_value(staged_manifest).unwrap()
        );
        let staged_trial = bekiper_core::inspection::toml_str_to_json_value(
            &std::fs::read_to_string(output.result_toml()).unwrap(),
        )
        .unwrap();
        // TOML omits null fields at its export boundary.
        let direct_trial = bekiper_core::inspection::toml_str_to_json_value(
            &bekiper_core::inspection::json_value_to_toml_string(
                &serde_json::to_value(&direct.trial).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(staged_trial, direct_trial);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn source_video_clock_is_projected_by_keys_before_crop_and_bad_clocks_fail() {
    let mut experiment: ExperimentConfig =
        toml::from_str("name='clock'\ntask_type='y_maze'\nfps=2.0\nmeters_per_pixel=0.001")
            .unwrap();
    experiment.frame_timing = Some(bekiper_core::video::FrameTiming {
        fps: 2.,
        is_vfr: true,
        timestamps: (0..20)
            .map(|i| i as f64 * 0.5 + (i / 4) as f64 * 0.05)
            .collect(),
    });
    let source = df! {"frame" => [5_u64,6,7], "x" => [1.,2.,3.]}.unwrap();
    let aligned = augment::attach_source_clock(source.lazy(), &experiment)
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        aligned
            .column("elapsed_s")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        vec![2.55, 3.05, 3.55]
    );
    let (timing, fps) =
        behaviour::resolve_behaviour_timing_and_fps(&experiment, &aligned.lazy()).unwrap();
    assert_eq!(fps, 2.);
    assert_eq!(timing.timestamps, [2.55, 3.05, 3.55]);
    for clock in [
        vec![Some(1.), Some(1.)],
        vec![Some(1.), None],
        vec![Some(2.), Some(1.)],
    ] {
        let input = df! {"elapsed_s" => clock}.unwrap();
        assert!(behaviour::resolve_behaviour_timing_and_fps(&experiment, &input.lazy()).is_err());
    }
}

#[test]
fn retained_source_image_resolves_height_for_analysis_and_inspection() {
    let dir = std::env::temp_dir().join(format!("bekiper-source-height-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let image = dir.join("source.png");
    image::RgbImage::new(400, 240).save(&image).unwrap();
    let mut experiment: ExperimentConfig = toml::from_str(
        "name='source_image'\ntask_type='open_field'\nfps=30.0\nmeters_per_pixel=0.001",
    )
    .unwrap();
    experiment.frame_image = Some(image);
    assert_eq!(
        resolve_source_resolution(&experiment).unwrap(),
        Some((400, 240))
    );
    experiment.video_resolution = Some((400, 0));
    assert!(resolve_source_resolution(&experiment).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "pose-video")]
#[test]
fn native_preparation_detects_the_nort_box_and_replaces_the_shared_scale() {
    let root = std::env::temp_dir().join(format!("bekiper-nort-box-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let video = root.join("nort.mp4");
    // 232x216 video: dark walls framing a lighter floor. Exact per-axis box
    // geometry is covered on raw frames in bekiper-detect; this checks wiring.
    let status = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("color=c=0x141414:size=232x216:rate=10:duration=1,drawbox=x=20:y=8:w=200:h=200:c=0xbebebe:t=fill")
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&video)
        .status()
        .unwrap();
    assert!(status.success());
    let mut experiment: crate::config::ExperimentConfig = toml::from_str(&format!(
        r#"
name = "nort"
task_type = "object_recognition"
fps = 10.0
meters_per_pixel = 0.002
video = "{}"
[detection]
backend = ["manual_annotation"]
[detection.arena_calibration]
search_margin_fraction = 0.0
center_zone_rule = {{ method = "frame_centered", arena_width_meters = 0.40, arena_height_meters = 0.40, center_width_meters = 0.20, center_height_meters = 0.20, detect_box = true }}
"#,
        video.display()
    ))
    .unwrap();
    let mut perimeters = vec![PerimeterSpec::Rectangle {
        label: "familiar".into(),
        center_x: 60.0,
        center_y: 60.0,
        width: 20.0,
        height: 20.0,
    }];
    let shared = ScaleConfig::Manual(0.002)
        .resolve_with_info(&perimeters)
        .unwrap()
        .1;
    let mut scale = Some(shared);
    prepare_video_inputs(&mut experiment, &mut perimeters, &mut scale).unwrap();
    let scale = scale.unwrap();
    assert_eq!(scale.method, "arena_box_detected");
    let arena = perimeters.iter().find(|p| p.label() == "arena").unwrap();
    let PerimeterSpec::Rectangle {
        center_x,
        width,
        height,
        ..
    } = arena
    else {
        panic!("expected arena rectangle");
    };
    assert!((center_x - 120.0).abs() <= 3.0, "box center {center_x}");
    assert!((scale.meters_per_pixel - 0.40 / width).abs() < 1e-12);
    assert!((scale.meters_per_pixel_y() - 0.40 / height).abs() < 1e-12);
    // The vertical scale is recorded only for a non-square box.
    assert_eq!(
        scale.meters_per_pixel_y.is_some(),
        width != height,
        "{width}x{height}"
    );
    assert!(
        (scale.meters_per_pixel - 0.002).abs() < 0.0002,
        "{}",
        scale.meters_per_pixel
    );
    assert!(perimeters.iter().any(|p| p.label() == "center_zone"));
    std::fs::remove_dir_all(root).unwrap();
}

const LABELS: [&str; 7] = [
    "nose",
    "center_ear",
    "tail_base",
    "base_tail",
    "mouse_center",
    "left_ear",
    "right_ear",
];

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("bekiper-analysis-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Pose frame with every label at `position(frame)` pixels (nose offset +10 x).
fn pose_frame(frames: usize, position: impl Fn(usize) -> (f64, f64)) -> DataFrame {
    let mut columns = vec![Column::new(
        "frame".into(),
        (0..frames as u64).collect::<Vec<_>>(),
    )];
    for label in LABELS {
        let dx = if label == "nose" { 10.0 } else { 0.0 };
        columns.extend([
            Column::new(
                format!("{label}_x").into(),
                (0..frames).map(|i| position(i).0 + dx).collect::<Vec<_>>(),
            ),
            Column::new(
                format!("{label}_y").into(),
                (0..frames).map(|i| position(i).1).collect::<Vec<_>>(),
            ),
            Column::new(format!("{label}_likelihood").into(), vec![1.0; frames]),
        ]);
    }
    DataFrame::new(frames, columns).unwrap()
}

fn labels() -> Vec<String> {
    LABELS.map(str::to_owned).to_vec()
}

fn arena() -> Vec<PerimeterSpec> {
    vec![PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 200.,
        center_y: 80.,
        width: 380.,
        height: 140.,
    }]
}

fn manual_scale(perimeters: &[PerimeterSpec]) -> ScaleDerivationInfo {
    ScaleConfig::Manual(0.001)
        .resolve_with_info(perimeters)
        .unwrap()
        .1
}

fn oft_experiment(name: &str) -> ExperimentConfig {
    toml::from_str(&format!(
        "name='{name}'\ntask_type='open_field'\nfps=2.0\nmeters_per_pixel=0.001"
    ))
    .unwrap()
}

#[test]
fn direct_analysis_rejects_unsafe_names_scales_and_empty_frames() {
    let pipeline = crate::config::PipelineConfig::default();
    let geometry = arena();
    let scale = manual_scale(&geometry);
    let bad_scale = ScaleDerivationInfo {
        meters_per_pixel: f64::NAN,
        ..scale.clone()
    };
    let raw = pose_frame(4, |i| (100.0 + i as f64, 80.0));
    let good = oft_experiment("trial");
    let unsafe_name = oft_experiment("..");
    let err = |result: Result<ExperimentAnalysis>| result.err().unwrap().to_string();
    let run = |experiment: &ExperimentConfig, scale: &ScaleDerivationInfo, raw: DataFrame| {
        analyze_loaded_pose(
            experiment,
            &pipeline,
            None,
            &geometry,
            scale,
            raw,
            &labels(),
        )
    };
    assert!(err(run(&unsafe_name, &scale, raw.clone())).contains("unsafe experiment output name"));
    assert!(err(run(&good, &bad_scale, raw.clone())).contains("finite and positive"));
    assert!(err(run(&good, &scale, raw.clear())).contains("pose input is empty"));
    assert!(
        augment_loaded_pose(&good, &pipeline, &labels(), &geometry, &bad_scale, &raw)
            .unwrap_err()
            .to_string()
            .contains("finite and positive")
    );
    let staged = |experiment: &ExperimentConfig, augmented: DataFrame| {
        analyze_augmented_pose_with_pipeline(
            experiment,
            &pipeline,
            None,
            &geometry,
            &scale,
            DataFrame::empty(),
            augmented,
        )
    };
    assert!(err(staged(&unsafe_name, raw.clone())).contains("unsafe experiment output name"));
    assert!(err(staged(&good, raw.clear())).contains("augmentation removed every frame"));
    assert!(
        metric_perimeters(&good, &pipeline, &geometry, &bad_scale)
            .unwrap_err()
            .to_string()
            .contains("finite and positive")
    );
    let inverted = crate::config::PipelineConfig {
        invert_y: true,
        ..Default::default()
    };
    assert!(
        metric_perimeters(&good, &inverted, &geometry, &scale)
            .unwrap_err()
            .to_string()
            .contains("Y inversion needs source resolution")
    );
    let config: IngressConfig =
        toml::from_str("[project]\nproject_name='p'\ndata_directory='.'").unwrap();
    assert!(err(analyze_experiment(&unsafe_name, &config, &geometry, &scale)).contains("unsafe"));
    assert!(
        err(analyze_experiment(&good, &config, &geometry, &scale))
            .contains("exactly one precomputed pose source")
    );
    let unreadable = ExperimentConfig {
        recording_id: None,
        frame_image: Some("/nonexistent/source.png".into()),
        ..good.clone()
    };
    assert!(
        resolve_source_resolution(&unreadable)
            .unwrap_err()
            .to_string()
            .contains("cannot read source image dimensions")
    );
    assert!(!is_analysis_excluded(&good));
}

#[test]
fn direct_analysis_of_a_source_file_exports_every_artifact() {
    let dir = scratch("export");
    let source = dir.join("7_oft.parquet");
    let mut raw = pose_frame(20, |i| (100.0 + 8.0 * i as f64, 60.0 + 2.0 * i as f64));
    bekiper_pose::io::write_parquet(&mut raw, &source).unwrap();
    let mut config: IngressConfig =
        toml::from_str("[project]\nproject_name='p'\ndata_directory='.'").unwrap();
    config.animal_metadata.insert(
        7,
        crate::config::AnimalMetadata {
            animal: 7,
            cohort: "c1".into(),
            gene: "wt".into(),
            sex: "f".into(),
            age: "12w".into(),
        },
    );
    let geometry = arena();
    let scale = manual_scale(&geometry);
    let mut experiment = oft_experiment("7_oft");
    experiment.data_files = vec![source];
    let analysis = analyze_experiment(&experiment, &config, &geometry, &scale).unwrap();
    assert_eq!(analysis.trial.animal_metadata["animal"], 7);
    let trial = export_analysis(&dir.join("flat"), &experiment, analysis).unwrap();
    let output = ExperimentOutputDir::new(&dir.join("flat"), "7_oft");
    assert!(output.result_toml().is_file());
    assert!(output.spreadsheet_workbook().is_file());
    assert_eq!(trial.experiment, "7_oft");

    // Staged-style inputs carry no raw pose; the animal layout and the
    // presentation wrapper still export the scored trial.
    let augmented = augment_loaded_pose(
        &experiment,
        &config.pipeline,
        &labels(),
        &geometry,
        &scale,
        &raw,
    )
    .unwrap();
    let analysis = analyze_augmented_pose(
        &experiment,
        None,
        &geometry,
        &scale,
        DataFrame::empty(),
        augmented,
    )
    .unwrap();
    experiment.animal_id = Some(7);
    let settings = crate::config::InspectConfig {
        enabled: false,
        ..Default::default()
    };
    export_analysis_with_inspection(
        &dir.join("animals"),
        &experiment,
        analysis,
        &settings,
        &|| false,
    )
    .unwrap();
    let output = ExperimentOutputDir::for_animal(&dir.join("animals"), 7, "7_oft");
    assert!(output.result_toml().is_file());
    assert!(output.root().join("trial_context.json").is_file());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ymaze_scoring_rejects_movement_that_never_enters_an_arm() {
    let mut experiment: ExperimentConfig = toml::from_str(
        r#"
        name = "ymaze_center"
        task_type = "y_maze"
        fps = 2.0
        meters_per_pixel = 0.001
        [task_params.ymaze]
        start_mode = "video_start"
        proximal_threshold = 0.0
        zone_max_gap_seconds = 0.0
        pose_gap_fill_seconds = 0.0
        arm_membership_hysteresis = false
        [task_params.ymaze.temporal_policy]
        min_duration_seconds = 0.0
        "#,
    )
    .unwrap();
    experiment.video_resolution = Some((400, 240));
    let geometry = vec![PerimeterSpec::RadialMaze {
        label: "maze".into(),
        center_vertices: vec![(160., 60.), (200., 60.), (200., 100.), (160., 100.)],
        arms: vec![
            vec![(200., 60.), (340., 60.), (340., 100.), (200., 100.)],
            vec![(160., 10.), (200., 10.), (200., 60.), (160., 60.)],
            vec![(20., 60.), (160., 60.), (160., 100.), (20., 100.)],
        ],
    }];
    let scale = manual_scale(&geometry);
    // 12.5 s in the center: meaningful duration, no arm ever entered.
    let raw = pose_frame(25, |i| (170.0 + (i % 2) as f64, 80.0));
    let error = crate::traced(|| {
        analyze_loaded_pose(
            &experiment,
            &Default::default(),
            None,
            &geometry,
            &scale,
            raw,
            &labels(),
        )
    })
    .err()
    .unwrap()
    .to_string();
    for expected in [
        "invalid Y-maze source summary",
        "zero arms: A, B, C",
        "boolean true frames",
        "center_ear bbox",
        "base_tail bbox",
        "center bbox",
        "arm_C bbox",
    ] {
        assert!(error.contains(expected), "{expected}: {error}");
    }
}

#[test]
fn generated_arena_requires_complete_outlines_and_a_frame_size() {
    let mut experiment: ExperimentConfig = toml::from_str(
        r#"
        name = "nort"
        task_type = "object_recognition"
        fps = 10.0
        meters_per_pixel = 0.002
        [detection]
        backend = ["manual_annotation"]
        [detection.arena_calibration]
        center_zone_rule = { method = "frame_centered", arena_width_meters = 0.40, arena_height_meters = 0.40, center_width_meters = 0.20, center_height_meters = 0.20 }
        "#,
    )
    .unwrap();
    let rect = |label: &str| PerimeterSpec::Rectangle {
        label: label.into(),
        center_x: 50.0,
        center_y: 50.0,
        width: 20.0,
        height: 20.0,
    };
    let scale = manual_scale(&[]);
    let mut reviewed = vec![rect("arena"), rect("center_zone")];
    assert!(
        append_generated_arena(&experiment, &mut reviewed, Some(&scale))
            .unwrap()
            .is_none()
    );
    assert_eq!(reviewed.len(), 2, "reviewed outlines are kept as-is");
    let error = append_generated_arena(&experiment, &mut vec![rect("arena")], Some(&scale))
        .unwrap_err()
        .to_string();
    assert!(error.contains("both or neither"), "{error}");
    let error = append_generated_arena(&experiment, &mut Vec::new(), Some(&scale))
        .unwrap_err()
        .to_string();
    assert!(error.contains("needs the video frame size"), "{error}");
    experiment.detection = None;
    assert!(
        append_generated_arena(&experiment, &mut Vec::new(), Some(&scale))
            .unwrap()
            .is_none()
    );
    assert_eq!(
        arena_calibration_sample_count(&experiment),
        bekiper_detect::config::default_arena_sample_positions().len()
    );
    let configured: crate::config::DetectionConfig =
        toml::from_str("[arena_calibration]\nsample_positions = [0.25, 0.75]").unwrap();
    experiment.detection = Some(configured);
    assert_eq!(arena_calibration_sample_count(&experiment), 2);
}

#[test]
fn staged_trial_scoring_verifies_every_prerequisite() {
    use crate::preparation::{HashedFile, ReuseTrial};
    let dir = scratch("staged");
    let geometry = arena();
    let scale = manual_scale(&geometry);
    let experiment = oft_experiment("staged");
    let pipeline = crate::config::PipelineConfig::default();
    let raw = pose_frame(20, |i| (100.0 + 8.0 * i as f64, 80.0));
    let mut augmented =
        augment_loaded_pose(&experiment, &pipeline, &labels(), &geometry, &scale, &raw).unwrap();
    let metric = metric_perimeters(&experiment, &pipeline, &geometry, &scale).unwrap();
    let write = |name: &str, text: String| {
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        path
    };
    let augmented_path = dir.join("augmented.parquet");
    bekiper_pose::io::write_parquet(&mut augmented, &augmented_path).unwrap();
    let paths = [
        ("augmented", augmented_path),
        (
            "pixel_perimeters",
            write(
                "pixel.toml",
                bekiper_core::handoff::to_string(&PerimetersFile {
                    perimeters: geometry.clone(),
                })
                .unwrap(),
            ),
        ),
        (
            "perimeters",
            write(
                "metric.toml",
                bekiper_core::handoff::to_string(&PerimetersFile {
                    perimeters: metric.clone(),
                })
                .unwrap(),
            ),
        ),
        (
            "scale",
            write(
                "scale.toml",
                bekiper_core::handoff::to_string(&scale).unwrap(),
            ),
        ),
        (
            "frame_timing",
            write(
                "timing.toml",
                bekiper_core::handoff::to_string(&bekiper_core::video::FrameTiming::cfr(2.0))
                    .unwrap(),
            ),
        ),
    ];
    let trial = ReuseTrial {
        analysis_id: "staged".into(),
        preparation_fingerprint: "fingerprint".into(),
        prerequisites: paths
            .iter()
            .map(|(key, path)| ((*key).into(), HashedFile::capture(path).unwrap()))
            .collect(),
    };
    let analysis = staged::analyze_trial(&experiment, &pipeline, None, &trial).unwrap();
    assert_eq!(analysis.raw.height(), 0);
    assert_eq!(analysis.trial.experiment, "staged");

    let fails = |trial: &ReuseTrial, expected: &str| {
        let error = staged::analyze_trial(&experiment, &pipeline, None, trial)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(expected), "{expected}: {error}");
    };
    fails(
        &ReuseTrial {
            analysis_id: "other".into(),
            ..trial.clone()
        },
        "identity differs",
    );
    let mut fewer = trial.clone();
    fewer.prerequisites.remove("scale");
    fails(&fewer, "prerequisite set differs");
    let mut renamed = fewer.clone();
    renamed
        .prerequisites
        .insert("bogus".into(), trial.prerequisites["scale"].clone());
    fails(&renamed, "missing staged prerequisite: scale");
    let shifted = PerimetersFile {
        perimeters: geometry.iter().map(|p| p.scale_xy(0.002, 0.002)).collect(),
    };
    let mut mismatched = trial.clone();
    mismatched.prerequisites.insert(
        "perimeters".into(),
        HashedFile::capture(&write(
            "shifted.toml",
            bekiper_core::handoff::to_string(&shifted).unwrap(),
        ))
        .unwrap(),
    );
    fails(&mismatched, "differ from prepared pixel/scale projection");
    let empty = dir.join("empty.parquet");
    bekiper_pose::io::write_parquet(&mut augmented.clear(), &empty).unwrap();
    let mut emptied = trial.clone();
    emptied
        .prerequisites
        .insert("augmented".into(), HashedFile::capture(&empty).unwrap());
    fails(&emptied, "augmentation removed every frame");
    std::fs::write(dir.join("scale.toml"), "changed").unwrap();
    fails(&trial, "reuse evidence bytes differ");
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(feature = "pose-video")]
#[test]
fn oft_video_preparation_calibrates_the_arena_and_probes_the_clock() {
    let dir = scratch("oft-calibration");
    let video = dir.join("oft.mp4");
    // Light floor with a dark wall outline, as on raw frames in bekiper-detect.
    let status = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("color=c=0xbebebe:size=120x120:rate=10:duration=1,drawbox=x=20:y=22:w=81:h=81:c=0x141414:t=2")
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&video)
        .status()
        .unwrap();
    assert!(status.success());
    let mut experiment: ExperimentConfig = toml::from_str(&format!(
        "name='oft'\ntask_type='open_field'\nvideo='{}'\nmeters_per_pixel={{ method = 'arena_side', meters = 0.40 }}",
        video.display()
    ))
    .unwrap();
    assert_eq!(
        resolve_source_resolution(&experiment).unwrap(),
        Some((120, 120))
    );
    let mut perimeters = Vec::new();
    let mut scale = None;
    prepare_video_inputs(&mut experiment, &mut perimeters, &mut scale).unwrap();
    assert_eq!(experiment.video_resolution, Some((120, 120)));
    assert_eq!(
        experiment.frame_timing.as_ref().unwrap().timestamps.len(),
        10
    );
    assert_eq!(perimeters[0].label(), "arena");
    assert!(
        scale.is_none(),
        "a calibrated OFT arena keeps the configured scale"
    );
    // Exported trials with a source video carry their representative frames.
    let analysis = analyze_loaded_pose(
        &experiment,
        &Default::default(),
        None,
        &perimeters,
        &manual_scale(&perimeters),
        pose_frame(6, |i| (40.0 + 5.0 * i as f64, 60.0)),
        &labels(),
    )
    .unwrap();
    crate::traced(|| export_analysis(&dir.join("run"), &experiment, analysis)).unwrap();
    assert!(
        ExperimentOutputDir::new(&dir.join("run"), "oft")
            .frames_dir()
            .read_dir()
            .unwrap()
            .next()
            .is_some()
    );

    // Calibration needs an OFT video with an arena_side scale.
    let mut wrong = experiment.clone();
    wrong.meters_per_pixel = ScaleConfig::Manual(0.001);
    for (experiment, expected) in [
        (
            ExperimentConfig {
                video: None,
                ..experiment.clone()
            },
            "requires video",
        ),
        (
            ExperimentConfig {
                task_type: bekiper_core::types::TaskType::YMaze,
                ..experiment.clone()
            },
            "requires open_field",
        ),
        (wrong, "requires arena_side scale"),
    ] {
        let error = calibrate_arena_source(&experiment)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(expected), "{expected}: {error}");
    }
    // Without a video there is nothing to prepare.
    let mut no_video = oft_experiment("none");
    prepare_video_inputs(&mut no_video, &mut Vec::new(), &mut None).unwrap();
    assert!(no_video.video_resolution.is_none());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn analysis_errors_propagate_from_augmentation_scoring_and_export() {
    let geometry = arena();
    let scale = manual_scale(&geometry);
    let experiment = oft_experiment("propagate");
    let raw = pose_frame(6, |i| (100.0 + 8.0 * i as f64, 80.0));
    let error = analyze_loaded_pose(
        &experiment,
        &Default::default(),
        None,
        &geometry,
        &scale,
        raw.clone(),
        &[],
    )
    .err()
    .unwrap()
    .to_string();
    assert!(error.contains("pose input has no labels"), "{error}");

    // An existing directory where an export file belongs fails the export.
    let dir = scratch("export-errors");
    let analysis = |raw: DataFrame| {
        analyze_loaded_pose(
            &experiment,
            &Default::default(),
            None,
            &geometry,
            &scale,
            raw,
            &labels(),
        )
        .unwrap()
    };
    let output = ExperimentOutputDir::new(&dir, "propagate");
    output.ensure_dir().unwrap();
    let inspection = output.root().join("propagate_inspection.toml");
    std::fs::create_dir_all(&inspection).unwrap();
    assert!(export_analysis(&dir, &experiment, analysis(raw.clone())).is_err());
    std::fs::remove_dir_all(&inspection).unwrap();
    std::fs::create_dir_all(output.spreadsheet_workbook()).unwrap();
    assert!(export_analysis(&dir, &experiment, analysis(raw.clone())).is_err());
    std::fs::remove_dir_all(output.spreadsheet_workbook()).unwrap();
    let settings = crate::config::InspectConfig {
        enabled: false,
        ..Default::default()
    };
    export_analysis_with_inspection(&dir, &experiment, analysis(raw), &settings, &|| false)
        .unwrap();
    assert!(output.result_toml().is_file());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ymaze_validation_skips_other_details_and_reports_without_behaviour_frames() {
    let experiment: ExperimentConfig = toml::from_str(
        r#"
        name = "ymaze_direct"
        task_type = "y_maze"
        fps = 2.0
        meters_per_pixel = 0.001
        [task_params.ymaze]
        start_mode = "video_start"
        proximal_threshold = 0.0
        zone_max_gap_seconds = 0.0
        pose_gap_fill_seconds = 0.0
        arm_membership_hysteresis = false
        [task_params.ymaze.temporal_policy]
        min_duration_seconds = 0.0
        "#,
    )
    .unwrap();
    let geometry = vec![PerimeterSpec::RadialMaze {
        label: "maze".into(),
        center_vertices: vec![(160., 60.), (200., 60.), (200., 100.), (160., 100.)],
        arms: vec![
            vec![(200., 60.), (340., 60.), (340., 100.), (200., 100.)],
            vec![(160., 10.), (200., 10.), (200., 60.), (160., 60.)],
            vec![(20., 60.), (160., 60.), (160., 100.), (20., 100.)],
        ],
    }];
    let scale = manual_scale(&geometry);
    let pipeline = crate::config::PipelineConfig::default();
    let raw = pose_frame(25, |i| (170.0 + (i % 2) as f64, 80.0));
    let augmented =
        augment_loaded_pose(&experiment, &pipeline, &labels(), &geometry, &scale, &raw).unwrap();
    let metric = metric_perimeters(&experiment, &pipeline, &geometry, &scale).unwrap();
    let (_, detailed, _) = behaviour::analyze_pose(&experiment, &metric, augmented.lazy()).unwrap();
    // Without behaviour frames or a radial maze, only arm totals are reported.
    let rect = vec![arena()[0].clone()];
    let error = summary::validate_ymaze_result(&experiment, detailed.as_ref(), &rect, None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("arm durations") && !error.contains("bbox"),
        "{error}"
    );
    summary::validate_ymaze_result(&experiment, None, &rect, None).unwrap();
    assert_eq!(
        summary::variant_analysis_role(&ExperimentConfig {
            name: "__proximal25".into(),
            ..experiment.clone()
        }),
        None
    );
}

#[test]
fn single_frame_nort_analysis_samples_one_skeleton_and_binds_the_centroid() {
    let experiment: ExperimentConfig = toml::from_str(
        r#"
        name = "nort_single"
        task_type = "object_recognition"
        fps = 2.0
        meters_per_pixel = 0.001
        [task_params.exploration]
        window_mode = "configured_full_trial"
        speed_filter_mode = "disabled"
        "#,
    )
    .unwrap();
    let geometry = vec![
        PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 200.,
            center_y: 80.,
            width: 380.,
            height: 140.,
        },
        PerimeterSpec::Rectangle {
            label: "center_zone".into(),
            center_x: 200.,
            center_y: 80.,
            width: 190.,
            height: 70.,
        },
        PerimeterSpec::Circle {
            label: "object_1".into(),
            center_x: 310.,
            center_y: 100.,
            radius: 10.,
        },
        PerimeterSpec::Circle {
            label: "object_2".into(),
            center_x: 100.,
            center_y: 80.,
            radius: 10.,
        },
    ];
    let scale = manual_scale(&geometry);
    let analysis = analyze_loaded_pose(
        &experiment,
        &Default::default(),
        None,
        &geometry,
        &scale,
        pose_frame(1, |_| (200.0, 80.0)),
        &labels(),
    )
    .unwrap();
    let features = &analysis.manifest.feature_inspections;
    assert!(features.iter().any(|f| f.label() == "skeleton_geometry_0"));
    assert!(features.iter().any(|f| matches!(
        f,
        bekiper_core::inspection::FeatureInspection::CenterPeriphery { coordinate_labels, .. }
            if coordinate_labels == &["mouse_center".to_string()]
    )));
}

#[test]
fn arena_sampling_defaults_when_no_positions_are_configured() {
    let config = bekiper_detect::config::ArenaCalibrationConfig {
        sample_positions: Vec::new(),
        ..Default::default()
    };
    assert_eq!(
        arena_calibration_frame_indices(101, &config).len(),
        bekiper_detect::config::default_arena_sample_positions().len()
    );
}

#[test]
fn staged_stages_report_missing_inputs_and_reuse_annotation_scale() {
    let dir = scratch("staged-stages");
    let mut config: IngressConfig = toml::from_str(
        "[project]\nproject_name='p'\ndata_directory='.'\n[pipeline]\ninvert_y = true",
    )
    .unwrap();
    let mut experiment = oft_experiment("staged_oft");
    experiment.video_resolution = Some((400, 240));
    let output = ExperimentOutputDir::new(&dir, "staged_oft");
    output.ensure_dir().unwrap();
    output.ensure_checkpoints_dir().unwrap();
    let error = |result: Result<()>| result.unwrap_err().to_string();
    assert!(
        error(behaviour::run_behaviour(&experiment, &output)).contains("run augment stage first")
    );
    assert!(
        error(augment::run_augment(&experiment, &config, &output)).contains("run pose stage first")
    );

    // Labels come from the DLC source file named by the experiment.
    let source = dir.join("source.csv");
    let header = |row: &str, value: &dyn Fn(&str) -> String| {
        std::iter::once(row.to_string())
            .chain(LABELS.iter().flat_map(|l| [value(l), value(l), value(l)]))
            .collect::<Vec<_>>()
            .join(",")
    };
    std::fs::write(
        &source,
        format!(
            "{}\n{}\n{}\n",
            header("scorer", &|_| "dlc".into()),
            header("bodyparts", &|l| l.to_string()),
            header("coords", &|_| "x".into()),
        ),
    )
    .unwrap();
    experiment.data_files = vec![source];
    std::fs::write(
        output.pixel_perimeters_toml(),
        toml::to_string_pretty(&PerimetersFile {
            perimeters: arena(),
        })
        .unwrap(),
    )
    .unwrap();
    let mut raw = pose_frame(6, |i| (100.0 + 8.0 * i as f64, 80.0));
    bekiper_pose::io::write_parquet(&mut raw, &output.poses_parquet()).unwrap();
    let mut annotated = manual_scale(&arena());
    annotated.method = "annotation_diagonal".into();
    std::fs::write(
        output.annotation_scale_toml(),
        toml::to_string_pretty(&annotated).unwrap(),
    )
    .unwrap();
    crate::traced(|| augment::run_augment(&experiment, &config, &output)).unwrap();
    let scale: ScaleDerivationInfo =
        toml::from_str(&std::fs::read_to_string(output.scale_toml()).unwrap()).unwrap();
    assert_eq!(scale.method, "annotation_diagonal");
    crate::traced(|| behaviour::run_behaviour(&experiment, &output)).unwrap();

    std::fs::write(output.behaviour_result_toml(), "= broken").unwrap();
    assert!(
        error(summary::run_summary(&experiment, &config, &output))
            .contains("invalid behaviour_result.toml")
    );
    std::fs::remove_file(output.behaviour_result_toml()).unwrap();
    let mut unresolved = experiment.clone();
    unresolved.video_resolution = None;
    assert!(
        error(summary::run_summary(&unresolved, &config, &output))
            .contains("raw tracking Y inversion needs source resolution")
    );
    config.pipeline.invert_y = false;
    let mut with_video = experiment.clone();
    let video = dir.join("empty.mp4");
    std::fs::write(&video, "").unwrap();
    with_video.video = Some(video);
    assert!(summary::run_summary(&with_video, &config, &output).is_err());
    std::fs::create_dir_all(output.spreadsheet_workbook()).unwrap();
    assert!(summary::run_summary(&experiment, &config, &output).is_err());
    std::fs::remove_dir_all(output.spreadsheet_workbook()).unwrap();
    summary::run_summary(&experiment, &config, &output).unwrap();
    std::fs::remove_file(output.behaviour_parquet()).unwrap();
    assert!(
        error(summary::run_summary(&experiment, &config, &output))
            .contains("requires evaluated behaviour data")
    );

    // Imen NORT habituation trials are never summarized.
    let mut habituation = experiment.clone();
    habituation.task_type = bekiper_core::types::TaskType::ObjectRecognition;
    habituation.session_context = Some(bekiper_core::inspection::IngressSessionContext {
        profile: "videomap".into(),
        stage: Some("0".into()),
        phase: Some("A".into()),
        ..Default::default()
    });
    crate::traced(|| summary::run_summary(&habituation, &config, &output)).unwrap();
    assert!(
        helpers::load_perimeters(&ExperimentOutputDir::new(&dir, "absent"))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        helpers::resolve_python(),
        std::path::PathBuf::from("python3")
    );
    let unlabeled = dir.join("unlabeled.parquet");
    bekiper_pose::io::write_parquet(&mut df! {"frame" => [0u64]}.unwrap(), &unlabeled).unwrap();
    assert!(
        helpers::discover_labels(&oft_experiment("x"), &unlabeled)
            .unwrap_err()
            .to_string()
            .contains("could not discover body-part labels")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn behaviour_timing_and_columns_are_validated_before_scoring() {
    use polars::prelude::IntoLazy;
    // A recorded clock too fine to give a finite rate falls back to manual fps.
    let instant = df! {"elapsed_s" => [0.0, 5e-324]}.unwrap().lazy();
    let (_, fps) =
        behaviour::resolve_behaviour_timing_and_fps(&oft_experiment("timing"), &instant).unwrap();
    assert_eq!(fps, 2.0);
    let empty = df! {"frame" => [0u64]}.unwrap().lazy();
    let mut unknown: ExperimentConfig =
        toml::from_str("name='unknown'\ntask_type='open_field'\nmeters_per_pixel=0.001").unwrap();
    unknown.frame_timing = None;
    assert!(
        behaviour::resolve_behaviour_timing_and_fps(&unknown, &empty)
            .unwrap_err()
            .to_string()
            .contains("behaviour timing requires")
    );
    let error = behaviour::analyze_pose(&oft_experiment("columns"), &arena(), empty)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("missing required column(s)"), "{error}");
}

#[test]
fn analysis_and_export_report_unreadable_sources_and_empty_stages() {
    let dir = scratch("unreadable-sources");
    let geometry = arena();
    let scale = manual_scale(&geometry);
    let raw = pose_frame(6, |i| (100.0 + 8.0 * i as f64, 80.0));
    let mut experiment = oft_experiment("unreadable");
    experiment.frame_image = Some(dir.join("absent.png"));
    let run = |experiment: &ExperimentConfig| {
        analyze_loaded_pose(
            experiment,
            &Default::default(),
            None,
            &geometry,
            &scale,
            raw.clone(),
            &labels(),
        )
    };
    assert!(
        run(&experiment).is_err(),
        "assembly resolves the source image size"
    );
    experiment.frame_image = None;
    let video = dir.join("empty.mp4");
    std::fs::write(&video, "").unwrap();
    experiment.video = Some(video.clone());
    experiment.video_resolution = Some((400, 240));
    let analysis = run(&experiment).unwrap();
    assert!(
        export_analysis(&dir, &experiment, analysis).is_err(),
        "an undecodable video fails"
    );
    let feature = bekiper_core::inspection::FeatureInspection::Trajectory {
        label: "first".into(),
        coordinate_labels: vec!["nose".into()],
        description: String::new(),
        representative_frame_index: Some(0),
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    };
    assert!(
        summary::extract_representative_frames(
            &video,
            &[feature],
            None,
            &ExperimentOutputDir::new(&dir, "frames")
        )
        .is_err()
    );
    std::fs::remove_dir_all(dir).unwrap();
}
