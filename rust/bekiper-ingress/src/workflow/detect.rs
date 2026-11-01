include!("detect/dry_run_detect.rs");
include!("detect/run_detect_manual_annotation.rs");

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bekiper_detection_dry_run_{label}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_config(dir: &Path, body: &str) -> std::path::PathBuf {
        let path = dir.join("bekiper_config.toml");
        std::fs::write(&path, body).unwrap();
        path
    }

    fn nort_experiment(
        rule: Option<&str>,
        resolution: (u32, u32),
    ) -> crate::config::ExperimentConfig {
        let detection = rule.map(|rule| {
            toml::from_str::<crate::config::DetectionConfig>(&format!(
                "backend = [\"manual_annotation\"]\n[arena_calibration]\ncenter_zone_rule = {rule}\n"
            ))
            .unwrap()
        });
        crate::config::ExperimentConfig {
            recording_id: None,
            name: "nort".into(),
            task_type: bekiper_core::types::TaskType::ObjectRecognition,
            fps: bekiper_core::video::FpsConfig::Manual(30.0),
            meters_per_pixel: bekiper_core::scale::ScaleConfig::Manual(0.001),
            data_files: Vec::new(),
            video: None,
            pose_model: None,
            detection,
            task_params: None,
            trial_start_perimeter: None,
            annotation_dir: None,
            frame_timing: None,
            video_resolution: Some(resolution),
            frame_image: None,
            timestamps_file: None,
            animal_id: None,
            session_context: None,
        }
    }

    const FRAME_CENTERED: &str = "{ method = \"frame_centered\", arena_width_meters = 0.40, arena_height_meters = 0.40, center_width_meters = 0.20, center_height_meters = 0.20 }";
    const FRAME_CENTERED_BOX: &str = "{ method = \"frame_centered\", arena_width_meters = 0.40, arena_height_meters = 0.40, center_width_meters = 0.20, center_height_meters = 0.20, detect_box = true }";

    #[test]
    fn frame_centered_rule_centers_on_each_videos_own_frame() {
        use super::super::inputs::{AnnotationInput, materialize_manual_annotation};
        use bekiper_core::inspection::{PerimeterSpec, PerimetersFile};
        let dir = temp_dir("frame_centered");
        let plugins = dir.join("plugins");
        std::fs::create_dir_all(&plugins).unwrap();
        // Shared Neo/Imen calibration, measured on one 224x224 video.
        std::fs::write(
            plugins.join("meters_per_pixel-diagonal-0.565685.csv"),
            "mpp,8,218,215,7,A2_training_3.png,224,224\n",
        )
        .unwrap();
        let annotation = AnnotationInput {
            specs: vec![PerimeterSpec::Rectangle {
                label: "familiar".into(),
                center_x: 48.0,
                center_y: 49.0,
                width: 25.0,
                height: 38.0,
            }],
            labels: vec!["familiar".into()],
            detail: "recorded".into(),
            scale_plugin_dir: plugins,
        };
        let output = ExperimentOutputDir::new(&dir.join("run"), "nort");
        let mpp = 0.565685 / (207.0f64.powi(2) + 211.0f64.powi(2)).sqrt();
        // A 232x216 video (not the calibration image's size) is centered on
        // its own frame; with detect_box but no video it falls back the same way.
        for rule in [FRAME_CENTERED, FRAME_CENTERED_BOX] {
            materialize_manual_annotation(
                &nort_experiment(Some(rule), (232, 216)),
                &annotation,
                &output,
            )
            .unwrap();
            let written: PerimetersFile =
                toml::from_str(&std::fs::read_to_string(output.pixel_perimeters_toml()).unwrap())
                    .unwrap();
            let rect = |label: &str| {
                written.perimeters.iter().find_map(|p| match p {
                    PerimeterSpec::Rectangle {
                        label: l,
                        center_x,
                        center_y,
                        width,
                        height,
                    } if l == label => Some((*center_x, *center_y, *width, *height)),
                    _ => None,
                })
            };
            let (ax, ay, aw, _) = rect("arena").unwrap();
            let (cx, cy, cw, _) = rect("center_zone").unwrap();
            assert_eq!((ax, ay, cx, cy), (116.0, 108.0, 116.0, 108.0));
            assert!((aw - 0.40 / mpp).abs() < 1e-9 && (cw - 0.20 / mpp).abs() < 1e-9);
            assert_eq!(written.perimeters.len(), 3, "objects are kept");
            let scale: bekiper_core::scale::ScaleDerivationInfo =
                toml::from_str(&std::fs::read_to_string(output.annotation_scale_toml()).unwrap())
                    .unwrap();
            assert_eq!(scale.method, "annotation_diagonal+frame_center_fallback");
        }
        // Without the rule nothing is generated.
        materialize_manual_annotation(&nort_experiment(None, (232, 216)), &annotation, &output)
            .unwrap();
        let plain: PerimetersFile =
            toml::from_str(&std::fs::read_to_string(output.pixel_perimeters_toml()).unwrap())
                .unwrap();
        assert_eq!(plain.perimeters.len(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fresh_manual_detection_writes_scale_in_separate_data_directory() {
        use super::super::inputs::{AnnotationInput, materialize_manual_annotation};
        let dir = temp_dir("fresh_manual_scale");
        let plugins = dir.join("plugins");
        std::fs::create_dir_all(&plugins).unwrap();
        std::fs::write(
            plugins.join("meters_per_pixel-diagonal-0.565685.csv"),
            "mpp,8,218,215,7,A2_training_3.png,224,224\n",
        )
        .unwrap();
        let output = ExperimentOutputDir::new(&dir.join("run"), "manual");
        let annotation = AnnotationInput {
            specs: Vec::new(),
            labels: Vec::new(),
            detail: "recorded".into(),
            scale_plugin_dir: plugins,
        };
        materialize_manual_annotation(&nort_experiment(None, (224, 224)), &annotation, &output)
            .unwrap();
        assert!(output.pixel_perimeters_toml().is_file());
        let scale: bekiper_core::scale::ScaleDerivationInfo =
            toml::from_str(&std::fs::read_to_string(output.annotation_scale_toml()).unwrap())
                .unwrap();
        assert_eq!(scale.method, "annotation_diagonal");
        assert!((scale.meters_per_pixel - 0.001914).abs() < 1e-4);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn neo_imen_config_expands_for_detection_dry_run() {
        let temp_root = temp_dir("neo_imen_config");
        for dir in [
            "NORT videos",
            "OFT videos",
            "Y-maze videos",
            "deeplabcut/poses",
            "annotations/nort",
            "annotations/ymaze",
            "output",
        ] {
            std::fs::create_dir_all(temp_root.join(dir)).unwrap();
        }
        for video in [
            "NORT videos/NORT (post)/T1/Config A/post.T1.configA.video-10.mp4",
            "OFT videos/OFT (post)/post.video-1.mp4",
            "Y-maze videos/Y-maze (post)/post.video-1.mp4",
        ] {
            let path = temp_root.join(video);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, []).unwrap();

            let pose = temp_root
                .join("deeplabcut/poses")
                .join(format!("{}.h5", video.strip_suffix(".mp4").unwrap()));
            std::fs::create_dir_all(pose.parent().unwrap()).unwrap();
            std::fs::write(pose, []).unwrap();
        }
        let config_path = temp_root.join("bekiper_config.toml");
        std::fs::write(
            temp_root.join("annotations/nort/metadata.csv"),
            "Phase,PhasePart,TrialNumber,Stage,Animal\nA,2,7,0,1\n",
        )
        .unwrap();
        let config_body = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/neo/imen/bekiper_config.toml"
        ))
        .replace(
            "\n[aggregate.metadata]\npath = \"animal_metadata.xlsx\"\nsheet = \"Updated_Mice-overview\"\nmissing_policy = \"exclude_missing\"\n",
            "\n",
        );
        std::fs::write(&config_path, config_body).unwrap();
        // The NORT whisker age table reads per-animal ages from this workbook.
        std::fs::copy(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../examples/neo/imen/mice_overview.xlsx"
            ),
            temp_root.join("mice_overview.xlsx"),
        )
        .unwrap();
        let base_dir = config_path.parent().unwrap();
        let mut config = IngressConfig::from_file(&config_path).unwrap();

        config
            .expand_for_validation(base_dir, &[Stage::Detect])
            .unwrap();

        assert_eq!(config.pipeline.pose_label_model.as_str(), "imen_dlc");
        // One NORT video expanded into the whisker_age/whisker20mm/whisker30mm/
        // nose_only variants, one OFT video, and one Y-maze video expanded into
        // the arm0_dwell1s/arm25/arm45/arm65 contexts.
        assert_eq!(config.experiments.len(), 9);
        // Animal 10 (from `video-10`) is post by its `NORT (post)` folder, so the
        // flat age table resolves at its end age (94 d start, 177 d end).
        let whisker = config
            .experiments
            .iter()
            .find(|e| e.name.ends_with("__whisker_age"))
            .and_then(|e| {
                e.task_params
                    .as_ref()?
                    .exploration
                    .as_ref()?
                    .whisker
                    .clone()
            })
            .unwrap();
        assert_eq!(whisker.age_days, Some(177.0));
        assert!(matches!(
            whisker.length,
            Some(bekiper_feature::heuristic::solo::whisker_config::WhiskerLengthSpec::Meters(m))
                if (m.meters - 0.0117).abs() < 1e-12
        ));
        assert!(config.experiments.iter().any(|experiment| {
            experiment.task_type == bekiper_core::types::TaskType::YMaze
                && experiment
                    .task_params
                    .as_ref()
                    .and_then(|params| params.tail_coords.as_ref())
                    .is_some_and(|coords| coords.x == "base_tail_x" && coords.y == "base_tail_y")
        }));
        assert!(config.experiments.iter().any(|experiment| {
            matches!(
                experiment.task_type,
                bekiper_core::types::TaskType::OpenField
                    | bekiper_core::types::TaskType::ObjectRecognition
            ) && experiment
                .task_params
                .as_ref()
                .and_then(|params| params.centroid_coords.as_ref())
                .is_some_and(|coords| coords.x == "body_center_x" && coords.y == "body_center_y")
        }));

        std::fs::remove_dir_all(&temp_root).ok();
    }

    #[test]
    fn dry_run_reports_multiple_missing_videos() {
        let dir = temp_dir("missing_videos");
        let config_path = write_config(
            &dir,
            r#"
            [project]
            project_name = "dry"
            data_directory = "."

            [[experiments]]
            name = "missing_a"
            task_type = "object_recognition"
            fps = 30.0
            meters_per_pixel = 0.001
            video = "missing-a.mp4"

            [experiments.detection]
            recipe = "nort_training"
            backend = ["classical_cv"]

            [[experiments]]
            name = "missing_b"
            task_type = "object_recognition"
            fps = 30.0
            meters_per_pixel = 0.001
            video = "missing-b.mp4"

            [experiments.detection]
            recipe = "nort_training"
            backend = ["classical_cv"]
            "#,
        );

        let summary = analyze_detection_dry_run(&config_path).unwrap();
        let path_errors = summary
            .diagnostics
            .iter()
            .filter(|diagnostic| {
                diagnostic.category == "path/video" && diagnostic.severity == DryRunSeverity::Error
            })
            .count();

        assert_eq!(path_errors, 2);
        assert!(summary.errors >= 2);

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn dry_run_reports_manual_annotation_miss_without_aborting() {
        let dir = temp_dir("annotation_miss");
        let annotations = dir.join("annotations");
        std::fs::create_dir_all(&annotations).unwrap();
        let config_path = write_config(
            &dir,
            r#"
            [project]
            project_name = "dry"
            data_directory = "."

            [[experiments]]
            name = "annotation_miss"
            task_type = "object_recognition"
            fps = 30.0
            meters_per_pixel = 0.001
            video = "pre.T1.configA.video-1.mp4"

            [experiments.detection]
            recipe = "nort_training"
            backend = ["manual_annotation"]
            "#,
        );

        let mut config = IngressConfig::from_file(&config_path).unwrap();
        config.experiments[0].annotation_dir = Some(annotations);

        let mut summary = DetectionDryRunSummary {
            total_experiments: 1,
            ..DetectionDryRunSummary::default()
        };
        let mut cache = HashSet::new();
        let output_dir = dir.join("output");
        let resolver = PipelineInputResolver::new(&output_dir);
        let exp_dir = resolver.exp_dir(&config.experiments[0]);
        analyze_experiment_dry_run(
            &config.experiments[0],
            &resolver,
            &exp_dir,
            &dir,
            &config_path,
            &mut summary,
            &mut cache,
        );

        assert!(summary.diagnostics.iter().any(|diagnostic| {
            diagnostic.category == "annotation" && diagnostic.severity == DryRunSeverity::Error
        }));
        assert!(summary.errors >= 1);

        std::fs::remove_dir_all(dir).ok();
    }
}
