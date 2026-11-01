#[cfg(test)]
mod shared_variant_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bekiper_core::inspection::{PerimeterSpec, PerimetersFile};
    use bekiper_core::scale::ScaleConfig;
    use bekiper_core::types::{CoordinateColumns, TaskType};
    use polars::prelude::*;

    use crate::config::{TaskParamsOverrides, YMazeOverrides};

    use super::*;

    /// Counts the heavy operations and writes small deterministic outputs.
    #[derive(Default)]
    struct CountingRunner {
        resolution: AtomicUsize,
        timestamps: AtomicUsize,
        detect: AtomicUsize,
        pose: AtomicUsize,
        augment: AtomicUsize,
    }

    impl CountingRunner {
        fn counts(&self) -> [usize; 5] {
            [
                &self.resolution,
                &self.timestamps,
                &self.detect,
                &self.pose,
                &self.augment,
            ]
            .map(|counter| counter.load(Ordering::SeqCst))
        }
    }

    fn geometry() -> String {
        toml::to_string(&PerimetersFile {
            perimeters: vec![PerimeterSpec::RadialMaze {
                label: "maze".into(),
                center_vertices: vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
                arms: vec![
                    vec![(0.1, -0.2), (1.1, -0.2), (1.1, 0.2), (0.1, 0.2)],
                    vec![(-0.2, 0.1), (0.2, 0.1), (0.2, 1.1), (-0.2, 1.1)],
                    vec![(-1.1, -0.2), (-0.1, -0.2), (-0.1, 0.2), (-1.1, 0.2)],
                ],
            }],
        })
        .unwrap()
    }

    impl StageRunner for CountingRunner {
        fn video_resolution(&self, _video: &Path) -> Result<(u32, u32)> {
            self.resolution.fetch_add(1, Ordering::SeqCst);
            Ok((224, 224))
        }

        fn frame_timestamps(&self, _video: &Path) -> Result<FrameTiming> {
            self.timestamps.fetch_add(1, Ordering::SeqCst);
            Ok(FrameTiming::cfr(2.0))
        }

        fn detect(
            &self,
            _experiment: &ExperimentConfig,
            dir: &ExperimentOutputDir,
            allow_gpu: bool,
            _no_auto_detect: bool,
        ) -> Result<()> {
            if allow_gpu {
                return Err(BekiperError::Data("no gpu detection in the fixture".into()));
            }
            self.detect.fetch_add(1, Ordering::SeqCst);
            dir.ensure_checkpoints_dir()?;
            std::fs::write(dir.pixel_perimeters_toml(), geometry())?;
            Ok(())
        }

        fn pose(
            &self,
            _experiment: &ExperimentConfig,
            dir: &ExperimentOutputDir,
            _model: &mut Option<OnnxPoseModel>,
        ) -> Result<()> {
            self.pose.fetch_add(1, Ordering::SeqCst);
            dir.ensure_checkpoints_dir()?;
            let mut poses = df! { "frame" => (0u64..13).collect::<Vec<_>>() }.unwrap();
            bekiper_pose::io::write_parquet(&mut poses, &dir.poses_parquet())?;
            Ok(())
        }

        fn augment(
            &self,
            _experiment: &ExperimentConfig,
            _config: &IngressConfig,
            dir: &ExperimentOutputDir,
            _model: &mut Option<OnnxPoseModel>,
        ) -> Result<()> {
            self.augment.fetch_add(1, Ordering::SeqCst);
            dir.ensure_checkpoints_dir()?;
            let x = [
                0.0, 0.9, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, -0.9, -0.9, -0.9, 0.0,
            ];
            let y = [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            let mut augmented = df! {
                "frame" => (0u64..13).collect::<Vec<_>>(),
                "mouse_center_x" => x, "mouse_center_y" => y,
                "nose_x" => x, "nose_y" => y,
                "center_ear_x" => x, "center_ear_y" => y,
                "tail_base_x" => x, "tail_base_y" => y
            }
            .unwrap();
            bekiper_pose::io::write_parquet(&mut augmented, &dir.augmented_parquet())?;
            std::fs::write(dir.perimeters_toml(), geometry())?;
            Ok(())
        }
    }

    fn variant(
        root: &Path,
        recording: Option<&str>,
        variant: &str,
        task_params: TaskParamsOverrides,
    ) -> ExperimentConfig {
        ExperimentConfig {
            name: match recording {
                Some(base) => format!("{base}__{variant}"),
                None => format!("rec__{variant}"),
            },
            task_type: TaskType::YMaze,
            fps: FpsConfig::FromVideo,
            meters_per_pixel: ScaleConfig::Manual(0.001),
            data_files: Vec::new(),
            video: Some(root.join("rec.mp4")),
            pose_model: None,
            detection: None,
            task_params: Some(task_params),
            trial_start_perimeter: None,
            annotation_dir: None,
            frame_timing: None,
            video_resolution: None,
            frame_image: None,
            timestamps_file: None,
            animal_id: None,
            session_context: None,
            recording_id: recording.map(str::to_string),
        }
    }

    fn gap(seconds: f64) -> TaskParamsOverrides {
        TaskParamsOverrides {
            ymaze: Some(YMazeOverrides {
                zone_max_gap_seconds: Some(seconds),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn variants(root: &Path, recording: Option<&str>) -> Vec<ExperimentConfig> {
        let mut head = gap(0.2);
        head.head_coords = Some(CoordinateColumns {
            x: "nose_x".into(),
            y: "nose_y".into(),
        });
        vec![
            variant(root, recording, "a", gap(0.0)),
            variant(root, recording, "b", gap(0.1)),
            variant(root, recording, "c", head),
        ]
    }

    fn run(root: &Path, experiments: Vec<ExperimentConfig>, runner: &CountingRunner) {
        std::fs::create_dir_all(root).unwrap();
        if !root.join("rec.mp4").exists() {
            // A real clip: summary decodes representative frames from it.
            let status = std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=224x224:rate=2:duration=6.5",
                ])
                .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "4"])
                .arg(root.join("rec.mp4"))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .expect("ffmpeg must be on PATH to run the shared-variant tests");
            assert!(status.success());
        }
        let mut config: IngressConfig = toml::from_str(
            r#"
            [project]
            project_name = "shared"
            data_directory = "."
            output_dir = "output"
            [project.runtime]
            num_threads = 2
            memory_scheduler_enabled = false
            "#,
        )
        .unwrap();
        config.experiments = experiments;
        let scheduler = ParallelScheduler::new(&config.project.runtime).unwrap();
        let opts = RunOptions {
            no_inspect: true,
            no_aggregate: true,
            ..RunOptions::default()
        };
        let stages = [
            Stage::Detect,
            Stage::Pose,
            Stage::Augment,
            Stage::Behaviour,
            Stage::Summary,
        ];
        run_experiments(
            &mut config,
            &opts,
            &stages,
            root,
            &root.join("output"),
            &root.join("bekiper_config.toml"),
            &scheduler,
            runner,
        )
        .unwrap();
    }

    fn result_files(root: &Path, name: &str) -> Vec<(String, Vec<u8>)> {
        let dir = root.join("output/data").join(name);
        let root_text = root.to_string_lossy().into_owned();
        [
            "result.toml".to_string(),
            "behaviour.parquet".to_string(),
            format!("{name}_evaluated.parquet"),
            format!("{name}_inspection.toml"),
        ]
        .into_iter()
        .map(|file| {
            let bytes = std::fs::read(dir.join(&file))
                .unwrap_or_else(|e| panic!("{}: {e}", dir.join(&file).display()));
            // Outputs may name the run root; compare modulo that.
            let bytes = match String::from_utf8(bytes.clone()) {
                Ok(text) => text.replace(&root_text, "ROOT").into_bytes(),
                Err(_) => bytes,
            };
            (file, bytes)
        })
        .collect()
    }

    fn fresh_root(tag: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("bekiper-shared-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn variants_share_stage_work_and_match_standalone_runs() {
        let shared_root = fresh_root("shared");
        let runner = CountingRunner::default();
        run(&shared_root, variants(&shared_root, Some("rec")), &runner);
        // Probe, detect and pose run once for the recording; augment once per
        // distinct head_coords (a and b share, c differs).
        assert_eq!(runner.counts(), [1, 1, 1, 1, 2]);

        let mut standalone_counts = [0usize; 5];
        for experiment in variants(&shared_root, Some("rec")) {
            let alone_root = fresh_root(&format!("alone-{}", experiment.name));
            let alone_runner = CountingRunner::default();
            let mut alone = experiment.clone();
            alone.recording_id = None;
            alone.video = Some(alone_root.join("rec.mp4"));
            run(&alone_root, vec![alone], &alone_runner);
            for (total, count) in standalone_counts.iter_mut().zip(alone_runner.counts()) {
                *total += count;
            }
            for ((file, shared_bytes), (_, alone_bytes)) in
                result_files(&shared_root, &experiment.name)
                    .into_iter()
                    .zip(result_files(&alone_root, &experiment.name))
            {
                if file.ends_with(".toml") {
                    let (shared_text, alone_text) = (
                        String::from_utf8_lossy(&shared_bytes),
                        String::from_utf8_lossy(&alone_bytes),
                    );
                    // Map-ordered tables (colours, columns) may print in any order.
                    let sorted = |text: &str| {
                        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
                        lines.sort();
                        lines
                    };
                    let (shared_lines, alone_lines) = (sorted(&shared_text), sorted(&alone_text));
                    let differing: Vec<_> = shared_lines
                        .iter()
                        .zip(&alone_lines)
                        .filter(|(a, b)| a != b)
                        .collect();
                    assert_eq!(shared_lines.len(), alone_lines.len());
                    assert!(
                        differing.is_empty(),
                        "{} {file}: {differing:?}",
                        experiment.name
                    );
                } else {
                    assert!(shared_bytes == alone_bytes, "{} {file}", experiment.name);
                }
            }
            std::fs::remove_dir_all(&alone_root).ok();
        }
        assert_eq!(standalone_counts, [3, 3, 3, 3, 3]);

        // Each variant keeps the legacy layout and names what it consumed.
        for name in ["rec__a", "rec__b", "rec__c"] {
            let dir = ExperimentOutputDir::new(&shared_root.join("output"), name);
            assert!(dir.augmented_parquet().exists(), "{name}");
            assert!(dir.poses_parquet().exists(), "{name}");
            assert!(dir.pixel_perimeters_toml().exists(), "{name}");
            assert!(dir.frame_timing_toml().exists(), "{name}");
            let marker = crate::stage_share::read_marker(dir.root()).unwrap();
            assert_eq!(marker.recording, "rec");
            assert_eq!(marker.consumed.len(), 4);
        }
        let a = crate::stage_share::read_marker(
            ExperimentOutputDir::new(&shared_root.join("output"), "rec__a").root(),
        )
        .unwrap();
        let b = crate::stage_share::read_marker(
            ExperimentOutputDir::new(&shared_root.join("output"), "rec__b").root(),
        )
        .unwrap();
        let c = crate::stage_share::read_marker(
            ExperimentOutputDir::new(&shared_root.join("output"), "rec__c").root(),
        )
        .unwrap();
        assert_eq!(
            a.consumed["augment"].fingerprint,
            b.consumed["augment"].fingerprint
        );
        assert_ne!(
            a.consumed["augment"].fingerprint,
            c.consumed["augment"].fingerprint
        );
        assert_eq!(
            a.consumed["detect"].fingerprint,
            c.consumed["detect"].fingerprint
        );

        // A second run finds everything fresh.
        let rerun = CountingRunner::default();
        run(&shared_root, variants(&shared_root, Some("rec")), &rerun);
        assert_eq!(rerun.counts(), [0, 0, 0, 0, 0]);

        // Changing an augment-read parameter recomputes only that variant's
        // augment output, and its stale behaviour output is replaced.
        let mut changed = variants(&shared_root, Some("rec"));
        changed[0].task_params.as_mut().unwrap().nose_coords = Some(CoordinateColumns {
            x: "center_ear_x".into(),
            y: "center_ear_y".into(),
        });
        let refreshed = CountingRunner::default();
        run(&shared_root, changed, &refreshed);
        assert_eq!(refreshed.counts(), [0, 0, 0, 0, 1]);
        let marker_a = crate::stage_share::read_marker(
            ExperimentOutputDir::new(&shared_root.join("output"), "rec__a").root(),
        )
        .unwrap();
        assert_ne!(
            marker_a.consumed["augment"].fingerprint,
            a.consumed["augment"].fingerprint
        );

        std::fs::remove_dir_all(&shared_root).ok();
    }

    #[test]
    fn legacy_checkpoints_are_adopted_not_recomputed() {
        let root = fresh_root("legacy");
        // A pre-sharing run: the same variants without a recording id.
        let legacy_runner = CountingRunner::default();
        run(&root, variants(&root, None), &legacy_runner);
        assert_eq!(legacy_runner.counts(), [3, 3, 3, 3, 3]);

        // The same data now run as variants of one recording reuses it.
        let runner = CountingRunner::default();
        run(&root, variants(&root, Some("rec")), &runner);
        // Only the probe (not adoptable: legacy roots never stored the video info) reruns.
        assert_eq!(runner.counts(), [1, 1, 0, 0, 0]);
        std::fs::remove_dir_all(&root).ok();
    }
}
