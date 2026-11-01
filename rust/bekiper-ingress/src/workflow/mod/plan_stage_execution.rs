fn plan_stage_execution(remaining_stages: &[Stage], config: &IngressConfig) -> StageExecutionPlan {
    let augment_uses_shared_model = remaining_stages.contains(&Stage::Augment)
        && (config.pipeline.skeleton.is_some() || config.pipeline.has_model_skeleton_repair());

    let sequential_shared_model: Vec<Stage> = remaining_stages
        .iter()
        .copied()
        .filter(|stage| {
            *stage == Stage::Pose || (augment_uses_shared_model && *stage == Stage::Augment)
        })
        .collect();
    let parallel_cpu: Vec<Stage> = remaining_stages
        .iter()
        .copied()
        .filter(|stage| !sequential_shared_model.contains(stage))
        .collect();

    StageExecutionPlan {
        sequential_shared_model,
        parallel_cpu,
    }
}

fn is_stage_fresh(
    stage: Stage,
    exp_dir: &ExperimentOutputDir,
    config: &ExperimentConfig,
    project: &IngressConfig,
) -> bool {
    if stage == Stage::Inspect {
        return !project.inspect.enabled || crate::inspection::is_fresh(exp_dir, &project.inspect);
    }
    let input_path: Option<&Path> = match stage {
        Stage::Detect => config.video.as_deref(),
        Stage::Pose => config
            .data_files
            .first()
            .map(|p| p.as_path())
            .or(config.video.as_deref()),
        // For dependent stages, just check existence of output
        Stage::Augment | Stage::Behaviour | Stage::Summary | Stage::Inspect => None,
        // Aggregate is a batch stage with no per-trial input/output.
        Stage::Aggregate => None,
    };
    exp_dir.is_fresh(stage, input_path)
}

fn run_stage(
    stage: Stage,
    experiment: &ExperimentConfig,
    config: &IngressConfig,
    exp_dir: &ExperimentOutputDir,
    _config_path: &Path,
) -> Result<()> {
    match stage {
        Stage::Detect => Ok(()), // handled separately in run_project
        Stage::Pose => run_pose(experiment, &config.gpu, exp_dir),
        Stage::Augment => run_augment(experiment, config, exp_dir),
        Stage::Behaviour => run_behaviour(experiment, exp_dir),
        Stage::Summary => run_summary(experiment, config, exp_dir),
        Stage::Inspect => run_inspect(experiment, config, exp_dir),
        // Aggregate runs once after the per-trial loop, not per experiment.
        // The run_project entry point invokes it separately; it's a no-op here.
        Stage::Aggregate => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use bekiper_core::config::{ProjectConfig, RuntimeSettings};
    use bekiper_core::inspection::IngressSessionContext;
    use bekiper_core::scale::ScaleConfig;
    use bekiper_core::types::{DetectionBackend, DetectionRecipe, TaskType};
    use bekiper_core::video::{FpsConfig, FrameTiming};

    use crate::config::{
        AggregateConfig, BackendPriority, DetectionConfig, ExperimentConfig, GpuConfig,
        PipelineConfig,
    };

    use super::*;

    fn config_with_pipeline(pipeline: PipelineConfig) -> IngressConfig {
        IngressConfig {
            inspect: Default::default(),
            review: Default::default(),
            project: ProjectConfig {
                project_name: "test".into(),
                data_directory: ".".into(),
                output_dir: Some("output".into()),
                runtime: Default::default(),
            },
            pipeline,
            gpu: GpuConfig::default(),
            aggregate: AggregateConfig::default(),
            post_analysis: crate::config::PostAnalysisConfig::default(),
            pose_model: None,
            scan: Vec::new(),
            animal_scan: Vec::new(),
            ingress_profile: Vec::new(),
            experiments: Vec::new(),
            animal_metadata: HashMap::new(),
            override_record: Vec::new(),
            toml_paths: Vec::new(),
        }
    }

    fn experiment_with_video(name: &str, fps: FpsConfig) -> ExperimentConfig {
        ExperimentConfig {
            recording_id: None,
            name: name.into(),
            task_type: TaskType::OpenField,
            fps,
            meters_per_pixel: ScaleConfig::Manual(0.001),
            data_files: Vec::new(),
            video: Some("video.mp4".into()),
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
        }
    }

    fn experiment_with_pose_data_and_detection(name: &str, data_file: PathBuf) -> ExperimentConfig {
        ExperimentConfig {
            recording_id: None,
            name: name.into(),
            task_type: TaskType::OpenField,
            fps: FpsConfig::Manual(30.0),
            meters_per_pixel: ScaleConfig::Manual(0.001),
            data_files: vec![data_file],
            video: Some("video.mp4".into()),
            pose_model: None,
            detection: Some(DetectionConfig {
                recipe: Some(DetectionRecipe::Oft),
                backend: BackendPriority::default(),
                perimeters: None,
                pipeline_overrides: None,
                arena_calibration: None,
            }),
            task_params: None,
            trial_start_perimeter: None,
            annotation_dir: None,
            frame_timing: None,
            video_resolution: None,
            frame_image: None,
            timestamps_file: None,
            animal_id: None,
            session_context: None,
        }
    }

    fn imen_habituation_experiment(name: &str) -> ExperimentConfig {
        let mut experiment = experiment_with_video(name, FpsConfig::Manual(30.0));
        experiment.task_type = TaskType::ObjectRecognition;
        experiment.session_context = Some(IngressSessionContext {
            profile: "videomap".into(),
            phase: Some("A".into()),
            phase_part: Some("1".into()),
            stage: Some("0".into()),
            ..IngressSessionContext::default()
        });
        experiment
    }

    fn imen_oft_habituation_experiment(name: &str) -> ExperimentConfig {
        let mut experiment = experiment_with_video(name, FpsConfig::Manual(30.0));
        experiment.task_type = TaskType::OpenField;
        experiment.video = Some("Imen/OFT videos/pre/video-1.mp4".into());
        experiment
    }

    fn test_scheduler() -> ParallelScheduler {
        ParallelScheduler::new(&RuntimeSettings {
            num_threads: Some(2),
            memory_scheduler_enabled: false,
            max_ram_usage_fraction: 0.80,
            ..RuntimeSettings::default()
        })
        .unwrap()
    }

    #[test]
    fn run_option_runtime_overrides_beat_loaded_config() {
        let mut config = config_with_pipeline(PipelineConfig::default());
        config.aggregate.enabled = true;
        config.aggregate.output_filename = "file.ods".into();
        config.pipeline.invert_y = false;
        config.project.runtime.num_threads = Some(2);
        config.project.runtime.memory_scheduler_enabled = false;
        config.project.runtime.max_ram_usage_fraction = 0.65;

        apply_run_option_overrides(
            &mut config,
            &RunOptions {
                no_aggregate: true,
                aggregate_output: Some("cli.xlsx".into()),
                invert_y: Some(true),
                num_threads: Some(7),
                memory_scheduler_enabled: Some(true),
                max_ram_usage_fraction: Some(0.9),
                ..RunOptions::default()
            },
        );

        assert!(!config.aggregate.enabled);
        assert_eq!(config.aggregate.output_filename, "cli.xlsx");
        assert!(config.pipeline.invert_y);
        assert_eq!(config.project.runtime.num_threads, Some(7));
        assert!(config.project.runtime.memory_scheduler_enabled);
        assert!((config.project.runtime.max_ram_usage_fraction - 0.9).abs() < 1e-12);
    }

    #[test]
    fn cli_runtime_overrides_beat_env_and_file_values() {
        let mut config = config_with_pipeline(PipelineConfig::default());
        config.project.runtime.num_threads = Some(2);
        config.project.runtime.memory_scheduler_enabled = false;
        config.project.runtime.max_ram_usage_fraction = 0.6;

        let env = HashMap::from([
            ("BEKIPER_RUNTIME_NUM_THREADS", "4"),
            ("BEKIPER_RUNTIME_MEMORY_SCHEDULER", "true"),
            ("BEKIPER_RUNTIME_MAX_RAM_USAGE", "0.75"),
        ]);
        config.apply_env_overrides_from(|key| env.get(key).map(|value| (*value).to_string()));

        apply_run_option_overrides(
            &mut config,
            &RunOptions {
                num_threads: Some(6),
                memory_scheduler_enabled: Some(false),
                max_ram_usage_fraction: Some(0.9),
                ..RunOptions::default()
            },
        );

        assert_eq!(config.project.runtime.num_threads, Some(6));
        assert!(!config.project.runtime.memory_scheduler_enabled);
        assert!((config.project.runtime.max_ram_usage_fraction - 0.9).abs() < 1e-12);
    }

    #[test]
    fn stage_execution_serializes_augment_for_model_skeleton_repair() {
        let plan = plan_stage_execution(
            &[Stage::Pose, Stage::Augment, Stage::Behaviour],
            &config_with_pipeline(PipelineConfig::default()),
        );

        assert_eq!(
            plan.sequential_shared_model,
            vec![Stage::Pose, Stage::Augment]
        );
        assert_eq!(plan.parallel_cpu, vec![Stage::Behaviour]);
    }

    #[test]
    fn stage_execution_serializes_augment_when_skeleton_repair_enabled() {
        let pipeline = PipelineConfig {
            skeleton: Some(crate::config::SkeletonRepairConfig {
                preset: bekiper_pose::skeleton::SkeletonPreset::MouseTopView,
                repair: crate::config::SkeletonRepairMode::RedetectThenInterpolate,
                relative_tolerance: 0.35,
                mad_multiplier: 4.0,
                min_baseline_frames: 30,
            }),
            ..PipelineConfig::default()
        };

        let plan = plan_stage_execution(
            &[Stage::Pose, Stage::Augment, Stage::Behaviour],
            &config_with_pipeline(pipeline),
        );

        assert_eq!(
            plan.sequential_shared_model,
            vec![Stage::Pose, Stage::Augment]
        );
        assert_eq!(plan.parallel_cpu, vec![Stage::Behaviour]);
    }

    #[test]
    fn no_detect_preflight_rejects_missing_perimeters() {
        let tmp = std::env::temp_dir().join("bekiper_no_detect_missing_perimeters");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("checkpoint/trial")).unwrap();
        let data_file = tmp.join("pose.h5");
        std::fs::write(&data_file, b"h5").unwrap();
        std::fs::write(tmp.join("checkpoint/trial/poses.parquet"), b"pose").unwrap();

        let mut config = config_with_pipeline(PipelineConfig::default());
        config.experiments = vec![experiment_with_pose_data_and_detection("trial", data_file)];

        let err =
            validate_no_detect_inputs(&config, &tmp.join("output"), &[Stage::Augment]).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("--no-detect preflight failed"),
            "{message}"
        );
        assert!(message.contains("pixel_perimeters.toml"), "{message}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn no_detect_preflight_reports_resolvable_manual_source_not_materialized() {
        let tmp = std::env::temp_dir().join("bekiper_no_detect_manual_source_note");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("checkpoint/trial")).unwrap();
        let annotation_dir = tmp.join("annotations");
        let plugin_dir = annotation_dir.join("plugin_files");
        std::fs::create_dir_all(&plugin_dir).unwrap();
        std::fs::write(tmp.join("checkpoint/trial/poses.parquet"), b"pose").unwrap();
        std::fs::write(
            plugin_dir.join("perimeter-rectangle-training_1.csv"),
            "familiar,48,49,25,38,img.png,224,224\nnovel,152,150,28,27,img.png,224,224\n",
        )
        .unwrap();
        std::fs::write(
            annotation_dir.join("video_map.csv"),
            "video_number,perimeter_id\n1,training_1\n",
        )
        .unwrap();

        let mut config = config_with_pipeline(PipelineConfig::default());
        let mut experiment = experiment_with_pose_data_and_detection("trial", tmp.join("pose.h5"));
        experiment.detection = Some(DetectionConfig {
            recipe: Some(DetectionRecipe::NortTraining),
            backend: BackendPriority(vec![DetectionBackend::ManualAnnotation]),
            perimeters: None,
            pipeline_overrides: None,
            arena_calibration: None,
        });
        experiment.video = Some(tmp.join("pre.video-1.mp4"));
        experiment.annotation_dir = Some(annotation_dir);
        config.experiments = vec![experiment];

        let err =
            validate_no_detect_inputs(&config, &tmp.join("output"), &[Stage::Augment]).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("manual annotations resolve via"),
            "{message}"
        );
        assert!(
            message.contains("--only detect --no-auto-detect"),
            "{message}"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn no_auto_detect_rejects_automatic_only_missing_perimeter() {
        let tmp = std::env::temp_dir().join("bekiper_no_auto_detect_auto_only");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("checkpoint/trial")).unwrap();

        let mut config = config_with_pipeline(PipelineConfig::default());
        config.experiments = vec![experiment_with_pose_data_and_detection(
            "trial",
            tmp.join("pose.h5"),
        )];

        let err = validate_no_auto_detect_inputs(
            &config,
            &tmp.join("output"),
            &[Stage::Detect, Stage::Augment],
            false,
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("--no-auto-detect preflight failed"),
            "{message}"
        );
        assert!(message.contains("automatic detection"), "{message}");
        assert!(message.contains("classical_cv"), "{message}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn detect_output_preflight_rejects_missing_perimeters_before_downstream() {
        let tmp = std::env::temp_dir().join("bekiper_detect_output_missing_perimeters");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("checkpoint/trial")).unwrap();
        let data_file = tmp.join("pose.h5");
        std::fs::write(&data_file, b"h5").unwrap();
        std::fs::write(tmp.join("checkpoint/trial/poses.parquet"), b"pose").unwrap();

        let mut config = config_with_pipeline(PipelineConfig::default());
        config.experiments = vec![experiment_with_pose_data_and_detection("trial", data_file)];

        let err = validate_detect_outputs_before_downstream(
            &config,
            &tmp.join("output"),
            &[Stage::Augment],
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("detect output preflight failed"),
            "{message}"
        );
        assert!(message.contains("pixel_perimeters.toml"), "{message}");
        assert!(!message.contains("rerun from detect stage"), "{message}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn detect_output_preflight_rejects_downstream_without_detection_config() {
        let tmp = std::env::temp_dir().join("bekiper_detect_output_no_detection_config");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("checkpoint/trial")).unwrap();

        let mut config = config_with_pipeline(PipelineConfig::default());
        config.experiments = vec![experiment_with_video("trial", FpsConfig::Manual(30.0))];

        let err = validate_detect_outputs_before_downstream(
            &config,
            &tmp.join("output"),
            &[Stage::Augment],
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("detect output preflight failed"),
            "{message}"
        );
        assert!(
            message.contains("no perimeter source configured"),
            "{message}"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn analysis_exclusion_filter_skips_imen_habituation_for_downstream() {
        let tmp = std::env::temp_dir().join("bekiper_detect_output_habituation_filtered");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("checkpoint/habituation")).unwrap();

        let mut config = config_with_pipeline(PipelineConfig::default());
        config.experiments = vec![imen_habituation_experiment("habituation")];

        filter_analysis_excluded_experiments(&mut config.experiments, &[Stage::Augment]);

        assert!(config.experiments.is_empty());
        validate_detect_outputs_before_downstream(&config, &tmp.join("output"), &[Stage::Augment])
            .unwrap();

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn analysis_exclusion_filter_keeps_imen_oft_videos_for_locomotion_analysis_stages() {
        for stage in [
            Stage::Detect,
            Stage::Augment,
            Stage::Behaviour,
            Stage::Summary,
            Stage::Inspect,
        ] {
            let mut experiments = vec![
                imen_oft_habituation_experiment("oft_videos_oft_pre_pre_video_1"),
                experiment_with_video("real_open_field_trial", FpsConfig::Manual(30.0)),
            ];

            filter_analysis_excluded_experiments(&mut experiments, &[stage]);

            assert_eq!(experiments.len(), 2, "stage {stage:?}");
            assert_eq!(experiments[0].name, "oft_videos_oft_pre_pre_video_1");
            assert_eq!(experiments[1].name, "real_open_field_trial");
        }
    }

    #[test]
    fn analysis_exclusion_filter_keeps_imen_habituation_for_pose_only() {
        let mut config = config_with_pipeline(PipelineConfig::default());
        config.experiments = vec![
            imen_habituation_experiment("habituation"),
            imen_oft_habituation_experiment("oft_videos_oft_pre_pre_video_1"),
        ];

        filter_analysis_excluded_experiments(&mut config.experiments, &[Stage::Pose]);

        assert_eq!(config.experiments.len(), 2);
    }

    #[test]
    fn detect_output_preflight_allows_pose_only_for_precomputed_pose_data() {
        let tmp = std::env::temp_dir().join("bekiper_detect_output_pose_only_precomputed");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("checkpoint/trial")).unwrap();
        let data_file = tmp.join("pose.h5");
        std::fs::write(&data_file, b"h5").unwrap();

        let mut config = config_with_pipeline(PipelineConfig::default());
        config.experiments = vec![experiment_with_pose_data_and_detection("trial", data_file)];

        validate_detect_outputs_before_downstream(&config, &tmp.join("output"), &[Stage::Pose])
            .unwrap();

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn no_detect_error_message_caps_long_lists() {
        let errors = (0..55)
            .map(|idx| format!("missing input {idx}"))
            .collect::<Vec<_>>();

        let message = format_no_detect_errors(&errors);

        assert!(
            message.contains("--no-detect preflight failed with 55 missing input(s)"),
            "{message}"
        );
        assert!(message.contains("missing input 0"), "{message}");
        assert!(message.contains("missing input 49"), "{message}");
        assert!(!message.contains("missing input 50"), "{message}");
        assert!(
            message.contains("... and 5 more missing input(s)"),
            "{message}"
        );
    }

    #[test]
    fn probe_video_timing_populates_resolution_for_manual_fps_video() {
        let tmp = std::env::temp_dir().join("bekiper_probe_video_resolution_manual_fps");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let mut experiments = vec![experiment_with_video("trial", FpsConfig::Manual(30.0))];
        let timing_calls = AtomicUsize::new(0);
        let scheduler = test_scheduler();

        probe_video_timing_with(
            &scheduler,
            &mut experiments,
            &tmp,
            &tmp.join("output"),
            &|_| Ok((1920, 1080)),
            &|_| {
                timing_calls.fetch_add(1, Ordering::Relaxed);
                Ok(FrameTiming::cfr(30.0))
            },
        )
        .unwrap();

        assert_eq!(experiments[0].video_resolution, Some((1920, 1080)));
        assert!(experiments[0].frame_timing.is_none());
        assert_eq!(timing_calls.load(Ordering::Relaxed), 0);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn probe_video_timing_does_not_double_rebase_absolute_video_paths() {
        let tmp = std::env::temp_dir().join("bekiper_probe_video_timing_absolute_video");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let absolute_video = tmp.join("video.mp4");
        let mut experiment = experiment_with_video("trial", FpsConfig::Manual(30.0));
        experiment.video = Some(absolute_video.clone());
        let mut experiments = vec![experiment];
        let seen_path = Arc::new(Mutex::new(None::<PathBuf>));
        let scheduler = test_scheduler();

        probe_video_timing_with(
            &scheduler,
            &mut experiments,
            &tmp,
            &tmp.join("output"),
            &{
                let seen_path = Arc::clone(&seen_path);
                move |path| {
                    *seen_path.lock().unwrap() = Some(path.to_path_buf());
                    Ok((1920, 1080))
                }
            },
            &|_| Ok(FrameTiming::cfr(30.0)),
        )
        .unwrap();

        assert_eq!(*seen_path.lock().unwrap(), Some(absolute_video));

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn probe_video_timing_extracts_timestamps_for_from_video_fps() {
        let tmp = std::env::temp_dir().join("bekiper_probe_video_timing_from_video");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let mut experiments = vec![experiment_with_video("trial", FpsConfig::FromVideo)];
        let timing_calls = AtomicUsize::new(0);
        let scheduler = test_scheduler();

        probe_video_timing_with(
            &scheduler,
            &mut experiments,
            &tmp,
            &tmp.join("output"),
            &|_| Ok((1280, 720)),
            &|_| {
                timing_calls.fetch_add(1, Ordering::Relaxed);
                Ok(FrameTiming::cfr(24.0))
            },
        )
        .unwrap();

        assert_eq!(experiments[0].video_resolution, Some((1280, 720)));
        assert_eq!(
            experiments[0]
                .frame_timing
                .as_ref()
                .map(|timing| timing.fps),
            Some(24.0)
        );
        assert_eq!(timing_calls.load(Ordering::Relaxed), 1);

        std::fs::remove_dir_all(&tmp).ok();
    }
}
