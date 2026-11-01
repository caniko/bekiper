#[test]
fn run_augment_rejects_missing_skeleton_labels() {
    let exp_dir = prepare_output("missing_labels");
    write_pixel_perimeters(&exp_dir, vec![]);

    let mut df = polars::df! {
        "frame" => &[0i64, 1],
        "nose_x" => &[0.0, 1.0],
        "nose_y" => &[0.0, 0.0],
        "nose_likelihood" => &[0.9, 0.9],
    }
    .unwrap();
    write_parquet(&mut df, &exp_dir.poses_parquet()).unwrap();

    let pipeline = PipelineConfig {
        skeleton: Some(SkeletonRepairConfig {
            preset: SkeletonPreset::MouseTopView,
            repair: SkeletonRepairMode::RedetectThenInterpolate,
            relative_tolerance: 0.35,
            mad_multiplier: 4.0,
            min_baseline_frames: 2,
        }),
        ..PipelineConfig::default()
    };

    let error = run_augment(&experiment("exp"), &test_config(pipeline), &exp_dir).unwrap_err();
    assert!(format!("{error}").contains("missing labels"));
}

#[test]
fn run_augment_requires_pixel_perimeters() {
    let exp_dir = prepare_output("missing_pixel_perimeters");

    let mut df = polars::df! {
        "frame" => &[0i64, 1],
        "nose_x" => &[0.0, 1.0],
        "nose_y" => &[0.0, 0.0],
        "nose_likelihood" => &[0.9, 0.9],
    }
    .unwrap();
    write_parquet(&mut df, &exp_dir.poses_parquet()).unwrap();

    let error = run_augment(
        &experiment("exp"),
        &test_config(PipelineConfig::default()),
        &exp_dir,
    )
    .unwrap_err();
    assert!(format!("{error}").contains("rerun from detect stage"));
}

fn skeleton_violation_pose() -> DataFrame {
    let labels = SkeletonPreset::MouseTopView.resolve().labels();
    let frames: Vec<i64> = (0..40).collect();
    let mut columns: Vec<Column> = vec![Column::new("frame".into(), &frames)];
    for (idx, label) in labels.iter().enumerate() {
        let base = idx as f64 * 10.0;
        let mut xs = vec![base; frames.len()];
        if label == "tail_end" {
            xs[20] = base + 150.0;
        }
        columns.push(Column::new(format!("{label}_x").into(), &xs));
        columns.push(Column::new(
            format!("{label}_y").into(),
            &vec![0.0; frames.len()],
        ));
        columns.push(Column::new(
            format!("{label}_likelihood").into(),
            &vec![0.99; frames.len()],
        ));
    }
    DataFrame::new(frames.len(), columns).unwrap()
}

fn redetect_pipeline() -> PipelineConfig {
    PipelineConfig {
        skeleton: Some(SkeletonRepairConfig {
            preset: SkeletonPreset::MouseTopView,
            repair: SkeletonRepairMode::RedetectThenInterpolate,
            relative_tolerance: 0.35,
            mad_multiplier: 4.0,
            min_baseline_frames: 30,
        }),
        ..PipelineConfig::default()
    }
}

fn with_pose_model(mut experiment: ExperimentConfig) -> ExperimentConfig {
    experiment.video = Some("/nonexistent/video.mp4".into());
    experiment.pose_model = Some(
        toml::from_str("path = '/nonexistent/model.onnx'\nspec = 'superanimal_topviewmouse'")
            .unwrap(),
    );
    experiment
}

#[test]
fn redetection_requests_fail_without_a_loadable_model_or_in_direct_analysis() {
    let labels: Vec<String> = SkeletonPreset::MouseTopView
        .resolve()
        .labels()
        .iter()
        .map(|l| l.to_string())
        .collect();
    let modelled = with_pose_model(experiment("exp"));
    let error = augment_pose(
        skeleton_violation_pose().lazy(),
        &modelled,
        &redetect_pipeline(),
        &labels,
        &[],
        MetersPerPixel(0.001),
        None,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(
        error.contains("cannot perform ONNX re-detection"),
        "{error}"
    );

    // Without a video and model, flagged frames are only interpolated.
    augment_pose(
        skeleton_violation_pose().lazy(),
        &experiment("exp"),
        &redetect_pipeline(),
        &labels,
        &[],
        MetersPerPixel(0.001),
        None,
    )
    .unwrap()
    .collect()
    .unwrap();
    let exp_dir = prepare_output("redetect_model_missing");
    write_pixel_perimeters(&exp_dir, vec![]);
    write_parquet(&mut skeleton_violation_pose(), &exp_dir.poses_parquet()).unwrap();
    assert!(run_augment(&modelled, &test_config(redetect_pipeline()), &exp_dir).is_err());
}

#[test]
fn direct_augmentation_validates_labels_inversion_and_source_clocks() {
    let labels = vec!["nose".to_string()];
    let pose = || {
        polars::df! {"frame" => [0i64, 1, 2], "nose_x" => [1.0, 2.0, 3.0], "nose_y" => [1.0, 2.0, 3.0], "nose_likelihood" => [1.0; 3]}
            .unwrap()
            .lazy()
    };
    let augment = |experiment: &ExperimentConfig, pipeline: &PipelineConfig| {
        augment_pose(
            pose(),
            experiment,
            pipeline,
            &labels,
            &[],
            MetersPerPixel(0.001),
            None,
        )
        .and_then(|lf| Ok(lf.collect()?))
        .map_err(|error| error.to_string())
    };
    let inverted = PipelineConfig {
        invert_y: true,
        ..PipelineConfig::default()
    };
    assert!(
        augment(&experiment("exp"), &inverted)
            .unwrap_err()
            .contains("no source frame height")
    );
    let mut clocked = experiment("exp");
    clocked.frame_timing = Some(bekiper_core::video::FrameTiming {
        fps: 2.0,
        is_vfr: true,
        timestamps: vec![0.0, 0.0, 1.0],
    });
    assert!(
        augment(&clocked, &PipelineConfig::default())
            .unwrap_err()
            .contains("strictly increasing")
    );
    clocked.frame_timing.as_mut().unwrap().timestamps = vec![0.0, 0.5];
    assert!(
        augment(&clocked, &PipelineConfig::default())
            .unwrap_err()
            .contains("no timestamp for frame 2")
    );

    let dir = test_dir("timestamps_file");
    let timestamps = dir.join("timestamps.csv");
    std::fs::write(
        &timestamps,
        "0,2023-08-19T12:07:03.0+02:00\n1,2023-08-19T12:07:03.5+02:00\n2,2023-08-19T12:07:04.0+02:00\n",
    )
    .unwrap();
    let mut recorded = experiment("exp");
    recorded.timestamps_file = Some(dir.join("absent.csv"));
    assert!(augment(&recorded, &PipelineConfig::default()).is_err());
    recorded.timestamps_file = Some(timestamps);
    let df = augment(&recorded, &PipelineConfig::default()).unwrap();
    assert_eq!(
        df.column("elapsed_s").unwrap().f64().unwrap().get(2),
        Some(1.0)
    );
    use bekiper_pose::pipeline::PipelineStep;
    assert_eq!(SnoutDirectionStep::default().name(), "snout_direction");
    assert_eq!(EarDirectionStep.name(), "ear_direction");
}

#[test]
fn staged_inversion_reports_an_unwritable_metric_perimeter_checkpoint() {
    let exp_dir = prepare_output("invert_unwritable");
    write_pixel_perimeters(&exp_dir, vec![]);
    write_parquet(
        &mut imen_pose_data(&[0, 1, 2], vec![10.0; 3]),
        &exp_dir.poses_parquet(),
    )
    .unwrap();
    std::fs::create_dir_all(exp_dir.perimeters_toml()).unwrap();
    let mut trial = experiment("exp");
    trial.video_resolution = Some((100, 100));
    let pipeline = PipelineConfig {
        invert_y: true,
        pose_label_model: PoseLabelModel::ImenDlc,
        ..PipelineConfig::default()
    };
    assert!(run_augment(&trial, &test_config(pipeline), &exp_dir).is_err());
}
