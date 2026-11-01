#[test]
fn run_augment_without_video_falls_back_to_interpolation() {
    let exp_dir = prepare_output("no_video_fallback");
    write_pixel_perimeters(&exp_dir, vec![]);

    let labels = SkeletonPreset::MouseTopView.resolve().labels();
    let frames: Vec<i64> = (0..40).collect();
    let mut columns: Vec<Column> = vec![Column::new("frame".into(), &frames)];
    for (idx, label) in labels.iter().enumerate() {
        let base = idx as f64 * 10.0;
        let mut xs = vec![base; frames.len()];
        if label == "tail_end" {
            xs[20] = base + 150.0;
        }
        let ys = vec![0.0; frames.len()];
        let liks = vec![0.99; frames.len()];
        columns.push(Column::new(format!("{label}_x").into(), &xs));
        columns.push(Column::new(format!("{label}_y").into(), &ys));
        columns.push(Column::new(format!("{label}_likelihood").into(), &liks));
    }
    let mut df = DataFrame::new(frames.len(), columns).unwrap();
    write_parquet(&mut df, &exp_dir.poses_parquet()).unwrap();

    let pipeline = PipelineConfig {
        skeleton: Some(SkeletonRepairConfig {
            preset: SkeletonPreset::MouseTopView,
            repair: SkeletonRepairMode::RedetectThenInterpolate,
            relative_tolerance: 0.35,
            mad_multiplier: 4.0,
            min_baseline_frames: 30,
        }),
        ..PipelineConfig::default()
    };

    run_augment(&experiment("exp"), &test_config(pipeline), &exp_dir).unwrap();

    let augmented = read_augmented(&exp_dir);

    assert_eq!(
        augmented
            .column("tail_end_skeleton_violation")
            .unwrap()
            .bool()
            .unwrap()
            .get(20),
        Some(true)
    );
    assert_eq!(
        augmented
            .column("skeleton_redetected")
            .unwrap()
            .bool()
            .unwrap()
            .get(20),
        Some(false)
    );
    assert!(augmented.column("tail_end_x").unwrap().get(20).unwrap() != AnyValue::Null);
}

#[test]
fn run_augment_uses_model_owned_imen_skeleton_by_default() {
    let exp_dir = prepare_output("imen_model_owned_skeleton");
    write_pixel_perimeters(&exp_dir, vec![]);

    let frames: Vec<i64> = (0..40).collect();
    let mut tail_x = vec![10.0; frames.len()];
    tail_x[20] = 100.0;
    let mut df = imen_pose_data(&frames, tail_x);
    write_parquet(&mut df, &exp_dir.poses_parquet()).unwrap();

    let pipeline = PipelineConfig {
        pose_label_model: PoseLabelModel::ImenDlc,
        ..PipelineConfig::default()
    };

    run_augment(&experiment("exp"), &test_config(pipeline), &exp_dir).unwrap();

    let augmented = read_augmented(&exp_dir);

    assert!(augmented.column("center_ear_x").is_ok());
    assert_eq!(
        augmented
            .column("tail_skeleton_violation")
            .unwrap()
            .bool()
            .unwrap()
            .get(20),
        Some(true)
    );
    assert_eq!(
        augmented
            .column("skeleton_redetected")
            .unwrap()
            .bool()
            .unwrap()
            .get(20),
        Some(false)
    );
    assert!(augmented.column("tail_x").unwrap().get(20).unwrap() != AnyValue::Null);
}
