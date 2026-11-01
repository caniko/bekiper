#[test]
fn run_augment_rerun_keeps_meter_perimeters_stable() {
    let exp_dir = prepare_output("rerun_keeps_meter_perimeters_stable");

    write_pixel_perimeters(
        &exp_dir,
        vec![PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 50.0,
            center_y: 50.0,
            width: 100.0,
            height: 100.0,
        }],
    );

    let mut df = polars::df! {
        "frame" => &[0i64, 1, 2],
        "nose_x" => &[25.0, 50.0, 75.0],
        "nose_y" => &[25.0, 50.0, 75.0],
        "nose_likelihood" => &[0.99, 0.99, 0.99],
    }
    .unwrap();
    write_parquet(&mut df, &exp_dir.poses_parquet()).unwrap();

    let mut experiment = experiment("exp");
    experiment.meters_per_pixel = ScaleConfig::Manual(0.01);
    let config = test_config(PipelineConfig::default());

    run_augment(&experiment, &config, &exp_dir).unwrap();
    let first = std::fs::read_to_string(exp_dir.perimeters_toml()).unwrap();
    let augmented = read_augmented(&exp_dir);
    assert_eq!(augmented.column("nose_x").unwrap().null_count(), 0);
    assert_eq!(augmented.column("nose_y").unwrap().null_count(), 0);

    run_augment(&experiment, &config, &exp_dir).unwrap();
    let second = std::fs::read_to_string(exp_dir.perimeters_toml()).unwrap();

    assert_eq!(
        first, second,
        "augment rerun should not rescale meter perimeters"
    );
    assert!(first.contains("width = 1.0"));
    assert!(first.contains("height = 1.0"));
}
