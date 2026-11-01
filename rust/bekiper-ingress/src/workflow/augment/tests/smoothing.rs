#[test]
fn run_augment_kalman_smoothing_changes_augmented_output() {
    fn run_case(name: &str, smoothing_method: SmoothingMethod) -> DataFrame {
        let exp_dir = prepare_output(name);
        write_pixel_perimeters(&exp_dir, vec![]);

        let mut df = polars::df! {
            "frame" => &[0i64, 1, 2, 3, 4],
            "nose_x" => &[0.0, 1.0, 20.0, 3.0, 4.0],
            "nose_y" => &[0.0, 0.0, 0.0, 0.0, 0.0],
            "nose_likelihood" => &[0.99, 0.99, 0.05, 0.99, 0.99],
        }
        .unwrap();
        write_parquet(&mut df, &exp_dir.poses_parquet()).unwrap();

        let pipeline = PipelineConfig {
            likelihood_threshold: 0.0,
            smoothing_window: 1,
            smoothing_method,
            max_velocity: 1_000.0,
            ..PipelineConfig::default()
        };

        run_augment(&experiment("exp"), &test_config(pipeline), &exp_dir).unwrap();
        read_augmented(&exp_dir)
    }

    let median = run_case("median_smoothing_output", SmoothingMethod::Median);
    let kalman = run_case("kalman_smoothing_output", SmoothingMethod::Kalman);
    let median_spike = median
        .column("nose_x")
        .unwrap()
        .f64()
        .unwrap()
        .get(2)
        .unwrap();
    let kalman_spike = kalman
        .column("nose_x")
        .unwrap()
        .f64()
        .unwrap()
        .get(2)
        .unwrap();

    assert!(
        (median_spike - 0.02).abs() < 1e-10,
        "median window 1 should preserve the spike, got {median_spike}"
    );
    assert!(
        kalman_spike < median_spike,
        "Kalman smoothing should downweight the low-confidence spike"
    );
}
