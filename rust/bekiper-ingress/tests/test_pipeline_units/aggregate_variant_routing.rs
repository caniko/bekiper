#[test]
fn aggregate_routes_analysis_role_variant_to_configured_sheet() {
    let tmp = test_dir("aggregate_variant_analysis_role");
    let result_path = tmp.join("result.toml");
    std::fs::write(
        &result_path,
        r#"
        experiment = "synthetic_ymaze_trial__proximal65"
        analysis_role = "ymaze__proximal65"
        fps = 15.0
        meters_per_pixel = 0.001
        task_type = "y_maze"

        [animal_metadata]
        animal = 14

        [detailed]
        proximal_threshold = 0.65
        start_mode = "video_start"
        total_duration_seconds = 10.0
        analysis_duration_seconds = 10.0
        total_distance_meters = 1.0
        total_mean_speed = 0.1
        zone_summaries = []
        area_motion_summaries = []

        [detailed.analysis]
        zone_sequence = []
        arm_sequence = []
        raw_arm_sequence = []
        arm_entries = [0, 0, 0]
        center_entries = 0
        correct_alternations = 0
        incorrect_alternations = 0
        neutral_triads = 0
        spontaneous_alternation_pct = 0.0
        arm_preference = 0.0
        permutation_distribution = { abc = 0, acb = 0, bac = 0, bca = 0, cab = 0, cba = 0 }
        sum_of_permutation_distribution = 0
        incorrect_triad_distribution = {}
        neutral_triad_distribution = {}
        "#,
    )
    .unwrap();

    let trial = bekiper_ingress::aggregate::TrialResult::load(&result_path).unwrap();
    let mut cfg = bekiper_ingress::config::AggregateConfig::default();
    cfg.recipe_to_sheet
        .insert("ymaze__proximal65".into(), "ymaze_proximal65".into());

    let sheets = bekiper_ingress::aggregate::build_sheet_dataframes(vec![trial], &cfg).unwrap();
    assert_eq!(sheets.len(), 1);
    assert_eq!(sheets[0].0, "ymaze_proximal65");
    assert_eq!(sheets[0].1.height(), 1);

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn experiment_output_dir_uses_variant_experiment_name_once() {
    let output_root = std::path::Path::new("/tmp/bekiper_variant_output_layout");
    let exp_dir = bekiper_ingress::stage::ExperimentOutputDir::new(
        output_root,
        "y_maze_videos_trial__proximal65",
    );

    assert_eq!(
        exp_dir.root(),
        output_root
            .join("data")
            .join("y_maze_videos_trial__proximal65")
    );
    assert_eq!(exp_dir.result_toml(), exp_dir.root().join("result.toml"));
}
