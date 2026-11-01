#[test]
fn animal_overview_pulls_cross_experiment_columns() {
    let root = unique_pipeline_dir("animal_overview_pulls_cross_experiment_columns");
    let data = root.join("data");

    write_overview_ymaze_result(
        &data.join("animal_42_ymaze_proximal25").join("result.toml"),
        "animal_42_ymaze_proximal25",
        "ymaze__proximal25",
        0.25,
        25.0,
    );
    write_overview_ymaze_result(
        &data.join("animal_42_ymaze_proximal65").join("result.toml"),
        "animal_42_ymaze_proximal65",
        "ymaze__proximal65",
        0.65,
        65.0,
    );
    write_overview_oft_result(&data.join("animal_42_oft").join("result.toml"));
    write_overview_nort_result(&data.join("animal_42_nort").join("result.toml"));

    let config_path = root.join("bekiper_config.toml");
    std::fs::write(
        &config_path,
        r#"
        [project]
        project_name = "animal_overview"
        data_directory = "."

        [aggregate.recipe_to_sheet]
        ymaze__proximal25 = "ymaze_proximal25"
        ymaze__proximal65 = "ymaze_proximal65"
        oft = "oft"
        nort_novelty = "nort_novelty"
        "#,
    )
    .unwrap();
    let config = bekiper_ingress::config::IngressConfig::from_file(&config_path).unwrap();

    bekiper_ingress::aggregate::run_aggregate(&config, &root).unwrap();

    let workbook = read_ods_workbook(&root.join("results.ods"));
    let overview = workbook
        .get("animal_overview")
        .expect("animal_overview sheet");
    assert_eq!(overview.len(), 2, "header plus one animal row");

    let header = &overview[0];
    assert_eq!(header[0], "animal");
    let row = &overview[1];
    assert_eq!(row[column_index(header, "animal")], "42");

    // Overview columns follow sheet emission order. Keep Y-maze variants
    // adjacent in recipe_to_sheet when clustered threshold columns are desired.
    assert_populated(
        row,
        header,
        "ymaze_proximal25_spontaneous_alternation_pct_standard",
    );
    assert_populated(
        row,
        header,
        "ymaze_proximal65_spontaneous_alternation_pct_standard",
    );
    assert_populated(row, header, "oft_center_time_pct");
    assert_populated(row, header, "nort_novelty_pct_novel");

    std::fs::remove_dir_all(&root).ok();
}

fn assert_populated(row: &[String], header: &[String], column: &str) {
    let value = &row[column_index(header, column)];
    assert!(!value.is_empty(), "{column} should be populated");
}

fn write_overview_ymaze_result(
    path: &std::path::Path,
    experiment: &str,
    analysis_role: &str,
    proximal_threshold: f64,
    spontaneous_alternation_pct: f64,
) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        format!(
            r#"
            experiment = "{experiment}"
            analysis_role = "{analysis_role}"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "y_maze"

            [animal_metadata]
            animal = 42
            cohort = "c1"
            gene = "wt"
            sex = "f"
            age = "12w"

            [detailed]
            proximal_threshold = {proximal_threshold}
            start_mode = "video_start"
            total_duration_seconds = 120.0
            analysis_duration_seconds = 120.0
            total_distance_meters = 2.0
            total_mean_speed = 0.02
            seconds_in_arms = 90.0
            seconds_in_arms_and_center = 100.0
            freezing_seconds = 1.0
            area_motion_summaries = []
            excluded = false

            [detailed.analysis]
            arm_sequence = ["A", "B", "C", "A"]
            raw_arm_sequence = ["A", "B", "C", "A"]
            zone_sequence = ["A", "center", "B", "center", "C", "A"]
            first_arm_entered = "A"
            last_arm_entered = "A"
            correct_alternations = 2
            incorrect_alternations = 0
            neutral_triads = 0
            spontaneous_alternation_pct = {spontaneous_alternation_pct}
            arm_preference = 0.25
            sum_of_permutation_distribution = 2
            incorrect_triad_distribution = {{}}
            neutral_triad_distribution = {{}}

            [detailed.analysis.permutation_distribution]
            abc = 1
            acb = 0
            bac = 0
            bca = 1
            cab = 0
            cba = 0

            [[detailed.zone_summaries]]
            zone_name = "in_arm_A"
            entries = 2
            duration_seconds = 30.0
            distance_meters = 0.8
            mean_speed = 0.02

            [[detailed.zone_summaries]]
            zone_name = "in_arm_B"
            entries = 1
            duration_seconds = 30.0
            distance_meters = 0.6
            mean_speed = 0.02

            [[detailed.zone_summaries]]
            zone_name = "in_arm_C"
            entries = 1
            duration_seconds = 30.0
            distance_meters = 0.6
            mean_speed = 0.02

            [[detailed.zone_summaries]]
            zone_name = "in_center"
            entries = 2
            duration_seconds = 10.0
            distance_meters = 0.1
            mean_speed = 0.01
            "#,
        ),
    )
    .unwrap();
}

fn write_overview_oft_result(path: &std::path::Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        r#"
        experiment = "animal_42_oft"
        analysis_role = "oft"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "open_field"

        [animal_metadata]
        animal = 42
        cohort = "c1"
        gene = "wt"
        sex = "f"
        age = "12w"

        [session_context]
        fecal_boli_count = 3

        [detailed]
        total_duration_seconds = 300.0
        total_distance_meters = 12.0
        total_mean_speed = 0.04
        freezing_seconds = 4.0
        immobility_pct = 5.0
        rearing_seconds = 8.0
        grooming_seconds = 2.0
        center_time_pct = 42.0
        periphery_time_pct = 58.0
        excluded = false

        [detailed.center_summary]
        entries = 4
        duration_seconds = 126.0
        latency_seconds = 12.0
        distance_meters = 5.0
        mean_speed = 0.04

        [detailed.periphery_summary]
        entries = 5
        duration_seconds = 174.0
        distance_meters = 7.0
        mean_speed = 0.04
        "#,
    )
    .unwrap();
}

fn write_overview_nort_result(path: &std::path::Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        r#"
        experiment = "animal_42_nort"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"

        [animal_metadata]
        animal = 42
        cohort = "c1"
        gene = "wt"
        sex = "f"
        age = "12w"

        [detailed]
        protocol = "nort_novelty"
        excluded = false
        total_duration_seconds = 180.0
        total_distance_meters = 7.0
        total_mean_speed = 0.04
        freezing_seconds = 3.0
        analysis_window_seconds = 120.0
        side_bias = 0.1
        object_bias = -0.1
        object_labels = ["familiar", "novel"]
        exploration_times = [6.0, 14.0]
        bout_counts = [3, 7]

        [detailed.indices]
        discrimination_index = 0.4
        recognition_index = 0.7
        pct_novel = 70.0
        log_ratio = 0.85
        novel_seconds = 14.0
        familiar_seconds = 6.0
        total_seconds = 20.0
        novel_bouts = 7
        familiar_bouts = 3
        "#,
    )
    .unwrap();
}
