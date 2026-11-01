#[test]
fn oft_extract_populates_all_static_columns() {
    let t = parse(
        r#"
        experiment = "42_seq1_oft"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "open_field"

        [animal_metadata]
        animal = 42
        cohort = "B"
        gene = "wt"
        sex = "m"
        age = "10"

        [session_context]
        fecal_boli_count = 4

        [detailed]
        excluded = false
        total_duration_seconds = 300.0
        total_distance_meters = 12.5
        total_mean_speed = 0.041
        freezing_seconds = 15.0
        immobility_pct = 5.0
        rearing_seconds = 7.2
        grooming_seconds = 3.1
        center_time_pct = 22.0
        periphery_time_pct = 78.0

        [detailed.center_summary]
        zone_name = "in_center"
        entries = 14
        duration_seconds = 66.0
        distance_meters = 2.4
        mean_speed = 0.036
        latency_seconds = 5.1

        [detailed.periphery_summary]
        zone_name = "in_periphery"
        entries = 14
        duration_seconds = 234.0
        distance_meters = 10.1
        mean_speed = 0.043
        latency_seconds = 0.0

        [detailed.center_motion_summary]
        zone_name = "in_center"
        total_displacement_meters = 2.3
        median_speed_mps = 0.031
        median_acceleration_mps2 = 0.12
        freezing_seconds = 4.5

        [detailed.periphery_motion_summary]
        zone_name = "in_periphery"
        total_displacement_meters = 10.0
        median_speed_mps = 0.045
        median_acceleration_mps2 = 0.17
        freezing_seconds = 10.5
        "#,
    );

    let row = OftExperiment.extract_row(&t);

    // Every static column present
    for c in OftExperiment.static_columns() {
        assert!(row.contains_key(&c), "missing OFT static column {c}");
    }

    // Metadata spot checks
    assert_eq!(row.get("animal"), Some(&CellValue::Int(42)));
    assert_eq!(row.get("cohort"), Some(&CellValue::String("B".into())));
    assert_eq!(row.get("age"), Some(&CellValue::String("10".into())));
    assert_eq!(row.get("fps"), Some(&CellValue::Float(30.0)));

    // Motion / duration
    assert_eq!(row.get("total_duration_s"), Some(&CellValue::Float(300.0)));
    assert_eq!(row.get("total_distance_m"), Some(&CellValue::Float(12.5)));
    assert_eq!(row.get("total_mean_speed"), Some(&CellValue::Float(0.041)));

    // Freezing / posture
    assert_eq!(row.get("freezing_s"), Some(&CellValue::Float(15.0)));
    assert_eq!(row.get("immobility_pct"), Some(&CellValue::Float(5.0)));
    assert_eq!(row.get("rearing_s"), Some(&CellValue::Float(7.2)));
    assert_eq!(row.get("grooming_s"), Some(&CellValue::Float(3.1)));
    assert_eq!(row.get("fecal_boli_count"), Some(&CellValue::Int(4)));

    // Zone-derived columns
    assert_eq!(row.get("center_entries"), Some(&CellValue::Int(14)));
    assert_eq!(row.get("center_duration_s"), Some(&CellValue::Float(66.0)));
    assert_eq!(row.get("latency_to_center_s"), Some(&CellValue::Float(5.1)));
    assert_eq!(row.get("center_distance_m"), Some(&CellValue::Float(2.4)));
    assert_eq!(row.get("center_mean_speed"), Some(&CellValue::Float(0.036)));
    assert_eq!(row.get("center_time_pct"), Some(&CellValue::Float(22.0)));
    assert_eq!(
        row.get("center_motion_distance_m"),
        Some(&CellValue::Float(2.3))
    );
    assert_eq!(
        row.get("center_motion_median_speed"),
        Some(&CellValue::Float(0.031))
    );
    assert_eq!(
        row.get("center_motion_median_acceleration"),
        Some(&CellValue::Float(0.12))
    );
    assert_eq!(
        row.get("center_motion_freezing_s"),
        Some(&CellValue::Float(4.5))
    );
    assert_eq!(row.get("periphery_entries"), Some(&CellValue::Int(14)));
    assert_eq!(
        row.get("periphery_duration_s"),
        Some(&CellValue::Float(234.0))
    );
    assert_eq!(
        row.get("periphery_distance_m"),
        Some(&CellValue::Float(10.1))
    );
    assert_eq!(
        row.get("periphery_mean_speed"),
        Some(&CellValue::Float(0.043))
    );
    assert_eq!(row.get("periphery_time_pct"), Some(&CellValue::Float(78.0)));
    assert_eq!(
        row.get("periphery_motion_distance_m"),
        Some(&CellValue::Float(10.0))
    );
    assert_eq!(
        row.get("periphery_motion_median_speed"),
        Some(&CellValue::Float(0.045))
    );
    assert_eq!(
        row.get("periphery_motion_median_acceleration"),
        Some(&CellValue::Float(0.17))
    );
    assert_eq!(
        row.get("periphery_motion_freezing_s"),
        Some(&CellValue::Float(10.5))
    );

    // Trailing
    assert_eq!(
        row.get("experiment"),
        Some(&CellValue::String("42_seq1_oft".into()))
    );
    assert_eq!(row.get("excluded"), Some(&CellValue::Bool(false)));
}

#[test]
fn oft_extract_missing_zone_summaries_yields_nulls() {
    // If zone_summary sub-objects are absent (e.g. task failed before
    // full_analyze), zone-derived columns must be Null rather than
    // panicking or producing garbage.
    let t = parse(
        r#"
        experiment = "7_seq0_oft"
        fps = 25.0
        meters_per_pixel = 0.001
        task_type = "open_field"

        [animal_metadata]
        animal = 7
        cohort = "A"
        gene = "wt"
        sex = "f"
        age = "8"

        [detailed]
        total_duration_seconds = 100.0
        total_distance_meters = 3.0
        total_mean_speed = 0.03
        freezing_seconds = 1.0
        "#,
    );

    let row = OftExperiment.extract_row(&t);

    // Metadata + trial-level scalars populated
    assert_eq!(row.get("animal"), Some(&CellValue::Int(7)));
    assert_eq!(row.get("fps"), Some(&CellValue::Float(25.0)));
    assert_eq!(row.get("total_duration_s"), Some(&CellValue::Float(100.0)));

    // Zone-derived columns all Null
    assert_eq!(row.get("center_entries"), Some(&CellValue::Null));
    assert_eq!(row.get("center_duration_s"), Some(&CellValue::Null));
    assert_eq!(row.get("latency_to_center_s"), Some(&CellValue::Null));
    assert_eq!(row.get("center_distance_m"), Some(&CellValue::Null));
    assert_eq!(row.get("center_mean_speed"), Some(&CellValue::Null));
    assert_eq!(row.get("center_time_pct"), Some(&CellValue::Null));
    assert_eq!(row.get("center_motion_distance_m"), Some(&CellValue::Null));
    assert_eq!(
        row.get("center_motion_median_speed"),
        Some(&CellValue::Null)
    );
    assert_eq!(
        row.get("center_motion_median_acceleration"),
        Some(&CellValue::Null)
    );
    assert_eq!(row.get("center_motion_freezing_s"), Some(&CellValue::Null));
    assert_eq!(row.get("periphery_entries"), Some(&CellValue::Null));
    assert_eq!(row.get("periphery_duration_s"), Some(&CellValue::Null));
    assert_eq!(row.get("periphery_distance_m"), Some(&CellValue::Null));
    assert_eq!(row.get("periphery_mean_speed"), Some(&CellValue::Null));
    assert_eq!(row.get("periphery_time_pct"), Some(&CellValue::Null));
    assert_eq!(
        row.get("periphery_motion_distance_m"),
        Some(&CellValue::Null)
    );
    assert_eq!(
        row.get("periphery_motion_median_speed"),
        Some(&CellValue::Null)
    );
    assert_eq!(
        row.get("periphery_motion_median_acceleration"),
        Some(&CellValue::Null)
    );
    assert_eq!(
        row.get("periphery_motion_freezing_s"),
        Some(&CellValue::Null)
    );

    // Posture metrics (absent) also Null
    assert_eq!(row.get("rearing_s"), Some(&CellValue::Null));
    assert_eq!(row.get("grooming_s"), Some(&CellValue::Null));
    assert_eq!(row.get("fecal_boli_count"), Some(&CellValue::Null));
    assert_eq!(row.get("immobility_pct"), Some(&CellValue::Null));

    // Excluded absent -> Null (downstream treats missing as "not excluded")
    assert_eq!(row.get("excluded"), Some(&CellValue::Null));
}

// -- YMazeExperiment -----------------------------------------------

fn ymaze_fixture_toml(zones_in_order: bool) -> String {
    let zones_in_a_b_c_center = r#"
        [[detailed.zone_summaries]]
        zone_name = "in_arm_A"
        entries = 6
        duration_seconds = 40.0
        distance_meters = 1.5
        mean_speed = 0.0375
        latency_seconds = 2.0

        [[detailed.zone_summaries]]
        zone_name = "in_arm_B"
        entries = 5
        duration_seconds = 38.0
        distance_meters = 1.4
        mean_speed = 0.037
        latency_seconds = 3.1

        [[detailed.zone_summaries]]
        zone_name = "in_arm_C"
        entries = 7
        duration_seconds = 45.0
        distance_meters = 1.7
        mean_speed = 0.038
        latency_seconds = 4.2

        [[detailed.zone_summaries]]
        zone_name = "in_center"
        entries = 17
        duration_seconds = 60.0
        distance_meters = 2.1
        mean_speed = 0.035
        latency_seconds = 0.0

        [[detailed.area_motion_summaries]]
        zone_name = "in_arm_A"
        total_displacement_meters = 1.4
        median_speed_mps = 0.032
        median_acceleration_mps2 = 0.14
        freezing_seconds = 2.1

        [[detailed.area_motion_summaries]]
        zone_name = "in_arm_B"
        total_displacement_meters = 1.3
        median_speed_mps = 0.031
        median_acceleration_mps2 = 0.13
        freezing_seconds = 1.8

        [[detailed.area_motion_summaries]]
        zone_name = "in_arm_C"
        total_displacement_meters = 1.6
        median_speed_mps = 0.034
        median_acceleration_mps2 = 0.15
        freezing_seconds = 2.4

        [[detailed.area_motion_summaries]]
        zone_name = "in_center"
        total_displacement_meters = 2.0
        median_speed_mps = 0.028
        median_acceleration_mps2 = 0.11
        freezing_seconds = 6.0

    "#;

    let zones_permuted = r#"
        [[detailed.zone_summaries]]
        zone_name = "in_arm_C"
        entries = 7
        duration_seconds = 45.0
        distance_meters = 1.7
        mean_speed = 0.038
        latency_seconds = 4.2

        [[detailed.zone_summaries]]
        zone_name = "in_center"
        entries = 17
        duration_seconds = 60.0
        distance_meters = 2.1
        mean_speed = 0.035
        latency_seconds = 0.0

        [[detailed.zone_summaries]]
        zone_name = "in_arm_A"
        entries = 6
        duration_seconds = 40.0
        distance_meters = 1.5
        mean_speed = 0.0375
        latency_seconds = 2.0

        [[detailed.zone_summaries]]
        zone_name = "in_arm_B"
        entries = 5
        duration_seconds = 38.0
        distance_meters = 1.4
        mean_speed = 0.037
        latency_seconds = 3.1

        [[detailed.area_motion_summaries]]
        zone_name = "in_arm_C"
        total_displacement_meters = 1.6
        median_speed_mps = 0.034
        median_acceleration_mps2 = 0.15
        freezing_seconds = 2.4

        [[detailed.area_motion_summaries]]
        zone_name = "in_center"
        total_displacement_meters = 2.0
        median_speed_mps = 0.028
        median_acceleration_mps2 = 0.11
        freezing_seconds = 6.0

        [[detailed.area_motion_summaries]]
        zone_name = "in_arm_A"
        total_displacement_meters = 1.4
        median_speed_mps = 0.032
        median_acceleration_mps2 = 0.14
        freezing_seconds = 2.1

        [[detailed.area_motion_summaries]]
        zone_name = "in_arm_B"
        total_displacement_meters = 1.3
        median_speed_mps = 0.031
        median_acceleration_mps2 = 0.13
        freezing_seconds = 1.8
    "#;

    let zones = if zones_in_order {
        zones_in_a_b_c_center
    } else {
        zones_permuted
    };

    format!(
        r#"
        experiment = "99_seq2_ymaze"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "y_maze"

        [animal_metadata]
        animal = 99
        cohort = "C"
        gene = "APPPS1"
        sex = "f"
        age = "12"

        [detailed]
        freezing_seconds = 12.3
        total_distance_meters = 6.7
        total_mean_speed = 0.04
        total_duration_seconds = 183.0
        analysis_duration_seconds = 180.0
        proximal_threshold = 0.65
        start_mode = "first_detection"
        zone_max_gap_seconds = 0.1
        arm_membership_hysteresis = true
        pose_gap_fill_seconds = 0.2
        seconds_in_arms = 123.0
        seconds_in_arms_and_center = 183.0

        [detailed.analysis]
        zone_sequence = ["A", "Center", "B", "Center", "C"]
        raw_arm_sequence = [
            "A", "A", "A", "A", "A", "A",
            "B", "B", "B", "B", "B",
            "C", "C", "C", "C", "C", "C", "C",
        ]
        arm_sequence = ["A", "B", "C", "A"]
        first_arm_entered = "A"
        last_arm_entered = "A"
        arm_entries = [6, 5, 7]
        center_entries = 17
        correct_alternations = 10
        incorrect_alternations = 5
        neutral_triads = 2
        spontaneous_alternation_pct = 66.67
        arm_preference = 2.94
        sum_of_permutation_distribution = 10

        [detailed.analysis.permutation_distribution]
        abc = 2
        acb = 1
        bac = 1
        bca = 3
        cab = 2
        cba = 1

        [detailed.analysis.incorrect_triad_distribution]
        aab = 1
        aac = 0
        aba = 2
        aca = 0
        bab = 0
        bba = 0
        bbc = 0
        bcb = 1
        cac = 1
        cbc = 0
        cca = 0
        ccb = 0

        [detailed.analysis.neutral_triad_distribution]
        aaa = 0
        abb = 1
        acc = 0
        baa = 0
        bbb = 0
        bcc = 1
        caa = 0
        cbb = 0
        ccc = 0

        {zones}
        "#
    )
}
