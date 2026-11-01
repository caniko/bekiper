#[test]
fn cheeseboard_extract_uses_typed_columns() {
    let t = parse(
        r#"
        experiment = "555_seq1_cheeseboard"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "cheeseboard"

        [animal_metadata]
        animal = 555
        cohort = "D"
        gene = "wt"
        sex = "m"
        age = "16"

        [detailed]
        excluded = false
        enclosure_label = "board"
        reward_labels = ["reward_a", "reward_b"]
        start_area_exit_frame = 12
        start_area_exit_seconds = 0.4
        reward_arrival_frame = 18
        reward_arrival_seconds = 0.6
        seconds_to_find_reward = 0.2
        seconds_in_reward_area = 1.5
        reward_entries = 2
        search_path_distance_meters = 1.25
        search_direct_distance_meters = 1.0
        search_mean_speed_mps = 2.5
        search_tortuosity = 1.25
        total_distance_meters = 9.5
        total_mean_speed_mps = 0.04
        total_duration_seconds = 240.0

        [detailed.enclosure_summary]
        entries = 3
        duration_seconds = 90.0
        distance_meters = 7.5
        mean_speed = 0.0833333333
        latency_seconds = 0.5
        "#,
    );

    let row = CheeseboardExperiment.extract_row(&t);

    assert_eq!(row.get("animal"), Some(&CellValue::Int(555)));
    assert_eq!(row.get("cohort"), Some(&CellValue::String("D".into())));
    assert_eq!(row.get("gene"), Some(&CellValue::String("wt".into())));
    assert_eq!(row.get("fps"), Some(&CellValue::Float(30.0)));
    assert_eq!(
        row.get("enclosure_label"),
        Some(&CellValue::String("board".into()))
    );
    assert_eq!(row.get("enclosure_entries"), Some(&CellValue::Int(3)));
    assert_eq!(
        row.get("enclosure_duration_s"),
        Some(&CellValue::Float(90.0))
    );
    assert_eq!(
        row.get("enclosure_distance_m"),
        Some(&CellValue::Float(7.5))
    );
    assert_eq!(
        row.get("enclosure_mean_speed_mps"),
        Some(&CellValue::Float(0.0833333333))
    );
    assert_eq!(row.get("enclosure_latency_s"), Some(&CellValue::Float(0.5)));
    assert_eq!(
        row.get("reward_labels"),
        Some(&CellValue::String("reward_a|reward_b".into()))
    );
    assert_eq!(row.get("reward_count"), Some(&CellValue::Int(2)));
    assert_eq!(row.get("start_area_exit_frame"), Some(&CellValue::Int(12)));
    assert_eq!(
        row.get("start_area_exit_seconds"),
        Some(&CellValue::Float(0.4))
    );
    assert_eq!(row.get("reward_arrival_frame"), Some(&CellValue::Int(18)));
    assert_eq!(
        row.get("reward_arrival_seconds"),
        Some(&CellValue::Float(0.6))
    );
    assert_eq!(
        row.get("seconds_to_find_reward"),
        Some(&CellValue::Float(0.2))
    );
    assert_eq!(
        row.get("seconds_in_reward_area"),
        Some(&CellValue::Float(1.5))
    );
    assert_eq!(row.get("reward_entries"), Some(&CellValue::Int(2)));
    assert_eq!(
        row.get("search_path_distance_meters"),
        Some(&CellValue::Float(1.25))
    );
    assert_eq!(
        row.get("search_direct_distance_meters"),
        Some(&CellValue::Float(1.0))
    );
    assert_eq!(
        row.get("search_mean_speed_mps"),
        Some(&CellValue::Float(2.5))
    );
    assert_eq!(row.get("search_tortuosity"), Some(&CellValue::Float(1.25)));
    assert_eq!(
        row.get("total_distance_meters"),
        Some(&CellValue::Float(9.5))
    );
    assert_eq!(
        row.get("total_mean_speed_mps"),
        Some(&CellValue::Float(0.04))
    );
    assert_eq!(
        row.get("total_duration_seconds"),
        Some(&CellValue::Float(240.0))
    );
    assert_eq!(
        row.get("experiment"),
        Some(&CellValue::String("555_seq1_cheeseboard".into()))
    );
    assert_eq!(row.get("excluded"), Some(&CellValue::Bool(false)));
}

#[test]
fn generic_extract_preserves_metadata_on_empty_detailed() {
    let t = parse(
        r#"
        experiment = "empty_seq0_cb"
        fps = 24.0
        meters_per_pixel = 0.001
        task_type = "cheeseboard"

        [animal_metadata]
        animal = 1
        cohort = "Z"
        gene = "wt"
        sex = "m"
        age = "5"

        [detailed]
        "#,
    );
    let row = GenericExperiment.extract_row(&t);

    // Only the metadata prefix + experiment + excluded should be present.
    let mut keys: Vec<&String> = row.keys().collect();
    keys.sort();
    let mut expected = vec![
        "age",
        "animal",
        "cohort",
        "excluded",
        "experiment",
        "fps",
        "gene",
        "sex",
    ];
    expected.sort();
    let keys_str: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    assert_eq!(keys_str, expected);

    assert_eq!(row.get("animal"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("fps"), Some(&CellValue::Float(24.0)));
    assert_eq!(
        row.get("experiment"),
        Some(&CellValue::String("empty_seq0_cb".into()))
    );
    assert_eq!(row.get("excluded"), Some(&CellValue::Null));
}

#[test]
fn generic_extract_omits_missing_cheeseboard_start_fields() {
    let t = parse(
        r#"
        experiment = "missing_start_seq0_cb"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "cheeseboard"

        [animal_metadata]
        animal = 77
        cohort = "E"
        gene = "wt"
        sex = "f"
        age = "14"

        [detailed]
        excluded = false
        enclosure_label = "board"
        reward_labels = ["reward_a"]
        reward_arrival_frame = 8
        reward_arrival_seconds = 0.2
        seconds_in_reward_area = 4.0
        reward_entries = 1
        search_path_distance_meters = 0.45
        search_direct_distance_meters = 0.41
        search_mean_speed_mps = 2.25
        search_tortuosity = 1.1
        total_distance_meters = 8.0
        total_mean_speed_mps = 0.33
        total_duration_seconds = 240.0
        "#,
    );

    let row = GenericExperiment.extract_row(&t);

    assert_eq!(row.get("animal"), Some(&CellValue::Int(77)));
    assert_eq!(row.get("reward_arrival_frame"), Some(&CellValue::Int(8)));
    assert_eq!(
        row.get("start_area_exit_frame"),
        None,
        "missing Cheeseboard start-area fields should stay absent in the flattened export"
    );
    assert_eq!(
        row.get("start_area_exit_seconds"),
        None,
        "missing Cheeseboard start-area fields should stay absent in the flattened export"
    );
    assert_eq!(
        row.get("seconds_to_find_reward"),
        None,
        "missing Cheeseboard latency should stay absent in the flattened export"
    );
    assert_eq!(
        row.get("reward_labels[0]"),
        Some(&CellValue::String("reward_a".into()))
    );
}

// -- experiment_for dispatcher --------------------------------------

#[test]
fn experiment_for_trial_dispatches_by_task_type() {
    // Spot-check: each dispatcher arm returns an extractor whose
    // static column schema contains a task-specific marker column.
    let nort = experiment_for_trial(&parse(
        r#"
        experiment = "42_seq0_nortnovelty"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"

        [detailed]
        protocol = "nort_novelty"
        "#,
    ));
    assert!(nort.static_columns().iter().any(|c| c == "DI"));

    let oft = experiment_for_trial(&parse(
        r#"
        experiment = "42_seq0_oft"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "open_field"

        [detailed]
        excluded = false
        "#,
    ));
    assert!(oft.static_columns().iter().any(|c| c == "center_entries"));
    assert!(
        oft.static_columns()
            .iter()
            .any(|c| c == "periphery_entries")
    );

    let ymaze = experiment_for_trial(&parse(
        r#"
        experiment = "42_seq0_ymaze"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "y_maze"

        [detailed]
        excluded = false
        "#,
    ));
    assert!(
        ymaze
            .static_columns()
            .iter()
            .any(|c| c == "raw_arm_A_entries")
    );
    assert!(
        ymaze
            .static_columns()
            .iter()
            .any(|c| c == "spontaneous_alternation_pct")
    );

    let cheeseboard = experiment_for_trial(&parse(
        r#"
        experiment = "42_seq0_cheeseboard"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "cheeseboard"

        [detailed]
        excluded = false
        "#,
    ));
    assert!(
        cheeseboard
            .static_columns()
            .iter()
            .any(|c| c == "start_area_exit_seconds")
    );
    assert!(
        cheeseboard
            .static_columns()
            .iter()
            .any(|c| c == "reward_labels")
    );
    assert!(
        cheeseboard
            .static_columns()
            .iter()
            .any(|c| c == "enclosure_entries")
    );
}

// -- build_sheet_dataframes ----------------------------------------

/// Build a single NORT trial with tweakable knobs. Keeps the tests
/// below readable — the TOML literal needs to stay valid for
/// [`TrialResult::load`] round-trip, so we use format!() here rather
/// than an ad-hoc struct builder.
#[allow(clippy::too_many_arguments)]
fn nort_trial(
    experiment: &str,
    animal: i64,
    total_distance_m: f64,
    freezing_s: f64,
    excluded: bool,
    object_labels: &[&str],
    exploration_times: &[f64],
    bout_counts: &[i64],
) -> TrialResult {
    let labels_toml = object_labels
        .iter()
        .map(|l| format!("\"{l}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let exp_times_toml = exploration_times
        .iter()
        .map(|v| format!("{v:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let bouts_toml = bout_counts
        .iter()
        .map(|v| format!("{v}"))
        .collect::<Vec<_>>()
        .join(", ");
    parse(&format!(
        r#"
        experiment = "{experiment}"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"

        [animal_metadata]
        animal = {animal}
        cohort = "cohortA"
        gene = "wt"
        sex = "m"
        age = "10"

        [detailed]
        excluded = {excluded}
        total_duration_seconds = 300.0
        total_distance_meters = {total_distance_m:?}
        total_mean_speed = 0.05
        freezing_seconds = {freezing_s:?}
        analysis_window_seconds = 120.0
        object_labels = [{labels_toml}]
        exploration_times = [{exp_times_toml}]
        bout_counts = [{bouts_toml}]
        "#
    ))
}
