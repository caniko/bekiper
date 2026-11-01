// Coverage for the spec's NA-on-zero-denominator rules:
// - `spontaneous_alternation_pct` is NA when correct + incorrect == 0.
// - `spontaneous_alternation_pct_standard` is NA when total_triads == 0.
// - `neutral_triads_pct` is NA when total_triads == 0.
// Plus boundary coverage for the two quality flags
// (`low_valid_entries_flag`, `low_usable_triads_flag`).

fn empty_ymaze_trial_toml() -> &'static str {
    r#"
    experiment = "y_maze_videos_y_maze_pre_pre_video_0__proximal25"
    analysis_role = "ymaze__proximal25"
    fps = 30.0
    meters_per_pixel = 0.001
    task_type = "y_maze"

    [detailed]
    proximal_threshold = 0.25
    start_mode = "first_detection"
    arm_membership_hysteresis = true
    total_duration_seconds = 120.0
    analysis_duration_seconds = 120.0
    total_distance_meters = 0.0
    total_mean_speed = 0.0
    seconds_in_arms = 0.0
    seconds_in_arms_and_center = 0.0
    freezing_seconds = 0.0
    area_motion_summaries = []
    excluded = false

    [detailed.analysis]
    arm_sequence = []
    raw_arm_sequence = []
    zone_sequence = []
    correct_alternations = 0
    incorrect_alternations = 0
    neutral_triads = 0
    arm_preference = 0.0

    [[detailed.zone_summaries]]
    zone_name = "in_arm_A"
    entries = 0
    duration_seconds = 0.0

    [[detailed.zone_summaries]]
    zone_name = "in_arm_B"
    entries = 0
    duration_seconds = 0.0

    [[detailed.zone_summaries]]
    zone_name = "in_arm_C"
    entries = 0
    duration_seconds = 0.0

    [[detailed.zone_summaries]]
    zone_name = "in_center"
    entries = 0
    duration_seconds = 0.0
    "#
}

#[test]
fn ymaze_zero_denominator_emits_null_percentages() {
    let trial = parse(empty_ymaze_trial_toml());
    let row = YMazeExperiment.extract_row(&trial);

    assert_eq!(
        row.get("spontaneous_alternation_pct"),
        Some(&CellValue::Null),
        "SA% must be Null when correct + incorrect == 0",
    );
    assert_eq!(
        row.get("spontaneous_alternation_pct_standard"),
        Some(&CellValue::Null),
        "SA%_standard must be Null when total_triads == 0",
    );
    assert_eq!(
        row.get("neutral_triads_pct"),
        Some(&CellValue::Null),
        "neutral_triads_pct must be Null when total_triads == 0",
    );
}

#[test]
fn ymaze_zero_denominator_derived_counters_are_zero_not_null() {
    let trial = parse(empty_ymaze_trial_toml());
    let row = YMazeExperiment.extract_row(&trial);

    assert_eq!(row.get("raw_total_arm_entries"), Some(&CellValue::Int(0)));
    assert_eq!(row.get("valid_total_arm_entries"), Some(&CellValue::Int(0)));
    assert_eq!(row.get("removed_short_entries"), Some(&CellValue::Int(0)));
    assert_eq!(row.get("total_triads"), Some(&CellValue::Int(0)));
    assert_eq!(row.get("usable_triads"), Some(&CellValue::Int(0)));
    assert_eq!(row.get("neutral_triads"), Some(&CellValue::Int(0)));
    assert_eq!(row.get("correct_alternations"), Some(&CellValue::Int(0)));
    assert_eq!(row.get("incorrect_alternations"), Some(&CellValue::Int(0)));
}

#[test]
fn ymaze_low_flags_fire_for_zero_entry_trial() {
    let trial = parse(empty_ymaze_trial_toml());
    let row = YMazeExperiment.extract_row(&trial);

    assert_eq!(
        row.get("low_valid_entries_flag"),
        Some(&CellValue::Bool(true)),
        "valid_total = 0 < 8",
    );
    assert_eq!(
        row.get("low_usable_triads_flag"),
        Some(&CellValue::Bool(true)),
        "usable_triads = 0 < 5",
    );
}

fn ymaze_trial_with_counts(
    valid_total: usize,
    correct: i64,
    incorrect: i64,
    neutral: i64,
) -> String {
    // Build a valid_sequence of length `valid_total`. Same length on the raw
    // side so `removed_short_entries == 0`.
    let labels = ["A", "B", "C"];
    let seq: Vec<String> = (0..valid_total)
        .map(|idx| format!("\"{}\"", labels[idx % 3]))
        .collect();
    let seq_toml = seq.join(", ");

    format!(
        r#"
        experiment = "y_maze_videos_y_maze_pre_pre_video_1__proximal25"
        analysis_role = "ymaze__proximal25"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "y_maze"

        [detailed]
        proximal_threshold = 0.25
        start_mode = "first_detection"
        arm_membership_hysteresis = true
        total_duration_seconds = 120.0
        analysis_duration_seconds = 120.0
        total_distance_meters = 0.0
        total_mean_speed = 0.0
        seconds_in_arms = 0.0
        seconds_in_arms_and_center = 0.0
        freezing_seconds = 0.0
        area_motion_summaries = []
        excluded = false

        [detailed.analysis]
        arm_sequence = [{seq_toml}]
        raw_arm_sequence = [{seq_toml}]
        zone_sequence = [{seq_toml}]
        correct_alternations = {correct}
        incorrect_alternations = {incorrect}
        neutral_triads = {neutral}
        arm_preference = 0.0

        [[detailed.zone_summaries]]
        zone_name = "in_arm_A"
        entries = 0
        duration_seconds = 0.0

        [[detailed.zone_summaries]]
        zone_name = "in_arm_B"
        entries = 0
        duration_seconds = 0.0

        [[detailed.zone_summaries]]
        zone_name = "in_arm_C"
        entries = 0
        duration_seconds = 0.0

        [[detailed.zone_summaries]]
        zone_name = "in_center"
        entries = 0
        duration_seconds = 0.0
        "#,
    )
}

#[test]
fn ymaze_low_valid_entries_flag_boundary_at_eight() {
    // valid_total == 7 → flag fires.
    let trial = parse(&ymaze_trial_with_counts(7, 0, 0, 0));
    let row = YMazeExperiment.extract_row(&trial);
    assert_eq!(
        row.get("low_valid_entries_flag"),
        Some(&CellValue::Bool(true))
    );

    // valid_total == 8 → flag clears.
    let trial = parse(&ymaze_trial_with_counts(8, 0, 0, 0));
    let row = YMazeExperiment.extract_row(&trial);
    assert_eq!(
        row.get("low_valid_entries_flag"),
        Some(&CellValue::Bool(false))
    );
}

#[test]
fn ymaze_low_usable_triads_flag_boundary_at_five() {
    // usable == 4 → flag fires.
    let trial = parse(&ymaze_trial_with_counts(6, 2, 2, 0));
    let row = YMazeExperiment.extract_row(&trial);
    assert_eq!(row.get("usable_triads"), Some(&CellValue::Int(4)));
    assert_eq!(
        row.get("low_usable_triads_flag"),
        Some(&CellValue::Bool(true))
    );

    // usable == 5 → flag clears.
    let trial = parse(&ymaze_trial_with_counts(7, 3, 2, 0));
    let row = YMazeExperiment.extract_row(&trial);
    assert_eq!(row.get("usable_triads"), Some(&CellValue::Int(5)));
    assert_eq!(
        row.get("low_usable_triads_flag"),
        Some(&CellValue::Bool(false))
    );
}

#[test]
fn ymaze_neutral_triads_pct_uses_total_triads_denominator() {
    // 7 valid entries → total_triads = 5. neutral = 1 → 20%.
    let trial = parse(&ymaze_trial_with_counts(7, 2, 2, 1));
    let row = YMazeExperiment.extract_row(&trial);

    assert_eq!(row.get("total_triads"), Some(&CellValue::Int(5)));
    match row.get("neutral_triads_pct") {
        Some(CellValue::Float(value)) => {
            assert!(
                (*value - 20.0).abs() < 1e-9,
                "neutral_triads_pct = {value}, expected 20.0"
            );
        }
        other => panic!("expected neutral_triads_pct Float, got {other:?}"),
    }
}

#[test]
fn ymaze_spontaneous_alternation_pct_standard_uses_total_triads() {
    // valid_total = 7 → total_triads = 5. correct = 3, incorrect = 2,
    // neutral = 0 → SA%_standard = 3/5 = 60%.
    let trial = parse(&ymaze_trial_with_counts(7, 3, 2, 0));
    let row = YMazeExperiment.extract_row(&trial);

    match row.get("spontaneous_alternation_pct_standard") {
        Some(CellValue::Float(value)) => {
            assert!(
                (*value - 60.0).abs() < 1e-9,
                "SA%_standard = {value}, expected 60.0"
            );
        }
        other => panic!("expected SA%_standard Float, got {other:?}"),
    }
}
