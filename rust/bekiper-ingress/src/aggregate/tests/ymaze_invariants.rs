use polars::prelude::{AnyValue, DataFrame};

#[test]
fn ymaze_imen_proximal_sheet_invariants_hold() {
    let cfg = AggregateConfig {
        recipe_to_sheet: [
            (
                "ymaze__proximal25".to_string(),
                "ymaze_proximal25".to_string(),
            ),
            (
                "ymaze__proximal45".to_string(),
                "ymaze_proximal45".to_string(),
            ),
            (
                "ymaze__proximal65".to_string(),
                "ymaze_proximal65".to_string(),
            ),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    let rows = vec![
        parse(&imen_ymaze_trial(
            10,
            "proximal25",
            0.25,
            &["A", "B", "C"],
            &["A", "B", "C"],
            1,
            0,
            0,
            100.0,
        )),
        parse(&imen_ymaze_trial(
            11,
            "proximal45",
            0.45,
            &["A", "B", "A", "C", "B"],
            &["A", "A", "B", "A", "C", "B"],
            2,
            1,
            0,
            66.67,
        )),
        parse(&imen_ymaze_trial(
            12,
            "proximal65",
            0.65,
            &["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            &["A", "B", "C", "A", "A", "B", "C", "C", "A", "B", "C", "A"],
            8,
            0,
            0,
            100.0,
        )),
    ];

    let sheets = build_sheet_dataframes(rows, &cfg).expect("aggregate sheets");
    let sheet_names = sheets
        .iter()
        .map(|(sheet_name, _, _)| sheet_name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        sheet_names,
        ["ymaze_proximal25", "ymaze_proximal45", "ymaze_proximal65"]
    );

    for (sheet_name, df, _) in sheets {
        assert!(
            sheet_name.starts_with("ymaze_proximal"),
            "unexpected Y-maze sheet {sheet_name}"
        );
        assert_no_removed_columns(&sheet_name, &df);
        for row in 0..df.height() {
            assert_ymaze_entry_invariants(&sheet_name, &df, row);
        }
    }
}

fn assert_no_removed_columns(sheet_name: &str, df: &DataFrame) {
    let removed_columns = df
        .get_column_names()
        .iter()
        .map(|column| column.as_str())
        .filter(|column| column.starts_with("dist"))
        .collect::<Vec<_>>();
    assert!(
        removed_columns.is_empty(),
        "{sheet_name}: unexpected removed columns: {removed_columns:?}"
    );
}

fn assert_ymaze_entry_invariants(sheet_name: &str, df: &DataFrame, row: usize) {
    let raw_a = i64_cell(df, "raw_arm_A_entries", row);
    let raw_b = i64_cell(df, "raw_arm_B_entries", row);
    let raw_c = i64_cell(df, "raw_arm_C_entries", row);
    let raw_total = i64_cell(df, "raw_total_arm_entries", row);
    let valid_a = i64_cell(df, "valid_arm_A_entries", row);
    let valid_b = i64_cell(df, "valid_arm_B_entries", row);
    let valid_c = i64_cell(df, "valid_arm_C_entries", row);
    let valid_total = i64_cell(df, "valid_total_arm_entries", row);
    let total_triads = i64_cell(df, "total_triads", row);
    let removed_short_entries = i64_cell(df, "removed_short_entries", row);
    let usable_triads = i64_cell(df, "usable_triads", row);
    let correct_alternations = i64_cell(df, "correct_alternations", row);
    let incorrect_alternations = i64_cell(df, "incorrect_alternations", row);
    let low_valid_entries_flag = bool_cell(df, "low_valid_entries_flag", row);
    let low_usable_triads_flag = bool_cell(df, "low_usable_triads_flag", row);

    assert_eq!(
        removed_short_entries + valid_total,
        raw_total,
        "{sheet_name} row {row}: removed_short_entries must remain raw_total_arm_entries - valid_total_arm_entries even when it is negative"
    );
    assert_eq!(
        valid_a + valid_b + valid_c,
        valid_total,
        "{sheet_name} row {row}: valid per-arm entries do not sum to valid_total_arm_entries"
    );
    assert_eq!(
        raw_a + raw_b + raw_c,
        raw_total,
        "{sheet_name} row {row}: raw per-arm entries do not sum to raw_total_arm_entries"
    );
    assert_eq!(
        total_triads,
        (valid_total - 2).max(0),
        "{sheet_name} row {row}: total_triads should equal max(valid_total_arm_entries - 2, 0)"
    );
    assert_eq!(
        usable_triads,
        correct_alternations + incorrect_alternations,
        "{sheet_name} row {row}: usable_triads should equal correct_alternations + incorrect_alternations"
    );
    assert_eq!(
        low_valid_entries_flag,
        valid_total < 8,
        "{sheet_name} row {row}: low_valid_entries_flag should track valid_total_arm_entries < 8"
    );
    assert_eq!(
        low_usable_triads_flag,
        usable_triads < 5,
        "{sheet_name} row {row}: low_usable_triads_flag should track usable_triads < 5"
    );
}

#[allow(clippy::too_many_arguments)]
fn imen_ymaze_trial(
    video_id: u32,
    variant: &str,
    proximal_threshold: f64,
    valid_sequence: &[&str],
    raw_sequence: &[&str],
    correct_alternations: i64,
    incorrect_alternations: i64,
    neutral_triads: i64,
    spontaneous_alternation_pct: f64,
) -> String {
    let experiment = format!("y_maze_videos_y_maze_pre_pre_video_{video_id}__{variant}");
    let analysis_role = format!("ymaze__{variant}");
    let valid_sequence_toml = sequence_toml_for_invariants(valid_sequence);
    let raw_sequence_toml = sequence_toml_for_invariants(raw_sequence);
    let first = valid_sequence.first().copied().unwrap_or("");
    let last = valid_sequence.last().copied().unwrap_or("");
    let (raw_a, raw_b, raw_c) = arm_counts_for_invariants(raw_sequence);

    format!(
        r#"
        experiment = "{experiment}"
        analysis_role = "{analysis_role}"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "y_maze"

        [detailed]
        proximal_threshold = {proximal_threshold}
        start_mode = "first_detection"
        arm_membership_hysteresis = true
        pose_gap_fill_seconds = 0.2
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
        arm_sequence = [{valid_sequence_toml}]
        raw_arm_sequence = [{raw_sequence_toml}]
        zone_sequence = [{raw_sequence_toml}]
        first_arm_entered = "{first}"
        last_arm_entered = "{last}"
        correct_alternations = {correct_alternations}
        incorrect_alternations = {incorrect_alternations}
        neutral_triads = {neutral_triads}
        spontaneous_alternation_pct = {spontaneous_alternation_pct}
        arm_preference = 0.25

        [[detailed.zone_summaries]]
        zone_name = "in_arm_A"
        entries = {raw_a}
        duration_seconds = 30.0

        [[detailed.zone_summaries]]
        zone_name = "in_arm_B"
        entries = {raw_b}
        duration_seconds = 30.0

        [[detailed.zone_summaries]]
        zone_name = "in_arm_C"
        entries = {raw_c}
        duration_seconds = 30.0

        [[detailed.zone_summaries]]
        zone_name = "in_center"
        entries = 1
        duration_seconds = 10.0
        "#
    )
}

fn sequence_toml_for_invariants(sequence: &[&str]) -> String {
    sequence
        .iter()
        .map(|label| format!("\"{label}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

fn arm_counts_for_invariants(sequence: &[&str]) -> (usize, usize, usize) {
    (
        sequence.iter().filter(|label| **label == "A").count(),
        sequence.iter().filter(|label| **label == "B").count(),
        sequence.iter().filter(|label| **label == "C").count(),
    )
}

fn i64_cell(df: &DataFrame, column: &str, row: usize) -> i64 {
    match df.column(column).unwrap().get(row).unwrap() {
        AnyValue::Int64(value) => value,
        AnyValue::Int32(value) => value.into(),
        AnyValue::UInt32(value) => value.into(),
        other => panic!("expected {column} row {row} to be an integer, got {other:?}"),
    }
}

fn bool_cell(df: &DataFrame, column: &str, row: usize) -> bool {
    match df.column(column).unwrap().get(row).unwrap() {
        AnyValue::Boolean(value) => value,
        other => panic!("expected {column} row {row} to be a boolean, got {other:?}"),
    }
}
