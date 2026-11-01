//! Golden snapshot for the Y-maze aggregate entry-count projection.
//!
//! To regenerate after an intentional schema or semantic change:
//!
//! ```text
//! UPDATE_GOLDEN=1 cargo test -p bekiper-ingress --test test_golden_ymaze_entries
//! ```

use std::path::Path;

use bekiper_ingress::aggregate::{TrialResult, build_sheet_dataframes};
use polars::prelude::*;
use serde::Serialize;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/golden_ymaze_entries.toml",
);

#[derive(Debug, Serialize)]
struct GoldenRows {
    trial: Vec<GoldenTrial>,
}

#[derive(Debug, Serialize)]
struct GoldenTrial {
    animal_trial_id: String,
    #[serde(rename = "raw_arm_A_entries")]
    raw_arm_a_entries: i64,
    #[serde(rename = "raw_arm_B_entries")]
    raw_arm_b_entries: i64,
    #[serde(rename = "raw_arm_C_entries")]
    raw_arm_c_entries: i64,
    raw_total_arm_entries: i64,
    removed_short_entries: i64,
    raw_arm_entry_sequence: String,
    #[serde(rename = "valid_arm_A_entries")]
    valid_arm_a_entries: i64,
    #[serde(rename = "valid_arm_B_entries")]
    valid_arm_b_entries: i64,
    #[serde(rename = "valid_arm_C_entries")]
    valid_arm_c_entries: i64,
    valid_total_arm_entries: i64,
    valid_arm_entry_sequence: Option<String>,
    spontaneous_alternation_pct: Option<f64>,
    spontaneous_alternation_pct_standard: Option<f64>,
    total_triads: i64,
    usable_triads: i64,
    neutral_triads_pct: Option<f64>,
    low_valid_entries_flag: bool,
    low_usable_triads_flag: bool,
}

#[test]
fn golden_ymaze_entries_match() {
    let actual = build_golden_rows();
    let actual_toml = toml::Value::try_from(&actual).expect("serialize golden projection");

    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        write_pretty_toml(Path::new(GOLDEN), &actual_toml);
        return;
    }

    let expected: toml::Value = toml::from_str(
        &std::fs::read_to_string(GOLDEN)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", GOLDEN)),
    )
    .expect("golden fixture TOML parses");
    assert_eq!(
        actual_toml, expected,
        "golden mismatch; rerun with UPDATE_GOLDEN=1 if the Y-maze aggregate change is intentional"
    );
}

fn write_pretty_toml(path: &Path, value: &toml::Value) {
    let mut rendered = toml::to_string_pretty(value).expect("render golden TOML");
    if !rendered.ends_with('\n') {
        rendered.push('\n');
    }
    std::fs::write(path, rendered)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

fn build_golden_rows() -> GoldenRows {
    let trials = vec![
        ymaze_trial(700, 0, "fixture_zero_valid", &[], &["A", "B"], 0, 0, 0, 0.0),
        ymaze_trial(
            701,
            1,
            "fixture_short",
            &["A", "B", "C"],
            &["A", "B", "C"],
            1,
            0,
            0,
            100.0,
        ),
        ymaze_trial(
            702,
            2,
            "fixture_mid_noise",
            &["A", "B", "A", "C", "B"],
            &["A", "A", "B", "A", "C", "B"],
            2,
            1,
            0,
            66.67,
        ),
        ymaze_trial(
            703,
            3,
            "fixture_long_reentry",
            &["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            &["A", "B", "C", "A", "A", "B", "C", "C", "A", "B", "C", "A"],
            8,
            0,
            0,
            100.0,
        ),
    ];

    let sheets = build_sheet_dataframes(trials, &Default::default()).expect("aggregate sheets");
    assert_eq!(sheets.len(), 1);
    let (_, df, _) = sheets.into_iter().next().expect("Y-maze sheet");

    let mut trial = (0..df.height())
        .map(|row| GoldenTrial {
            animal_trial_id: string_cell(&df, "animal_trial_id", row),
            raw_arm_a_entries: int_cell(&df, "raw_arm_A_entries", row),
            raw_arm_b_entries: int_cell(&df, "raw_arm_B_entries", row),
            raw_arm_c_entries: int_cell(&df, "raw_arm_C_entries", row),
            raw_total_arm_entries: int_cell(&df, "raw_total_arm_entries", row),
            removed_short_entries: int_cell(&df, "removed_short_entries", row),
            raw_arm_entry_sequence: string_cell(&df, "raw_arm_entry_sequence", row),
            valid_arm_a_entries: int_cell(&df, "valid_arm_A_entries", row),
            valid_arm_b_entries: int_cell(&df, "valid_arm_B_entries", row),
            valid_arm_c_entries: int_cell(&df, "valid_arm_C_entries", row),
            valid_total_arm_entries: int_cell(&df, "valid_total_arm_entries", row),
            valid_arm_entry_sequence: optional_string_cell(&df, "valid_arm_entry_sequence", row),
            spontaneous_alternation_pct: optional_float_cell(
                &df,
                "spontaneous_alternation_pct",
                row,
            ),
            spontaneous_alternation_pct_standard: optional_float_cell(
                &df,
                "spontaneous_alternation_pct_standard",
                row,
            ),
            total_triads: int_cell(&df, "total_triads", row),
            usable_triads: int_cell(&df, "usable_triads", row),
            neutral_triads_pct: optional_float_cell(&df, "neutral_triads_pct", row),
            low_valid_entries_flag: bool_cell(&df, "low_valid_entries_flag", row),
            low_usable_triads_flag: bool_cell(&df, "low_usable_triads_flag", row),
        })
        .collect::<Vec<_>>();
    trial.sort_by(|left, right| left.animal_trial_id.cmp(&right.animal_trial_id));

    GoldenRows { trial }
}

#[allow(clippy::too_many_arguments)]
fn ymaze_trial(
    animal: u32,
    trial: u32,
    experiment: &str,
    valid_sequence: &[&str],
    raw_sequence: &[&str],
    correct_alternations: i64,
    incorrect_alternations: i64,
    neutral_triads: i64,
    spontaneous_alternation_pct: f64,
) -> TrialResult {
    let valid_sequence_toml = sequence_toml(valid_sequence);
    let raw_sequence_toml = sequence_toml(raw_sequence);
    let first = valid_sequence.first().copied().unwrap_or("");
    let last = valid_sequence.last().copied().unwrap_or("");
    let (raw_a, raw_b, raw_c) = arm_counts(raw_sequence);

    toml::from_str(&format!(
        r#"
        experiment = "{experiment}"
        analysis_role = "ymaze__golden"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "y_maze"

        [animal_metadata]
        animal = {animal}

        [session_context]
        trial = {trial}

        [detailed]
        proximal_threshold = 0.25
        start_mode = "first_detection"
        zone_max_gap_seconds = 0.1
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
    ))
    .expect("synthetic Y-maze TrialResult parses")
}

fn sequence_toml(sequence: &[&str]) -> String {
    sequence
        .iter()
        .map(|label| format!("\"{label}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

fn arm_counts(sequence: &[&str]) -> (usize, usize, usize) {
    (
        sequence.iter().filter(|label| **label == "A").count(),
        sequence.iter().filter(|label| **label == "B").count(),
        sequence.iter().filter(|label| **label == "C").count(),
    )
}

fn string_cell(df: &DataFrame, column: &str, row: usize) -> String {
    match df.column(column).unwrap().get(row).unwrap() {
        AnyValue::String(value) => value.to_string(),
        AnyValue::StringOwned(value) => value.to_string(),
        other => panic!("expected {column} row {row} to be a string, got {other:?}"),
    }
}

fn optional_string_cell(df: &DataFrame, column: &str, row: usize) -> Option<String> {
    match df.column(column).unwrap().get(row).unwrap() {
        AnyValue::String(value) => Some(value.to_string()),
        AnyValue::StringOwned(value) => Some(value.to_string()),
        AnyValue::Null => None,
        other => panic!("expected {column} row {row} to be a string or null, got {other:?}"),
    }
}

fn int_cell(df: &DataFrame, column: &str, row: usize) -> i64 {
    match df.column(column).unwrap().get(row).unwrap() {
        AnyValue::Int64(value) => value,
        AnyValue::Int32(value) => value.into(),
        AnyValue::UInt32(value) => value.into(),
        other => panic!("expected {column} row {row} to be an integer, got {other:?}"),
    }
}

fn optional_float_cell(df: &DataFrame, column: &str, row: usize) -> Option<f64> {
    match df.column(column).unwrap().get(row).unwrap() {
        AnyValue::Float64(value) => Some(value),
        AnyValue::Float32(value) => Some(value.into()),
        AnyValue::Null => None,
        other => panic!("expected {column} row {row} to be a float or null, got {other:?}"),
    }
}

fn bool_cell(df: &DataFrame, column: &str, row: usize) -> bool {
    match df.column(column).unwrap().get(row).unwrap() {
        AnyValue::Boolean(value) => value,
        other => panic!("expected {column} row {row} to be a boolean, got {other:?}"),
    }
}
