use bekiper_core::types::TaskType;
use serde_json::Value;

use super::*;
use crate::config::{AggregateConfig, AggregationRule};

fn parse(toml_str: &str) -> TrialResult {
    toml::from_str(toml_str).expect("valid TrialResult TOML")
}

// -- parse_recipe_from_experiment -----------------------------------

#[test]
fn nort_parse_recipe_basic() {
    assert_eq!(
        parse_recipe_from_experiment("1035_seq3_oulupdate"),
        Some("oulupdate".to_string())
    );
    assert_eq!(
        parse_recipe_from_experiment("42_seq0_oultraining"),
        Some("oultraining".to_string())
    );
}

#[test]
fn nort_parse_recipe_preserves_underscored_recipe() {
    // The recipe segment is everything after the second underscore, so
    // compound recipe names survive intact.
    assert_eq!(
        parse_recipe_from_experiment("42_seq0_nort_training"),
        Some("nort_training".to_string())
    );
}

#[test]
fn parse_recipe_preserves_variant_suffix_for_legacy_names() {
    assert_eq!(
        parse_recipe_from_experiment("1035_seq3_ymaze"),
        Some("ymaze".to_string())
    );
    assert_eq!(
        parse_recipe_from_experiment("1035_seq3_ymaze__proximal65"),
        Some("ymaze__proximal65".to_string())
    );
}

fn routing_key_for(trial: &TrialResult) -> Option<String> {
    experiment_for_trial(trial)
        .sheet_key(trial)
        .map(|sheet_key| sheet_key.routing_key())
}

#[test]
fn nort_parse_recipe_rejects_malformed() {
    assert_eq!(parse_recipe_from_experiment("1035_seq_oulupdate"), None);
    assert_eq!(parse_recipe_from_experiment("just_two"), None);
    assert_eq!(parse_recipe_from_experiment(""), None);
    assert_eq!(parse_recipe_from_experiment("nostructure"), None);
}

#[test]
fn aggregate_sheet_key_uses_structured_fields_for_expanded_imen_names() {
    let nort = parse(
        r#"
        experiment = "nort_videos_nort_pre_t2_config_b_pre_t2_configb_video_13"
        fps = 15.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"

        [detailed]
        protocol = "nort_novelty"
        "#,
    );
    assert_eq!(routing_key_for(&nort), Some("nort_novelty".into()));

    let open_field = parse(
        r#"
        experiment = "oft_videos_oft_pre_pre_video_1"
        fps = 15.0
        meters_per_pixel = 0.001
        task_type = "open_field"
        "#,
    );
    assert_eq!(
        routing_key_for(&open_field),
        Some("nort_habituation".into())
    );

    let role = parse(
        r#"
        experiment = "legacy_or_renamed_trial"
        analysis_role = "nort_habituation"
        fps = 15.0
        meters_per_pixel = 0.001
        task_type = "open_field"
        "#,
    );
    assert_eq!(routing_key_for(&role), Some("nort_habituation".into()));

    let true_open_field = parse(
        r#"
        experiment = "open_field_videos_pre_video_1"
        fps = 15.0
        meters_per_pixel = 0.001
        task_type = "open_field"
        "#,
    );
    assert_eq!(routing_key_for(&true_open_field), Some("open_field".into()));

    let ymaze = parse(
        r#"
        experiment = "y_maze_videos_y_maze_pre_pre_video_14"
        fps = 15.0
        meters_per_pixel = 0.001
        task_type = "y_maze"
        "#,
    );
    assert_eq!(routing_key_for(&ymaze), Some("y_maze".into()));
}

#[test]
fn aggregate_sheet_key_infers_imen_nort_protocol_from_expanded_name() {
    let t1 = parse(
        r#"
        experiment = "nort_videos_nort_pre_t1_config_a_pre_t1_configa_video_22"
        fps = 15.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"
        "#,
    );
    assert_eq!(routing_key_for(&t1), Some("nort_training".into()));

    let t2 = parse(
        r#"
        experiment = "nort_videos_nort_pre_t2_config_b_pre_t2_configb_video_13"
        fps = 15.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"
        "#,
    );
    assert_eq!(routing_key_for(&t2), Some("nort_novelty".into()));
}

#[test]
fn aggregate_sheet_key_preserves_legacy_recipe_names() {
    let trial = parse(
        r#"
        experiment = "1035_seq3_oulupdate"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"
        "#,
    );
    assert_eq!(routing_key_for(&trial), Some("oulupdate".into()));
}

#[test]
fn aggregate_sheet_key_uses_analysis_role_for_descriptive_variant_names() {
    assert_eq!(
        parse_recipe_from_experiment("y_maze_videos_y_maze_pre_pre_video_14__proximal45"),
        None
    );

    let ymaze = parse(
        r#"
        experiment = "y_maze_videos_y_maze_pre_pre_video_14__proximal45"
        analysis_role = "ymaze__proximal45"
        fps = 15.0
        meters_per_pixel = 0.001
        task_type = "y_maze"
        "#,
    );
    assert_eq!(routing_key_for(&ymaze), Some("ymaze__proximal45".into()));
}

// -- TrialResult deserialization ------------------------------------

#[test]
fn nort_trial_result_parses_real_result_toml_shape() {
    // Mirrors the actual shape of result.toml written by run_summary()
    // for an OUL update trial (see
    // examples/neo/dejana/output/1035_seq3_oulupdate/result.toml).
    let t = parse(
        r#"
        experiment = "1035_seq3_oulupdate"
        fps = 30.0
        meters_per_pixel = 0.000756839164395596
        task_type = "object_recognition"

        [animal_metadata]
        age = "9"
        animal = 1035
        cohort = "20"
        gene = "APPPS1"
        sex = "m"

        [detailed]
        analysis_window_seconds = 120.0
        bout_counts = []
        excluded = false
        exploration_times = []
        freezing_seconds = 0.0
        object_labels = []
        total_distance_meters = 22.24
        total_duration_seconds = 302.96
        total_mean_speed = 0.0734

        [[summaries]]
        name = "freezing"
        seconds = 0.0
        total_frames = 9089
        true_frames = 0
        "#,
    );
    assert_eq!(t.experiment, "1035_seq3_oulupdate");
    assert_eq!(t.task_type, TaskType::ObjectRecognition);
    assert!((t.fps - 30.0).abs() < 1e-9);
    assert_eq!(
        t.animal_metadata.get("animal").and_then(Value::as_i64),
        Some(1035)
    );
}

// -- NortExperiment ---------------------------------

#[test]
fn object_recognition_extract_pairwise_training_trial() {
    // A training trial with no object interactions: indices absent,
    // but observation totals should still be emitted via the shared
    // object-recognition metric units.
    let t = parse(
        r#"
        experiment = "721_seq0_oultraining"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"

        [animal_metadata]
        animal = 721
        cohort = "18"
        gene = "control"
        sex = "m"
        age = "11"

        [detailed]
        excluded = false
        total_duration_seconds = 906.5
        total_distance_meters = 58.74
        total_mean_speed = 0.0648
        freezing_seconds = 294.31
        analysis_window_seconds = 120.0
        first_object_approach_time_s = 4.0
        passes_15s_exploration_threshold = false
        object_labels = ["object_1", "object_2", "arena", "center_zone"]
        exploration_times = [2.0, 1.0, 0.0, 0.0]
        bout_counts = [2, 1, 0, 0]

        [detailed.observation_totals]
        object_labels = ["object_1", "object_2"]
        observation_seconds = [2.0, 1.0]
        observation_bouts = [2, 1]
        total_seconds = 3.0
        total_bouts = 3

        [[detailed.pairwise_observations]]
        lhs_label = "object_1"
        rhs_label = "object_2"
        absolute_delta_seconds = 1.0
        "#,
    );

    let ext = NortExperiment::from_trial(&t);
    let row = ext.extract_row(&t);

    // Metadata
    assert_eq!(row.get("animal"), Some(&CellValue::Int(721)));
    assert_eq!(row.get("cohort"), Some(&CellValue::String("18".into())));
    assert_eq!(row.get("gene"), Some(&CellValue::String("control".into())));
    assert_eq!(row.get("sex"), Some(&CellValue::String("m".into())));
    assert_eq!(row.get("age"), Some(&CellValue::String("11".into())));

    // Trial identifiers
    assert_eq!(
        row.get("experiment"),
        Some(&CellValue::String("721_seq0_oultraining".into()))
    );
    assert_eq!(row.get("fps"), Some(&CellValue::Float(30.0)));
    assert_eq!(row.get("excluded"), Some(&CellValue::Bool(false)));

    // Motion
    assert_eq!(row.get("total_distance_m"), Some(&CellValue::Float(58.74)));
    assert_eq!(row.get("total_duration_s"), Some(&CellValue::Float(906.5)));
    assert_eq!(row.get("freezing_s"), Some(&CellValue::Float(294.31)));
    assert_eq!(row.get("total_mean_speed"), Some(&CellValue::Float(0.0648)));

    // Pairwise observation metrics populated from the detailed payload.
    assert_eq!(row.get("total_observation_s"), Some(&CellValue::Float(3.0)));
    assert_eq!(
        row.get("first_object_approach_time_s"),
        Some(&CellValue::Float(4.0))
    );
    assert_eq!(
        row.get("passes_15s_exploration_threshold"),
        Some(&CellValue::Bool(false))
    );
    assert_eq!(row.get("total_observation_bouts"), Some(&CellValue::Int(3)));
    assert_eq!(
        row.get("object_1_observation_s"),
        Some(&CellValue::Float(2.0))
    );
    assert_eq!(
        row.get("object_2_observation_s"),
        Some(&CellValue::Float(1.0))
    );
    assert_eq!(
        row.get("pair_object_1_object_2_delta_s"),
        Some(&CellValue::Float(1.0))
    );
    assert_eq!(
        row.get("pair_object_2_object_1_delta_s"),
        Some(&CellValue::Float(1.0))
    );

    // Every static column is present
    for c in ext.static_columns() {
        assert!(
            row.contains_key(&c),
            "missing static column {c} from pairwise training row"
        );
    }
}

#[test]
fn object_recognition_extract_novelty_trial_with_indices() {
    // A populated NORT novelty trial should use the novelty sheet spec.
    let t = parse(
        r#"
        experiment = "42_seq4_nortnovelty"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"

        [animal_metadata]
        animal = 42
        cohort = "A"
        gene = "wt"
        sex = "f"
        age = "12"

        [session_context]
        object_a = "D"
        object_b = "U"

        [detailed]
        protocol = "nort_novelty"
        excluded = false
        total_duration_seconds = 120.0
        total_distance_meters = 5.0
        total_mean_speed = 0.04
        freezing_seconds = 1.2
        analysis_window_seconds = 120.0
        latency_to_center = 4.3
        median_speed_observing = 0.02
        median_speed_not_observing = 0.06
        first_approach_frame = 312
        first_object_approach_time_s = 10.4
        passes_15s_exploration_threshold = true
        side_bias = 0.1
        object_bias = -0.05
        object_labels = ["familiar", "novel"]
        exploration_times = [6.5, 13.5]
        bout_counts = [5, 9]

        [detailed.indices]
        discrimination_index = 0.35
        recognition_index = 0.675
        pct_novel = 67.5
        log_ratio = 0.73
        novel_seconds = 13.5
        familiar_seconds = 6.5
        total_seconds = 20.0
        novel_bouts = 9
        familiar_bouts = 5

        [detailed.observation_totals]
        object_labels = ["familiar", "novel"]
        observation_seconds = [6.5, 13.5]
        observation_bouts = [5, 9]
        total_seconds = 20.0
        total_bouts = 14
        "#,
    );

    let row = NortExperiment::from_trial(&t).extract_row(&t);

    // NORT indices populated from [detailed.indices]
    assert_eq!(row.get("DI"), Some(&CellValue::Float(0.35)));
    assert_eq!(row.get("RI"), Some(&CellValue::Float(0.675)));
    assert_eq!(row.get("pct_novel"), Some(&CellValue::Float(67.5)));
    assert_eq!(row.get("log_ratio"), Some(&CellValue::Float(0.73)));
    assert_eq!(row.get("novel_s"), Some(&CellValue::Float(13.5)));
    assert_eq!(row.get("familiar_s"), Some(&CellValue::Float(6.5)));
    assert_eq!(
        row.get("total_exploration_s"),
        Some(&CellValue::Float(20.0))
    );
    assert_eq!(row.get("novel_bouts"), Some(&CellValue::Int(9)));
    assert_eq!(row.get("familiar_bouts"), Some(&CellValue::Int(5)));
    assert_eq!(
        row.get("total_observation_s"),
        Some(&CellValue::Float(20.0))
    );

    // Bias scores
    assert_eq!(row.get("side_bias"), Some(&CellValue::Float(0.1)));
    assert_eq!(row.get("object_bias"), Some(&CellValue::Float(-0.05)));
    assert_eq!(
        row.get("object_A_label"),
        Some(&CellValue::String("familiar".into()))
    );
    assert_eq!(
        row.get("object_B_label"),
        Some(&CellValue::String("novel".into()))
    );
    assert_eq!(
        row.get("object_A_placement"),
        Some(&CellValue::String("D".into()))
    );
    assert_eq!(
        row.get("object_B_placement"),
        Some(&CellValue::String("U".into()))
    );
    assert_eq!(
        row.get("object_A_exploration_s"),
        Some(&CellValue::Float(6.5))
    );
    assert_eq!(
        row.get("object_B_exploration_s"),
        Some(&CellValue::Float(13.5))
    );
    assert_eq!(row.get("object_A_bouts"), Some(&CellValue::Int(5)));
    assert_eq!(row.get("object_B_bouts"), Some(&CellValue::Int(9)));
    assert_eq!(
        row.get("protocol"),
        Some(&CellValue::String("nort_novelty".into()))
    );

    // Approach context
    assert_eq!(row.get("latency_to_center"), Some(&CellValue::Float(4.3)));
    assert_eq!(
        row.get("median_speed_observing"),
        Some(&CellValue::Float(0.02))
    );
    assert_eq!(
        row.get("median_speed_not_observing"),
        Some(&CellValue::Float(0.06))
    );
    assert_eq!(row.get("first_approach_frame"), Some(&CellValue::Int(312)));
    assert_eq!(
        row.get("first_object_approach_time_s"),
        Some(&CellValue::Float(10.4))
    );
    assert_eq!(
        row.get("passes_15s_exploration_threshold"),
        Some(&CellValue::Bool(true))
    );

    // Per-object observation totals are emitted in object label order.
    assert_eq!(
        row.get("familiar_observation_s"),
        Some(&CellValue::Float(6.5))
    );
    assert_eq!(
        row.get("novel_observation_s"),
        Some(&CellValue::Float(13.5))
    );
}

// -- OftExperiment -----------------------------------------------
