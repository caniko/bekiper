use std::collections::HashMap;
use std::path::Path;

use polars::prelude::*;

use bekiper_behaviour::{AnalysisResult, TaskDetail, TaskParams};
use bekiper_core::inspection::*;
use bekiper_core::scale::{ScaleConfig, ScaleDerivationInfo};
use bekiper_core::types::{MetersPerPixel, TaskType};
use bekiper_core::video::{FrameTiming, VideoMetadata};
use bekiper_feature::heuristic::HeuristicSummary;
use bekiper_feature::zone::ZoneSummary;
use bekiper_ingress::aggregate::{TrialResult, build_sheet_dataframes};
use bekiper_ingress::config::AggregateConfig;
use bekiper_ingress::config::{IngressConfig, PipelineConfig};
use bekiper_ingress::export::{export_inspection, export_spreadsheet};
use bekiper_ingress::stage::{ExperimentOutputDir, Stage};
use bekiper_pose::io::write_parquet;
use bekiper_pose::pipeline::AugmentedBuilder;
use bekiper_pose::steps::likelihood::LikelihoodFilter;
use bekiper_pose::steps::smoothing::MedianSmoother;
use bekiper_pose::steps::transform::CoordinateTransform;
use bekiper_pose::steps::velocity::HighVelocityFilter;

// ---------------------------------------------------------------------------
// Local wrapper types for TOML serialization
// ---------------------------------------------------------------------------

/// Wrapper for serializing summaries as a TOML file (TOML requires a root table).
#[derive(serde::Serialize, serde::Deserialize)]
struct SummariesFile {
    summaries: Vec<HeuristicSummary>,
    #[serde(default)]
    decompositions: Vec<HeuristicDecompositionSpec>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("bekiper_test_pipeline_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn load_typed_detail(path: &Path, task_type: TaskType) -> Option<TaskDetail> {
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(path).unwrap();
    let json = toml_str_to_json_value(&content).unwrap();
    Some(TaskDetail::from_json_value(task_type, json).unwrap())
}

/// DLC SuperAnimal-TopViewMouse labels used in the imen config.
/// We use a representative subset (the ones referenced by TaskParams defaults).
const LABELS: &[&str] = &[
    "nose",
    "left_ear",
    "right_ear",
    "center_ear",
    "mouse_center",
    "tail_base",
    "left_faw",
    "right_faw",
];

/// Generate a synthetic pose DataFrame with N frames and the given labels.
/// Coordinates are in pixel space (0..640 x 0..480) with high-confidence.
fn synthetic_poses(n: usize, labels: &[&str]) -> DataFrame {
    let frames: Vec<i64> = (0..n as i64).collect();
    let mut columns: Vec<Column> = vec![Column::new("frame".into(), &frames)];

    for (li, label) in labels.iter().enumerate() {
        // Each label gets a gentle linear trajectory with slight per-label offset
        let offset = li as f64 * 20.0;
        let xs: Vec<f64> = (0..n).map(|i| 200.0 + offset + (i as f64) * 0.5).collect();
        let ys: Vec<f64> = (0..n).map(|i| 150.0 + offset + (i as f64) * 0.3).collect();
        let liks: Vec<f64> = (0..n)
            .map(|i| {
                // Make a few frames low-confidence to exercise likelihood filter
                if i % 50 == 0 && i > 0 { 0.1 } else { 0.95 }
            })
            .collect();

        columns.push(Column::new(format!("{label}_x").into(), &xs));
        columns.push(Column::new(format!("{label}_y").into(), &ys));
        columns.push(Column::new(format!("{label}_likelihood").into(), &liks));
    }

    DataFrame::new(n, columns).unwrap()
}

/// Generate poses with a velocity spike to exercise the velocity filter.
fn synthetic_poses_with_spike(n: usize, labels: &[&str]) -> DataFrame {
    let frames: Vec<i64> = (0..n as i64).collect();
    let mut columns: Vec<Column> = vec![Column::new("frame".into(), &frames)];

    for (li, label) in labels.iter().enumerate() {
        let offset = li as f64 * 20.0;
        let xs: Vec<f64> = (0..n)
            .map(|i| {
                if i == n / 2 {
                    9999.0 // velocity spike
                } else {
                    200.0 + offset + (i as f64) * 0.5
                }
            })
            .collect();
        let ys: Vec<f64> = (0..n).map(|i| 150.0 + offset + (i as f64) * 0.3).collect();
        let liks: Vec<f64> = vec![0.95; n];

        columns.push(Column::new(format!("{label}_x").into(), &xs));
        columns.push(Column::new(format!("{label}_y").into(), &ys));
        columns.push(Column::new(format!("{label}_likelihood").into(), &liks));
    }

    DataFrame::new(n, columns).unwrap()
}

/// NORT perimeters: arena + two objects (in meter-space).
fn nort_perimeters() -> Vec<PerimeterSpec> {
    vec![
        PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 0.20,
            center_y: 0.20,
            width: 0.40,
            height: 0.40,
        },
        PerimeterSpec::Circle {
            label: "object_left".into(),
            center_x: 0.10,
            center_y: 0.20,
            radius: 0.02,
        },
        PerimeterSpec::Circle {
            label: "object_right".into(),
            center_x: 0.30,
            center_y: 0.20,
            radius: 0.02,
        },
    ]
}

fn oul_perimeters() -> Vec<PerimeterSpec> {
    vec![
        PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 0.20,
            center_y: 0.20,
            width: 0.40,
            height: 0.40,
        },
        PerimeterSpec::Rectangle {
            label: "center_zone".into(),
            center_x: 0.20,
            center_y: 0.20,
            width: 0.20,
            height: 0.20,
        },
        PerimeterSpec::Circle {
            label: "object_1".into(),
            center_x: 0.10,
            center_y: 0.20,
            radius: 0.02,
        },
        PerimeterSpec::Circle {
            label: "object_2".into(),
            center_x: 0.30,
            center_y: 0.20,
            radius: 0.02,
        },
    ]
}

/// OFT perimeters: single arena rectangle (in meter-space).
fn oft_perimeters() -> Vec<PerimeterSpec> {
    vec![PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 0.20,
        center_y: 0.20,
        width: 0.40,
        height: 0.40,
    }]
}

/// Y-maze perimeters: radial maze with 3 arms (in meter-space).
fn ymaze_perimeters() -> Vec<PerimeterSpec> {
    // A simplified Y-maze: equilateral center + 3 arms radiating outward
    let arm_len = 0.15;
    let arm_width = 0.05;
    let cx = 0.20;
    let cy = 0.20;

    // 3 arms at 120° apart (pointing up, lower-left, lower-right)
    let angles = [
        std::f64::consts::FRAC_PI_2,                                     // up
        std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::FRAC_PI_3, // lower-left
        std::f64::consts::FRAC_PI_2 + 4.0 * std::f64::consts::FRAC_PI_3, // lower-right
    ];

    let center_r = 0.03;
    let center_vertices: Vec<(f64, f64)> = angles
        .iter()
        .map(|a| (cx + center_r * a.cos(), cy + center_r * a.sin()))
        .collect();

    let arms: Vec<Vec<(f64, f64)>> = angles
        .iter()
        .map(|a| {
            let dx = a.cos();
            let dy = a.sin();
            let perp_x = -dy * arm_width / 2.0;
            let perp_y = dx * arm_width / 2.0;
            let base_x = cx + dx * center_r;
            let base_y = cy + dy * center_r;
            let tip_x = cx + dx * (center_r + arm_len);
            let tip_y = cy + dy * (center_r + arm_len);
            vec![
                (base_x + perp_x, base_y + perp_y),
                (base_x - perp_x, base_y - perp_y),
                (tip_x - perp_x, tip_y - perp_y),
                (tip_x + perp_x, tip_y + perp_y),
            ]
        })
        .collect();

    vec![PerimeterSpec::RadialMaze {
        label: "ymaze".into(),
        center_vertices,
        arms,
    }]
}

fn cheeseboard_perimeters() -> Vec<PerimeterSpec> {
    vec![
        PerimeterSpec::Circle {
            label: "board".into(),
            center_x: 0.20,
            center_y: 0.20,
            radius: 0.20,
        },
        PerimeterSpec::Circle {
            label: "reward_a".into(),
            center_x: 0.14,
            center_y: 0.20,
            radius: 0.03,
        },
        PerimeterSpec::Circle {
            label: "reward_b".into(),
            center_x: 0.26,
            center_y: 0.20,
            radius: 0.02,
        },
    ]
}

fn cheeseboard_detail(
    start_area_exit_frame: Option<usize>,
    start_area_exit_seconds: Option<f64>,
    reward_arrival_frame: Option<usize>,
    reward_arrival_seconds: Option<f64>,
    seconds_to_find_reward: Option<f64>,
) -> TaskDetail {
    let summary =
        |zone_name: &str, entries: u32, duration_seconds: f64, latency_seconds| ZoneSummary {
            zone_name: zone_name.into(),
            entries,
            duration_seconds,
            distance_meters: duration_seconds * 0.01,
            mean_speed: 0.5,
            latency_seconds,
        };

    TaskDetail::Cheeseboard(bekiper_behaviour::reward_tracing::CheeseboardResult {
        enclosure_label: "board".into(),
        reward_labels: vec!["reward_a".into(), "reward_b".into()],
        enclosure_summary: summary("board", 1, 240.0, Some(0.4)),
        reward_summaries: vec![
            summary("reward_a", 0, 0.0, Some(1.2)),
            summary("reward_b", 1, 18.7, Some(0.9)),
        ],
        start_area_exit_frame,
        start_area_exit_seconds,
        reward_arrival_frame,
        reward_arrival_seconds,
        seconds_to_find_reward,
        seconds_in_reward_area: 18.7,
        reward_entries: 2,
        search_path_distance_meters: Some(1.23),
        start_to_reward_displacement_meters: Some(1.23),
        search_direct_distance_meters: Some(0.98),
        search_mean_speed_mps: Some(2.46),
        search_tortuosity: Some(1.25),
        total_distance_meters: 18.7,
        total_mean_speed_mps: 0.62,
        total_duration_seconds: 240.0,
        evaluated_df: None,
    })
}

fn default_pipeline_config() -> PipelineConfig {
    PipelineConfig {
        likelihood_threshold: 0.9,
        smoothing_window: 5,
        max_velocity: 50.0,
        high_velocity_iterations: 3,
        displacement_sigma_threshold: 0.0,
        likelihood_trim: false,
        crop_start_seconds: 0.0,
        crop_end_seconds: 0.0,
        target_trial_length_seconds: 0.0,
        invert_y: false,
        midpoint_groups: Vec::new(),
        skeleton: None,
        ..PipelineConfig::default()
    }
}

fn label_columns(labels: &[&str]) -> (Vec<String>, Vec<String>, Vec<String>) {
    let x_cols: Vec<String> = labels.iter().map(|l| format!("{l}_x")).collect();
    let y_cols: Vec<String> = labels.iter().map(|l| format!("{l}_y")).collect();
    let lik_cols: Vec<String> = labels.iter().map(|l| format!("{l}_likelihood")).collect();
    (x_cols, y_cols, lik_cols)
}

/// Build the augmentation pipeline matching the imen config parameters.
fn build_augmentation_pipeline(
    labels: &[&str],
    config: &PipelineConfig,
    mpp: f64,
) -> AugmentedBuilder {
    let (x_cols, y_cols, lik_cols) = label_columns(labels);
    let coord_cols: Vec<String> = x_cols.iter().chain(y_cols.iter()).cloned().collect();

    AugmentedBuilder::new()
        .add_step(LikelihoodFilter::new(
            config.likelihood_threshold,
            lik_cols,
            coord_cols,
        ))
        .add_step(HighVelocityFilter {
            max_velocity: config.max_velocity,
            x_columns: x_cols.clone(),
            y_columns: y_cols.clone(),
            max_iterations: 3,
            sigma_threshold: 0.0,
        })
        .add_step(MedianSmoother {
            window_size: config.smoothing_window,
            columns: x_cols.iter().chain(y_cols.iter()).cloned().collect(),
        })
        .add_step(CoordinateTransform {
            meters_per_pixel: mpp,
            meters_per_pixel_y: None,
            invert_y: false,
            y_height: 0.0,
            x_columns: x_cols,
            y_columns: y_cols,
        })
}

// =========================================================================
// 1. Config parsing — imen TOML roundtrip
// =========================================================================

#[test]
fn parse_imen_config_structure() {
    // Minimal reproduction of the imen config structure (without requiring
    // actual video directories to exist).
    let toml = r#"
[project]
project_name = "imen_pilot"
data_directory = "."
output_dir = "output"

[pipeline]
likelihood_threshold = 0.9
smoothing_window = 5
max_velocity = 50.0

[[experiments]]
name = "nort_t1_video10"
task_type = "object_recognition"
fps = 30.0
meters_per_pixel = 0.001

[[experiments]]
name = "oft_video42"
task_type = "open_field"
fps = 30.0
meters_per_pixel = 0.001

[[experiments]]
name = "ymaze_video01"
task_type = "y_maze"
fps = 30.0
meters_per_pixel = 0.001
"#;

    let tmp = test_dir("config_parse");
    let config_path = tmp.join("config.toml");
    std::fs::write(&config_path, toml).unwrap();

    let config = IngressConfig::from_file(&config_path).unwrap();

    assert_eq!(config.project.project_name, "imen_pilot");
    assert_eq!(config.pipeline.likelihood_threshold, 0.9);
    assert_eq!(config.pipeline.smoothing_window, 5);
    assert_eq!(config.pipeline.max_velocity, 50.0);
    assert_eq!(config.experiments.len(), 3);

    assert_eq!(config.experiments[0].task_type, TaskType::ObjectRecognition);
    assert_eq!(config.experiments[1].task_type, TaskType::OpenField);
    assert_eq!(config.experiments[2].task_type, TaskType::YMaze);

    std::fs::remove_dir_all(&tmp).ok();
}
