//! Integration tests: end-to-end analysis → export → read-back roundtrip.

use std::collections::HashMap;

use polars::prelude::*;

use bekiper_behaviour::AnalysisResult;
use bekiper_core::inspection::*;
use bekiper_core::types::MetersPerPixel;
use bekiper_core::video::VideoMetadata;
use bekiper_feature::heuristic::HeuristicSummary;
use bekiper_ingress::export::export_inspection;

/// Build an AnalysisResult as if it came from a real ObjectRecognition.analyze() run.
fn realistic_analysis_result() -> AnalysisResult {
    let n = 200;
    let nose_x: Vec<f64> = (0..n).map(|i| 0.1 + (i as f64) * 0.002).collect();
    let nose_y: Vec<f64> = (0..n).map(|i| 0.2 + (i as f64) * 0.001).collect();
    // Heuristic boolean columns: body_proximity true for last 30 frames
    let body_prox: Vec<bool> = (0..n).map(|i| i >= 170).collect();
    // Olfaction true for last 10 frames
    let olfaction: Vec<bool> = (0..n).map(|i| i >= 190).collect();

    let df = df! {
        "nose_x" => &nose_x,
        "nose_y" => &nose_y,
        "body_proximity" => &body_prox,
        "olfaction" => &olfaction,
    }
    .unwrap();

    AnalysisResult {
        task_type: bekiper_core::types::TaskType::ObjectRecognition,
        summaries: vec![
            HeuristicSummary {
                name: "body_proximity".into(),
                true_frames: 30,
                total_frames: 200,
                seconds: 1.0,
            },
            HeuristicSummary {
                name: "olfaction".into(),
                true_frames: 10,
                total_frames: 200,
                seconds: 0.333,
            },
        ],
        decompositions: vec![],
        evaluated_df: Some(df),
    }
}

fn manifest_for_result(result: &AnalysisResult) -> InspectionManifest {
    let heuristic_metas: Vec<HeuristicMeta> = result
        .summaries
        .iter()
        .map(|s| HeuristicMeta {
            name: s.name.clone(),
            result_column: s.name.clone(),
            true_frames: s.true_frames,
            total_frames: s.total_frames,
            seconds: s.seconds,
            truth_spans: vec![],
        })
        .collect();

    InspectionManifest {
        video: VideoMetadata {
            fps: 30.0,
            total_frames: 200,
            resolution: (640, 480),
            meters_per_pixel: MetersPerPixel(0.001),
        },
        video_path: Some("/data/videos/trial01.mp4".into()),
        session_context: None,
        frame_image: None,
        labels: vec!["nose".into()],
        label_colors: HashMap::from([("nose".into(), "#FF0000".into())]),
        coordinate_columns: HashMap::from([(
            "nose".into(),
            CoordinateColumnSpec {
                x: "nose_x".into(),
                y: "nose_y".into(),
            },
        )]),
        perimeters: vec![PerimeterSpec::Circle {
            label: "object1".into(),
            center_x: 0.5,
            center_y: 0.4,
            radius: 0.05,
        }],
        heuristics: heuristic_metas,
        settings: InspectionSettings {
            invert_y: false,
            minimum_seconds_tolerance: 0.5,
            maximum_seconds_distraction: 0.333,
            meters_per_pixel: 0.001,
        },
        scale_derivation: None,
        quadrants: Vec::new(),
        feature_inspections: Vec::new(),
        pixel_perimeters: Vec::new(),
        pixel_quadrants: Vec::new(),
    }
}

#[test]
fn end_to_end_export_and_read_back() {
    let tmp = std::env::temp_dir().join("bekiper_test_e2e_export");
    let _ = std::fs::remove_dir_all(&tmp);

    let result = realistic_analysis_result();
    let manifest = manifest_for_result(&result);

    export_inspection(&tmp, "object_recognition", &result, &manifest).unwrap();

    // Verify Parquet: correct shape, columns, and data
    let pq_path = tmp.join("object_recognition_evaluated.parquet");
    assert!(pq_path.exists());
    let df = LazyFrame::scan_parquet(pq_path.to_str().unwrap().into(), Default::default())
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(df.height(), 200);
    assert!(df.column("nose_x").is_ok());
    assert!(df.column("body_proximity").is_ok());
    assert!(df.column("olfaction").is_ok());

    // Verify boolean column content
    let bp = df.column("body_proximity").unwrap().bool().unwrap();
    let true_count: u64 = bp.sum().unwrap() as u64;
    assert_eq!(true_count, 30, "body_proximity should have 30 true frames");

    // Verify TOML manifest roundtrip
    let toml_path = tmp.join("object_recognition_inspection.toml");
    assert!(toml_path.exists());
    let loaded = InspectionManifest::from_toml(&toml_path).unwrap();

    assert_eq!(loaded.heuristics.len(), 2);
    assert_eq!(loaded.heuristics[0].name, "body_proximity");
    assert_eq!(loaded.heuristics[0].true_frames, 30);
    assert_eq!(loaded.heuristics[1].name, "olfaction");
    assert_eq!(loaded.heuristics[1].true_frames, 10);
    assert_eq!(loaded.video.fps, 30.0);
    assert_eq!(loaded.video.total_frames, 200);
    assert_eq!(loaded.labels, vec!["nose"]);
    assert_eq!(loaded.perimeters.len(), 1);
    assert!(loaded.video_path.is_some());
    assert_eq!(loaded.coordinate_columns.len(), 1);

    // Verify perimeter spec
    match &loaded.perimeters[0] {
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } => {
            assert_eq!(label, "object1");
            assert!((center_x - 0.5).abs() < 1e-10);
            assert!((center_y - 0.4).abs() < 1e-10);
            assert!((radius - 0.05).abs() < 1e-10);
        }
        other => panic!("expected Circle, got {other:?}"),
    }

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn export_multiple_experiments_to_same_dir() {
    let tmp = std::env::temp_dir().join("bekiper_test_multi_export");
    let _ = std::fs::remove_dir_all(&tmp);

    let result1 = realistic_analysis_result();
    let manifest1 = manifest_for_result(&result1);

    let mut result2 = realistic_analysis_result();
    result2.task_type = bekiper_core::types::TaskType::ObjectRecognition;
    let manifest2 = manifest_for_result(&result2);

    export_inspection(&tmp, "experiment_A", &result1, &manifest1).unwrap();
    export_inspection(&tmp, "experiment_B", &result2, &manifest2).unwrap();

    // Both sets of files should exist
    assert!(tmp.join("experiment_A_evaluated.parquet").exists());
    assert!(tmp.join("experiment_A_inspection.toml").exists());
    assert!(tmp.join("experiment_B_evaluated.parquet").exists());
    assert!(tmp.join("experiment_B_inspection.toml").exists());

    // Each should be independently readable
    let m1 = InspectionManifest::from_toml(&tmp.join("experiment_A_inspection.toml")).unwrap();
    let m2 = InspectionManifest::from_toml(&tmp.join("experiment_B_inspection.toml")).unwrap();
    assert_eq!(m1.heuristics.len(), 2);
    assert_eq!(m2.heuristics.len(), 2);

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn ingress_config_from_toml_roundtrip() {
    let tmp = std::env::temp_dir().join("bekiper_test_config_rt");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let toml_content = r#"
[project]
project_name = "test_project"
data_directory = "/data/experiments"

[project.runtime]
minimum_seconds_tolerance = 0.5
maximum_seconds_distraction = 0.333
num_threads = 4

[[experiments]]
name = "trial_01"
data_files = ["/data/experiments/trial01.csv"]
task_type = "object_recognition"
fps = 30.0
meters_per_pixel = 0.001

[[experiments]]
name = "trial_02"
data_files = ["/data/experiments/trial02.csv", "/data/experiments/trial02b.csv"]
task_type = "y_maze"
fps = 60.0
meters_per_pixel = 0.0005
"#;

    let config_path = tmp.join("project.toml");
    std::fs::write(&config_path, toml_content).unwrap();

    let config = bekiper_ingress::config::IngressConfig::from_file(&config_path).unwrap();

    assert_eq!(config.project.project_name, "test_project");
    assert_eq!(
        config.project.data_directory,
        std::path::PathBuf::from("/data/experiments")
    );
    assert_eq!(config.project.runtime.num_threads, Some(4));
    assert_eq!(config.experiments.len(), 2);
    assert_eq!(config.experiments[0].name, "trial_01");
    assert_eq!(
        config.experiments[0].task_type,
        bekiper_core::types::TaskType::ObjectRecognition
    );
    assert!(config.experiments[0].effective_fps().is_ok());
    assert!((config.experiments[0].effective_fps().unwrap() - 30.0).abs() < 1e-10);
    assert_eq!(config.experiments[1].data_files.len(), 2);
    assert!((config.experiments[1].effective_fps().unwrap() - 60.0).abs() < 1e-10);

    std::fs::remove_dir_all(&tmp).ok();
}
