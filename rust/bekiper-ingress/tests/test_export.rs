use std::collections::HashMap;

use polars::prelude::*;

use bekiper_behaviour::AnalysisResult;
use bekiper_core::inspection::*;
use bekiper_core::types::MetersPerPixel;
use bekiper_core::video::VideoMetadata;
use bekiper_feature::heuristic::HeuristicSummary;
use bekiper_ingress::export::export_inspection;

fn sample_manifest() -> InspectionManifest {
    InspectionManifest {
        video: VideoMetadata {
            fps: 30.0,
            total_frames: 100,
            resolution: (640, 480),
            meters_per_pixel: MetersPerPixel(0.001),
        },
        video_path: None,
        session_context: None,
        frame_image: None,
        labels: vec!["nose".into()],
        label_colors: HashMap::new(),
        coordinate_columns: HashMap::new(),
        perimeters: vec![],
        heuristics: vec![HeuristicMeta {
            name: "test_h".into(),
            result_column: "test_h".into(),
            true_frames: 50,
            total_frames: 100,
            seconds: 1.67,
            truth_spans: vec![],
        }],
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

fn sample_result_with_df() -> AnalysisResult {
    let df = df! {
        "nose_x" => &[0.1, 0.2, 0.3],
        "nose_y" => &[0.4, 0.5, 0.6],
        "test_h" => &[true, false, true],
    }
    .unwrap();

    AnalysisResult {
        task_type: bekiper_core::types::TaskType::ObjectRecognition,
        summaries: vec![HeuristicSummary {
            name: "test_h".into(),
            true_frames: 2,
            total_frames: 3,
            seconds: 0.067,
        }],
        decompositions: vec![],
        evaluated_df: Some(df),
    }
}

fn sample_result_without_df() -> AnalysisResult {
    AnalysisResult {
        task_type: bekiper_core::types::TaskType::ObjectRecognition,
        summaries: vec![],
        decompositions: vec![],
        evaluated_df: None,
    }
}

#[test]
fn export_writes_parquet_and_toml() {
    let tmp = std::env::temp_dir().join("bekiper_test_export");
    let _ = std::fs::remove_dir_all(&tmp);

    let result = sample_result_with_df();
    let manifest = sample_manifest();

    export_inspection(&tmp, "test_experiment", &result, &manifest).unwrap();

    let parquet_path = tmp.join("test_experiment_evaluated.parquet");
    let toml_path = tmp.join("test_experiment_inspection.toml");

    assert!(parquet_path.exists());
    assert!(toml_path.exists());

    // Verify Parquet content
    let df = LazyFrame::scan_parquet(parquet_path.to_str().unwrap().into(), Default::default())
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(df.shape(), (3, 3));
    assert!(df.column("nose_x").is_ok());
    assert!(df.column("test_h").is_ok());

    // Verify TOML content
    let loaded = InspectionManifest::from_toml(&toml_path).unwrap();
    assert_eq!(loaded.heuristics.len(), 1);
    assert_eq!(loaded.heuristics[0].name, "test_h");

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn export_without_df_writes_only_toml() {
    let tmp = std::env::temp_dir().join("bekiper_test_export_no_df");
    let _ = std::fs::remove_dir_all(&tmp);

    let result = sample_result_without_df();
    let manifest = sample_manifest();

    export_inspection(&tmp, "no_df_experiment", &result, &manifest).unwrap();

    let parquet_path = tmp.join("no_df_experiment_evaluated.parquet");
    let toml_path = tmp.join("no_df_experiment_inspection.toml");

    assert!(!parquet_path.exists());
    assert!(toml_path.exists());

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn export_creates_output_dir() {
    let tmp = std::env::temp_dir().join("bekiper_test_export_mkdir/nested/deep");
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("bekiper_test_export_mkdir"));

    let result = sample_result_without_df();
    let manifest = sample_manifest();

    export_inspection(&tmp, "test", &result, &manifest).unwrap();
    assert!(tmp.join("test_inspection.toml").exists());

    std::fs::remove_dir_all(std::env::temp_dir().join("bekiper_test_export_mkdir")).ok();
}
