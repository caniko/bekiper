//! Audit regressions remain included in repository health measurements.
const IMEN_NORT_MODEL_CONFIG: &str = r#"
        [project]
        project_name = "audit"
        data_directory = "."

        [pipeline]
        pose_label_model = "imen_dlc"

        [[experiments]]
        name = "trial"
        task_type = "object_recognition"
        video = "video.mp4"
        meters_per_pixel = 0.001

        [experiments.pose_model]
        path = "models/model.onnx"
        spec = "models/model.json"
        "#;

const ANNOTATION_COVERAGE_CONFIG: &str = r#"
        [project]
        project_name = "coverage"
        data_directory = "."

        [pose_model]
        path = "models/model.onnx"
        spec = "models/model.json"

        [[scan]]
        video_dir = "videos"
        task_type = "object_recognition"
        meters_per_pixel = 0.001
        backend = ["manual_annotation"]
        annotations = { dir = "annotations/nort" }

        [scan.recipe_rules]
        "**/T1/**" = "nort_training"
        "#;

const LIKELIHOOD_CONFIG: &str = r#"
        [project]
        project_name = "likelihood"
        data_directory = "."
        output_dir = "output"

        [[experiments]]
        name = "trial1"
        data_files = ["trial1.parquet"]
        task_type = "open_field"
        fps = 30.0
        meters_per_pixel = 0.001
        "#;

fn write_config(directory: &Path, contents: &str) -> PathBuf {
    let path = directory.join("bekiper_config.toml");
    std::fs::write(&path, contents).unwrap();
    path
}

fn reset_fixture_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&path);
    path
}

use super::*;
use bekiper_pose::PoseBackend;
use bekiper_pose::infer::{ModelSpec, Normalization, OutputLayout};
use bekiper_pose::skeleton::{Edge, Node, Skeleton};

fn test_model_spec_json() -> String {
    serde_json::to_string(&ModelSpec {
        backend: PoseBackend::DeepLabCut,
        labels: vec![
            "mouse_center".into(),
            "nose".into(),
            "center_ear".into(),
            "tail_base".into(),
        ],
        input_height: 64,
        input_width: 64,
        input_channels: 3,
        channels_first: true,
        normalization: Normalization::ZeroOne,
        output_layout: OutputLayout::Heatmaps,
        skeleton: Skeleton::new(
            "test_superanimal_subset",
            vec![
                Node::new("mouse_center"),
                Node::new("nose"),
                Node::new("center_ear"),
                Node::new("tail_base"),
            ],
            vec![
                Edge::new("nose", "center_ear"),
                Edge::new("center_ear", "mouse_center"),
                Edge::new("mouse_center", "tail_base"),
            ],
        ),
        midpoint_groups: Vec::new(),
        label_aliases: Vec::new(),
    })
    .unwrap()
}

fn imen_model_spec_json() -> String {
    serde_json::to_string(&ModelSpec {
        backend: PoseBackend::DeepLabCut,
        labels: vec![
            "nose".into(),
            "left_ear".into(),
            "right_ear".into(),
            "tail".into(),
        ],
        input_height: 64,
        input_width: 64,
        input_channels: 3,
        channels_first: true,
        normalization: Normalization::ZeroOne,
        output_layout: OutputLayout::Heatmaps,
        skeleton: Skeleton::imen_dlc("tail"),
        midpoint_groups: vec![bekiper_core::types::MidpointGroup {
            output_label: "center_ear".into(),
            source_labels: vec!["left_ear".into(), "right_ear".into()],
        }],
        label_aliases: Vec::new(),
    })
    .unwrap()
}

include!("regressions/config_validation.rs");
include!("regressions/reporting.rs");
include!("regressions/external_tools.rs");
include!("regressions/ymaze_alignment.rs");
include!("regressions/annotation_coverage.rs");
include!("regressions/tracking_quality.rs");
include!("regressions/likelihood.rs");
