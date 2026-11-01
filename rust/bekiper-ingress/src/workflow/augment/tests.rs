#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use bekiper_core::config::ProjectConfig;
    use bekiper_core::scale::ScaleConfig;
    use bekiper_core::types::TaskType;
    use bekiper_core::video::FpsConfig;
    use bekiper_pose::io::write_parquet;
    use bekiper_pose::skeleton::{Edge, Node, Skeleton, SkeletonPreset};
    use polars::prelude::{AnyValue, LazyFrame};

    use crate::config::{
        AggregateConfig, GpuConfig, IngressConfig, PipelineConfig, PoseLabelModel,
        SkeletonRepairConfig, SmoothingMethod,
    };

    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bekiper_augment_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn line_skeleton() -> Skeleton {
        Skeleton::new(
            "line",
            vec![Node::new("a"), Node::new("b"), Node::new("c")],
            vec![Edge::new("a", "b"), Edge::new("b", "c")],
        )
    }

    fn test_config(pipeline: PipelineConfig) -> IngressConfig {
        IngressConfig {
            inspect: Default::default(),
            review: Default::default(),
            project: ProjectConfig {
                project_name: "test".into(),
                data_directory: ".".into(),
                output_dir: Some("output".into()),
                runtime: Default::default(),
            },
            pipeline,
            gpu: GpuConfig::default(),
            aggregate: AggregateConfig::default(),
            post_analysis: crate::config::PostAnalysisConfig::default(),
            pose_model: None,
            scan: Vec::new(),
            animal_scan: Vec::new(),
            ingress_profile: Vec::new(),
            experiments: Vec::new(),
            animal_metadata: HashMap::new(),
            override_record: Vec::new(),
            toml_paths: Vec::new(),
        }
    }

    fn experiment(name: &str) -> ExperimentConfig {
        ExperimentConfig {
            recording_id: None,
            name: name.into(),
            task_type: TaskType::OpenField,
            fps: FpsConfig::Manual(30.0),
            meters_per_pixel: ScaleConfig::Manual(0.001),
            data_files: Vec::new(),
            video: None,
            pose_model: None,
            detection: None,
            task_params: None,
            trial_start_perimeter: None,
            annotation_dir: None,
            frame_timing: None,
            video_resolution: None,
            frame_image: None,
            timestamps_file: None,
            animal_id: None,
            session_context: None,
        }
    }

    fn write_pixel_perimeters(exp_dir: &ExperimentOutputDir, perimeters: Vec<PerimeterSpec>) {
        std::fs::create_dir_all(exp_dir.checkpoints_dir()).unwrap();
        let pf = PerimetersFile { perimeters };
        std::fs::write(
            exp_dir.pixel_perimeters_toml(),
            toml::to_string_pretty(&pf).unwrap(),
        )
        .unwrap();
    }

    fn prepare_output(name: &str) -> ExperimentOutputDir {
        let exp_dir = ExperimentOutputDir::new(&test_dir(name), "exp");
        exp_dir.ensure_dir().unwrap();
        std::fs::create_dir_all(exp_dir.checkpoints_dir()).unwrap();
        exp_dir
    }

    fn read_augmented(exp_dir: &ExperimentOutputDir) -> DataFrame {
        LazyFrame::scan_parquet(
            exp_dir.augmented_parquet().to_str().unwrap().into(),
            Default::default(),
        )
        .unwrap()
        .collect()
        .unwrap()
    }

    fn imen_pose_data(frames: &[i64], tail_x: Vec<f64>) -> DataFrame {
        polars::df! {
            "frame" => frames,
            "nose_x" => vec![0.0; frames.len()],
            "nose_y" => vec![0.0; frames.len()],
            "nose_likelihood" => vec![0.99; frames.len()],
            "left_ear_x" => vec![-1.0; frames.len()],
            "left_ear_y" => vec![1.0; frames.len()],
            "left_ear_likelihood" => vec![0.99; frames.len()],
            "right_ear_x" => vec![1.0; frames.len()],
            "right_ear_y" => vec![1.0; frames.len()],
            "right_ear_likelihood" => vec![0.99; frames.len()],
            "tail_x" => tail_x,
            "tail_y" => vec![10.0; frames.len()],
            "tail_likelihood" => vec![0.99; frames.len()],
        }
        .unwrap()
    }

    include!("tests/redetection.rs");

    include!("tests/preflight.rs");

    include!("tests/smoothing.rs");

    include!("tests/skeleton.rs");

    include!("tests/midpoints.rs");

    include!("tests/calibration.rs");
}
