//! Generate fresh native trial measurements and export an explicitly synthetic
//! three-assay cohort. This is a reproducible integration demonstration.
//! Usage: cargo run -p bekiper-ingress --example post_analysis_synthetic_cohort --
//!        <project-config.toml> <fresh-run-root>
use std::{error::Error, fs, path::PathBuf};

use bekiper_behaviour::{TaskParams, factory::build_task};
use bekiper_core::{
    inspection::PerimeterSpec,
    types::{ObjectRecognitionProtocol, TaskType},
    video::FrameTiming,
};
use bekiper_ingress::{
    aggregate::{TrialResult, export_aggregate_trials},
    config::{IngressConfig, TaskParamsOverrides},
};
use bekiper_post_analysis::{RunOptions, inputs};
use polars::prelude::*;
use serde_json::{Value, json};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Pose = ([f64; 2], [f64; 2]);
const FPS: f64 = 10.0;

fn frame(poses: &[Pose]) -> Result<LazyFrame> {
    let mut columns = vec![Column::new(
        "frame".into(),
        (0..poses.len() as u64).collect::<Vec<_>>(),
    )];
    for (label, offset) in [
        ("center_ear", 0.0),
        ("nose", -0.03),
        ("mouse_center", 0.04),
        ("left_ear", 0.01),
        ("right_ear", 0.01),
    ] {
        let xs = poses
            .iter()
            .map(|(head, tail)| head[0] + offset * (tail[0] - head[0]).signum())
            .collect::<Vec<_>>();
        let ys = poses
            .iter()
            .map(|(head, _)| {
                head[1]
                    + if label == "left_ear" {
                        0.01
                    } else if label == "right_ear" {
                        -0.01
                    } else {
                        0.0
                    }
            })
            .collect::<Vec<_>>();
        columns.push(Column::new(format!("{label}_x").into(), xs));
        columns.push(Column::new(format!("{label}_y").into(), ys));
    }
    for (axis, index) in [("x", 0), ("y", 1)] {
        columns.push(Column::new(
            format!("tail_base_{axis}").into(),
            poses
                .iter()
                .map(|(_, tail)| tail[index])
                .collect::<Vec<_>>(),
        ));
    }
    columns.push(Column::new(
        "snout_dx".into(),
        poses
            .iter()
            .map(|(head, tail)| -0.03 * (tail[0] - head[0]).signum())
            .collect::<Vec<_>>(),
    ));
    columns.push(Column::new("snout_dy".into(), vec![0.0; poses.len()]));
    Ok(DataFrame::new(poses.len(), columns)?.lazy())
}

fn arena() -> PerimeterSpec {
    PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: 0.2,
        center_y: 0.2,
        width: 0.4,
        height: 0.4,
    }
}

fn maze() -> (Vec<PerimeterSpec>, Value) {
    let center = vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)];
    let arms = vec![
        vec![(0.1, 0.0), (0.6, 0.05), (1.1, 0.0), (0.6, -0.05)],
        vec![(0.0, 0.1), (0.05, 0.6), (0.0, 1.1), (-0.05, 0.6)],
        vec![(-0.1, 0.0), (-0.6, 0.05), (-1.1, 0.0), (-0.6, -0.05)],
    ];
    let geometry = json!({"version":"imen_radial_vertex_span_v1","canonical_vertices":true,
        "center":center,"arms":{"A":arms[0],"B":arms[1],"C":arms[2]}});
    (
        vec![PerimeterSpec::RadialMaze {
            label: "maze".into(),
            center_vertices: center,
            arms,
        }],
        geometry,
    )
}

fn poses(assay: &str, index: usize, post: bool) -> Vec<Pose> {
    if assay == "ymaze" {
        let sequences = [
            "ABCABBAC",
            "ABCABCABA",
            "ABABACBA",
            "ABCABCABCA",
            "ABBBACCA",
            "ABCAABCB",
            "ABBABCACAB",
            "ABCABACB",
        ];
        let mut poses = vec![([0.0, 0.0], [0.0, 0.0])];
        for (step, arm) in sequences[(index + usize::from(post)) % sequences.len()]
            .chars()
            .enumerate()
        {
            let radius = [0.5, 0.6, 0.9][(step + index) % 3];
            let point = match arm {
                'A' => [radius, 0.0],
                'B' => [0.0, radius],
                _ => [-radius, 0.0],
            };
            poses.push((point, [0.0, 0.0])); // First landmark alone never counts.
            poses.push((point, point)); // Raw crossing; effective policy qualifies visits.
            poses.push(([0.0, 0.0], [0.0, 0.0]));
        }
        return poses;
    }
    (0..240)
        .map(|sample| {
            if assay == "nort" {
                let left_samples = 85 + (index % 4) * 10 + usize::from(post) * 5;
                if sample < left_samples {
                    ([0.16, 0.2], [0.24, 0.2])
                } else {
                    ([0.24, 0.2], [0.16, 0.2])
                }
            } else {
                let center_samples = 60 + (index % 8) * 7 + usize::from(post) * 3;
                let x = if sample < center_samples { 0.20 } else { 0.035 };
                let y = 0.18 + 0.002 * (sample as f64 * (0.2 + index as f64 * 0.003)).sin();
                ([x, y], [x + 0.015, y])
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let config_path = PathBuf::from(arguments.next().ok_or("project config argument required")?)
        .canonicalize()?;
    let root = PathBuf::from(arguments.next().ok_or("fresh output root required")?);
    if arguments.next().is_some() || root.exists() {
        return Err("expected config and a fresh output root".into());
    }
    let mut config = IngressConfig::from_file(&config_path)?;
    let plan_path = config
        .post_analysis
        .resolved_plan_path(config_path.parent().unwrap())
        .ok_or("registered plan required")?;
    config.post_analysis.plan_path = Some(plan_path.clone());
    let mut plan = inputs::resolved_config(&plan_path, &RunOptions::default(), &|_| None)?;
    if plan["run_mode"] != "synthetic" || plan["synthetic_fixture"] != true {
        return Err("demonstration requires explicit synthetic registration".into());
    }
    for assay in ["oft", "ymaze", "nort"] {
        plan["inputs"][format!("{assay}_path")] = Value::Null;
    }
    let (metadata, _) = inputs::load_inputs(&plan, plan_path.parent().unwrap())?;
    let mut trials = Vec::new();
    for (index, row) in metadata.iter().enumerate() {
        let assay = row["assay"].as_str().ok_or("assay required")?;
        // A registered but unobserved post session exercises missing pairing.
        if row["animal_id"] == "SYNTHETIC_1_1" && assay == "ymaze" && row["timepoint"] == "post" {
            continue;
        }
        let post = row["timepoint"] == "post";
        let points = poses(assay, index, post);
        let task_type = match assay {
            "oft" => TaskType::OpenField,
            "ymaze" => TaskType::YMaze,
            "nort" => TaskType::ObjectRecognition,
            _ => return Err("unsupported synthetic assay".into()),
        };
        let thresholds = if assay == "ymaze" {
            vec![25, 45, 65]
        } else {
            vec![0]
        };
        for threshold in thresholds {
            let (perimeters, geometry) = if assay == "ymaze" {
                maze()
            } else if assay == "nort" {
                // Deliberate reversed object order: identity, never array position, owns roles.
                (
                    vec![
                        arena(),
                        PerimeterSpec::Circle {
                            label: "right".into(),
                            center_x: 0.30,
                            center_y: 0.20,
                            radius: 0.02,
                        },
                        PerimeterSpec::Circle {
                            label: "left".into(),
                            center_x: 0.10,
                            center_y: 0.20,
                            radius: 0.02,
                        },
                    ],
                    Value::Null,
                )
            } else {
                (vec![arena()], Value::Null)
            };
            let overrides: TaskParamsOverrides = serde_json::from_value(json!({
                "ymaze":{"proximal_threshold":f64::from(threshold)/100.0,"start_mode":"first_detection",
                    "temporal_policy":plan["assays"]["ymaze"]["temporal_policy"]},
                "exploration":{"analysis_window_seconds":360.0,"window_mode":"full_trial_short_trial","enable_qualia":false,"perimeter_erosion_meters":0.0}
            }))?;
            let protocol = (assay == "nort").then_some(if row["phase"] == "training" {
                ObjectRecognitionProtocol::NortTraining
            } else {
                ObjectRecognitionProtocol::NortNovelty
            });
            let task = build_task(
                task_type,
                &perimeters,
                TaskParams::new(FPS),
                protocol,
                None,
                Some(&overrides),
            )?;
            let (result, detail) = task.analyze_detailed(frame(&points)?, &FrameTiming::cfr(FPS));
            let mut detailed = detail
                .ok_or("native task omitted detailed output")?
                .as_json_value();
            detailed["synthetic_observation"] = true.into();
            detailed["qc_flags"] = json!({"source_warning":true,"metadata_warning":false});
            if assay == "nort" && row["phase"] == "test" {
                // Synthetic imported comparison: the historical index-based
                // native helper interprets array[0] as novel. Registered object
                // identities subsequently own recomputation, preserving drift.
                detailed["indices"] =
                    serde_json::to_value(bekiper_behaviour::nort::NortIndices::compute(
                        detailed["exploration_times"][0]
                            .as_f64()
                            .ok_or("first object time required")?,
                        detailed["exploration_times"][1]
                            .as_f64()
                            .ok_or("second object time required")?,
                        detailed["bout_counts"][0]
                            .as_u64()
                            .ok_or("first object bouts required")? as u32,
                        detailed["bout_counts"][1]
                            .as_u64()
                            .ok_or("second object bouts required")? as u32,
                    ))?;
            }
            if assay == "ymaze" {
                detailed["geometry"] = geometry;
                detailed["first_detection_s"] = 0.0.into();
                detailed["coordinate_samples"] = json!(points.iter().enumerate().map(|(sample,(head,tail))|
                    json!({"timestamp_s":sample as f64/FPS,"head_center_x":head[0],"head_center_y":head[1],
                        "tail_base_x":tail[0],"tail_base_y":tail[1]})).collect::<Vec<_>>());
            }
            trials.push(TrialResult { experiment:format!("native-{index}-{threshold}"), task_type, analysis_role:None,
                fps:FPS, meters_per_pixel:1.0, animal_metadata:json!({"animal":row["animal_id"]}),
                session_context:json!({"session_id":row["session_id"],"timepoint":row["timepoint"],"phase":row["phase"]}),
                summaries:serde_json::to_value(result.summaries)?.as_array().ok_or("summary array required")?.clone(), detailed });
        }
    }
    fs::create_dir_all(root.join("data"))?;
    fs::write(
        root.join("data/native_trials.json"),
        serde_json::to_vec_pretty(&trials)?,
    )?;
    for trial in &trials {
        let directory = root.join("data").join(&trial.experiment);
        fs::create_dir(&directory)?;
        fs::write(
            directory.join("result.toml"),
            bekiper_core::inspection::json_value_to_toml_string(&serde_json::to_value(trial)?)?,
        )?;
    }
    // The same fresh native checkpoint cohort is executable through `bekiper
    // run --only aggregate` with a different immutable run ID and CLI controls.
    let mut project: toml::Value = toml::from_str(&fs::read_to_string(&config_path)?)?;
    project["project"]
        .as_table_mut()
        .ok_or("project table required")?
        .insert("output_dir".into(), ".".into());
    project["post_analysis"]["plan_path"] = plan_path.to_string_lossy().into_owned().into();
    fs::write(
        root.join("bekiper_config.toml"),
        toml::to_string_pretty(&project)?,
    )?;
    let count = trials.len();
    export_aggregate_trials(&config, &root, trials)?;
    println!(
        "{}",
        json!({"run_root":root,"native_trial_count":count,"run_mode":"synthetic","real_data_execution":false})
    );
    Ok(())
}
