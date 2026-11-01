use std::path::{Path, PathBuf};

use bekiper_ingress::aggregate::aggregate_state_path_for_config;
use bekiper_ingress::stage::Stage;
use bekiper_ingress::workflow::{RunOptions, run_project};

fn write_result_toml(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        r#"
        experiment = "721_seq0_oultraining"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"

        [detailed]
        protocol = "oul_training"
        excluded = false
        total_duration_seconds = 120.0
        total_distance_meters = 5.0
        total_mean_speed = 0.04
        freezing_seconds = 1.2
        analysis_window_seconds = 120.0
        object_labels = []
        exploration_times = []
        bout_counts = []

        [detailed.effective_parameters.values]
        "scale.method" = "arena_side"
        "#,
    )
    .unwrap();
}

fn setup_config(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("bekiper_ingress_aggregate_workflow_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("output/trial_a")).unwrap();

    let config_path = dir.join("bekiper_config.toml");
    std::fs::write(
        &config_path,
        r#"
        [project]
        project_name = "aggregate"
        data_directory = "."
        output_dir = "output"
        "#,
    )
    .unwrap();
    write_result_toml(&dir.join("output/trial_a/result.toml"));
    (dir, config_path)
}

#[test]
fn enabled_cohort_plan_is_checked_before_any_pipeline_outputs() {
    let dir = std::env::temp_dir().join(format!("bekiper-plan-preflight-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let config = dir.join("bekiper_config.toml");
    std::fs::write(
        &config,
        r#"
        [project]
        project_name = "imen"
        data_directory = "."
        output_dir = "fresh-output"
        [post_analysis]
        enabled = true
        plan_path = "missing-registered-plan.json"
        "#,
    )
    .unwrap();
    let error = run_project(&config, &RunOptions::default()).unwrap_err();
    assert!(error.to_string().contains("post-analysis plan"), "{error}");
    assert!(!dir.join("fresh-output").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn run_project_only_aggregate_writes_default_workbook() {
    let (dir, config_path) = setup_config("only");

    run_project(
        &config_path,
        &RunOptions {
            only_stage: Some(Stage::Aggregate),
            ..Default::default()
        },
    )
    .unwrap();

    assert!(dir.join("output/results.ods").exists());
    let config = bekiper_ingress::config::IngressConfig::from_file(&config_path).unwrap();
    assert!(aggregate_state_path_for_config(&config, &dir.join("output")).exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn run_project_no_aggregate_skips_workbook() {
    let (dir, config_path) = setup_config("skip");

    run_project(
        &config_path,
        &RunOptions {
            only_stage: Some(Stage::Aggregate),
            no_aggregate: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(!dir.join("output/results.ods").exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn run_project_aggregate_output_override_wins() {
    let (dir, config_path) = setup_config("override");

    run_project(
        &config_path,
        &RunOptions {
            only_stage: Some(Stage::Aggregate),
            aggregate_output: Some("renamed.xlsx".into()),
            ..Default::default()
        },
    )
    .unwrap();

    assert!(dir.join("output/renamed.ods").exists());
    assert!(!dir.join("output/results.ods").exists());
    let mut config = bekiper_ingress::config::IngressConfig::from_file(&config_path).unwrap();
    config.aggregate.output_filename = "renamed.xlsx".into();
    assert!(aggregate_state_path_for_config(&config, &dir.join("output")).exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn fresh_behaviour_rerun_consumes_upstream_checkpoints_without_source_decoding() {
    use bekiper_core::inspection::{PerimeterSpec, PerimetersFile};
    use bekiper_feature::tolerance::DwellOverrides;
    use bekiper_ingress::{aggregate::TrialResult, stage::ExperimentOutputDir};
    use polars::prelude::*;

    let dir = std::env::temp_dir().join(format!("bekiper-upstream-reuse-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let perimeters = PerimetersFile {
        perimeters: vec![PerimeterSpec::RadialMaze {
            label: "maze".into(),
            center_vertices: vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
            arms: vec![
                vec![(0.1, -0.2), (1.1, -0.2), (1.1, 0.2), (0.1, 0.2)],
                vec![(-0.2, 0.1), (0.2, 0.1), (0.2, 1.1), (-0.2, 1.1)],
                vec![(-1.1, -0.2), (-0.1, -0.2), (-0.1, 0.2), (-1.1, 0.2)],
            ],
        }],
    };
    let source = dir.join("pose.h5");
    // Any attempt to decode/infer/augment this source will fail. Only the
    // admitted augmented coordinates and meter-space geometry are usable.
    std::fs::write(&source, b"upstream source must not be decoded").unwrap();
    for (label, minimum, entries) in [("default", None, 3), ("zero", Some(0.0), 4)] {
        let run_root = dir.join(label);
        let trial = ExperimentOutputDir::new(&run_root, "affected_ymaze");
        trial.ensure_dir().unwrap();
        trial.ensure_checkpoints_dir().unwrap();
        let x = [
            0.0, 0.9, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, -0.9, -0.9, -0.9, 0.0,
        ];
        let y = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut augmented = df! {
            "frame" => (0u64..13).collect::<Vec<_>>(),
            "mouse_center_x" => x, "mouse_center_y" => y,
            "nose_x" => x, "nose_y" => y,
            "center_ear_x" => x, "center_ear_y" => y,
            "tail_base_x" => x, "tail_base_y" => y
        }
        .unwrap();
        bekiper_pose::io::write_parquet(&mut augmented, &trial.augmented_parquet()).unwrap();
        let augmented_bytes = std::fs::read(trial.augmented_parquet()).unwrap();
        let geometry = toml::to_string(&perimeters).unwrap();
        std::fs::write(trial.perimeters_toml(), &geometry).unwrap();
        std::fs::write(trial.pixel_perimeters_toml(), &geometry).unwrap();
        let config_path = dir.join(format!("{label}.toml"));
        std::fs::write(
            &config_path,
            format!(
                r#"
[project]
project_name = "synthetic-reuse"
data_directory = "."
output_dir = "{label}"
[project.runtime]
num_threads = 1
memory_scheduler_enabled = false
[[experiments]]
name = "affected_ymaze"
task_type = "y_maze"
data_files = ["pose.h5"]
fps = 2.0
meters_per_pixel = 0.001
"#
            ),
        )
        .unwrap();
        run_project(
            &config_path,
            &RunOptions {
                from_stage: Some(Stage::Behaviour),
                no_inspect: true,
                no_aggregate: true,
                ymaze_dwell: DwellOverrides {
                    seconds: minimum,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let result = TrialResult::load(&trial.root().join("result.toml")).unwrap();
        assert_eq!(
            result.detailed["analysis"]["arm_sequence"]
                .as_array()
                .unwrap()
                .len(),
            entries
        );
        assert_eq!(
            result.detailed["scoring_provenance"]["minimum_dwell_seconds"],
            minimum.unwrap_or(1.0)
        );
        assert_eq!(
            std::fs::read(trial.augmented_parquet()).unwrap(),
            augmented_bytes
        );
        assert_eq!(
            std::fs::read_to_string(trial.perimeters_toml()).unwrap(),
            geometry
        );
        assert!(!trial.poses_parquet().exists());
        assert!(!run_root.join("data/unaffected_oft").exists());
    }
    assert_eq!(
        std::fs::read(&source).unwrap(),
        b"upstream source must not be decoded"
    );
    std::fs::remove_dir_all(dir).unwrap();
}
