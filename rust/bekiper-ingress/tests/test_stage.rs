use std::path::Path;

use bekiper_ingress::stage::{
    ExperimentOutputDir, Stage, aggregate_output_path, aggregate_state_path,
};

#[test]
fn stage_name_detect() {
    assert_eq!(Stage::Detect.name(), "detect");
}

#[test]
fn stage_name_all() {
    let expected = [
        "detect",
        "pose",
        "augment",
        "behaviour",
        "summary",
        "inspect",
        "aggregate",
    ];
    let names: Vec<&str> = Stage::all().iter().map(|s| s.name()).collect();
    assert_eq!(names, expected);
}

#[test]
fn stage_from_name_valid() {
    assert_eq!(Stage::from_name("pose").unwrap(), Stage::Pose);
}

#[test]
fn stage_from_name_invalid() {
    assert!(Stage::from_name("nonexistent").is_err());
}

#[test]
fn stage_all_order() {
    let all = Stage::all();
    assert_eq!(all.len(), 7);
    assert_eq!(all[0], Stage::Detect);
    assert_eq!(all[5], Stage::Inspect);
    assert_eq!(all[6], Stage::Aggregate);
}

#[test]
fn stage_from_onward_beginning() {
    let stages = Stage::from_onward(Stage::Detect);
    assert_eq!(stages.len(), 7);
    assert_eq!(stages[0], Stage::Detect);
    assert_eq!(stages[6], Stage::Aggregate);
}

#[test]
fn stage_from_onward_middle() {
    let stages = Stage::from_onward(Stage::Behaviour);
    assert_eq!(
        stages,
        vec![
            Stage::Behaviour,
            Stage::Summary,
            Stage::Inspect,
            Stage::Aggregate
        ]
    );
}

#[test]
fn stage_from_onward_last() {
    let stages = Stage::from_onward(Stage::Aggregate);
    assert_eq!(stages, vec![Stage::Aggregate]);
}

#[test]
fn stage_from_onward_aggregate_via_name() {
    // `--from aggregate` parses through from_name and then resolves the
    // tail of the pipeline.
    let parsed = Stage::from_name("aggregate").unwrap();
    assert_eq!(Stage::from_onward(parsed), vec![Stage::Aggregate]);
}

#[test]
fn stage_ordering() {
    assert!(Stage::Detect < Stage::Pose);
    assert!(Stage::Pose < Stage::Augment);
    assert!(Stage::Augment < Stage::Behaviour);
    assert!(Stage::Behaviour < Stage::Summary);
    assert!(Stage::Summary < Stage::Inspect);
    assert!(Stage::Inspect < Stage::Aggregate);
}

#[test]
fn output_dir_paths() {
    let dir = ExperimentOutputDir::new(Path::new("/tmp/output"), "mouse_01");
    assert_eq!(dir.root(), Path::new("/tmp/output/data/mouse_01"));
    assert_eq!(
        dir.pixel_perimeters_toml(),
        Path::new("/tmp/checkpoint/mouse_01/pixel_perimeters.toml")
    );
    assert_eq!(
        dir.perimeters_toml(),
        Path::new("/tmp/checkpoint/mouse_01/perimeters.toml")
    );
    assert_eq!(
        dir.poses_parquet(),
        Path::new("/tmp/checkpoint/mouse_01/poses.parquet")
    );
    assert_eq!(
        dir.augmented_parquet(),
        Path::new("/tmp/output/data/mouse_01/augmented.parquet")
    );
    assert_eq!(
        dir.behaviour_parquet(),
        Path::new("/tmp/output/data/mouse_01/behaviour.parquet")
    );
    assert_eq!(
        dir.result_toml(),
        Path::new("/tmp/output/data/mouse_01/result.toml")
    );
    assert_eq!(
        dir.inspection_toml(),
        Path::new("/tmp/output/data/mouse_01/mouse_01_inspection.toml")
    );
    assert_eq!(
        dir.evaluated_parquet(),
        Path::new("/tmp/output/data/mouse_01/mouse_01_evaluated.parquet")
    );
    assert_eq!(
        dir.spreadsheet_workbook(),
        Path::new("/tmp/output/data/mouse_01/mouse_01.ods")
    );
    assert_eq!(
        dir.scale_toml(),
        Path::new("/tmp/checkpoint/mouse_01/scale.toml")
    );
    assert_eq!(
        dir.behaviour_summaries_toml(),
        Path::new("/tmp/checkpoint/mouse_01/behaviour_summaries.toml")
    );
    assert_eq!(
        dir.behaviour_result_toml(),
        Path::new("/tmp/checkpoint/mouse_01/behaviour_result.toml")
    );
    assert_eq!(
        dir.results_parquet(),
        Path::new("/tmp/output/data/mouse_01/results.parquet")
    );
    assert_eq!(
        dir.figures_dir(),
        Path::new("/tmp/output/data/mouse_01/figures")
    );
    assert_eq!(
        dir.videos_dir(),
        Path::new("/tmp/output/data/mouse_01/videos")
    );
    assert_eq!(
        dir.frames_dir(),
        Path::new("/tmp/output/data/mouse_01/frames")
    );
    assert_eq!(
        dir.pose_chunks_dir(),
        Path::new("/tmp/checkpoint/mouse_01/pose_chunks")
    );
}

#[test]
fn stage_artifacts_detect_and_augment_use_split_perimeter_paths() {
    let dir = ExperimentOutputDir::new(Path::new("/tmp/output"), "mouse_01");

    let detect = dir.stage_artifacts(Stage::Detect).unwrap();
    assert_eq!(detect.primary_output, dir.pixel_perimeters_toml());
    assert!(detect.owned_paths.contains(&dir.pixel_perimeters_toml()));
    assert!(!detect.owned_paths.contains(&dir.perimeters_toml()));

    let augment = dir.stage_artifacts(Stage::Augment).unwrap();
    assert_eq!(augment.primary_output, dir.augmented_parquet());
    assert!(augment.owned_paths.contains(&dir.perimeters_toml()));
    assert!(augment.owned_paths.contains(&dir.scale_toml()));
}

#[test]
fn stage_artifacts_summary_include_manifest_and_frames() {
    let dir = ExperimentOutputDir::new(Path::new("/tmp/output"), "mouse_01");
    let artifacts = dir.stage_artifacts(Stage::Summary).unwrap();
    assert_eq!(artifacts.primary_output, dir.result_toml());
    assert!(artifacts.owned_paths.contains(&dir.result_toml()));
    assert!(artifacts.owned_paths.contains(&dir.inspection_toml()));
    assert!(artifacts.owned_paths.contains(&dir.evaluated_parquet()));
    assert!(artifacts.owned_paths.contains(&dir.spreadsheet_workbook()));
    assert!(artifacts.owned_paths.contains(&dir.frames_dir()));
}

#[test]
fn purge_paths_from_summary_include_downstream_outputs() {
    let dir = ExperimentOutputDir::new(Path::new("/tmp/output"), "mouse_01");
    let paths = dir.purge_paths_from(Stage::Summary);
    assert!(paths.contains(&dir.result_toml()));
    assert!(paths.contains(&dir.inspection_toml()));
    assert!(paths.contains(&dir.evaluated_parquet()));
    assert!(paths.contains(&dir.spreadsheet_workbook()));
    assert!(paths.contains(&dir.frames_dir()));
    assert!(paths.contains(&dir.figures_dir()));
    assert!(paths.contains(&dir.videos_dir()));
    assert!(!paths.contains(&dir.behaviour_parquet()));
}

#[test]
fn purge_paths_from_augment_include_meter_perimeters_but_not_pixel_perimeters() {
    let dir = ExperimentOutputDir::new(Path::new("/tmp/output"), "mouse_01");
    let paths = dir.purge_paths_from(Stage::Augment);
    assert!(paths.contains(&dir.perimeters_toml()));
    assert!(!paths.contains(&dir.pixel_perimeters_toml()));
}

#[test]
fn aggregate_output_path_uses_run_root_and_filename() {
    assert_eq!(
        aggregate_output_path(Path::new("/tmp/output"), "renamed.xlsx"),
        Path::new("/tmp/output/renamed.ods")
    );
}

#[test]
fn aggregate_state_path_uses_run_root_and_filename() {
    assert_eq!(
        aggregate_state_path(Path::new("/tmp/output"), "renamed.xlsx"),
        Path::new("/tmp/output/.renamed.ods.aggregate.toml")
    );
}

#[test]
fn output_dir_ensure_creates_dir() {
    let tmp = std::env::temp_dir().join("bekiper_test_stage_ensure");
    let _ = std::fs::remove_dir_all(&tmp);

    let dir = ExperimentOutputDir::new(&tmp, "exp1");
    assert!(!dir.root().exists());

    dir.ensure_dir().unwrap();
    assert!(dir.root().exists());
    assert!(dir.root().is_dir());

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn output_dir_is_fresh_missing_output() {
    let tmp = std::env::temp_dir().join("bekiper_test_stage_fresh_missing");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let dir = ExperimentOutputDir::new(&tmp, "exp1");
    // Output file does not exist, so not fresh.
    assert!(!dir.is_fresh(Stage::Detect, None));

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn output_dir_is_fresh_no_input() {
    let tmp = std::env::temp_dir().join("bekiper_test_stage_fresh_no_input");
    let _ = std::fs::remove_dir_all(&tmp);

    let dir = ExperimentOutputDir::new(&tmp, "exp1");
    dir.ensure_dir().unwrap();

    // Create the checkpoints dir and output file so it exists.
    let perim = dir.pixel_perimeters_toml();
    std::fs::create_dir_all(perim.parent().unwrap()).unwrap();
    std::fs::write(&perim, b"{}").unwrap();

    // No input path provided -- output exists, so fresh.
    assert!(dir.is_fresh(Stage::Detect, None));

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn output_dir_is_fresh_newer_output() {
    let tmp = std::env::temp_dir().join("bekiper_test_stage_fresh_newer");
    let _ = std::fs::remove_dir_all(&tmp);

    let dir = ExperimentOutputDir::new(&tmp, "exp1");
    dir.ensure_dir().unwrap();

    // Create input first.
    let input = tmp.join("input.csv");
    std::fs::write(&input, b"data").unwrap();

    std::thread::sleep(std::time::Duration::from_millis(50));

    // Create output after input so it is newer.
    std::fs::create_dir_all(dir.checkpoints_dir()).unwrap();
    std::fs::write(dir.poses_parquet(), b"parquet").unwrap();

    assert!(dir.is_fresh(Stage::Pose, Some(&input)));

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn output_dir_is_fresh_older_output() {
    let tmp = std::env::temp_dir().join("bekiper_test_stage_fresh_older");
    let _ = std::fs::remove_dir_all(&tmp);

    let dir = ExperimentOutputDir::new(&tmp, "exp1");
    dir.ensure_dir().unwrap();

    // Create output first.
    std::fs::write(dir.augmented_parquet(), b"parquet").unwrap();

    std::thread::sleep(std::time::Duration::from_millis(50));

    // Create input after output so output is stale.
    let input = tmp.join("input.csv");
    std::fs::write(&input, b"data").unwrap();

    assert!(!dir.is_fresh(Stage::Augment, Some(&input)));

    let _ = std::fs::remove_dir_all(&tmp);
}
