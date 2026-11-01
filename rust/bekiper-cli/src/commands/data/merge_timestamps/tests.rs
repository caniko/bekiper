use super::*;

fn test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bekiper_merge_timestamps_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_pose_parquet(path: &Path, frames: usize) {
    let frames_col: Vec<i64> = (0..frames as i64).collect();
    let xs: Vec<f64> = (0..frames).map(|i| i as f64).collect();
    let ys: Vec<f64> = (0..frames).map(|i| 10.0 + i as f64).collect();
    let liks: Vec<f64> = vec![0.95; frames];
    let mut df = polars::df! {
        "frame" => frames_col,
        "nose_x" => xs,
        "nose_y" => ys,
        "nose_likelihood" => liks,
    }
    .unwrap();
    bekiper_pose::io::write_parquet(&mut df, path).unwrap();
}

fn write_timestamps_csv(path: &Path, count: usize) {
    let mut content = String::new();
    for i in 0..count {
        let millis = i * 40;
        let seconds = millis / 1000;
        let fraction = (millis % 1000) * 1_000_000;
        content.push_str(&format!(
            "ignored,2023-08-19T12:07:{seconds:02}.{fraction:09}+02:00\n"
        ));
    }
    std::fs::write(path, content).unwrap();
}

#[test]
fn merge_writes_timestamped_parquet_and_keeps_sources_by_default() {
    let root = test_dir("writes_output");
    let unit = root.join("animal_1");
    std::fs::create_dir_all(&unit).unwrap();

    let coord = unit.join("1.coordinates-7_example.parquet");
    let timestamp = unit.join("1.timestamps-7_example.csv");
    write_pose_parquet(&coord, 3);
    write_timestamps_csv(&timestamp, 3);

    run(MergeTimestampsArgs {
        dataset_dir: root.clone(),
        force: false,
        delete_source: false,
        delimiter: ".".into(),
    })
    .unwrap();

    let output = unit.join("1.coordinates-7_example-timestamped.parquet");
    assert!(output.exists());
    assert!(coord.exists());
    assert!(timestamp.exists());

    let df = scan_file(&output).unwrap().collect().unwrap();
    let elapsed: Vec<f64> = df
        .column("elapsed_s")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(elapsed.len(), 3);
    assert!((elapsed[1] - 0.04).abs() < 1e-6);

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn merge_skips_mismatched_timestamp_lengths() {
    let root = test_dir("mismatch");
    let unit = root.join("animal_2");
    std::fs::create_dir_all(&unit).unwrap();

    let coord = unit.join("2.coordinates-8_example.parquet");
    let timestamp = unit.join("2.timestamps-8_example.csv");
    write_pose_parquet(&coord, 3);
    write_timestamps_csv(&timestamp, 2);

    run(MergeTimestampsArgs {
        dataset_dir: root.clone(),
        force: false,
        delete_source: false,
        delimiter: ".".into(),
    })
    .unwrap();

    let output = unit.join("2.coordinates-8_example-timestamped.parquet");
    assert!(!output.exists());
    assert!(coord.exists());
    assert!(timestamp.exists());

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn merge_deletes_sources_when_requested() {
    let root = test_dir("delete_source");
    let unit = root.join("animal_3");
    std::fs::create_dir_all(&unit).unwrap();

    let coord = unit.join("3.coordinates-9_example.parquet");
    let timestamp = unit.join("3.timestamps-9_example.csv");
    write_pose_parquet(&coord, 2);
    write_timestamps_csv(&timestamp, 2);

    run(MergeTimestampsArgs {
        dataset_dir: root.clone(),
        force: false,
        delete_source: true,
        delimiter: ".".into(),
    })
    .unwrap();

    let output = unit.join("3.coordinates-9_example-timestamped.parquet");
    assert!(output.exists());
    assert!(!coord.exists());
    assert!(!timestamp.exists());

    std::fs::remove_dir_all(&root).ok();
}
