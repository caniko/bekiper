//! Integration tests: multi-step pipeline, confinement, and IO roundtrips.

use polars::prelude::*;

use bekiper_core::types::MidpointGroup;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_pose::pipeline::AugmentedBuilder;
use bekiper_pose::skeleton::{Edge, Node, Skeleton};
use bekiper_pose::steps::confinement::ConfinementFilter;
use bekiper_pose::steps::likelihood::LikelihoodFilter;
use bekiper_pose::steps::midpoint::MidpointComputer;
use bekiper_pose::steps::skeleton::SkeletonRepairStep;
use bekiper_pose::steps::transform::CoordinateTransform;

/// Synthetic raw tracking data in pixel coordinates, mimicking DLC output.
fn raw_pixel_df(n: usize) -> LazyFrame {
    let frames: Vec<i64> = (0..n as i64).collect();
    // Smooth trajectory in pixel space (640x480 resolution)
    let nose_x: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 2.0).collect();
    let nose_y: Vec<f64> = (0..n).map(|i| 200.0 + (i as f64) * 1.5).collect();
    let nose_lik: Vec<f64> = (0..n)
        .map(|i| if i % 10 == 0 { 0.3 } else { 0.95 })
        .collect();

    let left_ear_x: Vec<f64> = nose_x.iter().map(|x| x - 10.0).collect();
    let left_ear_y: Vec<f64> = nose_y.iter().map(|y| y + 5.0).collect();
    let left_ear_lik: Vec<f64> = vec![0.99; n];

    let right_ear_x: Vec<f64> = nose_x.iter().map(|x| x + 10.0).collect();
    let right_ear_y: Vec<f64> = nose_y.iter().map(|y| y + 5.0).collect();
    let right_ear_lik: Vec<f64> = vec![0.99; n];

    // Inject a teleport artifact at frame 5
    let mut tail_base_x: Vec<f64> = nose_x.iter().map(|x| x - 30.0).collect();
    let mut tail_base_y: Vec<f64> = nose_y.iter().map(|y| y - 10.0).collect();
    if n > 5 {
        tail_base_x[5] = 999.0; // teleport
        tail_base_y[5] = 999.0;
    }
    let tail_base_lik: Vec<f64> = vec![0.99; n];

    df! {
        "frame" => &frames,
        "nose_x" => &nose_x,
        "nose_y" => &nose_y,
        "nose_likelihood" => &nose_lik,
        "left_ear_x" => &left_ear_x,
        "left_ear_y" => &left_ear_y,
        "left_ear_likelihood" => &left_ear_lik,
        "right_ear_x" => &right_ear_x,
        "right_ear_y" => &right_ear_y,
        "right_ear_likelihood" => &right_ear_lik,
        "tail_base_x" => &tail_base_x,
        "tail_base_y" => &tail_base_y,
        "tail_base_likelihood" => &tail_base_lik,
    }
    .unwrap()
    .lazy()
}

#[test]
fn full_pipeline_transforms_coordinates() {
    let mpp = 0.001; // 1 pixel = 1mm
    let pipeline = AugmentedBuilder::new().add_step(CoordinateTransform {
        meters_per_pixel: mpp,
        meters_per_pixel_y: None,
        invert_y: true,
        y_height: 480.0,
        x_columns: vec!["nose_x".into(), "left_ear_x".into(), "right_ear_x".into()],
        y_columns: vec!["nose_y".into(), "left_ear_y".into(), "right_ear_y".into()],
    });

    let df = pipeline.build_and_collect(raw_pixel_df(20)).unwrap();

    // Check that coordinates are now in meters (original pixel 100 → 0.1m)
    let nose_x = df.column("nose_x").unwrap().f64().unwrap();
    let first_x = nose_x.get(0).unwrap();
    assert!(
        (first_x - 0.1).abs() < 1e-6,
        "expected ~0.1m, got {first_x}"
    );

    // Y should be inverted: original 200px → (480 - 200) * 0.001 = 0.28m
    let nose_y = df.column("nose_y").unwrap().f64().unwrap();
    let first_y = nose_y.get(0).unwrap();
    assert!(
        (first_y - 0.28).abs() < 1e-6,
        "expected ~0.28m, got {first_y}"
    );
}

#[test]
fn likelihood_filter_nulls_low_confidence() {
    let pipeline = AugmentedBuilder::new().add_step(LikelihoodFilter::new(
        0.5,
        vec!["nose_likelihood".into()],
        vec!["nose_x".into()],
    ));

    let df = pipeline.build_and_collect(raw_pixel_df(20)).unwrap();

    let nose_x = df.column("nose_x").unwrap();
    // Every 10th frame has likelihood=0.3 (below 0.5 threshold), so those should be null
    assert!(
        nose_x.get(0).unwrap() == AnyValue::Null,
        "frame 0 should be nulled (lik=0.3)"
    );
    assert!(
        nose_x.get(1).unwrap() != AnyValue::Null,
        "frame 1 should be kept (lik=0.95)"
    );
    assert!(
        nose_x.get(10).unwrap() == AnyValue::Null,
        "frame 10 should be nulled"
    );
}

#[test]
fn midpoint_computer_adds_center_ear() {
    let pipeline = AugmentedBuilder::new().add_step(MidpointComputer {
        groups: vec![MidpointGroup {
            output_label: "center_ear".into(),
            source_labels: vec!["left_ear".into(), "right_ear".into()],
        }],
    });

    let df = pipeline.build_and_collect(raw_pixel_df(10)).unwrap();

    // center_ear_x should be average of left_ear_x and right_ear_x
    assert!(df.column("center_ear_x").is_ok());
    assert!(df.column("center_ear_y").is_ok());

    let ce_x = df.column("center_ear_x").unwrap().f64().unwrap();
    let le_x = df.column("left_ear_x").unwrap().f64().unwrap();
    let re_x = df.column("right_ear_x").unwrap().f64().unwrap();

    for i in 0..10 {
        let expected = (le_x.get(i).unwrap() + re_x.get(i).unwrap()) / 2.0;
        let actual = ce_x.get(i).unwrap();
        assert!(
            (actual - expected).abs() < 1e-10,
            "center_ear_x[{i}]: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn multi_step_pipeline_chains_correctly() {
    let mpp = 0.001;
    let pipeline = AugmentedBuilder::new()
        .add_step(LikelihoodFilter::new(
            0.5,
            vec!["nose_likelihood".into()],
            vec!["nose_x".into()],
        ))
        .add_step(CoordinateTransform {
            meters_per_pixel: mpp,
            meters_per_pixel_y: None,
            invert_y: false,
            y_height: 480.0,
            x_columns: vec!["nose_x".into()],
            y_columns: vec!["nose_y".into()],
        })
        .add_step(MidpointComputer {
            groups: vec![MidpointGroup {
                output_label: "center_ear".into(),
                source_labels: vec!["left_ear".into(), "right_ear".into()],
            }],
        });

    let df = pipeline.build_and_collect(raw_pixel_df(20)).unwrap();

    // Likelihood filter ran → nulls present
    let nose_x = df.column("nose_x").unwrap();
    assert!(nose_x.get(0).unwrap() == AnyValue::Null);

    // Coordinate transform ran → non-null values in meters
    let val = nose_x.get(1).unwrap();
    if let AnyValue::Float64(v) = val {
        assert!(v < 1.0, "should be in meters, not pixels");
    }

    // Midpoint computed
    assert!(df.column("center_ear_x").is_ok());
}

#[test]
fn confinement_filter_restricts_to_enclosure() {
    // Create a circle enclosure in pixel space centered at (200, 300) with radius 100px
    let enclosure = Perimeter::new("arena", Circle::new(200.0, 300.0, 100.0));
    let pipeline = AugmentedBuilder::new().add_step(ConfinementFilter {
        perimeter: enclosure,
        coords: bekiper_core::types::CoordinateColumns::new("nose_x", "nose_y"),
    });

    let n = 50;
    let df = pipeline.build_and_collect(raw_pixel_df(n)).unwrap();

    // Only rows where nose is inside the circle should remain
    assert!(df.height() < n, "some rows should be filtered out");
    assert!(df.height() > 0, "some rows should be inside the enclosure");
}

#[test]
fn pipeline_build_and_sink_parquet_roundtrip() {
    let pipeline = AugmentedBuilder::new().add_step(MidpointComputer {
        groups: vec![MidpointGroup {
            output_label: "center_ear".into(),
            source_labels: vec!["left_ear".into(), "right_ear".into()],
        }],
    });

    let tmp_dir = std::env::temp_dir().join("bekiper_test_pipeline_sink");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let parquet_path = tmp_dir.join("output.parquet");

    pipeline
        .build_and_sink_parquet(raw_pixel_df(30), &parquet_path)
        .unwrap();

    // Read back and verify
    let df = LazyFrame::scan_parquet(parquet_path.to_str().unwrap().into(), Default::default())
        .unwrap()
        .collect()
        .unwrap();

    assert_eq!(df.height(), 30);
    assert!(df.column("center_ear_x").is_ok());
    assert!(df.column("center_ear_y").is_ok());

    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn io_scan_csv_and_parquet_roundtrip() {
    let tmp_dir = std::env::temp_dir().join("bekiper_test_io_roundtrip_pose");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();

    // Write a CSV
    let csv_path = tmp_dir.join("test.csv");
    let mut df = df! {
        "x" => &[1.0, 2.0, 3.0],
        "y" => &[4.0, 5.0, 6.0],
    }
    .unwrap();
    CsvWriter::new(std::fs::File::create(&csv_path).unwrap())
        .finish(&mut df)
        .unwrap();

    // Scan CSV
    let lf = bekiper_pose::io::scan_file(&csv_path).unwrap();
    let loaded = lf.collect().unwrap();
    assert_eq!(loaded.shape(), (3, 2));

    // Write as Parquet
    let pq_path = tmp_dir.join("test.parquet");
    bekiper_pose::io::write_parquet(&mut df, &pq_path).unwrap();

    // Scan Parquet
    let lf2 = bekiper_pose::io::scan_file(&pq_path).unwrap();
    let loaded2 = lf2.collect().unwrap();
    assert_eq!(loaded2.shape(), (3, 2));

    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn midpoint_generated_labels_participate_in_skeleton_repair() {
    let raw = df! {
        "frame" => &[0i64, 1, 2],
        "left_ear_x" => &[0.0, 0.0, 0.0],
        "left_ear_y" => &[0.0, 0.0, 0.0],
        "right_ear_x" => &[2.0, 2.0, 8.0],
        "right_ear_y" => &[0.0, 0.0, 0.0],
    }
    .unwrap()
    .lazy();

    let skeleton = Skeleton::new(
        "ears",
        vec![
            Node::new("left_ear"),
            Node::new("center_ear"),
            Node::new("right_ear"),
        ],
        vec![
            Edge::new("left_ear", "center_ear"),
            Edge::new("center_ear", "right_ear"),
        ],
    );

    let pipeline = AugmentedBuilder::new()
        .add_step(MidpointComputer {
            groups: vec![MidpointGroup {
                output_label: "center_ear".into(),
                source_labels: vec!["left_ear".into(), "right_ear".into()],
            }],
        })
        .add_step(SkeletonRepairStep {
            skeleton,
            relative_tolerance: 0.35,
            mad_multiplier: 4.0,
            min_baseline_frames: 2,
        });

    let df = pipeline.build_and_collect(raw).unwrap();
    assert!(df.column("center_ear_x").is_ok());
    assert!(df.column("center_ear_skeleton_violation").is_ok());
}
