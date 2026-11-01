use polars::prelude::*;

use bekiper_core::types::MidpointGroup;
use bekiper_pose::pipeline::PipelineStep;
use bekiper_pose::skeleton::{Edge, Node, Skeleton};
use bekiper_pose::steps::likelihood::LikelihoodFilter;
use bekiper_pose::steps::midpoint::MidpointComputer;
use bekiper_pose::steps::skeleton::SkeletonRepairStep;
use bekiper_pose::steps::smoothing::{
    ConfidenceKalmanSmoother, KalmanSmootherColumns, MedianSmoother, SkeletonKinematicSmoother,
};
use bekiper_pose::steps::transform::CoordinateTransform;
use bekiper_pose::steps::velocity::HighVelocityFilter;

fn line_skeleton() -> Skeleton {
    Skeleton::new(
        "line",
        vec![Node::new("a"), Node::new("b"), Node::new("c")],
        vec![Edge::new("a", "b"), Edge::new("b", "c")],
    )
}

#[test]
fn likelihood_filter_nulls_low_confidence() {
    let df = df! {
        "nose_x" => &[1.0, 2.0, 3.0],
        "nose_likelihood" => &[0.9, 0.1, 0.8],
    }
    .unwrap();

    let step = LikelihoodFilter::new(0.5, vec!["nose_likelihood".into()], vec!["nose_x".into()]);
    assert_eq!(step.name(), "likelihood_filter");

    let result = step.apply(df.lazy()).collect().unwrap();
    let x = result.column("nose_x").unwrap().f64().unwrap();
    assert!((x.get(0).unwrap() - 1.0).abs() < 1e-10);
    assert!(x.get(1).is_none()); // below threshold → null
    assert!((x.get(2).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn high_velocity_filter_nulls_teleports() {
    let df = df! {
        "x" => &[0.0, 100.0, 101.0, 102.0],
        "y" => &[0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();

    let step = HighVelocityFilter {
        max_velocity: 50.0,
        x_columns: vec!["x".into()],
        y_columns: vec!["y".into()],
        max_iterations: 3,
        sigma_threshold: 0.0,
    };
    assert_eq!(step.name(), "high_velocity_filter");

    let result = step.apply(df.lazy()).collect().unwrap();
    let x = result.column("x").unwrap().f64().unwrap();
    // Frame 1 had velocity 100 > 50 → nullified
    assert!(x.get(1).is_none());
}

#[test]
fn coordinate_transform_pixels_to_meters() {
    let df = df! {
        "x" => &[100.0, 200.0],
        "y" => &[50.0, 100.0],
    }
    .unwrap();

    let step = CoordinateTransform {
        meters_per_pixel: 0.01,
        meters_per_pixel_y: None,
        invert_y: false,
        y_height: 0.0,
        x_columns: vec!["x".into()],
        y_columns: vec!["y".into()],
    };
    assert_eq!(step.name(), "coordinate_transform");

    let result = step.apply(df.lazy()).collect().unwrap();
    let x = result.column("x").unwrap().f64().unwrap();
    assert!((x.get(0).unwrap() - 1.0).abs() < 1e-10);
    assert!((x.get(1).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn coordinate_transform_invert_y() {
    let df = df! {
        "x" => &[100.0],
        "y" => &[0.0],
    }
    .unwrap();

    let step = CoordinateTransform {
        meters_per_pixel: 0.01,
        meters_per_pixel_y: None,
        invert_y: true,
        y_height: 200.0, // 200px
        x_columns: vec!["x".into()],
        y_columns: vec!["y".into()],
    };

    let result = step.apply(df.lazy()).collect().unwrap();
    let y = result.column("y").unwrap().f64().unwrap();
    // height_m = 200 * 0.01 = 2.0, pixel_to_meters(0) = 0.0, result = 2.0 - 0.0 = 2.0
    assert!((y.get(0).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn midpoint_computer_averages() {
    let df = df! {
        "left_ear_x" => &[0.0, 2.0],
        "left_ear_y" => &[0.0, 4.0],
        "right_ear_x" => &[2.0, 6.0],
        "right_ear_y" => &[4.0, 8.0],
    }
    .unwrap();

    let step = MidpointComputer {
        groups: vec![MidpointGroup {
            output_label: "center_ear".into(),
            source_labels: vec!["left_ear".into(), "right_ear".into()],
        }],
    };
    assert_eq!(step.name(), "midpoint_computer");

    let result = step.apply(df.lazy()).collect().unwrap();
    let cx = result.column("center_ear_x").unwrap().f64().unwrap();
    let cy = result.column("center_ear_y").unwrap().f64().unwrap();
    assert!((cx.get(0).unwrap() - 1.0).abs() < 1e-10);
    assert!((cy.get(0).unwrap() - 2.0).abs() < 1e-10);
    assert!((cx.get(1).unwrap() - 4.0).abs() < 1e-10);
    assert!((cy.get(1).unwrap() - 6.0).abs() < 1e-10);
}

#[test]
fn median_smoother_smooths_columns() {
    let df = df! {
        "x" => &[1.0, 100.0, 1.0, 1.0, 1.0],
    }
    .unwrap();

    let step = MedianSmoother {
        window_size: 3,
        columns: vec!["x".into()],
    };
    assert_eq!(step.name(), "median_smoother");

    let result = step.apply(df.lazy()).collect().unwrap();
    let x = result.column("x").unwrap().f64().unwrap();
    // The spike at index 1 should be smoothed
    assert!((x.get(1).unwrap() - 1.0).abs() < 1e-10);
}

fn nose_kalman_step(max_gap_frames: usize) -> ConfidenceKalmanSmoother {
    ConfidenceKalmanSmoother {
        columns: vec![KalmanSmootherColumns {
            label: "nose".into(),
            x_col: "nose_x".into(),
            y_col: "nose_y".into(),
            likelihood_col: Some("nose_likelihood".into()),
        }],
        process_noise: 0.25,
        measurement_noise: 2.0,
        min_likelihood: 0.05,
        max_gap_frames,
    }
}

#[test]
fn kalman_smoother_keeps_clean_linear_motion_stable() {
    let df = df! {
        "nose_x" => &[0.0, 1.0, 2.0, 3.0, 4.0],
        "nose_y" => &[10.0, 10.0, 10.0, 10.0, 10.0],
        "nose_likelihood" => &[0.99, 0.99, 0.99, 0.99, 0.99],
    }
    .unwrap();

    let result = nose_kalman_step(30).apply(df.lazy()).collect().unwrap();
    let x = result.column("nose_x").unwrap().f64().unwrap();
    let y = result.column("nose_y").unwrap().f64().unwrap();
    assert!((x.get(0).unwrap() - 0.0).abs() < 0.5);
    assert!((x.get(4).unwrap() - 4.0).abs() < 0.5);
    assert!((y.get(2).unwrap() - 10.0).abs() < 0.1);
}

#[test]
fn kalman_smoother_suppresses_low_likelihood_spike() {
    let df = df! {
        "nose_x" => &[0.0, 1.0, 100.0, 3.0, 4.0],
        "nose_y" => &[0.0, 0.0, 0.0, 0.0, 0.0],
        "nose_likelihood" => &[0.99, 0.99, 0.05, 0.99, 0.99],
    }
    .unwrap();

    let result = nose_kalman_step(30).apply(df.lazy()).collect().unwrap();
    let x = result.column("nose_x").unwrap().f64().unwrap();
    assert!(
        x.get(2).unwrap() < 40.0,
        "low-confidence spike should be downweighted, got {:?}",
        x.get(2)
    );
}

#[test]
fn kalman_smoother_fills_short_null_gap() {
    let df = df! {
        "nose_x" => &[Some(0.0), Some(1.0), None, Some(3.0), Some(4.0)],
        "nose_y" => &[Some(0.0), Some(0.0), None, Some(0.0), Some(0.0)],
        "nose_likelihood" => &[Some(0.99), Some(0.99), Some(0.0), Some(0.99), Some(0.99)],
    }
    .unwrap();

    let result = nose_kalman_step(30).apply(df.lazy()).collect().unwrap();
    let x = result.column("nose_x").unwrap().f64().unwrap();
    assert!(x.get(2).is_some());
    assert!((x.get(2).unwrap() - 2.0).abs() < 1.0);
}

#[test]
fn kalman_smoother_preserves_long_null_gap() {
    let df = df! {
        "nose_x" => &[Some(0.0), Some(1.0), None, None, None, Some(5.0)],
        "nose_y" => &[Some(0.0), Some(0.0), None, None, None, Some(0.0)],
        "nose_likelihood" => &[Some(0.99), Some(0.99), Some(0.0), Some(0.0), Some(0.0), Some(0.99)],
    }
    .unwrap();

    let result = nose_kalman_step(2).apply(df.lazy()).collect().unwrap();
    let x = result.column("nose_x").unwrap().f64().unwrap();
    assert!(x.get(2).is_none());
    assert!(x.get(3).is_none());
    assert!(x.get(4).is_none());
}

#[test]
fn kalman_smoother_uses_likelihood_weighting() {
    let low_conf_df = df! {
        "nose_x" => &[0.0, 1.0, 20.0, 3.0, 4.0],
        "nose_y" => &[0.0, 0.0, 0.0, 0.0, 0.0],
        "nose_likelihood" => &[0.99, 0.99, 0.05, 0.99, 0.99],
    }
    .unwrap();
    let high_conf_df = df! {
        "nose_x" => &[0.0, 1.0, 20.0, 3.0, 4.0],
        "nose_y" => &[0.0, 0.0, 0.0, 0.0, 0.0],
        "nose_likelihood" => &[0.99, 0.99, 0.99, 0.99, 0.99],
    }
    .unwrap();

    let low = nose_kalman_step(30)
        .apply(low_conf_df.lazy())
        .collect()
        .unwrap();
    let high = nose_kalman_step(30)
        .apply(high_conf_df.lazy())
        .collect()
        .unwrap();
    let low_x = low.column("nose_x").unwrap().f64().unwrap().get(2).unwrap();
    let high_x = high
        .column("nose_x")
        .unwrap()
        .f64()
        .unwrap()
        .get(2)
        .unwrap();
    assert!(low_x < high_x);
}

#[test]
fn skeleton_kinematic_smoother_projects_short_gap_to_bone_length() {
    let df = df! {
        "nose_x" => &[0.0, 0.0, 0.0, 0.0, 0.0],
        "nose_y" => &[0.0, 0.0, 0.0, 0.0, 0.0],
        "nose_likelihood" => &[0.99, 0.99, 0.99, 0.99, 0.99],
        "tail_x" => &[10.0, 10.0, 80.0, 10.0, 10.0],
        "tail_y" => &[0.0, 0.0, 0.0, 0.0, 0.0],
        "tail_likelihood" => &[0.99, 0.99, 0.01, 0.99, 0.99],
    }
    .unwrap();
    let columns = vec![
        KalmanSmootherColumns {
            label: "nose".into(),
            x_col: "nose_x".into(),
            y_col: "nose_y".into(),
            likelihood_col: Some("nose_likelihood".into()),
        },
        KalmanSmootherColumns {
            label: "tail".into(),
            x_col: "tail_x".into(),
            y_col: "tail_y".into(),
            likelihood_col: Some("tail_likelihood".into()),
        },
    ];
    let independent = ConfidenceKalmanSmoother {
        columns: columns.clone(),
        process_noise: 1.0,
        measurement_noise: 2.0,
        min_likelihood: 0.05,
        max_gap_frames: 30,
    }
    .apply(df.clone().lazy())
    .collect()
    .unwrap();
    let skeletal = SkeletonKinematicSmoother {
        skeleton: Skeleton::new(
            "line",
            vec![Node::new("nose"), Node::new("tail")],
            vec![Edge::new("nose", "tail")],
        ),
        columns,
        process_noise: 1.0,
        measurement_noise: 2.0,
        min_likelihood: 0.05,
        max_gap_frames: 30,
        constraint_iterations: 2,
        constraint_strength: 1.0,
    }
    .apply(df.lazy())
    .collect()
    .unwrap();

    let independent_tail = independent
        .column("tail_x")
        .unwrap()
        .f64()
        .unwrap()
        .get(2)
        .unwrap();
    let skeletal_tail = skeletal
        .column("tail_x")
        .unwrap()
        .f64()
        .unwrap()
        .get(2)
        .unwrap();
    assert!(
        (skeletal_tail - 10.0).abs() < (independent_tail - 10.0).abs(),
        "skeleton-aware interpolation should be closer to learned bone length: independent={independent_tail}, skeletal={skeletal_tail}"
    );
}

#[test]
fn skeleton_repair_nulls_stretched_leaf_only() {
    let df = df! {
        "frame" => &[0i64, 1, 2],
        "a_x" => &[0.0, 0.0, 0.0],
        "a_y" => &[0.0, 0.0, 0.0],
        "b_x" => &[1.0, 1.0, 1.0],
        "b_y" => &[0.0, 0.0, 0.0],
        "c_x" => &[2.0, 2.0, 10.0],
        "c_y" => &[0.0, 0.0, 0.0],
    }
    .unwrap();

    let step = SkeletonRepairStep {
        skeleton: line_skeleton(),
        relative_tolerance: 0.35,
        mad_multiplier: 4.0,
        min_baseline_frames: 2,
    };

    let result = step.apply(df.lazy()).collect().unwrap();
    let b_x = result.column("b_x").unwrap().f64().unwrap();
    let c_x = result.column("c_x").unwrap().f64().unwrap();
    let c_flag = result
        .column("c_skeleton_violation")
        .unwrap()
        .bool()
        .unwrap();

    assert_eq!(b_x.get(2), Some(1.0));
    assert!(c_x.get(2).is_none());
    assert_eq!(c_flag.get(2), Some(true));
}

#[test]
fn skeleton_repair_nulls_broken_hub() {
    let df = df! {
        "frame" => &[0i64, 1, 2],
        "a_x" => &[0.0, 0.0, 0.0],
        "a_y" => &[0.0, 0.0, 0.0],
        "b_x" => &[1.0, 1.0, 5.0],
        "b_y" => &[0.0, 0.0, 0.0],
        "c_x" => &[2.0, 2.0, 2.0],
        "c_y" => &[0.0, 0.0, 0.0],
    }
    .unwrap();

    let step = SkeletonRepairStep {
        skeleton: line_skeleton(),
        relative_tolerance: 0.35,
        mad_multiplier: 4.0,
        min_baseline_frames: 2,
    };

    let result = step.apply(df.lazy()).collect().unwrap();
    let b_x = result.column("b_x").unwrap().f64().unwrap();
    let b_flag = result
        .column("b_skeleton_violation")
        .unwrap()
        .bool()
        .unwrap();

    assert!(b_x.get(2).is_none());
    assert_eq!(b_flag.get(2), Some(true));
}

#[test]
fn skeleton_repair_leaves_stable_rows_untouched() {
    let df = df! {
        "frame" => &[0i64, 1, 2, 3],
        "a_x" => &[0.0, 0.1, 0.2, 0.3],
        "a_y" => &[0.0, 0.0, 0.0, 0.0],
        "b_x" => &[1.0, 1.1, 1.2, 1.3],
        "b_y" => &[0.0, 0.0, 0.0, 0.0],
        "c_x" => &[2.0, 2.1, 2.2, 2.3],
        "c_y" => &[0.0, 0.0, 0.0, 0.0],
    }
    .unwrap();

    let step = SkeletonRepairStep {
        skeleton: line_skeleton(),
        relative_tolerance: 0.35,
        mad_multiplier: 4.0,
        min_baseline_frames: 2,
    };

    let result = step.apply(df.lazy()).collect().unwrap();
    let counts = result
        .column("skeleton_violation_count")
        .unwrap()
        .u32()
        .unwrap();
    assert!(counts.into_no_null_iter().all(|count| count == 0));
}

#[test]
fn coordinate_transform_scales_each_axis_with_its_own_metres_per_pixel() {
    let df = df! { "nose_x" => [100.0], "nose_y" => [200.0] }.unwrap();
    for (invert_y, expected_y) in [(false, 200.0 * 0.002), (true, (210.0 - 200.0) * 0.002)] {
        let step = CoordinateTransform {
            meters_per_pixel: 0.001,
            meters_per_pixel_y: Some(0.002),
            invert_y,
            y_height: 210.0,
            x_columns: vec!["nose_x".into()],
            y_columns: vec!["nose_y".into()],
        };
        let out = step.apply(df.clone().lazy()).collect().unwrap();
        let x = out.column("nose_x").unwrap().f64().unwrap().get(0).unwrap();
        let y = out.column("nose_y").unwrap().f64().unwrap().get(0).unwrap();
        assert!((x - 0.1).abs() < 1e-12, "{x}");
        assert!((y - expected_y).abs() < 1e-12, "invert {invert_y}: {y}");
    }
}

fn f64s(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
    df.column(name)
        .unwrap()
        .f64()
        .unwrap()
        .into_iter()
        .collect()
}

#[test]
fn likelihood_filter_keeps_values_exactly_at_threshold() {
    let df = df! { "nose_x" => [1.0, 2.0], "nose_likelihood" => [0.5, 0.4999] }.unwrap();
    let step = LikelihoodFilter::new(0.5, vec!["nose_likelihood".into()], vec!["nose_x".into()]);
    let out = step.apply(df.lazy()).collect().unwrap();
    assert_eq!(f64s(&out, "nose_x"), [Some(1.0), None]);
}

#[test]
fn likelihood_trimmer_slices_to_the_reliable_span() {
    use bekiper_pose::steps::likelihood::LikelihoodTrimmer;
    let df = df! {
        "frame" => [0i64, 1, 2, 3, 4],
        "a_likelihood" => [0.1, 0.9, 0.5, 0.9, 0.2],
        "b_likelihood" => [0.9, 0.9, 0.9, 0.9, 0.9],
    }
    .unwrap();
    let step = LikelihoodTrimmer {
        threshold: 0.5,
        likelihood_columns: vec!["a_likelihood".into(), "b_likelihood".into()],
    };
    assert_eq!(step.name(), "likelihood_trimmer");
    let out = step.apply(df.clone().lazy()).collect().unwrap();
    // Exactly-at-threshold frame 2 is reliable, so the span is frames 1..=3.
    let frames: Vec<i64> = out
        .column("frame")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(frames, [1, 2, 3]);
    assert!(out.column("_lik_ok").is_err());

    // No reliable frame keeps every row and never leaks the helper column.
    let strict = LikelihoodTrimmer {
        threshold: 0.95,
        likelihood_columns: vec!["a_likelihood".into()],
    };
    let out = strict.apply(df.clone().lazy()).collect().unwrap();
    assert_eq!(out.height(), 5);
    assert!(out.column("_lik_ok").is_err());

    let none = LikelihoodTrimmer {
        threshold: 0.5,
        likelihood_columns: vec![],
    };
    assert_eq!(none.apply(df.lazy()).collect().unwrap().height(), 5);
}

#[test]
fn temporal_crop_trims_start_end_and_target_length() {
    use bekiper_pose::steps::crop::TemporalCrop;
    let df = df! { "frame" => (0i64..10).collect::<Vec<_>>() }.unwrap();
    let crop = |start: f64, end: f64, target: f64| {
        let step = TemporalCrop {
            crop_start_seconds: start,
            crop_end_seconds: end,
            target_length_seconds: target,
            fps: 2.0,
        };
        assert_eq!(step.name(), "temporal_crop");
        let out = step.apply(df.clone().lazy()).collect().unwrap();
        out.column("frame")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>()
    };
    assert_eq!(crop(1.0, 0.5, 0.0), [2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(crop(1.0, 0.0, 1.5), [2, 3, 4]);
    // A target longer than what remains leaves the crop unchanged.
    assert_eq!(crop(4.0, 0.0, 10.0), [8, 9]);
    // Cropping everything keeps the recording rather than emptying it.
    assert_eq!(crop(3.0, 3.0, 0.0).len(), 10);
    assert_eq!(crop(20.0, 0.0, 0.0).len(), 10);
}

#[test]
fn confinement_nulls_or_drops_frames_outside_the_enclosure() {
    use bekiper_core::types::CoordinateColumns;
    use bekiper_perimeter::perimeter::Perimeter;
    use bekiper_perimeter::rectangle::Rectangle;
    use bekiper_pose::steps::confinement::{ConfinementFilter, ConfinementNuller};
    let df = df! {
        "c_x" => [0.0, 5.0, 0.5], "c_y" => [0.0, 0.0, 0.5],
        "n_x" => [0.1, 5.1, 0.6], "n_y" => [0.1, 0.1, 0.6],
    }
    .unwrap();
    let arena = Perimeter::new("arena", Rectangle::new(0.0, 0.0, 2.0, 2.0));
    let nuller = ConfinementNuller {
        perimeter: arena.clone(),
        reference: CoordinateColumns::new("c_x", "c_y"),
        columns_to_null: vec!["n_x".into(), "n_y".into()],
    };
    assert_eq!(nuller.name(), "confinement_nuller");
    let out = nuller.apply(df.clone().lazy()).collect().unwrap();
    assert_eq!(out.height(), 3);
    assert_eq!(f64s(&out, "n_x"), [Some(0.1), None, Some(0.6)]);

    let filter = ConfinementFilter {
        perimeter: arena,
        coords: CoordinateColumns::new("c_x", "c_y"),
    };
    assert_eq!(filter.name(), "confinement_filter");
    let out = filter.apply(df.lazy()).collect().unwrap();
    assert_eq!(f64s(&out, "c_x"), [Some(0.0), Some(0.5)]);
}

#[test]
fn high_velocity_sigma_gate_nulls_displacement_outliers() {
    // Velocity gate disabled (huge max); the sigma gate alone catches the jump.
    let mut xs = vec![0.0; 20];
    xs[10] = 50.0;
    let df = df! { "x" => xs, "y" => vec![0.0; 20] }.unwrap();
    let step = HighVelocityFilter {
        max_velocity: 1e9,
        x_columns: vec!["x".into()],
        y_columns: vec!["y".into()],
        max_iterations: 3,
        sigma_threshold: 2.0,
    };
    let out = step.apply(df.clone().lazy()).collect().unwrap();
    let x = f64s(&out, "x");
    assert_eq!(x[10], None);
    // The return to baseline is the second large displacement.
    assert_eq!(x[11], None);
    assert_eq!(x[9], Some(0.0));
    assert!(out.column("_disp").is_err());

    let disabled = HighVelocityFilter {
        sigma_threshold: 0.0,
        ..step
    };
    let out = disabled.apply(df.lazy()).collect().unwrap();
    assert_eq!(f64s(&out, "x")[10], Some(50.0));
}

#[test]
fn midpoint_requires_every_source_keypoint() {
    let df = df! {
        "left_ear_x" => [Some(0.0), None], "left_ear_y" => [Some(0.0), Some(0.0)],
        "right_ear_x" => [Some(2.0), Some(2.0)], "right_ear_y" => [Some(2.0), Some(2.0)],
    }
    .unwrap();
    let step = MidpointComputer {
        groups: vec![MidpointGroup {
            output_label: "center_ear".into(),
            source_labels: vec!["left_ear".into(), "right_ear".into()],
        }],
    };
    let out = step.apply(df.lazy()).collect().unwrap();
    assert_eq!(f64s(&out, "center_ear_x"), [Some(1.0), None]);
    assert_eq!(f64s(&out, "center_ear_y")[0], Some(1.0));
}

#[test]
fn median_smoother_interpolates_gaps_and_keeps_edges() {
    let df = df! { "x" => [Some(1.0), None, Some(3.0), Some(4.0), Some(40.0)] }.unwrap();
    let step = MedianSmoother {
        window_size: 3,
        columns: vec!["x".into()],
    };
    let out = step.apply(df.lazy()).collect().unwrap();
    let x = f64s(&out, "x");
    assert_eq!(x.len(), 5);
    assert!(x.iter().all(Option::is_some), "{x:?}");
}

fn repair_step(skeleton: Skeleton, min_baseline_frames: usize) -> SkeletonRepairStep {
    SkeletonRepairStep {
        skeleton,
        relative_tolerance: 0.35,
        mad_multiplier: 4.0,
        min_baseline_frames,
    }
}

fn line_df(c_x: [Option<f64>; 5]) -> DataFrame {
    df! {
        "frame" => [0i64, 1, 2, 3, 4],
        "a_x" => [0.0; 5], "a_y" => [0.0; 5],
        "b_x" => [Some(1.0), None, Some(1.0), Some(1.0), Some(1.0)], "b_y" => [0.0; 5],
        "c_x" => c_x, "c_y" => [0.0; 5],
    }
    .unwrap()
}

#[test]
fn skeleton_repair_fit_rejects_edges_outside_the_node_set() {
    let df = line_df([Some(2.0); 5]);
    for (edge, expected) in [
        (Edge::new("x", "a"), "source 'x'"),
        (Edge::new("a", "y"), "target 'y'"),
    ] {
        let skeleton = Skeleton::new("bad", vec![Node::new("a")], vec![edge]);
        let error = repair_step(skeleton, 1).fit(&df).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
    }
    // No observed pair and no minimum: the median baseline is undefined.
    let empty = df.slice(0, 0);
    let error = repair_step(line_skeleton(), 0)
        .fit(&empty)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cannot compute median baseline"), "{error}");
}

#[test]
fn skeleton_repair_evaluation_api_reports_flagged_rows() {
    // `b` is missing on frame 1, so no edge is evaluated there; `c` jumps on
    // frame 4 and is the only flagged leaf.
    let df = line_df([Some(2.0), Some(2.0), Some(2.0), Some(2.0), Some(9.0)]);
    let mut skeleton = line_skeleton();
    skeleton.nodes.push(Node::new("isolated"));
    let mut df_isolated = df.clone();
    df_isolated
        .with_column(Column::new("isolated_x".into(), [5.0; 5]))
        .unwrap();
    df_isolated
        .with_column(Column::new("isolated_y".into(), [5.0; 5]))
        .unwrap();
    let step = repair_step(skeleton, 1);
    assert_eq!(step.name(), "skeleton_repair");
    let fitted = step.fit(&df_isolated).unwrap();
    assert_eq!(fitted.label_index("c"), Some(2));
    assert_eq!(fitted.label_index("zzz"), None);
    assert_eq!(fitted.label_name(3), "isolated");
    let row = fitted.row_coordinates(&df_isolated, 1).unwrap();
    assert_eq!(row[1], None);
    assert!(fitted.evaluate_frame(&row).is_empty());

    let mut evaluation = fitted.evaluate(&df_isolated).unwrap();
    assert_eq!(evaluation.count(4), 1);
    assert!(evaluation.is_violated(4, "c"));
    assert!(!evaluation.is_violated(4, "isolated"));
    assert!(!evaluation.is_violated(4, "zzz"));
    assert_eq!(
        evaluation.label_flags("c"),
        Some(&[false, false, false, false, true][..])
    );
    assert_eq!(evaluation.label_flags("zzz"), None);
    let flagged = evaluation.flagged_rows(&df_isolated).unwrap();
    assert_eq!(flagged.len(), 1);
    assert_eq!((flagged[0].row_index, flagged[0].frame), (4, 4));
    assert_eq!(flagged[0].labels, ["c"]);

    // A null frame index on a flagged row cannot be reported.
    let mut null_frame = df_isolated.clone();
    null_frame
        .with_column(Column::new(
            "frame".into(),
            [Some(0i64), Some(1), Some(2), Some(3), None],
        ))
        .unwrap();
    assert!(evaluation.flagged_rows(&null_frame).is_err());
    let mut float_frame = df_isolated.clone();
    float_frame
        .with_column(Column::new("frame".into(), [0.0, 1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    assert!(evaluation.flagged_rows(&float_frame).is_err());

    // Clearing an unflagged or unknown label changes nothing.
    evaluation.clear_label(0, "c");
    evaluation.clear_label(4, "zzz");
    assert_eq!(evaluation.count(4), 1);
    evaluation.clear_label(4, "c");
    assert_eq!(evaluation.count(4), 0);
    assert!(evaluation.flagged_rows(&df_isolated).unwrap().is_empty());

    // Redetected rows are recorded alongside the repaired coordinates.
    let redetected = std::collections::BTreeSet::from([1usize]);
    let repaired = fitted
        .apply(df_isolated.clone(), &evaluation, &redetected)
        .unwrap();
    let flags: Vec<bool> = repaired
        .column("skeleton_redetected")
        .unwrap()
        .bool()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(flags, [false, true, false, false, false]);

    // Applying an evaluation to a frame of a different height fails.
    assert!(
        fitted
            .apply(df_isolated.slice(0, 2), &evaluation, &redetected)
            .is_err()
    );
}

#[test]
fn skeleton_repair_without_nodes_cannot_attach_row_counts() {
    let df = line_df([Some(2.0); 5]);
    let fitted = repair_step(Skeleton::new("empty", vec![], vec![]), 1)
        .fit(&df)
        .unwrap();
    let evaluation = fitted.evaluate(&df).unwrap();
    assert!(fitted.apply(df, &evaluation, &Default::default()).is_err());
}

#[test]
fn skeleton_kinematic_smoother_reports_its_name() {
    let step = SkeletonKinematicSmoother {
        skeleton: line_skeleton(),
        columns: vec![],
        process_noise: 1.0,
        measurement_noise: 1.0,
        min_likelihood: 0.5,
        max_gap_frames: 1,
        constraint_iterations: 1,
        constraint_strength: 1.0,
    };
    assert_eq!(step.name(), "skeleton_kinematic_smoother");
}

#[test]
fn skeleton_repair_skips_edges_without_enough_baseline_frames() {
    let df = line_df([Some(2.0), Some(2.0), Some(2.0), Some(2.0), Some(9.0)]);
    let fitted = repair_step(line_skeleton(), 10).fit(&df).unwrap();
    let evaluation = fitted.evaluate(&df).unwrap();
    assert!((0..5).all(|row| evaluation.count(row) == 0));
}

/// Counts emitted events; enabling every callsite evaluates its fields.
#[derive(Default)]
struct CountingSubscriber(std::sync::atomic::AtomicUsize);

impl tracing::Subscriber for CountingSubscriber {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

#[test]
fn timestamp_joiner_warns_once_when_skipping_an_unreadable_file() {
    use bekiper_pose::steps::timestamp::TimestampJoiner;
    let subscriber = std::sync::Arc::new(CountingSubscriber::default());
    let joiner = TimestampJoiner {
        timestamps_file: "/nonexistent/bekiper/timestamps.csv".into(),
    };
    let df = df! { "frame" => [0i64, 1] }.unwrap();
    let out = tracing::subscriber::with_default(subscriber.clone(), || {
        joiner.apply(df.lazy()).collect().unwrap()
    });
    assert!(out.column("elapsed_s").is_err());
    assert_eq!(subscriber.0.load(std::sync::atomic::Ordering::Relaxed), 1);
}
