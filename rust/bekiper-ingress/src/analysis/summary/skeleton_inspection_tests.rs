use super::*;

#[cfg(feature = "pose-video")]
#[test]
fn representative_frames_follow_source_ordinals_with_irregular_timestamps() {
    use bekiper_core::inspection::FeatureInspection;
    use polars::prelude::*;
    use std::process::{Command, Stdio};
    let root = std::env::temp_dir().join(format!(
        "bekiper-representative-pts-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let video = root.join("irregular.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=3",
            "-vf",
            "setpts=if(lt(N\\,15)\\,N\\,15+2*(N-15))/(10*TB)",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-bf",
            "2",
        ])
        .arg(&video)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());
    let dir = ExperimentOutputDir::new(&root, "trial");
    let features = [0, 1].map(|row| FeatureInspection::Trajectory {
        label: format!("source_{row}"),
        coordinate_labels: vec!["nose".into()],
        description: "Source ordinal".into(),
        representative_frame_index: Some(row),
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    });
    let evaluated = df!("frame" => [7u64, 29]).unwrap();
    extract_representative_frames(&video, &features, Some(&evaluated), &dir).unwrap();
    for index in [7, 29] {
        let expected = root.join(format!("expected_{index}.png"));
        let status = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&video)
            .args(["-vf", &format!("select=eq(n\\,{index})"), "-frames:v", "1"])
            .arg(&expected)
            .status()
            .unwrap();
        assert!(status.success());
        let actual = image::open(dir.frames_dir().join(format!("frame_{index:06}.png")))
            .unwrap()
            .to_rgb8();
        assert!(
            actual == image::open(expected).unwrap().to_rgb8(),
            "source frame {index}"
        );
    }
    let unavailable = df!("frame" => [30u64, 31]).unwrap();
    assert!(extract_representative_frames(&video, &features, Some(&unavailable), &dir).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_nose_speed_joins_pose_quality_without_duplicate_object_traces() {
    use polars::prelude::*;
    let experiment: ExperimentConfig = toml::from_str(
            "name = 'scalar_fixture'\ntask_type = 'object_recognition'\nfps = 10.0\nmeters_per_pixel = 0.001",
        ).unwrap();
    let summaries = ["exploring_B", "exploring_A"].map(|name| HeuristicSummary {
        name: name.into(),
        true_frames: 1,
        total_frames: 2,
        seconds: 0.1,
    });
    let df = df!(
        "nose_x" => [0.1, 0.2], "nose_y" => [0.2, 0.2],
        "__bk_diag__exploring_A__nose_speed_mps" => [None, Some(0.3)],
        "__bk_diag__exploring_B__nose_speed_mps" => [None, Some(0.3)],
    )
    .unwrap();
    let manifest = build_inspection_manifest(
        &experiment,
        &["nose".into()],
        &summaries,
        &[],
        &[],
        &[],
        2,
        0.001,
        None,
        None,
        Some(&df),
    );
    let traces = manifest
        .feature_inspections
        .iter()
        .filter(|feature| {
            matches!(
                feature,
                bekiper_core::inspection::FeatureInspection::TimeSeries { .. }
            ) && feature.story_key() == Some("pose_quality")
        })
        .collect::<Vec<_>>();
    assert_eq!(traces.len(), 1);
    let bekiper_core::inspection::FeatureInspection::TimeSeries {
        value_column, unit, ..
    } = traces[0]
    else {
        panic!("expected scalar trace");
    };
    assert_eq!(value_column, "__bk_diag__exploring_A__nose_speed_mps");
    assert_eq!(unit, "m/s");
    assert_eq!(df.column(value_column).unwrap().f64().unwrap().get(0), None);
}

#[test]
fn retained_tracking_columns_are_metric_and_reject_frame_misalignment() {
    use polars::prelude::*;
    let mut manifest: InspectionManifest = toml::from_str(include_str!(
        "../../../../../python/bekiper-inspect/tests/fixtures/golden_inspection.toml"
    ))
    .unwrap();
    let mut df = df!("frame" => [0_u64, 1], "nose_x" => [0.1, 0.2]).unwrap();
    let source =
        df!("frame" => [0_u64, 1], "nose_x" => [100.0, 200.0], "nose_y" => [50.0, 60.0]).unwrap();
    retain_tracking_stage(
        &mut df,
        &source,
        &["nose".into()],
        "raw_pose",
        (0.001, 0.001),
        &mut manifest,
    )
    .unwrap();
    assert_eq!(
        df.column("__bk_stage__raw_pose__nose_x")
            .unwrap()
            .f64()
            .unwrap()
            .get(1),
        Some(0.2)
    );
    let wrong = source
        .lazy()
        .with_column(lit(9_u64).alias("frame"))
        .collect()
        .unwrap();
    assert!(
        retain_tracking_stage(
            &mut df,
            &wrong,
            &["nose".into()],
            "raw_pose",
            (0.001, 0.001),
            &mut manifest
        )
        .is_err()
    );
}

#[test]
fn tracking_aligns_cropped_source_keys_and_omits_unavailable_midpoints() {
    use polars::prelude::*;
    let mut manifest: InspectionManifest = toml::from_str(include_str!(
        "../../../../../python/bekiper-inspect/tests/fixtures/golden_inspection.toml"
    ))
    .unwrap();
    manifest.coordinate_columns.clear();
    manifest.feature_inspections.clear();
    let mut evaluated = df!("frame" => [2_u64, 4], "elapsed_s" => [0.2, 0.4]).unwrap();
    let raw = df!("frame" => [0_u64, 1, 2, 3, 4],
            "nose_x" => [0., 10., 20., 30., 40.],
            "nose_y" => [Some(0.), Some(10.), None, Some(30.), Some(40.)])
    .unwrap();
    let labels = vec!["nose".into(), "center_ear".into()];
    retain_tracking_stage(
        &mut evaluated,
        &raw,
        &labels,
        "raw_pose",
        (0.01, 0.01),
        &mut manifest,
    )
    .unwrap();
    assert_eq!(
        evaluated
            .column("__bk_stage__raw_pose__nose_x")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        [Some(0.2), Some(0.4)]
    );
    assert_eq!(
        evaluated
            .column("__bk_stage__raw_pose__nose_y")
            .unwrap()
            .null_count(),
        1
    );
    emit_tracking_inspections(&labels, &mut manifest);
    assert_eq!(manifest.feature_inspections.len(), 3);
    assert!(
        !manifest
            .feature_inspections
            .iter()
            .any(|f| f.label().contains("center_ear"))
    );
    let missing = evaluated
        .clone()
        .lazy()
        .with_column(lit(7_u64).alias("frame"))
        .collect()
        .unwrap();
    assert!(
        retain_tracking_stage(
            &mut missing.clone(),
            &raw,
            &labels,
            "raw_pose",
            (0.01, 0.01),
            &mut manifest
        )
        .is_err()
    );
    assert!(
        retain_tracking_stage(
            &mut evaluated,
            &raw.drop("frame").unwrap(),
            &labels,
            "raw_pose",
            (0.01, 0.01),
            &mut manifest
        )
        .is_err()
    );
}

#[test]
fn skeleton_feature_uses_base_tail_for_imen_ymaze_labels() {
    let labels = ["nose", "left_ear", "right_ear", "center_ear", "base_tail"]
        .into_iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    let feature = skeleton_feature_from_labels(&labels).unwrap();
    match feature {
        bekiper_core::inspection::FeatureInspection::Trajectory {
            label,
            coordinate_labels,
            title_override,
            ..
        } => {
            assert_eq!(label, "skeleton_pose");
            assert_eq!(
                coordinate_labels,
                vec!["nose", "left_ear", "center_ear", "right_ear", "base_tail"]
            );
            assert_eq!(title_override.as_deref(), Some("Skeleton pose zoom"));
        }
        other => panic!("expected trajectory-backed skeleton inspection, got {other:?}"),
    }
}

#[test]
fn skeleton_feature_uses_tail_for_imen_oft_nort_labels() {
    let labels = ["nose", "left_ear", "right_ear", "center_ear", "tail"]
        .into_iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    let feature = skeleton_feature_from_labels(&labels).unwrap();
    match feature {
        bekiper_core::inspection::FeatureInspection::Trajectory {
            coordinate_labels, ..
        } => assert!(coordinate_labels.contains(&"tail".to_string())),
        other => panic!("expected trajectory-backed skeleton inspection, got {other:?}"),
    }
}

#[cfg(feature = "pose-video")]
#[test]
fn representative_frames_need_rows_inside_the_evaluated_table() {
    use bekiper_core::inspection::FeatureInspection;
    let dir = ExperimentOutputDir::new(
        &std::env::temp_dir().join(format!(
            "bekiper-representative-rows-{}",
            std::process::id()
        )),
        "trial",
    );
    let feature = FeatureInspection::Trajectory {
        label: "late".into(),
        coordinate_labels: vec!["nose".into()],
        description: String::new(),
        representative_frame_index: Some(5),
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    };
    let evaluated = polars::df! {"frame" => [0u64, 1]}.unwrap();
    let error = extract_representative_frames(
        Path::new("/nonexistent.mp4"),
        std::slice::from_ref(&feature),
        Some(&evaluated),
        &dir,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("Representative row 5 is outside"), "{error}");
    extract_representative_frames(Path::new("/nonexistent.mp4"), &[], Some(&evaluated), &dir)
        .unwrap();
}

#[test]
fn tracking_rejects_untyped_missing_or_misclocked_keys() {
    use polars::prelude::*;
    let mut manifest: InspectionManifest = toml::from_str(include_str!(
        "../../../../../python/bekiper-inspect/tests/fixtures/golden_inspection.toml"
    ))
    .unwrap();
    let labels = ["nose".to_string()];
    let mut retain = |df: DataFrame, source: DataFrame| {
        let mut df = df;
        retain_tracking_stage(
            &mut df,
            &source,
            &labels,
            "raw_pose",
            (1.0, 1.0),
            &mut manifest,
        )
        .map(|()| df)
        .map_err(|error| error.to_string())
    };
    let source =
        df!("frame" => [0_u64, 1], "nose_x" => [1.0, 2.0], "nose_y" => [3.0, 4.0]).unwrap();
    for (df, message) in [
        (df!("frame" => [0.0, 1.0]).unwrap(), "must be integers"),
        (
            df!("frame" => [Some(0_u64), None]).unwrap(),
            "nonnegative and non-null",
        ),
        (
            df!("frame" => [0_u64, 5]).unwrap(),
            "missing source frame 5",
        ),
        (
            df!("frame" => [0_u64, 1], "elapsed_s" => [0.0, 9.0]).unwrap(),
            "elapsed_s does not align",
        ),
    ] {
        let source = if message.contains("elapsed") {
            source
                .clone()
                .lazy()
                .with_column(lit(0.5).alias("elapsed_s"))
                .collect()
                .unwrap()
        } else {
            source.clone()
        };
        let error = retain(df, source).unwrap_err();
        assert!(error.contains(message), "{message}: {error}");
    }
    // Unkeyed tables of equal height align by position.
    let unkeyed = retain(
        df!("nose_x" => [0.0, 0.0]).unwrap(),
        source.drop("frame").unwrap(),
    )
    .unwrap();
    assert!(unkeyed.column("__bk_stage__raw_pose__nose_x").is_ok());
}

#[test]
fn state_partitions_refuse_invalid_clocks_and_classify_overlaps_and_gaps() {
    use bekiper_core::inspection::FeatureInspection;
    use polars::prelude::*;
    let spec = |name: &str, column: &str| -> HeuristicDecompositionSpec {
        toml::from_str(&format!(
            "heuristic_name = '{name}'\nresult_column = '{column}'"
        ))
        .unwrap()
    };
    let objects = [1, 2].map(|i| PerimeterSpec::Circle {
        label: format!("object_{i}"),
        center_x: 0.0,
        center_y: 0.0,
        radius: 1.0,
    });
    let specs = [
        spec("body_proximity_object_1", "bp1"),
        spec("body_proximity_object_2", "bp2"),
        spec("olfaction_object_1", "olf1"),
    ];
    let states = |rows: usize| {
        df!(
            "bp1" => [Some(true), None, Some(true)][..rows].to_vec(),
            "bp2" => [Some(true), Some(false), Some(false)][..rows].to_vec()
        )
        .unwrap()
    };
    let invalid = states(1)
        .lazy()
        .with_column(lit(0.0).alias("sample_duration_s"))
        .collect()
        .unwrap();
    assert!(inspection_states::state_inspections(&invalid, &specs, &objects, 30.0).is_empty());
    assert!(inspection_states::state_inspections(&states(3), &specs, &objects, 0.0).is_empty());
    let features = inspection_states::state_inspections(&states(3), &specs, &objects, 2.0);
    let Some(FeatureInspection::StateDistribution { frame_counts, .. }) = features.first() else {
        panic!("expected a session composition: {features:?}");
    };
    // object_1, object_2, overlapping, unassigned, missing
    assert_eq!(frame_counts, &[1, 0, 1, 0, 1]);
    let empty = df!("bp1" => Vec::<bool>::new(), "bp2" => Vec::<bool>::new(), "elapsed_s" => Vec::<f64>::new()).unwrap();
    let features = inspection_states::state_inspections(&empty, &specs, &objects, 2.0);
    let Some(FeatureInspection::StateDistribution { percentages, .. }) = features.first() else {
        panic!("expected an empty composition");
    };
    assert!(percentages.iter().all(|p| *p == 0.0));
}

#[test]
fn vector_examples_need_every_ray_column_and_sample_single_candidates() {
    use polars::prelude::*;
    let spec: HeuristicDecompositionSpec = toml::from_str(
        r#"
        heuristic_name = "observation_object_1"
        result_column = "obs1"
        [[components]]
        label = "angle"
        result_column = "angle_ok"
        ray_origin_label = "nose"
        ray_direction_x_column = "dx"
        ray_direction_y_column = "dy"
        ray_target_x_column = "tx"
        ray_target_y_column = "ty"
        value_column = "angle"
        value_unit = "degrees"
        "#,
    )
    .unwrap();
    let partial = df!("dx" => [1.0]).unwrap();
    assert!(inspection_states::vector_examples(&partial, std::slice::from_ref(&spec)).is_empty());
    let single = df!(
        "dx" => [1.0, 0.0], "dy" => [0.0, 0.0], "tx" => [1.0, 1.0], "ty" => [1.0, 1.0],
        "angle" => [10.0, 20.0]
    )
    .unwrap();
    let examples = inspection_states::vector_examples(&single, &[spec]);
    assert_eq!(examples.len(), 1);
    assert_eq!(examples[0].representative_frame_index(), Some(0));
}
