use bekiper_behaviour::BehaviouralTask;
use bekiper_behaviour::open_field::OpenFieldTest;
use bekiper_core::types::CoordinateColumns;
use polars::prelude::*;

#[test]
fn standard_creates_arena() {
    let oft = OpenFieldTest::standard((0.0, 0.0), 0.40, 0.08, 30.0);
    assert!((oft.arena.shape.width - 0.40).abs() < 1e-10);
    assert!((oft.arena.shape.height - 0.40).abs() < 1e-10);
}

#[test]
fn standard_creates_center_zone() {
    let oft = OpenFieldTest::standard((0.0, 0.0), 0.40, 0.08, 30.0);
    let expected_side = 0.40 - 2.0 * 0.08; // 0.24
    assert!((oft.center.shape.width - expected_side).abs() < 1e-10);
    assert!((oft.center.shape.height - expected_side).abs() < 1e-10);
}

#[test]
fn name_is_open_field_test() {
    let oft = OpenFieldTest::standard((0.0, 0.0), 0.40, 0.08, 30.0);
    assert_eq!(oft.task_type(), bekiper_core::types::TaskType::OpenField);
}

#[test]
fn heuristics_includes_periphery_rearing_grooming() {
    let oft = OpenFieldTest::standard((0.0, 0.0), 0.40, 0.08, 30.0);
    let heuristics = oft.heuristics();
    // center, periphery, freezing, rearing, grooming = 5
    assert_eq!(heuristics.len(), 5);
    let names: Vec<&str> = heuristics.iter().map(|h| h.name()).collect();
    assert!(names.contains(&"in_center"));
    assert!(names.contains(&"in_periphery"));
    assert!(names.contains(&"freezing"));
    assert!(names.contains(&"rearing"));
    assert!(names.contains(&"grooming"));
}

#[test]
fn open_field_accepts_imen_coordinate_overrides_without_superanimal_columns() {
    let mut oft = OpenFieldTest::standard((0.0, 0.0), 0.40, 0.08, 30.0);
    oft.params.centroid_coords = CoordinateColumns::new("center_ear_x", "center_ear_y");
    oft.params.head_coords = CoordinateColumns::new("center_ear_x", "center_ear_y");
    oft.params.tail_coords = CoordinateColumns::new("tail_x", "tail_y");

    let n = 60;
    let x = vec![0.0; n];
    let y = vec![0.0; n];
    let nose_x = vec![0.02; n];
    let tail_x = vec![-0.04; n];
    let df = df! {
        "center_ear_x" => &x,
        "center_ear_y" => &y,
        "nose_x" => &nose_x,
        "nose_y" => &y,
        "tail_x" => &tail_x,
        "tail_y" => &y,
    }
    .unwrap();

    let result = oft.full_analyze(df.lazy());

    assert!(result.evaluated_df.is_some());
}

fn recipe_task(fps: f64) -> OpenFieldTest {
    let mut task = OpenFieldTest::standard((0.0, 0.0), 0.40, 0.08, fps);
    task.params.centroid_coords = CoordinateColumns::new("x", "y");
    task.params.head_coords = CoordinateColumns::new("x", "y");
    task.params.tail_coords = CoordinateColumns::new("tail_x", "tail_y");
    task
}

fn recipe_poses(xs: &[Option<f64>]) -> DataFrame {
    let ys: Vec<_> = xs.iter().map(|x| x.map(|_| 0.0)).collect();
    df! {
        "x" => xs, "y" => &ys,
        "nose_x" => &vec![0.02; xs.len()], "nose_y" => &vec![0.0; xs.len()],
        "tail_x" => &vec![-0.04; xs.len()], "tail_y" => &vec![0.0; xs.len()],
        "frame" => &(0..xs.len() as u64).collect::<Vec<_>>(),
    }
    .unwrap()
}

#[test]
fn oft_recipe_selects_300_seconds_from_first_valid_centroid() {
    let mut xs = vec![None; 10];
    xs.extend(vec![Some(0.16); 300]);
    // This late center visit must not contaminate the required window.
    xs.extend(vec![Some(0.0); 5]);
    let result = recipe_task(1.0).full_analyze(recipe_poses(&xs).lazy());
    assert_eq!(result.total_duration_seconds, 300.0);
    assert_eq!(result.center_summary.entries, 0);
    assert_eq!(result.center_summary.duration_seconds, 0.0);
    assert_eq!(result.periphery_time_pct, 100.0);
    let evaluated = result.evaluated_df.unwrap();
    assert_eq!(evaluated.height(), 300);
    assert_eq!(
        evaluated.column("frame").unwrap().u64().unwrap().get(0),
        Some(10)
    );
}

#[test]
fn oft_recipe_distance_includes_short_zone_visits_and_boundary_crossings() {
    let result = recipe_task(10.0)
        .full_analyze(recipe_poses(&[Some(0.0), Some(0.16), Some(0.0), Some(0.16)]).lazy());
    assert!((result.total_distance_meters - 0.48).abs() < 1e-12);
    assert!((result.total_mean_speed - 1.2).abs() < 1e-12);
    assert_eq!(result.center_summary.entries, 0);
}

#[test]
fn oft_recipe_distance_does_not_invent_motion_across_missing_pose() {
    let result = recipe_task(10.0)
        .full_analyze(recipe_poses(&[Some(0.0), None, Some(0.16), Some(0.18)]).lazy());
    assert!((result.total_distance_meters - 0.02).abs() < 1e-12);
}

#[test]
fn oft_recipe_dwell_uses_measured_seconds_without_fractional_fps_truncation() {
    for (frames, expected_entries) in [(29, 0), (30, 1)] {
        let result = recipe_task(29.97).full_analyze(recipe_poses(&vec![Some(0.0); frames]).lazy());
        assert_eq!(result.center_summary.entries, expected_entries);
        // Percent of the analysed seconds: the whole short recording is centre.
        let expected_pct = if frames == 30 { 100.0 } else { 0.0 };
        assert!((result.center_time_pct - expected_pct).abs() < 1e-12);
    }
}

#[test]
fn oft_recipe_vfr_dwell_accepts_exactly_one_second() {
    for (second_end, expected_entries) in [(5.99, 0), (6.0, 1)] {
        let timing = bekiper_core::video::FrameTiming {
            fps: 10.0,
            timestamps: vec![5.0, 5.4, second_end, 6.4],
            is_vfr: true,
        };
        let task = recipe_task(10.0);
        let (_, detail) = task.analyze_detailed(
            recipe_poses(&[Some(0.0), Some(0.0), Some(0.16), Some(0.16)]).lazy(),
            &timing,
        );
        let bekiper_behaviour::TaskDetail::OpenField(result) = detail.unwrap() else {
            panic!("OFT detail")
        };
        assert_eq!(result.center_summary.entries, expected_entries);
        if expected_entries == 1 {
            assert_eq!(result.center_summary.duration_seconds, 1.0);
            assert_eq!(result.center_summary.latency_seconds, Some(0.0));
        }
    }
}

#[test]
fn oft_center_and_periphery_entries_reject_short_visits_without_bridging_transitions() {
    // The center ends 12 cm from the floor center; the peripheral band extends
    // from that boundary to the wall at 20 cm (8 cm wide).
    let mut xs = vec![Some(0.121); 10];
    xs.extend(vec![Some(0.119); 9]);
    xs.extend(vec![Some(0.121); 10]);
    xs.extend(vec![Some(0.119); 10]);
    xs.extend(vec![Some(0.121); 9]);
    let result = recipe_task(10.0).full_analyze(recipe_poses(&xs).lazy());
    assert_eq!(result.center_summary.entries, 1);
    assert_eq!(result.periphery_summary.entries, 2);
    assert!((result.center_summary.duration_seconds - 1.0).abs() < 1e-12);
    assert!((result.periphery_summary.duration_seconds - 2.0).abs() < 1e-12);
    assert!((result.center_summary.latency_seconds.unwrap() - 2.9).abs() < 1e-12);
    let df = result.evaluated_df.unwrap();
    assert!(
        df.column("in_center_raw")
            .unwrap()
            .bool()
            .unwrap()
            .get(10)
            .unwrap()
    );
    assert!(
        !df.column("in_center")
            .unwrap()
            .bool()
            .unwrap()
            .get(10)
            .unwrap()
    );
}

#[test]
fn oft_window_clips_last_sample_support_and_retains_incomplete_recordings() {
    let mut task = recipe_task(2.0);
    task.params.oft_window_seconds = 1.25;
    task.params.min_dwell_seconds = 1.0;
    let result = task.full_analyze(recipe_poses(&[Some(0.0); 5]).lazy());
    assert_eq!(result.total_duration_seconds, 1.25);
    assert_eq!(result.center_summary.duration_seconds, 1.25);
    assert_eq!(result.center_time_pct, 100.0);
    assert!(result.window.complete);
    let df = result.evaluated_df.unwrap();
    assert_eq!(
        df.column("sample_duration_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(2),
        Some(0.25)
    );
    let short = recipe_task(2.0).full_analyze(recipe_poses(&[Some(0.0); 2]).lazy());
    assert!(!short.window.complete);
    assert_eq!(short.total_duration_seconds, 1.0);
    assert_eq!(short.center_summary.entries, 1);
    let absent = recipe_task(2.0).full_analyze(recipe_poses(&[None; 2]).lazy());
    assert!(!absent.window.complete);
    assert_eq!(absent.total_duration_seconds, 0.0);
    assert_eq!(absent.center_summary.entries, 0);
}

#[test]
fn oft_window_keeps_sparse_source_keys_and_legacy_row_keys_after_trim() {
    let mut task = recipe_task(2.0);
    task.params.oft_window_seconds = 1.0;
    let mut df = recipe_poses(&[None, Some(0.0), Some(0.0), Some(0.16)]);
    df.with_column(Series::new("frame".into(), [70u64, 71, 73, 74]).into_column())
        .unwrap();
    let result = task.full_analyze(df.clone().lazy());
    assert_eq!(result.window.source_start_frame, 71);
    assert_eq!(result.window.source_end_frame, 74);
    assert_eq!(result.center_motion_islands[0].start_frame, 71);
    assert_eq!(result.center_motion_islands[0].end_frame, 74);
    let result = task.full_analyze(df.drop("frame").unwrap().lazy());
    assert_eq!(result.window.source_start_frame, 1);
    assert_eq!(result.window.source_end_frame, 3);
    assert_eq!(
        result
            .evaluated_df
            .unwrap()
            .column("frame")
            .unwrap()
            .u64()
            .unwrap()
            .get(0),
        Some(1)
    );
}

#[test]
fn oft_freezing_reports_immobility_even_in_a_grooming_like_posture() {
    // Stationary for 5 s with a compact 6 cm body, which also matches the
    // grooming/rearing proxies; freezing must not drop it (2026-10-21 decision).
    let compact = recipe_task(10.0).full_analyze(recipe_poses(&vec![Some(0.16); 50]).lazy());
    assert!(compact.immobility_seconds > 0.0);
    assert_eq!(compact.freezing_seconds, compact.immobility_seconds);
    assert!(
        (compact.immobility_pct
            - 100.0 * compact.immobility_seconds / compact.total_duration_seconds)
            .abs()
            < 1e-12
    );
}

#[test]
fn oft_quadrant_grid_spans_the_arena() {
    let quadrants = recipe_task(10.0).quadrants();
    assert_eq!(quadrants.len(), 1);
    let grid = &quadrants[0];
    assert_eq!(grid.label, "open_field_grid");
    assert_eq!(
        (grid.x_min, grid.x_max, grid.y_min, grid.y_max),
        (-0.2, 0.2, -0.2, 0.2)
    );
}

#[test]
fn oft_full_video_window_uses_the_recording_as_denominator() {
    let mut task = recipe_task(1.0);
    task.params.oft_window_seconds = 0.0;
    // Leading nulls are kept: a full-video window starts at frame 0.
    let mut xs = vec![None, Some(0.16)];
    xs.extend(vec![Some(0.0); 2]);
    let result = task.full_analyze(recipe_poses(&xs).lazy());
    assert_eq!(result.window.policy, "full_video_half_open_v1");
    assert!(result.window.complete);
    assert_eq!(result.window.source_start_frame, 0);
    assert_eq!(result.total_duration_seconds, 4.0);
    assert_eq!(result.center_summary.entries, 1);
    assert_eq!(result.center_time_pct, 50.0);
}

#[test]
fn oft_empty_recording_reports_zero_percentages() {
    for window in [0.0, 300.0] {
        let mut task = recipe_task(10.0);
        task.params.oft_window_seconds = window;
        let result = task.full_analyze(recipe_poses(&[]).lazy());
        assert_eq!(result.total_duration_seconds, 0.0);
        assert_eq!(result.total_mean_speed, 0.0);
        assert_eq!(result.center_time_pct, 0.0);
        assert_eq!(result.periphery_time_pct, 0.0);
        assert_eq!(result.immobility_pct, 0.0);
        assert_eq!(result.window.source_end_frame, 0);
        // A requested window the recording cannot fill is incomplete.
        assert_eq!(result.window.complete, window == 0.0);
    }
}

#[test]
fn task_params_without_oft_window_default_to_300_seconds() {
    let params: bekiper_behaviour::TaskParams = toml::from_str(
        r#"
        fps = 30.0
        freezing_window_seconds = 1.0
        freezing_displacement_threshold = 0.01
        min_dwell_seconds = 1.0
        peripheral_band_fraction = 0.2
        centroid_coords = { x = "x", y = "y" }
        nose_coords = { x = "nx", y = "ny" }
        head_coords = { x = "hx", y = "hy" }
        tail_coords = { x = "tx", y = "ty" }
        "#,
    )
    .unwrap();
    assert_eq!(
        params.oft_window_seconds,
        bekiper_behaviour::params::DEFAULT_OFT_WINDOW_SECONDS
    );
}

#[test]
fn oft_detailed_analysis_returns_a_round_tripping_detail() {
    use bekiper_behaviour::TaskDetail;
    use bekiper_core::types::TaskType;
    let (_, detail) = recipe_task(10.0).analyze_detailed(
        recipe_poses(&[Some(0.0); 12]).lazy(),
        &bekiper_core::video::FrameTiming::cfr(10.0),
    );
    let detail = detail.unwrap();
    assert_eq!(detail.task_type(), TaskType::OpenField);
    let restored =
        TaskDetail::from_json_value(TaskType::OpenField, detail.as_json_value()).unwrap();
    let TaskDetail::OpenField(result) = restored else {
        unreachable!()
    };
    assert_eq!(result.center_summary.entries, 1);
}

#[test]
fn oft_zone_distance_and_mean_speed_cover_only_that_zones_frames() {
    // 10 fps: 1.1 s walking 10 cm inside the centre, a 5 cm step across the
    // boundary, then 1.1 s walking 4 cm in the periphery. A step belongs to
    // the zone it ends in, so the crossing counts toward the periphery.
    let mut xs: Vec<_> = (0..11).map(|i| Some(i as f64 * 0.01)).collect();
    xs.extend((0..11).map(|i| Some(0.15 + i as f64 * 0.004)));
    let result = recipe_task(10.0).full_analyze(recipe_poses(&xs).lazy());
    let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
    let (center, periphery) = (&result.center_summary, &result.periphery_summary);
    assert!(close(center.duration_seconds, 1.1) && close(periphery.duration_seconds, 1.1));
    assert!(
        close(center.distance_meters, 0.10),
        "{}",
        center.distance_meters
    );
    assert!(
        close(periphery.distance_meters, 0.09),
        "{}",
        periphery.distance_meters
    );
    assert!(close(center.mean_speed, 0.10 / 1.1));
    assert!(close(periphery.mean_speed, 0.09 / 1.1));
    assert!(close(result.total_distance_meters, 0.19));
    assert!(close(
        result.center_motion_summary.total_displacement_meters,
        0.10
    ));
}

#[test]
fn oft_grooming_and_rearing_seconds_count_proxy_bouts_at_task_level() {
    // Motionless for 4 s: a 6 cm body for 2 s (grooming range), then a 3 cm
    // body (≤5 cm and ≤70 % of the 4.5 cm median: rearing).
    let n = 40;
    let half = |a: f64, b: f64| {
        (0..n)
            .map(|i| if i < n / 2 { a } else { b })
            .collect::<Vec<_>>()
    };
    let poses = df! {
        "x" => vec![0.16; n], "y" => vec![0.0; n],
        "nose_x" => half(0.18, 0.17), "nose_y" => vec![0.0; n],
        "tail_x" => half(0.12, 0.14), "tail_y" => vec![0.0; n],
        "frame" => (0..n as u64).collect::<Vec<_>>(),
    }
    .unwrap();
    let result = recipe_task(10.0).full_analyze(poses.lazy());
    // Frame 0 has no speed; grooming frames 1-19, rearing frames 20-39.
    assert!(
        (result.grooming_seconds - 1.9).abs() < 1e-9,
        "{}",
        result.grooming_seconds
    );
    assert!(
        (result.rearing_seconds - 2.0).abs() < 1e-9,
        "{}",
        result.rearing_seconds
    );
}
