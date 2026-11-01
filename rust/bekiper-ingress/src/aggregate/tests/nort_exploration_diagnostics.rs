#[test]
fn nort_novelty_exports_arena_zone_metrics() {
    for protocol in ["nort_training", "nort_novelty"] {
        let mut trial = nort_trial(
            protocol,
            12,
            15.0,
            20.0,
            false,
            &["familiar", "novel"],
            &[5.0, 10.0],
            &[2, 3],
        );
        let summary = bekiper_feature::zone::ZoneSummary {
            zone_name: "in_center".into(),
            entries: 3,
            duration_seconds: 12.0,
            distance_meters: 1.5,
            mean_speed: 0.125,
            latency_seconds: Some(2.0),
        };
        trial.detailed["zone_metrics"] = serde_json::json!({
            "center_summary": summary, "periphery_summary": summary,
            "quadrant_summaries": [{"zone_name":"q_0_0", "entries":1,"duration_seconds":4.0,
                "distance_meters":0.5,"mean_speed":0.125,"latency_seconds":0.0}],
            "quadrant_motion_summaries": []
        });
        let experiment = super::object_recognition::NortExperiment::from_trial(&trial);
        let row = experiment.extract_row(&trial);
        assert_eq!(row.get("center_duration_s"), Some(&CellValue::Float(12.0)));
        assert_eq!(row.get("center_entries"), Some(&CellValue::Int(3)));
        assert_eq!(row.get("center_distance_m"), Some(&CellValue::Float(1.5)));
        assert_eq!(row.get("center_mean_speed"), Some(&CellValue::Float(0.125)));
        assert_eq!(
            row.get("periphery_duration_s"),
            Some(&CellValue::Float(12.0))
        );
        assert_eq!(
            row.get("quadrant_1_1_duration_s"),
            Some(&CellValue::Float(4.0))
        );
    }
}

#[test]
fn nort_exploration_diagnostics_sheet_has_two_rows_per_trial() {
    let trial = parse(
        r#"
        experiment = "nort_videos_nort_post_t2_config_b_post_t2_configb_video_40"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"

        [animal_metadata]
        animal = 12

        [session_context]
        animal = "12"
        trial = "90"

        [detailed]
        protocol = "nort_novelty"
        excluded = true
        total_duration_seconds = 120.0
        total_distance_meters = 4.0
        total_mean_speed = 0.05
        freezing_seconds = 1.0
        analysis_window_seconds = 120.0
        object_labels = ["familiar", "novel"]
        exploration_times = [0.0, 0.0]
        bout_counts = [0, 0]

        [[detailed.exploration_attributions]]
        frames_in_window = 3600
        frames_pose_ok = 3500
        frames_outside_object = 3000
        frames_nose_near = 200
        frames_nose_contact = 50
        frames_within_angle = 100
        frames_contact_or_angle = 120
        frames_slow_nose = 80
        frames_all_gates_raw = 20
        frames_after_tolerance = 0
        bouts_pre_filter = 4
        bouts_post_filter = 0
        bout_lengths_ms_pre = [100, 200]
        bout_lengths_ms_post = []
        candidate_bouts_pre_merge = 4
        bouts_post_merge = 2
        bouts_removed_by_duration_filter = 1
        bouts_removed_by_speed_filter = 1
        final_valid_bouts = 0
        bouts_removed_by_angle_filter = 0
        candidate_frames_removed_by_angle_criterion = 3
        median_nose_speed_per_bout = []
        median_head_angle_per_bout = []

        [[detailed.exploration_attributions]]
        frames_in_window = 3600
        frames_pose_ok = 3500
        frames_outside_object = 3100
        frames_nose_near = 150
        frames_nose_contact = 30
        frames_within_angle = 90
        frames_contact_or_angle = 100
        frames_slow_nose = 70
        frames_all_gates_raw = 10
        frames_after_tolerance = 0
        bouts_pre_filter = 2
        bouts_post_filter = 0
        bout_lengths_ms_pre = [300]
        bout_lengths_ms_post = []
        candidate_bouts_pre_merge = 2
        bouts_post_merge = 1
        bouts_removed_by_duration_filter = 1
        bouts_removed_by_speed_filter = 0
        final_valid_bouts = 0
        bouts_removed_by_angle_filter = 0
        candidate_frames_removed_by_angle_criterion = 1
        median_nose_speed_per_bout = []
        median_head_angle_per_bout = []
        "#,
    );

    let df = super::object_recognition::build_nort_exploration_diagnostics_dataframe(&[trial])
        .unwrap()
        .expect("diagnostics sheet");

    assert_eq!(df.height(), 2);
    assert_eq!(
        df.column("phase").unwrap().str().unwrap().get(0),
        Some("novelty")
    );
    assert_eq!(
        df.column("object_label").unwrap().str().unwrap().get(0),
        Some("A")
    );
    assert_eq!(
        df.column("object_label").unwrap().str().unwrap().get(1),
        Some("B")
    );
    assert_eq!(
        df.column("frames_pose_ok").unwrap().i64().unwrap().get(0),
        Some(3500)
    );
    assert_eq!(
        df.column("bout_lengths_ms_pre")
            .unwrap()
            .str()
            .unwrap()
            .get(0),
        Some("[100,200]")
    );
    assert_eq!(
        df.column("n_candidate_bouts_pre_merge")
            .unwrap()
            .i64()
            .unwrap()
            .get(0),
        Some(4)
    );
    assert_eq!(
        df.column("n_bouts_post_merge")
            .unwrap()
            .i64()
            .unwrap()
            .get(0),
        Some(2)
    );
    assert_eq!(
        df.column("n_final_valid_bouts")
            .unwrap()
            .i64()
            .unwrap()
            .get(0),
        Some(0)
    );
    assert_eq!(
        df.column("n_bouts_post_merge")
            .unwrap()
            .i64()
            .unwrap()
            .get(0)
            .unwrap()
            - df.column("n_bouts_removed_by_duration_filter")
                .unwrap()
                .i64()
                .unwrap()
                .get(0)
                .unwrap()
            - df.column("n_bouts_removed_by_speed_filter")
                .unwrap()
                .i64()
                .unwrap()
                .get(0)
                .unwrap()
            - df.column("n_bouts_removed_by_angle_filter")
                .unwrap()
                .i64()
                .unwrap()
                .get(0)
                .unwrap(),
        df.column("n_final_valid_bouts")
            .unwrap()
            .i64()
            .unwrap()
            .get(0)
            .unwrap()
    );
    let speed_lists = df
        .column("median_nose_speed_per_bout")
        .unwrap()
        .list()
        .unwrap();
    assert_eq!(speed_lists.get_as_series(0).unwrap().len(), 0);
    let angle_lists = df
        .column("median_head_angle_per_bout")
        .unwrap()
        .list()
        .unwrap();
    assert_eq!(angle_lists.get_as_series(0).unwrap().len(), 0);
}

#[test]
fn nort_exploration_diagnostics_do_not_extend_primary_nort_sheet() {
    let trial = parse(
        r#"
        experiment = "nort_videos_nort_post_t2_config_b_post_t2_configb_video_40"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"

        [session_context]
        animal = "12"
        trial = "90"

        [detailed]
        protocol = "nort_novelty"
        excluded = true
        total_duration_seconds = 120.0
        total_distance_meters = 4.0
        total_mean_speed = 0.05
        freezing_seconds = 1.0
        analysis_window_seconds = 120.0
        object_labels = ["familiar", "novel"]
        exploration_times = [0.0, 0.0]
        bout_counts = [0, 0]

        [[detailed.exploration_attributions]]
        frames_in_window = 3600
        frames_pose_ok = 3500
        frames_outside_object = 3000
        frames_nose_near = 200
        frames_nose_contact = 50
        frames_within_angle = 100
        frames_contact_or_angle = 120
        frames_slow_nose = 80
        frames_all_gates_raw = 20
        frames_after_tolerance = 0
        bouts_pre_filter = 4
        bouts_post_filter = 0
        bout_lengths_ms_pre = [100, 200]
        bout_lengths_ms_post = []
        "#,
    );

    let sheets = build_sheet_dataframes(vec![trial], &AggregateConfig::default()).unwrap();
    let (_, df, _) = sheets
        .iter()
        .find(|(name, _, _)| name == "nort_novelty")
        .expect("nort_novelty sheet");

    assert_eq!(df.height(), 1);
    assert!(df.column("total_exploration_s").is_ok());
    assert!(df.column("frames_in_window").is_err());
    assert!(df.column("bout_lengths_ms_pre").is_err());
}
