#[test]
fn aggregate_multi_recipe_split() {
    let rows = vec![
        nort_trial(
            "721_seq0_oultraining",
            721,
            58.0,
            290.0,
            false,
            &[],
            &[],
            &[],
        ),
        nort_trial(
            "721_seq3_oulupdate",
            721,
            22.0,
            30.0,
            false,
            &["arena", "ulp"],
            &[0.0, 10.0],
            &[0, 4],
        ),
        nort_trial(
            "721_seq4_oultest",
            721,
            15.0,
            15.0,
            false,
            &["arena", "ulp", "urp"],
            &[0.0, 13.0, 6.0],
            &[0, 9, 5],
        ),
    ];

    let cfg = AggregateConfig::default();
    let sheets = build_sheet_dataframes(rows, &cfg).expect("build ok");

    let names: Vec<&str> = sheets.iter().map(|(n, _, _)| n.as_str()).collect();
    // No rename map -> leftover path -> alphabetical order.
    assert_eq!(names, vec!["oultest", "oultraining", "oulupdate"]);

    // Every sheet has one row.
    for (name, df, _) in &sheets {
        assert_eq!(df.height(), 1, "sheet {name} should have one row");
    }
}

#[test]
fn oft_pre_post_display_aliases_have_distinct_recording_and_analysis_keys() {
    let rows = ["pre", "post"]
        .into_iter()
        .map(|timepoint| {
            parse(&format!(
                r#"
            experiment = "oft_videos_oft_{timepoint}_{timepoint}_video_2"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "open_field"
            [session_context]
            animal = "2"
            trial = "1"
            stage = "0"
            phase_part = "{}"
            [detailed]
            total_distance_meters = 1.0
        "#,
                if timepoint == "pre" { 1 } else { 2 }
            ))
        })
        .collect();
    let sheets = build_sheet_dataframes(rows, &AggregateConfig::default()).unwrap();
    let df = &sheets[0].1;
    assert_eq!(df.height(), 2);
    let alias = df.column("animal_trial_id").unwrap().str().unwrap();
    assert_eq!(alias.get(0), Some("A2_T1"));
    assert_eq!(alias.get(1), Some("A2_T1"));
    let recording = df.column("recording_id").unwrap().str().unwrap();
    let analysis = df.column("analysis_id").unwrap().str().unwrap();
    assert_ne!(recording.get(0), recording.get(1));
    assert_ne!(analysis.get(0), analysis.get(1));
    let points = df.column("timepoint").unwrap().str().unwrap();
    assert_eq!(points.get(0), Some("pre"));
    assert_eq!(points.get(1), Some("post"));
    assert_eq!(
        df.column("trial_id").unwrap().i64().unwrap().get(0),
        Some(1)
    );
}

#[test]
fn aggregate_explicit_passthrough_preserves_trial_rows() {
    let rows = vec![
        nort_trial(
            "721_seq0_oultraining",
            721,
            60.0,
            300.0,
            false,
            &[],
            &[],
            &[],
        ),
        nort_trial(
            "721_seq1_oultraining",
            721,
            30.0,
            150.0,
            true,
            &[],
            &[],
            &[],
        ),
        nort_trial(
            "721_seq2_oultraining",
            721,
            90.0,
            450.0,
            false,
            &[],
            &[],
            &[],
        ),
    ];

    let mut cfg = AggregateConfig::default();
    cfg.aggregation
        .insert("oultraining".into(), AggregationRule::Passthrough);

    let sheets = build_sheet_dataframes(rows, &cfg).expect("build ok");
    assert_eq!(sheets.len(), 1);
    let (name, df, _) = &sheets[0];
    assert_eq!(name, "oultraining");
    assert_eq!(
        df.height(),
        3,
        "passthrough must preserve one row per trial"
    );

    let dist = df.column("total_distance_m").unwrap().f64().unwrap();
    assert_eq!(dist.get(0), Some(60.0));
    assert_eq!(dist.get(1), Some(30.0));
    assert_eq!(dist.get(2), Some(90.0));

    let freez = df.column("freezing_s").unwrap().f64().unwrap();
    assert_eq!(freez.get(0), Some(300.0));
    assert_eq!(freez.get(1), Some(150.0));
    assert_eq!(freez.get(2), Some(450.0));

    let excluded = df.column("excluded").unwrap().bool().unwrap();
    assert_eq!(excluded.get(0), Some(false));
    assert_eq!(excluded.get(1), Some(true));
    assert_eq!(excluded.get(2), Some(false));

    let cohort = df
        .column("cohort")
        .unwrap()
        .str()
        .unwrap()
        .get(0)
        .unwrap()
        .to_string();
    assert_eq!(cohort, "cohortA");
}

#[test]
fn aggregate_recipe_rename_via_config() {
    let rows = vec![
        nort_trial(
            "721_seq0_oultraining",
            721,
            58.0,
            290.0,
            false,
            &[],
            &[],
            &[],
        ),
        nort_trial("721_seq3_oulupdate", 721, 22.0, 30.0, false, &[], &[], &[]),
        nort_trial("721_seq4_oultest", 721, 15.0, 15.0, false, &[], &[], &[]),
    ];

    let mut cfg = AggregateConfig::default();
    cfg.recipe_to_sheet
        .insert("oultraining".into(), "training".into());
    cfg.recipe_to_sheet
        .insert("oulupdate".into(), "update".into());
    cfg.recipe_to_sheet.insert("oultest".into(), "test".into());

    let sheets = build_sheet_dataframes(rows, &cfg).expect("build ok");
    let names: Vec<&str> = sheets.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(names, vec!["training", "update", "test"]);
}

#[test]
fn imen_passthrough_preserves_composite_identity_rows() {
    let rows = vec![
        parse(
            r#"
            experiment = "nort_videos_nort_post_t1_config_b_post_t1_configb_video_40"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "object_recognition"

            [animal_metadata]
            animal = 12
            group_name = "metadata present"
            hcar1 = "KO"

            [session_context]
            animal = "12"
            trial = "90"

            [detailed]
            protocol = "nort_training"
            excluded = true
            total_duration_seconds = 120.0
            total_distance_meters = 4.0
            total_mean_speed = 0.05
            freezing_seconds = 1.0
            analysis_window_seconds = 120.0
            object_labels = ["familiar", "variable"]
            exploration_times = [1.0, 0.0]
            bout_counts = [1, 0]
            "#,
        ),
        parse(
            r#"
            experiment = "nort_videos_nort_post_t1_config_b_post_t1_configb_video_44"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "object_recognition"

            [animal_metadata]
            animal = 12
            group_name = "metadata present"
            hcar1 = "KO"

            [session_context]
            animal = "12"
            trial = "91"

            [detailed]
            protocol = "nort_training"
            excluded = false
            total_duration_seconds = 120.0
            total_distance_meters = 8.0
            total_mean_speed = 0.05
            freezing_seconds = 2.0
            analysis_window_seconds = 120.0
            object_labels = ["familiar", "variable"]
            exploration_times = [2.0, 0.0]
            bout_counts = [1, 0]
            "#,
        ),
    ];

    let mut cfg = AggregateConfig::default();
    cfg.recipe_to_sheet
        .insert("nort_training".into(), "nort_training".into());

    let sheets = build_sheet_dataframes(rows, &cfg).expect("build ok");
    let (_, df, _) = &sheets[0];
    assert_eq!(df.height(), 2);
    assert_eq!(
        df.column("animal_trial_id").unwrap().str().unwrap().get(0),
        Some("A40_T90")
    );
    assert_eq!(
        df.column("animal_trial_id").unwrap().str().unwrap().get(1),
        Some("A44_T91")
    );
    assert_eq!(df.column("animal").unwrap().i64().unwrap().get(0), Some(40));
    assert_eq!(df.column("animal").unwrap().i64().unwrap().get(1), Some(44));
    assert_eq!(
        df.column("animal_id").unwrap().i64().unwrap().get(0),
        Some(40)
    );
    assert_eq!(
        df.column("trial_id").unwrap().i64().unwrap().get(1),
        Some(91)
    );
}

#[test]
fn imen_identity_uses_nort_trial_number_metadata_and_ymaze_video_fallback() {
    let nort = parse(
        r#"
        experiment = "nort_videos_nort_t1_pre_video_40"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "object_recognition"

        [animal_metadata]
        animal = 12
        group_name = "metadata present"

        [session_context]
        animal = "12"
        trial_number = "90"

        [detailed]
        protocol = "nort_training"
        excluded = false
        total_duration_seconds = 120.0
        total_distance_meters = 8.0
        total_mean_speed = 0.05
        freezing_seconds = 2.0
        analysis_window_seconds = 120.0
        object_labels = ["familiar", "variable"]
        exploration_times = [2.0, 0.0]
        bout_counts = [1, 0]
        "#,
    );
    let ymaze = parse(
        r#"
        experiment = "y_maze_videos_y_maze_pre_pre_video_38"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "y_maze"

        [detailed]
        excluded = false
        arm_entries = 4
        correct_alternations = 1
        incorrect_alternations = 1
        neutral_triads = 0
        spontaneous_alternation_pct = 50.0
        "#,
    );

    let mut cfg = AggregateConfig::default();
    cfg.recipe_to_sheet
        .insert("nort_training".into(), "nort_training".into());
    cfg.recipe_to_sheet.insert("y_maze".into(), "ymaze".into());

    let sheets = build_sheet_dataframes(vec![nort, ymaze], &cfg).expect("build ok");
    let nort_df = sheets
        .iter()
        .find(|(name, _, _)| name == "nort_training")
        .map(|(_, df, _)| df)
        .expect("nort sheet");
    assert_eq!(
        nort_df
            .column("animal_trial_id")
            .unwrap()
            .str()
            .unwrap()
            .get(0),
        Some("A40_T90")
    );
    assert_eq!(
        nort_df.column("animal").unwrap().i64().unwrap().get(0),
        Some(40)
    );

    let ymaze_df = sheets
        .iter()
        .find(|(name, _, _)| name == "ymaze")
        .map(|(_, df, _)| df)
        .expect("ymaze sheet");
    assert_eq!(
        ymaze_df
            .column("animal_trial_id")
            .unwrap()
            .str()
            .unwrap()
            .get(0),
        Some("A38_T38")
    );
    assert_eq!(
        ymaze_df.column("animal_id").unwrap().i64().unwrap().get(0),
        Some(38)
    );
}

#[test]
fn ymaze_passthrough_keeps_entry_counts_as_per_trial_integers() {
    let trial_a = parse(
        r#"
        experiment = "y_maze_videos_y_maze_pre_pre_video_48"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "y_maze"

        [detailed]
        total_duration_seconds = 300.0

        [[detailed.zone_summaries]]
        zone_name = "in_center"
        entries = 3
        duration_seconds = 10.0
        distance_meters = 1.0
        mean_speed = 0.1
        "#,
    );
    let trial_b = parse(
        r#"
        experiment = "y_maze_videos_y_maze_pre_pre_video_61"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "y_maze"

        [detailed]
        total_duration_seconds = 300.0

        [[detailed.zone_summaries]]
        zone_name = "in_center"
        entries = 4
        duration_seconds = 20.0
        distance_meters = 2.0
        mean_speed = 0.1
        "#,
    );

    let row_a = YMazeExperiment.extract_row(&trial_a);
    let row_b = YMazeExperiment.extract_row(&trial_b);
    assert_eq!(row_a.get("center_entries"), Some(&CellValue::Int(3)));
    assert_eq!(row_b.get("center_entries"), Some(&CellValue::Int(4)));
}

#[test]
fn aggregate_per_object_column_union_across_trials() {
    let rows = vec![
        nort_trial(
            "721_seq4_oultest",
            721,
            15.0,
            15.0,
            false,
            &["arena", "ulp", "urp"],
            &[0.0, 13.0, 6.0],
            &[0, 9, 5],
        ),
        nort_trial(
            "842_seq4_oultest",
            842,
            20.0,
            20.0,
            false,
            &["arena", "llp", "lrp"],
            &[0.0, 11.0, 7.0],
            &[0, 8, 4],
        ),
    ];

    let cfg = AggregateConfig::default();
    let sheets = build_sheet_dataframes(rows, &cfg).expect("build ok");
    assert_eq!(sheets.len(), 1);
    let (_, df, _) = &sheets[0];

    // All four per-object columns present in the union.
    for col in [
        "ulp_exploration_s",
        "urp_exploration_s",
        "llp_exploration_s",
        "lrp_exploration_s",
    ] {
        assert!(
            df.schema().contains(col),
            "missing union column {col} in {:?}",
            df.schema()
        );
    }

    assert_eq!(df.height(), 2);

    // Row 0 (animal 721) should have ulp/urp populated and llp/lrp null.
    let ulp = df.column("ulp_exploration_s").unwrap().f64().unwrap();
    let urp = df.column("urp_exploration_s").unwrap().f64().unwrap();
    let llp = df.column("llp_exploration_s").unwrap().f64().unwrap();
    let lrp = df.column("lrp_exploration_s").unwrap().f64().unwrap();

    assert_eq!(ulp.get(0), Some(13.0));
    assert_eq!(urp.get(0), Some(6.0));
    assert_eq!(llp.get(0), None);
    assert_eq!(lrp.get(0), None);

    // Row 1 (animal 842) should be the inverse.
    assert_eq!(ulp.get(1), None);
    assert_eq!(urp.get(1), None);
    assert_eq!(llp.get(1), Some(11.0));
    assert_eq!(lrp.get(1), Some(7.0));
}

#[test]
fn aggregate_metadata_columns_feed_animal_summary() {
    let mut trial = nort_trial(
        "721_seq0_oultraining",
        721,
        60.0,
        300.0,
        false,
        &["arena", "familiar", "variable"],
        &[0.0, 10.0, 5.0],
        &[0, 2, 1],
    );
    trial.animal_metadata = serde_json::json!({
        "animal": 721,
        "group": 3,
        "group_name": "Group 3 WT lactate",
        "sex": "male",
        "hcar1": "KO",
        "age_group": "old",
        "treatment_onset": "late",
        "five_xfad": "wt",
        "age_group_nr": 1,
        "start_age_days": 99,
        "start_age_months": 3.3,
        "end_age_days": 182,
        "end_age_months": 6.06,
        "treatment": "lactate",
        "round": 1,
        "comments": ""
    });

    let mut cfg = AggregateConfig::default();
    cfg.recipe_to_sheet
        .insert("oultraining".into(), "nort_training".into());

    let sheets = build_sheet_dataframes(vec![trial], &cfg).expect("build ok");
    let (_, df, _) = &sheets[0];
    assert!(df.schema().contains("group_name"));
    assert!(!df.schema().contains("cohort"));
    assert_eq!(
        df.column("group_name").unwrap().str().unwrap().get(0),
        Some("Group 3 WT lactate")
    );

    let summary = super::pipeline::build_animal_overview_dataframe(&sheets)
        .unwrap()
        .expect("summary");
    assert_eq!(summary.height(), 1);
    assert_eq!(
        summary.column("animal").unwrap().i64().unwrap().get(0),
        Some(721)
    );
    assert_eq!(
        summary
            .column("nort_training_total_distance_m")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        Some(60.0)
    );
}
