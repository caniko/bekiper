use indexmap::IndexMap;

// -- Cell, dataframe and identity edge cases --------------------------

fn bare_trial(experiment: &str, task_type: TaskType) -> TrialResult {
    TrialResult {
        experiment: experiment.into(),
        task_type,
        analysis_role: None,
        fps: 30.0,
        meters_per_pixel: 0.001,
        animal_metadata: Value::Null,
        session_context: Value::Null,
        summaries: Vec::new(),
        detailed: Value::Null,
    }
}

#[test]
fn json_cells_collapse_containers_and_keep_large_unsigned_counts() {
    let big = serde_json::json!(u64::MAX);
    assert_eq!(CellValue::from_json(&big), CellValue::Int(u64::MAX as i64));
    assert_eq!(
        CellValue::from_json(&serde_json::json!([1])),
        CellValue::Null
    );
    assert_eq!(
        CellValue::from_json(&serde_json::json!({"a": 1})),
        CellValue::Null
    );
    let trial = bare_trial("t", TaskType::OpenField);
    assert_eq!(
        cell::pluck_trial_metadata(&trial, "unknown"),
        CellValue::Null
    );
}

#[test]
fn mixed_columns_pick_one_dtype_and_render_incompatible_values() {
    let row = |pairs: &[(&str, CellValue)]| {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect::<IndexMap<_, _>>()
    };
    let df = dataframe::dataframe_from_extracted(
        vec!["float".into(), "flag".into(), "list".into(), "text".into()],
        vec![
            row(&[
                ("float", CellValue::Float(0.5)),
                ("flag", CellValue::Bool(true)),
                ("list", CellValue::FloatList(vec![1.0, 2.5])),
                ("text", CellValue::String("a".into())),
            ]),
            row(&[
                ("float", CellValue::Bool(true)),
                ("flag", CellValue::Null),
                ("list", CellValue::Null),
                ("text", CellValue::Int(3)),
            ]),
            row(&[("text", CellValue::Float(1.5))]),
            row(&[("text", CellValue::FloatList(vec![1.0, 2.5]))]),
            row(&[("text", CellValue::Bool(false))]),
        ],
    )
    .unwrap();
    assert_eq!(df.column("float").unwrap().f64().unwrap().get(1), Some(1.0));
    assert_eq!(df.column("flag").unwrap().bool().unwrap().get(1), None);
    assert!(df.column("list").unwrap().get(1).unwrap().is_null());
    let text = df.column("text").unwrap().str().unwrap();
    assert_eq!(
        text.into_iter().collect::<Vec<_>>(),
        [
            Some("a"),
            Some("3"),
            Some("1.5"),
            Some("[1,2.5]"),
            Some("false")
        ]
    );
}

#[test]
fn every_experiment_reports_only_non_static_columns_as_dynamic() {
    let oft = bare_trial("oft_trial", TaskType::OpenField);
    let ymaze = bare_trial("ymaze_trial", TaskType::YMaze);
    let nort = nort_trial("nort_trial", 1, 1.0, 0.0, false, &["a"], &[1.0], &[1]);
    for (experiment, trial) in [
        (Box::new(OftExperiment) as Box<dyn Experiment>, &oft),
        (Box::new(YMazeExperiment), &ymaze),
        (Box::new(GenericExperiment), &oft),
        (experiment_for_trial(&nort), &nort),
    ] {
        let statics = experiment.static_columns();
        let dynamic = experiment.dynamic_columns(trial);
        assert!(dynamic.iter().all(|column| !statics.contains(column)));
        assert!(
            dynamic
                .iter()
                .all(|column| experiment.extract_row(trial).contains_key(column))
        );
    }
    assert!(GenericExperiment.animal_overview_columns().is_empty());
    assert_eq!(GenericExperiment.id(), ExperimentId("generic"));
    assert!(
        GenericExperiment
            .static_columns()
            .contains(&"fps".to_string())
    );
    assert_eq!(
        GenericExperiment.sheet_key(&oft).unwrap().routing_key(),
        "open_field"
    );
    assert_eq!(
        GenericExperiment
            .sheet_key(&bare_trial(
                "1_seq0_oultraining",
                TaskType::ObjectRecognition
            ))
            .unwrap()
            .routing_key(),
        "oultraining"
    );
}

#[test]
fn nort_extraction_falls_back_for_an_untyped_detail_and_unknown_roles() {
    let mut trial = nort_trial("nort_trial", 1, 1.0, 0.0, false, &["a"], &[1.0], &[1]);
    trial.detailed["protocol"] = serde_json::json!(5);
    trial.analysis_role = Some("custom".into());
    let row = experiment_for_trial(&trial).extract_row(&trial);
    assert_eq!(row["experiment"], CellValue::String("nort_trial".into()));
}

#[test]
fn aggregate_identity_falls_back_through_row_cells_and_requires_imen_animals() {
    let trial = bare_trial("descriptive", TaskType::OpenField);
    for (cell, expected) in [
        (CellValue::Int(4), Some(4)),
        (CellValue::Float(5.0), Some(5)),
        (CellValue::Float(5.5), None),
        (CellValue::String(" 6 ".into()), Some(6)),
        (CellValue::Bool(true), None),
    ] {
        let row = IndexMap::from([("animal".to_string(), cell)]);
        let identity = identity::resolve_aggregate_identity(&trial, &row).unwrap();
        assert_eq!(identity.animal_id, expected);
        assert_eq!(
            identity.trial_id, expected,
            "trial falls back to the animal"
        );
    }
    let error = identity::resolve_aggregate_identity(
        &bare_trial("oft_videos_unnumbered", TaskType::OpenField),
        &IndexMap::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("missing animal_id"), "{error}");
}

#[test]
fn heterogeneous_sheets_use_the_first_trials_experiment() {
    let mut oft = bare_trial("1_oft", TaskType::OpenField);
    oft.analysis_role = Some("oft".into());
    let mut ymaze = bare_trial("2_ymaze", TaskType::YMaze);
    ymaze.analysis_role = Some("ymaze".into());
    let cfg = AggregateConfig {
        recipe_to_sheet: IndexMap::from([
            ("oft".to_string(), "combined".to_string()),
            ("y_maze".to_string(), "combined".to_string()),
            ("ymaze".to_string(), "combined".to_string()),
        ]),
        ..Default::default()
    };
    let sheets = crate::traced(|| build_sheet_dataframes(vec![oft, ymaze], &cfg)).unwrap();
    assert_eq!(sheets.len(), 1);
    assert_eq!(sheets[0].0, "combined");
    assert_eq!(sheets[0].2.id(), OftExperiment.id());
    assert_eq!(sheets[0].1.height(), 2);
}

#[test]
fn animal_overview_reads_every_cell_type_and_skips_unkeyed_rows() {
    use polars::prelude::*;
    let n = 5;
    let df = DataFrame::new(
        n,
        vec![
            Column::new(
                "animal".into(),
                [Some(1.0), Some(2.5), None, Some(4.0), Some(5.0)],
            ),
            Column::new("group".into(), [true, false, true, true, true]),
            Column::new("group_name".into(), [1i8; 5]),
            Column::new("sex".into(), [1i16; 5]),
            Column::new("hcar1".into(), [1i32; 5]),
            Column::new("five_xfad".into(), [1u32; 5]),
            Column::new("age_group_nr".into(), [1u64, 1, 1, u64::MAX, 1]),
            Column::new("start_age_days".into(), [1.5f32; 5]),
            Column::new_scalar(
                "start_age_months".into(),
                Scalar::new(DataType::String, AnyValue::StringOwned("x".into())),
                n,
            ),
            Column::new(
                "end_age_days".into(),
                [Some(" "), Some("a"), None, Some("b"), Some("c")],
            ),
            Column::new("end_age_months".into(), [1i32; 5])
                .cast(&DataType::Date)
                .unwrap(),
        ],
    )
    .unwrap();
    let strings = DataFrame::new(1, vec![Column::new("animal".into(), ["7"])]).unwrap();
    let unkeyed = DataFrame::new(1, vec![Column::new("other".into(), [1i64])]).unwrap();
    let sheets: Vec<AggregateSheet> = vec![
        ("oft".into(), df, Box::new(OftExperiment)),
        ("ymaze".into(), strings, Box::new(YMazeExperiment)),
        ("none".into(), unkeyed, Box::new(OftExperiment)),
    ];
    let overview = pipeline::build_animal_overview_dataframe(&sheets)
        .unwrap()
        .unwrap();
    // Float keys need an integral value; a null animal is skipped.
    assert_eq!(
        overview
            .column("animal")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        [Some("1"), Some("4"), Some("5"), Some("7")]
    );
    let column = |name: &str| overview.column(name).unwrap().clone();
    assert_eq!(column("group").bool().unwrap().get(0), Some(true));
    assert_eq!(column("five_xfad").i64().unwrap().get(0), Some(1));
    assert_eq!(column("start_age_days").f64().unwrap().get(0), Some(1.5));
    assert_eq!(column("start_age_months").str().unwrap().get(0), Some("x"));
    assert_eq!(
        column("end_age_months").str().unwrap().get(0),
        Some("1970-01-02")
    );
    let big = column("age_group_nr");
    let big = big.str().unwrap();
    assert_eq!(big.get(0), Some("1"));
    assert_eq!(big.get(1), Some(u64::MAX.to_string().as_str()));
}

#[test]
fn ymaze_rows_tally_mixed_sequences_and_saturate_oversized_counts() {
    let mut trial = bare_trial("ymaze_tally", TaskType::YMaze);
    trial.detailed = serde_json::json!({"analysis": {
        "arm_sequence": "not a sequence",
        "raw_arm_sequence": ["A", 1, "X", "B", "C"],
        "raw_correct_alternations": 1,
        "raw_incorrect_alternations": 1,
        "raw_spontaneous_alternation_pct": 50.0,
        "incorrect_triad_distribution": {"aab": u64::MAX, "aac": 1},
        "neutral_triad_distribution": {"aaa": 2}
    }});
    let row = YMazeExperiment.extract_row(&trial);
    assert_eq!(
        row["raw_spontaneous_alternation_pct"],
        CellValue::Float(50.0)
    );
    assert_eq!(row["incorrect_sum"], CellValue::Int(i64::MAX));
    assert_eq!(row["neutral_sum"], CellValue::Int(2));
    assert!(matches!(
        row["raw_spontaneous_alternation_pct_standard"],
        CellValue::Float(value) if (value - 100.0 / 3.0).abs() < 1e-9
    ));
}

#[test]
fn enriched_rows_fill_null_identity_cells_from_the_extractor() {
    let mut trial = bare_trial("ymaze_enriched", TaskType::YMaze);
    trial.analysis_role = Some("ymaze".into());
    trial.animal_metadata = serde_json::json!({"group_name": "treated"});
    let sheets = build_sheet_dataframes(vec![trial], &AggregateConfig::default()).unwrap();
    let df = &sheets[0].1;
    assert_eq!(
        df.column("group_name").unwrap().str().unwrap().get(0),
        Some("treated")
    );
    assert_eq!(
        df.column("timepoint").unwrap().str().unwrap().get(0),
        Some("unknown")
    );
}

#[test]
fn nort_zone_names_without_a_quadrant_prefix_are_kept() {
    let mut trial = nort_trial(
        "nort_zones",
        1,
        1.0,
        0.0,
        false,
        &["a", "b"],
        &[1.0, 2.0],
        &[1, 1],
    );
    trial.detailed["zone_metrics"] = serde_json::json!({
        "center_summary": null,
        "quadrant_summaries": [{"zone_name": "corner", "entries": 1, "duration_seconds": 2.0,
            "distance_meters": 0.1, "mean_speed": 0.05, "latency_seconds": 0.0}],
        "quadrant_motion_summaries": []
    });
    let row = experiment_for_trial(&trial).extract_row(&trial);
    assert!(
        row.keys().any(|key| key.starts_with("corner_")),
        "{:?}",
        row.keys()
    );
}

#[test]
fn parameters_sheet_unions_keys_one_row_per_analysis() {
    let trial = |name: &str, key: &str| TrialResult {
        experiment: name.into(),
        task_type: bekiper_core::types::TaskType::OpenField,
        analysis_role: None,
        fps: 30.0,
        meters_per_pixel: 0.002,
        animal_metadata: serde_json::Value::Null,
        session_context: serde_json::Value::Null,
        summaries: Vec::new(),
        detailed: serde_json::json!({"effective_parameters": {"values": {key: 0.5, "scale.method": "arena_side", "flag": true, "table": [[0, 1]]}}}),
    };
    let df =
        super::pipeline::build_parameters_dataframe(&[trial("a", "x.one"), trial("b", "x.two")])
            .unwrap()
            .unwrap();
    // analysis_id, task, x.one, scale.method, flag, table, x.two
    assert_eq!(df.shape(), (2, 7));
}
