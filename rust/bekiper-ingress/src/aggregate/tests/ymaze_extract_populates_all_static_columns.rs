#[test]
fn ymaze_scoring_provenance_survives_aggregate_cell_conversion() {
    let mut trial = parse(&ymaze_fixture_toml(true));
    let historical = YMazeExperiment.extract_row(&trial);
    assert_eq!(historical.get("scoring_provenance"), Some(&CellValue::Null));
    let evidence = serde_json::json!({
        "geometry_version": "imen_radial_vertex_span_v1",
        "entry_rule": "dual_landmark_false_to_true_v1",
        "apply_dwell_filter": false,
        "minimum_dwell_seconds": 0,
        "entry_frames": {"in_arm_A": [12, 31]}
    });
    trial.detailed["scoring_provenance"] = evidence.clone();
    let row = YMazeExperiment.extract_row(&trial);
    let Some(CellValue::String(text)) = row.get("scoring_provenance") else {
        panic!("structured scoring provenance must be a JSON text cell");
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(text).unwrap(),
        evidence
    );
}

#[test]
fn ymaze_entry_difference_does_not_claim_unverified_dwell_rejections() {
    let mut trial = parse(&ymaze_fixture_toml(true));
    let historical = YMazeExperiment.extract_row(&trial);
    assert_eq!(
        historical["raw_to_scored_entry_difference"],
        CellValue::Int(14)
    );
    assert_eq!(historical["removed_short_entries"], CellValue::Int(14));
    assert_eq!(historical["dwell_rejected_entries"], CellValue::Null);

    trial.detailed["scoring_provenance"] = serde_json::json!({
        "minimum_dwell_seconds": 0.0,
        "dwell_bouts": {
            "in_arm_A": [{"accepted": true}],
            "in_arm_B": [{"accepted": true}],
            "in_arm_C": [],
            "in_center": [{"accepted": false, "rejection_reason": "minimum_dwell"}]
        }
    });
    let row = YMazeExperiment.extract_row(&trial);
    assert_eq!(row["raw_to_scored_entry_difference"], CellValue::Int(14));
    assert_eq!(row["dwell_rejected_entries"], CellValue::Int(0));

    trial.detailed["scoring_provenance"]["dwell_bouts"]["in_arm_A"] =
        serde_json::json!([{"accepted": false, "rejection_reason": "minimum_dwell"}]);
    let row = YMazeExperiment.extract_row(&trial);
    assert_eq!(row["dwell_rejected_entries"], CellValue::Int(1));

    trial.detailed["scoring_provenance"]["dwell_bouts"]
        .as_object_mut()
        .unwrap()
        .remove("in_arm_C");
    assert_eq!(
        YMazeExperiment.extract_row(&trial)["dwell_rejected_entries"],
        CellValue::Null
    );
    trial.detailed["scoring_provenance"]["dwell_bouts"]["in_arm_C"] = serde_json::json!([]);

    trial.detailed["scoring_provenance"]["dwell_bouts"]["in_arm_B"] =
        serde_json::json!([{"accepted": false, "rejection_reason": "unknown"}]);
    assert_eq!(
        YMazeExperiment.extract_row(&trial)["dwell_rejected_entries"],
        CellValue::Null
    );
}

#[test]
fn ymaze_entry_difference_retains_signed_count_changes() {
    let mut trial = parse(&ymaze_fixture_toml(true));
    trial.detailed["analysis"]["raw_arm_sequence"] = serde_json::json!(["A"]);
    let row = YMazeExperiment.extract_row(&trial);
    assert_eq!(row["raw_to_scored_entry_difference"], CellValue::Int(-3));
    assert_eq!(row["removed_short_entries"], CellValue::Int(-3));
    assert_eq!(row["dwell_rejected_entries"], CellValue::Null);
}

#[test]
fn ymaze_extract_populates_all_static_columns() {
    let t = parse(&ymaze_fixture_toml(true));
    let row = YMazeExperiment.extract_row(&t);

    for c in YMazeExperiment.static_columns() {
        assert!(row.contains_key(&c), "missing YMaze static column {c}");
    }

    // Metadata
    assert_eq!(row.get("animal"), Some(&CellValue::Int(99)));
    assert_eq!(
        row.get("timepoint"),
        Some(&CellValue::String("unknown".into()))
    );
    assert_eq!(row.get("fps"), Some(&CellValue::Float(30.0)));
    assert_eq!(row.get("total_duration_s"), Some(&CellValue::Float(183.0)));
    assert_eq!(
        row.get("analysis_duration_s"),
        Some(&CellValue::Float(180.0))
    );
    assert_eq!(row.get("total_distance_m"), Some(&CellValue::Float(6.7)));
    assert_eq!(row.get("total_mean_speed"), Some(&CellValue::Float(0.04)));
    assert_eq!(row.get("proximal_threshold"), Some(&CellValue::Float(0.65)));
    assert_eq!(
        row.get("start_mode"),
        Some(&CellValue::String("first_detection".into()))
    );
    assert_eq!(
        row.get("zone_max_gap_seconds"),
        Some(&CellValue::Float(0.1))
    );
    assert_eq!(
        row.get("arm_membership_hysteresis"),
        Some(&CellValue::Bool(true))
    );
    assert_eq!(
        row.get("pose_gap_fill_seconds"),
        Some(&CellValue::Float(0.2))
    );

    // Arm A - `raw_*` comes from the truly raw arm sequence, while `valid_*`
    // is derived from the mitigated `arm_sequence` used for headline SA.
    assert_eq!(row.get("raw_total_arm_entries"), Some(&CellValue::Int(18)));
    assert_eq!(row.get("raw_arm_A_entries"), Some(&CellValue::Int(6)));
    assert_eq!(row.get("arm_A_duration_s"), Some(&CellValue::Float(40.0)));
    assert_eq!(row.get("arm_A_distance_m"), Some(&CellValue::Float(1.5)));
    assert_eq!(row.get("arm_A_mean_speed"), Some(&CellValue::Float(0.0375)));
    assert_eq!(row.get("arm_A_latency_s"), Some(&CellValue::Float(2.0)));
    assert_eq!(
        row.get("arm_A_motion_distance_m"),
        Some(&CellValue::Float(1.4))
    );
    assert_eq!(
        row.get("arm_A_motion_median_speed"),
        Some(&CellValue::Float(0.032))
    );
    assert_eq!(
        row.get("arm_A_motion_median_acceleration"),
        Some(&CellValue::Float(0.14))
    );
    assert_eq!(
        row.get("arm_A_motion_freezing_s"),
        Some(&CellValue::Float(2.1))
    );
    // Arm B
    assert_eq!(row.get("raw_arm_B_entries"), Some(&CellValue::Int(5)));
    assert_eq!(row.get("arm_B_duration_s"), Some(&CellValue::Float(38.0)));
    assert_eq!(row.get("arm_B_distance_m"), Some(&CellValue::Float(1.4)));
    assert_eq!(row.get("arm_B_latency_s"), Some(&CellValue::Float(3.1)));
    // Arm C
    assert_eq!(row.get("raw_arm_C_entries"), Some(&CellValue::Int(7)));
    assert_eq!(row.get("arm_C_duration_s"), Some(&CellValue::Float(45.0)));
    assert_eq!(row.get("arm_C_distance_m"), Some(&CellValue::Float(1.7)));
    assert_eq!(row.get("arm_C_latency_s"), Some(&CellValue::Float(4.2)));

    // Center
    assert_eq!(row.get("center_entries"), Some(&CellValue::Int(17)));
    assert_eq!(row.get("center_duration_s"), Some(&CellValue::Float(60.0)));
    assert_eq!(row.get("center_distance_m"), Some(&CellValue::Float(2.1)));
    assert_eq!(row.get("center_mean_speed"), Some(&CellValue::Float(0.035)));
    assert_eq!(row.get("center_latency_s"), Some(&CellValue::Float(0.0)));
    assert_eq!(
        row.get("center_motion_distance_m"),
        Some(&CellValue::Float(2.0))
    );
    assert_eq!(
        row.get("center_motion_median_speed"),
        Some(&CellValue::Float(0.028))
    );
    assert_eq!(
        row.get("center_motion_median_acceleration"),
        Some(&CellValue::Float(0.11))
    );
    assert_eq!(
        row.get("center_motion_freezing_s"),
        Some(&CellValue::Float(6.0))
    );

    // Alternation
    assert_eq!(
        row.get("zone_entry_sequence"),
        Some(&CellValue::String("A Center B Center C".into()))
    );
    assert_eq!(
        row.get("raw_arm_entry_sequence"),
        Some(&CellValue::String(
            "A A A A A A B B B B B C C C C C C C".into()
        ))
    );
    assert_eq!(
        row.get("valid_arm_entry_sequence"),
        Some(&CellValue::String("A B C A".into()))
    );
    assert_eq!(
        row.get("first_arm_entered"),
        Some(&CellValue::String("A".into()))
    );
    assert_eq!(
        row.get("last_arm_entered"),
        Some(&CellValue::String("A".into()))
    );
    // Valid counts: 2 × "A" + 1 × "B" + 1 × "C" in ["A", "B", "C", "A"].
    assert_eq!(row.get("valid_arm_A_entries"), Some(&CellValue::Int(2)));
    assert_eq!(row.get("valid_arm_B_entries"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("valid_arm_C_entries"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("valid_total_arm_entries"), Some(&CellValue::Int(4)));
    assert_eq!(row.get("removed_short_entries"), Some(&CellValue::Int(14)));
    assert_eq!(row.get("total_triads"), Some(&CellValue::Int(2)));
    assert_eq!(row.get("correct_alternations"), Some(&CellValue::Int(10)));
    assert_eq!(row.get("incorrect_alternations"), Some(&CellValue::Int(5)));
    assert_eq!(row.get("usable_triads"), Some(&CellValue::Int(15)));
    assert_eq!(row.get("neutral_triads"), Some(&CellValue::Int(2)));
    assert_eq!(
        row.get("neutral_triads_pct"),
        Some(&CellValue::Float(100.0))
    );
    assert_eq!(
        row.get("spontaneous_alternation_pct"),
        Some(&CellValue::Float(66.67))
    );
    // SA(2) = correct_alternations / (valid_total_arm_entries - 2) * 100
    //       = 10 / (4 - 2) * 100 = 500.0 for this synthetic fixture.
    assert_eq!(
        row.get("spontaneous_alternation_pct_standard"),
        Some(&CellValue::Float(500.0))
    );
    assert_eq!(row.get("arm_preference"), Some(&CellValue::Float(2.94)));
    assert_eq!(row.get("perm_ABC"), Some(&CellValue::Int(2)));
    assert_eq!(row.get("perm_ACB"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("perm_BAC"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("perm_BCA"), Some(&CellValue::Int(3)));
    assert_eq!(row.get("perm_CAB"), Some(&CellValue::Int(2)));
    assert_eq!(row.get("perm_CBA"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("permutation_sum"), Some(&CellValue::Int(10)));
    assert_eq!(row.get("incorrect_AAB"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("incorrect_ABA"), Some(&CellValue::Int(2)));
    assert_eq!(row.get("incorrect_BCB"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("incorrect_CAC"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("incorrect_sum"), Some(&CellValue::Int(5)));
    assert_eq!(row.get("neutral_ABB"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("neutral_BCC"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("neutral_sum"), Some(&CellValue::Int(2)));
    assert_eq!(row.get("seconds_in_arms"), Some(&CellValue::Float(123.0)));
    assert_eq!(
        row.get("seconds_in_arms_and_center"),
        Some(&CellValue::Float(183.0))
    );

    // Other
    assert_eq!(row.get("freezing_s"), Some(&CellValue::Float(12.3)));
    assert_eq!(
        row.get("low_valid_entries_flag"),
        Some(&CellValue::Bool(true))
    );
    assert_eq!(
        row.get("low_usable_triads_flag"),
        Some(&CellValue::Bool(false))
    );

    // Trailing
    assert_eq!(
        row.get("experiment"),
        Some(&CellValue::String("99_seq2_ymaze".into()))
    );
    // `excluded` absent from fixture -> Null
    assert_eq!(row.get("excluded"), Some(&CellValue::Null));
}

#[test]
fn ymaze_extract_uses_zone_name_not_index() {
    // Permuted zone_summaries array order must still resolve arm A/B/C
    // + center correctly via zone_name lookup.
    let t = parse(&ymaze_fixture_toml(false));
    let row = YMazeExperiment.extract_row(&t);

    // Raw and valid sequence-derived counts are independent of the
    // permuted zone-summary array order.
    assert_eq!(row.get("raw_arm_A_entries"), Some(&CellValue::Int(6)));
    assert_eq!(row.get("raw_arm_B_entries"), Some(&CellValue::Int(5)));
    assert_eq!(row.get("raw_arm_C_entries"), Some(&CellValue::Int(7)));
    assert_eq!(row.get("valid_arm_A_entries"), Some(&CellValue::Int(2)));
    assert_eq!(row.get("valid_arm_B_entries"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("valid_arm_C_entries"), Some(&CellValue::Int(1)));
    assert_eq!(row.get("arm_A_duration_s"), Some(&CellValue::Float(40.0)));
    assert_eq!(row.get("arm_C_distance_m"), Some(&CellValue::Float(1.7)));
    assert_eq!(
        row.get("arm_A_motion_distance_m"),
        Some(&CellValue::Float(1.4))
    );
    assert_eq!(
        row.get("arm_C_motion_freezing_s"),
        Some(&CellValue::Float(2.4))
    );
    assert_eq!(row.get("center_entries"), Some(&CellValue::Int(17)));
    assert_eq!(row.get("center_duration_s"), Some(&CellValue::Float(60.0)));
    assert_eq!(
        row.get("center_motion_median_speed"),
        Some(&CellValue::Float(0.028))
    );
}

#[test]
fn ymaze_extract_missing_new_scalar_fields_yields_nulls() {
    let t = parse(
        r#"
        experiment = "legacy_ymaze"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "y_maze"

        [detailed]
        total_duration_seconds = 120.0

        [detailed.analysis]
        arm_sequence = ["A", "B"]
        "#,
    );
    let row = YMazeExperiment.extract_row(&t);

    // Historical result.toml files do not know the variant identity or
    // post-trim duration, so the aggregate should expose empty cells rather
    // than inferring values from defaults or filenames.
    assert_eq!(row.get("proximal_threshold"), Some(&CellValue::Null));
    assert_eq!(row.get("start_mode"), Some(&CellValue::Null));
    assert_eq!(row.get("zone_max_gap_seconds"), Some(&CellValue::Null));
    assert_eq!(row.get("arm_membership_hysteresis"), Some(&CellValue::Null));
    assert_eq!(row.get("pose_gap_fill_seconds"), Some(&CellValue::Null));
    assert_eq!(row.get("analysis_duration_s"), Some(&CellValue::Null));
    assert_eq!(row.get("first_arm_entered"), Some(&CellValue::Null));
    assert_eq!(row.get("last_arm_entered"), Some(&CellValue::Null));
    assert_eq!(row.get("total_triads"), Some(&CellValue::Int(0)));
}

// -- GenericExperiment -------------------------------------------

#[test]
fn generic_extract_flattens_arbitrary_detailed() {
    // A Cheeseboard-shaped fixture with the legacy start/reward fields plus
    // nested objects + arrays in `detailed`. The generic extractor should
    // flatten everything into dot-pathed columns alongside the metadata
    // prefix.
    let t = parse(
        r#"
        experiment = "555_seq1_cheeseboard"
        fps = 30.0
        meters_per_pixel = 0.001
        task_type = "cheeseboard"

        [animal_metadata]
        animal = 555
        cohort = "D"
        gene = "wt"
        sex = "m"
        age = "16"

        [detailed]
        excluded = false
        trial_duration_seconds = 240.0
        enclosure_label = "board"
        reward_labels = ["reward_a", "reward_b"]
        start_area_exit_frame = 12
        start_area_exit_seconds = 0.4
        reward_arrival_frame = 27
        reward_arrival_seconds = 0.9
        seconds_to_find_reward = 0.5
        seconds_in_reward_area = 18.7
        reward_entries = 3
        search_path_distance_meters = 1.23
        search_direct_distance_meters = 0.98
        search_mean_speed_mps = 2.46
        search_tortuosity = 1.25
        total_distance_meters = 18.7
        total_mean_speed_mps = 0.62
        total_duration_seconds = 240.0

        [detailed.enclosure_summary]
        zone_name = "board"
        entries = 1
        duration_seconds = 240.0
        distance_meters = 18.7
        mean_speed = 0.62
        latency_seconds = 0.4

        [[detailed.reward_summaries]]
        zone_name = "reward_a"
        entries = 0
        duration_seconds = 0.0
        distance_meters = 0.0
        mean_speed = 0.0
        latency_seconds = 1.2

        [[detailed.reward_summaries]]
        zone_name = "reward_b"
        entries = 1
        duration_seconds = 18.7
        distance_meters = 1.1
        mean_speed = 0.3
        latency_seconds = 0.9

        [[detailed.zone_summaries]]
        zone_name = "target"
        entries = 1
        duration_seconds = 2.0

        [[detailed.zone_summaries]]
        zone_name = "arena"
        entries = 1
        duration_seconds = 238.0
        "#,
    );

    let row = GenericExperiment.extract_row(&t);

    // Metadata prefix
    assert_eq!(row.get("animal"), Some(&CellValue::Int(555)));
    assert_eq!(row.get("cohort"), Some(&CellValue::String("D".into())));
    assert_eq!(row.get("gene"), Some(&CellValue::String("wt".into())));
    assert_eq!(row.get("fps"), Some(&CellValue::Float(30.0)));

    // Flattened Cheeseboard leaves
    assert_eq!(
        row.get("trial_duration_seconds"),
        Some(&CellValue::Float(240.0))
    );
    assert_eq!(
        row.get("enclosure_label"),
        Some(&CellValue::String("board".into()))
    );
    assert_eq!(
        row.get("reward_labels[0]"),
        Some(&CellValue::String("reward_a".into()))
    );
    assert_eq!(
        row.get("reward_labels[1]"),
        Some(&CellValue::String("reward_b".into()))
    );
    assert_eq!(row.get("start_area_exit_frame"), Some(&CellValue::Int(12)));
    assert_eq!(
        row.get("start_area_exit_seconds"),
        Some(&CellValue::Float(0.4))
    );
    assert_eq!(row.get("reward_arrival_frame"), Some(&CellValue::Int(27)));
    assert_eq!(
        row.get("reward_arrival_seconds"),
        Some(&CellValue::Float(0.9))
    );
    assert_eq!(
        row.get("seconds_to_find_reward"),
        Some(&CellValue::Float(0.5))
    );
    assert_eq!(
        row.get("seconds_in_reward_area"),
        Some(&CellValue::Float(18.7))
    );
    assert_eq!(row.get("reward_entries"), Some(&CellValue::Int(3)));
    assert_eq!(
        row.get("search_path_distance_meters"),
        Some(&CellValue::Float(1.23))
    );
    assert_eq!(
        row.get("search_direct_distance_meters"),
        Some(&CellValue::Float(0.98))
    );
    assert_eq!(
        row.get("search_mean_speed_mps"),
        Some(&CellValue::Float(2.46))
    );
    assert_eq!(row.get("search_tortuosity"), Some(&CellValue::Float(1.25)));
    assert_eq!(
        row.get("enclosure_summary.zone_name"),
        Some(&CellValue::String("board".into()))
    );
    assert_eq!(
        row.get("reward_summaries[0].zone_name"),
        Some(&CellValue::String("reward_a".into()))
    );
    assert_eq!(
        row.get("reward_summaries[1].duration_seconds"),
        Some(&CellValue::Float(18.7))
    );

    // Legacy nested object/array flattening still works alongside Cheeseboard fields.
    assert_eq!(
        row.get("reward.errors"),
        None,
        "old placeholder reward subtree is no longer part of this fixture"
    );

    // Array of objects -> indexed keys
    assert_eq!(
        row.get("zone_summaries[0].zone_name"),
        Some(&CellValue::String("target".into()))
    );
    assert_eq!(
        row.get("zone_summaries[0].entries"),
        Some(&CellValue::Int(1))
    );
    assert_eq!(
        row.get("zone_summaries[1].zone_name"),
        Some(&CellValue::String("arena".into()))
    );
    assert_eq!(
        row.get("zone_summaries[1].duration_seconds"),
        Some(&CellValue::Float(238.0))
    );

    // Trailing identifiers
    assert_eq!(
        row.get("experiment"),
        Some(&CellValue::String("555_seq1_cheeseboard".into()))
    );
    assert_eq!(row.get("excluded"), Some(&CellValue::Bool(false)));
}

// -- CheeseboardExperiment ---------------------------------------

#[test]
fn ymaze_zone_entries_are_not_overwritten_by_crossing_sequence_counts() {
    let mut trial = parse(&ymaze_fixture_toml(true));
    let zone = trial.detailed["zone_summaries"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|zone| zone["zone_name"] == "in_arm_A")
        .unwrap();
    zone["entries"] = serde_json::json!(5);
    let row = YMazeExperiment.extract_row(&trial);
    // Criteria #25 zone entries and #1 raw crossing counts are separate cells.
    assert_eq!(row.get("arm_A_entries"), Some(&CellValue::Int(5)));
    assert_eq!(row.get("raw_arm_A_entries"), Some(&CellValue::Int(6)));
}
