#[test]
fn oft_fragments_coalesce_but_tracking_gaps_break_dwell() {
    let continuous = [
        json!({"zone":"center","start_s":4,"end_s":4.6}),
        json!({"zone":"center","start_s":4.6,"end_s":6}),
    ];
    let result = events::oft_zone_intervals(&continuous, 4.0, "qualified_only", "missing").unwrap();
    assert_eq!(result["center_entries"], 1);
    assert_eq!(result["center_time_s"], 2.0);
    assert_eq!(result["latency_to_first_center_entry_s"], 0.0);
    assert_eq!(result["event_debug"][0]["component_indices"], json!([0, 1]));
    let gap = [
        json!({"zone":"center","start_s":0,"end_s":0.6}),
        json!({"zone":"center","start_s":0.7,"end_s":1.3}),
    ];
    assert_eq!(
        events::oft_zone_intervals(&gap, 0.0, "qualified_only", "missing").unwrap()["center_entries"],
        0
    );
    let empty = events::oft_zone_intervals(&[], 0.0, "all_observed", "right_censored_300").unwrap();
    assert_eq!(empty["latency_to_first_center_entry_s"], 300.0);
    assert_eq!(empty["latency_right_censored"], true);
}

#[test]
fn oft_analysed_duration_denominator_and_nort_zero_rules_are_preserved() {
    // Percent of the analysed seconds (owner decision 2026-10-22), capped at
    // the 300 s window; 300 s when the duration is unknown.
    for (duration, expected) in [
        (json!(150), 50.0),
        (json!(299.5), 75.0 / 2.995),
        (json!(400), 25.0),
        (json!(null), 25.0),
    ] {
        assert_eq!(
            metrics::oft_scores(&json!({"center_time_s":75,"analysis_duration_s":duration}))
                .unwrap()["center_time_pct"],
            expected
        );
    }
    let empty = metrics::nort_scores(None, None, Some(0.0), Some(0.0)).unwrap();
    for field in ["DI", "RI", "percent_novel", "LR"] {
        assert!(empty[field].is_null());
    }
    let zero = metrics::nort_scores(None, None, Some(0.0), Some(1.0)).unwrap();
    assert_eq!(zero["DI"], -1.0);
    assert!(zero["LR"].is_null());
    let train = metrics::nort_scores(Some(3.0), Some(1.0), None, None).unwrap();
    assert_eq!(train["object_bias"], 0.5);
    assert_eq!(train["side_bias"], -0.5);
}

#[test]
fn unresolved_oft_interval_policies_withhold_dependent_percentage() {
    let mut input = row("oft");
    input["center_intervals"] = json!([{"zone":"center","start_s":0,"end_s":10}]);
    input["first_detection_s"] = json!(0);
    input["center_time_s"] = json!(12);
    input["center_time_pct"] = json!(4);
    let (rows, _) = metrics::score_records(&[input], &config()).unwrap();
    assert!(rows[0]["center_time_s"].is_null());
    assert!(rows[0]["center_time_pct"].is_null());
    assert_eq!(
        rows[0]["missing_reasons"]["center_time_pct"],
        "blocked:D04/D05:interval_policy"
    );
    assert_eq!(rows[0]["upstream_center_time_pct"], 4);
}

#[test]
fn oft_reconstruction_respects_short_recordings_and_clears_stale_censoring() {
    let mut input = row("oft");
    input["first_detection_s"] = json!(4);
    input["analysis_duration_s"] = json!(150);
    input["latency_right_censored"] = json!(true);
    input["center_intervals"] = json!([{"zone":"center","start_s":4,"end_s":6}]);
    let (rows, debug) = metrics::score_records(&[input.clone()], &oft_interval_config()).unwrap();
    assert_eq!(rows[0]["latency_right_censored"], false);
    assert_eq!(rows[0]["upstream_latency_right_censored"], true);
    assert_eq!(debug[0]["animal_id"], input["animal_id"]);
    assert!(
        rows[0]["qc_flags"]["incomplete_recording"]
            .as_bool()
            .unwrap()
    );
    input["center_intervals"] = json!([{"zone":"center","start_s":154,"end_s":155}]);
    assert!(metrics::score_records(&[input], &oft_interval_config()).is_err());
}
