#[test]
fn tied_spearman_matches_scipy_and_reports_pairwise_missingness() {
    let fixture = fixture();
    let expected = &fixture["spearman"];
    let mut rows: Vec<_> = expected["x"].as_array().unwrap().iter().enumerate().map(|(i, x)| json!({
        "animal_id":format!("a{i}"),"session_id":format!("s{i}"),"spontaneous_alternation_pct":x,"valid_total_arm_entries":expected["y"][i]
    })).collect();
    rows.push(json!({"animal_id":"missing","session_id":"smissing","spontaneous_alternation_pct":null,"valid_total_arm_entries":10}));
    let tests = spearman_pairs(&rows, Some("allow_unverified")).unwrap();
    close(&tests[0]["rho"], &expected["rho"], 1e-12);
    close(&tests[0]["p_raw"], &expected["p_raw"], 1e-10);
    assert_eq!(tests[0]["n_animals"], 7);
    assert_eq!(
        tests[0]["subset"][7]["reason"],
        "model_specific_missingness"
    );
    assert_eq!(tests[1]["n_animals"], 0);
    assert_eq!(tests[1]["reason"], "insufficient_pairs");
}

#[test]
fn pairing_preserves_identity_and_verification_of_both_timepoints() {
    let mut rows = vec![
        json!({"animal_id":"a","assay":"ymaze","phase":"session","session_id":"post-a","timepoint":"post","spontaneous_alternation_pct":75,"input_verification":{"status":"verified_clarified_entry_definition"}}),
        json!({"animal_id":"b","assay":"ymaze","phase":"session","session_id":"pre-b","timepoint":"pre","spontaneous_alternation_pct":99}),
        json!({"animal_id":"a","assay":"ymaze","phase":"session","session_id":"pre-a","timepoint":"pre","spontaneous_alternation_pct":60,"input_verification":{"status":"unresolved"}}),
    ];
    let (paired, ledger) = pairing(
        &rows,
        "spontaneous_alternation_pct",
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(paired[0]["DeltaSA"], 15.0);
    assert!(
        ledger
            .iter()
            .any(|row| row["reason"] == "unmatched_prepost")
    );
    let (paired, ledger) = pairing(
        &rows,
        "spontaneous_alternation_pct",
        Some("require_verified"),
    )
    .unwrap();
    assert!(paired.is_empty());
    assert!(
        ledger
            .iter()
            .any(|row| row["reason"] == "upstream_unverified:pre:unresolved")
    );
    rows.push(rows[2].clone());
    assert!(
        pairing(
            &rows,
            "spontaneous_alternation_pct",
            Some("allow_unverified")
        )
        .is_err()
    );
}

#[test]
fn one_sample_checks_match_scipy_and_preserve_distinct_chance_references() {
    let fixture = fixture();
    let oracle = &fixture["within_group"];
    let rows: Vec<_> = oracle["values"].as_array().unwrap().iter().enumerate().map(|(index, value)| json!({
        "animal_id": format!("SYNTHETIC_{index}"), "session_id": format!("s{index}"), "group": "G5",
        "DI": value, "percent_novel": 50.0 + value.as_f64().unwrap() * 50.0
    })).collect();
    let tests = within_group(
        &rows,
        &["DI".into(), "percent_novel".into()],
        0.95,
        Some("allow_unverified"),
        &["G5".into(), "G6".into()],
    )
    .unwrap();
    assert_eq!(tests.len(), 4);
    assert_eq!(tests[0]["null"], 0.0);
    assert_eq!(tests[1]["null"], 50.0);
    for (index, factor) in [(0, 1.0), (1, 50.0)] {
        let test = &tests[index];
        assert_eq!(test["df"], oracle["df"]);
        assert_eq!(test["status"], "computed");
        for field in ["estimate", "standard_error", "ci_lower", "ci_upper"] {
            close(
                &json!(test[field].as_f64().unwrap() / factor),
                &oracle[field],
                1e-12,
            );
        }
        close(&test["statistic"], &oracle["statistic"], 1e-12);
        probability_close(&test["p_raw"], &oracle["p_raw"]);
    }
    assert!(
        tests[2..]
            .iter()
            .all(|test| test["n_animals"] == 0 && test["p_raw"].is_null())
    );
}

#[test]
fn sensitivity_pairs_remain_separate_and_non_sa_endpoints_do_not_emit_deltasa() {
    let mut rows = Vec::new();
    for threshold in [25, 45] {
        for timepoint in ["pre", "post"] {
            rows.push(json!({"animal_id":"a", "session_id":format!("{threshold}_{timepoint}"), "assay":"ymaze",
                "phase":"session", "timepoint":timepoint, "threshold_pct":threshold,
                "spontaneous_alternation_pct":if timepoint == "pre" { 50.0 } else { 75.0 },
                "total_distance_m":if timepoint == "pre" { 10.0 } else { 12.0 }}));
        }
    }
    let (paired, ledger) = pairing(
        &rows,
        "spontaneous_alternation_pct",
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(paired.len(), 2);
    assert_eq!(ledger.len(), 2);
    assert!(paired.iter().all(|row| row["DeltaSA"] == 25.0));
    assert_ne!(paired[0]["threshold_pct"], paired[1]["threshold_pct"]);
    assert!(
        analysis_subset(
            &paired,
            &["spontaneous_alternation_pct"],
            Some("allow_unverified")
        )
        .is_err()
    );
    let (paired, _) = pairing(&rows, "total_distance_m", Some("allow_unverified")).unwrap();
    assert!(paired.iter().all(|row| row["paired_difference"] == 2.0));
    assert!(paired.iter().all(|row| row["DeltaSA"].is_null()));
}
