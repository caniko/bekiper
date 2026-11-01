#[test]
fn planned_empty_groups_stay_in_holm_family_and_stale_adjustments_are_cleared() {
    let rows = vec![
        json!({"animal_id":"a","session_id":"s1","group":"G5","DI":0.1}),
        json!({"animal_id":"b","session_id":"s2","group":"G5","DI":0.2}),
        json!({"animal_id":"c","session_id":"s3","group":"G5","DI":0.3}),
    ];
    let groups = vec!["G5".into(), "G6".into()];
    let mut tests = within_group(
        &rows,
        &["DI".into()],
        0.95,
        Some("allow_unverified"),
        &groups,
    )
    .unwrap();
    assert_eq!(tests.len(), 2);
    assert_eq!(tests[1]["group"], "G6");
    assert_eq!(tests[1]["n_animals"], 0);
    assert_eq!(tests[1]["reason"], "insufficient_data");
    tests[1]["p_adjusted"] = json!(0.01);
    adjust_family(&mut tests, "holm", "registered_two_groups").unwrap();
    assert_eq!(tests[0]["family_size"], 2);
    close(
        &tests[0]["p_adjusted"],
        &json!((tests[0]["p_raw"].as_f64().unwrap() * 2.0).min(1.0)),
        1e-12,
    );
    assert!(tests[1]["p_adjusted"].is_null());
    let empty = within_group(
        &[],
        &["DI".into(), "percent_novel".into()],
        0.95,
        Some("allow_unverified"),
        &groups,
    )
    .unwrap();
    assert_eq!(empty.len(), 4);
    assert!(empty.iter().all(|test| test["p_raw"].is_null()));
}

#[test]
fn holm_counts_null_attempts_and_validates_p_values() {
    let mut tests = vec![
        json!({"p_raw":0.04}),
        json!({"p_raw":null}),
        json!({"p_raw":0.01}),
        json!({"p_raw":0.03}),
    ];
    adjust_family(&mut tests, "holm", "registered_family").unwrap();
    close(&tests[0]["p_adjusted"], &json!(0.09), 1e-12);
    close(&tests[2]["p_adjusted"], &json!(0.04), 1e-12);
    close(&tests[3]["p_adjusted"], &json!(0.09), 1e-12);
    assert!(tests.iter().all(|test| test["family_size"] == 4));
    assert!(adjust_family(&mut [json!({"p_raw":-0.1})], "holm", "family").is_err());
    assert!(adjust_family(&mut [json!({"p_raw":0.0})], "holm", "family").is_err());
}
