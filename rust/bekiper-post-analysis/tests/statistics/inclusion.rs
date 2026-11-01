#[test]
fn unsupported_model_censoring_and_missing_contrast_reference_are_explicit() {
    let fixture = fixture();
    let rows = fixture["rows"].as_array().unwrap();
    let mut settings = fixture["settings"].clone();
    settings["model_family"] = "mixed_model".into();
    let model = fit_factorial(
        rows,
        "DI",
        &fixture["levels"],
        &fixture["groups"],
        &settings,
        false,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(
        model["reason"],
        "unsupported_registered_model_or_covariance"
    );
    let model = fit_factorial(
        rows,
        "DI",
        &fixture["levels"],
        &fixture["groups"],
        &fixture["settings"],
        true,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(
        model["reason"],
        "blocked:D09:contrast_covariate_reference_missing"
    );
    let censored: Vec<_> = rows
        .iter()
        .cloned()
        .map(|mut row| {
            row["latency_to_first_center_entry_s"] = row["DI"].clone();
            row["pre_latency_right_censored"] = true.into();
            row
        })
        .collect();
    let model = fit_factorial(
        &censored,
        "latency_to_first_center_entry_s",
        &fixture["levels"],
        &fixture["groups"],
        &fixture["settings"],
        true,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(
        model["reason"],
        "D09:censored_latency_requires_registered_censoring_model"
    );
}

#[test]
fn duplicate_animal_sessions_cannot_enter_models_or_qc_checks() {
    let rows = vec![
        json!({"animal_id":"a","session_id":"s1","group":"G5","DI":0.1,"spontaneous_alternation_pct":50,"valid_total_arm_entries":8}),
        json!({"animal_id":"a","session_id":"s2","group":"G5","DI":0.2,"spontaneous_alternation_pct":60,"valid_total_arm_entries":9}),
    ];
    assert!(
        within_group(
            &rows,
            &["DI".into()],
            0.95,
            Some("allow_unverified"),
            &["G5".into()]
        )
        .is_err()
    );
    assert!(spearman_pairs(&rows, Some("allow_unverified")).is_err());
}

#[test]
fn inclusion_ledger_keeps_qc_exclusions_and_endpoint_missingness_distinct() {
    let rows = vec![
        json!({"animal_id":"flagged", "session_id":"s1", "DI":0.3, "qc_flags":["low_tracking"]}),
        json!({"animal_id":"excluded", "session_id":"s2", "DI":0.4, "excluded":true, "exclusion_reason":"registered_metadata", "qc_flags":["low_tracking"]}),
        json!({"animal_id":"missing", "session_id":"s3", "DI":null, "missing_reasons":{"DI":"zero_exploration"}}),
    ];
    let (used, ledger) = analysis_subset(&rows, &["DI"], Some("allow_unverified")).unwrap();
    assert_eq!(used.len(), 1);
    assert_eq!(used[0]["animal_id"], "flagged");
    assert_eq!(ledger[0]["qc_flags"], json!(["low_tracking"]));
    assert!(ledger[0]["reason"].is_null());
    assert_eq!(ledger[1]["reason"], "excluded:registered_metadata");
    assert_eq!(ledger[1]["exclusion_reason"], "registered_metadata");
    assert_eq!(ledger[2]["reason"], "model_specific_missingness");
    assert_eq!(ledger[2]["missing_fields"], json!(["DI"]));
    assert_eq!(ledger[2]["missing_reasons"]["DI"], "zero_exploration");
    let (used, ledger) = analysis_subset(&rows, &["DI"], None).unwrap();
    assert!(used.is_empty());
    assert_eq!(
        ledger[0]["reason"],
        "blocked:D12:upstream_verification_requirement"
    );
}

#[test]
fn invalid_censoring_indicators_are_not_treated_as_uncensored() {
    let fixture = fixture();
    let mut rows = fixture["rows"].as_array().unwrap().clone();
    for row in &mut rows {
        row["latency_to_first_center_entry_s"] = row["DI"].clone();
        row["latency_right_censored"] = "false".into();
    }
    assert!(
        fit_factorial(
            &rows,
            "latency_to_first_center_entry_s",
            &fixture["levels"],
            &fixture["groups"],
            &fixture["settings"],
            false,
            Some("allow_unverified")
        )
        .is_err()
    );
}
