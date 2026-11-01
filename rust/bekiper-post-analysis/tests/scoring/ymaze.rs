#[test]
fn count_domain_is_checked_even_without_a_sequence_or_relationship_total() {
    for field in [
        "raw_total_arm_entries",
        "valid_total_arm_entries",
        "raw_arm_A_entries",
        "valid_arm_C_entries",
        "dwell_rejected_entries",
        "total_triads",
        "usable_triads",
        "correct_alternations",
        "incorrect_alternations",
        "neutral_triads",
    ] {
        for value in [
            json!(-1),
            json!(0.5),
            json!(true),
            json!("0"),
            json!(9_007_199_254_740_992_u64),
        ] {
            let mut supplied = json!({"assay":"ymaze"});
            supplied[field] = value;
            let error = metrics::validate_supplied(&supplied)
                .unwrap_err()
                .to_string();
            assert!(error.starts_with(field), "{field}: {error}");
        }
        for value in [Value::Null, json!(0)] {
            let mut supplied = json!({"assay":"ymaze"});
            supplied[field] = value;
            metrics::validate_supplied(&supplied).unwrap();
        }
    }
}

#[test]
fn ymaze_entry_differences_are_signed_integers_with_explicit_identity() {
    for field in ["raw_to_scored_entry_difference", "removed_short_entries"] {
        for value in [
            json!(0.5),
            json!(true),
            json!("-1"),
            json!(-9_007_199_254_740_992_i64),
        ] {
            let mut supplied = json!({"assay":"ymaze"});
            supplied[field] = value;
            assert!(metrics::validate_supplied(&supplied).is_err(), "{supplied}");
        }
        for value in [Value::Null, json!(-1), json!(0), json!(1)] {
            let mut supplied = json!({"assay":"ymaze"});
            supplied[field] = value;
            metrics::validate_supplied(&supplied).unwrap();
        }
    }
    let signed = json!({"assay":"ymaze","raw_total_arm_entries":3,"valid_total_arm_entries":4,
        "raw_to_scored_entry_difference":-1,"removed_short_entries":-1,"dwell_rejected_entries":0});
    metrics::validate_supplied(&signed).unwrap();
    let mut contradictory = signed.clone();
    contradictory["raw_to_scored_entry_difference"] = json!(0);
    assert!(metrics::validate_supplied(&contradictory).is_err());
    contradictory = signed;
    contradictory["dwell_rejected_entries"] = json!(4);
    assert!(metrics::validate_supplied(&contradictory).is_err());
}

#[test]
fn historical_ymaze_difference_does_not_establish_a_dwell_rejection_reason() {
    let mut input = row("ymaze");
    input["threshold_pct"] = json!(45);
    input["raw_arm_entry_sequence"] = json!("ABCABC");
    input["valid_arm_entry_sequence"] = json!("ABC");
    input["removed_short_entries"] = json!(2);
    let (rows, _) = metrics::score_records(&[input.clone()], &config()).unwrap();
    assert_eq!(rows[0]["raw_to_scored_entry_difference"], 3);
    assert_eq!(rows[0]["removed_short_entries"], 2);
    assert!(rows[0]["dwell_rejected_entries"].is_null());
    assert_eq!(
        rows[0]["missing_reasons"]["dwell_rejected_entries"],
        "not_measured:dwell_bout_evidence"
    );
    assert_eq!(rows[0]["qc_flags"]["removal_reason_unverified"], true);

    input["valid_arm_entry_sequence"] = json!("ABCABCABC");
    input["removed_short_entries"] = json!(-3);
    let (rows, _) = metrics::score_records(&[input], &config()).unwrap();
    assert_eq!(rows[0]["raw_to_scored_entry_difference"], -3);
    assert!(rows[0]["dwell_rejected_entries"].is_null());
}

#[test]
fn all_27_source_table_triads_and_sequence_goldens_are_conditional_on_policy() {
    let cases: Value = serde_json::from_str(include_str!(
        "../../../../behavioral_post_analysis_harness/tests/golden_cases.json"
    ))
    .unwrap();
    assert_eq!(cases["ymaze_triplets"].as_array().unwrap().len(), 27);
    for case in cases["ymaze_triplets"].as_array().unwrap() {
        let result =
            metrics::ymaze_scores(&case["triplet"], Some("source_table"), None, None).unwrap();
        assert_eq!(result["window_labels"], json!([case["expected"]]));
    }
    for group in ["ymaze_sequences", "ymaze_short_sequences"] {
        for case in cases[group].as_array().unwrap() {
            let result = metrics::ymaze_scores(
                &case["sequence"],
                Some("source_table"),
                Some("zero_windows"),
                None,
            )
            .unwrap();
            for (key, expected) in case["expected"].as_object().unwrap() {
                if let Some(value) = expected.as_f64() {
                    assert!(
                        (result[key].as_f64().unwrap() - value).abs() < 1e-10,
                        "{key}"
                    );
                } else {
                    assert_eq!(&result[key], expected, "{key}");
                }
            }
        }
    }
    let blocked = metrics::ymaze_scores(&json!("ABCABC"), None, None, None).unwrap();
    assert_eq!(
        blocked["missing_reasons"]["spontaneous_alternation_pct"],
        "blocked:D01"
    );
}

#[test]
fn principal_entry_reconstruction_matches_shared_native_golden_without_dwell() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../bekiper-behaviour/src/radial_arm/tests/golden/principal_entry_parity.json"
    ))
    .unwrap();
    let geometry = json!({"version":fixture["geometry_version"],"canonical_vertices":true,"center":fixture["center"],"arms":fixture["arms"]});
    for case in fixture["cases"].as_array().unwrap() {
        let samples: Vec<Value> = case["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|values| {
                Value::Object(
                    fixture["sample_columns"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(values.as_array().unwrap())
                        .map(|(column, value)| (column.as_str().unwrap().into(), value.clone()))
                        .collect(),
                )
            })
            .collect();
        for threshold in case["thresholds"].as_array().unwrap() {
            let threshold = threshold.as_u64().unwrap() as u8;
            let result = events::ymaze_coordinate_entries(&samples, &geometry, threshold).unwrap();
            let expected = &case["expected"][threshold.to_string()];
            assert_eq!(
                result["valid_arm_entry_sequence"], expected["sequence"],
                "{}",
                case["name"]
            );
            for arm in ["A", "B", "C"] {
                let frames: Vec<_> = result["event_debug"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|r| r["arm"] == arm && r["entry"] == true)
                    .map(|r| r["sample_index"].clone())
                    .collect();
                assert_eq!(json!(frames), expected["entry_frames"][arm]);
            }
            if expected["arm_A_predicate"].is_array() {
                let states: Vec<_> = result["event_debug"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|r| r["arm"] == "A")
                    .map(|r| r["predicate"].clone())
                    .collect();
                assert_eq!(json!(states), expected["arm_A_predicate"]);
            }
        }
    }
}

#[test]
fn historical_ymaze_mitigation_requires_audit_and_dwell_requires_amendment() {
    let no_dwell = events::ymaze_provenance_with_policy(
        &json!({"scoring_provenance":{"apply_dwell_filter":false,"minimum_dwell_seconds":0}}),
        &json!({}),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(no_dwell["status"], "unresolved");
    assert!(
        !no_dwell["reasons"]
            .as_array()
            .unwrap()
            .contains(&json!("no_dwell"))
    );
    let mitigation = events::ymaze_provenance_with_policy(
        &json!({"scoring_provenance":{"apply_dwell_filter":false,"zone_max_gap_seconds":1}}),
        &json!({}),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(mitigation["status"], "requires_event_set_audit");
    let dwell = events::ymaze_provenance_with_policy(
        &json!({"scoring_provenance":{"apply_dwell_filter":true}}),
        &json!({}),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(
        dwell["status"],
        "requires_regeneration_or_registered_amendment"
    );
}

#[test]
fn absent_sequence_does_not_let_triad_counts_bypass_d01() {
    let mut input = row("ymaze");
    input["threshold_pct"] = json!(45);
    input["valid_total_arm_entries"] = json!(8);
    input["correct_alternations"] = json!(4);
    input["incorrect_alternations"] = json!(1);
    input["neutral_triads"] = json!(1);
    input["total_triads"] = json!(6);
    input["usable_triads"] = json!(5);
    let mut config = config();
    config["decision_resolutions"]
        .as_object_mut()
        .unwrap()
        .remove("D01");
    let (rows, _) = metrics::score_records(&[input], &config).unwrap();
    for field in [
        "correct_alternations",
        "incorrect_alternations",
        "neutral_triads",
        "usable_triads",
        "low_quality",
    ] {
        assert!(rows[0][field].is_null(), "unblocked {field}");
        assert_eq!(rows[0]["missing_reasons"][field], "blocked:D01");
    }
    assert_eq!(rows[0]["valid_total_arm_entries"], 8);
    assert_eq!(rows[0]["total_triads"], 6);
    assert_eq!(rows[0]["upstream_correct_alternations"], 4);
}

#[test]
fn sequence_representation_changes_are_not_event_set_disagreements() {
    let mut input = row("ymaze");
    input["threshold_pct"] = json!(45);
    input["valid_arm_entry_sequence"] = json!("ABCABCAB");
    let saved = input.clone();
    let (rows, _) = metrics::score_records(&[input.clone()], &config()).unwrap();
    assert_eq!(input, saved);
    assert!(rows[0]["disagreements"].is_null());
    assert_eq!(
        rows[0]["valid_arm_entry_sequence"],
        json!(["A", "B", "C", "A", "B", "C", "A", "B"])
    );
}

#[test]
fn unresolved_short_sequence_total_has_an_explicit_missing_reason() {
    let result = metrics::ymaze_scores(&json!("A"), None, None, None).unwrap();
    assert!(result["total_triads"].is_null());
    assert_eq!(
        result["missing_reasons"]["total_triads"],
        "blocked:D02:insufficient_entries"
    );
}
