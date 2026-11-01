//! Cross-line (imen_arm_start_crossline_v1) reconstruction against the shared
//! synthetic fixture also asserted by the native task tests.
use bekiper_feature::tolerance::TemporalPolicy;
use bekiper_post_analysis::events;
use serde_json::{Value, json};

#[test]
fn crossline_reconstruction_matches_shared_native_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../bekiper-behaviour/src/radial_arm/tests/golden/crossline_entry_parity.json"
    ))
    .unwrap();
    assert_eq!(fixture["geometry_version"], events::GEOMETRY_ARM_START);
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
                        .map(|(c, v)| (c.as_str().unwrap().into(), v.clone()))
                        .collect(),
                )
            })
            .collect();
        for threshold in case["thresholds"].as_array().unwrap() {
            let pct = threshold.as_u64().unwrap() as u8;
            let result = events::ymaze_coordinate_entries(&samples, &geometry, pct).unwrap();
            let expected = &case["expected"][pct.to_string()];
            assert_eq!(
                result["valid_arm_entry_sequence"], expected["sequence"],
                "{pct}"
            );
            assert_eq!(result["geometry_version"], events::GEOMETRY_ARM_START);
            for arm in ["A", "B", "C"] {
                let frames: Vec<_> = result["event_debug"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|r| r["arm"] == arm && r["entry"] == true)
                    .map(|r| r["sample_index"].clone())
                    .collect();
                assert_eq!(json!(frames), expected["entry_frames"][arm], "{pct} {arm}");
            }
        }
    }
    // The radial rule keeps rejecting 0%.
    let radial = json!({"version":events::GEOMETRY,"canonical_vertices":true,"center":fixture["center"],"arms":fixture["arms"]});
    assert!(events::ymaze_coordinate_entries(&[], &radial, 0).is_err());
}

#[test]
fn crossline_zone_gap_is_policy_only_when_it_matches_the_registered_value() {
    let row = |gap: f64| {
        json!({"threshold_pct":25,"upstream_revision":"r","scoring_provenance":{
            "geometry_version":events::GEOMETRY_ARM_START,"apply_dwell_filter":false,
            "minimum_dwell_seconds":0.0,"requires_both_landmarks_same_arm":true,
            "arm_membership_hysteresis":false,"zone_max_gap_seconds":gap,"pose_gap_fill_seconds":0.0,
            "proximal_threshold":0.25,"entry_rule":events::ENTRY_RULE,
            "head_columns":{"x":"hx","y":"hy"},"tail_columns":{"x":"tx","y":"ty"}}})
    };
    let roles = json!({"head_center":{"x":"hx","y":"hy"},"tail_base":{"x":"tx","y":"ty"}});
    let policy = TemporalPolicy::default();
    let check = |gap, registered| {
        events::ymaze_provenance_registered(
            &row(gap),
            &roles,
            &policy,
            events::GEOMETRY_ARM_START,
            registered,
        )
        .unwrap()["status"]
            .clone()
    };
    assert_eq!(check(1.0, 1.0), "verified_clarified_entry_definition");
    assert_eq!(check(1.0, 0.0), "requires_event_set_audit");
    // Radial semantics unchanged: any gap is a mitigation.
    assert_eq!(
        events::ymaze_provenance_with_policy(&row(1.0), &roles, &policy).unwrap()["status"],
        "requires_regeneration_or_registered_amendment"
    );
}

#[test]
fn zero_percent_is_a_primary_context_only_under_crossline_geometry() {
    use bekiper_post_analysis::contracts::RunOptions;
    use bekiper_post_analysis::inputs::{allowed_thresholds, resolve_primary_runs};
    let plan = |geometry: &str| {
        json!({"assays":{"ymaze":{"threshold_geometry":geometry}},
            "execution":{"ymaze_primary_thresholds":[0,25,45,65]}})
    };
    let mut cross = plan(events::GEOMETRY_ARM_START);
    assert_eq!(allowed_thresholds(&cross), [0, 25, 45, 65]);
    resolve_primary_runs(&mut cross, &RunOptions::default()).unwrap();
    let mut radial = plan(events::GEOMETRY);
    assert_eq!(allowed_thresholds(&radial), [25, 45, 65]);
    assert!(resolve_primary_runs(&mut radial, &RunOptions::default()).is_err());
    radial["execution"]["ymaze_primary_thresholds"] = json!([25, 45, 65]);
    resolve_primary_runs(&mut radial, &RunOptions::default()).unwrap();
}

/// rv Y1: native bridges same-arm visits separated by <= zone_max_gap_seconds
/// (1.0 s registered for the 25/45/65 contexts, whose dwell policy has gap 0).
/// Reconstruction must match native: A held, 0.5 s out, A again -> one entry.
#[test]
fn rv_y1_reconstruction_bridges_registered_zone_gap_like_native() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../bekiper-behaviour/src/radial_arm/tests/golden/crossline_entry_parity.json"
    ))
    .unwrap();
    let geometry = json!({"version":events::GEOMETRY_ARM_START,"canonical_vertices":true,"center":fixture["center"],"arms":fixture["arms"]});
    let sample = |i: usize, x: f64| {
        json!({"timestamp_s": i as f64 * 0.1, "head_center_x": x, "head_center_y": 0.0,
               "tail_base_x": x, "tail_base_y": 0.0})
    };
    // 0.3 s in arm A (x=0.9), 0.5 s in centre (x=0), 0.3 s in arm A.
    let xs = [0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9];
    let samples: Vec<Value> = xs.iter().enumerate().map(|(i, x)| sample(i, *x)).collect();
    // The 25/45/65 contexts' registered policy: zero dwell, policy gap 0.
    let result = events::ymaze_coordinate_entries_with_zone_gap(
        &samples,
        &geometry,
        25,
        &TemporalPolicy::default(),
        None,
        1.0,
    )
    .unwrap();
    assert_eq!(
        result["valid_arm_entry_sequence"],
        json!(["A"]),
        "registered zone_max_gap_seconds=1.0 bridges the 0.5 s excursion"
    );
}

/// rv Y2: a dwell override must reach every registered per-threshold policy,
/// which cross-line scoring prefers over the base `temporal_policy`.
#[test]
fn rv_y2_dwell_override_reaches_threshold_policies() {
    let mut config = json!({"assays":{"ymaze":{"threshold_geometry":events::GEOMETRY_ARM_START,
        "temporal_policy":{"min_duration_seconds":1.0},
        "threshold_policies":{"0":{"min_duration_seconds":1.0},"25":{"min_duration_seconds":1.0}}}}});
    let overrides = bekiper_feature::tolerance::DwellOverrides {
        seconds: Some(0.0),
        ..Default::default()
    };
    bekiper_post_analysis::inputs::resolve_ymaze_dwell(&mut config, overrides).unwrap();
    let ymaze = &config["assays"]["ymaze"];
    assert_eq!(ymaze["temporal_policy"]["min_duration_seconds"], 0.0);
    for pct in ["0", "25"] {
        assert_eq!(
            ymaze["threshold_policies"][pct]["min_duration_seconds"],
            0.0
        );
    }
}

#[test]
fn contexts_and_render_workers_options_are_validated_into_the_execution_block() {
    use bekiper_post_analysis::contracts::{ContextSpec, RunOptions};
    use bekiper_post_analysis::inputs::resolve_primary_runs;
    let context = |assay: &str, variant: &str| ContextSpec {
        assay: assay.into(),
        variant: variant.into(),
    };
    let plan = || json!({"assays":{"ymaze":{"threshold_geometry":events::GEOMETRY_ARM_START}},"execution":{}});
    let mut config = plan();
    resolve_primary_runs(
        &mut config,
        &RunOptions {
            contexts: Some(vec![
                context("ymaze", "arm45"),
                context("nort", "nose_only"),
            ]),
            render_workers: Some(3),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(config["execution"]["render_workers"], 3);
    assert_eq!(config["execution"]["contexts"][1]["variant"], "nose_only");
    let mut nameless = plan();
    let error = resolve_primary_runs(
        &mut nameless,
        &RunOptions {
            contexts: Some(vec![context("ymaze", "primary")]),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("must name its threshold"),
        "{error}"
    );
}

#[test]
fn nort_provenance_expects_the_plan_registered_reach() {
    let matches = |config: Value| {
        let row = json!({"assay": "nort", "variant": "whisker20mm",
            "scoring_provenance": {"zone_extension_m": 0.02},
            "_rule_audit": {"rules": {"zone_extension_m": 0.02}}});
        events::assay_provenance(&row, &config).unwrap()["checks"]["zone_extension_m"].clone()
    };
    assert_eq!(matches(json!({"assays": {"nort": {}}})), false);
    assert_eq!(
        matches(json!({"assays": {"nort": {"zone_extension_m": 0.02}}})),
        true
    );
}
