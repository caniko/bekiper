use bekiper_feature::heuristic::combined::exploration::{
    BoutFilterMode, BoutFilterSpec, IMEN_MAX_GAP_SECONDS, IMEN_MIN_BOUT_SECONDS,
    object_exploration_with_bout_filters,
};
use bekiper_post_analysis::events;
use polars::prelude::*;
use serde_json::{Value, json};

fn native(raw: &[bool], angles: &[f64], contacts: &[bool]) -> (f64, u32) {
    let df = df! { "exploring_A_raw" => raw }.unwrap();
    let angles = Float64Chunked::from_slice("angles".into(), angles);
    let contacts = BooleanChunked::from_slice("contacts".into(), contacts);
    let (seconds, bouts, _) = object_exploration_with_bout_filters(
        &df,
        "exploring_A_raw",
        10.0,
        IMEN_MIN_BOUT_SECONDS,
        IMEN_MAX_GAP_SECONDS,
        0.0,
        BoutFilterSpec {
            strict_boundaries: false,
            speed_mode: BoutFilterMode::Disabled,
            angle_mode: BoutFilterMode::PerBoutMedian,
            max_nose_speed: 0.05,
            max_angle_deg: 45.0,
            nose_speed_mps: None,
            angle_deg: Some(&angles),
            nose_contact: Some(&contacts),
        },
    );
    (seconds, bouts)
}

fn candidate(id: &str, start: f64, end: f64, angles: &[f64], contacts: &[bool]) -> Value {
    json!({"segment_id":id,"object_id":"A","start_s":start,"end_s":end,
        "candidate_eligible":true,"angles_degrees":angles,"nose_contacts":contacts})
}

#[test]
fn imen_config_and_plan_share_inclusive_nort_policy_without_speed_exclusion() {
    use bekiper_feature::heuristic::combined::exploration::NortBoutPolicy;
    let config: Value = toml::from_str(include_str!(
        "../../../examples/neo/imen/bekiper_config.toml"
    ))
    .unwrap();
    let plan: Value = toml::from_str(include_str!(
        "../../../examples/neo/imen/post_analysis_plan.toml"
    ))
    .unwrap();
    let exploration = &config["ingress_profile"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|profile| profile["tasks"].as_array().unwrap())
        .find(|task| task["task_type"] == "object_recognition")
        .unwrap()["task_params"]["exploration"];
    let native_policy: NortBoutPolicy = serde_json::from_value(json!({
        "min_bout_seconds": exploration["min_bout_seconds"],
        "max_gap_seconds": exploration["max_gap_seconds"],
        "max_nose_speed": exploration["max_nose_speed"],
        "max_angle_deg": exploration["max_angle_deg"],
        "speed_filter_mode": exploration["speed_filter_mode"],
        "strict_boundaries": exploration["strict_boundaries"],
        "whisker_enabled": exploration["whisker"]["enabled"],
    }))
    .unwrap();
    assert!(native_policy.whisker_enabled);
    let standalone_policy: NortBoutPolicy =
        serde_json::from_value(plan["assays"]["nort"]["bout_policy"].clone()).unwrap();
    assert_eq!(native_policy, standalone_policy);
    // 20 fps puts the 0.25 s minimum bout and the 1.0 s gap on frame boundaries.
    const FPS: f64 = 20.0;
    for (frames, gap_frames, expected_bouts, expected_seconds) in
        [(4, 21, 0, 0.0), (5, 21, 2, 0.5), (4, 20, 1, 1.4)]
    {
        for speed in [None, Some(0.8)] {
            let mut raw = vec![true; frames];
            raw.extend(vec![false; gap_frames]);
            raw.extend(vec![true; frames]);
            let df = df! { "exploring_A_raw" => &raw }.unwrap();
            let speeds =
                speed.map(|v| Float64Chunked::from_slice("speed".into(), &vec![v; raw.len()]));
            let angles = Float64Chunked::from_slice("angle".into(), &vec![30.0; raw.len()]);
            let native = object_exploration_with_bout_filters(
                &df,
                "exploring_A_raw",
                FPS,
                native_policy.min_bout_seconds,
                native_policy.max_gap_seconds,
                0.0,
                BoutFilterSpec {
                    strict_boundaries: native_policy.strict_boundaries,
                    speed_mode: native_policy.speed_filter_mode,
                    angle_mode: BoutFilterMode::PerBoutMedian,
                    max_nose_speed: native_policy.max_nose_speed,
                    max_angle_deg: native_policy.max_angle_deg,
                    nose_speed_mps: speeds.as_ref(),
                    angle_deg: Some(&angles),
                    ..BoutFilterSpec::unfiltered()
                },
            );
            let first_end = frames as f64 / FPS;
            let second_start = (frames + gap_frames) as f64 / FPS;
            let mut first = candidate(
                "first",
                0.0,
                first_end,
                &vec![30.0; frames],
                &vec![false; frames],
            );
            let mut second = candidate(
                "second",
                second_start,
                second_start + first_end,
                &vec![30.0; frames],
                &vec![false; frames],
            );
            second["gap_before"] = json!({"start_s":first_end,"end_s":second_start,
                "angles_degrees":vec![30.0; gap_frames]});
            if let Some(speed) = speed {
                first["nose_speeds_m_per_s"] = json!(vec![speed; frames]);
                second["nose_speeds_m_per_s"] = json!(vec![speed; frames]);
                second["gap_before"]["nose_speeds_m_per_s"] = json!(vec![speed; gap_frames]);
            }
            let standalone = events::nort_bouts_with_policy(
                &[first, second],
                "include_gaps",
                "include_gaps",
                &standalone_policy,
            )
            .unwrap();
            assert_eq!(
                native.1, expected_bouts,
                "frames={frames}, gap={gap_frames}, speed={speed:?}"
            );
            assert!((native.0 - expected_seconds).abs() < 1e-12);
            assert_eq!(standalone["valid_bouts"], expected_bouts);
            assert!(
                (standalone["object_exploration_s"]["A"].as_f64().unwrap() - expected_seconds)
                    .abs()
                    < 1e-12
            );
            assert_eq!(standalone["removed_speed_bouts"], 0);
        }
    }
}

#[test]
fn recipe_native_and_standalone_agree_at_strict_gap_and_speed_boundaries() {
    use bekiper_feature::heuristic::combined::exploration::NortBoutPolicy;
    let policy = NortBoutPolicy::recipe();
    for gap_frames in [4, 5] {
        for speed in [0.049, 0.05, 0.051] {
            let mut raw = vec![true; 5];
            raw.extend(vec![false; gap_frames]);
            raw.extend(vec![true; 5]);
            let df = df! { "exploring_A_raw" => &raw }.unwrap();
            let speeds = Float64Chunked::from_slice("speed".into(), &vec![speed; raw.len()]);
            let angles = Float64Chunked::from_slice("angle".into(), &vec![30.0; raw.len()]);
            let native = object_exploration_with_bout_filters(
                &df,
                "exploring_A_raw",
                10.0,
                policy.min_bout_seconds,
                policy.max_gap_seconds,
                0.0,
                BoutFilterSpec {
                    strict_boundaries: true,
                    speed_mode: BoutFilterMode::PerBoutMedian,
                    angle_mode: BoutFilterMode::PerBoutMedian,
                    max_angle_deg: 45.0,
                    nose_speed_mps: Some(&speeds),
                    angle_deg: Some(&angles),
                    ..BoutFilterSpec::unfiltered()
                },
            );
            let first_end = 0.5;
            let second_start = (5 + gap_frames) as f64 / 10.0;
            let mut first = candidate("first", 0.0, first_end, &[30.0; 5], &[false; 5]);
            first["nose_speeds_m_per_s"] = json!(vec![speed; 5]);
            let mut second = candidate(
                "second",
                second_start,
                second_start + 0.5,
                &[30.0; 5],
                &[false; 5],
            );
            second["nose_speeds_m_per_s"] = json!(vec![speed; 5]);
            second["gap_before"] = json!({"start_s":first_end,"end_s":second_start,
                "angles_degrees":vec![30.0; gap_frames],"nose_speeds_m_per_s":vec![speed; gap_frames]});
            let standalone = events::nort_bouts_with_policy(
                &[first, second],
                "include_gaps",
                "include_gaps",
                &policy,
            )
            .unwrap();
            assert!(
                (standalone["object_exploration_s"]["A"].as_f64().unwrap() - native.0).abs()
                    < 1e-12
            );
            assert_eq!(standalone["valid_bouts"], native.1);
        }
    }
}

#[test]
fn native_parity_includes_explicit_gap_angles_and_retains_observed_only_policy() {
    let raw = [
        true, true, false, false, false, false, false, false, false, false, true, true,
    ];
    let angles = [
        30.0, 30.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 30.0, 30.0,
    ];
    let first = candidate("first", 0.0, 0.2, &[30.0; 2], &[false; 2]);
    let mut second = candidate("second", 1.0, 1.2, &[30.0; 2], &[false; 2]);
    second["gap_before"] = json!({"start_s":0.2,"end_s":1.0,
        "angles_degrees":vec![90.0; 8],"nose_contacts":vec![false; 8]});
    let parts = [second, first]; // Evidence follows the segment, not input order.
    let result = events::nort_bouts(&parts, "include_gaps", "include_gaps").unwrap();
    let expected = native(&raw, &angles, &[false; 12]);
    assert_eq!(expected, (0.0, 0));
    assert_eq!(result["object_exploration_s"]["A"], expected.0);
    assert_eq!(result["valid_bouts"], expected.1);
    let merged = result["bout_debug"].as_array().unwrap().last().unwrap();
    assert_eq!(merged["median_angle_degrees"], 90.0);
    assert_eq!(merged["gap_median_policy"], "include_gaps");
    assert_eq!(merged["gap_evidence"][0]["start_s"], 0.2);
    let historical = events::nort_bouts(&parts, "include_gaps", "observed_samples_only").unwrap();
    assert_eq!(historical["object_exploration_s"]["A"], 1.2);
    assert_eq!(
        historical["bout_debug"].as_array().unwrap().last().unwrap()["median_angle_degrees"],
        30.0
    );
}

#[test]
fn native_parity_contact_rescues_a_high_angle_bout_without_speed_samples() {
    for contact in [true, false] {
        let contacts = [false, contact, false, false, false];
        let part = candidate("contact", 0.0, 0.5, &[60.0; 5], &contacts);
        let result = events::nort_bouts(&[part], "include_gaps", "observed_samples_only").unwrap();
        let expected = native(&[true; 5], &[60.0; 5], &contacts);
        assert_eq!(result["object_exploration_s"]["A"], expected.0);
        assert_eq!(result["valid_bouts"], expected.1);
        assert_eq!(result["removed_speed_bouts"], 0);
        assert_eq!(
            result["bouts_rescued_by_contact_override"],
            u32::from(contact)
        );
        let merged = result["bout_debug"].as_array().unwrap().last().unwrap();
        assert_eq!(merged["contact_override_applied"], contact);
        assert!(merged["median_nose_speed_m_per_s"].is_null());
    }
}

#[test]
fn native_parity_gap_contacts_require_the_selected_gap_sample_policy() {
    let first = candidate("first", 0.0, 0.2, &[60.0; 2], &[false; 2]);
    let mut second = candidate("second", 0.3, 0.5, &[60.0; 2], &[false; 2]);
    second["gap_before"] = json!({"start_s":0.2,"end_s":0.3,
        "angles_degrees":[60.0],"nose_contacts":[true],"nose_speeds_m_per_s":[0.8]});
    let parts = [first, second];
    let result = events::nort_bouts(&parts, "include_gaps", "include_gaps").unwrap();
    let expected = native(
        &[true, true, false, true, true],
        &[60.0; 5],
        &[false, false, true, false, false],
    );
    assert_eq!(expected, (0.5, 1));
    assert_eq!(result["object_exploration_s"]["A"], expected.0);
    assert_eq!(result["valid_bouts"], expected.1);
    assert_eq!(
        result["bout_debug"].as_array().unwrap().last().unwrap()["median_nose_speed_m_per_s"],
        0.8
    );
    let observed = events::nort_bouts(&parts, "include_gaps", "observed_samples_only").unwrap();
    assert_eq!(observed["valid_bouts"], 0);
}

#[test]
fn native_parity_rejects_missing_or_misaligned_gap_evidence() {
    let first = candidate("first", 0.0, 0.2, &[30.0; 2], &[false; 2]);
    let second = candidate("second", 1.0, 1.2, &[30.0; 2], &[false; 2]);
    for evidence in [
        Value::Null,
        json!({"start_s":0.3,"end_s":1.0,"angles_degrees":[30.0]}),
        json!({"start_s":0.2,"end_s":0.9,"angles_degrees":[30.0]}),
        json!({"start_s":0.2,"end_s":1.0,"angles_degrees":[]}),
    ] {
        let mut second = second.clone();
        second["gap_before"] = evidence;
        let error = events::nort_bouts(&[first.clone(), second], "include_gaps", "include_gaps")
            .unwrap_err();
        assert!(error.to_string().contains("gap_before"), "{error:#}");
    }
}

#[test]
fn native_parity_rejects_unaligned_or_nonboolean_contact_samples() {
    for contacts in [json!([true]), json!([true, 1]), json!(true)] {
        let mut part = candidate("bad", 0.0, 0.5, &[60.0; 2], &[false; 2]);
        part["nose_contacts"] = contacts;
        assert!(events::nort_bouts(&[part], "include_gaps", "observed_samples_only").is_err());
    }
}

/// Whisker-fan candidates: native bout rescue (`nose_contact` mask carries
/// contact OR whisker hit) and standalone `whisker_contacts` agree; a disabled
/// policy ignores the whisker field.
#[test]
fn whisker_hits_rescue_a_high_angle_bout_like_contact_when_enabled() {
    use bekiper_feature::heuristic::combined::exploration::NortBoutPolicy;
    const FPS: f64 = 20.0;
    let frames = 10;
    let raw = vec![true; frames];
    let df = df! { "exploring_A_raw" => &raw }.unwrap();
    let angles = Float64Chunked::from_slice("angle".into(), &vec![90.0; frames]);
    let mut hits = vec![false; frames];
    hits[3] = true;
    let hit_mask = BooleanChunked::from_slice("hit".into(), &hits);
    let none_mask = BooleanChunked::from_slice("none".into(), &vec![false; frames]);
    let policy = NortBoutPolicy {
        whisker_enabled: true,
        ..NortBoutPolicy::default()
    };
    let native = |contact: &BooleanChunked| {
        object_exploration_with_bout_filters(
            &df,
            "exploring_A_raw",
            FPS,
            policy.min_bout_seconds,
            policy.max_gap_seconds,
            0.0,
            BoutFilterSpec {
                angle_mode: BoutFilterMode::PerBoutMedian,
                max_angle_deg: policy.max_angle_deg,
                angle_deg: Some(&angles),
                nose_contact: Some(contact),
                ..BoutFilterSpec::unfiltered()
            },
        )
    };
    assert_eq!(native(&none_mask).1, 0);
    assert_eq!(native(&hit_mask).1, 1);

    let mut segment = candidate("only", 0.0, 0.5, &vec![90.0; frames], &vec![false; frames]);
    segment["whisker_contacts"] = json!(hits);
    let run = |policy: &NortBoutPolicy| {
        events::nort_bouts_with_policy(&[segment.clone()], "include_gaps", "include_gaps", policy)
            .unwrap()
    };
    let enabled = run(&policy);
    assert_eq!(enabled["valid_bouts"], 1);
    let merged = enabled["bout_debug"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["stage"] == "merged")
        .unwrap()
        .clone();
    assert_eq!(
        merged["bout_contact_policy"],
        "any_observed_nose_or_whisker_contact"
    );
    let disabled = run(&NortBoutPolicy::default());
    assert_eq!(disabled["valid_bouts"], 0);
}
