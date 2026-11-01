use bekiper_post_analysis::{events, metrics};
use serde_json::{Value, json};

fn segment(id: &str, object: &str, start: f64, end: f64) -> Value {
    json!({"segment_id":id,"object_id":object,"start_s":start,"end_s":end,
        "candidate_eligible":true,"angles_degrees":[0,0,90],"nose_speeds_m_per_s":[0.01,0.01,0.01]})
}

fn nort(parts: &[Value], gaps: &str) -> Value {
    events::nort_bouts(parts, gaps, "observed_samples_only").unwrap()
}

fn row(assay: &str) -> Value {
    json!({"animal_id":"fixture_a","assay":assay,"session_id":"fixture_session","timepoint":"post",
        "phase":if assay == "nort" {"test"} else {"session"},"measurement_available":true,
        "excluded":false,"exclusion_reason":null,"synthetic_observation":true})
}

fn config() -> Value {
    json!({"run_mode":"synthetic","synthetic_fixture":true,
        "inputs":{"units_and_calibration":{
            "oft":{"time":"s","distance":"m","speed":"m/s"},
            "ymaze":{"time":"s","distance":"m","speed":"m/s"},
            "nort":{"time":"s","distance":"m","speed":"m/s"}}},
        "assays":{"oft":{},"ymaze":{"triad_policy":"source_table","fewer_than_three_entries_policy":"zero_windows",
            "primary_threshold_arm_length_pct":45},"nort":{}},
        "decision_resolutions":{
            "D01":{"value":{"assays.ymaze.triad_policy":"source_table"},"scope":["assays.ymaze"],
                "origin":"development_fixture","rationale":"fixture only","approval_or_plan_reference":"test"},
            "D02":{"value":{"assays.ymaze.fewer_than_three_entries_policy":"zero_windows"},"scope":["assays.ymaze"],
                "origin":"development_fixture","rationale":"fixture only","approval_or_plan_reference":"test"},
            "D03":{"value":{"assays.ymaze.primary_threshold_arm_length_pct":45},"scope":["assays.ymaze"],
                "origin":"development_fixture","rationale":"fixture only","approval_or_plan_reference":"test"}}})
}

fn register_choice(config: &mut Value, decision: &str, path: &str, value: Value) {
    let mut target = &mut *config;
    let fields: Vec<_> = path.split('.').collect();
    for field in &fields[..fields.len() - 1] {
        target = &mut target[*field];
    }
    target[fields[fields.len() - 1]] = value.clone();
    if config["decision_resolutions"][decision].is_null() {
        config["decision_resolutions"][decision] = json!({"value":{},"scope":["assays"],
            "origin":"development_fixture","rationale":"regression fixture only","approval_or_plan_reference":"scoring tests"});
    }
    config["decision_resolutions"][decision]["value"][path] = value;
}

fn oft_interval_config() -> Value {
    let mut config = config();
    for (decision, path, value) in [
        (
            "D05",
            "assays.oft.center_occupancy_policy",
            "qualified_only",
        ),
        (
            "D05",
            "assays.oft.no_center_entry_latency_policy",
            "missing",
        ),
        (
            "D04",
            "assays.oft.window_and_tracking_policy",
            "half_open_300s_from_first_detection",
        ),
        (
            "D05",
            "assays.oft.starting_in_center_policy",
            "count_as_entry_at_origin",
        ),
        (
            "D04",
            "assays.oft.entry_timestamp_policy",
            "initial_crossing",
        ),
    ] {
        register_choice(&mut config, decision, path, json!(value));
    }
    config
}

include!("scoring/ymaze.rs");
include!("scoring/nort.rs");
include!("scoring/lineage.rs");
include!("scoring/oft.rs");
