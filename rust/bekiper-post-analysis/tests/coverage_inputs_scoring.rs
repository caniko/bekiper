//! Edge cases of input adapters, event reconstruction and session scoring that
//! the behaviour-level suites do not reach: explicit NA reasons, rejected
//! registrations and malformed upstream tables.
use bekiper_post_analysis::{RunOptions, contracts, events, fixtures, genotype, inputs, metrics};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path};
use tempfile::tempdir;

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
        "assays":{"oft":{},"ymaze":{},"nort":{}},"decision_resolutions":{}})
}

fn register_choice(config: &mut Value, decision: &str, path: &str, value: Value) {
    let mut target = &mut *config;
    let fields: Vec<_> = path.split('.').collect();
    for field in &fields[..fields.len() - 1] {
        target = &mut target[*field];
    }
    target[fields[fields.len() - 1]] = value.clone();
    if config["decision_resolutions"][decision].is_null() {
        config["decision_resolutions"][decision] = json!({"value":{},"scope":["assays","statistics","outputs"],
            "origin":"development_fixture","rationale":"coverage fixture only","approval_or_plan_reference":"coverage tests"});
    }
    config["decision_resolutions"][decision]["value"][path] = value;
}

fn score(input: &Value, config: &Value) -> anyhow::Result<Value> {
    metrics::score_records(std::slice::from_ref(input), config).map(|(rows, _)| rows[0].clone())
}

fn segment(id: &str, object: &str, start: f64, end: f64) -> Value {
    json!({"segment_id":id,"object_id":object,"start_s":start,"end_s":end,
        "candidate_eligible":true,"angles_degrees":[0,0,90],"nose_speeds_m_per_s":[0.01,0.01,0.01]})
}

fn ods(path: &Path, rows: &str) {
    let xml = format!(
        r#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><table:table table:name="s">{rows}</table:table></office:document-content>"#
    );
    let mut archive = zip::ZipWriter::new(fs::File::create(path).unwrap());
    archive
        .start_file("content.xml", zip::write::SimpleFileOptions::default())
        .unwrap();
    archive.write_all(xml.as_bytes()).unwrap();
    archive.finish().unwrap();
}

const HEADER: &str = r#"<table:table-row><table:table-cell><text:p>animal</text:p></table:table-cell><table:table-cell><text:p>value</text:p></table:table-cell></table:table-row>"#;

// ---------------------------------------------------------------- contracts

#[test]
fn contract_helpers_follow_dotted_paths_and_create_missing_reason_maps() {
    let record = json!({"a":{"b":{"c":3}}});
    assert_eq!(contracts::at(&record, "a.b.c"), 3);
    assert!(contracts::at(&record, "a.x.c").is_null());
    let mut row = json!({"field":1,"missing_reasons":null});
    contracts::missing(&mut row, "field", "not_measured");
    assert!(row["field"].is_null());
    assert_eq!(row["missing_reasons"], json!({"field":"not_measured"}));
}

#[test]
fn unknown_policy_labels_are_explicit() {
    assert_eq!(genotype::label("Z"), "Unknown HCAR1 policy");
    assert!(genotype::project(&[], &json!({}), "A").is_err());
}

// ------------------------------------------------------------------- config

#[test]
fn registration_scopes_must_all_be_strings() {
    let mut config = json!({"run_mode":"production","x":{"y":1},"decision_resolutions":{"D1":{
        "value":{"x.y":1},"scope":["x"],"origin":"researcher_decision","rationale":"r",
        "approval_or_plan_reference":"p"}}});
    assert_eq!(inputs::choice(&config, "x.y", "D1"), Some(&json!(1)));
    config["decision_resolutions"]["D1"]["scope"] = json!(["x", 1]);
    assert_eq!(inputs::choice(&config, "x.y", "D1"), None);
}

#[test]
fn sensitivity_and_primary_threshold_layers_validate_every_source() {
    assert!(inputs::validate_sensitivity_thresholds(&json!("25,45")).is_err());
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.json");
    fs::write(&path, r#"{"run_mode":"development"}"#).unwrap();
    let resolve = |options: &RunOptions, env: &dyn Fn(&str) -> Option<String>| {
        inputs::resolved_config(&path, options, env)
    };
    // Constructor values win and are recorded in execution.
    let options = RunOptions {
        primary_thresholds: Some(vec![45]),
        primary_workers: Some(2),
        ..Default::default()
    };
    let config = resolve(&options, &|_| None).unwrap();
    assert_eq!(config["execution"]["ymaze_primary_thresholds"], json!([45]));
    assert_eq!(config["execution"]["primary_workers"], 2);
    let env = |pairs: &'static [(&'static str, &'static str)]| {
        move |key: &str| {
            pairs
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| (*value).to_owned())
        }
    };
    let config = resolve(
        &Default::default(),
        &env(&[
            ("BEKIPER_YMAZE_PRIMARY_THRESHOLDS", " 25, 65"),
            ("BEKIPER_POST_ANALYSIS_SENSITIVITY_THRESHOLDS", "45, 25"),
        ]),
    )
    .unwrap();
    assert_eq!(
        config["execution"]["ymaze_primary_thresholds"],
        json!([25, 65])
    );
    assert_eq!(
        config["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"],
        json!([45, 25])
    );
    let config = resolve(
        &Default::default(),
        &env(&[("BEKIPER_YMAZE_PRIMARY_THRESHOLDS", "  ")]),
    )
    .unwrap();
    assert_eq!(config["execution"]["ymaze_primary_thresholds"], json!([]));
    for bad in [
        &[("BEKIPER_YMAZE_PRIMARY_THRESHOLDS", "25,x")][..],
        &[("BEKIPER_YMAZE_PRIMARY_THRESHOLDS", "44")],
        &[("BEKIPER_PRIMARY_ANALYSIS_WORKERS", "0")],
        &[("BEKIPER_PRIMARY_ANALYSIS_WORKERS", "many")],
        &[("BEKIPER_POST_ANALYSIS_SENSITIVITY_THRESHOLDS", "25,abc")],
    ] {
        assert!(resolve(&Default::default(), &env(bad)).is_err(), "{bad:?}");
    }
    fs::write(
        &path,
        r#"{"run_mode":"development","outputs":{"directory":"from-plan"}}"#,
    )
    .unwrap();
    assert_eq!(
        resolve(&Default::default(), &|_| None).unwrap()["outputs"]["directory"],
        "from-plan"
    );
    fs::write(
        &path,
        r#"{"run_mode":"development","outputs":{"directory":5}}"#,
    )
    .unwrap();
    assert!(
        format!("{:#}", resolve(&Default::default(), &|_| None).unwrap_err())
            .contains("outputs.directory must be a string")
    );
}

#[test]
fn subgroup_plans_need_both_registrations_and_default_outputs_follow_the_plan() {
    let plan = json!({"strata":[{"field":"sex","levels":["female"],"assays":["oft"]}],
        "outputs":["models"],"multiplicity":{"family":"subgroups","method":"holm"}});
    let mut config = json!({"run_mode":"production","statistics":{"subgroup_analysis_plan":plan},
        "outputs":{"report":{"enabled":true}},"decision_resolutions":{"D09":{
        "value":{"statistics.subgroup_analysis_plan":plan},"scope":["statistics"],
        "origin":"researcher_decision","rationale":"r","approval_or_plan_reference":"p"}}});
    // D09 alone never admits subgroup inference.
    assert_eq!(inputs::subgroup_plan(&config).unwrap(), None);
    config["decision_resolutions"]["D12"] = config["decision_resolutions"]["D09"].clone();
    assert_eq!(inputs::subgroup_plan(&config).unwrap(), Some(plan));
    assert_eq!(
        inputs::requested_output_ids(&config, &[]).unwrap(),
        [
            "metrics",
            "models",
            "figures",
            "subgroup_inference",
            "report"
        ]
    );
}

// ------------------------------------------------------------------- cohort

fn write_json(path: &Path, value: &Value) {
    fs::write(path, value.to_string()).unwrap();
}

fn metadata(assay: &str, phase: &str) -> Value {
    json!({"animal_id":"7","assay":assay,"session_id":"s1","timepoint":"pre","phase":phase,
        "five_xfad":"negative","hcar1":"WT","treatment":"vehicle","group":"G0","sex":"female",
        "age_treatment_window":"adult","excluded":false,"exclusion_reason":null})
}

#[test]
fn cohort_requires_metadata_rule_roles_and_matching_sessions() {
    let dir = tempdir().unwrap();
    assert!(
        format!(
            "{:#}",
            inputs::load_inputs(&json!({"inputs":{}}), dir.path()).unwrap_err()
        )
        .contains("metadata_path is required")
    );
    write_json(
        &dir.path().join("metadata.json"),
        &json!([metadata("ymaze", "session")]),
    );
    fs::write(dir.path().join("implementation.txt"), "rule source").unwrap();
    let sha =
        inputs::fingerprint(&dir.path().join("implementation.txt")).unwrap()["sha256"].clone();
    let mut config = json!({"inputs":{"metadata_path":"metadata.json","upstream_rule_provenance":{
        "oft":{"evidence_files":[{"role":"implementation","sha256":sha,"path":"implementation.txt"}]}}}});
    // A non-audit evidence role is fingerprinted but never parsed.
    let (rows, descriptors) = inputs::load_inputs(&config, dir.path()).unwrap();
    assert_eq!(rows[0]["measurement_available"], false);
    assert!(
        descriptors
            .iter()
            .any(|d| d["role"] == "oft_evidence:implementation")
    );
    config["inputs"]["upstream_rule_provenance"]["oft"]["evidence_files"][0]
        .as_object_mut()
        .unwrap()
        .remove("role");
    assert!(
        format!(
            "{:#}",
            inputs::load_inputs(&config, dir.path()).unwrap_err()
        )
        .contains("unique roles")
    );
    // Y-maze proximal_threshold fractions become the explicit percentage.
    let config = json!({"inputs":{"metadata_path":"metadata.json","ymaze_path":"ymaze.json"}});
    let mut measurement = metadata("ymaze", "session");
    for field in ["five_xfad", "hcar1", "treatment", "group", "sex"] {
        measurement.as_object_mut().unwrap().remove(field);
    }
    measurement["proximal_threshold"] = json!(0.45);
    write_json(&dir.path().join("ymaze.json"), &json!([measurement]));
    let (rows, _) = inputs::load_inputs(&config, dir.path()).unwrap();
    assert_eq!(rows[0]["threshold_pct"], 45.0);
    assert_eq!(rows[0]["measurement_available"], true);
    measurement["session_id"] = json!("unmatched");
    write_json(&dir.path().join("ymaze.json"), &json!([measurement]));
    assert!(
        format!(
            "{:#}",
            inputs::load_inputs(&config, dir.path()).unwrap_err()
        )
        .contains("no unique matching metadata session")
    );
    // Metadata validation failures stop the join before any measurement is read.
    let mut incomplete = metadata("ymaze", "session");
    incomplete.as_object_mut().unwrap().remove("hcar1");
    write_json(&dir.path().join("metadata.json"), &json!([incomplete]));
    assert!(inputs::load_inputs(&config, dir.path()).is_err());
}

// ------------------------------------------------------------------- tables

#[test]
fn json_and_extension_errors_name_the_supported_formats() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rows.json");
    fs::write(&path, r#"{"animal_id":"1"}"#).unwrap();
    assert!(
        format!("{:#}", inputs::read_table(&path, None).unwrap_err()).contains("must be an array")
    );
    let path = dir.path().join("rows.xlsx");
    fs::write(&path, "binary").unwrap();
    assert!(
        format!("{:#}", inputs::read_table(&path, None).unwrap_err())
            .contains("Unsupported table extension")
    );
}

#[test]
fn parquet_cells_keep_their_numeric_type_and_reject_nonfinite_values() {
    use polars::prelude::*;
    let dir = tempdir().unwrap();
    let path = dir.path().join("typed.parquet");
    let write = |columns: Vec<Column>| {
        let mut frame = DataFrame::new(1, columns).unwrap();
        ParquetWriter::new(fs::File::create(&path).unwrap())
            .finish(&mut frame)
            .unwrap();
    };
    write(vec![
        Series::new("s".into(), ["text"]).into(),
        Series::new("i8".into(), [-8i8]).into(),
        Series::new("i16".into(), [-16i16]).into(),
        Series::new("i32".into(), [-32i32]).into(),
        Series::new("u32".into(), [32u32]).into(),
        Series::new("u64".into(), [64u64]).into(),
        Series::new("f32".into(), [1.5f32]).into(),
        Series::new("f64".into(), [2.25f64]).into(),
        Series::new("list".into(), [Series::new("".into(), [1i64, 2])]).into(),
    ]);
    let row = &inputs::read_table(&path, None).unwrap()[0];
    assert_eq!(
        row,
        &json!({"s":"text","i8":-8,"i16":-16,"i32":-32,"u32":32,"u64":64,
            "f32":1.5,"f64":2.25,"list":[1,2]})
    );
    let nonfinite_struct = StructChunked::from_series(
        "nested".into(),
        1,
        [Series::new("x".into(), [f64::NAN])].iter(),
    )
    .unwrap()
    .into_series();
    for column in [
        Series::new("f32".into(), [f32::INFINITY]),
        Series::new("f64".into(), [f64::NAN]),
        Series::new("list".into(), [Series::new("".into(), [f64::NAN])]),
        nonfinite_struct,
    ] {
        write(vec![column.into()]);
        assert!(
            format!("{:#}", inputs::read_table(&path, None).unwrap_err()).contains("nonfinite")
        );
    }
    write(vec![Series::new("bytes".into(), [&b"raw"[..]]).into()]);
    assert!(
        format!("{:#}", inputs::read_table(&path, None).unwrap_err())
            .contains("Unsupported Parquet cell type")
    );
}

#[test]
fn ods_rejects_untyped_cells_and_populated_unnamed_source_columns() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("table.ods");
    let cell = |attrs: &str, text: &str| {
        format!(r#"<table:table-cell {attrs}><text:p>{text}</text:p></table:table-cell>"#)
    };
    // Covered (merged) cells and explicit empty strings decode to null.
    ods(
        &path,
        &format!(
            "{HEADER}<table:table-row>{}{}</table:table-row><table:table-row>{}<table:covered-table-cell/></table:table-row>",
            cell(r#"office:value-type="string""#, "1"),
            cell(r#"office:value-type="string""#, " "),
            cell(
                r#"office:value-type="boolean" office:boolean-value="true""#,
                ""
            ),
        ),
    );
    let rows = inputs::read_table(&path, Some("s")).unwrap();
    assert_eq!(rows[0], json!({"animal":"1","value":null}));
    assert_eq!(rows[1], json!({"animal":true,"value":null}));
    for (attrs, message) in [
        (
            r#"office:value-type="float" office:value="ten""#,
            "ODS numeric value",
        ),
        (
            r#"office:value-type="boolean" office:boolean-value="maybe""#,
            "Invalid ODS boolean",
        ),
        (
            r#"office:value-type="date" office:date-value="2026-01-01""#,
            "Unsupported ODS value type: date",
        ),
    ] {
        ods(
            &path,
            &format!(
                "{HEADER}<table:table-row>{}</table:table-row>",
                cell(attrs, "x")
            ),
        );
        let error = format!("{:#}", inputs::read_table(&path, Some("s")).unwrap_err());
        assert!(error.contains(message), "{error}");
    }
    // Source sheets keep unnamed blank columns only while they stay empty.
    let header = format!(
        "<table:table-row>{}<table:table-cell/>{}</table:table-row>",
        cell("", "animal"),
        cell("", "hcar1")
    );
    ods(
        &path,
        &format!(
            "{header}<table:table-row>{}<table:table-cell/>{}</table:table-row>",
            cell("", "37"),
            cell("", "Het/KO")
        ),
    );
    let rows = inputs::read_ods_source(&path, "s").unwrap();
    assert_eq!(
        rows[0],
        json!({"animal":"37","hcar1":"Het/KO","source_row":2})
    );
    ods(
        &path,
        &format!(
            "{header}<table:table-row>{}{}{}</table:table-row>",
            cell("", "37"),
            cell("", "stray"),
            cell("", "WT")
        ),
    );
    assert!(
        format!("{:#}", inputs::read_ods_source(&path, "s").unwrap_err())
            .contains("Populated unnamed ODS column at row 2")
    );
}

#[test]
fn compact_sequences_and_integral_float_identifiers_normalize_exactly() {
    let row = inputs::normalize(
        &json!({"valid_arm_entry_sequence":"ABCA","animal_id":99.0,"session_id":1.5}),
        &json!({}),
    )
    .unwrap();
    assert_eq!(row["valid_arm_entry_sequence"], json!(["A", "B", "C", "A"]));
    assert_eq!(row["animal_id"], "99");
    assert_eq!(row["session_id"], "1.5");
}

// ------------------------------------------------------------------- events

fn nort_row(phase: &str) -> Value {
    let mut row = row("nort");
    row["phase"] = json!(phase);
    row
}

#[test]
fn object_roles_absent_phase_or_configuration_is_unavailable_not_inferred() {
    let row = nort_row("test");
    assert_eq!(
        events::nort_object_roles(&row, &json!({"training":{"TA":"a","TB":"b"}})).unwrap(),
        None
    );
    let mapping = json!({"test":{"TN":"a","TF":"b"}});
    assert_eq!(events::nort_object_roles(&row, &mapping).unwrap(), None);
    let mut configured = row.clone();
    configured["object_configuration"] = json!("left_right_counterbalanced");
    assert_eq!(
        events::nort_object_roles(&configured, &mapping).unwrap(),
        Some(json!({"TN":"a","TF":"b"}))
    );
}

#[test]
fn physical_sides_need_exact_session_evidence_and_optional_identity_checks() {
    let row = nort_row("training");
    let mut assignment = row.clone();
    assignment["roles"] = json!({"TA":"a","TB":"b"});
    let mapping = json!({"sessions":[assignment.clone()]});
    assert_eq!(events::nort_physical_sides(&row, &mapping).unwrap(), None);
    assignment["physical_sides"] = json!({"TA":"left","TB":"right"});
    assert_eq!(
        events::nort_physical_sides(&row, &json!({"sessions":[assignment]})).unwrap(),
        Some(json!({"TA":"left","TB":"right"}))
    );
}

#[test]
fn geometry_windows_and_landmark_roles_fail_closed() {
    assert!(!events::polygon_contains(0.0, 0.0, &[]));
    let window = json!({"start_s":0.0,"end_s":10.0});
    assert!(events::validate_window(&[json!({"timestamp_s":-1.0})], &window, false).is_err());
    for roles in [
        json!({"head_center":{"x":"hx"},"tail_base":{"x":"tx","y":"ty"}}),
        json!({"head_center":{"x":"hx","y":1},"tail_base":{"x":"tx","y":"ty"}}),
        json!({"head_center":{"x":"hx","y":"hx"},"tail_base":{"x":"tx","y":"ty"}}),
        json!({"head_center":{"x":"hx","y":" "},"tail_base":{"x":"tx","y":"ty"}}),
    ] {
        assert!(!events::complete_landmark_roles(&roles), "{roles}");
    }
    let mut row = row("ymaze");
    row["threshold_pct"] = json!(45);
    row["scoring_provenance"] = json!({"geometry_version":"other_geometry",
        "requires_both_landmarks_same_arm":false,"apply_dwell_filter":true,"minimum_dwell_seconds":1.0});
    let verification = events::ymaze_provenance(&row, &json!({})).unwrap();
    let reasons = verification["reasons"].as_array().unwrap();
    assert!(reasons.contains(&json!("different_geometry")));
    assert!(reasons.contains(&json!("single_landmark_or_proxy")));
}

#[test]
fn nort_candidates_record_rejection_and_speed_boundary_inclusivity() {
    use bekiper_feature::heuristic::combined::exploration::{BoutFilterMode, NortBoutPolicy};
    let mut rejected = segment("r", "A", 0.0, 1.0);
    rejected["candidate_eligible"] = json!(false);
    let result = events::nort_bouts(&[rejected], "exclude_gaps", "observed_samples_only").unwrap();
    assert_eq!(result["valid_bouts"], 0);
    assert_eq!(
        result["bout_debug"][0]["rejection_reasons"],
        json!(["upstream_candidate_rejection"])
    );
    // Same-object overlap passes object arbitration but is still invalid evidence.
    let overlapping = [segment("a", "A", 0.0, 1.0), segment("b", "A", 0.5, 1.5)];
    let error = events::nort_bouts(&overlapping, "exclude_gaps", "observed_samples_only")
        .unwrap_err()
        .to_string();
    assert!(!error.contains("different-object"), "{error}");
    let mut part = segment("s", "A", 0.0, 1.0);
    part["nose_speeds_m_per_s"] = json!([0.05]);
    for (strict, valid) in [(false, 1), (true, 0)] {
        let policy = NortBoutPolicy {
            speed_filter_mode: BoutFilterMode::PerBoutMedian,
            max_nose_speed: 0.05,
            strict_boundaries: strict,
            ..Default::default()
        };
        let result = events::nort_bouts_with_policy(
            std::slice::from_ref(&part),
            "exclude_gaps",
            "observed_samples_only",
            &policy,
        )
        .unwrap();
        assert_eq!(result["valid_bouts"], valid, "strict={strict}");
    }
}

#[test]
fn oft_latency_is_the_first_qualified_center_entry_across_periphery_runs() {
    let intervals = [
        json!({"start_s":10.0,"end_s":12.0,"zone":"center"}),
        json!({"start_s":12.0,"end_s":20.0,"zone":"periphery"}),
        json!({"start_s":20.0,"end_s":25.0,"zone":"center"}),
    ];
    let result = events::oft_zone_intervals(&intervals, 5.0, "qualified_only", "missing").unwrap();
    assert_eq!(result["center_entries"], 2);
    assert_eq!(result["center_time_s"], 7.0);
    assert_eq!(result["latency_to_first_center_entry_s"], 5.0);
}

// ------------------------------------------------------------------ metrics

#[test]
fn triad_policies_report_short_sequences_and_validate_custom_mappings() {
    let none = metrics::ymaze_scores(&json!("AB"), None, Some("zero_windows"), None).unwrap();
    assert_eq!(none["total_triads"], 0);
    let short = metrics::ymaze_scores(&json!("AB"), Some("source_table"), None, None).unwrap();
    assert!(short["total_triads"].is_null());
    assert_eq!(
        short["missing_reasons"]["total_triads"],
        "blocked:D02:insufficient_entries"
    );
    assert_eq!(
        short["missing_reasons"]["spontaneous_alternation_pct"],
        "undefined:insufficient_entries"
    );
    assert!(metrics::ymaze_scores(&json!("ABC"), Some("custom"), None, None).is_err());
    let arms = ["A", "B", "C"];
    let mut mapping = serde_json::Map::new();
    for a in arms {
        for b in arms {
            for c in arms {
                mapping.insert(format!("{a}{b}{c}"), json!("neutral"));
            }
        }
    }
    let mapping = Value::Object(mapping);
    let custom =
        metrics::ymaze_scores(&json!("ABCA"), Some("custom"), None, Some(&mapping)).unwrap();
    assert_eq!(custom["neutral_triads"], 2);
    assert_eq!(custom["neutral_triads_pct"], 100.0);
    assert_eq!(
        custom["missing_reasons"]["spontaneous_alternation_pct"],
        "undefined:zero_usable_triads"
    );
    let mut invalid = mapping.clone();
    invalid["ABC"] = json!("correctish");
    assert!(metrics::ymaze_scores(&json!("ABC"), Some("custom"), None, Some(&invalid)).is_err());
    invalid.as_object_mut().unwrap().remove("ABC");
    assert!(metrics::ymaze_scores(&json!("ABC"), Some("custom"), None, Some(&invalid)).is_err());
}

#[test]
fn oft_scores_bound_center_time_and_retain_disagreeing_upstream_percentages() {
    assert!(metrics::oft_scores(&json!({"center_time_s":301.0})).is_err());
    let mut input = row("oft");
    input["center_time_s"] = json!(30.0);
    input["center_time_pct"] = json!(50.0);
    input["prior_lineage"] = json!({"upstream_center_time_pct":40.0});
    let scored = score(&input, &config()).unwrap();
    assert_eq!(scored["center_time_pct"], 10.0);
    assert_eq!(scored["upstream_center_time_pct"], 50.0);
    assert!(
        scored["disagreements"]
            .as_array()
            .unwrap()
            .contains(&json!("center_time_pct:analysed_duration_denominator"))
    );
    assert_eq!(
        scored["prior_lineage"]["previous"]["upstream_center_time_pct"],
        40.0
    );
}

#[test]
fn oft_interval_reconstruction_with_unknown_duration_and_invalid_zone() {
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
    let mut input = row("oft");
    input["first_detection_s"] = json!(0.0);
    input["center_intervals"] = json!([{"start_s":0.0,"end_s":2.0,"zone":"center"}]);
    let scored = score(&input, &config).unwrap();
    assert_eq!(scored["center_time_s"], 2.0);
    assert_eq!(scored["qc_flags"]["incomplete_recording"], Value::Null);
    input["analysis_duration_s"] = json!(300.0);
    input["center_intervals"][0]["zone"] = json!("edge");
    assert!(format!("{:#}", score(&input, &config).unwrap_err()).contains("Unknown OFT zone"));
    let mut censored = row("oft");
    censored["latency_to_first_center_entry_s"] = json!(300.0);
    censored["latency_right_censored"] = json!(true);
    censored["analysis_duration_s"] = json!(120.0);
    assert!(metrics::validate_supplied(&censored).is_ok());
    censored["latency_right_censored"] = json!(false);
    assert!(metrics::validate_supplied(&censored).is_err());
    assert!(metrics::validate_supplied(&json!({"assay":"unscored"})).is_ok());
}

fn ymaze_config() -> Value {
    let mut config = config();
    register_choice(
        &mut config,
        "D01",
        "assays.ymaze.triad_policy",
        json!("source_table"),
    );
    register_choice(
        &mut config,
        "D03",
        "assays.ymaze.primary_threshold_arm_length_pct",
        json!(45),
    );
    config
}

#[test]
fn unavailable_ymaze_sessions_need_no_threshold_and_keep_identity_missing() {
    let mut input = row("ymaze");
    input["measurement_available"] = json!(false);
    let scored = score(&input, &ymaze_config()).unwrap();
    assert!(scored["primary_threshold"].is_null());
    assert_eq!(
        scored["missing_reasons"]["primary_threshold"],
        "not_measured:threshold_identity"
    );
}

#[test]
fn ymaze_invalid_sequences_mappings_and_provenance_fail_closed() {
    let mut input = row("ymaze");
    input["threshold_pct"] = json!(45);
    input["valid_arm_entry_sequence"] = json!(["A", "D", "B"]);
    assert!(score(&input, &ymaze_config()).is_err());
    input
        .as_object_mut()
        .unwrap()
        .remove("valid_arm_entry_sequence");
    let mut custom = ymaze_config();
    register_choice(
        &mut custom,
        "D01",
        "assays.ymaze.triad_policy",
        json!("custom"),
    );
    assert!(format!("{:#}", score(&input, &custom).unwrap_err()).contains("full mapping"));
    // The same unregistered mapping fails when a sequence is supplied.
    input["valid_arm_entry_sequence"] = json!("ABC");
    assert!(format!("{:#}", score(&input, &custom).unwrap_err()).contains("full mapping"));
    input
        .as_object_mut()
        .unwrap()
        .remove("valid_arm_entry_sequence");
    input["scoring_provenance"] = json!(5);
    assert!(score(&input, &ymaze_config()).is_err());
}

fn ymaze_coordinates() -> (Value, Value) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../bekiper-behaviour/src/radial_arm/tests/golden/principal_entry_parity.json"
    ))
    .unwrap();
    let mut input = row("ymaze");
    input["threshold_pct"] = json!(45);
    input["geometry"] = json!({"version":fixture["geometry_version"],"canonical_vertices":true,"center":fixture["center"],"arms":fixture["arms"]});
    input["analysis_duration_s"] = json!(360);
    input["first_detection_s"] = json!(0);
    input["coordinate_samples"] = json!(
        [(0.0, 0.0), (0.9, 0.0), (0.0, 0.0)]
            .into_iter()
            .enumerate()
            .map(|(i, (x, y))| json!({"timestamp_s":i as f64 / 30.0,
                "head_center_x":x,"head_center_y":y,"tail_base_x":x,"tail_base_y":y}))
            .collect::<Vec<_>>()
    );
    let mut config = ymaze_config();
    register_choice(
        &mut config,
        "D04",
        "assays.ymaze.window_and_tracking_policy",
        json!({"interval":"half_open","origin":"first_detection_s","duration":"analysis_duration_s","out_of_window":"reject",
            "missing_pose":"observed_samples_break_on_missing_no_interpolation"}),
    );
    (input, config)
}

#[test]
fn ymaze_coordinates_without_registered_window_or_sample_end_are_explicit() {
    let (input, mut config) = ymaze_coordinates();
    // Zero dwell needs no exclusive final-sample boundary.
    config["assays"]["ymaze"]["temporal_policy"] =
        serde_json::to_value(bekiper_feature::tolerance::TemporalPolicy::default()).unwrap();
    let scored = score(&input, &config).unwrap();
    assert_eq!(
        scored["input_verification"]["status"],
        "verified_clarified_entry_definition"
    );
    let mut broken = input.clone();
    broken["geometry"] = json!({});
    assert!(score(&broken, &config).is_err());
    let scored = score(&input, &ymaze_config()).unwrap();
    assert!(
        scored["blocked_outputs"]
            .as_array()
            .unwrap()
            .contains(&json!("ymaze_coordinate_reconstruction:D04"))
    );
}

fn nort_reconstruction_config() -> Value {
    let mut config = config();
    register_choice(
        &mut config,
        "D07",
        "assays.nort.merged_gap_duration_policy",
        json!("exclude_gaps"),
    );
    register_choice(
        &mut config,
        "D07",
        "assays.nort.merged_gap_median_policy",
        json!("observed_samples_only"),
    );
    register_choice(
        &mut config,
        "D10",
        "assays.nort.object_role_and_location_mapping",
        json!({"test":{"TN":"right","TF":"left"}}),
    );
    register_choice(
        &mut config,
        "D14",
        "assays.nort.window_and_tracking_policy",
        json!({"interval":"half_open","origin":"analysis_start_s","duration":"analysis_duration_s","out_of_window":"reject","protocol_departures":"flag_only"}),
    );
    config
}

#[test]
fn nort_reconstruction_reports_unexplored_objects_and_unregistered_merges() {
    let config = nort_reconstruction_config();
    let mut input = row("nort");
    input["analysis_start_s"] = json!(0);
    input["analysis_duration_s"] = json!(100);
    input["object_configuration"] = json!("counterbalanced");
    input["candidate_segments"] = json!([segment("r", "right", 4.0, 10.0)]);
    let scored = score(&input, &config).unwrap();
    assert_eq!(scored["TN"], 6.0);
    assert!(scored["TF"].is_null() && scored["DI"].is_null());
    let mut unregistered = config.clone();
    unregistered["decision_resolutions"]
        .as_object_mut()
        .unwrap()
        .remove("D07");
    let scored = score(&input, &unregistered).unwrap();
    assert!(
        scored["blocked_outputs"]
            .as_array()
            .unwrap()
            .contains(&json!("nort_bout_reconstruction:D07/D10"))
    );
    // A registered window needs a positive duration; zero is not a session.
    input["analysis_duration_s"] = json!(0);
    assert!(
        format!("{:#}", score(&input, &config).unwrap_err()).contains("finite positive session")
    );
}

#[test]
fn nort_side_bias_follows_physical_sides_and_pseudocount_requires_d13() {
    use bekiper_feature::heuristic::combined::exploration::NortBoutPolicy;
    let mut input = nort_row("training");
    input["TA"] = json!(1.0);
    input["TB"] = json!(3.0);
    let mut assignment = input.clone();
    assignment["roles"] = json!({"TA":"a","TB":"b"});
    assignment["physical_sides"] = json!({"TA":"left","TB":"right"});
    let mut config = config();
    config["assays"]["nort"]["bout_policy"] =
        serde_json::to_value(NortBoutPolicy::default()).unwrap();
    register_choice(
        &mut config,
        "D10",
        "assays.nort.object_role_and_location_mapping",
        json!({"sessions":[assignment]}),
    );
    let scored = score(&input, &config).unwrap();
    assert_eq!(scored["side_bias"], 0.5);
    assert_eq!(scored["object_bias"], -0.5);
    input["TA"] = json!(0.0);
    input["TB"] = json!(0.0);
    let scored = score(&input, &config).unwrap();
    assert!(scored["side_bias"].is_null());
    assert_eq!(scored["physical_side_mapping"]["TB"], "right");
    let mut test = row("nort");
    test["TN"] = json!(4.0);
    test["TF"] = json!(0.0);
    test["protocol_metadata"] = json!({"familiarization_s":300,"delay_h":5,"test_s":360});
    let mut config = self::config();
    config["assays"]["nort"]["lr_pseudocount"] = json!({"enabled":true});
    let scored = score(&test, &config).unwrap();
    assert!(scored["LR"].is_null());
    assert_eq!(
        scored["missing_reasons"]["LR"],
        "blocked:D13:pseudocount_definition"
    );
    assert_eq!(
        scored["protocol_timing_departures"],
        json!(["familiarization_s"])
    );
    assert_eq!(scored["qc_flags"]["protocol_timing_departure"], true);
}

// ----------------------------------------------------------------- fixtures

#[test]
fn policy_fixture_rejects_unparseable_plans_before_claiming_output() {
    let dir = tempdir().unwrap();
    let plan = dir.path().join("plan.toml");
    fs::write(&plan, "statistics = [unterminated").unwrap();
    let output = dir.path().join("fixture");
    assert!(fixtures::prepare_hcar1(&plan, &plan, &output).is_err());
    assert!(!output.exists());
}
