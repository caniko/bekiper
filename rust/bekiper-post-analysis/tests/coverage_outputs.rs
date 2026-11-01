//! Presentation outputs fail closed on malformed figure inventories and keep
//! unavailable, disabled and blocked figures visible instead of inventing data.
use std::{fs, path::Path};

use bekiper_post_analysis::{
    Record, RunOptions, export, pipeline,
    plotting::{build_bundle, build_descriptive_bundle, disabled_bundle},
};
use serde_json::{Value, json};

fn config() -> Value {
    let mut value = json!({"run_mode":"synthetic","synthetic_fixture":true,
        "assays":{"ymaze":{"primary_threshold_arm_length_pct":45,"requested_sensitivity_thresholds_arm_length_pct":[65]}},
        "statistics":{"summary_uncertainty":"SEM"},
        "inclusion":{"upstream_verification_requirement":"allow_unverified"},
        "outputs":{"figure_formats":["svg"]},"decision_resolutions":{}});
    for (decision, path, setting) in [
        (
            "D03",
            "assays.ymaze.primary_threshold_arm_length_pct",
            json!(45),
        ),
        ("D09", "statistics.summary_uncertainty", json!("SEM")),
        (
            "D12",
            "inclusion.upstream_verification_requirement",
            json!("allow_unverified"),
        ),
    ] {
        value["decision_resolutions"][decision] = json!({"value":{path:setting},
            "scope":["assays","statistics","inclusion"],"origin":"development_fixture",
            "rationale":"test-only","approval_or_plan_reference":"coverage fixture"});
    }
    value
}

fn selecting(names: &[&str]) -> Value {
    let mut value = config();
    value["outputs"]["figure_selection"] = json!(names);
    value
}

fn verification(value: &mut Value, requirement: &str) {
    value["inclusion"]["upstream_verification_requirement"] = requirement.into();
    value["decision_resolutions"]["D12"]["value"]["inclusion.upstream_verification_requirement"] =
        requirement.into();
}

fn row(animal: &str, assay: &str, timepoint: &str, phase: &str) -> Value {
    json!({"animal_id":animal,"assay":assay,"session_id":format!("{animal}_{assay}_{timepoint}_{phase}"),
        "timepoint":timepoint,"phase":phase,"group":"G5","sex":"female","excluded":false,
        "exclusion_reason":null,"input_verification":{"status":"unresolved"},
        "five_xfad":"AD","hcar1":"WT","treatment":"lactate",
        "qc_flags":{"low_quality":true},"total_distance_m":2,"mean_speed_m_per_s":0.02,
        "center_time_pct":20,"center_entries":4,"latency_to_first_center_entry_s":10})
}

fn ymaze(animal: &str, timepoint: &str, sa: f64) -> Value {
    let mut value = row(animal, "ymaze", timepoint, "session");
    value["threshold_pct"] = 45.into();
    value["primary_threshold"] = true.into();
    value["spontaneous_alternation_pct"] = sa.into();
    value
}

fn figure<'a>(bundle: &'a Value, name: &str) -> &'a Value {
    bundle["figures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == name)
        .unwrap_or_else(|| panic!("missing figure {name}"))
}

fn unrendered<'a>(bundle: &'a Value, name: &str) -> &'a Value {
    bundle["unrendered_figures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == name)
        .unwrap_or_else(|| panic!("missing unrendered entry {name}"))
}

fn error(rows: &[Value], configuration: &Value) -> String {
    format!("{:#}", build_bundle(rows, configuration).unwrap_err())
}

fn state(seconds: [f64; 2], percentages: [f64; 2], frames: [u64; 2]) -> Value {
    json!({"kind":"state_distribution","label":"attention_state_distribution_native",
        "basis":"recorded half-open clock","source_columns":["observed"],
        "category_labels":["target","Unassigned"],"frame_counts":frames,"seconds":seconds,
        "percentages":percentages,"denominator_frames":frames[0] + frames[1],
        "denominator_seconds":seconds[0] + seconds[1]})
}

#[test]
fn empty_selection_disables_every_registered_figure() {
    let mut configuration = config();
    configuration["outputs"]["figure_selection"] = json!([]);
    let bundle = disabled_bundle(&configuration).unwrap();
    let omitted = bundle["unrendered_figures"].as_array().unwrap();
    assert!(!omitted.is_empty());
    assert!(
        omitted
            .iter()
            .all(|f| f["status"] == "disabled" && f["reason"] == "empty_figure_selection")
    );
}

#[test]
fn malformed_registrations_and_figure_lists_are_rejected() {
    let mut unknown = config();
    verification(&mut unknown, "trust_everything");
    assert!(error(&[], &unknown).contains("Unknown upstream verification requirement"));
    let mut formats = config();
    formats["outputs"]["figure_formats"] = json!("svg");
    assert!(error(&[], &formats).contains("figure_formats must be an array"));
    let mut selection = config();
    selection["outputs"]["figure_selection"] = json!("figure_oft");
    assert!(error(&[], &selection).contains("figure_selection must be an array"));
}

#[test]
fn selected_unavailable_and_empty_sensitivity_figures_remain_listed() {
    let bundle = build_bundle(
        &[row("a", "oft", "post", "session")],
        &selecting(&["diagnostic_nort_states"]),
    )
    .unwrap();
    assert!(bundle["figures"].as_array().unwrap().is_empty());
    assert_eq!(
        unrendered(&bundle, "diagnostic_nort_states")["reason"],
        "no_assay_measurements"
    );
    let mut empty = selecting(&["supplement_ymaze"]);
    empty["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"] = json!([]);
    let bundle = build_bundle(&[], &empty).unwrap();
    let supplement = unrendered(&bundle, "supplement_ymaze");
    assert_eq!(supplement["status"], "disabled");
    assert_eq!(supplement["reason"], "empty_sensitivity_selection");
    // A selected distribution family without measurements keeps no empty page.
    let bundle = build_bundle(&[], &selecting(&["figure_nort_distributions"])).unwrap();
    assert!(bundle["figures"].as_array().unwrap().is_empty());
    assert_eq!(
        unrendered(&bundle, "figure_nort_distributions")["status"],
        "unavailable"
    );
}

#[test]
fn nonnumeric_supplementary_values_fail_instead_of_dropping_animals() {
    let mut sensitivity = ymaze("a", "post", 50.0);
    sensitivity["threshold_pct"] = 65.into();
    let mut bad = sensitivity.clone();
    bad["spontaneous_alternation_pct"] = "fifty".into();
    assert!(error(&[bad], &selecting(&["supplement_ymaze"])).contains("nonnumeric plot value"));
    sensitivity["spontaneous_alternation_pct_standard"] = "fifty".into();
    assert!(
        error(&[sensitivity], &selecting(&["supplement_ymaze"]))
            .contains("spontaneous_alternation_pct_standard: nonnumeric")
    );
    let mut training = row("a", "nort", "post", "training");
    training["object_bias"] = 0.1.into();
    training["side_bias"] = "left".into();
    assert!(error(&[training], &selecting(&["supplement_nort"])).contains("side_bias"));
    let mut test = row("a", "nort", "post", "test");
    test["valid_bouts"] = "many".into();
    assert!(error(&[test], &selecting(&["supplement_nort"])).contains("valid_bouts"));
}

#[test]
fn nort_panels_reject_each_nonnumeric_exploration_field() {
    for (phase, field) in [
        ("training", "TA"),
        ("test", "TF"),
        ("test", "percent_novel"),
        ("test", "total_test_exploration"),
    ] {
        let mut value = row("a", "nort", "post", phase);
        value[field] = "x".into();
        assert!(
            error(&[value], &selecting(&["figure_nort"])).contains(field),
            "{field}"
        );
    }
    let mut training = row("a", "nort", "post", "training");
    training["TA"] = 3.into();
    training["TB"] = 2.into();
    training["object_bias"] = "x".into();
    assert!(error(&[training], &selecting(&["figure_nort_distributions"])).contains("object_bias"));
}

#[test]
fn ymaze_pairing_panels_fail_closed_on_inconsistent_inputs() {
    let configuration = selecting(&["figure_ymaze"]);
    let mut noncanonical = ymaze("a", "post", 50.0);
    noncanonical["timepoint"] = "mid".into();
    assert!(error(&[noncanonical], &configuration).contains("canonical pre/post"));
    let mut excluded = ymaze("a", "post", 50.0);
    excluded["excluded"] = "no".into();
    assert!(error(&[excluded], &configuration).contains("explicit boolean excluded"));
    // Each post value and delta is finite; their summary arithmetic overflows.
    let rows = [
        ymaze("a", "pre", 1.5e308),
        ymaze("a", "post", 0.0),
        ymaze("b", "pre", -1.5e308),
        ymaze("b", "post", 0.0),
    ];
    assert!(error(&rows, &configuration).contains("Summary arithmetic overflow"));
    // A post record that is visible only as missing group metadata still pairs;
    // its pre record must carry a plotting group.
    let mut pre = ymaze("a", "pre", 40.0);
    let mut post = ymaze("a", "post", 60.0);
    for value in [&mut pre, &mut post] {
        value["group"] = Value::Null;
    }
    post["genotype_eligibility"] = json!({"eligible":true});
    assert!(error(&[pre.clone(), post.clone()], &configuration).contains("group: required"));
    pre["genotype_eligibility"] = json!({"eligible":true});
    let bundle = build_bundle(&[pre, post], &configuration).unwrap();
    let paired = &figure(&bundle, "figure_ymaze")["panels"][2];
    assert!(paired["points"].as_array().unwrap().is_empty());
    assert!(
        paired["eligibility_ledger"]
            .as_array()
            .unwrap()
            .iter()
            .all(|entry| entry["reason"] == "missing_group_metadata")
    );
    // Reused pre-session identities cannot silently merge two pair series.
    let mut rows = Vec::new();
    for (pair, group, sa) in [("first", "G5", 40.0), ("second", "G6", 45.0)] {
        for (timepoint, session) in [("pre", "shared_pre"), ("post", pair)] {
            let mut value = ymaze("a", timepoint, sa);
            value["pair_id"] = pair.into();
            value["group"] = group.into();
            value["session_id"] = session.into();
            rows.push(value);
        }
    }
    assert!(error(&rows, &configuration).contains("repeated animal replicates"));
}

#[test]
fn point_eligibility_distinguishes_group_metadata_and_verification() {
    let mut missing_group = row("a", "oft", "post", "session");
    missing_group["group"] = Value::Null;
    missing_group["genotype_eligibility"] = json!({"eligible":true});
    let bundle = build_bundle(&[missing_group], &selecting(&["figure_oft"])).unwrap();
    let panel = &figure(&bundle, "figure_oft")["panels"][0];
    assert_eq!(
        panel["eligibility_ledger"][0]["reason"],
        "missing_group_metadata"
    );
    assert_eq!(panel["eligibility_ledger"][0]["qc_excluded"], false);
    let mut verified = selecting(&["figure_oft"]);
    verification(&mut verified, "require_verified");
    let bundle = build_bundle(&[row("a", "oft", "post", "session")], &verified).unwrap();
    let panel = &figure(&bundle, "figure_oft")["panels"][0];
    assert_eq!(
        panel["eligibility_ledger"][0]["reason"],
        "upstream_unverified:unresolved"
    );
    assert_eq!(panel["n_animals"], 0);
    let mut confirmed = row("b", "oft", "post", "session");
    confirmed["input_verification"]["status"] = "verified_source_rule_definition".into();
    confirmed["genotype_eligibility"] = json!({"eligible":true});
    let bundle = build_bundle(&[confirmed], &verified).unwrap();
    let panel = &figure(&bundle, "figure_oft")["panels"][0];
    assert_eq!(panel["eligibility_ledger"][0]["reason"], Value::Null);
    assert_eq!(panel["n_animals"], 1);
}

#[test]
fn subgroup_figures_follow_selection_and_primary_threshold_registration() {
    let mut configuration = config();
    let plan = json!({"strata":[{"field":"sex","levels":["female"],"assays":["nort","ymaze"]}],
        "outputs":["models"],"multiplicity":{"method":"holm","family":"fixture_sex_models"}});
    configuration["statistics"]["subgroup_analysis_plan"] = plan.clone();
    for decision in ["D09", "D12"] {
        configuration["decision_resolutions"][decision]["value"]["statistics.subgroup_analysis_plan"] =
            plan.clone();
    }
    let mut test = row("a", "nort", "post", "test");
    test["DI"] = "x".into();
    let mut only_subgroup = configuration.clone();
    only_subgroup["outputs"]["figure_selection"] = json!(["subgroup_nort_sex_0"]);
    assert!(error(&[test.clone()], &only_subgroup).contains("DI: nonnumeric"));
    let mut only_oft = configuration.clone();
    only_oft["outputs"]["figure_selection"] = json!(["figure_oft"]);
    let bundle = build_bundle(&[test], &only_oft).unwrap();
    assert_eq!(
        unrendered(&bundle, "subgroup_nort_sex_0")["status"],
        "not_selected"
    );
    let mut unthresholded = configuration;
    unthresholded["decision_resolutions"]
        .as_object_mut()
        .unwrap()
        .remove("D03");
    let bundle = build_bundle(&[ymaze("a", "post", 50.0)], &unthresholded).unwrap();
    for panel in figure(&bundle, "subgroup_ymaze_sex_0")["panels"]
        .as_array()
        .unwrap()
    {
        assert_eq!(panel["reason"], "blocked:D03:primary_threshold");
        assert_eq!(panel["n_animals"], 0);
    }
}

#[test]
fn native_state_panels_keep_unrecorded_and_zero_duration_sessions_missing() {
    let configuration = selecting(&["diagnostic_nort_states"]);
    let mut recorded = row("a", "nort", "post", "test");
    recorded["native_state_distributions"] = json!([state([0.5, 0.5], [50.0, 50.0], [1, 1])]);
    let mut zero = row("b", "nort", "post", "test");
    zero["native_state_distributions"] = json!([state([0.0, 0.0], [0.0, 0.0], [0, 0])]);
    let unrecorded = row("c", "nort", "post", "test");
    let bundle = build_bundle(&[recorded.clone(), zero, unrecorded], &configuration).unwrap();
    let panel = &figure(&bundle, "diagnostic_nort_states")["panels"][0];
    assert_eq!(panel["n_animals"], 1);
    let reasons = panel["eligibility_ledger"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| (entry["animal_id"].clone(), entry["reason"].clone()))
        .collect::<Vec<_>>();
    assert!(reasons.contains(&(json!("b"), json!("zero_session_duration"))));
    assert!(reasons.contains(&(json!("c"), json!("native_state_not_recorded"))));
    recorded["excluded"] = "no".into();
    assert!(error(&[recorded], &configuration).contains("explicit boolean excluded"));
}

#[test]
fn descriptive_bundle_rejects_malformed_formats_and_exclusions() {
    let value = json!({"animal_id":"a","session_id":"one","analysis_id":"one","assay":"oft",
        "timepoint":"post","phase":"session","group":"2","excluded":false,"value":2.0,
        "scalar_endpoints":["value"],"presentation_partition":{},"presentation_label":"OFT post"});
    assert!(
        build_descriptive_bundle(
            std::slice::from_ref(&value),
            &json!({"outputs":{"figure_formats":"png"}})
        )
        .is_err()
    );
    let mut excluded = value;
    excluded["excluded"] = "no".into();
    assert!(build_descriptive_bundle(&[excluded], &json!({})).is_err());
}

#[test]
fn package_paths_keep_logs_internal() {
    assert_eq!(
        export::relative_path("renderer.stderr.log"),
        "_internal/logs/renderer.stderr.log"
    );
    assert_eq!(
        export::relative_path("report.compiler.stdout.log"),
        "_internal/logs/report.compiler.stdout.log"
    );
    assert_eq!(
        export::relative_path("report.child.1.stdout.log"),
        "_internal/report/report.child.1.stdout.log"
    );
}

#[test]
fn figure_organization_validates_names_formats_and_regular_files() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let organize = |figure: Value| {
        let mut figures = [figure];
        export::organize_figures(root, &mut figures).map(|()| figures[0].clone())
    };
    for (figure, message) in [
        (json!({"formats":["svg"]}), "figure needs name"),
        (
            json!({"name":"../escape","formats":["svg"]}),
            "unsafe figure name",
        ),
        (json!({"name":"figure_oft"}), "figure needs formats"),
        (json!({"name":"figure_oft","formats":[5]}), "figure format"),
        (
            json!({"name":"figure_oft","formats":["tiff"]}),
            "unsupported figure format",
        ),
    ] {
        assert!(
            format!("{:#}", organize(figure).unwrap_err()).contains(message),
            "{message}"
        );
    }
    fs::write(root.join("figure_oft.svg"), "<svg/>").unwrap();
    let organized = export::path(root, "figure_nort.pdf");
    fs::create_dir_all(organized.parent().unwrap()).unwrap();
    fs::write(&organized, "%PDF-").unwrap();
    assert_eq!(
        organize(json!({"name":"figure_oft","formats":["svg"]})).unwrap()["files"],
        json!(["figures/oft/figure_oft.svg"])
    );
    assert!(!root.join("figure_oft.svg").exists());
    assert!(export::path(root, "figure_oft.svg").is_file());
    assert_eq!(
        organize(json!({"name":"figure_nort","formats":["pdf"]})).unwrap()["files"],
        json!(["figures/nort/figure_nort.pdf"])
    );
    // Missing artifacts and directories never satisfy a declared figure.
    assert!(organize(json!({"name":"figure_ymaze","formats":["png"]})).is_err());
    fs::create_dir(root.join("figure_ymaze.png")).unwrap();
    assert!(
        format!(
            "{:#}",
            organize(json!({"name":"figure_ymaze","formats":["png"]})).unwrap_err()
        )
        .contains("regular file")
    );
    fs::create_dir_all(export::path(root, "qc_nort.svg")).unwrap();
    assert!(
        format!(
            "{:#}",
            organize(json!({"name":"qc_nort","formats":["svg"]})).unwrap_err()
        )
        .contains("organized figure must be a regular file")
    );
}

fn session(animal: &str) -> Value {
    json!({"animal_id":animal,"assay":"oft","session_id":"session","timepoint":"post",
        "phase":"session","total_distance_m":1.0})
}

#[test]
fn finalized_index_lists_report_and_figure_files() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::write(root.join("report.pdf"), "%PDF-").unwrap();
    let figure = export::path(root, "figure_oft.svg");
    fs::create_dir_all(figure.parent().unwrap()).unwrap();
    fs::write(figure, "<svg/>").unwrap();
    let mut manifest = json!({"run_mode":"synthetic","status":"partial","real_data_execution":false,
        "resolved_config":{},"figures":[{"name":"figure_oft","files":["figures/oft/figure_oft.svg"]}]});
    export::finalize(root, &mut manifest, &[session("a")]).unwrap();
    assert_eq!(
        manifest["package_layout"]["paths"]["figure_oft.svg"],
        "figures/oft/figure_oft.svg"
    );
    let index = fs::read_to_string(root.join("README.md")).unwrap();
    assert!(index.contains("- [PDF report](report.pdf)"));
    assert!(index.contains("- [figures/oft/figure_oft.svg](figures/oft/figure_oft.svg)"));
    // The human source index is bound to the canonical record position.
    let metrics = fs::read_to_string(root.join("metrics.toml")).unwrap();
    assert!(metrics.contains("source_record_index = 0"));
    fs::write(
        root.join("metrics.toml"),
        metrics.replace("source_record_index = 0", "source_record_index = -1"),
    )
    .unwrap();
    assert!(export::verify_metrics_export(root, &manifest, &[session("a")]).is_err());
}

#[test]
fn finalize_rejects_unknown_assays_and_malformed_figure_paths() {
    let manifest =
        json!({"run_mode":"synthetic","status":"partial","resolved_config":{},"figures":[]});
    let directory = tempfile::tempdir().unwrap();
    let mut unknown = session("a");
    unknown["assay"] = "rotarod".into();
    let error = export::finalize(directory.path(), &mut manifest.clone(), &[unknown]).unwrap_err();
    assert!(error.to_string().contains("unsupported human metric assay"));
    for files in [json!([5]), json!([".."])] {
        let directory = tempfile::tempdir().unwrap();
        let mut manifest = manifest.clone();
        manifest["figures"] = json!([{"name":"figure_oft","files":files}]);
        assert!(export::finalize(directory.path(), &mut manifest, &[session("a")]).is_err());
    }
}

fn source_run(output: &Path) -> Record {
    pipeline::run_records(
        json!({"run_mode":"synthetic","synthetic_fixture":true,
            "inputs":{"units_and_calibration":{"oft":{"time":"s","distance":"m","speed":"m/s"}}},
            "assays":{"oft":{},"ymaze":{},"nort":{}},
            "outputs":{"requested_outputs":["metrics"]}}),
        vec![
            json!({"animal_id":"mouse","assay":"oft","session_id":"session",
            "timepoint":"post","phase":"session","excluded":false,"exclusion_reason":null,
            "measurement_available":true,"total_distance_m":1.0,"synthetic_observation":true}),
        ],
        vec![],
        &output.parent().unwrap().join("fixture.json"),
        &RunOptions {
            output: Some(output.into()),
            ..Default::default()
        },
    )
    .unwrap()
}

fn retained_failure(parent: &Path, name: &str) -> Value {
    let prefix = format!(".{name}.staging-");
    let staging = fs::read_dir(parent)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(&prefix)
        })
        .expect("retained staging evidence");
    serde_json::from_slice(&fs::read(staging.join("failure_manifest.json")).unwrap()).unwrap()
}

#[test]
fn failed_presentation_export_retains_evidence_and_publishes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source");
    source_run(&source);
    let path = source.join("run_manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["figures"] = json!([{"name":"figure_oft","formats":["svg"]}]);
    pipeline::write_json(&path, &manifest).unwrap();
    pipeline::write_artifact_manifest(&source).unwrap();
    let output = directory.path().join("presentation");
    let error = format!(
        "{:#}",
        export::export_existing(&source, &output).unwrap_err()
    );
    assert!(error.contains("failure evidence retained in"));
    assert!(!output.exists());
    let failure = retained_failure(directory.path(), "presentation");
    assert_eq!(failure["failed_stage"], "presentation_export");
    assert_eq!(failure["published"], false);
    pipeline::verify_artifacts(&source).unwrap();
}

#[test]
fn destination_created_during_export_is_never_replaced() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source");
    source_run(&source);
    // The destination appears after the claim but before the atomic publish.
    // The race is retried until the competing writer lands inside that window.
    for attempt in 0..100 {
        let name = format!("raced-{attempt}");
        let output = directory.path().join(&name);
        let parent = directory.path().to_path_buf();
        let target = output.clone();
        let racer = std::thread::spawn(move || {
            let prefix = format!(".{name}.staging-");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while std::time::Instant::now() < deadline && !target.exists() {
                let staged = fs::read_dir(&parent).unwrap().any(|entry| {
                    entry
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .starts_with(&prefix)
                });
                if staged {
                    let _ = fs::create_dir(&target);
                    return;
                }
            }
        });
        let result = export::export_existing(&source, &output);
        racer.join().unwrap();
        let Err(error) = result else { continue };
        let error = format!("{error:#}");
        assert!(error.contains("appeared while processing"), "{error}");
        assert!(error.contains("failure evidence retained in"));
        assert_eq!(fs::read_dir(&output).unwrap().count(), 0);
        let failure = retained_failure(directory.path(), &format!("raced-{attempt}"));
        assert_eq!(failure["failed_stage"], "presentation_publication");
        return;
    }
    panic!("the competing destination never landed inside the publication window");
}

#[test]
fn blank_decision_fields_never_resolve_a_registered_choice() {
    let mut value = config();
    assert_eq!(
        bekiper_post_analysis::inputs::choice(&value, "statistics.summary_uncertainty", "D09"),
        Some(&json!("SEM"))
    );
    // A whitespace-only rationale is not an approval.
    for field in ["origin", "rationale", "approval_or_plan_reference"] {
        let mut blank = value.clone();
        blank["decision_resolutions"]["D09"][field] = json!("  ");
        assert_eq!(
            bekiper_post_analysis::inputs::choice(&blank, "statistics.summary_uncertainty", "D09"),
            None,
            "{field}"
        );
    }
    value["decision_resolutions"]["D09"]["rationale"] = json!(7);
    assert_eq!(
        bekiper_post_analysis::inputs::choice(&value, "statistics.summary_uncertainty", "D09"),
        None
    );
}

#[test]
fn requested_spatial_location_bias_is_reported_blocked_not_measured() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("run");
    let manifest = pipeline::run_records(
        json!({"run_mode":"synthetic","synthetic_fixture":true,
            "inputs":{"units_and_calibration":{"oft":{"time":"s","distance":"m","speed":"m/s"}}},
            "assays":{"oft":{},"ymaze":{},"nort":{}},
            "outputs":{"requested_outputs":["metrics","spatial_location_bias"]}}),
        vec![
            json!({"animal_id":"mouse","assay":"oft","session_id":"session",
            "timepoint":"post","phase":"session","excluded":false,"exclusion_reason":null,
            "measurement_available":true,"total_distance_m":1.0,"synthetic_observation":true}),
        ],
        vec![],
        &parent.path().join("fixture.json"),
        &RunOptions {
            output: Some(output),
            ..Default::default()
        },
    )
    .unwrap();
    let status = &manifest["output_statuses"]["spatial_location_bias"];
    assert_eq!(status["status"], "blocked");
    assert_eq!(status["decision_ids"], json!(["D08"]));
    assert!(
        manifest["supplementary_output_statuses"]
            .as_array()
            .unwrap()
            .contains(status)
    );
}
