use bekiper_post_analysis::{RunOptions, inputs::*};
use serde_json::{Value, json};
use std::{fs, io::Write};
use tempfile::tempdir;

fn metadata() -> Value {
    json!({"animal_id":"99","assay":"oft","session_id":"s1","timepoint":"pre","phase":"session","five_xfad":"negative","hcar1":"wt","treatment":"vehicle","group":"control","sex":"female","age_treatment_window":"adult","excluded":false,"exclusion_reason":null})
}

fn approval(value: Value) -> Value {
    json!({"value":{"statistics.subgroup_analysis_plan":value},"scope":["statistics"],"origin":"researcher_decision","rationale":"registered before analysis","approval_or_plan_reference":"plan-v1"})
}

#[test]
fn toml_plan_resolves_policy_and_retains_original_config_byte_identity() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("analysis.toml");
    let text = "run_mode = 'development'\n[assays.ymaze.temporal_policy]\nmin_duration_seconds = 0.0\nduration_comparison = 'at_least'\n";
    fs::write(&path, text).unwrap();
    let config = resolved_config(&path, &Default::default(), &|_| None).unwrap();
    assert_eq!(
        config["assays"]["ymaze"]["temporal_policy"]["min_duration_seconds"],
        0.0
    );
    assert_eq!(
        config["_adapter_config_fingerprint"]["sha256"],
        fingerprint(&path).unwrap()["sha256"]
    );
    fs::write(&path, "run_mode = 'development'\nrun_mode = 'production'\n").unwrap();
    assert!(resolved_config(&path, &Default::default(), &|_| None).is_err());
}

#[test]
fn ymaze_default_is_one_second_and_generic_temporal_default_is_unfiltered() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("default.json");
    fs::write(&path, r#"{"run_mode":"development"}"#).unwrap();
    let config = resolved_config(&path, &Default::default(), &|_| None).unwrap();
    assert_eq!(
        config["assays"]["ymaze"]["temporal_policy"]["min_duration_seconds"],
        1.0
    );
    assert_eq!(
        config["assays"]["ymaze"]["temporal_policy"]["duration_comparison"],
        "at_least"
    );
    assert_eq!(
        bekiper_feature::tolerance::TemporalPolicy::default().min_duration_seconds,
        0.0
    );
    let options = RunOptions {
        ymaze_dwell: bekiper_feature::tolerance::DwellOverrides {
            comparison: Some(bekiper_feature::tolerance::DurationComparison::GreaterThan),
            ..Default::default()
        },
        ..Default::default()
    };
    let config = resolved_config(&path, &options, &|_| None).unwrap();
    assert_eq!(
        config["assays"]["ymaze"]["temporal_policy"]["min_duration_seconds"],
        1.0
    );
    assert_eq!(
        config["assays"]["ymaze"]["temporal_policy"]["duration_comparison"],
        "greater_than"
    );
}

#[test]
fn dwell_policy_has_one_constructor_environment_file_default_chain() {
    use bekiper_feature::tolerance::{DurationComparison, DwellOverrides};
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.json");
    fs::write(
        &path,
        json!({"run_mode":"development","assays":{"ymaze":{"temporal_policy":{
        "min_duration_seconds":0.5,"duration_comparison":"greater_than"}}}})
        .to_string(),
    )
    .unwrap();
    let env = |key: &str| match key {
        "BEKIPER_YMAZE_DWELL_SECONDS" => Some("1.0".into()),
        _ => None,
    };
    let from_file = resolved_config(&path, &Default::default(), &|_| None).unwrap();
    assert_eq!(
        from_file["assays"]["ymaze"]["temporal_policy"]["min_duration_seconds"],
        0.5
    );
    let from_env = resolved_config(&path, &Default::default(), &env).unwrap();
    assert_eq!(
        from_env["assays"]["ymaze"]["temporal_policy"]["min_duration_seconds"],
        1.0
    );
    assert_eq!(
        from_env["assays"]["ymaze"]["temporal_policy"]["duration_comparison"],
        "greater_than"
    );
    let options = RunOptions {
        ymaze_dwell: DwellOverrides {
            seconds: Some(0.0),
            comparison: Some(DurationComparison::AtLeast),
        },
        ..Default::default()
    };
    let from_cli = resolved_config(&path, &options, &env).unwrap();
    assert_eq!(
        from_cli["assays"]["ymaze"]["temporal_policy"]["min_duration_seconds"],
        0.0
    );
    assert_eq!(
        from_cli["assays"]["ymaze"]["temporal_policy"]["duration_comparison"],
        "at_least"
    );
    for bad in ["-1", "NaN", "inf", "junk"] {
        assert!(
            resolved_config(&path, &Default::default(), &|key| (key
                == "BEKIPER_YMAZE_DWELL_SECONDS")
                .then(|| bad.into()))
            .is_err()
        );
        assert!(
            resolved_config(&path, &options, &|key| (key
                == "BEKIPER_YMAZE_DWELL_SECONDS")
                .then(|| bad.into()))
            .is_ok()
        );
    }
    let invalid_comparison =
        |key: &str| (key == "BEKIPER_YMAZE_DWELL_COMPARISON").then(|| "unknown".into());
    assert!(resolved_config(&path, &Default::default(), &invalid_comparison).is_err());
    assert!(resolved_config(&path, &options, &invalid_comparison).is_ok());
}

#[test]
fn overlays_and_report_profile() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.json");
    fs::write(&path, json!({"run_mode":"development","outputs":{"directory":"file","requested_outputs":["metrics"]},"assays":{"ymaze":{"requested_sensitivity_thresholds_arm_length_pct":[25]}}}).to_string()).unwrap();
    let env = |key: &str| match key {
        "BEKIPER_POST_ANALYSIS_OUTPUT" => Some("env".into()),
        "BEKIPER_POST_ANALYSIS_SENSITIVITY_THRESHOLDS" => Some(String::new()),
        _ => None,
    };
    let config = resolved_config(&path, &RunOptions::default(), &env).unwrap();
    assert_eq!(config["outputs"]["directory"], "env");
    assert_eq!(
        config["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"],
        json!([])
    );
    assert_eq!(config["outputs"]["report"]["enabled"], false);
    let options = RunOptions {
        output: Some(dir.path().join("cli")),
        requested_outputs: Some(vec!["report".into(), "report_pdf".into()]),
        sensitivity_thresholds: Some(vec![65]),
        ..Default::default()
    };
    let config = resolved_config(&path, &options, &env).unwrap();
    assert_eq!(
        config["outputs"]["directory"],
        options.output.unwrap().to_string_lossy().as_ref()
    );
    assert_eq!(
        requested_output_ids(&config, &[]).unwrap(),
        ["report", "report_pdf"]
    );
    assert!(resolved_config(&path, &RunOptions::default(), &|_| Some(String::new())).is_err());
    assert!(validate_sensitivity_thresholds(&json!([25, 25])).is_err());
    assert!(validate_sensitivity_thresholds(&json!([25.0])).is_err());
    assert!(
        !requested_output_ids(&json!({}), &[])
            .unwrap()
            .contains(&"report".into())
    );
}

#[test]
fn normalization_preserves_all_count_fields_and_large_identifiers() {
    let counts = [
        "raw_total_arm_entries",
        "valid_total_arm_entries",
        "removed_short_entries",
        "raw_to_scored_entry_difference",
        "dwell_rejected_entries",
        "total_triads",
        "usable_triads",
        "correct_alternations",
        "incorrect_alternations",
        "neutral_triads",
        "valid_arm_A_entries",
        "raw_arm_C_entries",
        "valid_bouts",
        "removed_duration_bouts",
        "removed_speed_bouts",
        "removed_angle_bouts",
    ];
    for field in counts {
        let row = normalize(&json!({field:"8"}), &json!({})).unwrap();
        assert_eq!(row[field].as_f64(), Some(8.0), "{field} must be numeric");
    }
    for field in ["raw_to_scored_entry_difference", "removed_short_entries"] {
        let row = normalize(&json!({field:"-3"}), &json!({})).unwrap();
        assert_eq!(
            row[field], -3,
            "signed legacy differences must survive import"
        );
    }
    let row = normalize(&json!({"animal_id": 9007199254740993u64}), &json!({})).unwrap();
    assert_eq!(row["animal_id"], "9007199254740993");
    assert!(normalize(&json!({"five_xfad": {"status":"AD"}}), &json!({})).is_err());
}

#[test]
fn duplicate_json_keys_are_rejected_before_they_can_replace_exclusions() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("table.json");
    fs::write(&path, r#"[{"excluded":true,"excluded":false}]"#).unwrap();
    assert!(read_table(&path, None).is_err());
    fs::write(
        &path,
        r#"[{"scoring_provenance":{"minimum_dwell_seconds":1,"minimum_dwell_seconds":0}}]"#,
    )
    .unwrap();
    assert!(read_table(&path, None).is_err());
}

#[test]
fn arbitrary_threshold_fields_cannot_make_nort_or_oft_duplicate_sessions_unique() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("metadata.json"),
        json!([metadata()]).to_string(),
    )
    .unwrap();
    let mut first = metadata();
    first["threshold_pct"] = json!(25);
    let mut second = first.clone();
    second["threshold_pct"] = json!(65);
    fs::write(
        dir.path().join("oft.json"),
        json!([first, second]).to_string(),
    )
    .unwrap();
    let config = json!({"inputs":{"metadata_path":"metadata.json","oft_path":"oft.json"}});
    assert!(load_inputs(&config, dir.path()).is_err());
}

#[test]
fn reserved_rule_evidence_cannot_be_smuggled_through_column_aliases() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("metadata.json"),
        json!([metadata()]).to_string(),
    )
    .unwrap();
    let mut row = metadata();
    row["_rule_audit"] = json!({"verified":true});
    fs::write(dir.path().join("oft.json"), json!([row]).to_string()).unwrap();
    let config = json!({"inputs":{"metadata_path":"metadata.json","oft_path":"oft.json",
        "column_mapping":{"oft":{"_rule_audit":"imported_audit_claim"}}}});
    assert!(load_inputs(&config, dir.path()).is_err());
}

#[test]
fn malformed_metadata_and_config_shapes_do_not_turn_into_strings_or_defaults() {
    let mut row = metadata();
    row["sex"] = json!({"female":true});
    assert!(validate_metadata(&[row], &json!({"inputs":{}})).is_err());
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.json");
    fs::write(
        &path,
        json!({"run_mode":"development","outputs":{"directory":42}}).to_string(),
    )
    .unwrap();
    assert!(resolved_config(&path, &RunOptions::default(), &|_| None).is_err());
}

#[test]
fn malformed_approvals_and_synthetic_isolation() {
    let path = "statistics.subgroup_analysis_plan";
    let plan = json!({"strata":[{"field":"sex","levels":["female"],"assays":["oft"]}],"outputs":["models"],"multiplicity":{"method":"holm","family":"sex-models"},"plan_reference":"retained-registration-field"});
    let mut config = json!({"run_mode":"production","statistics":{"subgroup_analysis_plan":plan},"decision_resolutions":{"D09":approval(plan.clone()),"D12":approval(plan)}});
    assert_eq!(
        subgroup_plan(&config).unwrap(),
        Some(config["statistics"]["subgroup_analysis_plan"].clone())
    );
    for invalid in [
        json!(null),
        json!([]),
        json!("bad"),
        json!({"value":[]}),
        json!({"scope":"statistics"}),
    ] {
        let mut malformed = config.clone();
        malformed["decision_resolutions"]["D09"] = invalid;
        assert!(choice(&malformed, path, "D09").is_none());
    }
    config["decision_resolutions"]["D09"]["origin"] = json!("development_fixture");
    assert!(choice(&config, path, "D09").is_none());
    config["run_mode"] = json!("synthetic");
    assert!(choice(&config, path, "D09").is_none());
    config["synthetic_fixture"] = json!(true);
    assert!(choice(&config, path, "D09").is_some());
    config["statistics"]["subgroup_analysis_plan"]["strata"][0]["assays"] = json!(["unknown"]);
    let plan = config["statistics"]["subgroup_analysis_plan"].clone();
    config["decision_resolutions"]["D09"] = approval(plan.clone());
    config["decision_resolutions"]["D12"] = approval(plan);
    assert!(subgroup_plan(&config).is_err());
}

#[test]
fn metadata_cardinality_and_biology_are_checked() {
    let row = metadata();
    assert!(validate_metadata(&[row.clone(), row.clone()], &json!({"inputs":{}})).is_err());
    let mut paired = row.clone();
    paired["timepoint"] = json!("post");
    paired["sex"] = json!("male");
    assert!(validate_metadata(&[row.clone(), paired], &json!({"inputs":{}})).is_err());
    assert!(
        validate_metadata(
            std::slice::from_ref(&row),
            &json!({"inputs":{"factor_levels":{"hcar1":["knockout"]}}})
        )
        .is_err()
    );
    assert!(validate_metadata(&[row], &json!({"inputs":{"group_mapping":{"control":{"five_xfad":"positive","hcar1":"wt","treatment":"vehicle"}}}})).is_err());
    let mut invalid = metadata();
    invalid["excluded"] = json!(true);
    assert!(validate_metadata(&[invalid], &json!({"inputs":{}})).is_err());
}

#[test]
fn joins_retain_unavailable_sessions_and_reject_duplicates_and_reserved_audits() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("metadata.json"),
        json!([metadata()]).to_string(),
    )
    .unwrap();
    let mut config = json!({"inputs":{"metadata_path":"metadata.json"}});
    let (rows, descriptors) = load_inputs(&config, dir.path()).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["measurement_available"], false);
    assert_eq!(descriptors.len(), 1);
    let row = json!({"animal_id":"99","assay":"oft","session_id":"s1","timepoint":"pre","phase":"session","total_distance_m":2});
    fs::write(
        dir.path().join("oft.json"),
        json!([row.clone(), row.clone()]).to_string(),
    )
    .unwrap();
    config["inputs"]["oft_path"] = json!("oft.json");
    assert!(
        load_inputs(&config, dir.path())
            .unwrap_err()
            .to_string()
            .contains("many-to-many")
    );
    let mut reserved = row.clone();
    reserved["_rule_audit"] = json!({});
    fs::write(dir.path().join("oft.json"), json!([reserved]).to_string()).unwrap();
    assert!(
        load_inputs(&config, dir.path())
            .unwrap_err()
            .to_string()
            .contains("adapter-owned")
    );
    fs::write(dir.path().join("oft.json"), json!([row]).to_string()).unwrap();
    fs::write(
        dir.path().join("audit.json"),
        json!({"verified":true}).to_string(),
    )
    .unwrap();
    let hash = fingerprint(&dir.path().join("audit.json")).unwrap()["sha256"].clone();
    config["inputs"]["upstream_rule_provenance"] =
        json!({"oft":{"evidence_files":[{"role":"rule_audit","path":"audit.json","sha256":hash}]}});
    let (rows, descriptors) = load_inputs(&config, dir.path()).unwrap();
    assert_eq!(rows[0]["_rule_audit"]["verified"], true);
    assert_eq!(descriptors.len(), 3);
    fs::write(dir.path().join("audit.json"), "{}").unwrap();
    assert!(load_inputs(&config, dir.path()).is_err());
}

#[test]
fn json_csv_and_normalization() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("table.csv");
    fs::write(
        &path,
        "animal,excluded,freezing_s,valid_arm_entry_sequence\n99,false,,A A B\n",
    )
    .unwrap();
    let row = normalize(&read_table(&path, None).unwrap()[0], &json!({})).unwrap();
    assert_eq!(row["animal_id"], "99");
    assert_eq!(row["excluded"], false);
    assert!(row["freezing_s"].is_null());
    assert_eq!(row["valid_arm_entry_sequence"], json!(["A", "A", "B"]));
    fs::write(&path, "animal,animal\na,b\n").unwrap();
    assert!(read_table(&path, None).is_err());
    assert!(normalize(&json!({"animal":"a","animal_id":"b"}), &json!({})).is_err());
    assert!(normalize(&json!({"freezing_s":"NaN"}), &json!({})).is_err());
    assert!(normalize(&json!({"excluded":"0"}), &json!({})).is_err());
    let path = dir.path().join("table.json");
    fs::write(&path, "[{\"TA\": NaN}]").unwrap();
    assert!(read_table(&path, None).is_err());
    fs::write(&path, "[null]").unwrap();
    assert!(read_table(&path, None).is_err());
}

#[test]
fn ods_preserves_typed_values_and_requires_a_sheet() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("table.ods");
    let xml = r#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><table:table table:name="data"><table:table-row><table:table-cell><text:p>animal</text:p></table:table-cell><table:table-cell><text:p>excluded</text:p></table:table-cell><table:table-cell><text:p>freezing_s</text:p></table:table-cell></table:table-row><table:table-row><table:table-cell office:value-type="float" office:value="99"/><table:table-cell office:value-type="boolean" office:boolean-value="false"/><table:table-cell/></table:table-row></table:table></office:document-content>"#;
    let mut archive = zip::ZipWriter::new(fs::File::create(&path).unwrap());
    archive
        .start_file("content.xml", zip::write::SimpleFileOptions::default())
        .unwrap();
    archive.write_all(xml.as_bytes()).unwrap();
    archive.finish().unwrap();
    assert!(read_table(&path, None).is_err());
    assert!(read_table(&path, Some("missing")).is_err());
    let row = normalize(&read_table(&path, Some("data")).unwrap()[0], &json!({})).unwrap();
    assert_eq!(row["animal_id"], "99");
    assert_eq!(row["excluded"], false);
    assert!(row["freezing_s"].is_null());
}

#[test]
fn parquet_preserves_null_and_boolean_cells() {
    use polars::prelude::*;
    let dir = tempdir().unwrap();
    let path = dir.path().join("table.parquet");
    let mut frame =
        df!("animal" => [99i64], "excluded" => [false], "freezing_s" => [None::<f64>]).unwrap();
    ParquetWriter::new(fs::File::create(&path).unwrap())
        .finish(&mut frame)
        .unwrap();
    let row = normalize(&read_table(&path, None).unwrap()[0], &json!({})).unwrap();
    assert_eq!(row["animal_id"], "99");
    assert_eq!(row["excluded"], false);
    assert!(row["freezing_s"].is_null());
}

#[test]
fn typed_measurement_fields_reject_objects_and_qc_json_keeps_false_flags() {
    for field in ["center_time_s", "TN", "valid_total_arm_entries"] {
        assert!(normalize(&json!({field:{"value":1}}), &json!({})).is_err());
        assert!(normalize(&json!({field:true}), &json!({})).is_err());
    }
    let row = normalize(
        &json!({"qc_flags":r#"{"low_quality":false,"review_required":true}"#,
        "coordinate_samples":r#"[{"t_s":0,"valid":false}]"#,
        "valid_arm_entry_sequence":"[\"A\",\"A\",\"B\"]"}),
        &json!({}),
    )
    .unwrap();
    assert_eq!(row["qc_flags"]["low_quality"], false);
    assert_eq!(row["coordinate_samples"][0]["valid"], false);
    assert_eq!(row["valid_arm_entry_sequence"], json!(["A", "A", "B"]));
    assert!(
        normalize(
            &json!({"scoring_provenance":r#"{"verified":false,"verified":true}"#}),
            &json!({})
        )
        .is_err()
    );
}

#[test]
fn nested_parquet_provenance_round_trips() {
    use polars::prelude::*;
    let dir = tempdir().unwrap();
    let path = dir.path().join("table.parquet");
    let fields = [
        Series::new("geometry_version".into(), ["imen_radial_vertex_span_v1"]),
        Series::new("apply_dwell_filter".into(), [false]),
    ];
    let provenance = StructChunked::from_series("scoring_provenance".into(), 1, fields.iter())
        .unwrap()
        .into_series();
    let mut frame = DataFrame::new(
        1,
        vec![
            Series::new("animal".into(), [99i64]).into(),
            provenance.into(),
        ],
    )
    .unwrap();
    ParquetWriter::new(fs::File::create(&path).unwrap())
        .finish(&mut frame)
        .unwrap();
    let rows = read_table(&path, None).unwrap();
    assert_eq!(
        rows[0]["scoring_provenance"],
        json!({"geometry_version":"imen_radial_vertex_span_v1","apply_dwell_filter":false})
    );
}

#[test]
fn session_mapping_and_table_defaults_join_without_inventing_identifiers() {
    let dir = tempdir().unwrap();
    let mut meta = metadata();
    meta["timepoint"] = json!("baseline");
    fs::write(dir.path().join("metadata.json"), json!([meta]).to_string()).unwrap();
    fs::write(
        dir.path().join("oft.csv"),
        "animal,total_distance_m\n99,2.5\n",
    )
    .unwrap();
    let config = json!({"inputs":{"metadata_path":"metadata.json",
        "session_key_mapping":{"timepoint":{"baseline":"pre"}},
        "oft_path":{"path":"oft.csv","constants":{"assay":"oft","session_id":"s1","timepoint":"baseline","phase":"session"}}}});
    let (rows, descriptors) = load_inputs(&config, dir.path()).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["timepoint"], "pre");
    assert_eq!(rows[0]["total_distance_m"], 2.5);
    assert_eq!(
        descriptors[1]["bytes"],
        fs::metadata(dir.path().join("oft.csv")).unwrap().len()
    );
    fs::write(dir.path().join("oft.csv"), "total_distance_m\n2.5\n").unwrap();
    assert!(load_inputs(&config, dir.path()).is_err());
}

#[test]
fn metadata_and_measurement_qc_claims_remain_visible_alongside_conflicts() {
    let dir = tempdir().unwrap();
    let mut meta = metadata();
    meta["qc_flags"] = json!({"review_required":true,"low_quality":true});
    fs::write(dir.path().join("metadata.json"), json!([meta]).to_string()).unwrap();
    let mut row = metadata();
    row["qc_flags"] = json!({"low_quality":false,"upstream_tracking_gap":true});
    fs::write(dir.path().join("oft.json"), json!([row]).to_string()).unwrap();
    let config = json!({"inputs":{"metadata_path":"metadata.json","oft_path":"oft.json"}});
    let (rows, _) = load_inputs(&config, dir.path()).unwrap();
    assert_eq!(
        rows[0]["qc_flags"],
        json!({"review_required":true,"low_quality":false,"upstream_tracking_gap":true})
    );
    assert_eq!(rows[0]["metadata_qc_flags"]["low_quality"], true);
    assert_eq!(rows[0]["measurement_qc_flags"]["low_quality"], false);
    assert_eq!(
        rows[0]["upstream_qc_join_conflicts"]["low_quality"],
        json!({"metadata":true,"measurement":false})
    );
    assert_eq!(rows[0]["excluded"], false);
}

#[test]
fn empty_registered_assays_keep_their_output_attempts() {
    let config = json!({"inputs":{"ymaze_path":"empty-ymaze.json","nort_path":"empty-nort.json"}});
    let requested = requested_output_ids(&config, &[]).unwrap();
    assert!(requested.contains(&"correlations".into()));
    assert!(requested.contains(&"within_group_tests".into()));
}

#[test]
fn ymaze_threshold_identity_is_unique_and_conflicting_threshold_units_are_rejected() {
    let dir = tempdir().unwrap();
    let mut meta = metadata();
    meta["assay"] = json!("ymaze");
    fs::write(
        dir.path().join("metadata.json"),
        json!([meta.clone()]).to_string(),
    )
    .unwrap();
    let rows: Vec<_> = [25, 45, 65]
        .into_iter()
        .map(|pct| {
            let mut row = meta.clone();
            row["threshold_pct"] = json!(pct);
            row["valid_arm_entry_sequence"] = json!(["A", "A", "B"]);
            row
        })
        .collect();
    fs::write(dir.path().join("ymaze.json"), json!(rows).to_string()).unwrap();
    let config = json!({"inputs":{"metadata_path":"metadata.json","ymaze_path":"ymaze.json"}});
    let (output, _) = load_inputs(&config, dir.path()).unwrap();
    assert_eq!(output.len(), 3);
    assert_eq!(
        output
            .iter()
            .map(|r| r["threshold_pct"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [25, 45, 65]
    );
    let mut duplicate = rows.clone();
    duplicate.push(rows[0].clone());
    fs::write(dir.path().join("ymaze.json"), json!(duplicate).to_string()).unwrap();
    assert!(load_inputs(&config, dir.path()).is_err());
    let mut conflict = rows[0].clone();
    conflict["proximal_threshold"] = json!(0.45);
    fs::write(dir.path().join("ymaze.json"), json!([conflict]).to_string()).unwrap();
    assert!(load_inputs(&config, dir.path()).is_err());
}

#[test]
fn pinned_table_hashes_and_portable_fingerprints_are_preserved() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("metadata.json");
    fs::write(&path, json!([metadata()]).to_string()).unwrap();
    let hash = fingerprint(&path).unwrap()["sha256"].clone();
    let mut config = json!({"inputs":{"metadata_path":{"path":"metadata.json","sha256":hash}}});
    let (_, descriptors) = load_inputs(&config, dir.path()).unwrap();
    assert_eq!(descriptors[0]["name"], "metadata.json");
    assert!(descriptors[0].get("path").is_none());
    config["inputs"]["metadata_path"]["sha256"] = json!("0".repeat(64));
    assert!(load_inputs(&config, dir.path()).is_err());
}

#[test]
fn loaded_plan_fingerprint_tracks_resolved_bytes_even_after_path_replacement() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.json");
    fs::write(
        dir.path().join("metadata.json"),
        json!([metadata()]).to_string(),
    )
    .unwrap();
    fs::write(
        &path,
        json!({"run_mode":"development","inputs":{"metadata_path":"metadata.json"}}).to_string(),
    )
    .unwrap();
    let original = fingerprint(&path).unwrap();
    let config = resolved_config(&path, &RunOptions::default(), &|_| None).unwrap();
    fs::write(&path, r#"{"run_mode":"production"}"#).unwrap();
    let (_, descriptors) = load_inputs(&config, dir.path()).unwrap();
    let plan = descriptors
        .iter()
        .find(|r| r["role"] == "post_analysis_plan")
        .expect("decoded plan provenance");
    assert_eq!(plan["sha256"], original["sha256"]);
    assert_ne!(plan["sha256"], fingerprint(&path).unwrap()["sha256"]);
}

#[test]
fn malformed_shapes_and_oversized_floating_identifiers_fail_explicitly() {
    assert!(normalize(&json!({"animal_id":9007199254740992.0}), &json!({})).is_err());
    for invalid in [json!([]), json!(false), json!({"excluded":{}})] {
        assert!(normalize(&invalid, &json!({})).is_err());
    }
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("metadata.json"),
        json!([metadata()]).to_string(),
    )
    .unwrap();
    for field in [
        "column_mapping",
        "session_key_mapping",
        "upstream_rule_provenance",
    ] {
        let config = json!({"inputs":{"metadata_path":"metadata.json",field:[]}});
        assert!(
            load_inputs(&config, dir.path()).is_err(),
            "{field} malformed shape"
        );
    }
    let path = dir.path().join("config.json");
    fs::write(&path, r#"{"run_mode":"synthetic","run_mode":"production"}"#).unwrap();
    assert!(resolved_config(&path, &RunOptions::default(), &|_| None).is_err());
}

#[test]
fn ods_repeated_empty_tail_cells_and_header_groups_preserve_null_measurements() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("repeated.ods");
    let xml = r#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><table:table table:name="data"><table:table-header-rows><table:table-row><table:table-cell><text:p>animal</text:p></table:table-cell><table:table-cell><text:p>freezing_s</text:p></table:table-cell><table:table-cell table:number-columns-repeated="100"/></table:table-row></table:table-header-rows><table:table-row><table:table-cell office:value-type="float" office:value="9007199254740993"/><table:table-cell table:number-columns-repeated="100"/></table:table-row></table:table></office:document-content>"#;
    let mut archive = zip::ZipWriter::new(fs::File::create(&path).unwrap());
    archive
        .start_file("content.xml", zip::write::SimpleFileOptions::default())
        .unwrap();
    archive.write_all(xml.as_bytes()).unwrap();
    archive.finish().unwrap();
    let rows = read_ods(&path, Some("data")).unwrap();
    let row = normalize(&rows[0], &json!({})).unwrap();
    assert_eq!(row["animal_id"], "9007199254740993");
    assert!(row["freezing_s"].is_null());
}

#[test]
fn registered_factor_and_group_mappings_are_validated_even_with_empty_metadata() {
    let config = json!({"inputs":{"factor_levels":{"five_xfad":["negative","positive"],"hcar1":["wt","ko"],"treatment":["vehicle","lactate"]},
        "group_mapping":{"G0":{"five_xfad":"negative","hcar1":"wt","treatment":"vehicle"}}}});
    assert!(validate_metadata(&[], &config).is_ok());
    for invalid in [
        json!({"G0":{"five_xfad":"unexpected","hcar1":"wt","treatment":"vehicle"}}),
        json!({"G0":[]}),
        json!({"G0":{"five_xfad":"negative","hcar1":"wt"}}),
    ] {
        let mut config = config.clone();
        config["inputs"]["group_mapping"] = invalid;
        assert!(validate_metadata(&[], &config).is_err());
    }
    let mut invalid = config;
    invalid["inputs"]["factor_levels"]["treatment"] = json!(["vehicle", "vehicle"]);
    assert!(validate_metadata(&[], &invalid).is_err());
}
#[test]
fn source_ods_retains_literal_text_and_physical_rows_across_blank_repeats() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.ods");
    let mut archive = zip::ZipWriter::new(fs::File::create(&path).unwrap());
    archive
        .start_file("content.xml", zip::write::SimpleFileOptions::default())
        .unwrap();
    archive.write_all(br#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><table:table table:name="animal"><table:table-row><table:table-cell><text:p>Animal</text:p></table:table-cell><table:table-cell><text:p>HCAR1</text:p></table:table-cell></table:table-row><table:table-row table:number-rows-repeated="3"><table:table-cell/></table:table-row><table:table-row><table:table-cell><text:p>37</text:p></table:table-cell><table:table-cell><text:p> Het/KO</text:p></table:table-cell></table:table-row></table:table></office:document-content>"#).unwrap();
    archive.finish().unwrap();
    let before = fs::read(&path).unwrap();
    let rows = bekiper_post_analysis::inputs::read_ods_source(&path, "animal").unwrap();
    assert_eq!(rows[0]["source_row"], 5);
    assert_eq!(rows[0]["HCAR1"], " Het/KO");
    assert_eq!(fs::read(path).unwrap(), before);
}
