use super::*;
use serde_json::json;
use std::os::unix::process::ExitStatusExt;

fn example() -> (Record, Vec<Record>, Vec<Record>, Vec<Record>) {
    let manifest = json!({"schema_version":"2.0","status":"partial","run_mode":"synthetic",
        "synthetic_fixture":true,"real_data_execution":false,"requested_outputs":["models","report"],
        "output_statuses":{"models":{"status":"partial","reason":"rank deficient"},"report":{"status":"complete"}},
        "source_recipe":{"name":"BEHAVIORAL_RECIPE.md","sha256":"recipe-hash"},
        "ymaze_amendment":{"sha256":"amendment-hash"},"code":{"version":"0.1.0","source_sha256":"engine-hash"},
        "limitations":["D03: primary threshold unregistered"]});
    let rows = vec![
        json!({"animal_id":"fixture1","assay":"nort","excluded":false,
        "input_verification":{"status":"upstream_unverified"},"qc_flags":{"low_quality":true}}),
        json!({"animal_id":"fixture2","assay":"nort","excluded":true,"exclusion_reason":"registered metadata exclusion"}),
    ];
    let models = vec![
        json!({"model_id":"nort.DI","status":"fitted","endpoint":"DI","n_animals":15,
        "formula":"DI ~ five_xfad * hcar1 * treatment","settings":{"confidence_level":0.95},
        "coefficients":[{"term":"Intercept","estimate":0.1,"standard_error":0.02,"p_raw":1e-20}],
        "contrast":{"name":"(G5-G6)-(G1-G2)","estimate":0.25,"standard_error":0.05,"df":22,
            "ci_lower":0.15,"ci_upper":0.35,"p_raw":1e-20,"p_adjusted":8e-20,
            "correction_family":"primary_models","family_size":8,"correction_method":"holm"}}),
        json!({"model_id":"oft.latency","status":"error","n_animals":31,"reason":"rank_deficient:interactions_retained","contrast":null}),
    ];
    let ledger = vec![
        json!({"animal_id":"fixture1","assay":"nort","endpoint":"DI","phase":"test","timepoint":"post",
        "metric_status":"available","metadata_excluded":false,"eligible_before_model_policy":true}),
        json!({"animal_id":"fixture2","assay":"nort","endpoint":"DI","phase":"test","timepoint":"post",
        "metric_status":"available","metadata_excluded":true,"eligible_before_model_policy":false,"exclusion_reason":"registered metadata exclusion"}),
        json!({"animal_id":"fixture1","assay":"ymaze","endpoint":"primary_threshold","phase":"session","timepoint":"post",
        "metric_status":"blocked","missing_reason":"blocked:D03","metadata_excluded":false,"eligible_before_model_policy":false}),
    ];
    (manifest, rows, models, ledger)
}

#[test]
fn escaped_report_preserves_numeric_results_and_blockers() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, rows, models, ledger) = example();
    let config = json!({"outputs":{"report":{"enabled":false,"title":"Hostile \\input{secret} & 50% # _ ~ ^ $"}}});
    let result = write_report(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &config,
    )
    .unwrap();
    assert_eq!(result["pdf_status"], "not_requested");
    let markdown = std::fs::read_to_string(directory.path().join("report.md")).unwrap();
    assert!(markdown.contains("Synthetic"));
    assert!(markdown.contains("0.25"));
    assert!(markdown.contains("1e-20"));
    assert!(markdown.contains("8e-20"));
    assert!(markdown.contains("primary_models"));
    assert!(markdown.contains("rank_deficient"));
    assert!(markdown.contains("blocked:D03"));
    assert!(markdown.contains("15"));
    assert!(!directory.path().join("report.pdf").exists());
    let escaped = tex_escape("\\input{secret} & 50% # _ ~ ^ $");
    assert!(!escaped.contains("\\input{"));
    assert!(escaped.contains("\\textbackslash{}"));
    assert!(escaped.contains("\\&"));
    assert!(escaped.contains("\\%"));
    assert!(escaped.contains("\\{"));
}

#[test]
fn descriptive_report_counts_available_missing_and_excluded_records_separately() {
    let directory = tempfile::tempdir().unwrap();
    let rows = vec![
        json!({"animal_id":"a","session_id":"one","analysis_id":"one","assay":"oft","timepoint":"post","phase":"session","group":"2","excluded":false,"value":2.0,"scalar_endpoints":["value"],"presentation_partition":{},"presentation_label":"OFT post"}),
        json!({"animal_id":"b","session_id":"two","analysis_id":"two","assay":"oft","timepoint":"post","phase":"session","group":"2","excluded":false,"value":null,"scalar_endpoints":["value"],"presentation_partition":{},"presentation_label":"OFT post"}),
        json!({"animal_id":"c","session_id":"three","analysis_id":"three","assay":"oft","timepoint":"post","phase":"session","group":"2","excluded":true,"exclusion_reason":"recorded exclusion","value":900.0,"scalar_endpoints":["value"],"presentation_partition":{},"presentation_label":"OFT post"}),
    ];
    let bundle = crate::plotting::build_descriptive_bundle(
        &rows,
        &json!({"outputs":{"figures_enabled":false}}),
    )
    .unwrap();
    let mut ledger = bundle["eligibility_ledger"].as_array().unwrap().clone();
    ledger
        .push(json!({"analysis_id":"one","field":"sex","reason":"recorded_metadata_unavailable"}));
    let sections = report_sections(
        directory.path(),
        &json!({"analysis_mode":"recorded_descriptive"}),
        &rows,
        &[],
        &ledger,
        &json!({}),
    )
    .unwrap();
    let section = sections
        .iter()
        .find(|section| section.heading == "Inclusion and measurement availability")
        .expect("descriptive availability table");
    assert_eq!(section.tables[0].rows.len(), 1);
    assert_eq!(&section.tables[0].rows[0][1..], ["1", "2", "1", "1"]);
    let reasons = section.tables[1]
        .rows
        .iter()
        .map(|row| row[0].as_str())
        .collect::<BTreeSet<_>>();
    assert!(reasons.contains("not_measured:value"));
    assert!(reasons.contains("recorded exclusion"));
    assert!(reasons.contains("recorded_metadata_unavailable"));
}

#[test]
fn descriptive_report_keeps_same_threshold_variants_and_scoring_policies_separate() {
    let directory = tempfile::tempdir().unwrap();
    let mut rows = [("compare_0pct_dwell0s", 0.0), ("compare_0pct_dwell1s", 1.0), ("compare_0pct_dwell0s", 0.5)]
        .into_iter().enumerate().map(|(index, (variant, dwell))| {
            json!({"animal_id":"a","session_id":format!("session_{index}"),"analysis_id":format!("analysis_{index}"),
                "assay":"ymaze","timepoint":"post","phase":"session","group":"2","excluded":false,"threshold_pct":0,
                "value":2.0,"scalar_endpoints":["value"],"variant":variant,
                "presentation_partition":{"timepoint":"post","phase":"session","variant":variant,"scoring_lineage":{"minimum_dwell_seconds":dwell}},
                "presentation_label":"Recorded Y-maze"})
        }).collect::<Vec<_>>();
    for field in ["stage", "sequence"] {
        let mut row = rows[0].clone();
        row["analysis_id"] = field.into();
        row["presentation_partition"][field] = "2".into();
        rows.push(row);
    }
    let bundle = crate::plotting::build_descriptive_bundle(
        &rows,
        &json!({"outputs":{"figures_enabled":false}}),
    )
    .unwrap();
    let section = inclusion_section(
        directory.path(),
        &rows,
        bundle["eligibility_ledger"].as_array().unwrap(),
    )
    .unwrap();
    assert_eq!(section.tables[0].rows.len(), 5);
    for row in &section.tables[0].rows {
        assert_eq!(&row[1..], ["1", "1", "0", "0"]);
    }
    let sections = report_sections(
        directory.path(),
        &json!({"analysis_mode":"recorded_descriptive"}),
        &rows,
        &[],
        bundle["eligibility_ledger"].as_array().unwrap(),
        &json!({}),
    )
    .unwrap();
    let scope = sections
        .iter()
        .find(|section| section.heading == "Recorded descriptive measurements")
        .unwrap();
    assert_eq!(scope.tables[0].rows.len(), 5);
}

#[test]
fn descriptive_native_state_counts_keep_stage_sequence_and_scoring_lineage() {
    let directory = tempfile::tempdir().unwrap();
    let rows = [("1", "1", 0.5), ("2", "1", 0.5), ("1", "2", 0.5), ("1", "1", 1.0)]
        .into_iter().enumerate().map(|(index, (stage, sequence, dwell))| {
            json!({"animal_id":"a","session_id":format!("session_{index}"),"analysis_id":format!("analysis_{index}"),
                "assay":"nort","timepoint":"post","phase":"training","group":"2","excluded":false,
                "scalar_endpoints":[],"variant":null,"presentation_label":"Recorded NORT",
                "presentation_partition":{"timepoint":"post","phase":"training","variant":null,
                    "stage":stage,"sequence":sequence,"scoring_lineage":{"minimum_dwell_seconds":dwell}},
                "native_state_distributions":[{"kind":"state_distribution","label":"attention_state_distribution_native",
                    "basis":"recorded half-open clock","source_columns":["observed"],
                    "category_labels":["target","Unassigned"],"frame_counts":[1,1],"seconds":[0.5,0.5],
                    "percentages":[50.0,50.0],"denominator_frames":2,"denominator_seconds":1.0}]})
        }).collect::<Vec<_>>();
    let bundle = crate::plotting::build_descriptive_bundle(
        &rows,
        &json!({"outputs":{"figure_selection":["diagnostic_nort_states"]}}),
    )
    .unwrap();
    let ledger = bundle["eligibility_ledger"].as_array().unwrap();
    assert_eq!(ledger.len(), 8);
    for entry in ledger {
        let source = rows
            .iter()
            .find(|row| row["analysis_id"] == entry["analysis_id"])
            .unwrap();
        assert_eq!(
            entry["presentation_partition"],
            source["presentation_partition"]
        );
    }
    for figure in bundle["figures"].as_array().unwrap() {
        for panel in figure["panels"].as_array().unwrap() {
            let source = rows
                .iter()
                .find(|row| row["analysis_id"] == panel["points"][0]["analysis_id"])
                .unwrap();
            assert_eq!(panel["session_partition"], source["presentation_partition"]);
        }
    }
    let sections = report_sections(
        directory.path(),
        &json!({"analysis_mode":"recorded_descriptive"}),
        &rows,
        &[],
        ledger,
        &json!({}),
    )
    .unwrap();
    let inclusion = sections
        .iter()
        .find(|section| section.heading == "Inclusion and measurement availability")
        .unwrap();
    assert_eq!(inclusion.tables[0].rows.len(), 8);
    for row in &inclusion.tables[0].rows {
        assert_eq!(&row[1..], ["1", "1", "0", "0"]);
    }
}

#[test]
fn source_includes_actual_paired_exclusion_accounting_and_families() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, rows, models, ledger) = example();
    std::fs::write(directory.path().join("within_group_tests.json"), serde_json::to_vec(&json!([
        {"group":"empty_group","endpoint":"DI","n_animals":0,"status":"unavailable","reason":"insufficient_data","p_raw":null,"p_adjusted":null,"family_size":8,"correction_family":"DI_groups"}
    ])).unwrap()).unwrap();
    let sections = report_sections(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &json!({}),
    )
    .unwrap();
    let tex = render_tex("Fixture", &sections, &[], &themekit::DARK).replace("\\allowbreak{}", "");
    assert!(tex.contains("empty\\_group"));
    assert!(tex.contains("DI\\_groups"));
    assert!(tex.contains("insufficient\\_data"));
    assert!(tex.contains("QC flags are not exclusions"));
    assert!(tex.contains("(G5-G6)-(G1-G2)"));
    assert!(tex.contains("recipe-hash"));
    assert!(tex.contains("engine-hash"));
    assert!(tex.contains("metadata excluded"));
}

#[test]
fn literal_zero_p_is_reported_as_unavailable_instead_of_a_discovery() {
    assert!(p_value(&json!({"p_raw":0.0,"log_p_raw":-900.0}), "p_raw").contains("Underflow"));
    assert!(p_value(&json!({"p_raw":null,"log_p_raw":-900.0}), "p_raw").contains("log P"));
}

#[test]
fn adjusted_p_never_inherits_raw_log_probability() {
    let row = json!({"p_raw":1e-20,"log_p_raw":-46.0517,"p_adjusted":null});
    assert_eq!(p_value(&row, "p_adjusted"), "Unavailable");
    assert!(!p_value(&json!({"p_adjusted":0.0,"log_p_raw":-900.0}), "p_adjusted").contains("-900"));
}

#[test]
fn numeric_display_preserves_small_probabilities_and_effect_direction() {
    for number in [
        1e-300,
        1e-20,
        7.728343730224925e-6,
        -0.14999999999999866,
        15.161290322580646,
        123456.789,
    ] {
        let displayed: f64 = text(&json!(number)).parse().unwrap();
        assert!(displayed != 0.0);
        assert!(
            (displayed / number - 1.0).abs() < 5e-6,
            "display lost significant digits: {number} -> {displayed}"
        );
    }
    assert_eq!(text(&json!(31)), "31");
}

#[test]
fn invalid_compiler_type_is_rejected_before_writing() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, rows, models, ledger) = example();
    let result = write_report(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &json!({"outputs":{"report":{"enabled":true,"compiler":5}}}),
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("compiler must be a string")
    );
    assert!(!directory.path().join("report.md").exists());
}

#[test]
fn dependency_inventory_is_summarized_without_expanding_artifacts_into_tex() {
    let files = (0..3000)
        .map(|index| json!({"path":format!("figures/fixture_{index}.pdf"),"sha256":"a".repeat(64),"bytes":100}))
        .collect::<Vec<_>>();
    let manifest =
        json!({"dependencies":{"renderer":{"status":"rendered","python":"3.13", "files":files}}});
    let section = provenance::provenance_section(&manifest).unwrap();
    let tex = render_tex("Fixture", &[section], &[], &themekit::DARK).replace("\\allowbreak{}", "");
    assert!(tex.contains("3.13"));
    assert!(tex.contains("3000 artifact files"));
    assert!(tex.contains("run\\_manifest.json"));
    assert!(!tex.contains("fixture\\_2999"));
    assert!(tex.lines().all(|line| line.len() < 2000));
    assert_eq!(
        manifest["dependencies"]["renderer"]["files"]
            .as_array()
            .unwrap()
            .len(),
        3000
    );
}

#[test]
fn native_plan_fingerprint_is_shown_in_the_report() {
    let directory = tempfile::tempdir().unwrap();
    let (mut manifest, rows, models, ledger) = example();
    manifest["inputs"] = json!([{"name":"plan.json","role":"post_analysis_plan","sha256":"sealed-plan-hash","bytes":100}]);
    write_report(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &json!({}),
    )
    .unwrap();
    let markdown = std::fs::read_to_string(directory.path().join("report.md")).unwrap();
    assert!(markdown.contains("Configuration plan input | sealed-plan-hash"));
}

#[test]
fn compiler_failure_retains_tex_logs_and_failure_record() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, rows, models, ledger) = example();
    let config = json!({"outputs":{"report":{"enabled":true,"compiler":"latexmk"}}});
    let mut commands = Vec::new();
    let result = write_report_with_runner(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &config,
        &mut |command| {
            let environment = command
                .get_envs()
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(
                environment.get(std::ffi::OsStr::new("openin_any")),
                Some(&Some(std::ffi::OsStr::new("r")))
            );
            assert_eq!(
                environment.get(std::ffi::OsStr::new("openout_any")),
                Some(&Some(std::ffi::OsStr::new("p")))
            );
            let args = command
                .get_args()
                .map(|value| value.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            commands.push(args.clone());
            if args.contains(&"-v".into()) || args.contains(&"--version".into()) {
                return Ok(std::process::Output {
                    status: std::process::ExitStatus::from_raw(0),
                    stdout: b"fixture compiler version".to_vec(),
                    stderr: vec![],
                });
            }
            Ok(std::process::Output {
                status: std::process::ExitStatus::from_raw(256),
                stdout: b"compiler trace".to_vec(),
                stderr: b"deliberate compiler failure".to_vec(),
            })
        },
    );
    assert!(result.is_err());
    assert!(commands.iter().any(|args| args.contains(&"-norc".into()) && args.contains(&"-no-shell-escape".into())));
    assert!(crate::export::path(directory.path(), "report.tex").is_file());
    assert!(
        directory
            .path()
            .join("report.compiler.stdout.log")
            .is_file()
    );
    assert!(
        directory
            .path()
            .join("report.compiler.stderr.log")
            .is_file()
    );
    let build: Record =
        serde_json::from_slice(&std::fs::read(directory.path().join("report_build.json")).unwrap())
            .unwrap();
    assert_eq!(build["status"], "error");
    assert!(!directory.path().join("report.pdf").exists());
}

#[test]
fn missing_compiler_retains_source_and_spawn_diagnostic() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, rows, models, ledger) = example();
    let result = write_report_with_runner(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &json!({"outputs":{"report":{"enabled":true,"compiler":"lualatex"}}}),
        &mut |_| {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "fixture compiler missing",
            ))
        },
    );
    assert!(result.is_err());
    let build: Record =
        serde_json::from_slice(&std::fs::read(directory.path().join("report_build.json")).unwrap())
            .unwrap();
    assert_eq!(build["status"], "error");
    assert!(crate::export::path(directory.path(), "report.tex").is_file());
    assert!(
        std::fs::read_to_string(directory.path().join("report.compiler.stderr.log"))
            .unwrap()
            .contains("fixture compiler missing")
    );
}

#[test]
fn tex_commands_do_not_inherit_the_gpu_library_search_path() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, rows, models, ledger) = example();
    let _ = write_report_with_runner(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &json!({"outputs":{"report":{"enabled":true,"compiler":"lualatex"}}}),
        &mut |command| {
            assert!(
                command
                    .get_envs()
                    .any(|(name, value)| name == "LD_LIBRARY_PATH" && value.is_none())
            );
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "fixture compiler",
            ))
        },
    );
}

#[test]
fn successful_exit_without_pdf_is_an_error_after_both_direct_passes() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, rows, models, ledger) = example();
    let mut calls = 0;
    let result = write_report_with_runner(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &json!({"outputs":{"report":{"enabled":true,"compiler":"lualatex"}}}),
        &mut |_| {
            calls += 1;
            Ok(std::process::Output {
                status: std::process::ExitStatus::from_raw(0),
                stdout: b"fixture compiler".to_vec(),
                stderr: vec![],
            })
        },
    );
    assert!(result.unwrap_err().to_string().contains("missing PDF"));
    assert_eq!(calls, 3);
    let build: Record =
        serde_json::from_slice(&std::fs::read(directory.path().join("report_build.json")).unwrap())
            .unwrap();
    assert_eq!(build["status"], "error");
    assert!(!build["generator"]["source_sha256"].is_null());
    assert_eq!(build["record_inputs"][0]["role"], "manifest_snapshot");
}

#[test]
fn stale_pdf_is_never_accepted_as_compiler_output() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("report.pdf"), b"stale").unwrap();
    let (manifest, rows, models, ledger) = example();
    let result = write_report(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &json!({"outputs":{"report":{"enabled":true}}}),
    );
    assert!(result.unwrap_err().to_string().contains("already exists"));
    assert_eq!(
        std::fs::read(directory.path().join("report.pdf")).unwrap(),
        b"stale"
    );
}

#[test]
fn unsafe_figure_and_missing_vector_pdf_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    assert!(
        report_figures(
            directory.path(),
            &[json!({"name":"../outside","status":"rendered"})]
        )
        .is_err()
    );
    assert!(
        report_figures(
            directory.path(),
            &[json!({"name":"figure_nort","status":"rendered","formats":["svg"]})]
        )
        .is_err()
    );
    assert!(
        report_figures(
            directory.path(),
            &[json!({"name":"report","status":"rendered"})]
        )
        .is_err()
    );
}

fn fake_pdf() -> Vec<u8> {
    let mut bytes = b"%PDF-1.4\n".to_vec();
    bytes.extend([b'%'; 80]);
    bytes.extend(b"\n%%EOF\n");
    bytes
}

fn version_or(
    args: &[String],
    compile: impl FnOnce() -> std::io::Result<std::process::Output>,
) -> std::io::Result<std::process::Output> {
    if args.contains(&"-v".into()) || args.contains(&"--version".into()) {
        return Ok(std::process::Output {
            status: std::process::ExitStatus::from_raw(0),
            stdout: b"fixture compiler 1.0".to_vec(),
            stderr: vec![],
        });
    }
    compile()
}

#[test]
fn report_options_reject_non_boolean_enabled_and_blank_title() {
    let error = |options| ReportOptions::resolve(&options).err().unwrap().to_string();
    assert!(error(json!({"enabled":"yes"})).contains("enabled must be boolean"));
    assert!(error(json!({"title":"   "})).contains("nonempty string"));
    assert!(error(json!({"title":7})).contains("nonempty string"));
}

#[test]
fn compiled_report_publishes_pdf_and_captions_only_rendered_figures() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, rows, models, ledger) = example();
    let figure = crate::export::path(directory.path(), "figure_nort.pdf");
    std::fs::create_dir_all(figure.parent().unwrap()).unwrap();
    std::fs::write(&figure, fake_pdf()).unwrap();
    let figures = vec![
        json!({"name":"figure_nort","status":"rendered","formats":["svg","pdf","tiff"],"panels":[
            {"label":"DI","reference_line":0.0,"status":"rendered"},
            {"label":"Line one\nline two\tabcdefghijklmnopqrstuvwxyz","reference_line":null,"status":"partial"}]}),
        json!({"name":"figure_oft","status":"blocked"}),
        json!({"name":"figure_ymaze","status":"unavailable","formats":["png"]}),
    ];
    let config =
        json!({"outputs":{"report":{"enabled":true,"compiler":"latexmk","title":"Fixture"}}});
    let result = write_report_with_runner(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &figures,
        &config,
        &mut |command| {
            let args = command
                .get_args()
                .map(|value| value.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            let directory = command.get_current_dir().unwrap().to_path_buf();
            version_or(&args, || {
                std::fs::write(directory.join("_internal/report/report.pdf"), fake_pdf())?;
                Ok(std::process::Output {
                    status: std::process::ExitStatus::from_raw(0),
                    stdout: b"compiled".to_vec(),
                    stderr: vec![],
                })
            })
        },
    )
    .unwrap();
    assert_eq!(result["pdf_status"], "complete");
    assert_eq!(result["versions"]["latexmk"], "fixture compiler 1.0");
    assert_eq!(result["versions"]["lualatex"], "fixture compiler 1.0");
    assert_eq!(result["pdf"]["sha256"].as_str().unwrap().len(), 64);
    assert!(directory.path().join("report.pdf").is_file());
    assert!(
        !directory
            .path()
            .join("_internal/report/report.pdf")
            .exists()
    );
    let build: Record =
        serde_json::from_slice(&std::fs::read(directory.path().join("report_build.json")).unwrap())
            .unwrap();
    assert_eq!(build["status"], "complete");
    assert_eq!(build["figure_inputs"].as_array().unwrap().len(), 1);
    let tex = std::fs::read_to_string(crate::export::path(directory.path(), "report.tex")).unwrap();
    assert!(tex.contains("\\section{Selected figures}"));
    assert!(tex.contains("{figures/nort/figure_nort.pdf}"));
    let plain = tex.replace("\\allowbreak{}", "");
    assert!(plain.contains("DI; reference 0; status rendered\\par"));
    assert!(plain.contains("no reference line; status partial"));
    assert!(tex.contains("Line one line two abcdefghijklmnop\\allowbreak{}qrstuvwxyz"));
    assert!(!tex.contains("figure\\_oft"));
    let markdown = std::fs::read_to_string(directory.path().join("report.md")).unwrap();
    assert!(markdown.contains("- figure_nort: rendered"));
    assert!(markdown.contains("  - [pdf](figures/nort/figure_nort.pdf)"));
    assert!(markdown.contains("  - [svg](figures/nort/figure_nort.svg)"));
    assert!(!markdown.contains("tiff"));
    assert!(markdown.contains("- figure_oft: blocked"));
}

#[test]
fn existing_report_artifacts_are_never_replaced() {
    let (manifest, rows, models, ledger) = example();
    let enabled = json!({"outputs":{"report":{"enabled":true,"compiler":"lualatex"}}});
    let failing = &mut |_: Command| -> std::io::Result<std::process::Output> {
        Err(std::io::Error::other("fixture compiler missing"))
    };
    let markdown = tempfile::tempdir().unwrap();
    std::fs::write(markdown.path().join("report.md"), "prior").unwrap();
    let error = write_report(
        markdown.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &json!({}),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("could not create fresh report artifact report.md"));
    assert_eq!(
        std::fs::read(markdown.path().join("report.md")).unwrap(),
        b"prior"
    );
    let tex = tempfile::tempdir().unwrap();
    let source = crate::export::path(tex.path(), "report.tex");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::write(&source, "prior").unwrap();
    let error = write_report_with_runner(
        tex.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &enabled,
        failing,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("report.tex"));
    let build = tempfile::tempdir().unwrap();
    std::fs::write(build.path().join("report_build.json"), "prior").unwrap();
    let error = write_report_with_runner(
        build.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &enabled,
        failing,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("report_build.json"));
    assert_eq!(
        std::fs::read(build.path().join("report_build.json")).unwrap(),
        b"prior"
    );
}

#[test]
fn pdf_validation_requires_a_regular_signed_nontrivial_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = |name: &str| directory.path().join(name);
    std::fs::write(path("valid.pdf"), fake_pdf()).unwrap();
    validate_pdf(&path("valid.pdf")).unwrap();
    std::os::unix::fs::symlink(path("valid.pdf"), path("link.pdf")).unwrap();
    assert!(
        validate_pdf(&path("link.pdf"))
            .unwrap_err()
            .to_string()
            .contains("non-symlink")
    );
    std::fs::write(path("short.pdf"), b"%PD").unwrap();
    assert!(validate_pdf(&path("short.pdf")).is_err());
    std::fs::write(path("tiny.pdf"), b"%PDF-1.4").unwrap();
    assert!(
        validate_pdf(&path("tiny.pdf"))
            .unwrap_err()
            .to_string()
            .contains("did not produce")
    );
    let mut svg = b"<svg>".to_vec();
    svg.extend([b' '; 80]);
    std::fs::write(path("figure.pdf"), svg).unwrap();
    assert!(validate_pdf(&path("figure.pdf")).is_err());
}

#[test]
fn historical_config_fingerprint_takes_precedence_over_plan_inputs() {
    let manifest = json!({"config_fingerprint":{"sha256":"historical"},
        "inputs":[{"role":"post_analysis_plan","sha256":"current"}]});
    assert_eq!(config_input_hash(&manifest), "historical");
    assert_eq!(config_input_hash(&json!({})), &Value::Null);
}

#[test]
fn supplementary_sections_render_correlations_subgroups_summaries_and_pairing() {
    let directory = tempfile::tempdir().unwrap();
    let write = |name: &str, value: Value| {
        std::fs::write(
            directory.path().join(name),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap()
    };
    write(
        "correlations.json",
        json!([{"x":"spontaneous_alternation_pct","y":"total_distance_m",
        "method":"Spearman","n_animals":12,"rho":0.25,"p_raw":0.4,"p_adjusted":0.8,"status":"computed","reason":null}]),
    );
    write(
        "subgroup_analyses.json",
        json!([
        {"field":"sex","level":"female","status":"partial","assays":["nort"],"reason":"fixture subgroup",
            "models":[{"model_id":"nort.DI","status":"fitted","n_animals":6,"contrast":null}],
            "within_group_tests":[{"group":"G5","endpoint":"DI","null":0,"n_animals":3,"status":"computed"}]},
        {"field":"sex","level":"male","status":"blocked","assays":["nort"],"reason":"no animals","within_group_tests":[]}]),
    );
    write(
        "summaries.json",
        json!([{"figure":"figure_nort","panel":"DI","category":"G5","n_animals":3,
        "mean":0.125,"uncertainty":"SEM","error_half_width":0.01,"status":"available"}]),
    );
    write(
        "pairing_ledger.json",
        json!([
        {"animal_id":"a","endpoint":"DeltaSA","status":"unpaired","reason":"unmatched_prepost"},
        {"animal_id":"b","endpoint":"DeltaSA","status":"unpaired","reason":"unmatched_prepost"},
        {"endpoint":"DeltaSA","status":"unpaired","reason":"anonymous"}]),
    );
    write(
        "genotype_policies.json",
        json!({"comparison":{"interpretation":"Policies compared",
        "rows":[],"endpoint_interpretation":[
            {"model_id":"nort.DI","policy":"B","status":"comparison_unavailable"},
            {"model_id":"oft.total_distance_m","policy":"C","status":"compared","direction_changed":false}]},
        "policies":[{"policy":"B","label":"B: recorded","models":[],"within_group_tests":[],
            "eligibility_and_coding_audit":[],"subgroup_status":"not_requested"}]}),
    );
    let manifest =
        json!({"status":"partial","run_mode":"synthetic","hcar1_policies":{"primary":"A"}});
    let (_, rows, _, ledger) = example();
    let config = json!({"decision_resolutions":{"D01":{"origin":"development_fixture",
        "approval_or_plan_reference":"fixture","rationale":"fixture only"}}});
    let sections =
        report_sections(directory.path(), &manifest, &rows, &[], &ledger, &config).unwrap();
    let markdown = render_markdown("Fixture", &sections, &[]);
    assert!(markdown.contains("spontaneous_alternation_pct vs total_distance_m (Spearman)"));
    assert!(markdown.contains("computed: Unavailable"));
    assert!(markdown.contains("Registered subgroup: sex = female"));
    assert!(
        markdown.contains("Subgroup status: partial. Registered assays: nort. fixture subgroup")
    );
    assert!(markdown.contains("Registered subgroup: sex = female: within-group checks"));
    assert!(!markdown.contains("sex = male: within-group checks"));
    assert!(markdown.contains("figure_nort / DI | G5 | 3 | 0.125 | SEM | 0.01 / available"));
    assert!(markdown.contains("DeltaSA / unpaired / unmatched_prepost | 2"));
    assert!(!markdown.contains("anonymous"));
    assert!(markdown.contains("No model attempts supplied"));
    assert!(markdown.contains("D01 | Record present; see dependency-scoped statuses"));
    assert!(markdown.contains("D02 | Unresolved"));
    assert!(markdown.contains("nort.DI / B | comparison_unavailable"));
    assert!(markdown.contains("oft.total_distance_m / C | false"));
    let missing = tempfile::tempdir().unwrap();
    assert!(
        report_sections(missing.path(), &manifest, &rows, &[], &ledger, &config).is_err(),
        "a declared policy inventory requires its payload"
    );
}

#[test]
fn final_build_record_collision_fails_after_retaining_the_compiled_pdf() {
    let directory = tempfile::tempdir().unwrap();
    let (manifest, rows, models, ledger) = example();
    let error = write_report_with_runner(
        directory.path(),
        &manifest,
        &rows,
        &models,
        &ledger,
        &[],
        &json!({"outputs":{"report":{"enabled":true,"compiler":"latexmk"}}}),
        &mut |command| {
            let args = command
                .get_args()
                .map(|value| value.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            let directory = command.get_current_dir().unwrap().to_path_buf();
            version_or(&args, || {
                std::fs::write(directory.join("_internal/report/report.pdf"), fake_pdf())?;
                std::fs::write(directory.join("report_build.json"), "competing record")?;
                Ok(std::process::Output {
                    status: std::process::ExitStatus::from_raw(0),
                    stdout: vec![],
                    stderr: vec![],
                })
            })
        },
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("report_build.json"));
    assert_eq!(
        std::fs::read(directory.path().join("report_build.json")).unwrap(),
        b"competing record"
    );
    validate_pdf(&directory.path().join("report.pdf")).unwrap();
}

#[test]
fn report_pages_follow_the_figure_colour_mode() {
    let lookup = |value: Option<&'static str>| move |_: &str| value.map(str::to_owned);
    assert_eq!(report_mode(lookup(None)).unwrap(), themekit::Mode::Dark);
    assert_eq!(
        report_mode(lookup(Some("light"))).unwrap(),
        themekit::Mode::Light
    );
    assert!(report_mode(lookup(Some("sepia"))).is_err());
    let dark = render_tex("Fixture", &[], &[], &themekit::DARK);
    assert!(dark.contains("\\definecolor{pagebg}{HTML}{0F1318}"));
    assert!(dark.contains("\\pagecolor{pagebg}\\color{pageink}"));
    let light = render_tex("Fixture", &[], &[], &themekit::LIGHT);
    assert!(light.contains("{HTML}{FFFFFF}"));
}
