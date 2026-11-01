//! End-to-end rendering and orchestration failure paths: the Python renderer,
//! the LaTeX report compiler and the read-only verifiers run on synthetic
//! publications; every failure must retain evidence and publish nothing.
use bekiper_post_analysis::{
    Record, RunOptions, fixtures, inputs,
    pipeline::{self, publish_descriptive_with_cancel, run_records_with_cancel},
    verification,
};
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicBool, Ordering},
};

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_bekiper-post-analysis")
}

/// The uniquely named staging directory of a live or retained publication.
fn staging(output: &Path) -> Option<PathBuf> {
    let prefix = format!(".{}.staging-", output.file_name()?.to_str()?);
    fs::read_dir(output.parent()?)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix))
        })
}

fn staged(output: &Path, file: &str) -> bool {
    staging(output).is_some_and(|directory| directory.join(file).exists())
}

fn retained_failure(output: &Path) -> Record {
    inputs::load_json(&staging(output).unwrap().join("failure_manifest.json")).unwrap()
}

fn small_config(requests: Value) -> Record {
    json!({"run_mode":"synthetic","synthetic_fixture":true,
        "inputs":{"units_and_calibration":{
            "oft":{"time":"s","distance":"m","speed":"m/s"},
            "nort":{"time":"s","distance":"m","speed":"m/s"}}},
        "assays":{"oft":{},"ymaze":{},"nort":{"zero_training_total_policy":null}},
        "statistics":{"summary_uncertainty":"SEM"},
        "outputs":{"requested_outputs":requests}})
}

fn small_rows() -> Vec<Record> {
    vec![
        json!({"animal_id":"mouse 2","assay":"oft","session_id":"session 2",
            "timepoint":"post","phase":"session","excluded":false,"exclusion_reason":null,
            "measurement_available":true,"total_distance_m":3.0,"center_time_s":20.0,
            "mean_speed_m_per_s":0.01,"center_entries":2,"latency_to_first_center_entry_s":4.0,
            "qc_flags":{"tracking_incomplete":false},"synthetic_observation":true}),
        json!({"animal_id":"mouse 1","assay":"nort","session_id":"test 1",
            "timepoint":"post","phase":"test","excluded":false,"exclusion_reason":null,
            "measurement_available":true,"TN":12,"TF":8,"analysis_duration_s":360,
            "synthetic_observation":true}),
    ]
}

fn run_small(
    config: Record,
    rows: Vec<Record>,
    config_path: &Path,
    output: &Path,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> anyhow::Result<Record> {
    run_records_with_cancel(
        config,
        rows,
        vec![],
        config_path,
        &RunOptions {
            output: Some(output.into()),
            ..Default::default()
        },
        cancelled,
    )
}

fn register(config: &mut Record, decision: &str, path: &str, value: Value) {
    let mut target = &mut *config;
    for part in path.split('.') {
        target = &mut target[part];
    }
    *target = value.clone();
    if !config["decision_resolutions"][decision].is_object() {
        config["decision_resolutions"][decision] = json!({"value":{},"scope":["inclusion","outputs","inputs","statistics"],
            "origin":"development_fixture","rationale":"coverage fixture only","approval_or_plan_reference":"tests"});
    }
    config["decision_resolutions"][decision]["value"][path] = value;
}

/// The registered A/B/C fixture with one primary threshold context.
fn policy_fixture(parent: &Path, requests: Value, selection: Value) -> PathBuf {
    let fixture = parent.join("fixture");
    fixtures::prepare_hcar1(
        &repository().join("examples/neo/imen/post_analysis_plan.toml"),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/statistics_oracle.json"),
        &fixture,
    )
    .unwrap();
    let mut config = inputs::load_json(&fixture.join("config.json")).unwrap();
    config["execution"]["ymaze_primary_thresholds"] = json!([45]);
    config["outputs"]["requested_outputs"] = requests;
    config["outputs"]["figure_selection"] = selection;
    config["outputs"]["report"]["enabled"] = false.into();
    pipeline::write_json(&fixture.join("config.json"), &config).unwrap();
    fixture.join("config.json")
}

fn run_path(config: &Path, output: &Path) {
    pipeline::run(
        config,
        &RunOptions {
            output: Some(output.into()),
            ..Default::default()
        },
    )
    .unwrap();
}

type Edit = dyn Fn(&mut Record);

fn resign(output: &Path, child: &Path) {
    pipeline::write_artifact_manifest(child).unwrap();
    pipeline::write_artifact_manifest(output).unwrap();
}

/// Edit one published JSON payload, re-sign both inventories so only the
/// semantic verifier can object, then restore the published bytes.
fn verify_edited(output: &Path, file: &Path, edit: impl FnOnce(&mut Record)) -> anyhow::Result<()> {
    let child = output.join("primary45");
    let original = fs::read(file).unwrap();
    let mut value = inputs::load_json(file).unwrap();
    edit(&mut value);
    pipeline::write_json(file, &value).unwrap();
    resign(output, &child);
    let result = verification::verify_run(output).map(|_| ());
    fs::write(file, original).unwrap();
    resign(output, &child);
    result
}

fn executable(path: &Path, script: &str) {
    fs::write(path, script).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn png_at_120_dpi(path: &Path) {
    let mut encoder = png::Encoder::new(fs::File::create(path).unwrap(), 1, 1);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_pixel_dims(Some(png::PixelDimensions {
        xppu: 4724,
        yppu: 4724,
        unit: png::Unit::Meter,
    }));
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&[0])
        .unwrap();
}

/// The stand-ins run with a PATH holding only themselves, so they name `cp`
/// by its absolute location.
fn cp() -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|directory| directory.join("cp"))
        .find(|path| path.is_file())
        .unwrap()
}

/// Poppler is not part of the development shell. These stand-ins copy fixed
/// fixtures so the review orchestration, page selection and bounds checks run.
struct FakePoppler {
    bin: PathBuf,
    bbox: PathBuf,
}

impl FakePoppler {
    fn new(root: &Path) -> Self {
        let bin = root.join("fake-bin");
        fs::create_dir_all(&bin).unwrap();
        let bbox = root.join("fixture.bbox.html");
        let image = root.join("fixture.png");
        png_at_120_dpi(&image);
        executable(
            &bin.join("pdftotext"),
            &format!(
                "#!/bin/sh\nfor last; do :; done\n'{}' '{}' \"$last\"\n",
                cp().display(),
                bbox.display()
            ),
        );
        executable(
            &bin.join("pdftoppm"),
            &format!(
                "#!/bin/sh\nfor last; do :; done\n'{}' '{}' \"$last.png\"\n",
                cp().display(),
                image.display()
            ),
        );
        Self { bin, bbox }
    }

    fn pages(&self, pages: &[&str]) {
        let body: String = pages
            .iter()
            .map(|words| {
                let words: String = words
                    .split(' ')
                    .enumerate()
                    .map(|(i, word)| {
                        let x = 10 + 40 * i;
                        let y = if word == "OVERFLOW" { -50 } else { 10 };
                        format!(
                            r#"<word xMin="{x}" yMin="{y}" xMax="{}" yMax="20">{word}</word>"#,
                            x + 30
                        )
                    })
                    .collect();
                format!(r#"<page width="600" height="800">{words}</page>"#)
            })
            .collect();
        fs::write(
            &self.bbox,
            format!(
                r#"<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"><html><body>{body}</body></html>"#
            ),
        )
        .unwrap();
    }

    fn review(&self, input: &Path, output: &Path, cwd: &Path) -> Output {
        Command::new(binary())
            .current_dir(cwd)
            .env("PATH", &self.bin)
            .arg("review-pdfs")
            .arg("--input")
            .arg(input)
            .arg("--output")
            .arg(output)
            .output()
            .unwrap()
    }
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn rendered_policy_run_verifies_figures_report_pdf_source_identity_and_pdf_review() {
    let parent = tempfile::tempdir().unwrap();
    let config = policy_fixture(
        parent.path(),
        json!(["metrics", "figures", "report_pdf"]),
        json!(["figure_oft"]),
    );
    let output = parent.path().join("rendered");
    run_path(&config, &output);
    let verified = verification::verify_run(&output).unwrap();
    let context = &verified["contexts"][0];
    assert_eq!(context["png_count"], 3, "one PNG per HCAR1 policy figure");
    assert_eq!(context["report_pdf"], true);
    for policy in context["policies"].as_array().unwrap() {
        assert_eq!(policy["model_attempts"], 0, "models were not requested");
    }
    // The embedded engine identity matches this checkout; another tree does not.
    verification::verify_run_at_source(&output, Some(&repository())).unwrap();
    let foreign = parent.path().join("foreign");
    fs::create_dir_all(&foreign).unwrap();
    fs::write(foreign.join("Cargo.lock"), "# foreign\n").unwrap();
    assert!(verification::verify_run_at_source(&output, Some(&foreign)).is_err());

    // Figure inventory tampering is detected even after inventories are re-signed.
    let manifest = output.join("primary45/run_manifest.json");
    let figures = inputs::load_json(&manifest).unwrap()["figures"].clone();
    let sensitivity = figures
        .as_array()
        .unwrap()
        .iter()
        .position(|figure| figure["hcar1_policy"] == "B")
        .unwrap();
    let png = figures[0]["files"]
        .as_array()
        .unwrap()
        .iter()
        .position(|file| file.as_str().unwrap().ends_with(".png"))
        .unwrap();
    let cases: Vec<(&str, Box<Edit>)> = vec![
        (
            "invalid or repeated figure slug",
            Box::new(|m| m["figures"][0]["name"] = "Bad Name".into()),
        ),
        (
            "figure policy label",
            Box::new(|m| m["figures"][0]["hcar1_policy"] = Value::Null),
        ),
        (
            "unknown figure policy",
            Box::new(|m| m["figures"][0]["hcar1_policy"] = "D".into()),
        ),
        (
            "unlabelled sensitivity figure",
            Box::new(move |m| m["figures"][sensitivity]["name"] = "figure_relabelled".into()),
        ),
        (
            "outside artifact inventory",
            Box::new(|m| {
                m["figures"][0]["files"]
                    .as_array_mut()
                    .unwrap()
                    .push("figures/absent.svg".into())
            }),
        ),
        (
            "format inventory is incomplete",
            Box::new(|m| m["figures"][0]["formats"] = json!([])),
        ),
        (
            "missing, repeated or misbound",
            Box::new(|m| {
                let formats = m["figures"][0]["formats"].as_array_mut().unwrap();
                formats[0] = "gif".into();
            }),
        ),
        (
            "PNG DPI differs",
            Box::new(move |m| {
                m["figures"][0]["dpi"] = 7.into();
                assert!(m["figures"][0]["files"][png].is_string());
            }),
        ),
    ];
    for (expected, edit) in cases {
        let error = verify_edited(&output, &manifest, edit).unwrap_err();
        assert!(
            format!("{error:#}").contains(expected),
            "{expected}: {error:#}"
        );
    }
    verification::verify_run(&output).unwrap();

    // Retained PDF review: page 1, the policy-handling page and the B/C pages.
    let poppler = FakePoppler::new(parent.path());
    poppler.pages(&[
        "Title HCAR1 handling: primary A",
        "HCAR1 B: recorded",
        "HCAR1 C: assigned",
    ]);
    let review = poppler.review(&output, &parent.path().join("review"), parent.path());
    assert!(review.status.success(), "{}", stderr(&review));
    let receipt: Value = serde_json::from_slice(&review.stdout).unwrap();
    assert_eq!(receipt["status"], "bounds_verified_visual_review_pending");
    let report = &receipt["reports"][0];
    assert_eq!(report["source"], "primary45/report.pdf");
    assert_eq!(report["page_count"], 3);
    assert_eq!(
        report["representative_pages"],
        json!([
            "primary45-page-001.png",
            "primary45-page-002.png",
            "primary45-page-003.png"
        ])
    );
    assert!(
        parent
            .path()
            .join("review/primary45-page-002.png")
            .is_file()
    );

    // Words beyond the page are retained in a failed receipt.
    poppler.pages(&["Title OVERFLOW"]);
    let review = poppler.review(&output, Path::new("relative-review"), parent.path());
    assert!(!review.status.success());
    assert!(stderr(&review).contains("PDF text extends beyond the page"));
    let receipt = inputs::load_json(&parent.path().join("relative-review/receipt.json")).unwrap();
    assert_eq!(receipt["status"], "text_bounds_failed");
    assert_eq!(
        receipt["reports"][0]["words_outside_page"][0]["text"],
        "OVERFLOW"
    );

    // Tool, extraction and output-placement failures never publish a receipt.
    poppler.pages(&[]);
    let review = poppler.review(&output, &parent.path().join("no-pages"), parent.path());
    assert!(stderr(&review).contains("PDF extraction has no pages"));
    fs::write(&poppler.bbox, "<html><page").unwrap();
    let review = poppler.review(&output, &parent.path().join("malformed"), parent.path());
    assert!(!review.status.success() && !parent.path().join("malformed").exists());
    let review = poppler.review(&output, &output.join("inside"), parent.path());
    assert!(!review.status.success());
    poppler.pages(&["Title"]);
    executable(&poppler.bin.join("pdftoppm"), "#!/bin/sh\nexit 3\n");
    let review = poppler.review(&output, &parent.path().join("render-failed"), parent.path());
    assert!(stderr(&review).contains("PDF representative page render failed"));
    fs::remove_file(poppler.bin.join("pdftoppm")).unwrap();
    let review = poppler.review(&output, &parent.path().join("no-pdftoppm"), parent.path());
    assert!(stderr(&review).contains("Poppler pdftoppm"));
    executable(&poppler.bin.join("pdftotext"), "#!/bin/sh\nexit 2\n");
    let review = poppler.review(
        &output,
        &parent.path().join("extract-failed"),
        parent.path(),
    );
    assert!(
        stderr(&review).contains("primary45/report.pdf"),
        "{}",
        stderr(&review)
    );
    fs::remove_file(poppler.bin.join("pdftotext")).unwrap();
    let review = poppler.review(&output, &parent.path().join("no-pdftotext"), parent.path());
    assert!(stderr(&review).contains("Poppler pdftotext"));
    for name in ["review", "relative-review"] {
        assert!(parent.path().join(name).is_dir());
    }
    for name in [
        "no-pages",
        "render-failed",
        "no-pdftoppm",
        "extract-failed",
        "no-pdftotext",
    ] {
        assert!(!parent.path().join(name).exists(), "{name}");
    }
}

#[test]
fn policy_model_verification_reconciles_multiplicity_and_rejects_unsupported_methods() {
    let parent = tempfile::tempdir().unwrap();
    let config = policy_fixture(parent.path(), json!(["metrics", "models"]), json!([]));
    let output = parent.path().join("models");
    run_path(&config, &output);
    verification::verify_run(&output).unwrap();
    let child = output.join("primary45");
    let manifest = inputs::load_json(&child.join("run_manifest.json")).unwrap();
    let policies = child.join(
        manifest["package_layout"]["paths"]["genotype_policies.json"]
            .as_str()
            .unwrap(),
    );
    // Without a registered multiplicity plan the verifier skips family
    // reconciliation, and a contrast need not carry a correction family.
    verify_edited(&output, &policies, |record| {
        for policy in record["policies"].as_array_mut().unwrap() {
            policy["effective_config"]["statistics"]
                .as_object_mut()
                .unwrap()
                .remove("multiplicity");
        }
        record["policies"][1]["models"][0]["contrast"]
            .as_object_mut()
            .unwrap()
            .remove("correction_family");
    })
    .unwrap();
    let error = verify_edited(&output, &policies, |record| {
        let config = &mut record["policies"][0]["effective_config"];
        config["statistics"]["multiplicity"]["method"] = "bonferroni".into();
        config["decision_resolutions"]["D09"]["value"]["statistics.multiplicity"]["method"] =
            "bonferroni".into();
    })
    .unwrap_err();
    assert!(format!("{error:#}").contains("holm"), "{error:#}");
    verification::verify_run(&output).unwrap();
}

#[test]
fn single_context_pdf_report_is_verified_and_reviewed_under_the_report_prefix() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("report-run");
    let manifest = run_small(
        small_config(json!(["metrics", "report_pdf"])),
        small_rows(),
        &parent.path().join("absent.json"),
        &output,
        &|| false,
    )
    .unwrap();
    assert_eq!(manifest["report"]["pdf_status"], "complete");
    assert!(output.join("report.pdf").is_file());
    let verified = verification::verify_run(&output).unwrap();
    assert_eq!(verified["contexts"][0]["report_pdf"], true);
    assert!(
        verified["contexts"][0]["policies"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let poppler = FakePoppler::new(parent.path());
    poppler.pages(&["Policy sensitivity: none", "Sensitivity policy appendix"]);
    let review = poppler.review(&output, &parent.path().join("review"), parent.path());
    assert!(review.status.success(), "{}", stderr(&review));
    let receipt: Value = serde_json::from_slice(&review.stdout).unwrap();
    assert_eq!(
        receipt["reports"][0]["representative_pages"],
        json!(["report-page-001.png", "report-page-002.png"])
    );
    // A verified export without any report PDF has nothing to review.
    let plain = parent.path().join("plain");
    run_small(
        small_config(json!(["metrics"])),
        small_rows(),
        &parent.path().join("absent.json"),
        &plain,
        &|| false,
    )
    .unwrap();
    let review = poppler.review(&plain, &parent.path().join("empty-review"), parent.path());
    assert!(stderr(&review).contains("no report PDFs"));
}

#[test]
fn verification_rejects_unsupported_terminal_status_and_unbound_plans() {
    let parent = tempfile::tempdir().unwrap();
    let plain = parent.path().join("plain");
    run_small(
        small_config(json!(["metrics"])),
        small_rows(),
        &parent.path().join("absent.json"),
        &plain,
        &|| false,
    )
    .unwrap();
    // Every decision registration of a plan without HCAR1 policies is checked.
    let mut plan: Record = serde_json::to_value(
        toml::from_str::<toml::Value>(
            &fs::read_to_string(repository().join("examples/neo/imen/post_analysis_plan.toml"))
                .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    plan["statistics"]
        .as_object_mut()
        .unwrap()
        .remove("hcar1_policy_plan");
    for registration in plan["decision_resolutions"]
        .as_object_mut()
        .unwrap()
        .values_mut()
    {
        registration["value"]
            .as_object_mut()
            .unwrap()
            .remove("statistics.hcar1_policy_plan");
    }
    let path = parent.path().join("plan.json");
    pipeline::write_json(&path, &plan).unwrap();
    let report = verification::verify_plan(&path).unwrap();
    assert_eq!(report["status"], "registered_choices_verified");
}

#[test]
fn run_records_defaults_outputs_fingerprints_the_plan_and_ports_evidence_paths() {
    let parent = tempfile::tempdir().unwrap();
    let plan = parent.path().join("plan.json");
    fs::write(&plan, "{}\n").unwrap();
    let mut config = small_config(json!(null));
    config.as_object_mut().unwrap().remove("outputs");
    config["source"] = json!({"reference":"reference/BEHAVIORAL_RECIPE.md"});
    config["inputs"]["upstream_rule_provenance"] = json!({"rule":{"evidence_files":[
        {"path":"/retained/evidence/rule.json"},{"path":"/"},{"sha256":"0"}]},
        "undocumented":{"status":"no_files"}});
    let manifest = run_records_with_cancel(
        config,
        small_rows(),
        vec![],
        &plan,
        &RunOptions {
            output: Some(parent.path().join("run")),
            requested_outputs: Some(vec!["metrics".into(), "subgroup_inference".into()]),
            ..Default::default()
        },
        &|| false,
    )
    .unwrap();
    assert!(
        manifest["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["role"] == "post_analysis_plan")
    );
    let evidence = &manifest["resolved_config"]["inputs"]["upstream_rule_provenance"]["rule"]["evidence_files"];
    assert_eq!(evidence[0]["path"], "rule.json");
    assert_eq!(evidence[1]["path"], "evidence");
    assert!(evidence[2].get("path").is_none());
    assert_eq!(
        manifest["resolved_config"]["source"]["reference"],
        "BEHAVIORAL_RECIPE.md"
    );
    assert_eq!(
        manifest["output_statuses"]["subgroup_inference"]["status"],
        "blocked"
    );
}

#[test]
fn invalid_orchestration_inputs_fail_before_or_with_retained_evidence() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("thresholds");
    let error = run_records_with_cancel(
        small_config(json!(["metrics"])),
        small_rows(),
        vec![],
        &parent.path().join("absent.json"),
        &RunOptions {
            output: Some(output.clone()),
            sensitivity_thresholds: Some(vec![44]),
            ..Default::default()
        },
        &|| false,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("ensitivity"), "{error:#}");
    assert!(!output.exists() && staging(&output).is_none());

    let mut production = small_config(json!(["metrics"]));
    production["run_mode"] = "production".into();
    let output = parent.path().join("production");
    let error = run_small(
        production,
        small_rows(),
        &parent.path().join("absent.json"),
        &output,
        &|| false,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("synthetic observations cannot be a production"));
    assert_eq!(retained_failure(&output)["failed_stage"], "validation");

    let mut report = small_config(json!(["metrics", "report"]));
    report["outputs"]["report"] = json!({"enabled":"yes"});
    let output = parent.path().join("report-config");
    let error = run_small(
        report,
        small_rows(),
        &parent.path().join("absent.json"),
        &output,
        &|| false,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("outputs.report.enabled must be boolean"));
    assert_eq!(retained_failure(&output)["failed_stage"], "report");
    assert!(!output.exists());

    // An enabled Markdown report with compiler "none" requests no PDF, and the
    // report writer still rejects the unsupported compiler.
    let mut report = small_config(json!(["metrics", "report"]));
    report["outputs"]["report"] = json!({"enabled":true,"compiler":"none"});
    let output = parent.path().join("report-compiler");
    let error = run_small(
        report,
        small_rows(),
        &parent.path().join("absent.json"),
        &output,
        &|| false,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("report compiler must be latexmk or lualatex"));
    assert!(!output.exists());
}

#[test]
fn late_cancellation_racing_destinations_and_lost_staging_never_publish() {
    let parent = tempfile::tempdir().unwrap();
    let absent = parent.path().join("absent.json");

    let output = parent.path().join("late-cancel");
    let cancelled = || staged(&output, "run_manifest.json");
    let error = run_small(
        small_config(json!(["metrics"])),
        small_rows(),
        &absent,
        &output,
        &cancelled,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("cancelled before publication"));
    let failure = retained_failure(&output);
    assert_eq!(failure["status"], "cancelled");
    assert!(!staging(&output).unwrap().join("run_manifest.json").exists());
    assert!(!output.exists());

    let output = parent.path().join("raced");
    let racer = || {
        if staged(&output, "run_manifest.json") && !output.exists() {
            fs::create_dir(&output).unwrap();
        }
        false
    };
    let error = run_small(
        small_config(json!(["metrics"])),
        small_rows(),
        &absent,
        &output,
        &racer,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("destination appeared"));
    assert_eq!(retained_failure(&output)["failed_stage"], "publication");
    assert_eq!(fs::read_dir(&output).unwrap().count(), 0);

    let output = parent.path().join("lost");
    let lost = || {
        if let Some(directory) = staging(&output) {
            fs::remove_dir_all(directory).unwrap();
        }
        true
    };
    let error = run_small(
        small_config(json!(["metrics"])),
        small_rows(),
        &absent,
        &output,
        &lost,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("failed to retain failure manifest"));
    assert!(!output.exists());

    // A failed payload write after scoring is retained under its stage.
    let output = parent.path().join("blocked-write");
    let mut rows = small_rows();
    rows[0]["native_comparisons"] = json!([{"metric":"center_time_s","native":20.0}]);
    let blocker = || {
        if let Some(directory) = staging(&output)
            && directory.join("plotting_bundle.json").exists()
        {
            let _ = fs::create_dir(directory.join("native_comparisons.json"));
        }
        false
    };
    let error = run_small(
        small_config(json!(["metrics"])),
        rows,
        &absent,
        &output,
        &blocker,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("failure evidence retained"));
    assert!(!output.exists());
}

#[test]
fn explicit_recipe_references_are_copied_or_rejected() {
    let parent = tempfile::tempdir().unwrap();
    let references = parent.path().join("refs");
    fs::create_dir_all(&references).unwrap();
    fs::write(references.join("RECIPE.md"), "# recipe\n").unwrap();
    let config_path = parent.path().join("plan.json");
    let mut config = small_config(json!(["metrics"]));
    config["source"] = json!({"reference":"refs/RECIPE.md"});

    let output = parent.path().join("no-amendment");
    let error = run_small(config.clone(), small_rows(), &config_path, &output, &|| {
        false
    })
    .unwrap_err();
    assert!(format!("{error:#}").contains("requires YMAZE_AMENDMENT.json beside it"));

    fs::write(references.join("YMAZE_AMENDMENT.json"), "{}\n").unwrap();
    let output = parent.path().join("copied");
    let manifest = run_small(config.clone(), small_rows(), &config_path, &output, &|| {
        false
    })
    .unwrap();
    assert_eq!(manifest["source_recipe"]["bytes"], 9);
    assert_eq!(manifest["ymaze_amendment"]["bytes"], 3);

    config["source"]["reference"] = "refs/MISSING.md".into();
    let output = parent.path().join("missing");
    let error = run_small(config, small_rows(), &config_path, &output, &|| false).unwrap_err();
    assert!(format!("{error:#}").contains("configured source reference is missing"));

    // The embedded reference is written into the staging package; a blocked
    // write is retained as failure evidence.
    let output = parent.path().join("embedded-blocked");
    let blocker = || {
        if let Some(directory) = staging(&output) {
            let _ = fs::create_dir(directory.join("YMAZE_AMENDMENT.json"));
        }
        false
    };
    assert!(
        run_small(
            small_config(json!(["metrics"])),
            small_rows(),
            &config_path,
            &output,
            &blocker
        )
        .is_err()
    );
    assert!(!output.exists());
}

#[test]
fn model_scope_follows_registered_requests_and_rejects_unknown_verification_rules() {
    let mut config = small_config(json!(["correlations"]));
    let results = pipeline::run_models(&[json!({"timepoint":"pre"})], &config).unwrap();
    assert!(
        !results.correlations.is_empty(),
        "correlations scope Y-maze"
    );
    assert!(
        results
            .correlations
            .iter()
            .all(|row| row["status"] == "blocked")
    );
    config["outputs"]["requested_outputs"] = json!(["models"]);
    register(
        &mut config,
        "D12",
        "inclusion.upstream_verification_requirement",
        json!("trust_everything"),
    );
    let error = pipeline::run_models(&[], &config).err().unwrap();
    assert!(
        error
            .to_string()
            .contains("unknown upstream verification requirement")
    );
}

fn descriptive_rows() -> Vec<Record> {
    let mut rows = Vec::new();
    for timepoint in ["pre", "post"] {
        for (animal, recording, value) in [("a", "one", 7.), ("b", "two", 9.)] {
            rows.push(json!({"animal_id":animal,"assay":"ymaze","group":"recorded-2",
                "session_id":recording,"recording_id":recording,"analysis_id":format!("{timepoint}_{recording}"),
                "variant":"proximal25","analysis_role":"ymaze__proximal25","timepoint":timepoint,"phase":"session",
                "threshold_pct":25,"excluded":false,"value":value,"scalar_endpoints":["value"],
                "scoring_provenance":{"minimum_dwell_seconds":1.0},
                "presentation_partition":{"timepoint":timepoint,"phase":"session","variant":"proximal25"},
                "presentation_label":format!("Y-maze {timepoint}")}));
        }
    }
    rows
}

#[test]
fn descriptive_report_renders_figures_and_compiles_a_pdf() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("descriptive");
    let manifest = publish_descriptive_with_cancel(
        descriptive_rows(),
        vec![],
        vec![],
        json!({"outputs":{"report":{"enabled":true},"figure_selection":["figure_ymaze_distributions"]}}),
        &output,
        &|| false,
    )
    .unwrap();
    assert_eq!(
        manifest["requested_outputs"],
        json!(["metrics", "figures", "report", "report_pdf"])
    );
    assert_eq!(
        manifest["output_statuses"]["report_pdf"]["status"],
        "complete"
    );
    assert_eq!(
        manifest["resolved_config"]["outputs"]["figure_formats"],
        json!(["png", "pdf"])
    );
    assert!(output.join("report.pdf").is_file());
    pipeline::verify_artifacts(&output).unwrap();
}

#[test]
fn descriptive_late_cancellation_and_racing_destination_retain_evidence() {
    let parent = tempfile::tempdir().unwrap();
    let config = json!({"outputs":{"figures_enabled":false}});
    let output = parent.path().join("late");
    let cancelled = || staged(&output, "run_manifest.json");
    let error = publish_descriptive_with_cancel(
        descriptive_rows(),
        vec![],
        vec![],
        config.clone(),
        &output,
        &cancelled,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("failure evidence retained"));
    assert_eq!(retained_failure(&output)["status"], "cancelled");

    let output = parent.path().join("raced");
    let raced = AtomicBool::new(false);
    let racer = || {
        if staged(&output, "run_manifest.json") && !raced.swap(true, Ordering::SeqCst) {
            fs::create_dir(&output).unwrap();
        }
        false
    };
    let error = publish_descriptive_with_cancel(
        descriptive_rows(),
        vec![],
        vec![],
        config,
        &output,
        &racer,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("destination appeared"));
    assert_eq!(retained_failure(&output)["failed_stage"], "publication");
}

/// The installed binary resolves `python` from PATH. A stand-in renderer that
/// fails, or exits without its JSON receipt or figures, must not publish.
#[test]
fn renderer_failures_and_missing_figures_retain_failure_evidence() {
    let parent = tempfile::tempdir().unwrap();
    let bin = parent.path().join("fake-bin");
    fs::create_dir_all(&bin).unwrap();
    let mut config = small_config(json!(["metrics", "figures", "report_pdf"]));
    config["outputs"]["figure_selection"] = json!(["figure_oft"]);
    config["inputs"]["metadata_path"] = "metadata.json".into();
    config["inputs"]["oft_path"] = "oft.json".into();
    let metadata = json!([{"animal_id":"a1","assay":"oft","session_id":"s1","timepoint":"post",
        "phase":"session","five_xfad":"AD","hcar1":"WT","treatment":"lactate","group":"G5",
        "sex":"female","age_treatment_window":"registered_window","excluded":false,"exclusion_reason":null}]);
    let oft = json!([{"animal_id":"a1","assay":"oft","session_id":"s1","timepoint":"post",
        "phase":"session","total_distance_m":3.0,"center_time_s":20.0,"mean_speed_m_per_s":0.01,
        "center_entries":2,"latency_to_first_center_entry_s":4.0,"analysis_duration_s":300,
        "synthetic_observation":true}]);
    for (name, value) in [
        ("config.json", config),
        ("metadata.json", metadata),
        ("oft.json", oft),
    ] {
        pipeline::write_json(&parent.path().join(name), &value).unwrap();
    }
    let run = |output: &str| {
        Command::new(binary())
            .current_dir(parent.path())
            .env("PATH", &bin)
            .args([
                "--config",
                "config.json",
                "--output",
                output,
                "--no-sensitivity",
            ])
            .output()
            .unwrap()
    };
    executable(
        &bin.join("python"),
        "#!/bin/sh\necho 'renderer exploded' >&2\nexit 4\n",
    );
    let failed = run("failed");
    assert!(!failed.status.success());
    assert!(
        stderr(&failed).contains("renderer exploded"),
        "{}",
        stderr(&failed)
    );
    let failure = retained_failure(&parent.path().join("failed"));
    assert_eq!(failure["failed_stage"], "renderer");

    executable(&bin.join("python"), "#!/bin/sh\necho 'not a receipt'\n");
    let silent = run("silent");
    assert!(stderr(&silent).contains("renderer did not produce selected figure"));
    let staged = staging(&parent.path().join("silent")).unwrap();
    let bundle = inputs::load_json(&staged.join("plotting_bundle.json")).unwrap();
    assert_eq!(
        bundle["figures"][0]["formats"],
        json!(["svg", "pdf"]),
        "a requested PDF report adds vector figures to the default"
    );
    assert!(!parent.path().join("silent").exists());
}

#[test]
fn fully_measured_oft_session_reports_complete_metrics() {
    let parent = tempfile::tempdir().unwrap();
    let mut row = small_rows().remove(0);
    for (field, value) in [
        ("periphery_time_pct", 80.0),
        ("freezing_s", 1.0),
        ("grooming_s", 2.0),
        ("rearing_s", 3.0),
        ("immobility_s", 4.0),
        ("fecal_boli_count", 0.0),
    ] {
        row[field] = value.into();
    }
    row["analysis_duration_s"] = 300.into();
    let output = parent.path().join("complete");
    let manifest = run_small(
        small_config(json!(["metrics"])),
        vec![row],
        &parent.path().join("absent.json"),
        &output,
        &|| false,
    )
    .unwrap();
    let ledger = inputs::load_json(&bekiper_post_analysis::export::input_path(
        &output,
        "inclusion_ledger.json",
    ))
    .unwrap();
    assert_eq!(
        manifest["output_statuses"]["metrics"]["status"], "complete",
        "{ledger:#}"
    );
}

#[test]
fn rendered_figures_without_policies_verify_and_cancelled_renderers_never_start() {
    let parent = tempfile::tempdir().unwrap();
    let mut config = small_config(json!(["metrics", "figures"]));
    config["outputs"]["figure_selection"] = json!(["figure_oft"]);
    let mut rows = small_rows();
    for row in &mut rows {
        for (field, value) in [
            ("five_xfad", "AD"),
            ("hcar1", "WT"),
            ("treatment", "lactate"),
            ("group", "G5"),
        ] {
            row[field] = value.into();
        }
    }
    let output = parent.path().join("figures");
    let manifest = run_small(
        config.clone(),
        rows.clone(),
        &parent.path().join("absent.json"),
        &output,
        &|| false,
    )
    .unwrap();
    assert_eq!(manifest["figures"][0]["formats"], json!(["svg"]));
    let verified = verification::verify_run(&output).unwrap();
    assert!(
        verified["contexts"][0]["policies"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    let output = parent.path().join("cancelled-renderer");
    let cancelled = || staged(&output, "plotting_bundle.json");
    let error = run_small(
        config,
        rows,
        &parent.path().join("absent.json"),
        &output,
        &cancelled,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("rendering cancelled"));
    let retained = staging(&output).unwrap();
    assert_eq!(retained_failure(&output)["status"], "cancelled");
    assert!(!retained.join("renderer.stdout.log").exists());
}

#[test]
fn descriptive_report_errors_and_lost_staging_are_reported() {
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("titled");
    let error = publish_descriptive_with_cancel(
        descriptive_rows(),
        vec![],
        vec![],
        json!({"outputs":{"figures_enabled":false,"report":{"enabled":true,"title":" "}}}),
        &output,
        &|| false,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("title must be a nonempty string"));
    assert_eq!(
        retained_failure(&output)["failed_stage"],
        "descriptive_report"
    );

    let output = parent.path().join("lost");
    let lost = || {
        if let Some(directory) = staging(&output)
            && directory.join("run_manifest.json").exists()
        {
            fs::remove_dir_all(directory).unwrap();
        }
        staging(&output).is_none()
    };
    assert!(
        publish_descriptive_with_cancel(
            descriptive_rows(),
            vec![],
            vec![],
            json!({"outputs":{"figures_enabled":false}}),
            &output,
            &lost,
        )
        .is_err()
    );
    assert!(!output.exists() && staging(&output).is_none());
}

/// Adapter contracts that the cohort exports exercise from other crates are
/// pinned here as well, so this crate's own suite covers them.
#[test]
fn adapter_mappings_pair_identity_dwell_overrides_and_unsupported_genotypes() {
    let mapped =
        inputs::normalize(&json!({"Animal":"a1"}), &json!({"Animal":"animal_id"})).unwrap();
    assert_eq!(mapped["animal_id"], "a1");
    assert!(
        inputs::normalize(&json!({"Animal":"a1"}), &json!({"Animal":5}))
            .unwrap_err()
            .to_string()
            .contains("targets must be strings")
    );
    let mut row = json!({"animal_id":"a1"});
    inputs::apply_session_value_mappings(&mut row, &Value::Null).unwrap();
    assert_eq!(row, json!({"animal_id":"a1"}));

    let metadata = |pair: Value| {
        json!([{"animal_id":"a1","assay":"oft","session_id":"s1","timepoint":"post",
            "phase":"session","five_xfad":"AD","hcar1":"WT","treatment":"lactate","group":"G5",
            "sex":"female","age_treatment_window":"window","excluded":false,"exclusion_reason":null,
            "pair_id":pair}])
    };
    inputs::validate_metadata(metadata(json!("pair-1")).as_array().unwrap(), &json!({})).unwrap();
    assert!(
        inputs::validate_metadata(metadata(json!(" ")).as_array().unwrap(), &json!({}))
            .unwrap_err()
            .to_string()
            .contains("pair_id must be nonempty")
    );

    let mut dwell = json!({"assays":{"ymaze":{"principal_minimum_dwell_s":1.0}}});
    inputs::resolve_ymaze_dwell(
        &mut dwell,
        bekiper_feature::tolerance::DwellOverrides {
            seconds: Some(0.5),
            comparison: None,
        },
    )
    .unwrap();
    assert_eq!(dwell["assays"]["ymaze"]["principal_minimum_dwell_s"], 0.5);

    let mut config = json!({"run_mode":"synthetic","synthetic_fixture":true});
    register(
        &mut config,
        "D10",
        "statistics.hcar1_policy_plan",
        json!({"primary":"A","policies":["A","B","C"]}),
    );
    register(
        &mut config,
        "D10",
        "inputs.group_mapping",
        json!({"G5":{"five_xfad":"AD","hcar1":"WT","treatment":"lactate"}}),
    );
    let (coded, _) = bekiper_post_analysis::genotype::project(
        &[json!({"animal_id":"a1","hcar1":"unrecorded","group":"G5"})],
        &config,
        "A",
    )
    .unwrap();
    assert_eq!(
        coded[0]["genotype_eligibility"]["reason"],
        "missing_or_unsupported_recorded_genotype"
    );
}

#[test]
fn training_roles_and_per_arm_counts_are_checked_against_their_sources() {
    use bekiper_post_analysis::{events, metrics};
    let training = json!({"animal_id":"a1","assay":"nort","session_id":"s1","timepoint":"post","phase":"training"});
    // Exact training roles validate even before an object configuration exists.
    assert!(
        events::nort_object_roles(
            &training,
            &json!({"training":{"TA":"cube","TB":"cylinder"}})
        )
        .unwrap()
        .is_none()
    );
    assert!(
        events::nort_object_roles(
            &training,
            &json!({"training":{"TN":"cube","TF":"cylinder"}})
        )
        .is_err()
    );
    let mut ymaze = json!({"assay":"ymaze","valid_arm_entry_sequence":["A","B","A"],"valid_arm_A_entries":2,
        "valid_arm_B_entries":1,"valid_arm_C_entries":0});
    metrics::validate_supplied(&ymaze).unwrap();
    ymaze["valid_arm_C_entries"] = 1.into();
    assert!(
        metrics::validate_supplied(&ymaze)
            .unwrap_err()
            .to_string()
            .contains("contradicts its supplied sequence")
    );
}
