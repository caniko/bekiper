//! The registered Neo/Imen plan end to end on the synthetic HCAR1 fixture: one
//! primary context, every requested output, rendered main figures and the PDF.
use bekiper_post_analysis::{Record, RunOptions, fixtures, inputs, pipeline, verification};
use serde_json::json;
use std::{fs, path::Path};

fn load(child: &Path, manifest: &Record, name: &str) -> Record {
    inputs::load_json(&child.join(manifest["package_layout"]["paths"][name].as_str().unwrap()))
        .unwrap()
}

/// Edit one published payload, re-sign both inventories so only the semantic
/// verifier can object, then restore the published bytes.
fn verify_edited(output: &Path, file: &Path, edit: impl FnOnce(&mut Record)) -> String {
    let child = output.join("primary45");
    let original = fs::read(file).unwrap();
    let mut value = inputs::load_json(file).unwrap();
    edit(&mut value);
    pipeline::write_json(file, &value).unwrap();
    pipeline::write_artifact_manifest(&child).unwrap();
    pipeline::write_artifact_manifest(output).unwrap();
    let error = verification::verify_run(output).unwrap_err().to_string();
    fs::write(file, original).unwrap();
    pipeline::write_artifact_manifest(&child).unwrap();
    pipeline::write_artifact_manifest(output).unwrap();
    error
}

#[test]
fn imen_plan_publishes_every_requested_output_and_verifies_its_families() {
    let parent = tempfile::tempdir().unwrap();
    let fixture = parent.path().join("fixture");
    let output = parent.path().join("export");
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    fixtures::prepare_hcar1(
        &crate_dir.join("../../examples/neo/imen/post_analysis_plan.toml"),
        &crate_dir.join("tests/fixtures/statistics_oracle.json"),
        &fixture,
    )
    .unwrap();
    let mut config = inputs::load_json(&fixture.join("config.json")).unwrap();
    // The plan's requested outputs and enabled report stay as registered; one
    // threshold context and the fixture's main-figure selection keep it light.
    assert_eq!(
        config["outputs"]["requested_outputs"],
        json!([
            "metrics",
            "models",
            "correlations",
            "within_group_tests",
            "subgroup_inference",
            "figures",
            "report_pdf"
        ])
    );
    assert_eq!(config["outputs"]["report"]["enabled"], true);
    config["execution"]["ymaze_primary_thresholds"] = json!([45]);
    pipeline::write_json(&fixture.join("config.json"), &config).unwrap();
    pipeline::run(
        &fixture.join("config.json"),
        &RunOptions {
            output: Some(output.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    let verified = verification::verify_run(&output).unwrap();
    assert_eq!(verified["artifact_status"], "verified");
    let child = output.join("primary45");
    let manifest = inputs::load_json(&child.join("run_manifest.json")).unwrap();

    // NORT preference checks: DI against 0 and % novel against 50 in every group.
    let tests = load(&child, &manifest, "within_group_tests.json");
    let tests = tests.as_array().unwrap();
    assert_eq!(tests.len(), 16, "8 registered groups x 2 checks");
    for test in tests {
        let null = match test["endpoint"].as_str().unwrap() {
            "DI" => 0.0,
            "percent_novel" => 50.0,
            other => panic!("unregistered check {other}"),
        };
        assert_eq!(test["null"], null);
        assert_eq!(test["method"], "one_sample_t");
        assert_eq!(test["status"], "computed", "{}", test["group"]);
        assert!(test["estimate"].is_f64() && test["p_raw"].is_f64());
        assert!(test["p_adjusted"].as_f64().unwrap() >= test["p_raw"].as_f64().unwrap());
        assert_eq!(test["family_size"], 16);
        assert_eq!(
            test["correction_family"],
            "recipe_nort_preference_checks:HCAR1_policy_A"
        );
    }

    // Y-maze Spearman QC on post sessions only, and the baseline-adjusted SA model.
    let correlations = load(&child, &manifest, "correlations.json");
    let pairs: Vec<_> = correlations
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            assert_eq!(c["method"], "Spearman");
            assert_eq!(c["x"], "spontaneous_alternation_pct");
            assert_eq!(c["status"], "computed");
            assert!(
                c["subset"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|r| r["timepoint"] == "post")
            );
            c["y"].as_str().unwrap()
        })
        .collect();
    assert_eq!(
        pairs,
        [
            "valid_total_arm_entries",
            "total_distance_m",
            "mean_speed_m_per_s",
            "neutral_triads_pct"
        ]
    );
    let models = load(&child, &manifest, "models.json");
    let ymaze = models
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model_id"] == "ymaze.spontaneous_alternation_pct")
        .unwrap();
    assert_eq!(ymaze["status"], "fitted");
    assert!(ymaze["formula"].as_str().unwrap().contains("pre_outcome +"));
    let summaries = load(&child, &manifest, "summaries.json");
    let delta: Vec<_> = summaries
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["figure"] == "figure_ymaze" && s["panel"] == "DeltaSA")
        .collect();
    assert!(!delta.is_empty());
    assert!(
        delta
            .iter()
            .any(|s| s["status"] == "available" && s["mean"].is_f64())
    );

    // Registered exploratory strata: 2 sex + 4 age/treatment-window levels.
    let subgroups = load(&child, &manifest, "subgroup_analyses.json");
    let subgroups = subgroups.as_array().unwrap();
    assert_eq!(subgroups.len(), 6);
    for item in subgroups {
        for kind in ["models", "correlations", "within_group_tests"] {
            assert!(!item[kind].as_array().unwrap().is_empty(), "{kind}");
        }
        assert_eq!(
            item["within_group_tests"][0]["correction_family"],
            "recipe_exploratory_subgroups:HCAR1_policy_A"
        );
    }

    // B and C are compared against primary A per model.
    let policies_path = child.join(
        manifest["package_layout"]["paths"]["genotype_policies.json"]
            .as_str()
            .unwrap(),
    );
    let policies = inputs::load_json(&policies_path).unwrap();
    let comparison = &policies["comparison"];
    let estimate = |policy: &str, model: &Record| {
        comparison["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["policy"] == policy && r["model_id"] == *model)
            .unwrap()["estimate"]
            .as_f64()
            .unwrap()
    };
    let compared = comparison["endpoint_interpretation"].as_array().unwrap();
    assert_eq!(compared.len(), 16, "8 models x policies B and C");
    for row in compared {
        assert!(["B", "C"].contains(&row["policy"].as_str().unwrap()));
        assert_eq!(row["status"], "compared", "{}", row["model_id"]);
        let (a, b) = (
            estimate("A", &row["model_id"]),
            estimate(row["policy"].as_str().unwrap(), &row["model_id"]),
        );
        assert_eq!(
            row["direction_changed"],
            (a > 0.0, a < 0.0) != (b > 0.0, b < 0.0)
        );
        assert_eq!(row["estimate_difference_from_A"].as_f64().unwrap(), b - a);
    }

    // Rendered PNG figures and the compiled report PDF.
    let context = &verified["contexts"][0];
    assert_eq!(context["report_pdf"], true);
    assert_eq!(manifest["report"]["pdf_status"], "complete");
    assert!(child.join("report.pdf").metadata().unwrap().len() > 0);
    let pngs: Vec<_> = manifest["figures"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|f| f["files"].as_array().unwrap())
        .filter_map(|f| f.as_str().filter(|f| f.ends_with(".png")))
        .collect();
    assert_eq!(pngs.len(), 9, "main OFT/Y-maze/NORT figures for A, B and C");
    assert_eq!(context["png_count"], 9);
    assert!(pngs.iter().all(|png| child.join(png).is_file()));

    // Holm families are recomputed from the published raw attempts.
    let nort = verify_edited(&output, &policies_path, |value| {
        value["policies"][1]["within_group_tests"][0]["p_adjusted"] = json!(0.5);
    });
    assert!(
        nort.contains("nort_within_group correction differs"),
        "{nort}"
    );
    let subgroup = verify_edited(&output, &policies_path, |value| {
        value["policies"][2]["subgroup_analyses"][0]["within_group_tests"][0]["p_adjusted"] =
            json!(0.5);
    });
    assert!(
        subgroup.contains("subgroup correction differs"),
        "{subgroup}"
    );
    let main = child.join(
        manifest["package_layout"]["paths"]["subgroup_analyses.json"]
            .as_str()
            .unwrap(),
    );
    let published = verify_edited(&output, &main, |value| {
        value[0]["models"][0]["contrast"]["p_adjusted"] = json!(0.5);
    });
    assert!(
        published.contains("differs from primary policy A"),
        "{published}"
    );
}
