use bekiper_post_analysis::{RunOptions, fixtures, inputs, pipeline, verification};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn sources() -> (std::path::PathBuf, std::path::PathBuf) {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    (
        repo.join("examples/neo/imen/post_analysis_plan.toml"),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/statistics_oracle.json"),
    )
}

#[test]
fn native_fixture_runs_all_nine_policy_contexts_and_verifies_complete_attempts() {
    let parent = tempfile::tempdir().unwrap();
    let fixture = parent.path().join("fixture");
    let output = parent.path().join("export");
    let (plan, oracle) = sources();
    let plan_before = fs::read(&plan).unwrap();
    let receipt = fixtures::prepare_hcar1(&plan, &oracle, &fixture).unwrap();
    assert_eq!(receipt["animals"], 40);
    assert_eq!(receipt["metadata_sessions"], 320);
    assert_eq!(receipt["shared_measurements"], 480);
    assert_eq!(receipt["scope"], "synthetic_only");
    assert_eq!(fs::read(&plan).unwrap(), plan_before);
    assert!(fixtures::prepare_hcar1(&plan, &oracle, &fixture).is_err());
    let mut config = inputs::load_json(&fixture.join("config.json")).unwrap();
    config["outputs"]["report"]["enabled"] = false.into();
    config["outputs"]["requested_outputs"] = json!(["metrics", "models", "subgroup_inference"]);
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
    assert_eq!(verified["scientific_readiness"], "fixture_only");
    for (context, threshold) in verified["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .zip([25, 45, 65])
    {
        assert_eq!(context["threshold_pct"], threshold);
        for (policy, n) in context["policies"]
            .as_array()
            .unwrap()
            .iter()
            .zip([37, 39, 40])
        {
            assert_eq!(policy["candidate_animals"], n);
            assert_eq!(policy["model_attempts"], 8);
        }
    }
    // Re-sign inventories to isolate the semantic verifier from the byte verifier.
    let child = output.join("primary25");
    let manifest = inputs::load_json(&child.join("run_manifest.json")).unwrap();
    let path = child.join(
        manifest["package_layout"]["paths"]["genotype_policies.json"]
            .as_str()
            .unwrap(),
    );
    let mut policies = inputs::load_json(&path).unwrap();
    for (policy, n) in policies["policies"]
        .as_array()
        .unwrap()
        .iter()
        .zip([37, 39, 40])
    {
        for model in policy["models"].as_array().unwrap() {
            assert_eq!(model["n_animals"], n, "{}", model["model_id"]);
            assert_eq!(
                model["contrast"]["status"], "computed",
                "{}",
                model["model_id"]
            );
        }
    }
    policies["policies"][1]["models"]
        .as_array_mut()
        .unwrap()
        .pop();
    pipeline::write_json(&path, &policies).unwrap();
    pipeline::write_artifact_manifest(&child).unwrap();
    pipeline::write_artifact_manifest(&output).unwrap();
    assert!(
        verification::verify_run(&output)
            .unwrap_err()
            .to_string()
            .contains("model attempts")
    );
}

#[test]
fn registered_plan_is_byte_bound_and_stale_registration_is_rejected() {
    let (plan, _) = sources();
    let report = verification::verify_plan(&plan).unwrap();
    assert_eq!(report["checked_registrations"], 32);
    assert_eq!(report["analysis_admitted"], false);
    let mut plan: Value = serde_json::to_value(
        toml::from_str::<toml::Value>(&fs::read_to_string(plan).unwrap()).unwrap(),
    )
    .unwrap();
    plan["statistics"]["confidence_level"] = json!(0.90);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("changed-plan.json");
    pipeline::write_json(&path, &plan).unwrap();
    assert!(verification::verify_plan(&path).is_err());
}
