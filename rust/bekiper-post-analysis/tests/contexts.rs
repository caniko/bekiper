//! Generalised primary contexts: one variant per assay, no-plan 0%, parallel
//! workers, and equivalence with the legacy threshold expansion.
use bekiper_post_analysis::pipeline::{
    artifact_inventory, publish_descriptive_with_cancel, verify_artifacts,
};
use serde_json::{Value, json};
use std::path::Path;

const CROSSLINE: &str = "imen_arm_start_crossline_v1";

fn row(assay: &str, variant: Option<&str>, recording: &str, extra: Value) -> Value {
    let mut row = json!({"animal_id":"a","assay":assay,"group":"g","session_id":recording,
        "recording_id":recording,"analysis_id":format!("{recording}__{}", variant.unwrap_or("bare")),
        "variant":variant,"analysis_role":variant.map(|v| format!("{assay}__{v}")),
        "timepoint":"pre","phase":"session","excluded":false,"value":7.,"scalar_endpoints":["value"],
        "presentation_partition":{"timepoint":"pre","phase":"session","variant":variant},
        "presentation_label":format!("{assay} {}", variant.unwrap_or("recorded"))});
    row.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    row
}

fn ymaze(geometry: &str) -> Vec<Value> {
    let mut rows = Vec::new();
    for (variant, pct) in [
        ("arm0_dwell1s", 0),
        ("arm25", 25),
        ("arm45", 45),
        ("arm65", 65),
    ] {
        for recording in ["one", "two"] {
            rows.push(row(
                "ymaze",
                Some(variant),
                recording,
                json!({"threshold_pct":pct,"scoring_provenance":{"geometry_version":geometry}}),
            ));
        }
    }
    rows
}

fn nort() -> Vec<Value> {
    ["whisker_age", "whisker20mm", "nose_only"]
        .into_iter()
        .map(|variant| row("nort", Some(variant), "n1", json!({"phase":"test"})))
        .collect()
}

fn config(execution: Value) -> Value {
    json!({"outputs":{"figures_enabled":false},"execution":execution})
}

fn metrics(context: &Path) -> Vec<Value> {
    serde_json::from_slice(
        &std::fs::read(bekiper_post_analysis::export::input_path(
            context,
            "metrics.json",
        ))
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn zero_pct_is_accepted_without_a_plan_from_recorded_geometry() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("zero");
    let manifest = publish_descriptive_with_cancel(
        ymaze(CROSSLINE),
        vec![],
        vec![],
        config(json!({"ymaze_primary_thresholds":[0,25,45,65],"primary_workers":2})),
        &output,
        &|| false,
    )
    .unwrap();
    assert_eq!(manifest["primary_contexts"].as_array().unwrap().len(), 4);
    assert_eq!(manifest["primary_workers"], 2);
    for t in [0, 25, 45, 65] {
        let rows = metrics(&output.join(format!("primary{t}")));
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r["threshold_pct"] == t));
    }
    verify_artifacts(&output).unwrap();
    // The radial geometry still refuses 0%, and conflicting records error.
    let radial = publish_descriptive_with_cancel(
        ymaze("imen_radial_vertex_span_v1"),
        vec![],
        vec![],
        config(json!({"ymaze_primary_thresholds":[0]})),
        &dir.path().join("radial"),
        &|| false,
    );
    assert!(radial.is_err());
    let mut mixed = ymaze(CROSSLINE);
    mixed[0]["scoring_provenance"]["geometry_version"] = json!("imen_radial_vertex_span_v1");
    let error = publish_descriptive_with_cancel(
        mixed,
        vec![],
        vec![],
        config(json!({"ymaze_primary_thresholds":[25]})),
        &dir.path().join("mixed"),
        &|| false,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("conflicting Y-maze threshold geometries"));
}

#[test]
fn explicit_contexts_reproduce_the_legacy_threshold_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let legacy = dir.path().join("legacy");
    publish_descriptive_with_cancel(
        ymaze(CROSSLINE),
        vec![],
        vec![],
        config(json!({"ymaze_primary_thresholds":[0,25,45,65],"primary_workers":2})),
        &legacy,
        &|| false,
    )
    .unwrap();
    let explicit = dir.path().join("explicit");
    let contexts: Vec<_> = ["arm0_dwell1s", "arm25", "arm45", "arm65"]
        .iter()
        .map(|variant| json!({"assay":"ymaze","variant":variant}))
        .collect();
    let manifest = publish_descriptive_with_cancel(
        ymaze(CROSSLINE),
        vec![],
        vec![],
        config(json!({"contexts":contexts,"primary_workers":4})),
        &explicit,
        &|| false,
    )
    .unwrap();
    assert_eq!(manifest["publication_kind"], "independent_primary_contexts");
    verify_artifacts(&explicit).unwrap();
    let mut compared = 0;
    for (t, variant) in [
        (0, "arm0_dwell1s"),
        (25, "arm25"),
        (45, "arm45"),
        (65, "arm65"),
    ] {
        let old = artifact_inventory(&legacy.join(format!("primary{t}"))).unwrap();
        let new = artifact_inventory(&explicit.join(variant)).unwrap();
        for file in &old {
            let path = file["path"].as_str().unwrap();
            if path.ends_with("manifest.json") || path.ends_with("parameters.toml") {
                continue; // record the context identity and configuration
            }
            if let Some(same) = new.iter().find(|n| n["path"] == file["path"]) {
                assert_eq!(same["sha256"], file["sha256"], "{variant}: {path}");
                compared += 1;
            }
        }
    }
    assert!(compared >= 4, "compared {compared} files");
}

#[test]
fn nort_variant_contexts_score_exactly_one_variant_per_session() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("nort");
    let rows = [ymaze(CROSSLINE), nort()].concat();
    publish_descriptive_with_cancel(
        rows,
        vec![],
        vec![],
        config(json!({"contexts":[
            {"assay":"ymaze","variant":"arm45"},
            {"assay":"nort","variant":"whisker20mm"},
            {"assay":"nort","variant":"nose_only"}],"primary_workers":2})),
        &output,
        &|| false,
    )
    .unwrap();
    let variants = |context: &str, assay: &str| -> Vec<String> {
        metrics(&output.join(context))
            .iter()
            .filter(|r| r["assay"] == assay)
            .map(|r| r["variant"].as_str().unwrap().to_owned())
            .collect()
    };
    // The varied assay takes the context's variant, the others their principal.
    assert_eq!(variants("arm45", "nort"), ["whisker_age"]);
    assert_eq!(variants("whisker20mm", "nort"), ["whisker20mm"]);
    assert_eq!(variants("nose_only", "nort"), ["nose_only"]);
    assert_eq!(variants("whisker20mm", "ymaze"), ["arm45", "arm45"]);
    assert_eq!(variants("arm45", "ymaze"), ["arm45", "arm45"]);
    verify_artifacts(&output).unwrap();
}

/// Re-runs this test in a child whose PATH holds only a failing stand-in
/// `python`, so the parent's environment is never mutated.
fn with_failing_renderer(test: &str) -> bool {
    use std::os::unix::fs::PermissionsExt;
    if std::env::var_os("BEKIPER_FAILING_RENDERER_CHILD").is_some() {
        return true;
    }
    let dir = tempfile::tempdir().unwrap();
    let python = dir.path().join("python");
    std::fs::write(&python, "#!/bin/sh\necho 'renderer exploded' >&2\nexit 4\n").unwrap();
    std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env("PATH", dir.path())
        .env("BEKIPER_FAILING_RENDERER_CHILD", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    false
}

#[test]
fn renderer_failures_surface_with_the_worker_share_of_each_publication() {
    if !with_failing_renderer("renderer_failures_surface_with_the_worker_share_of_each_publication")
    {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let plain =
        |execution: Value| json!({"outputs":{"figures_enabled":true},"execution":execution});
    for (name, rows, execution) in [
        ("single", ymaze(CROSSLINE), json!({"render_workers":2})),
        (
            "contexts",
            ymaze(CROSSLINE),
            json!({"ymaze_primary_thresholds":[25,45],"primary_workers":2,"render_workers":4}),
        ),
        (
            "contexts_unset",
            ymaze(CROSSLINE),
            json!({"ymaze_primary_thresholds":[25,45],"primary_workers":2}),
        ),
    ] {
        let error = publish_descriptive_with_cancel(
            rows,
            vec![],
            vec![],
            plain(execution),
            &dir.path().join(name),
            &|| false,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("renderer exploded"),
            "{name}: {error:#}"
        );
    }
}
