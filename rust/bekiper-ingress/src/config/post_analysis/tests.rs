use super::*;
use std::collections::HashMap;

fn test_config() -> PostAnalysisConfig {
    PostAnalysisConfig::default()
}

fn lookup_from_map(map: &HashMap<String, String>) -> impl Fn(&str) -> Option<String> + '_ {
    move |k| map.get(k).cloned()
}

#[test]
fn primary_context_overlays_preserve_explicit_disable_and_constructor_priority() {
    let mut settings: PostAnalysisConfig =
        toml::from_str("ymaze_primary_thresholds = [25,45,65]\nprimary_workers = 3").unwrap();
    settings.apply_env_overrides_from(|key| match key {
        "BEKIPER_YMAZE_PRIMARY_THRESHOLDS" => Some("".into()),
        "BEKIPER_PRIMARY_ANALYSIS_WORKERS" => Some("2".into()),
        _ => None,
    });
    settings.validate().unwrap();
    assert_eq!(settings.ymaze_primary_thresholds, Some(vec![]));
    assert_eq!(settings.primary_workers, 2);
    settings.apply_env_overrides_from(|key| {
        matches!(
            key,
            "BEKIPER_YMAZE_PRIMARY_THRESHOLDS" | "BEKIPER_PRIMARY_ANALYSIS_WORKERS"
        )
        .then(|| "invalid".into())
    });
    assert!(settings.validate().is_err());
    settings.apply_overrides(&PostAnalysisOverrides {
        ymaze_primary_thresholds: Some(vec![25, 65]),
        primary_workers: Some(1),
        ..Default::default()
    });
    settings.validate().unwrap();
    assert_eq!(settings.ymaze_primary_thresholds, Some(vec![25, 65]));
    assert_eq!(settings.primary_workers, 1);
    let mut plan = serde_json::json!({"execution":{"ymaze_primary_thresholds":[25,45,65]}});
    bekiper_post_analysis::inputs::resolve_primary_runs(
        &mut plan,
        &bekiper_post_analysis::RunOptions {
            primary_thresholds: Some(vec![]),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        plan["execution"]["ymaze_primary_thresholds"],
        serde_json::json!([])
    );
}

#[test]
fn genotype_views_are_reserved_for_each_concurrent_primary_context() {
    let settings: PostAnalysisConfig = toml::from_str(
        "enabled = true\nymaze_primary_thresholds = [25,45,65]\nprimary_workers = 2",
    )
    .unwrap();
    let retained = 1024 * 1024;
    let plain = serde_json::json!({});
    let plan = serde_json::json!({"statistics":{"hcar1_policy_plan":{"primary":"A","policies":["A","B","C"]}}});
    let original = settings
        .cohort_workspace_bytes(retained, Some(&plain))
        .unwrap();
    assert_eq!(
        settings
            .cohort_workspace_bytes(retained, Some(&plan))
            .unwrap()
            - original,
        3 * retained * 2
    );
    assert!(
        settings
            .cohort_workspace_bytes(u64::MAX / 2, Some(&plan))
            .is_err()
    );
    let mut disabled = settings;
    disabled.enabled = false;
    assert_eq!(
        disabled
            .cohort_workspace_bytes(retained, Some(&plan))
            .unwrap(),
        disabled
            .cohort_workspace_bytes(retained, Some(&plain))
            .unwrap()
    );
}

#[test]
fn post_analysis_env_overlay_disabled() {
    let mut cfg = test_config();
    let mut env = HashMap::new();
    env.insert(
        "BEKIPER_POST_ANALYSIS_DISABLED".to_string(),
        "1".to_string(),
    );
    cfg.apply_env_overrides_from(lookup_from_map(&env));
    assert!(!cfg.enabled);
}

#[test]
fn post_analysis_env_overlay_enabled_explicit() {
    let mut cfg = test_config();
    cfg.enabled = false; // start disabled
    let mut env = HashMap::new();
    env.insert(
        "BEKIPER_POST_ANALYSIS_DISABLED".to_string(),
        "0".to_string(),
    );
    cfg.apply_env_overrides_from(lookup_from_map(&env));
    assert!(cfg.enabled);
}

#[test]
fn post_analysis_env_overlay_plan_path() {
    let mut cfg = test_config();
    let mut env = HashMap::new();
    env.insert(
        "BEKIPER_POST_ANALYSIS_PLAN_PATH".to_string(),
        "analysis/plan.json".to_string(),
    );
    cfg.apply_env_overrides_from(lookup_from_map(&env));
    assert_eq!(cfg.plan_path, Some(PathBuf::from("analysis/plan.json")));
}

#[test]
fn post_analysis_env_overlay_run_id() {
    let mut cfg = test_config();
    let mut env = HashMap::new();
    env.insert(
        "BEKIPER_POST_ANALYSIS_RUN_ID".to_string(),
        "run-2024-01-15".to_string(),
    );
    cfg.apply_env_overrides_from(lookup_from_map(&env));
    assert_eq!(cfg.run_id, Some("run-2024-01-15".to_string()));
}

#[test]
fn post_analysis_figures_env_overlays() {
    let mut cfg = test_config();
    let mut env = HashMap::new();
    env.insert(
        "BEKIPER_POST_ANALYSIS_FIGURES_ENABLED".to_string(),
        "0".to_string(),
    );
    env.insert(
        "BEKIPER_POST_ANALYSIS_FIGURES_SELECTION".to_string(),
        "figure_oft,figure_nort".to_string(),
    );
    env.insert(
        "BEKIPER_POST_ANALYSIS_FIGURES_DPI".to_string(),
        "300".to_string(),
    );
    env.insert(
        "BEKIPER_POST_ANALYSIS_FIGURES_FORMATS".to_string(),
        "pdf,svg".to_string(),
    );
    cfg.apply_env_overrides_from(lookup_from_map(&env));

    assert!(!cfg.figures.enabled);
    assert_eq!(
        cfg.figures.selection,
        Some(vec!["figure_oft".into(), "figure_nort".into()])
    );
    assert_eq!(cfg.figures.dpi, 300);
    assert_eq!(cfg.figures.formats, vec!["pdf", "svg"]);
}

#[test]
fn post_analysis_report_env_overlays() {
    let mut cfg = test_config();
    let mut env = HashMap::new();
    env.insert(
        "BEKIPER_POST_ANALYSIS_REPORT_ENABLED".to_string(),
        "0".to_string(),
    );
    env.insert(
        "BEKIPER_POST_ANALYSIS_REPORT_TITLE".to_string(),
        "Custom Title".to_string(),
    );
    env.insert(
        "BEKIPER_POST_ANALYSIS_REPORT_COMPILER".to_string(),
        "latexmk".to_string(),
    );
    cfg.apply_env_overrides_from(lookup_from_map(&env));

    assert!(!cfg.report.enabled);
    assert_eq!(cfg.report.title, "Custom Title");
    assert_eq!(cfg.report.compiler, "latexmk");
}

#[test]
fn post_analysis_resolved_plan_path_relative() {
    let mut cfg = test_config();
    cfg.plan_path = Some(PathBuf::from("analysis/plan.json"));
    let config_parent = PathBuf::from("/project/config");
    let resolved = cfg.resolved_plan_path(&config_parent);
    assert_eq!(
        resolved,
        Some(PathBuf::from("/project/config/analysis/plan.json"))
    );
}

#[test]
fn post_analysis_resolved_plan_path_absolute() {
    let mut cfg = test_config();
    cfg.plan_path = Some(PathBuf::from("/absolute/plan.json"));
    let config_parent = PathBuf::from("/project/config");
    let resolved = cfg.resolved_plan_path(&config_parent);
    assert_eq!(resolved, Some(PathBuf::from("/absolute/plan.json")));
}

#[test]
fn post_analysis_effective_frozen_plan_prefers_embedded() {
    let mut cfg = test_config();
    cfg.frozen_plan = Some(serde_json::json!({"embedded": true}));
    cfg.plan_path = Some(PathBuf::from("analysis/plan.json"));
    let config_parent = PathBuf::from("/project/config");
    let plan = cfg.effective_frozen_plan(&config_parent).unwrap();
    assert_eq!(plan, Some(serde_json::json!({"embedded": true})));
}

#[test]
fn post_analysis_effective_frozen_fingerprints_prefers_embedded() {
    let mut cfg = test_config();
    cfg.frozen_fingerprints = vec![serde_json::json!({"name": "embedded"})];
    cfg.frozen_plan = Some(serde_json::json!({"source_fingerprints": [{"name": "from_plan"}]}));
    let config_parent = PathBuf::from("/project/config");
    let fingerprints = cfg.effective_frozen_fingerprints(&config_parent).unwrap();
    assert_eq!(fingerprints, vec![serde_json::json!({"name": "embedded"})]);
}

#[test]
fn constructor_overrides_win_over_env_and_invalid_paths_fail() {
    let mut cfg = PostAnalysisConfig {
        enabled: true,
        plan_path: Some("analysis.json".into()),
        ..Default::default()
    };
    cfg.apply_env_overrides_from(|key| {
        (key == "BEKIPER_POST_ANALYSIS_FIGURES_ENABLED").then(|| "0".into())
    });
    cfg.apply_overrides(&PostAnalysisOverrides {
        figures_enabled: Some(true),
        run_id: Some("../outside".into()),
        ..Default::default()
    });
    assert!(cfg.figures.enabled);
    assert!(cfg.validate().is_err());
    cfg.run_id = Some("cohort-v1".into());
    cfg.validate().unwrap();
    assert!(
        cfg.effective_frozen_plan(&PathBuf::from("/nonexistent-project"))
            .is_err()
    );
}

#[test]
fn project_loader_validates_after_constructor_env_and_toml_layers() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "bekiper-post-analysis-layers-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("bekiper_config.toml");
    std::fs::write(
        &path,
        r#"
            [project]
            project_name = "synthetic-layer-test"
            data_directory = "."
            [post_analysis]
            enabled = false
            [post_analysis.figures]
            dpi = 100
        "#,
    )
    .unwrap();
    let options = PostAnalysisOverrides {
        enabled: Some(true),
        plan_path: Some("registered-plan.json".into()),
        figure_dpi: Some(300),
        ..Default::default()
    };
    let config = super::super::IngressConfig::from_file_with_post_analysis_overrides_from(
        &path,
        &options,
        |key| match key {
            "BEKIPER_POST_ANALYSIS_DISABLED" => Some("0".into()),
            "BEKIPER_POST_ANALYSIS_FIGURES_DPI" => Some("200".into()),
            _ => None,
        },
    )
    .unwrap();
    assert!(config.post_analysis.enabled);
    assert_eq!(config.post_analysis.figures.dpi, 300);
    assert_eq!(
        config.post_analysis.plan_path,
        Some("registered-plan.json".into())
    );
    let disabled = super::super::IngressConfig::from_file_with_post_analysis_overrides_from(
        &path,
        &PostAnalysisOverrides {
            enabled: Some(false),
            ..Default::default()
        },
        |key| (key == "BEKIPER_POST_ANALYSIS_DISABLED").then(|| "0".into()),
    )
    .unwrap();
    assert!(!disabled.post_analysis.enabled);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn contexts_and_render_workers_flow_through_env_overrides_and_validation() {
    let mut settings = test_config();
    settings.apply_env_overrides_from(|key| {
        (key == "BEKIPER_POST_ANALYSIS_RENDER_WORKERS").then(|| " 3 ".into())
    });
    assert_eq!(settings.render_workers, Some(3));
    settings.apply_env_overrides_from(|key| {
        (key == "BEKIPER_POST_ANALYSIS_RENDER_WORKERS").then(|| "x".into())
    });
    assert_eq!(settings.render_workers, Some(0));
    assert!(settings.validate().is_err());

    let context = |variant: &str| bekiper_post_analysis::contracts::ContextSpec {
        assay: "ymaze".into(),
        variant: variant.into(),
    };
    let mut settings = test_config();
    settings.apply_overrides(&PostAnalysisOverrides {
        contexts: Some(vec![context("arm25"), context("arm45")]),
        render_workers: Some(2),
        ..Default::default()
    });
    assert_eq!(settings.render_workers, Some(2));
    settings.primary_workers = 8;
    settings.validate().unwrap();
    // Concurrency follows the contexts, not the thresholds.
    assert_eq!(settings.primary_concurrency(None), 2);
    settings.contexts = Some(vec![context("arm25"), context("arm25")]);
    assert!(settings.validate().is_err());
    settings.contexts = Some(vec![context("arm25")]);
    settings.ymaze_primary_thresholds = Some(vec![25]);
    assert!(settings.validate().is_err());
}
