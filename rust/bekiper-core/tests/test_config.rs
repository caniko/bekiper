use bekiper_core::config::*;
use std::collections::HashMap;

fn env_lookup<'a>(
    map: &'a HashMap<&'static str, &'static str>,
) -> impl Fn(&str) -> Option<String> + 'a {
    move |k| map.get(k).map(|v| (*v).to_string())
}

#[test]
fn runtime_settings_defaults() {
    let settings = RuntimeSettings::default();
    assert!((settings.minimum_seconds_tolerance - 0.5).abs() < f64::EPSILON);
    assert!((settings.maximum_seconds_distraction - 1.0 / 3.0).abs() < f64::EPSILON);
    assert!(settings.num_threads.is_none());
    assert!(settings.memory_scheduler_enabled);
    assert!((settings.max_ram_usage_fraction - 0.80).abs() < f64::EPSILON);
}

#[test]
fn runtime_settings_serde_with_defaults() {
    let toml_str = "";
    let settings: RuntimeSettings = toml::from_str(toml_str).unwrap();
    assert!((settings.minimum_seconds_tolerance - 0.5).abs() < f64::EPSILON);
    assert!(settings.num_threads.is_none());
    assert!(settings.memory_scheduler_enabled);
    assert!((settings.max_ram_usage_fraction - 0.80).abs() < f64::EPSILON);
}

#[test]
fn runtime_settings_serde_with_overrides() {
    let toml_str = r#"
        minimum_seconds_tolerance = 1.0
        maximum_seconds_distraction = 0.5
        num_threads = 4
        memory_scheduler_enabled = false
        max_ram_usage_fraction = 0.65
    "#;
    let settings: RuntimeSettings = toml::from_str(toml_str).unwrap();
    assert!((settings.minimum_seconds_tolerance - 1.0).abs() < f64::EPSILON);
    assert!((settings.maximum_seconds_distraction - 0.5).abs() < f64::EPSILON);
    assert_eq!(settings.num_threads, Some(4));
    assert!(!settings.memory_scheduler_enabled);
    assert!((settings.max_ram_usage_fraction - 0.65).abs() < f64::EPSILON);
}

#[test]
fn project_config_serde() {
    let toml_str = r#"
        project_name = "my_experiment"
        data_directory = "/data/experiment1"
    "#;
    let config: ProjectConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.project_name, "my_experiment");
    assert_eq!(
        config.data_directory,
        std::path::PathBuf::from("/data/experiment1")
    );
    // Runtime should be defaults
    assert!((config.runtime.minimum_seconds_tolerance - 0.5).abs() < f64::EPSILON);
}

#[test]
fn project_config_with_runtime() {
    let toml_str = r#"
        project_name = "test"
        data_directory = "/tmp"
        [runtime]
        minimum_seconds_tolerance = 2.0
        num_threads = 8
    "#;
    let config: ProjectConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.project_name, "test");
    assert!((config.runtime.minimum_seconds_tolerance - 2.0).abs() < f64::EPSILON);
    assert_eq!(config.runtime.num_threads, Some(8));
}

#[test]
fn runtime_env_overlay_applies_all_supported_overrides() {
    let mut settings = RuntimeSettings::default();
    let env = HashMap::from([
        ("BEKIPER_RUNTIME_NUM_THREADS", "6"),
        ("BEKIPER_RUNTIME_MEMORY_SCHEDULER", "false"),
        ("BEKIPER_RUNTIME_MAX_RAM_USAGE", "0.72"),
    ]);

    settings.apply_env_overrides_from(env_lookup(&env));

    assert_eq!(settings.num_threads, Some(6));
    assert!(!settings.memory_scheduler_enabled);
    assert!((settings.max_ram_usage_fraction - 0.72).abs() < f64::EPSILON);
}

#[test]
fn runtime_env_overlay_rejects_invalid_values() {
    let mut settings = RuntimeSettings::default();
    let env = HashMap::from([
        ("BEKIPER_RUNTIME_NUM_THREADS", "0"),
        ("BEKIPER_RUNTIME_MEMORY_SCHEDULER", "maybe"),
        ("BEKIPER_RUNTIME_MAX_RAM_USAGE", "1.5"),
    ]);

    settings.apply_env_overrides_from(env_lookup(&env));

    assert!(settings.num_threads.is_none());
    assert!(settings.memory_scheduler_enabled);
    assert!((settings.max_ram_usage_fraction - 0.80).abs() < f64::EPSILON);
}

#[test]
fn project_config_roundtrip() {
    let config = ProjectConfig {
        project_name: "roundtrip".to_string(),
        data_directory: "/data".into(),
        output_dir: None,
        runtime: RuntimeSettings::default(),
    };
    let serialized = toml::to_string(&config).unwrap();
    let deserialized: ProjectConfig = toml::from_str(&serialized).unwrap();
    assert_eq!(deserialized.project_name, "roundtrip");
}
