use crate::config::{ExperimentConfig, IngressConfig};
use crate::stage::ExperimentOutputDir;
use bekiper_core::error::Result;

pub(super) fn run_inspect(
    experiment: &ExperimentConfig,
    config: &IngressConfig,
    dir: &ExperimentOutputDir,
) -> Result<()> {
    crate::inspection::capture_context(dir, experiment, &config.inspect)?;
    crate::inspection::render(dir, experiment, &config.inspect, &|| false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_inspect_stage_retains_configuration_and_fails_for_missing_analysis_inputs() {
        let root =
            std::env::temp_dir().join(format!("bekiper-missing-inspection-{}", std::process::id()));
        let config: IngressConfig = toml::from_str(
            r#"
            [project]
            project_name = "inspection forwarding"
            data_directory = "."
            [inspect.plot]
            fig_format = ".png"
            min_dpi = 300
            trajectory_gradient = true
            [[experiments]]
            name = "missing"
            task_type = "open_field"
            fps = 30.0
            meters_per_pixel = 0.001
        "#,
        )
        .unwrap();
        let dir = ExperimentOutputDir::new(&root, "missing");
        assert!(run_inspect(&config.experiments[0], &config, &dir).is_err());
        let captured: toml::Value = toml::from_str(
            &std::fs::read_to_string(dir.root().join("inspection_config.toml")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            captured["inspect"]["plot"]["min_dpi"].as_integer(),
            Some(300)
        );
        assert_eq!(
            captured["inspect"]["plot"]["trajectory_gradient"].as_bool(),
            Some(true)
        );
        let receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.root().join("inspection_receipt.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(receipt["status"], "failed");
        assert!(!crate::inspection::is_fresh(&dir, &config.inspect));
        std::fs::remove_dir_all(root).unwrap();
    }
}
