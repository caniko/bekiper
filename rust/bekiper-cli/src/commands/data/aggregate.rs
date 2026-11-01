use std::path::{Path, PathBuf};

use clap::Args;

use bekiper_core::identity::DEFAULT_CONFIG_FILENAME;
use bekiper_ingress::aggregate::run_aggregate;
use bekiper_ingress::config::IngressConfig;
use bekiper_ingress::stage::Stage;

#[derive(Args)]
pub struct AggregateArgs {
    /// Path to the project configuration file.
    #[arg(short, long, default_value = DEFAULT_CONFIG_FILENAME)]
    config: PathBuf,

    /// Existing run root to aggregate. Defaults to the configured output dir.
    #[arg(long)]
    run_root: Option<PathBuf>,

    /// Override the output spreadsheet filename. Legacy `.xlsx` values are
    /// accepted and normalized to `.ods`.
    #[arg(long)]
    output: Option<String>,
}

pub fn run(args: AggregateArgs) -> anyhow::Result<()> {
    let mut config = IngressConfig::from_file(&args.config)?;
    config.aggregate.enabled = true;
    if let Some(output_filename) = args.output {
        config.aggregate.output_filename = output_filename;
    }

    let base_dir = args.config.parent().unwrap_or(Path::new("."));
    config.expand(base_dir, &[Stage::Aggregate])?;
    let run_root = args.run_root.unwrap_or_else(|| {
        base_dir.join(
            config
                .project
                .output_dir
                .as_deref()
                .unwrap_or(Path::new("output")),
        )
    });

    run_aggregate(&config, &run_root)?;

    println!("Aggregation complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_result_toml(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            path,
            r#"
            experiment = "721_seq0_oultraining"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "object_recognition"

            [detailed]
            protocol = "oul_training"
            excluded = false
            total_duration_seconds = 120.0
            total_distance_meters = 5.0
            total_mean_speed = 0.04
            freezing_seconds = 1.2
            analysis_window_seconds = 120.0
            object_labels = []
            exploration_times = []
            bout_counts = []
            "#,
        )
        .unwrap();
    }

    #[test]
    fn aggregate_command_runs_even_when_config_disables_aggregate() {
        let dir = std::env::temp_dir().join("bekiper_cli_aggregate_command");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("output/data/trial_a")).unwrap();
        let config_path = dir.join("bekiper_config.toml");
        std::fs::write(
            &config_path,
            r#"
            [project]
            project_name = "aggregate"
            data_directory = "."
            output_dir = "output"

            [aggregate]
            enabled = false
            output_filename = "disabled.ods"
            "#,
        )
        .unwrap();
        write_result_toml(&dir.join("output/data/trial_a/result.toml"));

        run(AggregateArgs {
            config: config_path.clone(),
            run_root: None,
            output: Some("renamed.xlsx".into()),
        })
        .unwrap();

        assert!(dir.join("output/renamed.ods").exists());
        let mut config = IngressConfig::from_file(&config_path).unwrap();
        config.aggregate.output_filename = "renamed.xlsx".into();
        assert!(
            bekiper_ingress::aggregate::aggregate_state_path_for_config(
                &config,
                &dir.join("output")
            )
            .exists()
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
