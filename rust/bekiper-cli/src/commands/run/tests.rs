use super::*;
use clap::Parser;

fn parse(flags: &[&str]) -> RunArgs {
    let mut command = vec!["bekiper", "run"];
    command.extend_from_slice(flags);
    let crate::Cli::Run(args) = crate::Cli::try_parse_from(command).unwrap() else {
        panic!("expected project run command");
    };
    args
}

#[test]
fn project_cli_post_analysis_overrides_environment_and_toml() {
    let args = parse(&[
        "--post-analysis",
        "--post-analysis-plan",
        "cli-plan.json",
        "--post-analysis-run-id",
        "cli-run",
        "--post-analysis-figures",
        "--post-analysis-figure",
        "figure_oft",
        "--post-analysis-format",
        "svg",
        "--post-analysis-dpi",
        "180",
        "--post-analysis-report",
        "--post-analysis-report-title",
        "CLI report",
        "--post-analysis-report-compiler",
        "lualatex",
    ]);
    let parent = std::env::temp_dir().join(format!(
        "bekiper-cli-precedence-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&parent).unwrap();
    let path = parent.join("project.toml");
    std::fs::write(
        &path,
        r#"
        [project]
        project_name = "precedence"
        data_directory = "."
        [post_analysis]
        enabled = false
        plan_path = "file-plan.json"
        run_id = "file-run"
        [post_analysis.figures]
        enabled = false
        selection = ["figure_nort"]
        formats = ["pdf"]
        dpi = 120
        [post_analysis.report]
        enabled = false
        title = "File report"
        compiler = "latexmk"
    "#,
    )
    .unwrap();
    let lookup = |key: &str| match key {
        "BEKIPER_POST_ANALYSIS_DISABLED" => Some("true".into()),
        "BEKIPER_POST_ANALYSIS_PLAN_PATH" => Some("env-plan.json".into()),
        "BEKIPER_POST_ANALYSIS_RUN_ID" => Some("env-run".into()),
        "BEKIPER_POST_ANALYSIS_FIGURES_ENABLED" | "BEKIPER_POST_ANALYSIS_REPORT_ENABLED" => {
            Some("false".into())
        }
        "BEKIPER_POST_ANALYSIS_FIGURES_SELECTION" => Some("figure_ymaze".into()),
        "BEKIPER_POST_ANALYSIS_FIGURES_FORMATS" => Some("png".into()),
        "BEKIPER_POST_ANALYSIS_FIGURES_DPI" => Some("160".into()),
        "BEKIPER_POST_ANALYSIS_REPORT_TITLE" => Some("Environment report".into()),
        "BEKIPER_POST_ANALYSIS_REPORT_COMPILER" => Some("latexmk".into()),
        _ => None,
    };
    let config = IngressConfig::from_file_with_post_analysis_overrides_from(
        &path,
        &args.post_analysis.overrides(),
        lookup,
    )
    .unwrap()
    .post_analysis;
    assert!(config.enabled && config.figures.enabled && config.report.enabled);
    assert_eq!(config.plan_path, Some(PathBuf::from("cli-plan.json")));
    assert_eq!(config.run_id.as_deref(), Some("cli-run"));
    assert_eq!(config.figures.selection, Some(vec!["figure_oft".into()]));
    assert_eq!(config.figures.formats, ["svg"]);
    assert_eq!(config.figures.dpi, 180);
    assert_eq!(config.report.title, "CLI report");
    assert_eq!(config.report.compiler, "lualatex");
    let from_env = IngressConfig::from_file_with_post_analysis_overrides_from(
        &path,
        &parse(&[]).post_analysis.overrides(),
        lookup,
    )
    .unwrap()
    .post_analysis;
    assert_eq!(from_env.figures.dpi, 160);
    let from_file = IngressConfig::from_file_with_post_analysis_overrides_from(
        &path,
        &parse(&[]).post_analysis.overrides(),
        |_| None,
    )
    .unwrap()
    .post_analysis;
    assert_eq!(from_file.figures.dpi, 120);
    std::fs::remove_dir_all(parent).unwrap();
}

#[test]
fn project_cli_explicit_disable_and_invalid_raster_settings_are_preserved() {
    let args = parse(&[
        "--no-post-analysis",
        "--no-post-analysis-figures",
        "--no-post-analysis-report",
    ]);
    let overrides = args.post_analysis.overrides();
    assert_eq!(overrides.enabled, Some(false));
    assert_eq!(overrides.figures_enabled, Some(false));
    assert_eq!(overrides.report_enabled, Some(false));
    assert!(crate::Cli::try_parse_from(["bekiper", "run", "--post-analysis-dpi", "0"]).is_err());
    assert!(
        crate::Cli::try_parse_from(["bekiper", "run", "--post-analysis-format", "tex"]).is_err()
    );
}

#[test]
fn project_cli_accepts_explicit_dwell_duration_and_comparison() {
    let args = parse(&[
        "--ymaze-dwell-seconds",
        "1.0",
        "--ymaze-dwell-comparison",
        "greater_than",
    ]);
    assert_eq!(args.ymaze_dwell_seconds, Some(1.0));
    assert_eq!(
        args.ymaze_dwell_comparison,
        Some(bekiper_feature::tolerance::DurationComparison::GreaterThan)
    );
    assert!(
        crate::Cli::try_parse_from([
            "bekiper",
            "run",
            "--ymaze-dwell-comparison",
            "approximately"
        ])
        .is_err()
    );
}
