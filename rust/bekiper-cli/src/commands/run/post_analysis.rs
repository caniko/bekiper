use bekiper_ingress::config::PostAnalysisOverrides;
use clap::{ArgAction, Args};
use std::path::PathBuf;

/// Project presentation overrides; scientific policy stays in the registered plan.
#[derive(Default, Args)]
pub(super) struct PostAnalysisArgs {
    /// Independent primary Y-maze export contexts (25,45,65).
    #[arg(long, value_delimiter = ',')]
    ymaze_primary_thresholds: Option<Vec<u8>>,
    /// Maximum concurrent primary export contexts; default 1.
    #[arg(long)]
    primary_analysis_workers: Option<usize>,
    /// Total figure-render processes of the post-analysis publication.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    post_analysis_render_workers: Option<u32>,
    /// Enable cohort post-analysis after native trial summaries.
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_post_analysis")]
    post_analysis: bool,
    /// Disable cohort post-analysis.
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "post_analysis")]
    no_post_analysis: bool,
    /// Registered analysis plan, relative to the project config.
    #[arg(long)]
    post_analysis_plan: Option<PathBuf>,
    /// Fresh immutable cohort run identifier.
    #[arg(long)]
    post_analysis_run_id: Option<String>,
    /// Enable selected post-analysis figures.
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_post_analysis_figures")]
    post_analysis_figures: bool,
    /// Disable post-analysis figures.
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "post_analysis_figures")]
    no_post_analysis_figures: bool,
    /// Figure identifier to select (repeatable or comma-separated).
    #[arg(long, value_delimiter = ',')]
    post_analysis_figure: Option<Vec<String>>,
    /// Figure format (repeatable or comma-separated).
    #[arg(long, value_delimiter = ',', value_parser = ["svg", "png", "pdf"])]
    post_analysis_format: Option<Vec<String>>,
    /// Raster figure DPI.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    post_analysis_dpi: Option<u32>,
    /// Enable the post-analysis PDF report.
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_post_analysis_report")]
    post_analysis_report: bool,
    /// Disable the post-analysis report.
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "post_analysis_report")]
    no_post_analysis_report: bool,
    /// Post-analysis report title.
    #[arg(long)]
    post_analysis_report_title: Option<String>,
    /// PDF report compiler.
    #[arg(long, value_parser = ["latexmk", "lualatex"])]
    post_analysis_report_compiler: Option<String>,
}

impl PostAnalysisArgs {
    pub(super) fn overrides(&self) -> PostAnalysisOverrides {
        let toggle = |enabled, disabled| {
            if enabled {
                Some(true)
            } else if disabled {
                Some(false)
            } else {
                None
            }
        };
        PostAnalysisOverrides {
            ymaze_primary_thresholds: self.ymaze_primary_thresholds.clone(),
            primary_workers: self.primary_analysis_workers,
            contexts: None,
            render_workers: self.post_analysis_render_workers.map(|v| v as usize),
            enabled: toggle(self.post_analysis, self.no_post_analysis),
            plan_path: self.post_analysis_plan.clone(),
            run_id: self.post_analysis_run_id.clone(),
            figures_enabled: toggle(self.post_analysis_figures, self.no_post_analysis_figures),
            figure_selection: self.post_analysis_figure.clone(),
            figure_formats: self.post_analysis_format.clone(),
            figure_dpi: self.post_analysis_dpi,
            report_enabled: toggle(self.post_analysis_report, self.no_post_analysis_report),
            report_title: self.post_analysis_report_title.clone(),
            report_compiler: self.post_analysis_report_compiler.clone(),
        }
    }
}
