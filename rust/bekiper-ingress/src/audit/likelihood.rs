//! Raw-pose scoring and deterministic likelihood rankings.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::{
    AuditDiagnostic, AuditSeverity, LikelihoodAggregateRow, LikelihoodAuditReport,
    LikelihoodPairRow, audit_exp_output_dir, discover_pose_labels,
    focused_likelihood_labels_for_config, focused_likelihood_labels_for_task, fraction,
    likelihood_pair_row, load_audit_config, scan_parquet_df, task_phase_key, top_likelihood_pairs,
};
use crate::config::{ExperimentConfig, IngressConfig};
use crate::stage::ExperimentOutputDir;

pub fn check_likelihood(path: &Path, all_labels: bool) -> LikelihoodAuditReport {
    const TOP_N: usize = 20;
    let mut report = LikelihoodAuditReport::new("likelihood", Some(path.to_path_buf()));
    let validation_command = format!("bekiper check likelihood -c {}", path.display());
    let (config, output_root) = match load_audit_config(path) {
        Ok(context) => context,
        Err(diagnostic) => {
            report.push(*diagnostic);
            return report;
        }
    };
    report.summary.experiments_expected = config.experiments.len();
    report.summary.likelihood_threshold = config.pipeline.likelihood_threshold;
    report.summary.all_labels = all_labels;
    report.summary.pose_label_model = config.pipeline.pose_label_model;
    report.summary.focused_labels = focused_likelihood_labels_for_config(&config);

    let mut rankings = LikelihoodRankings::default();
    for experiment in &config.experiments {
        let exp_dir = audit_exp_output_dir(&output_root, experiment);
        for row in inspect_experiment(
            &mut report,
            &config,
            experiment,
            &exp_dir,
            &validation_command,
        ) {
            rankings.add(row);
        }
    }
    rankings.finish(&mut report, TOP_N);
    report.push(
        AuditDiagnostic::new(
            AuditSeverity::Ok,
            "likelihood",
            format!(
                "inspected {}/{} experiment(s), {} frame(s), {} scored label stream(s); overall low-likelihood {:.1}% at threshold {:.3}",
                report.summary.experiments_inspected,
                report.summary.experiments_expected,
                report.summary.total_frames,
                report.summary.labels_scored,
                report.summary.low_likelihood_fraction() * 100.0,
                report.summary.likelihood_threshold
            ),
        )
        .artifact_path(path)
        .validation_command(validation_command),
    );
    report
}

fn inspect_experiment(
    report: &mut LikelihoodAuditReport,
    config: &IngressConfig,
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
    validation_command: &str,
) -> Vec<LikelihoodPairRow> {
    let poses_path = exp_dir.poses_parquet();
    let failure = |message, cause: String, hint| {
        AuditDiagnostic::new(AuditSeverity::Error, "likelihood", message)
            .experiment(&experiment.name)
            .artifact_path(&poses_path)
            .cause(cause)
            .fix_hint(hint)
            .validation_command(validation_command)
    };
    if !poses_path.exists() {
        report.summary.experiments_missing_poses += 1;
        report.push(failure(
            "poses.parquet is missing",
            "likelihood scoring requires the raw pose checkpoint".into(),
            "run the pose stage for this experiment",
        ));
        return Vec::new();
    }
    let raw_df = match scan_parquet_df(&poses_path) {
        Ok(df) => df,
        Err(error) => {
            report.push(failure(
                "poses.parquet is unreadable",
                error.to_string(),
                "regenerate poses.parquet for this experiment",
            ));
            return Vec::new();
        }
    };
    report.summary.experiments_inspected += 1;
    report.summary.total_frames += raw_df.height();
    let focused =
        focused_likelihood_labels_for_task(config.pipeline.pose_label_model, experiment.task_type);
    let mut rows = Vec::new();
    for label in discover_pose_labels(&raw_df)
        .into_iter()
        .filter(|label| report.summary.all_labels || focused.contains(&label.as_str()))
    {
        match likelihood_pair_row(
            &raw_df,
            &experiment.name,
            &label,
            config.pipeline.likelihood_threshold,
        ) {
            Some(row) => {
                report.summary.labels_scored += 1;
                report.summary.low_likelihood_frames += row.low_likelihood_frames;
                report.summary.scored_likelihood_values += row.frames;
                rows.push(row);
            }
            None => report.summary.labels_skipped_missing_likelihood += 1,
        }
    }
    if !exp_dir.augmented_parquet().exists() {
        report.summary.augmented_missing_count += 1;
    }
    rows
}

#[derive(Default)]
struct LikelihoodRankings {
    pairs: Vec<LikelihoodPairRow>,
    labels: HashMap<String, LikelihoodAccumulator>,
    experiments: HashMap<String, LikelihoodAccumulator>,
    task_phases: HashMap<String, LikelihoodAccumulator>,
}

impl LikelihoodRankings {
    fn add(&mut self, row: LikelihoodPairRow) {
        self.labels.entry(row.label.clone()).or_default().add(&row);
        self.experiments
            .entry(row.experiment.clone())
            .or_default()
            .add(&row);
        if let Some(phase) = task_phase_key(&row.experiment) {
            self.task_phases.entry(phase).or_default().add(&row);
        }
        self.pairs.push(row);
    }

    fn finish(self, report: &mut LikelihoodAuditReport, limit: usize) {
        report.worst_pairs = top_likelihood_pairs(self.pairs, limit);
        report.worst_labels = top_likelihood_aggregates(self.labels, limit);
        report.worst_experiments = top_likelihood_aggregates(self.experiments, limit);
        report.task_phase_summary = top_likelihood_aggregates(self.task_phases, usize::MAX);
    }
}

#[derive(Default)]
struct LikelihoodAccumulator {
    experiments: HashSet<String>,
    labels: HashSet<String>,
    frames: usize,
    low_likelihood_frames: usize,
    likelihood_sum: f64,
    likelihood_count: usize,
}

impl LikelihoodAccumulator {
    fn add(&mut self, row: &LikelihoodPairRow) {
        self.experiments.insert(row.experiment.clone());
        self.labels.insert(row.label.clone());
        self.frames += row.frames;
        self.low_likelihood_frames += row.low_likelihood_frames;
        if let Some(mean) = row.mean_likelihood {
            self.likelihood_sum += mean * row.frames as f64;
            self.likelihood_count += row.frames;
        }
    }

    fn into_row(self, key: String) -> LikelihoodAggregateRow {
        LikelihoodAggregateRow {
            key,
            experiments: self.experiments.len(),
            labels: self.labels.len(),
            frames: self.frames,
            low_likelihood_frames: self.low_likelihood_frames,
            low_likelihood_fraction: fraction(self.low_likelihood_frames, self.frames),
            mean_likelihood: (self.likelihood_count > 0)
                .then_some(self.likelihood_sum / self.likelihood_count as f64),
        }
    }
}

fn top_likelihood_aggregates(
    rows: HashMap<String, LikelihoodAccumulator>,
    limit: usize,
) -> Vec<LikelihoodAggregateRow> {
    let mut rows = rows
        .into_iter()
        .map(|(key, acc)| acc.into_row(key))
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        b.low_likelihood_fraction
            .total_cmp(&a.low_likelihood_fraction)
            .then_with(|| b.low_likelihood_frames.cmp(&a.low_likelihood_frames))
            .then_with(|| a.key.cmp(&b.key))
    });
    rows.truncate(limit);
    rows
}
