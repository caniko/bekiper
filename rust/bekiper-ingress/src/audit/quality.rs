//! Audit quality responsibilities.
use super::*;

pub fn check_pose_quality(path: &Path) -> AuditReport {
    let mut report = AuditReport::new("pose-quality", Some(path.to_path_buf()));
    let validation_command = format!("bekiper check pose-quality -c {}", path.display());
    let (config, output_root) = match load_audit_config(path) {
        Ok(context) => context,
        Err(diagnostic) => {
            report.push(*diagnostic);
            return report;
        }
    };
    let mut inspected = 0usize;
    for experiment in &config.experiments {
        let exp_dir = audit_exp_output_dir(&output_root, experiment);
        match pose_quality_diagnostics_for_experiment(
            experiment,
            &exp_dir,
            &config.pipeline.quality,
            config.pipeline.likelihood_threshold,
            config.pipeline.max_velocity,
            &validation_command,
        ) {
            Ok(diagnostics) => {
                inspected += 1;
                report.extend(diagnostics);
            }
            Err(error) => report.push(
                AuditDiagnostic::new(
                    AuditSeverity::Error,
                    "pose quality",
                    "pose quality report failed",
                )
                .experiment(&experiment.name)
                .cause(error.to_string())
                .fix_hint("repair the pose artifacts for this experiment")
                .validation_command(&validation_command),
            ),
        }
    }

    if inspected == 0 {
        report.push(
            AuditDiagnostic::new(
                AuditSeverity::Skipped,
                "pose quality",
                "no experiments were available to inspect",
            )
            .validation_command(validation_command),
        );
    }

    report
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct LabelPoseQuality {
    pub(super) label: String,
    pub(super) total_frames: usize,
    pub(super) low_likelihood_count: Option<usize>,
    pub(super) raw_missing_count: usize,
    pub(super) augmented_missing_count: Option<usize>,
    pub(super) longest_gap_frames: usize,
    pub(super) short_gap_count: usize,
    pub(super) medium_gap_count: usize,
    pub(super) long_gap_count: usize,
    pub(super) interpolation_burden_count: Option<usize>,
    pub(super) augmented_jump_count: Option<usize>,
    pub(super) worst_augmented_displacement: Option<f64>,
}

impl LabelPoseQuality {
    fn low_likelihood_fraction(&self) -> Option<f64> {
        self.low_likelihood_count
            .map(|count| fraction(count, self.total_frames))
    }

    fn raw_missing_fraction(&self) -> f64 {
        fraction(self.raw_missing_count, self.total_frames)
    }

    fn augmented_missing_fraction(&self) -> Option<f64> {
        self.augmented_missing_count
            .map(|count| fraction(count, self.total_frames))
    }

    fn interpolation_burden_fraction(&self) -> Option<f64> {
        self.interpolation_burden_count
            .map(|count| fraction(count, self.total_frames))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GapStats {
    pub(super) longest: usize,
    pub(super) short: usize,
    pub(super) medium: usize,
    pub(super) long: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct JumpStats {
    pub(super) count: usize,
    pub(super) worst_displacement: f64,
}

pub(super) fn pose_quality_diagnostics_for_experiment(
    experiment: &ExperimentConfig,
    exp_dir: &ExperimentOutputDir,
    quality: &PoseQualityConfig,
    likelihood_threshold: f64,
    max_velocity: f64,
    validation_command: &str,
) -> Result<Vec<AuditDiagnostic>> {
    let poses_path = exp_dir.poses_parquet();
    if !poses_path.exists() {
        return Ok(vec![
            AuditDiagnostic::new(
                AuditSeverity::Error,
                "pose quality",
                "poses.parquet is missing",
            )
            .experiment(&experiment.name)
            .artifact_path(&poses_path)
            .cause("pose quality metrics require the raw pose checkpoint")
            .fix_hint("run the pose stage for this experiment")
            .validation_command(validation_command),
        ]);
    }

    let raw_df = scan_parquet_df(&poses_path)?;
    let augmented_path = exp_dir.augmented_parquet();
    let augmented_df = if augmented_path.exists() {
        Some(scan_parquet_df(&augmented_path)?)
    } else {
        None
    };

    let labels = discover_pose_labels(&raw_df);
    if labels.is_empty() {
        return Ok(vec![
            AuditDiagnostic::new(
                AuditSeverity::Error,
                "pose quality",
                "poses.parquet has no *_x/*_y label columns",
            )
            .experiment(&experiment.name)
            .artifact_path(&poses_path)
            .cause("pose quality metrics require coordinate columns named <label>_x and <label>_y")
            .fix_hint("regenerate poses.parquet from a supported pose source")
            .validation_command(validation_command),
        ]);
    }

    let scale = load_pose_quality_scale(exp_dir);
    let jump_threshold = scale
        .map(|meters_per_pixel| max_velocity * meters_per_pixel)
        .unwrap_or(max_velocity);
    let label_metrics = compute_pose_quality_metrics(
        &raw_df,
        augmented_df.as_ref(),
        &labels,
        likelihood_threshold,
        jump_threshold,
    )?;

    let mut diagnostics = Vec::new();
    diagnostics.push(summary_diagnostic(
        experiment,
        &poses_path,
        &augmented_path,
        &label_metrics,
        validation_command,
    ));

    let fps = experiment.effective_fps().unwrap_or(30.0).max(1.0);
    let max_gap_frames = (quality.max_null_gap_seconds * fps).ceil() as usize;
    for metric in &label_metrics {
        if let Some(frac) = metric.low_likelihood_fraction()
            && frac > quality.max_low_likelihood_fraction
        {
            diagnostics.push(
                AuditDiagnostic::new(
                    AuditSeverity::Warning,
                    "pose quality",
                    format!(
                        "{} low-likelihood rate is {:.1}% ({}/{} frames)",
                        metric.label,
                        frac * 100.0,
                        metric.low_likelihood_count.unwrap_or(0),
                        metric.total_frames
                    ),
                )
                .experiment(&experiment.name)
                .artifact_path(&poses_path)
                .cause(format!(
                    "likelihood is below {likelihood_threshold:.3} more often than the configured {:.1}% threshold",
                    quality.max_low_likelihood_fraction * 100.0
                ))
                .fix_hint("inspect model fit, crop/scale settings, and label visibility before relying on interpolation")
                .validation_command(validation_command),
            );
        }

        if metric.longest_gap_frames > max_gap_frames {
            diagnostics.push(
                AuditDiagnostic::new(
                    AuditSeverity::Warning,
                    "pose quality",
                    format!(
                        "{} longest invalid gap is {} frame(s), above {:.3}s",
                        metric.label, metric.longest_gap_frames, quality.max_null_gap_seconds
                    ),
                )
                .experiment(&experiment.name)
                .artifact_path(&poses_path)
                .cause(format!(
                    "gap buckets: short={}, medium={}, long={}",
                    metric.short_gap_count, metric.medium_gap_count, metric.long_gap_count
                ))
                .fix_hint("long gaps should be handled by model/crop/rerun checks rather than interpolation alone")
                .validation_command(validation_command),
            );
        } else if metric.longest_gap_frames > 0 {
            diagnostics.push(
                AuditDiagnostic::new(
                    AuditSeverity::Ok,
                    "pose quality",
                    format!(
                        "{} has invalid gaps, longest {} frame(s); interpolation tuning may help",
                        metric.label, metric.longest_gap_frames
                    ),
                )
                .experiment(&experiment.name)
                .artifact_path(&poses_path)
                .validation_command(validation_command),
            );
        }

        if let Some(jump_count) = metric.augmented_jump_count
            && jump_count > quality.max_augmented_jump_count as usize
        {
            diagnostics.push(
                AuditDiagnostic::new(
                    AuditSeverity::Warning,
                    "pose quality",
                    format!(
                        "{} augmented trajectory still has {} high-velocity jump(s); worst displacement {:.4}",
                        metric.label,
                        jump_count,
                        metric.worst_augmented_displacement.unwrap_or(0.0)
                    ),
                )
                .experiment(&experiment.name)
                .artifact_path(&augmented_path)
                .cause("augmented output still exceeds the configured velocity gate")
                .fix_hint("tune max_velocity, enable/check skeleton repair, or inspect remaining teleports")
                .validation_command(validation_command),
            );
        }
    }

    if augmented_df.is_none() {
        diagnostics.push(
            AuditDiagnostic::new(
                AuditSeverity::Warning,
                "pose quality",
                "augmented.parquet is missing; post-repair metrics are unavailable",
            )
            .experiment(&experiment.name)
            .artifact_path(&augmented_path)
            .cause("interpolation burden, augmented null rate, and residual jump metrics require augmentation output")
            .fix_hint("run the augment stage for this experiment")
            .validation_command(validation_command),
        );
    }

    diagnostics.extend(skeleton_diagnostics(
        experiment,
        augmented_df.as_ref(),
        &augmented_path,
        validation_command,
    )?);
    diagnostics.extend(ymaze_alignment_diagnostics(
        experiment,
        augmented_df.as_ref(),
        exp_dir,
        validation_command,
    )?);

    Ok(diagnostics)
}

pub(super) fn compute_pose_quality_metrics(
    raw_df: &DataFrame,
    augmented_df: Option<&DataFrame>,
    labels: &[String],
    likelihood_threshold: f64,
    augmented_jump_threshold: f64,
) -> Result<Vec<LabelPoseQuality>> {
    labels
        .iter()
        .map(|label| {
            compute_label_pose_quality(
                raw_df,
                augmented_df,
                label,
                likelihood_threshold,
                augmented_jump_threshold,
            )
        })
        .collect()
}

pub(super) fn compute_label_pose_quality(
    raw_df: &DataFrame,
    augmented_df: Option<&DataFrame>,
    label: &str,
    likelihood_threshold: f64,
    augmented_jump_threshold: f64,
) -> Result<LabelPoseQuality> {
    let total_frames = raw_df.height();
    let raw_invalid = invalid_coordinate_mask(raw_df, label)?;
    let low_likelihood = likelihood_invalid_mask(raw_df, label, likelihood_threshold)?;
    let default_likelihood = vec![false; total_frames];
    let likelihood_mask = low_likelihood.as_ref().unwrap_or(&default_likelihood);
    let combined_invalid: Vec<bool> = raw_invalid
        .iter()
        .zip(likelihood_mask)
        .map(|(coord_invalid, likelihood_invalid)| *coord_invalid || *likelihood_invalid)
        .collect();
    let gap_stats = gap_stats(&combined_invalid);

    let augmented_invalid = augmented_df
        .map(|df| invalid_coordinate_mask(df, label))
        .transpose()?;
    let augmented_missing_count = augmented_invalid
        .as_ref()
        .map(|mask| mask.iter().filter(|value| **value).count());
    let interpolation_burden_count = if let (Some(aug_df), Some(aug_mask)) =
        (augmented_df, augmented_invalid.as_ref())
    {
        let comparable = total_frames.min(aug_df.height()).min(aug_mask.len());
        Some(
            combined_invalid
                .iter()
                .take(comparable)
                .zip(aug_mask.iter())
                .filter(|(raw_invalid, augmented_invalid)| **raw_invalid && !**augmented_invalid)
                .count(),
        )
    } else {
        None
    };
    let jumps = augmented_df
        .map(|df| jump_stats(df, label, augmented_jump_threshold))
        .transpose()?;

    Ok(LabelPoseQuality {
        label: label.to_string(),
        total_frames,
        low_likelihood_count: low_likelihood
            .as_ref()
            .map(|mask| mask.iter().filter(|value| **value).count()),
        raw_missing_count: raw_invalid.iter().filter(|value| **value).count(),
        augmented_missing_count,
        longest_gap_frames: gap_stats.longest,
        short_gap_count: gap_stats.short,
        medium_gap_count: gap_stats.medium,
        long_gap_count: gap_stats.long,
        interpolation_burden_count,
        augmented_jump_count: jumps.map(|stats| stats.count),
        worst_augmented_displacement: jumps.map(|stats| stats.worst_displacement),
    })
}

pub(super) fn invalid_coordinate_mask(df: &DataFrame, label: &str) -> Result<Vec<bool>> {
    let x_name = format!("{label}_x");
    let y_name = format!("{label}_y");
    let x_nulls = df
        .column(&x_name)
        .map_err(|_| BekiperError::MissingColumn(x_name.clone()))?
        .is_null();
    let y_nulls = df
        .column(&y_name)
        .map_err(|_| BekiperError::MissingColumn(y_name.clone()))?
        .is_null();

    Ok(x_nulls
        .into_iter()
        .zip(&y_nulls)
        .map(|(x, y)| x.unwrap_or(true) || y.unwrap_or(true))
        .collect())
}

pub(super) fn likelihood_invalid_mask(
    df: &DataFrame,
    label: &str,
    threshold: f64,
) -> Result<Option<Vec<bool>>> {
    let name = format!("{label}_likelihood");
    let Ok(series) = df.column(&name) else {
        return Ok(None);
    };
    let values = series
        .f64()
        .map_err(|_| BekiperError::Data(format!("{name} must be Float64")))?;
    Ok(Some(
        values
            .into_iter()
            .map(|value| value.is_none_or(|v| v < threshold))
            .collect(),
    ))
}

pub(super) fn gap_stats(mask: &[bool]) -> GapStats {
    let mut longest = 0usize;
    let mut short = 0usize;
    let mut medium = 0usize;
    let mut long = 0usize;
    let mut current = 0usize;

    for value in mask.iter().copied().chain(std::iter::once(false)) {
        if value {
            current += 1;
            continue;
        }
        if current == 0 {
            continue;
        }
        longest = longest.max(current);
        match current {
            1..=5 => short += 1,
            6..=15 => medium += 1,
            _ => long += 1,
        }
        current = 0;
    }

    GapStats {
        longest,
        short,
        medium,
        long,
    }
}

pub(super) fn jump_stats(df: &DataFrame, label: &str, threshold: f64) -> Result<JumpStats> {
    let x_name = format!("{label}_x");
    let y_name = format!("{label}_y");
    let x = df
        .column(&x_name)
        .map_err(|_| BekiperError::MissingColumn(x_name.clone()))?
        .f64()
        .map_err(|_| BekiperError::Data(format!("{x_name} must be Float64")))?;
    let y = df
        .column(&y_name)
        .map_err(|_| BekiperError::MissingColumn(y_name.clone()))?
        .f64()
        .map_err(|_| BekiperError::Data(format!("{y_name} must be Float64")))?;

    let mut count = 0usize;
    let mut worst_displacement = 0.0f64;
    let mut previous: Option<(f64, f64)> = None;
    for (x, y) in x.into_iter().zip(y) {
        let current = x.zip(y);
        if let (Some((px, py)), Some((cx, cy))) = (previous, current) {
            let displacement = ((cx - px).powi(2) + (cy - py).powi(2)).sqrt();
            if displacement > threshold {
                count += 1;
                worst_displacement = worst_displacement.max(displacement);
            }
        }
        previous = current;
    }

    Ok(JumpStats {
        count,
        worst_displacement,
    })
}

pub(super) fn summary_diagnostic(
    experiment: &ExperimentConfig,
    poses_path: &Path,
    augmented_path: &Path,
    metrics: &[LabelPoseQuality],
    validation_command: &str,
) -> AuditDiagnostic {
    let label_count = metrics.len();
    let frame_count = metrics.first().map(|m| m.total_frames).unwrap_or(0);
    let worst_low = metrics
        .iter()
        .filter_map(LabelPoseQuality::low_likelihood_fraction)
        .fold(0.0, f64::max);
    let worst_raw_missing = metrics
        .iter()
        .map(LabelPoseQuality::raw_missing_fraction)
        .fold(0.0, f64::max);
    let worst_aug_missing = metrics
        .iter()
        .filter_map(LabelPoseQuality::augmented_missing_fraction)
        .fold(0.0, f64::max);
    let worst_interp = metrics
        .iter()
        .filter_map(LabelPoseQuality::interpolation_burden_fraction)
        .fold(0.0, f64::max);
    let jump_count: usize = metrics
        .iter()
        .filter_map(|metric| metric.augmented_jump_count)
        .sum();

    AuditDiagnostic::new(
        AuditSeverity::Ok,
        "pose quality",
        format!(
            "inspected {label_count} label(s), {frame_count} frame(s); worst low-likelihood {:.1}%, raw missing {:.1}%, augmented missing {:.1}%, interpolation burden {:.1}%, residual jumps {jump_count}",
            worst_low * 100.0,
            worst_raw_missing * 100.0,
            worst_aug_missing * 100.0,
            worst_interp * 100.0
        ),
    )
    .experiment(&experiment.name)
    .artifact_path(if augmented_path.exists() {
        augmented_path
    } else {
        poses_path
    })
    .validation_command(validation_command)
}

pub(super) fn skeleton_diagnostics(
    experiment: &ExperimentConfig,
    augmented_df: Option<&DataFrame>,
    augmented_path: &Path,
    validation_command: &str,
) -> Result<Vec<AuditDiagnostic>> {
    let Some(df) = augmented_df else {
        return Ok(Vec::new());
    };
    let height = df.height();
    if height == 0 {
        return Ok(Vec::new());
    }

    let mut diagnostics = Vec::new();
    if let Ok(series) = df.column("skeleton_violation_count") {
        let counts = series
            .u32()
            .map_err(|_| BekiperError::Data("skeleton_violation_count must be UInt32".into()))?;
        let violated = counts
            .into_iter()
            .filter(|value| value.is_some_and(|count| count > 0))
            .count();
        if violated > 0 {
            diagnostics.push(
                AuditDiagnostic::new(
                    AuditSeverity::Warning,
                    "pose quality",
                    format!(
                        "skeleton repair flagged {:.1}% of augmented frames ({violated}/{height})",
                        fraction(violated, height) * 100.0
                    ),
                )
                .experiment(&experiment.name)
                .artifact_path(augmented_path)
                .cause("topology checks found anatomically implausible keypoints")
                .fix_hint("inspect label/model preset compatibility and recurring failed labels")
                .validation_command(validation_command),
            );
        }
    }

    if let Ok(series) = df.column("skeleton_redetected") {
        let redetected = series
            .bool()
            .map_err(|_| BekiperError::Data("skeleton_redetected must be Boolean".into()))?
            .into_iter()
            .filter(|value| value.unwrap_or(false))
            .count();
        if redetected > 0 {
            diagnostics.push(
                AuditDiagnostic::new(
                    AuditSeverity::Ok,
                    "pose quality",
                    format!(
                        "skeleton redetection repaired {:.1}% of augmented frames ({redetected}/{height})",
                        fraction(redetected, height) * 100.0
                    ),
                )
                .experiment(&experiment.name)
                .artifact_path(augmented_path)
                .validation_command(validation_command),
            );
        }
    }

    Ok(diagnostics)
}

pub(super) fn load_pose_quality_scale(exp_dir: &ExperimentOutputDir) -> Option<f64> {
    let content = std::fs::read_to_string(exp_dir.scale_toml()).ok()?;
    let info: bekiper_core::scale::ScaleDerivationInfo = toml::from_str(&content).ok()?;
    Some(info.meters_per_pixel)
}

pub(super) fn fraction(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    }
}
