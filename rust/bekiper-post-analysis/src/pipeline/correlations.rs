//! Pairwise Y-maze descriptive checks.
use super::*;

pub(super) fn ymaze_correlations(
    subset: &[Record],
    config: &Record,
    requirement: Option<&str>,
) -> Result<Vec<Record>> {
    let mut requirements = INCLUSION_REQUIREMENTS.to_vec();
    requirements.extend([
        (
            "statistics.spearman_missing_and_constant_input_policy",
            "D09",
            Some("pairwise_complete_constant_missing"),
        ),
        ("statistics.test_tails", "D09", Some("two-sided")),
    ]);
    let mut failed = failed_choices(config, &requirements);
    if choice(
        config,
        "assays.ymaze.primary_threshold_arm_length_pct",
        "D03",
    )
    .is_none()
    {
        failed.push("D03:primary_threshold".into());
    }
    if failed.is_empty() {
        let post: Vec<_> = subset
            .iter()
            .filter(|row| row["timepoint"] == "post")
            .cloned()
            .collect();
        statistics::spearman_pairs(&post, requirement)
    } else {
        Ok([
                    "valid_total_arm_entries",
                    "total_distance_m",
                    "mean_speed_m_per_s",
                    "neutral_triads_pct",
                ].into_iter().map(|other| json!({"x":crate::contracts::ymaze_score_fields()[0],"y":other,"method":"Spearman","status":"blocked","reason":failed,"n_animals":0,"p_raw":null})).collect())
    }
}
