//! Native descriptive means and registered uncertainty.
use super::*;

pub(super) fn summary(
    category: &str,
    points: &[&Record],
    policy: &PlotPolicy<'_>,
) -> Result<Record> {
    let censored = points
        .iter()
        .filter(|point| point["censored"] == true)
        .count();
    if censored > 0 {
        return Ok(
            json!({"category":category,"n_animals":points.iter().map(|p| p["animal_id"].to_string()).collect::<BTreeSet<_>>().len(),"n_observations":points.len(),"n_right_censored":censored,
            "mean":null,"sem":null,"error_half_width":null,"uncertainty":policy.uncertainty,
            "status":"blocked:D09:censored_latency_summary","reason":"censoring_summary_method_unavailable",
            "population_policy":policy.population}),
        );
    }
    let n = points.len();
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (index, point) in points.iter().enumerate() {
        let value = number(point, "value")?.context("summary point value")?;
        let delta = value - mean;
        mean += delta / (index + 1) as f64;
        m2 += delta * (value - mean);
    }
    ensure!(
        mean.is_finite() && m2.is_finite(),
        "Summary arithmetic overflow"
    );
    let sem = (n > 1 && policy.population != "recorded_descriptive")
        .then(|| (m2.max(0.0) / (n - 1) as f64 / n as f64).sqrt());
    let error = match (policy.uncertainty, sem) {
        (Some("SEM"), Some(sem)) => Some(sem),
        (Some("95% CI"), Some(sem)) => {
            Some(sem * StudentsT::new(0.0, 1.0, (n - 1) as f64)?.inverse_cdf(0.975))
        }
        _ => None,
    };
    ensure!(
        error.is_none_or(f64::is_finite),
        "Summary interval overflow"
    );
    let status = if policy.population == "recorded_descriptive" && n > 0 {
        "available"
    } else if policy.uncertainty.is_none() {
        "blocked:D09:summary_uncertainty"
    } else if n < 2 {
        "unavailable"
    } else {
        "available"
    };
    Ok(
        json!({"category":category,"n_animals":points.iter().map(|p| p["animal_id"].to_string()).collect::<BTreeSet<_>>().len(),"n_observations":n,"mean":mean,"sem":sem,"error_half_width":error,
        "uncertainty":policy.uncertainty,"status":status,"reason":if n<2 && policy.population != "recorded_descriptive" {Some("insufficient_animals_for_uncertainty")} else {None},
        "population_policy":policy.population}),
    )
}
