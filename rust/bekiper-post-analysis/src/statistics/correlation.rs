//! Tied-rank Spearman checks with pairwise populations.
use super::*;

pub(super) fn ranks(values: &[f64]) -> Vec<f64> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut result = vec![0.0; values.len()];
    let mut start = 0;
    while start < indices.len() {
        let mut end = start + 1;
        while end < indices.len() && values[indices[end]] == values[indices[start]] {
            end += 1;
        }
        let average = (start + 1 + end) as f64 / 2.0;
        for &index in &indices[start..end] {
            result[index] = average;
        }
        start = end;
    }
    result
}

pub fn spearman_pairs(rows: &[Record], requirement: Option<&str>) -> Result<Vec<Record>> {
    let mut results = Vec::new();
    for other in [
        "valid_total_arm_entries",
        "total_distance_m",
        "mean_speed_m_per_s",
        "neutral_triads_pct",
    ] {
        let (used, subset) =
            analysis_subset(rows, &["spontaneous_alternation_pct", other], requirement)?;
        let mut result = json!({"x":"spontaneous_alternation_pct","y":other,"method":"Spearman",
            "p_method":"student_t_asymptotic","n_animals":used.len(),"rho":null,"p_raw":null,"log_p_raw":null,
            "reason":null,"subset":subset,"status":"unavailable"});
        let x = used
            .iter()
            .map(|row| number(row, "spontaneous_alternation_pct"))
            .collect::<Result<Vec<_>>>()?;
        let y = used
            .iter()
            .map(|row| number(row, other))
            .collect::<Result<Vec<_>>>()?;
        if !x.is_empty()
            && (x.iter().all(|value| *value == x[0]) || y.iter().all(|value| *value == y[0]))
        {
            result["reason"] = "constant_input".into();
        } else if used.len() < 3 {
            result["reason"] = "insufficient_pairs".into();
        } else {
            let rank_x = ranks(&x);
            let rank_y = ranks(&y);
            let mean = (used.len() + 1) as f64 / 2.0;
            let cross: f64 = rank_x
                .iter()
                .zip(&rank_y)
                .map(|(x, y)| (x - mean) * (y - mean))
                .sum();
            let sum_x: f64 = rank_x.iter().map(|value| (value - mean).powi(2)).sum();
            let sum_y: f64 = rank_y.iter().map(|value| (value - mean).powi(2)).sum();
            let rho = (cross / (sum_x * sum_y).sqrt()).clamp(-1.0, 1.0);
            result["rho"] = rho.into();
            result["df"] = (used.len() - 2).into();
            if rho.abs() == 1.0 {
                result["reason"] = "numerical_p_zero:not_reported_as_literal_zero".into();
            } else {
                let statistic = rho * (((used.len() - 2) as f64) / (1.0 - rho * rho)).sqrt();
                let (probability, log_probability) = t_probability(statistic, used.len() - 2)?;
                result["p_raw"] = json!(probability);
                result["log_p_raw"] = json!(log_probability);
                if probability.is_some() {
                    result["status"] = "computed".into();
                } else {
                    result["reason"] = "p_underflow:see_log_p_raw".into();
                }
            }
        }
        results.push(result);
    }
    Ok(results)
}
