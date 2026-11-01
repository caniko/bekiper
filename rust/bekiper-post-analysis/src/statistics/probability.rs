//! Student-t tail and log-tail probabilities.
use super::*;

/// Stable log of the two-sided Student-t tail. The incomplete-beta continued
/// fraction is evaluated in log space only when the ordinary SF underflows.
pub(super) fn log_t_probability(statistic: f64, df: usize) -> Result<f64> {
    let absolute = statistic.abs();
    if absolute == 0.0 {
        return Ok(0.0);
    }
    ensure!(
        absolute.is_finite() && df > 0,
        "Finite statistic and positive degrees of freedom required"
    );
    let freedom = df as f64;
    let log_ratio = 2.0 * absolute.ln() - freedom.ln();
    let log_x = if log_ratio > 0.0 {
        -log_ratio - (-log_ratio).exp().ln_1p()
    } else {
        -log_ratio.exp().ln_1p()
    };
    let x = log_x.exp();
    let a = freedom / 2.0;
    let b = 0.5;
    if x == 0.0 {
        return Ok(a * log_x - a.ln() - ln_beta(a, b));
    }
    ensure!(
        x < (a + 1.0) / (a + b + 2.0),
        "Underflow tail outside lower incomplete-beta branch"
    );
    let floor = f64::MIN_POSITIVE / f64::EPSILON;
    let nonzero = |value: f64| {
        if value.abs() < floor {
            floor.copysign(value) // coverage: unreachable — Lentz guard; terms stay bounded away from zero below the branch limit checked above
        } else {
            value
        }
    };
    let mut c = 1.0;
    let mut d = 1.0 / nonzero(1.0 - (a + b) * x / (a + 1.0));
    let mut fraction = d;
    let mut converged = false;
    for iteration in 1..=1000 {
        let m = iteration as f64;
        let twice = 2.0 * m;
        let even = m * (b - m) * x / ((a - 1.0 + twice) * (a + twice));
        d = 1.0 / nonzero(1.0 + even * d);
        c = nonzero(1.0 + even / c);
        fraction *= d * c;
        let odd = -(a + m) * (a + b + m) * x / ((a + twice) * (a + 1.0 + twice));
        d = 1.0 / nonzero(1.0 + odd * d);
        c = nonzero(1.0 + odd / c);
        let change = d * c;
        fraction *= change;
        if (change - 1.0).abs() <= 4.0 * f64::EPSILON {
            converged = true;
            break;
        }
    }
    ensure!(
        converged && fraction > 0.0 && fraction.is_finite(),
        "Student-t log tail did not converge"
    );
    let result = a * log_x + b * (-x).ln_1p() - ln_beta(a, b) + fraction.ln() - a.ln();
    ensure!(
        result.is_finite() && result <= 0.0,
        "Invalid Student-t log probability"
    );
    Ok(result)
}

pub(super) fn t_probability(statistic: f64, df: usize) -> Result<(Option<f64>, Option<f64>)> {
    let distribution = StudentsT::new(0.0, 1.0, df as f64)?;
    let probability = 2.0 * distribution.sf(statistic.abs());
    if probability.is_finite() && probability > 0.0 {
        let probability = probability.min(1.0);
        Ok((Some(probability), Some(probability.ln())))
    } else {
        let log_probability = log_t_probability(statistic, df)?;
        // The SF implementation can lose a representable tail when t*t
        // overflows. Recover it from the log-domain calculation when possible.
        let probability = log_probability.exp();
        Ok((
            (probability > 0.0).then_some(probability),
            Some(log_probability),
        ))
    }
}

pub fn t_summary(estimate: f64, standard_error: f64, df: usize, confidence: f64) -> Result<Record> {
    ensure!(
        estimate.is_finite() && standard_error.is_finite() && standard_error >= 0.0,
        "Student-t summary requires finite estimate and nonnegative standard error"
    );
    ensure!(
        confidence.is_finite() && confidence > 0.0 && confidence < 1.0,
        "Register a valid confidence level"
    );
    ensure!(
        df > 0,
        "Student-t inference requires positive degrees of freedom"
    );
    let mut result = json!({"estimate":estimate,"standard_error":standard_error,"statistic":null,"df":df,
        "p_raw":null,"log_p_raw":null,"ci_lower":null,"ci_upper":null,"reason":null});
    if standard_error == 0.0 {
        result["reason"] = "undefined:zero_standard_error".into();
        return Ok(result);
    }
    let statistic = estimate / standard_error;
    if !statistic.is_finite() {
        result["reason"] = "undefined:statistic_overflow".into();
        return Ok(result);
    }
    let (probability, log_probability) = t_probability(statistic, df)?;
    let distribution = StudentsT::new(0.0, 1.0, df as f64)?;
    // Use the lower tail to avoid rounding (1 + confidence) / 2 to one.
    let radius = -distribution.inverse_cdf((1.0 - confidence) / 2.0) * standard_error;
    result["statistic"] = statistic.into();
    result["p_raw"] = json!(probability);
    result["log_p_raw"] = json!(log_probability);
    if (estimate - radius).is_finite() && (estimate + radius).is_finite() {
        result["ci_lower"] = (estimate - radius).into();
        result["ci_upper"] = (estimate + radius).into();
    } else {
        result["reason"] = "undefined:confidence_interval_overflow".into();
    }
    if probability.is_none() {
        result["reason"] = "p_underflow:see_log_p_raw".into();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_tail_is_exact_at_zero_and_matches_the_ordinary_tail_near_the_centre() {
        assert_eq!(log_t_probability(0.0, 3).unwrap(), 0.0);
        // |t| <= sqrt(df) takes the direct log(1 + t^2/df) branch.
        let (probability, _) = t_probability(5.0, 100).unwrap();
        let log = log_t_probability(5.0, 100).unwrap();
        assert!(
            (log / probability.unwrap().ln() - 1.0).abs() < 1e-10,
            "{log}"
        );
        // Central tails are outside the continued fraction's convergent branch.
        assert!(log_t_probability(1.0, 4).is_err());
        assert!(log_t_probability(f64::INFINITY, 3).is_err());
        assert!(log_t_probability(1.0, 0).is_err());
    }
}
