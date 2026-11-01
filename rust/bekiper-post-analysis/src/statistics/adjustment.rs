//! Multiplicity accounting including null attempts.
use super::*;

/// Holm uses every planned attempt in m, including unavailable tests. Log-only
/// probabilities remain usable for ordering/correction without emitting zero.
pub fn adjust_family(tests: &mut [Record], method: &str, family: &str) -> Result<()> {
    ensure!(
        matches!(method, "holm" | "none"),
        "Supported multiplicity methods are registered holm or explicit none"
    );
    ensure!(!family.is_empty(), "Multiplicity family must be named");
    let mut available = Vec::new();
    for (index, test) in tests.iter().enumerate() {
        ensure!(test.is_object(), "Correction attempts must be objects");
        if !test["p_raw"].is_null() {
            let probability = number(test, "p_raw")?;
            ensure!(
                probability > 0.0 && probability <= 1.0,
                "P value must be >0 and <=1; underflow must use null and log_p_raw"
            );
            available.push((index, probability.ln()));
        } else if !test["log_p_raw"].is_null() {
            let probability = number(test, "log_p_raw")?;
            ensure!(probability <= 0.0, "Log P value must be <=0");
            available.push((index, probability));
        }
    }
    available.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let family_size = tests.len();
    for test in tests.iter_mut() {
        test["p_adjusted"] = Value::Null;
        test["log_p_adjusted"] = Value::Null;
        test["correction_method"] = method.into();
        test["correction_family"] = family.into();
        test["family_size"] = family_size.into();
        test.as_object_mut()
            .context("correction attempt object")?
            .remove("adjustment_reason");
    }
    let mut running = f64::NEG_INFINITY;
    for (rank, (index, log_probability)) in available.into_iter().enumerate() {
        let corrected = if method == "holm" {
            running = running.max((log_probability + ((family_size - rank) as f64).ln()).min(0.0));
            running
        } else {
            log_probability
        };
        tests[index]["log_p_adjusted"] = corrected.into();
        let probability = corrected.exp();
        if probability > 0.0 {
            tests[index]["p_adjusted"] = probability.into();
        } else {
            tests[index]["adjustment_reason"] = "p_underflow:see_log_p_adjusted".into();
        }
    }
    Ok(())
}
