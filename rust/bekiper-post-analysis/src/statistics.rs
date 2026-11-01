//! Registered Gaussian OLS with full factorial treatment coding. Sessions,
//! frames and sensitivity thresholds never become independent animal replicates.
//! Unsupported inference is explicit; rank failures never simplify a model.
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};
use nalgebra::{DMatrix, DVector};
use serde_json::{Value, json};
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::function::beta::ln_beta;

use crate::contracts::{FACTORS, Record};

pub const FACTOR_FORMULA: &str = "5XFAD_status * HCAR1_status * treatment";
pub const WEIGHTS: [(&str, f64); 4] = [("G5", 1.0), ("G6", -1.0), ("G1", -1.0), ("G2", 1.0)];

fn outcome(result: &mut Record, status: &str, reason: &str) {
    result["status"] = status.into();
    result["reason"] = reason.into();
}

fn matrix_json(matrix: &DMatrix<f64>) -> Value {
    json!(
        (0..matrix.nrows())
            .map(|row| (0..matrix.ncols())
                .map(|column| matrix[(row, column)])
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    )
}

/// OLS coefficients and covariance are derived from a column-scaled SVD,
/// avoiding normal-equation inversion. The full interaction design is retained.
pub fn fit_factorial(
    rows: &[Record],
    endpoint: &str,
    levels: &Record,
    groups: &Record,
    settings: &Record,
    baseline: bool,
    requirement: Option<&str>,
) -> Result<Record> {
    let additional: Vec<String> = match &settings["additional_covariates"] {
        Value::Null => Vec::new(),
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .context("Additional covariates require explicit field names")
            })
            .collect::<Result<_>>()?,
        _ => bail!("Additional covariates require a list"),
    };
    let mut covariates = if baseline {
        vec!["pre_outcome".into()]
    } else {
        Vec::new()
    };
    covariates.extend(additional.iter().cloned());
    ensure!(
        !covariates.iter().any(|name| name == endpoint),
        "Outcome cannot also be a covariate"
    );
    let mut formula = format!(
        "{endpoint} ~ {}{FACTOR_FORMULA}",
        if baseline { "pre_outcome + " } else { "" }
    );
    if !additional.is_empty() {
        formula.push_str(&format!(" + {}", additional.join(" + ")));
    }
    let mut result = json!({"endpoint":endpoint,"formula":formula,"status":"blocked","reason":null,
        "n_animals":0,"settings":settings,"coefficients":[],"contrast":null,"subset":[],
        "numerical_method":"column_scaled_svd"});
    for (unsupported, reason) in [
        (
            settings["model_family"] != "gaussian_ols"
                || settings["covariance_estimator"] != "classical",
            "unsupported_registered_model_or_covariance",
        ),
        (
            settings["factor_coding"] != "treatment" || settings["test_tails"] != "two-sided",
            "unsupported_registered_coding_or_tail",
        ),
        (
            settings["confidence_interval_method"] != "student_t",
            "unsupported_registered_ci_method",
        ),
        (
            settings["contrast_estimand_and_weighting"]["weighting"] != "fixed_reference",
            "unsupported_registered_contrast_weighting",
        ),
    ] {
        if unsupported {
            outcome(&mut result, "blocked", reason);
            return Ok(result);
        }
    }
    let confidence = number(settings, "confidence_level")?;
    ensure!(
        confidence > 0.0 && confidence < 1.0,
        "Register a valid confidence level"
    );
    validate_contrast_mapping(groups, &settings["contrast_roles"])?;
    let fields: Vec<_> = std::iter::once(endpoint)
        .chain(covariates.iter().map(String::as_str))
        .chain(FACTORS)
        .collect();
    let (used, subset) = analysis_subset(rows, &fields, requirement)?;
    result["subset"] = subset.into();
    result["n_animals"] = used.len().into();
    result["factor_cell_counts"] = estimability::cell_counts(&used, levels)?;
    if endpoint == "latency_to_first_center_entry_s" {
        let mut censored = false;
        for row in &used {
            let post = optional_flag(row, "latency_right_censored")?.unwrap_or(false);
            let pre = if baseline {
                optional_flag(row, "pre_latency_right_censored")?.unwrap_or(false)
            } else {
                false
            };
            censored |= post || pre;
        }
        if censored {
            outcome(
                &mut result,
                "blocked",
                "D09:censored_latency_requires_registered_censoring_model",
            );
            return Ok(result);
        }
    }
    if used.is_empty() {
        outcome(&mut result, "unavailable", "insufficient_data");
        return Ok(result);
    }
    let (matrix, names) = design(&used, levels, &covariates)?;
    let n = matrix.nrows();
    let p = matrix.ncols();
    result["factor_levels"] = levels.clone();
    result["coefficient_names"] = json!(names);
    result["columns"] = p.into();
    let y = DVector::from_vec(
        used.iter()
            .map(|row| number(row, endpoint))
            .collect::<Result<_>>()?,
    );
    let scales: Vec<f64> = (0..p)
        .map(|column| {
            matrix
                .column(column)
                .iter()
                .fold(0.0_f64, |norm, value| norm.hypot(*value))
        })
        .map(|scale| if scale == 0.0 { 1.0 } else { scale })
        .collect();
    let mut normalized = matrix.clone();
    for (column, scale) in scales.iter().enumerate() {
        normalized
            .column_mut(column)
            .scale_mut(1.0 / scale.max(f64::MIN_POSITIVE));
    }
    let Some(svd) = nalgebra::linalg::SVD::try_new(normalized, true, true, f64::EPSILON, 100_000)
    else {
        outcome(&mut result, "error", "svd_nonconvergence"); // coverage: unreachable — finite, column-normalized design converges within 100000 sweeps
        return Ok(result); // coverage: unreachable — SVD nonconvergence branch above
    };
    let largest = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = f64::EPSILON * n.max(p) as f64 * largest;
    let rank = svd
        .singular_values
        .iter()
        .filter(|&&value| value > tolerance)
        .count();
    result["rank"] = rank.into();
    result["rank_tolerance"] = tolerance.into();
    result["singular_values_scaled"] = json!(svd.singular_values.as_slice());
    let retained: Vec<_> = svd
        .singular_values
        .iter()
        .enumerate()
        .filter(|(_, s)| **s > tolerance)
        .map(|(k, _)| k)
        .collect();
    if n <= rank {
        outcome(
            &mut result,
            "error",
            "insufficient_residual_degrees_of_freedom",
        );
        return Ok(result);
    }
    let df = n - rank;
    let mut scaled_beta = svd
        .solve(&y, tolerance)
        .map_err(|message| anyhow::anyhow!(message))?;
    // Residual correction in the same retained SVD subspace improves solve accuracy
    // for repeated singular values without altering the retained design.
    for _ in 0..2 {
        let residual = &y
            - &matrix
                * DVector::from_iterator(p, (0..p).map(|index| scaled_beta[index] / scales[index]));
        scaled_beta += svd
            .solve(&residual, tolerance)
            .map_err(|message| anyhow::anyhow!(message))?;
    }
    let beta = DVector::from_iterator(p, (0..p).map(|index| scaled_beta[index] / scales[index]));
    let fitted = &matrix * &beta;
    let residuals = &y - &fitted;
    let residual_norm = residuals
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value));
    let y_norm = y.iter().fold(0.0_f64, |norm, value| norm.hypot(*value));
    let variance = (residual_norm / (df as f64).sqrt()).powi(2);
    if beta
        .iter()
        .chain(fitted.iter())
        .chain(residuals.iter())
        .any(|value| !value.is_finite())
        || !variance.is_finite()
    {
        outcome(&mut result, "error", "numerical_nonfinite_model_result");
        return Ok(result);
    }
    result["residual_df"] = df.into();
    result["residual_variance"] = variance.into();
    // Relative backward-error check: changing response units must not turn
    // genuine residual variation into a numerically exact fit.
    if y_norm == 0.0 || residual_norm / y_norm <= f64::EPSILON * n.max(p) as f64 {
        outcome(&mut result, "error", "numerically_zero_residual_variation");
        return Ok(result);
    }
    if variance == 0.0 {
        outcome(&mut result, "error", "residual_variance_underflow");
        return Ok(result);
    }
    let v_transpose = svd
        .v_t
        .as_ref()
        .context("SVD omitted V despite requested covariance")?;
    let u = svd
        .u
        .as_ref()
        .context("SVD omitted U despite requested diagnostics")?;
    let mut covariance = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            covariance[(i, j)] = retained
                .iter()
                .map(|&k| {
                    v_transpose[(k, i)] * v_transpose[(k, j)] / svd.singular_values[k].powi(2)
                })
                .sum::<f64>()
                * variance
                / scales[i]
                / scales[j];
        }
    }
    if covariance.iter().any(|value| !value.is_finite()) {
        outcome(&mut result, "error", "numerical_nonfinite_covariance");
        return Ok(result);
    }
    let coefficients = names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let mut vector = DVector::zeros(p);
            vector[index] = 1.0;
            if !estimability::estimable(&vector,&scales,v_transpose,&retained,n) {
                return Ok(json!({"term":name,"status":"not_estimable","reason":"coefficient_outside_design_row_space",
                    "estimable":false,"estimate":null,"standard_error":null,"ci_lower":null,"ci_upper":null,"p_raw":null}));
            }
            let mut coefficient = t_summary(
                beta[index],
                covariance[(index, index)].max(0.0).sqrt(),
                df,
                confidence,
            )?; // coverage: unreachable — beta, covariance, df and confidence are validated finite above
            coefficient["term"] = name.clone().into();
            coefficient["status"] = "computed".into();
            coefficient["estimable"] = true.into();
            Ok(coefficient)
        })
        .collect::<Result<Vec<_>>>()?;
    result["coefficients"] = coefficients.into();
    result["covariance_matrix"] = matrix_json(&covariance);
    result["covariance_matrix_definition"] = if rank < p {
        "generalized_solution_covariance_for_estimable_functions_only"
    } else {
        "identified_coefficients"
    }
    .into();
    result["diagnostics"] = json!((0..n).map(|index| json!({"animal_id":used[index]["animal_id"],
        "session_id":used[index]["session_id"],"observed":y[index],"fitted":fitted[index],"residual":residuals[index],
        "leverage":retained.iter().map(|&column| u[(index,column)].powi(2)).sum::<f64>()})).collect::<Vec<_>>());
    result["normality_diagnostic"] = json!({"method":"Shapiro-Wilk residuals","status":"unavailable",
        "reason":"not_implemented_in_native_backend","statistic":null,"p_raw":null});
    let mut references = json!({});
    let configured_references = &settings["contrast_estimand_and_weighting"]["reference_values"];
    for covariate in &covariates {
        let reference = &configured_references[covariate];
        if reference.is_null() {
            outcome(
                &mut result,
                "blocked",
                "blocked:D09:contrast_covariate_reference_missing",
            );
            return Ok(result);
        }
        if levels.get(covariate).is_some() {
            if !categorical_levels(levels, covariate)?
                .iter()
                .any(|level| reference == level)
            {
                outcome(
                    &mut result,
                    "blocked",
                    "blocked:D09:categorical_covariate_reference",
                );
                return Ok(result);
            }
            references[covariate] = reference.clone();
        } else {
            let value = if reference == "mean" {
                used.iter().try_fold(0.0, |mean, row| {
                    Ok::<_, anyhow::Error>(mean + number(row, covariate)? / n as f64)
                })?
            } else {
                number(configured_references, covariate)?
            };
            ensure!(value.is_finite(), "Contrast reference overflow");
            references[covariate] = value.into();
        }
    }
    let targets: Vec<_> = WEIGHTS
        .iter()
        .map(|(group, _)| {
            let mut target = groups[*group].clone();
            for covariate in &covariates {
                target[covariate] = references[covariate].clone();
            }
            target
        })
        .collect();
    let (target_matrix, _) = design(&targets, levels, &covariates)?;
    let vector = DVector::from_iterator(
        p,
        (0..p).map(|column| {
            WEIGHTS
                .iter()
                .enumerate()
                .map(|(index, (_, weight))| weight * target_matrix[(index, column)])
                .sum()
        }),
    );
    let summarize = |vector: &DVector<f64>| -> Result<Record> {
        if !estimability::estimable(vector, &scales, v_transpose, &retained, n) {
            return Ok(
                json!({"status":"not_estimable","estimable":false,"reason":"linear_function_outside_design_row_space",
                "estimate":null,"standard_error":null,"ci_lower":null,"ci_upper":null,"p_raw":null,"df":df}),
            );
        }
        let contrast_variance = retained
            .iter()
            .map(|&component| {
                let coordinate = (0..p)
                    .map(|column| {
                        v_transpose[(component, column)] * vector[column] / scales[column]
                    })
                    .sum::<f64>();
                (coordinate / svd.singular_values[component]).powi(2)
            })
            .sum::<f64>()
            * variance;
        let mut summary = t_summary(
            vector.dot(&beta),
            contrast_variance.max(0.0).sqrt(),
            df,
            confidence,
        )?;
        summary["status"] = "computed".into();
        summary["estimable"] = true.into();
        Ok(summary)
    };
    let mut contrast = summarize(&vector)?;
    let estimates = &target_matrix * &beta;
    ensure!(
        estimates.iter().all(|value| value.is_finite()),
        "Contrast group estimate overflow"
    );
    contrast["name"] = "(G5-G6)-(G1-G2)".into();
    contrast["weights"] = json!({"G5":1,"G6":-1,"G1":-1,"G2":1});
    contrast["reference_values"] = references;
    contrast["coefficient_vector"] = json!(vector.as_slice());
    let mut group_estimates = json!({});
    for (index, (group, _)) in WEIGHTS.iter().enumerate() {
        let target = target_matrix.row(index).transpose();
        group_estimates[*group] =
            if estimability::estimable(&target, &scales, v_transpose, &retained, n) {
                json!(estimates[index])
            } else {
                Value::Null
            };
    }
    contrast["group_estimates"] = group_estimates;
    result["contrast"] = contrast;
    let cells = result["factor_cell_counts"]
        .as_array()
        .context("Factor cells")?;
    let targets: Vec<_> = cells
        .iter()
        .map(|cell| {
            let mut target = cell.clone();
            // Numeric 'mean' references have already been resolved for the contrast.
            for covariate in &covariates {
                target[covariate] = result["contrast"]["reference_values"][covariate].clone();
            }
            target
        })
        .collect();
    let (cell_matrix, _) = design(&targets, levels, &covariates)?;
    let cell_estimates: Vec<_> = cells
        .iter()
        .enumerate()
        .map(|(i, cell)| {
            let mut summary = summarize(&cell_matrix.row(i).transpose())?;
            for factor in FACTORS {
                summary[factor] = cell[factor].clone();
            }
            summary["n_animals"] = cell["n_animals"].clone();
            Ok(summary)
        })
        .collect::<Result<_>>()?;
    let mut lactate_effects = Vec::new();
    for genotype in categorical_levels(levels, "hcar1")? {
        let mut indexes = Vec::new();
        for treatment in ["lactate", "vehicle"] {
            let i = cells
                .iter()
                .position(|c| {
                    c["five_xfad"] == settings["contrast_roles"]["ad_level"]
                        && c["hcar1"] == genotype
                        && c["treatment"] == settings["contrast_roles"][treatment]
                })
                .context("Genotype contrast target")?;
            indexes.push(i);
        }
        let vector = (cell_matrix.row(indexes[0]) - cell_matrix.row(indexes[1])).transpose();
        let mut summary = summarize(&vector)?;
        summary["hcar1"] = genotype.into();
        summary["name"] = "AD background: lactate - vehicle".into();
        summary["inference_role"] = "supplementary_genotype_simple_effect".into();
        lactate_effects.push(summary);
    }
    result["factor_cell_estimates"] = json!(cell_estimates);
    result["genotype_lactate_effects"] = json!(lactate_effects);
    if rank < p {
        outcome(&mut result, "error", "rank_deficient:interactions_retained");
    } else {
        result["status"] = "fitted".into();
        result["reason"] = Value::Null;
    }
    Ok(result)
}

mod eligibility;
mod estimability;
pub use eligibility::{analysis_subset, verification_reason};
use eligibility::{
    excluded, identity, number, optional_flag, validate_requirement, verification_status,
};

mod pairing;
pub use pairing::pairing;

mod design;
use design::categorical_levels;
pub use design::{design, validate_contrast_mapping};

mod probability;
use probability::t_probability;
pub use probability::t_summary;

mod correlation;
pub use correlation::spearman_pairs;

mod checks;
pub use checks::within_group;

mod adjustment;
pub use adjustment::adjust_family;
