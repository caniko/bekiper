"""Registered Gaussian OLS inference; full factorial, no silent model simplification.

Linear algebra uses NumPy; distributions/correlations use the existing SciPy
stack. Unsupported registered families are reported, never substituted.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from scipy import stats

from .inputs import FACTORS
from .metrics import Record

WEIGHTS = {"G5": 1, "G6": -1, "G1": -1, "G2": 1}
FACTOR_FORMULA = "5XFAD_status * HCAR1_status * treatment"


def verification_reason(row: Record, requirement: str | None) -> str | None:
    if requirement is None:
        return "blocked:D12:upstream_verification_requirement"
    if requirement not in ["allow_unverified", "require_verified"]:
        raise ValueError("Unknown upstream verification requirement")
    status = row.get("input_verification", {}).get("status", "unresolved")
    if requirement == "require_verified" and status not in [
        "verified_clarified_entry_definition",
        "verified_source_rule_definition",
    ]:
        return f"upstream_unverified:{status}"
    return None


def analysis_subset(
    rows: list[Record], fields: list[str], requirement: str | None
) -> tuple[list[Record], list[Record]]:
    """One eligible observation per animal, with a reason for every supplied row."""
    used, subset = [], []
    for row in rows:
        reason = "excluded:" + str(row.get("exclusion_reason")) if row.get("excluded") else None
        reason = reason or verification_reason(row, requirement)
        if reason is None and any(row.get(field) is None for field in fields):
            reason = "model_specific_missingness"
        subset.append(
            {
                "animal_id": row["animal_id"],
                "session_id": row["session_id"],
                "reason": reason,
                "input_verification_status": row.get("input_verification", {}).get("status", "unresolved"),
            }
        )
        if reason is None:
            used.append(row)
    if len({r["animal_id"] for r in used}) != len(used):
        raise ValueError("Animal replicates in analysis: sessions/thresholds cannot be independent observations")
    return used, subset


def pairing(
    rows: list[Record], endpoint: str, *, verification_requirement: str | None = "allow_unverified"
) -> tuple[list[Record], list[Record]]:
    grouped: dict[tuple[Any, ...], dict[str, Record]] = {}
    for row in rows:
        key = (row["animal_id"], row["assay"], row["phase"], row.get("threshold_pct"), row.get("pair_id", "primary"))
        timepoint = row["timepoint"]
        if timepoint not in ["pre", "post"]:
            continue
        if timepoint in grouped.setdefault(key, {}):
            raise ValueError("Multiple sessions per animal/timepoint; explicit pair_id or aggregation plan required")
        grouped[key][timepoint] = row
    paired, ledger = [], []
    for key, parts in grouped.items():
        pre, post = parts.get("pre"), parts.get("post")
        reason = None
        if pre is None or post is None:
            reason = "unmatched_prepost"
        elif pre.get("excluded") or post.get("excluded"):
            reason = "excluded:paired_metadata"
        elif verification_reason(pre, verification_requirement):
            reason = verification_reason(pre, verification_requirement)
            if reason and reason.startswith("upstream_unverified:"):
                reason = reason.replace("upstream_unverified:", "upstream_unverified:pre:", 1)
        elif verification_reason(post, verification_requirement):
            reason = verification_reason(post, verification_requirement)
            if reason and reason.startswith("upstream_unverified:"):
                reason = reason.replace("upstream_unverified:", "upstream_unverified:post:", 1)
        elif pre.get(endpoint) is None or post.get(endpoint) is None:
            reason = "missing_pair_endpoint"
        ledger.append(
            {
                "animal_id": key[0],
                "assay": key[1],
                "phase": key[2],
                "threshold_pct": key[3],
                "endpoint": endpoint,
                "pre_session_id": pre["session_id"] if pre else None,
                "post_session_id": post["session_id"] if post else None,
                "reason": reason,
                "pair_id": key[4],
                "verification_requirement": verification_requirement,
                "pre_input_verification_status": pre.get("input_verification", {}).get("status", "unresolved")
                if pre
                else None,
                "post_input_verification_status": post.get("input_verification", {}).get("status", "unresolved")
                if post
                else None,
            }
        )
        if reason is None and pre is not None and post is not None:
            paired.append(
                {
                    **post,
                    "pre_outcome": pre[endpoint],
                    "DeltaSA": post[endpoint] - pre[endpoint],
                    "pre_session_id": pre["session_id"],
                    "pre_latency_right_censored": pre.get("latency_right_censored", False),
                }
            )
    return paired, ledger


def validate_contrast_mapping(groups: Record, roles: Record) -> None:
    expected = {
        "G5": (roles.get("ad_level"), roles.get("hcar1_wt"), roles.get("lactate")),
        "G6": (roles.get("ad_level"), roles.get("hcar1_wt"), roles.get("vehicle")),
        "G1": (roles.get("ad_level"), roles.get("hcar1_ko"), roles.get("lactate")),
        "G2": (roles.get("ad_level"), roles.get("hcar1_ko"), roles.get("vehicle")),
    }
    if not all(roles.get(r) for r in ["ad_level", "hcar1_wt", "hcar1_ko", "lactate", "vehicle"]):
        raise ValueError("Explicit factor semantics for planned contrast required")
    for group, levels in expected.items():
        if group not in groups or tuple(groups[group].get(f) for f in FACTORS) != levels:
            raise ValueError(f"Planned contrast role mismatch for {group}")


def design(rows: list[Record], levels: Record, covariates: list[str]) -> tuple[Any, list[str]]:
    for covariate in covariates:
        if covariate in levels and any(row[covariate] not in levels[covariate] for row in rows):
            raise ValueError("Unknown registered categorical covariate level")
    bases: list[list[tuple[str, str]]] = []
    for factor in FACTORS:
        if not levels.get(factor) or len(levels[factor]) < 2:
            raise ValueError("Register ordered categorical levels/reference for each biological factor")
        if any(row[factor] not in levels[factor] for row in rows):
            raise ValueError("Unknown model factor level")
        bases.append([(factor, level) for level in levels[factor][1:]])
    terms: list[tuple[tuple[str, str], ...]] = []
    for order in range(1, 4):
        for indices in itertools.combinations(range(3), order):
            terms.extend(itertools.product(*(bases[i] for i in indices)))
    covariate_terms = [(c, level) for c in covariates for level in (levels[c][1:] if c in levels else [None])]
    names = (
        ["Intercept"]
        + [":".join(f"{f}[{v}]" for f, v in term) for term in terms]
        + [c if level is None else f"{c}[{level}]" for c, level in covariate_terms]
    )
    values = [
        [1.0]
        + [float(all(row[f] == v for f, v in term)) for term in terms]
        + [float(row[c]) if level is None else float(row[c] == level) for c, level in covariate_terms]
        for row in rows
    ]
    return np.asarray(values, dtype=float), names


def t_summary(estimate: float, se: float, df: int, confidence: float) -> Record:
    if not math.isfinite(se) or se <= 0:
        return {
            "estimate": estimate,
            "standard_error": se,
            "statistic": None,
            "df": df,
            "p_raw": None,
            "ci_lower": None,
            "ci_upper": None,
            "reason": "undefined:zero_standard_error",
        }
    statistic = estimate / se
    log_p = float(math.log(2) + stats.t.logsf(abs(statistic), df))
    p = float(2 * stats.t.sf(abs(statistic), df))
    radius = float(stats.t.ppf((1 + confidence) / 2, df)) * se
    return {
        "estimate": estimate,
        "standard_error": se,
        "statistic": statistic,
        "df": df,
        "p_raw": p if p > 0 else None,
        "log_p_raw": log_p if math.isfinite(log_p) else None,
        "ci_lower": estimate - radius,
        "ci_upper": estimate + radius,
        "reason": None if p > 0 else "p_underflow:see_log_p_raw",
    }


def fit_factorial(
    rows: list[Record],
    endpoint: str,
    *,
    levels: Record,
    groups: Record,
    settings: Record,
    baseline: bool = False,
    verification_requirement: str | None = "allow_unverified",
) -> Record:
    covariates = (["pre_outcome"] if baseline else []) + settings.get("additional_covariates", [])
    formula = endpoint + " ~ " + ("pre_outcome + " if baseline else "") + FACTOR_FORMULA
    if settings.get("additional_covariates"):
        formula += " + " + " + ".join(settings["additional_covariates"])
    result: Record = {
        "endpoint": endpoint,
        "formula": formula,
        "status": "blocked",
        "n_animals": 0,
        "settings": settings,
        "coefficients": [],
        "contrast": None,
    }
    if settings.get("model_family") != "gaussian_ols" or settings.get("covariance_estimator") != "classical":
        return {**result, "reason": "unsupported_registered_model_or_covariance"}
    if settings.get("factor_coding") != "treatment" or settings.get("test_tails") != "two-sided":
        return {**result, "reason": "unsupported_registered_coding_or_tail"}
    if settings.get("confidence_interval_method") != "student_t":
        return {**result, "reason": "unsupported_registered_ci_method"}
    if settings.get("contrast_estimand_and_weighting", {}).get("weighting") != "fixed_reference":
        return {**result, "reason": "unsupported_registered_contrast_weighting"}
    confidence = settings.get("confidence_level")
    if not isinstance(confidence, (float, int)) or not 0 < confidence < 1:
        raise ValueError("Register a valid confidence level")
    used, subset = analysis_subset(rows, [endpoint, *covariates], verification_requirement)
    result.update(subset=subset, n_animals=len({r["animal_id"] for r in used}))
    if endpoint == "latency_to_first_center_entry_s" and any(
        r.get("latency_right_censored") or (baseline and r.get("pre_latency_right_censored")) for r in used
    ):
        return {**result, "reason": "D09:censored_latency_requires_registered_censoring_model"}
    if not used:
        return {**result, "status": "unavailable", "reason": "insufficient_data"}
    matrix, names = design(used, levels, covariates)
    rank = int(np.linalg.matrix_rank(matrix))
    result.update(factor_levels=levels, coefficient_names=names, rank=rank, columns=matrix.shape[1])
    if rank < matrix.shape[1]:
        return {**result, "status": "error", "reason": "rank_deficient:interactions_retained"}
    df = len(used) - matrix.shape[1]
    if df <= 0:
        return {**result, "status": "error", "reason": "insufficient_residual_degrees_of_freedom"}
    y = np.asarray([r[endpoint] for r in used], dtype=float)
    coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
    fitted = matrix @ coefficients
    residuals = y - fitted
    sigma_sq = float(residuals @ residuals / df)
    # A numerically exact fit cannot supply meaningful Student-t uncertainty.
    # This is a floating-point resolution check, not a biological variance gate.
    resolution = np.finfo(float).eps * max(float(np.linalg.norm(y)), 1.0)
    if float(np.linalg.norm(residuals)) <= resolution * max(matrix.shape):
        return {
            **result,
            "status": "error",
            "reason": "numerically_zero_residual_variation",
            "residual_df": df,
            "residual_variance": sigma_sq,
        }
    inverse = np.linalg.inv(matrix.T @ matrix)
    covariance = inverse * sigma_sq
    result["coefficients"] = [
        {"term": name, **t_summary(float(beta), float(se), df, confidence)}
        for name, beta, se in zip(names, coefficients, np.sqrt(np.diag(covariance)), strict=True)
    ]
    references = settings["contrast_estimand_and_weighting"].get("reference_values", {})
    reference_values = {}
    for covariate in covariates:
        if covariate not in references:
            return {**result, "reason": "blocked:D09:contrast_covariate_reference_missing"}
        if covariate in levels:
            if references[covariate] not in levels[covariate]:
                return {**result, "reason": "blocked:D09:categorical_covariate_reference"}
            reference_values[covariate] = references[covariate]
        else:
            reference_values[covariate] = (
                float(np.mean([r[covariate] for r in used]))
                if references[covariate] == "mean"
                else float(references[covariate])
            )
    validate_contrast_mapping(groups, settings.get("contrast_roles", {}))
    targets = [{**groups[group], **reference_values} for group in WEIGHTS]
    target_matrix, _ = design(targets, levels, covariates)
    vector = np.asarray(list(WEIGHTS.values())) @ target_matrix
    estimate = float(vector @ coefficients)
    se = float(np.sqrt(max(float(vector @ covariance @ vector), 0)))
    result["contrast"] = {
        "name": "(G5-G6)-(G1-G2)",
        "weights": WEIGHTS,
        "reference_values": reference_values,
        "coefficient_vector": vector.tolist(),
        "group_estimates": dict(zip(WEIGHTS, (target_matrix @ coefficients).tolist(), strict=True)),
        **t_summary(estimate, se, df, confidence),
    }
    leverage = np.sum((matrix @ inverse) * matrix, axis=1)
    diagnostics = [
        {
            "animal_id": row["animal_id"],
            "session_id": row["session_id"],
            "observed": float(value),
            "fitted": float(fit),
            "residual": float(resid),
            "leverage": float(h),
        }
        for row, value, fit, resid, h in zip(used, y, fitted, residuals, leverage, strict=True)
    ]
    result.update(
        status="fitted",
        reason=None,
        residual_df=df,
        residual_variance=sigma_sq,
        diagnostics=diagnostics,
        covariance_matrix=covariance.tolist(),
    )
    if len(used) >= 3 and np.ptp(residuals) > 0:
        shapiro = stats.shapiro(residuals)
        result["normality_diagnostic"] = {
            "method": "Shapiro-Wilk residuals",
            "statistic": float(shapiro.statistic),
            "p_raw": float(shapiro.pvalue),
        }
    return result


def spearman_pairs(rows: list[Record], *, verification_requirement: str | None = "allow_unverified") -> list[Record]:
    result = []
    for other in ["valid_total_arm_entries", "total_distance_m", "mean_speed_m_per_s", "neutral_triads_pct"]:
        used, subset = analysis_subset(rows, ["spontaneous_alternation_pct", other], verification_requirement)
        pairs = [(r["spontaneous_alternation_pct"], r[other]) for r in used]
        reason = "insufficient_pairs" if len(pairs) < 3 else None
        if pairs and (len({p[0] for p in pairs}) == 1 or len({p[1] for p in pairs}) == 1):
            reason = "constant_input"
        correlation = stats.spearmanr(*zip(*pairs, strict=True)) if reason is None else None
        p_value = float(correlation.pvalue) if correlation else None
        if p_value == 0:
            p_value = None
            reason = "numerical_p_zero:not_reported_as_literal_zero"
        result.append(
            {
                "x": "spontaneous_alternation_pct",
                "y": other,
                "method": "Spearman",
                "n_animals": len(pairs),
                "rho": float(correlation.statistic) if correlation else None,
                "p_raw": p_value,
                "reason": reason,
                "subset": subset,
                "status": "computed" if reason is None else "unavailable",
            }
        )
    return result


def adjust_family(tests: list[Record], method: str, family: str) -> None:
    """Holm across all registered attempts, including unavailable tests in m."""
    if method not in ["holm", "none"]:
        raise ValueError("Supported multiplicity methods are registered holm or explicit none")
    available = sorted([t for t in tests if t.get("p_raw") is not None], key=lambda t: t["p_raw"])
    running = 0.0
    for i, test in enumerate(available):
        running = max(running, min(1.0, (len(tests) - i) * test["p_raw"]))
        test["p_adjusted"] = running if method == "holm" else test["p_raw"]
    for test in tests:
        test.setdefault("p_adjusted", None)
        test.update(correction_method=method, correction_family=family, family_size=len(tests))


def within_group(
    rows: list[Record],
    checks: list[str],
    confidence: float,
    *,
    verification_requirement: str | None = "allow_unverified",
) -> list[Record]:
    if not checks or not set(checks) <= {"DI", "percent_novel"}:
        raise ValueError("Register DI and/or percent_novel checks")
    if not 0 < confidence < 1:
        raise ValueError("Register a valid confidence level")
    results = []
    for group in sorted({r["group"] for r in rows}):
        for endpoint in checks:
            null = 0 if endpoint == "DI" else 50
            used, subset = analysis_subset(
                [r for r in rows if r["group"] == group], [endpoint], verification_requirement
            )
            values = [r[endpoint] for r in used]
            result: Record = {
                "group": group,
                "endpoint": endpoint,
                "null": null,
                "method": "one_sample_t",
                "n_animals": len(values),
                "p_raw": None,
                "reason": "insufficient_data",
                "subset": subset,
            }
            if len(values) >= 2:
                result.update(
                    t_summary(
                        float(np.mean(values)) - null,
                        float(np.std(values, ddof=1) / math.sqrt(len(values))),
                        len(values) - 1,
                        confidence,
                    )
                )
            result["status"] = "computed" if result["reason"] is None else "unavailable"
            results.append(result)
    return results
