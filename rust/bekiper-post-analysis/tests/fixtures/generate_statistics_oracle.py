"""Independent cell-means/QR oracle, never imported by the production engine.

The native engine uses treatment coding and SVD. This oracle uses eight cell
indicators, QR covariance, and an explicit inclusion-exclusion change of basis.
All observations are deterministic synthetic test inputs.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import linalg, stats

levels = {
    "five_xfad": ["WT", "AD"],
    "hcar1": ["KO", "WT"],
    "treatment": ["vehicle", "lactate"],
    "sex": ["female", "male"],
}
roles = {"ad_level": "AD", "hcar1_wt": "WT", "hcar1_ko": "KO", "lactate": "lactate", "vehicle": "vehicle"}
groups = {
    "G5": {"five_xfad": "AD", "hcar1": "WT", "treatment": "lactate"},
    "G6": {"five_xfad": "AD", "hcar1": "WT", "treatment": "vehicle"},
    "G1": {"five_xfad": "AD", "hcar1": "KO", "treatment": "lactate"},
    "G2": {"five_xfad": "AD", "hcar1": "KO", "treatment": "vehicle"},
}
rows, cells = [], []
cell_means = [0.08, 0.13, 0.15, 0.20, 0.19, 0.26, 0.25, 0.52]
for cell, factors in enumerate(itertools.product(*(levels[name] for name in ["five_xfad", "hcar1", "treatment"]))):
    for rep in range([3, 4, 5, 6, 4, 5, 3, 7][cell]):
        pre = 0.03 * cell + 0.07 * rep + 0.025 * ((cell + rep) % 3)
        age = 7 + 0.2 * ((cell * 3 + rep) % 5) + 0.01 * rep * rep
        male = (rep + cell) % 2
        noise = [0.018, -0.029, 0.037, -0.011, -0.021, 0.024, -0.008][rep] + 0.004 * (cell % 3)
        outcome = cell_means[cell] + 0.65 * pre + 0.025 * (age - 7) + 0.032 * male + noise
        rows.append(
            {
                "animal_id": f"SYNTHETIC_{cell}_{rep}",
                "session_id": f"s_{cell}_{rep}",
                **dict(zip(["five_xfad", "hcar1", "treatment"], factors, strict=True)),
                "sex": "male" if male else "female",
                "age": age,
                "pre_outcome": pre,
                "DI": outcome,
                "excluded": False,
                "synthetic_observation": True,
            }
        )
        cells.append(cell)


def oracle(adjusted):
    indicator = np.eye(8)[cells]
    extras = np.asarray([[r["pre_outcome"], r["age"], float(r["sex"] == "male")] for r in rows])
    design = np.column_stack([indicator, extras]) if adjusted else indicator
    outcome = np.asarray([r["DI"] for r in rows])
    beta = linalg.lstsq(design, outcome, lapack_driver="gelsy")[0]
    fitted = design @ beta
    residual = outcome - fitted
    df = len(rows) - design.shape[1]
    variance = float(residual @ residual / df)
    q, r = linalg.qr(design, mode="economic")
    inverse_r = linalg.solve_triangular(r, np.eye(len(beta)))
    covariance = variance * inverse_r @ inverse_r.T
    # Inclusion-exclusion transforms eight cell intercepts to treatment-coded
    # main effects/interactions, independently of the native term builder.
    transform = np.zeros((len(beta), len(beta)))
    term_masks = [0, 4, 2, 1, 6, 5, 3, 7]
    for term, mask in enumerate(term_masks):
        for cell in range(8):
            if cell & mask == cell:
                transform[term, cell] = (-1) ** (mask.bit_count() - cell.bit_count())
    if adjusted:
        transform[8:, 8:] = np.eye(3)
    coefficients = transform @ beta
    native_covariance = transform @ covariance @ transform.T
    coefficient_errors = np.sqrt(np.diag(native_covariance))
    coefficient_statistics = coefficients / coefficient_errors
    coefficient_radius = float(stats.t.ppf(0.975, df)) * coefficient_errors
    vector = np.zeros(len(beta))
    vector[[7, 6, 5, 4]] = [1, -1, -1, 1]
    estimate = float(vector @ beta)
    standard_error = float(np.sqrt(vector @ covariance @ vector))
    statistic = estimate / standard_error
    radius = float(stats.t.ppf(0.975, df)) * standard_error
    references = (
        {"pre_outcome": float(extras[:, 0].mean()), "age": float(extras[:, 1].mean()), "sex": "female"}
        if adjusted
        else {}
    )
    group_estimates = {}
    for group, cell in [("G5", 7), ("G6", 6), ("G1", 5), ("G2", 4)]:
        target = np.zeros(len(beta))
        target[cell] = 1
        if adjusted:
            target[8:] = [references["pre_outcome"], references["age"], 0]
        group_estimates[group] = float(target @ beta)
    return {
        "n_animals": len(rows),
        "df": df,
        "residual_variance": variance,
        "coefficients": coefficients.tolist(),
        "covariance_matrix": native_covariance.tolist(),
        "coefficient_summaries": [
            {
                "standard_error": float(error),
                "statistic": float(statistic),
                "p_raw": float(2 * stats.t.sf(abs(statistic), df)),
                "ci_lower": float(coefficient - radius),
                "ci_upper": float(coefficient + radius),
            }
            for coefficient, error, statistic, radius in zip(
                coefficients, coefficient_errors, coefficient_statistics, coefficient_radius, strict=True
            )
        ],
        "fitted": fitted.tolist(),
        "residuals": residual.tolist(),
        "leverage": np.sum(q * q, axis=1).tolist(),
        "contrast": {
            "estimate": estimate,
            "standard_error": standard_error,
            "statistic": statistic,
            "p_raw": float(2 * stats.t.sf(abs(statistic), df)),
            "log_p_raw": float(np.log(2) + stats.t.logsf(abs(statistic), df)),
            "ci_lower": estimate - radius,
            "ci_upper": estimate + radius,
            "reference_values": references,
            "group_estimates": group_estimates,
        },
    }


settings = {
    "model_family": "gaussian_ols",
    "covariance_estimator": "classical",
    "factor_coding": "treatment",
    "test_tails": "two-sided",
    "confidence_interval_method": "student_t",
    "confidence_level": 0.95,
    "contrast_estimand_and_weighting": {"weighting": "fixed_reference", "reference_values": {}},
    "contrast_roles": roles,
    "additional_covariates": [],
}
spearman_x = [1, 1, 2, 3, 3, 4, 5]
spearman_y = [4, 1, 2, 2, 5, 3, 6]
spearman = stats.spearmanr(spearman_x, spearman_y)
within_values = [0.10, 0.04, -0.03, 0.22, 0.16, -0.01, 0.31]
within_result = stats.ttest_1samp(within_values, 0)
within_interval = within_result.confidence_interval(0.95)
fixture = {
    "notice": "Synthetic numerical oracle only; no biological observations or scientific approvals",
    "method": "independent eight-cell indicators, scipy.linalg QR covariance, inclusion-exclusion coding transform",
    "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__},
    "levels": levels,
    "groups": groups,
    "settings": settings,
    "rows": rows,
    "post_only": oracle(False),
    "baseline_adjusted": oracle(True),
    "spearman": {"x": spearman_x, "y": spearman_y, "rho": float(spearman.statistic), "p_raw": float(spearman.pvalue)},
    "within_group": {
        "values": within_values,
        "estimate": float(np.mean(within_values)),
        "standard_error": float(stats.sem(within_values)),
        "statistic": float(within_result.statistic),
        "df": int(within_result.df),
        "p_raw": float(within_result.pvalue),
        "ci_lower": float(within_interval.low),
        "ci_upper": float(within_interval.high),
    },
}
Path(__file__).with_name("statistics_oracle.json").write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
