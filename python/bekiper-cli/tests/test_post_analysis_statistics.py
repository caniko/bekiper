import itertools
import math

import pytest

from reference_post_analysis.statistics import (
    fit_factorial,
    pairing,
    spearman_pairs,
    validate_contrast_mapping,
    within_group,
)

LEVELS = {"five_xfad": ["WT", "AD"], "hcar1": ["KO", "WT"], "treatment": ["vehicle", "lactate"]}
GROUPS = {
    "G5": {"five_xfad": "AD", "hcar1": "WT", "treatment": "lactate"},
    "G6": {"five_xfad": "AD", "hcar1": "WT", "treatment": "vehicle"},
    "G1": {"five_xfad": "AD", "hcar1": "KO", "treatment": "lactate"},
    "G2": {"five_xfad": "AD", "hcar1": "KO", "treatment": "vehicle"},
}
ROLES = {"ad_level": "AD", "hcar1_wt": "WT", "hcar1_ko": "KO", "lactate": "lactate", "vehicle": "vehicle"}
SETTINGS = {
    "model_family": "gaussian_ols",
    "covariance_estimator": "classical",
    "factor_coding": "treatment",
    "test_tails": "two-sided",
    "confidence_interval_method": "student_t",
    "confidence_level": 0.95,
    "contrast_estimand_and_weighting": {"weighting": "fixed_reference", "reference_values": {}},
    "contrast_roles": ROLES,
}


def factorial_rows():
    rows = []
    for cell, (ad, hcar, treatment) in enumerate(itertools.product(*LEVELS.values())):
        mean = 0.2 * (treatment == "lactate") + 0.4 * (treatment == "lactate" and hcar == "WT" and ad == "AD")
        for replicate, noise in enumerate([-0.03, -0.01, 0.01, 0.03]):
            rows.append(
                {
                    "animal_id": f"a{cell}_{replicate}",
                    "session_id": f"s{cell}_{replicate}",
                    "five_xfad": ad,
                    "hcar1": hcar,
                    "treatment": treatment,
                    "DI": mean + noise,
                }
            )
    return rows


def test_model_based_planned_contrast_sign_and_uncertainty():
    result = fit_factorial(factorial_rows(), "DI", levels=LEVELS, groups=GROUPS, settings=SETTINGS)
    assert result["status"] == "fitted"
    assert result["contrast"]["estimate"] == pytest.approx(0.4)
    assert result["contrast"]["standard_error"] > 0
    # Independent balanced-cell oracle: pooled SSE=8*sum(noise^2), df=32-8;
    # contrast variance is four independent cell-mean variances, n=4 per cell.
    residual_variance = 8 * sum(n**2 for n in [-0.03, -0.01, 0.01, 0.03]) / 24
    assert result["contrast"]["standard_error"] == pytest.approx(math.sqrt(4 * residual_variance / 4))
    assert result["contrast"]["ci_lower"] < 0.4 < result["contrast"]["ci_upper"]
    assert 0 < result["contrast"]["p_raw"] < 1
    assert result["n_animals"] == 32
    assert len(result["coefficients"]) == 8


def test_missing_cell_is_rank_deficient_not_simplified():
    result = fit_factorial(factorial_rows()[4:], "DI", levels=LEVELS, groups=GROUPS, settings=SETTINGS)
    assert result["reason"] == "rank_deficient:interactions_retained"
    assert result["contrast"] is None


def test_wrong_contrast_mapping_fails():
    with pytest.raises(ValueError, match="role mismatch"):
        validate_contrast_mapping({**GROUPS, "G5": GROUPS["G1"]}, ROLES)


def test_pairing_by_animal_not_row_order_reports_unmatched():
    rows = [
        {
            "animal_id": animal,
            "assay": "ymaze",
            "phase": "session",
            "session_id": f"{animal}-{time}",
            "timepoint": time,
            "SA": value,
        }
        for animal, time, value in [("a", "post", 75), ("b", "pre", 99), ("a", "pre", 60)]
    ]
    paired, ledger = pairing(rows, "SA")
    assert paired[0]["DeltaSA"] == 15
    assert paired[0]["animal_id"] == "a"
    assert ledger[1]["reason"] == "unmatched_prepost"


@pytest.mark.parametrize("analysis", ["spearman", "within_group"])
def test_qc_tests_never_count_repeated_sessions_as_independent_animals(analysis):
    rows = [
        {
            "animal_id": "same_animal",
            "session_id": f"session_{i}",
            "group": "G5",
            "DI": 0.1 * i,
            "spontaneous_alternation_pct": 50 + i,
            "valid_total_arm_entries": 10 + i,
        }
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="Animal replicates"):
        if analysis == "spearman":
            spearman_pairs(rows)
        else:
            within_group(rows, ["DI"], 0.95)
