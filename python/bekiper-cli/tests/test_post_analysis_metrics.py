"""Recipe fixtures test the actual scorer; table-policy cases are conditional."""

import json
from pathlib import Path

import pytest

from reference_post_analysis.metrics import nort_scores, oft_scores, ymaze_scores

CASES = json.loads((Path(__file__).parents[3] / "behavioral_post_analysis_harness/tests/golden_cases.json").read_text())


@pytest.mark.parametrize("case", CASES["ymaze_sequences"] + CASES["ymaze_short_sequences"])
def test_conditional_table_sequences(case):
    result = ymaze_scores(case["sequence"], policy="source_table", short_policy="zero_windows")
    for key, value in case["expected"].items():
        assert result[key] == pytest.approx(value) if isinstance(value, float) else result[key] == value


@pytest.mark.parametrize("case", CASES["ymaze_triplets"])
def test_all_table_triads(case):
    assert ymaze_scores(case["triplet"], policy="source_table")["window_labels"] == [case["expected"]]


@pytest.mark.parametrize("case", CASES["nort_test_scores"])
def test_nort_test_arithmetic(case):
    result = nort_scores(TN=case["TN"], TF=case["TF"])
    for key, value in case["expected"].items():
        assert result[key] == pytest.approx(value) if value is not None else result[key] is None


@pytest.mark.parametrize("case", CASES["nort_training_scores"])
def test_nort_training_arithmetic(case):
    result = nort_scores(TA=case["TA"], TB=case["TB"])
    for key, value in case["expected"].items():
        assert result[key] == pytest.approx(value) if value is not None else result[key] is None


def test_fixed_oft_denominator_and_missing_secondary():
    result = oft_scores({"center_time_s": 75, "analysis_duration_s": 150})
    assert result["center_time_pct"] == 25
    assert result["freezing_s"] is None
    assert result["missing_reasons"]["freezing_s"] == "not_measured"


def test_no_unregistered_triad_policy():
    result = ymaze_scores("ABCABBAC", policy=None)
    assert result["spontaneous_alternation_pct"] is None
    assert result["missing_reasons"]["spontaneous_alternation_pct"] == "blocked:D01"


@pytest.mark.parametrize("times", [{"TN": -1, "TF": 1}, {"TA": float("nan"), "TB": 0}])
def test_reject_impossible_exploration(times):
    with pytest.raises(ValueError):
        nort_scores(**times)
