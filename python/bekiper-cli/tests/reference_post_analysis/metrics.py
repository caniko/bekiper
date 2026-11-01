"""Pure SRC-OFT/SRC-YM/SRC-NORT arithmetic, with typed missingness."""

from __future__ import annotations

import math
from typing import Any

Record = dict[str, Any]
CORRECT = {"ABC", "ACB", "BAC", "BCA", "CAB", "CBA"}
INCORRECT = {"AAB", "AAC", "ABA", "ACA", "BAB", "BBA", "BBC", "BCB", "CAC", "CBC", "CCA", "CCB"}
NEUTRAL = {"AAA", "ABB", "ACC", "BAA", "BBB", "BCC", "CAA", "CBB", "CCC"}
TABLE = {
    **dict.fromkeys(CORRECT, "correct"),
    **dict.fromkeys(INCORRECT, "incorrect"),
    **dict.fromkeys(NEUTRAL, "neutral"),
}
OFT_PRIMARY = [
    "total_distance_m",
    "mean_speed_m_per_s",
    "center_time_s",
    "center_time_pct",
    "center_entries",
    "latency_to_first_center_entry_s",
]
OFT_SECONDARY = ["periphery_time_pct", "freezing_s", "grooming_s", "rearing_s", "immobility_s", "fecal_boli_count"]
YM_SCORES = ["spontaneous_alternation_pct", "spontaneous_alternation_pct_standard", "neutral_triads_pct"]
COUNT_FIELDS = {
    "center_entries",
    "fecal_boli_count",
    "raw_total_arm_entries",
    "valid_total_arm_entries",
    "removed_short_entries",
    "total_triads",
    "usable_triads",
    "correct_alternations",
    "incorrect_alternations",
    "neutral_triads",
} | {f"{stage}_arm_{arm}_entries" for stage in ["raw", "valid"] for arm in "ABC"}


def nonnegative(value: Any, name: str, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: requires a finite nonnegative number")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name}: exceeds {maximum}")
    return float(value)


def missing(result: Record, field: str, reason: str) -> None:
    result[field] = None
    result.setdefault("missing_reasons", {})[field] = reason


def ymaze_scores(
    sequence: str | list[str], *, policy: str | None, short_policy: str | None = None, mapping: Record | None = None
) -> Record:
    """Sliding windows from valid entries; never deduplicate repeated arms."""
    labels = list(sequence.replace(" ", "")) if isinstance(sequence, str) else sequence
    if any(label not in "ABC" or len(label) != 1 for label in labels):
        raise ValueError("valid_arm_entry_sequence: only A/B/C labels are supported")
    result: Record = {"valid_total_arm_entries": len(labels), "valid_arm_entry_sequence": labels, "missing_reasons": {}}
    for arm in "ABC":
        result[f"valid_arm_{arm}_entries"] = labels.count(arm)
    if policy is None:
        for key in YM_SCORES + [
            "correct_alternations",
            "incorrect_alternations",
            "neutral_triads",
            "usable_triads",
            "low_quality",
            "window_labels",
        ]:
            missing(result, key, "blocked:D01")
        result["total_triads"] = len(labels) - 2 if len(labels) >= 3 else None
        return result
    if policy == "source_table":
        table = TABLE
    elif policy == "custom" and mapping is not None and set(mapping) == set(TABLE):
        if not set(mapping.values()) <= {"correct", "incorrect", "neutral"}:
            raise ValueError("Invalid custom triad categories")
        table = mapping
    else:
        raise ValueError("Triad policy requires source_table or a full registered custom mapping")
    windows = [table["".join(labels[i : i + 3])] for i in range(max(len(labels) - 2, 0))]
    counts = {category: windows.count(category) for category in ["correct", "incorrect", "neutral"]}
    total = len(windows)
    usable = counts["correct"] + counts["incorrect"]
    result.update(
        window_labels=windows,
        correct_alternations=counts["correct"],
        incorrect_alternations=counts["incorrect"],
        neutral_triads=counts["neutral"],
        usable_triads=usable,
        total_triads=total,
        low_quality=len(labels) < 8 or usable < 5,
    )
    if len(labels) < 3 and short_policy != "zero_windows":
        missing(result, "total_triads", "blocked:D02:insufficient_entries")
    for key, denominator, numerator in [
        (YM_SCORES[0], usable, counts["correct"]),
        (YM_SCORES[1], total, counts["correct"]),
        (YM_SCORES[2], total, counts["neutral"]),
    ]:
        if denominator > 0:
            result[key] = numerator / denominator * 100
        else:
            missing(
                result, key, "undefined:insufficient_entries" if len(labels) < 3 else "undefined:zero_usable_triads"
            )
    return result


def nort_scores(
    *, TA: float | None = None, TB: float | None = None, TN: float | None = None, TF: float | None = None
) -> Record:
    """No implicit pseudocount; source side_bias remains algebraic, not spatial."""
    result: Record = {"TA": TA, "TB": TB, "TN": TN, "TF": TF, "missing_reasons": {}}
    for key, value in list(result.items()):
        if key == "missing_reasons":
            continue
        if value is not None:
            nonnegative(value, key)
        else:
            missing(result, key, "not_measured")
    for a, b, total_name, indices in [
        (TA, TB, "total_training_exploration", ["object_bias", "side_bias"]),
        (TN, TF, "total_test_exploration", ["DI", "RI", "percent_novel", "LR"]),
    ]:
        if a is None or b is None:
            for key in [total_name] + indices:
                missing(result, key, "not_measured:component_missing")
            continue
        total = a + b
        result[total_name] = total
        if total == 0:
            for key in indices:
                missing(result, key, "undefined:zero_exploration_total")
        elif total_name == "total_training_exploration":
            result.update(object_bias=(a - b) / total, side_bias=(b - a) / total)
        else:
            result.update(DI=(a - b) / total, RI=a / total, percent_novel=100 * a / total)
            if a == 0 or b == 0:
                missing(result, "LR", "undefined:zero_component_no_pseudocount")
            else:
                result["LR"] = math.log(a / b)
    return result


def oft_scores(values: Record) -> Record:
    result: Record = {"missing_reasons": {}}
    for key in OFT_PRIMARY + OFT_SECONDARY:
        value = values.get(key)
        if value is None:
            missing(result, key, values.get("missing_reasons", {}).get(key, "not_measured"))
        else:
            nonnegative(value, key, 300 if key == "center_time_s" else 100 if key.endswith("pct") else None)
            if key in ["center_entries", "fecal_boli_count"] and int(value) != value:
                raise ValueError(f"{key}: requires an integer count")
            result[key] = value
    if result["center_time_s"] is not None:
        computed = result["center_time_s"] / 300 * 100
        result["center_time_pct"] = computed
        result["missing_reasons"].pop("center_time_pct", None)
        supplied = values.get("center_time_pct")
        if supplied is not None and not math.isclose(supplied, computed, abs_tol=1e-8):
            result["upstream_center_time_pct"] = supplied
            result["disagreements"] = ["center_time_pct:fixed_300s_denominator"]
    return result
