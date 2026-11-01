"""Column identities and finite scalar parsing shared by angle renderers."""

from typing import Any, cast

import numpy as np

from bekiper_inspect.manifest import FeatureHeuristic

from .angleoverlaysample import DIAGNOSTIC_COLUMN_PREFIX


def _parse_diag_result_column(result_column: str) -> tuple[str, str] | None:
    if not result_column.startswith(DIAGNOSTIC_COLUMN_PREFIX):
        return None
    stem = result_column[len(DIAGNOSTIC_COLUMN_PREFIX) :]
    if "__" not in stem:
        return None
    heuristic_name, component = stem.split("__", 1)
    if not heuristic_name or not component:
        return None
    return heuristic_name, component


def _heuristic_name_from_feature(feature: FeatureHeuristic, *, heuristic_name: str | None = None) -> str | None:
    if heuristic_name:
        return heuristic_name
    parsed = _parse_diag_result_column(feature.result_column)
    if parsed is not None:
        return parsed[0]
    label = str(feature.label)
    if label.startswith("heuristic_"):
        return label.removeprefix("heuristic_")
    return cast(str, feature.result_column) or None


def _float_or_none(value: object) -> float | None:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _float_or_nan(value: object) -> float:
    result = _float_or_none(value)
    return float("nan") if result is None else result
