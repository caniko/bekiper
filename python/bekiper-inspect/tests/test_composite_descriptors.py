"""Descriptor-driven composite panels, scored/diagnostic/folded labels and task stories."""

from __future__ import annotations

import copy
import tomllib

import numpy as np
import polars as pl
import pytest
import tomli_w
from conftest import GOLDEN_TOML, make_sample_df
from matplotlib import pyplot as plt

from bekiper_inspect import experiments
from bekiper_inspect.experiments import is_object
from bekiper_inspect.manifest import FeatureTimeSeries, InspectionManifest
from bekiper_inspect.plot.composite import (
    Variant,
    _frame,
    _overview_panel_count,
    _scored,
    _series_title,
    render_composite,
)
from bekiper_inspect.plot.feature._grouping import TASK_STORY_KEYS, group_into_stories
from bekiper_inspect.plot.feature._story_parts import _plot_objects_story as story

TASK = set(TASK_STORY_KEYS)
PANELS = ("object discrimination", "arm sequence", "ymaze arm sections", "center periphery")
SAMENESS = "1 same data as the object discrimination panel"
FOLD = "folded: the same gate is a component cell of its scored row"


def _build(tmp_path, keep=lambda f: True, perimeters=(), heuristics=None, features=()):
    """Golden manifest with features filtered/added; every referenced column exists."""
    data = tomllib.loads(GOLDEN_TOML.read_text())
    data["feature_inspections"] = [f for f in data["feature_inspections"] if keep(f)] + copy.deepcopy(list(features))
    for key in ("perimeters", "pixel_perimeters"):
        src = next(p for p in data[key] if p["label"] == "zone_a")
        data[key] += [{**src, "label": name} for name in perimeters]
    if heuristics is not None:
        row = data["heuristics"][0]
        data["heuristics"] = [{**row, "name": n, "result_column": n} for n in heuristics]
    df = make_sample_df()
    n = df.height

    def add_columns(node):
        nonlocal df
        if isinstance(node, list):
            for item in node:
                add_columns(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key in ("result_column", "value_column") and value not in df.columns:
                    df = df.with_columns(
                        pl.Series(value, np.linspace(0, 90, n) if key == "value_column" else np.arange(n) % 3 == 0)
                    )
                add_columns(value)

    add_columns(data)
    toml, parquet = tmp_path / "m.toml", tmp_path / "m.parquet"
    toml.write_text(tomli_w.dumps(data))
    df.write_parquet(parquet)
    return InspectionManifest.from_files(toml, parquet)


def _decomp(name, label=None):
    comp = {"label": "raw", "result_column": f"{name}_raw", "coordinate_labels": ["nose"], "perimeter_label": "arena"}
    return {
        "kind": "heuristic_decomposition",
        "description": name,
        "label": label or f"heuristic_decomposition_{name}",
        "heuristic_name": name,
        "result_column": f"{name}_final",
        "coordinate_labels": ["nose"],
        "perimeter_label": "arena",
        "show_final_result": True,
        "components": [comp],
    }


def _series(column):
    return {
        "kind": "time_series",
        "label": f"timeseries_{column}",
        "description": column,
        "unit": "deg",
        "value_column": column,
    }


def _render(manifest):
    fig, receipt = render_composite([Variant("recorded", manifest)], {"trial_id": "t"})
    titles = [ax.get_title().lower() for ax in fig.axes]
    plt.close("all")
    return titles, receipt


def _accounted(manifest, receipt):
    accounted = set(receipt.drawn) | set(receipt.folded) | set(receipt.omitted)
    return {f.label for f in manifest.feature_inspections} <= accounted


def _count(titles, text):
    return sum(text in t for t in titles)


@pytest.fixture
def nort_fields(monkeypatch):
    """Inject descriptor fields that object_recognition.toml may not carry yet."""
    real = experiments.descriptors

    def patched():
        out = []
        for d in real():
            if d["id"] == "object_recognition":
                extra = {
                    "diagnostic_prefixes": ["whisker_interaction_"],
                    "folded": [{"prefix": "proximity_diagnostic_", "reason": FOLD}],
                }
                d = {**d, "features": {**d["features"], **extra}}
            out.append(d)
        return tuple(out)

    monkeypatch.setattr(experiments, "descriptors", patched)


def test_golden_overview_panels_and_receipt(sample_manifest) -> None:
    titles, receipt = _render(sample_manifest)
    assert [_count(titles, p) for p in PANELS] == [1, 1, 1, 1]
    assert _overview_panel_count([Variant("recorded", sample_manifest)]) == 4
    assert _accounted(sample_manifest, receipt)
    for label in ("object_discrimination", "arm_sequence", "ymaze_arm_sections", "center_periphery"):
        assert receipt.drawn[label] == "1 overview"


def test_nort_scored_diagnostic_and_folded(tmp_path, nort_fields) -> None:
    objects = ("familiar", "novel")
    names = [*(f"exploring_{o}" for o in objects), "whisker_interaction_familiar"]
    disc = next(
        f for f in tomllib.loads(GOLDEN_TOML.read_text())["feature_inspections"] if f["kind"] == "object_discrimination"
    )
    features = [
        {**disc, "label": "preference_summary"},
        _decomp("rearing"),
        _decomp("grooming"),
        _decomp("proximity_diagnostic_x", "proximity_diagnostic_x"),
        _series("head__exploring_novel__within_angle_deg"),
    ]
    m = _build(tmp_path, lambda f: f["kind"] in {"perimeter", "object_discrimination"}, objects, names, features)
    scored, diagnostic = _scored(m)
    assert scored == ["exploring_familiar", "exploring_novel", "grooming", "rearing"]
    assert diagnostic == ["whisker_interaction_familiar"]
    titles, receipt = _render(m)
    assert receipt.folded["preference_summary"] == SAMENESS
    assert receipt.folded["proximity_diagnostic_x"] == FOLD
    assert _accounted(m, receipt) and _count(titles, "preference summary") == 0
    series = next(f for f in m.feature_inspections if isinstance(f, FeatureTimeSeries))
    assert _series_title(series, m).startswith("Head angle to the novel object")


def test_oul_object_series_titles(tmp_path) -> None:
    objs = [f"object_{i}" for i in range(1, 5)]
    features = [_series(f"head__exploring_{o}__within_angle_deg") for o in objs]
    m = _build(tmp_path, lambda f: f["kind"] == "perimeter", objs, [f"exploring_{o}" for o in objs], features)
    titles = [_series_title(f, m) for f in m.feature_inspections if isinstance(f, FeatureTimeSeries)]
    assert [t.split(" (")[0] for t in titles] == [f"Head angle to the object {i}" for i in range(1, 5)]
    assert _accounted(m, _render(m)[1])


def test_cheeseboard_has_no_overview_panels(tmp_path) -> None:
    m = _build(tmp_path, lambda f: f["kind"] not in TASK, ("start", "reward"))
    assert is_object("reward") and is_object("start")
    assert _overview_panel_count([Variant("recorded", m)]) == 0
    assert _accounted(m, _render(m)[1])


def test_oft_only_has_a_single_center_periphery_panel(tmp_path) -> None:
    keep = lambda f: (
        f["kind"] in {"perimeter", "zone", "center_periphery", "time_series"} or f["kind"].startswith("heuristic")
    )  # noqa: E731
    m = _build(tmp_path, keep)
    assert _overview_panel_count([Variant("recorded", m)]) == 1
    titles, receipt = _render(m)
    assert _count(titles, "center periphery") == 1 and _accounted(m, receipt)
    stories, _ = group_into_stories(m.feature_inspections, m.heuristics, m.perimeters)
    assert "center_periphery" in {s.key for s in stories}


def test_ymaze_only_groups_both_features_into_one_arm_sequence_story(tmp_path, monkeypatch) -> None:
    m = _build(tmp_path, lambda f: f["kind"] in {"arm_sequence", "ymaze_arm_sections"})
    m.feature_inspections.sort(
        key=lambda f: f.kind != "ymaze_arm_sections"
    )  # section first: story must still pick arm_sequence
    titles, receipt = _render(m)
    assert _count(titles, "arm sequence") == 1 and _count(titles, "ymaze arm sections") == 1
    assert _overview_panel_count([Variant("recorded", m)]) == 2 and _accounted(m, receipt)
    stories, _ = group_into_stories(m.feature_inspections, m.heuristics, m.perimeters)
    group = [s for s in stories if s.key == "arm_sequence"]
    assert len(group) == 1 and {f.kind for f in group[0].features} == {"arm_sequence", "ymaze_arm_sections"}
    drawn = []
    for kind in ("arm_sequence", "ymaze_arm_sections"):
        monkeypatch.setitem(
            story._TASK_RENDERERS, kind, lambda ax, manifest, feat, frame_shape, kind=kind: drawn.append(kind)
        )
    assert story._plot_task_story(m, group[0], _frame(m)) is not None
    assert drawn == ["arm_sequence"]


def test_unknown_task_falls_back_to_generic_panels(tmp_path) -> None:
    m = _build(tmp_path, lambda f: f["kind"] not in TASK)
    titles, receipt = _render(m)
    assert _overview_panel_count([Variant("recorded", m)]) == 0
    assert _accounted(m, receipt) and not any(p in t for t in titles for p in PANELS)
    assert {k for k, v in receipt.drawn.items() if v == "1 overview"} == {"perimeter_arena"}
