"""Reviewer cohort figures: named groups, endpoint classes and readable axes."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from PIL import Image

from bekiper_cli.post_analysis.review_figures import _named, build_panels, is_constant, write_review_figures
from bekiper_cli.post_analysis.render_bundle import draw_figure

CODES = ["1", "2", "3", "4", "5", "6", "7", "8", "unrecorded"]
GROUPS = {"5": "G5 · AD · HCAR1-WT · lactate", "unrecorded": "Unrecorded group"}


def _records(endpoint: str, values: dict[str, float]) -> tuple[list[dict], list[dict]]:
    points, summaries = [], []
    figure = "figure_oft_partition_0_endpoint_0"
    for code in CODES:
        for animal in (f"{code}a", f"{code}b"):
            points.append(
                {
                    "figure": figure,
                    "panel_id": "p0",
                    "assay": "oft",
                    "panel": endpoint,
                    "label": f"oft · post · A · recorded · {endpoint}",
                    "animal_id": animal,
                    "session_id": animal,
                    "category": code,
                    "value": values[code],
                    "pair_series": False,
                    "plot_pair_id": None,
                    "pair_order": 0,
                }
            )
        v = values[code]
        summaries.append(
            {
                "figure": figure,
                "panel_id": "p0",
                "category": code,
                "n_animals": 2,
                "mean": v,
                "sem": 0.5,
                "status": "available",
                "uncertainty": "95% CI",
                "error_half_width": 1.0,
                "distribution": {
                    "status": "available",
                    "box": {"whislo": v, "q1": v, "med": v, "q3": v, "whishi": v, "fliers": []},
                },
            }
        )
    return points, summaries


def test_groups_are_named_and_sem_is_labelled() -> None:
    points, summaries = _records("center_time_pct", {c: float(i) for i, c in enumerate(CODES)})
    panel = _named(next(iter(build_panels(points, summaries).values())), GROUPS)
    categories = {p["category"] for p in panel["points"]}
    assert "G5\nAD\nHCAR1-WT\nlactate" in categories and "Group 1" in categories and "Unrecorded\ngroup" in categories
    assert panel["label"] == "oft · post · A"
    assert all(s["uncertainty"] == "SEM" and s["error_half_width"] == 0.5 for s in panel["summaries"])


def test_nine_dense_categories_keep_the_axis_readable() -> None:
    points, summaries = _records("center_time_pct", {c: float(i) for i, c in enumerate(CODES)})
    panel = _named(next(iter(build_panels(points, summaries).values())), GROUPS)
    fig = draw_figure({"panels": [panel, panel], "panel_width": 6, "summary_label": "mean ± SEM"})
    try:
        assert fig.axes[0].get_position().height >= 0.45
    finally:
        plt.close(fig)
    # Nine groups: two wide columns, and upright stacked labels never overlap.
    fig = draw_figure({"panels": [panel, panel, panel], "panel_width": 8, "columns": 2, "summary_label": "mean ± SEM"})
    try:
        assert len({round(ax.get_position().x0, 3) for ax in fig.axes if ax.get_visible()}) == 2
        renderer = fig.canvas.get_renderer()
        boxes = [label.get_window_extent(renderer) for label in fig.axes[0].get_xticklabels()]
        assert all(a.x1 <= b.x0 for a, b in zip(boxes, boxes[1:], strict=False))
    finally:
        plt.close(fig)


def test_classes_criteria_constant_and_appendix(tmp_path) -> None:
    varied, varied_summaries = _records("center_time_pct", {c: float(i) for i, c in enumerate(CODES)})
    flat, flat_summaries = _records("analysis_duration_s", dict.fromkeys(CODES, 300.0))
    for record in flat + flat_summaries:
        record["panel_id"] = "p1"
    other, other_summaries = _records("rearing_s", {c: float(i) for i, c in enumerate(CODES)})
    for record in other + other_summaries:
        record["panel_id"] = "p2"
    # The recorded distribution figure repeats the partition in two panel styles.
    for panel_id in ("panel_0", "panel_1"):
        points, sums = _records("center_time_pct", {c: float(i) for i, c in enumerate(CODES)})
        for record in points + sums:
            record.update(figure="figure_oft_distributions_partition_0_endpoint_0", panel_id=panel_id)
        for summary in sums:
            summary["distribution"]["histogram"] = {"edges": [0.0, 4.0, 8.0], "counts": [1, 1]}
        varied += points
        varied_summaries += sums
    context = tmp_path / "primary45"
    (context / "_internal").mkdir(parents=True)
    (context / "statistics").mkdir()
    (context / "_internal" / "plotting_data.json").write_text(json.dumps(varied + flat + other))
    (context / "statistics" / "summaries.json").write_text(
        json.dumps(varied_summaries + flat_summaries + other_summaries)
    )
    out = tmp_path / "out"
    index = write_review_figures(context, out, GROUPS, {"oft": ["center_time_pct"]})
    classes = {e["endpoint"]: e["class"] for e in index["endpoints"]}
    assert classes == {"center_time_pct": "criteria", "analysis_duration_s": "constant", "rearing_s": "appendix"}
    criteria = next(e for e in index["endpoints"] if e["class"] == "criteria")
    assert [Image.open(out / f["path"]).format for f in criteria["webp"]] == ["WEBP", "WEBP"]
    assert criteria["webp"][1]["path"] == "criteria/oft/center_time_pct.distribution.webp"
    assert len(criteria["partitions"]) == 1 and criteria["webp"][0]["width"] > 0
    assert index["color_mode"] == "dark"
    appendix = next(e for e in index["endpoints"] if e["class"] == "appendix")
    assert appendix["webp"][0]["path"] == "appendix/oft/rearing_s.webp"
    assert next(e for e in index["endpoints"] if e["class"] == "constant")["webp"] == []
    assert criteria["partitions"][0]["groups"][4]["group"] == "G5 · AD · HCAR1-WT · lactate"
    assert is_constant([300.0, 300.0001]) and not is_constant([0.0, 1.0])


def test_threshold_contexts_draw_only_their_own_assays(tmp_path) -> None:
    """A Y-maze threshold context repeats OFT/NORT records; only the principal cohort draws them."""
    points, summaries = _records("center_time_pct", {c: float(i) for i, c in enumerate(CODES)})
    ymaze, ymaze_summaries = _records("spontaneous_alternation_pct", {c: float(i) for i, c in enumerate(CODES)})
    for record in ymaze + ymaze_summaries:
        record.update(figure="figure_ymaze_partition_0_endpoint_0", panel_id="p1")
    for record in ymaze:
        record["assay"] = "ymaze"
    drawn = {}
    for name in ("primary45", "primary25"):
        context = tmp_path / name
        (context / "_internal").mkdir(parents=True)
        (context / "statistics").mkdir()
        (context / "_internal" / "plotting_data.json").write_text(json.dumps(points + ymaze))
        (context / "statistics" / "summaries.json").write_text(json.dumps(summaries + ymaze_summaries))
        index = write_review_figures(context, tmp_path / f"out_{name}", GROUPS, {})
        drawn[name] = {e["assay"] for e in index["endpoints"]}
    assert drawn == {"primary45": {"oft", "ymaze"}, "primary25": {"ymaze"}}
    # Without a threshold-context assay, any context draws everything.
    context = tmp_path / "main"
    (context / "_internal").mkdir(parents=True)
    (context / "statistics").mkdir()
    (context / "_internal" / "plotting_data.json").write_text(json.dumps(points))
    (context / "statistics" / "summaries.json").write_text(json.dumps(summaries))
    assert {e["assay"] for e in write_review_figures(context, tmp_path / "out_main", GROUPS, {})["endpoints"]} == {
        "oft"
    }
