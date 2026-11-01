"""Reviewer cohort figures from one post-analysis context.

Reads ``_internal/plotting_data.json`` (points) and ``statistics/summaries.json``
(per-group summaries) and writes, under ``--output``:

* ``criteria/<assay>/<endpoint>.webp`` and ``appendix/<assay>/<endpoint>.webp``:
  one figure per endpoint with a panel per session partition and named groups,
  plus ``<endpoint>.distribution.webp`` with the recorded per-group histograms;
* ``cohort_index.json``: endpoint classes, per-group n/mean/SEM, source hashes
  and figure paths for the package builder.

Every figure is drawn in the active colour mode (dark by default). Constant
endpoints (e.g. an analysis duration that is 300 s everywhere) are listed for a
QC table instead of being drawn. Values are never recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from bekiper_inspect.experiments import context_assays
from bekiper_inspect.plot.composite import WEBP_OPTIONS
from matplotlib import pyplot as plt
from PIL import Image

from . import render_bundle
from .render_bundle import draw_figure, use_color_mode

Record = dict[str, Any]
DPI = 130


def _key(point: Record) -> tuple[str, str]:
    return point["figure"], point["panel_id"]


def _partition(label: str, endpoint: str) -> str:
    """'nort · post · test · recorded · DI' -> 'nort · post · test'."""
    parts = [p for p in label.split(" · ") if p not in (endpoint, "recorded")]
    return " · ".join(parts)


def _hash(records: list[Record]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def is_constant(values: list[float]) -> bool:
    """True when an endpoint barely varies, so a figure would show nothing."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return False
    # ponytail: fixed relative tolerance; a per-endpoint QC registry would be exact.
    return max(finite) - min(finite) <= 1e-3 * max(abs(sum(finite) / len(finite)), 1.0)


def build_panels(points: list[Record], summaries: list[Record]) -> dict[tuple[str, str], Record]:
    """Group points and summaries into figure panels; distribution panels are flagged."""
    panels: dict[tuple[str, str], Record] = {}
    for point in points:
        panel = panels.setdefault(
            _key(point),
            {
                "figure": point["figure"],
                "panel_id": point["panel_id"],
                "assay": point["assay"],
                "endpoint": point["panel"],
                "label": point["label"],
                "distribution": "_distributions_" in point["figure"],
                "population_policy": point.get("population_policy"),
                "points": [],
                "summaries": [],
            },
        )
        panel["points"].append(point)
    for summary in summaries:
        key = (summary["figure"], summary["panel_id"])
        if key in panels:
            panels[key]["summaries"].append(summary)
    return panels


def _group_name(code: Any, groups: dict[str, str], stacked: bool) -> str:
    name = groups.get(str(code), f"Group {code}" if str(code).isdigit() else str(code))
    if not stacked:
        return name
    # One factor per line ("G5\nAD\nHCAR1-WT\nlactate") keeps eight named groups upright and legible.
    return name.replace(" · ", "\n") if " · " in name else textwrap.fill(name, width=10)


def _named(panel: Record, groups: dict[str, str]) -> Record:
    """A draw_panel record with readable group names (boxplot, or histogram for distributions)."""
    histogram = panel.get("distribution", False)

    def rename(code: Any) -> str:
        return _group_name(code, groups, stacked=not histogram)

    summaries = sorted(panel["summaries"], key=lambda s: str(s["category"]).zfill(4))
    return {
        "label": _partition(panel["label"], panel["endpoint"]),
        "y_label": panel["endpoint"],
        "kind": "categorical",
        "style": "histogram" if histogram else "boxplot",
        "population_policy": panel.get("population_policy"),
        "n_animals": len({p["animal_id"] for p in panel["points"]}),
        "points": [{**p, "category": rename(p["category"])} for p in panel["points"]],
        "summaries": [_sem_summary(s, rename(s["category"])) for s in summaries],
    }


def _sem_summary(summary: Record, category: str) -> Record:
    """The recorded summary drawn as mean ± SEM (no error bar below two animals)."""
    sem = summary.get("sem") if (summary.get("n_animals") or 0) >= 2 and summary.get("mean") is not None else None
    return {**summary, "category": category, "error_half_width": sem, "uncertainty": "SEM" if sem is not None else None}


def render_bundle_palette() -> Record:
    """The palette the shared renderer is drawing with."""
    return render_bundle._palette


def _render(panels: list[Record], groups: dict[str, str], caption: str, path: Path, output: Path) -> Record:
    """Draw panels as one figure, save it as WebP atomically; returns its path, size and sha256."""
    named = [_named(p, groups) for p in panels]
    # More than eight groups need wider panels so the stacked labels keep apart.
    crowded = any(len(p["summaries"]) > 8 for p in named)
    figure = {
        "panels": named,
        "panel_width": 8 if crowded else 6,
        "columns": 2 if crowded else 3,
        "summary_label": caption,
    }
    fig = draw_figure(figure)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.webp")
        fig.savefig(tmp, format="webp", dpi=DPI, pil_kwargs=dict(WEBP_OPTIONS))
        tmp.replace(path)
    finally:
        plt.close(fig)
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": str(path.relative_to(output)),
        "width": width,
        "height": height,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_review_figures(
    context: Path,
    output: Path,
    groups: dict[str, str],
    primary: dict[str, list[str]],
    assays: list[str] | None = None,
) -> Record:
    points = json.loads((context / "_internal" / "plotting_data.json").read_text())
    summaries = json.loads((context / "statistics" / "summaries.json").read_text())
    panels = build_panels(points, summaries)
    del points, summaries
    if assays is None:
        # Default: the context's own assays (a threshold context repeats the others' records).
        assays = context_assays(context, {p["assay"] for p in panels.values()})
    by_endpoint: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for panel in panels.values():
        if assays is None or panel["assay"] in assays:
            by_endpoint[(panel["assay"], panel["endpoint"])].append(panel)
    entries: list[Record] = []
    for (assay, endpoint), members in sorted(by_endpoint.items()):
        members.sort(key=lambda p: p["label"])
        figures = [m for m in members if not m["distribution"]]
        # A recorded distribution figure repeats one partition's data in several
        # panel styles; its first panel is the histogram drawn here.
        first: dict[str, Record] = {}
        for m in sorted((m for m in members if m["distribution"]), key=lambda m: m["panel_id"], reverse=True):
            first[m["figure"]] = m
        distributions = sorted(first.values(), key=lambda m: m["label"])
        if not figures:
            continue
        values = [p["value"] for m in figures for p in m["points"]]
        kind = "criteria" if endpoint in primary.get(assay, []) else "appendix"
        if is_constant(values):
            kind = "constant"
        entry: Record = {
            "assay": assay,
            "endpoint": endpoint,
            "class": kind,
            "partitions": [
                {
                    "label": _partition(m["label"], endpoint),
                    "source_figure": m["figure"],
                    "source_sha256": _hash(m["points"]),
                    "groups": [
                        {
                            "group": _group_name(s["category"], groups, stacked=False),
                            "n_animals": s.get("n_animals"),
                            "mean": s.get("mean"),
                            "sem": s.get("sem"),
                        }
                        for s in sorted(m["summaries"], key=lambda s: str(s["category"]).zfill(4))
                    ],
                }
                for m in figures
            ],
            "range": [min(values, default=None), max(values, default=None)] if kind == "constant" else None,
            "webp": [],
        }
        if kind != "constant":
            target = output / kind / assay
            caption = "Each point is one recording; box = native quartiles, bar = mean ± SEM."
            entry["webp"].append(_render(figures, groups, caption, target / f"{endpoint}.webp", output))
            if distributions:
                caption = "Recorded per-group histograms (native bins)."
                entry["webp"].append(
                    _render(distributions, groups, caption, target / f"{endpoint}.distribution.webp", output)
                )
        entries.append(entry)
    index = {"context": context.name, "color_mode": render_bundle_palette()["mode"], "endpoints": entries}
    (output / "cohort_index.json").write_text(json.dumps(index, indent=2, default=str))
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=Path, help="JSON {group code: display name}")
    parser.add_argument("--primary", type=Path, help="JSON {assay: [criteria endpoints]}")
    parser.add_argument(
        "--assay", action="append", help="Only these assays (repeatable); default: the context's assays"
    )
    parser.add_argument("--theme", choices=["dark", "light"], help="Colour mode (default: env, then dark)")
    args = parser.parse_args(argv)
    groups = json.loads(args.groups.read_text()) if args.groups else {}
    primary = json.loads(args.primary.read_text()) if args.primary else {}
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(use_color_mode(args.theme)):
        write_review_figures(args.context, args.output, groups, primary, args.assay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
