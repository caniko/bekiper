"""One summary figure per animal (reviewer package).

Each panel is one recorded criteria endpoint of one protocol step (habituation,
T1 training, T2 test, Y-maze): the animal's group as faint points with a box per
timepoint, and the animal itself highlighted pre -> post. The job carries only
recorded values (selected by the Rust package builder); nothing is recomputed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from bekiper_inspect.plot.theme import get_active_theme

# ponytail: bump when the layout changes so cached summaries are re-rendered.
ANIMAL_SUMMARY_VERSION = 1

TIMEPOINTS = ("pre", "post")
COLUMNS = 4


def _panel(ax: Any, panel: dict[str, Any], pal: dict[str, Any]) -> None:
    group = panel.get("group", {})
    animal = panel.get("animal", {})
    xs = []
    for x, timepoint in enumerate(TIMEPOINTS):
        values = [v for v in group.get(timepoint, []) if v is not None and math.isfinite(v)]
        if values:
            rng = np.random.default_rng(x)
            jitter = rng.uniform(-0.12, 0.12, len(values))
            ax.scatter(x + jitter, values, s=10, color=pal["muted"], alpha=0.45, linewidths=0, zorder=1)
            ax.boxplot(
                [values],
                positions=[x],
                widths=0.42,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": pal["box_fill"], "alpha": 0.6, "edgecolor": pal["muted"]},
                medianprops={"color": pal["muted"]},
                whiskerprops={"color": pal["muted"]},
                capprops={"color": pal["muted"]},
                manage_ticks=False,
                zorder=2,
            )
        if animal.get(timepoint) is not None:
            xs.append((x, animal[timepoint]))
    if xs:
        ax.plot([x for x, _ in xs], [v for _, v in xs], color=pal["highlight"], linewidth=1.6, zorder=3)
        ax.scatter(
            [x for x, _ in xs],
            [v for _, v in xs],
            s=60,
            color=pal["highlight"],
            edgecolors=pal["ink"],
            linewidths=0.8,
            zorder=4,
        )
        for x, v in xs:
            ax.annotate(
                f"{v:.3g}", (x, v), textcoords="offset points", xytext=(8, 0), va="center", fontsize=8, color=pal["ink"]
            )
    n = {t: len(group.get(t, [])) for t in TIMEPOINTS}
    ax.set_xticks(range(len(TIMEPOINTS)), [f"{t}\n(group n={n[t]})" for t in TIMEPOINTS], fontsize=8)
    ax.set_xlim(-0.6, len(TIMEPOINTS) - 0.4)
    ax.set_title(f"{panel['step']} · {panel['endpoint']}", fontsize=9, loc="left")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def render_animal_summary(job: dict[str, Any]) -> Figure:
    """Draw one animal's summary from a package job (`title`, `subtitle`, `panels`)."""
    pal = get_active_theme().colors()
    panels = job.get("panels", [])
    rows = max(1, math.ceil(len(panels) / COLUMNS))
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(4.2 * COLUMNS, 3.0 * rows + 0.9), squeeze=False)
    for ax, panel in zip(axes.flat, panels, strict=False):
        _panel(ax, panel, pal)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    if not panels:
        axes.flat[0].set_visible(True)
        axes.flat[0].set_axis_off()
        axes.flat[0].text(0.5, 0.5, "No recorded criteria endpoints", ha="center", va="center")
    fig.suptitle(
        f"{job.get('title', '')} · {job.get('subtitle', '')}", x=0.01, ha="left", fontsize=13, fontweight="bold"
    )
    fig.text(
        0.01,
        0.005,
        "Highlighted: this animal (pre → post). Faint points and boxes: its group's recorded "
        f"values. Summary v{ANIMAL_SUMMARY_VERSION}.",
        fontsize=8,
        color=pal["muted"],
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig
