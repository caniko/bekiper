"""Disjoint native state quantities, with an explicit whole-session denominator."""

from __future__ import annotations

import math
import textwrap

import numpy as np
from matplotlib import pyplot as plt

from bekiper_inspect.plot.color import identity_color
from bekiper_inspect.plot.theme import get_active_theme


def draw_state_distribution(ax, feature, label_colors=None):
    labels = feature.category_labels
    seconds = np.asarray(feature.seconds, dtype=float)
    percentages = np.asarray(feature.percentages, dtype=float)
    if (
        not len(labels) == len(seconds) == len(percentages) == len(feature.frame_counts)
        or not np.isfinite(seconds).all()
        or (seconds < 0).any()
        or not np.isfinite(percentages).all()
        or (percentages < 0).any()
        or not math.isfinite(feature.denominator_seconds)
        or feature.denominator_seconds < 0
        or not math.isclose(seconds.sum(), feature.denominator_seconds, abs_tol=1e-8)
        or sum(feature.frame_counts) != feature.denominator_frames
        or (
            feature.denominator_seconds > 0
            and not np.allclose(percentages, 100 * seconds / feature.denominator_seconds)
        )
        or (feature.denominator_seconds == 0 and (percentages != 0).any())
    ):
        raise ValueError("Native state distribution/denominator mismatch")
    colors = [identity_color(label, label_colors) for label in labels]
    legend_labels = [
        f"{label}: {value:.2f} s ({percent:.1f}%)"
        for label, value, percent in zip(labels, seconds, percentages, strict=True)
    ]
    if feature.denominator_seconds <= 0:
        ax.text(0.5, 0.5, "No recorded session duration", transform=ax.transAxes, ha="center")
    elif feature.style == "pie":
        ax.pie(
            seconds,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 0.8},
        )
    elif feature.style == "stacked":
        left = 0.0
        for index, (value, color) in enumerate(zip(percentages, colors, strict=True)):
            ax.barh(0, value, left=left, color=color, label=legend_labels[index], edgecolor="white")
            left += value
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Whole-session duration (%)")
    else:
        raise ValueError(f"Unknown state distribution style: {feature.style}")
    if feature.style == "pie":
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, legend_labels, strict=True)]
    else:
        handles = ax.get_legend_handles_labels()[0]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=8,
        frameon=False,
        ncol=2 if len(labels) > 5 else 1,
    )
    description = "\n".join(textwrap.fill(line, width=48) for line in feature.description.splitlines())
    ax.set_title(
        f"{description}\nDenominator: {feature.denominator_seconds:.2f} s · {feature.denominator_frames} frames",
        fontsize=get_active_theme().fontsize,
    )


def plot_state_distributions(features, label_colors=None):
    columns = min(3, len(features))
    rows = math.ceil(len(features) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 5.7 * rows), squeeze=False, layout="constrained")
    for ax, feature in zip(axes.flat, features, strict=False):
        draw_state_distribution(ax, feature, label_colors)
    for ax in list(axes.flat)[len(features) :]:
        ax.set_visible(False)
    fig.suptitle("Native descriptive state composition", fontsize=get_active_theme().title_fontsize)
    fig.supxlabel(features[0].basis, fontsize=8)
    return fig
