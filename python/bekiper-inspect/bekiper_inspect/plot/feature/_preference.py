"""Object preference pies from native per-object measured durations."""

from __future__ import annotations

import math
import numpy as np
from matplotlib import pyplot as plt

from bekiper_inspect.plot.theme import get_active_theme
from bekiper_inspect.plot.color import identity_color


def plot_preference(feature, label_colors=None):
    channels = [("Exploration", feature.exploration_times)]
    channels.extend((c.heuristic_type, c.per_object_seconds) for c in feature.qualia_channels or [])
    channels.extend(sorted((feature.merged_channels or {}).items()))
    columns = min(3, len(channels))
    fig, axes = plt.subplots(
        math.ceil(len(channels) / columns),
        columns,
        figsize=(4.8 * columns, 4.5 * math.ceil(len(channels) / columns)),
        squeeze=False,
    )
    for ax, (label, times) in zip(axes.flat, channels, strict=False):
        times = np.asarray(times, dtype=float)
        if (
            len(times) == len(feature.object_labels)
            and np.isfinite(times).all()
            and (times >= 0).all()
            and times.sum() > 0
        ):
            ax.pie(
                times,
                labels=[f"{name}\n{seconds:.2f} s" for name, seconds in zip(feature.object_labels, times, strict=True)],
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
                colors=[identity_color(label, label_colors) for label in feature.object_labels],
            )
        else:
            ax.text(0.5, 0.5, "No measured exploration", ha="center", transform=ax.transAxes)
        ax.set_title(label)
    for ax in list(axes.flat)[len(channels) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Conditional object preference — {feature.protocol or 'recorded protocol'}\nDenominator: sum of measured object durations in each channel (overlap can count twice)",
        fontsize=get_active_theme().title_fontsize,
    )
    fig.tight_layout()
    return fig
