from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bekiper_inspect.manifest import InspectionManifest
from bekiper_inspect.plot.generic import (
    ax_hue_plot_coordinate_with_boolean_index,
    boolean_index_truth_sequences,
)
from bekiper_inspect.plot.perimeter import plot_perimeter_on_ax
from bekiper_inspect.plot.theme import get_active_theme


def plot_heuristic_summary(
    manifest: InspectionManifest,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Bar chart summarizing all heuristic results (percentage of frames True)."""
    theme = get_active_theme()
    effective_figsize = figsize or theme.summary_figsize
    if not manifest.heuristics:
        fig, ax = plt.subplots(figsize=effective_figsize)
        ax.text(0.5, 0.5, "No heuristic results", ha="center", va="center", transform=ax.transAxes)
        return fig

    names = [h.name for h in manifest.heuristics]
    percentages = [h.percentage for h in manifest.heuristics]
    seconds = [h.seconds for h in manifest.heuristics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=effective_figsize)

    ax1.barh(names, percentages, color=theme.bar_primary_color)
    ax1.set_xlabel("% of frames")
    ax1.set_title("Heuristic Results (percentage)")
    ax1.set_xlim(0, 100)

    ax2.barh(names, seconds, color=theme.bar_secondary_color)
    ax2.set_xlabel("Seconds")
    ax2.set_title("Heuristic Results (duration)")

    fig.tight_layout()
    return fig


def draw_timeline_on_ax(
    ax: Axes,
    manifest: InspectionManifest,
    heuristic_name: str,
    *,
    color: str | None = None,
    label: str | None = None,
    y_offset: float = 0.0,
    row_height: float = 1.0,
    show_stats: bool = True,
) -> None:
    """Draw a heuristic timeline as colored spans on an existing axes.

    Parameters
    ----------
    y_offset, row_height
        When stacking multiple heuristics on one axes, each row occupies
        ``[y_offset, y_offset + row_height]``.  The caller increments
        ``y_offset`` by ``row_height`` for each heuristic.
    """
    theme = get_active_theme()
    boolean_index = manifest.get_boolean_index(heuristic_name)
    fps = manifest.video.fps
    resolved_color = color or theme.bar_primary_color

    # Prefer pre-computed truth_spans from Rust when available.
    meta = next((h for h in manifest.heuristics if h.name == heuristic_name), None)
    if meta is not None and meta.truth_spans:
        sequences = meta.truth_spans
    else:
        sequences = boolean_index_truth_sequences(boolean_index)
    # All spans use a uniform color at low alpha so that regions where
    # multiple spans overlap appear darker through additive blending,
    # making temporal density immediately visible.
    for start, end in sequences:
        t_start = start / fps
        t_end = end / fps
        ax.axhspan(
            y_offset,
            y_offset + row_height,
            xmin=t_start / (len(boolean_index) / fps) if fps else 0,
            xmax=t_end / (len(boolean_index) / fps) if fps else 0,
            alpha=0.2,
            color=resolved_color,
        )

    total_seconds = len(boolean_index) / fps if fps else 0
    ax.set_xlim(0, total_seconds)

    if show_stats:
        meta = next((h for h in manifest.heuristics if h.name == heuristic_name), None)
        if meta is not None:
            display_label = label or heuristic_name
            ax.text(
                0.99,
                y_offset + row_height * 0.5,
                f"{display_label}: {meta.seconds:.1f}s ({meta.percentage:.1f}%)",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=max(theme.fontsize - 1, 6),
                color=theme.hint_color,
            )


def plot_heuristic_timeline(
    manifest: InspectionManifest,
    heuristic_name: str,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Timeline plot showing when a heuristic is True across frames."""
    theme = get_active_theme()
    effective_figsize = figsize or theme.timeline_figsize

    fig, ax = plt.subplots(figsize=effective_figsize)
    draw_timeline_on_ax(ax, manifest, heuristic_name, show_stats=False)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{heuristic_name} — timeline")

    meta = next(h for h in manifest.heuristics if h.name == heuristic_name)
    ax.text(
        0.99,
        0.95,
        f"{meta.seconds:.1f}s ({meta.percentage:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=theme.fontsize,
        color=theme.hint_color,
    )

    fig.tight_layout()
    return fig


def plot_heuristic_on_coordinates(
    manifest: InspectionManifest,
    heuristic_name: str,
    label: str,
    perimeter_label: str | None = None,
    plot_non_confinement: bool = True,
    plot_line: bool = False,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot coordinates color-coded by a heuristic boolean result, with optional perimeter overlay."""
    theme = get_active_theme()
    effective_figsize = figsize or theme.coordinates_figsize
    boolean_index = manifest.get_boolean_index(heuristic_name)
    coordinates = manifest.get_coordinates(label)

    fig, ax = plt.subplots(figsize=effective_figsize)
    ax_hue_plot_coordinate_with_boolean_index(
        ax,
        boolean_index,
        coordinates,
        plot_non_confinement=plot_non_confinement,
        plot_line=plot_line,
    )

    if perimeter_label:
        p = manifest.get_perimeter(perimeter_label)
        plot_perimeter_on_ax(ax, p, meters_per_pixel=manifest.settings.meters_per_pixel)
    else:
        for p in manifest.perimeters:
            plot_perimeter_on_ax(ax, p, meters_per_pixel=manifest.settings.meters_per_pixel)

    ax.set_title(f"{heuristic_name} — {label}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
