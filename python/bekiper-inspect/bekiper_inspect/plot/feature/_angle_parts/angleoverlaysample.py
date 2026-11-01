"""Shared helpers for angle-based feature overlays."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

from bekiper_inspect.manifest import (
    FeatureAngle,
    FeatureHeuristic,
    InspectionManifest,
    PerimeterSpec,
)
from bekiper_inspect.plot.theme import PlotTheme

from .._helpers import (
    _metric_to_frame_pixels,
    _resolve_metric_perimeter,
)

ANGLE_MIN_DISPLAY_BIN_PX = 24.0
ANGLE_MAX_DISPLAY_BIN_PX = 72.0
ANGLE_TARGET_DISPLAY_BINS = 36.0
ANGLE_LABEL_SAFE_BIN_PX = 56.0
ANGLE_DENSE_MIN_SAMPLES = 18
ANGLE_DENSE_REMOVED_FRACTION = 0.25
ANGLE_HEATMAP_BINS = 40
ANGLE_HEURISTIC_VECTOR_PX = 24.0
ANGLE_LABEL_GAP_PX = 6.0
ANGLE_LABEL_SWEEP_DEG = 18.0
ANGLE_LABEL_FARTHER_OFFSET_PX = 12.0
ANGLE_LABEL_ESCAPE_OFFSET_PX = 18.0
ANGLE_LABEL_TICK_PX = 7.0
ANGLE_LABEL_TICK_GAP_PX = 1.5
ANGLE_LABEL_TICK_COLOR = "#b8b8b8"
DIAGNOSTIC_COLUMN_PREFIX = "__bk_diag__"
GEOMETRY_EPSILON = 1e-9


@dataclass(frozen=True)
class AngleOverlaySample:
    frame_index: int
    origin_xy: tuple[float, float]
    primary_vector: tuple[float, float]
    comparison_vector: tuple[float, float] | None
    rendered_angle_deg: float
    display_color: str
    display_origin_xy: tuple[float, float] | None = None
    label_anchor_xy: tuple[float, float] | None = None
    association_anchor_xy: tuple[float, float] | None = None
    show_association_tick: bool = False


@dataclass(frozen=True)
class AngleOverlayStats:
    representative_angle_deg: float
    median_angle_deg: float
    p10_angle_deg: float
    p90_angle_deg: float
    threshold_angle_deg: float | None = None
    median_margin_deg: float | None = None


@dataclass(frozen=True)
class AngleHeatmap:
    values: np.ndarray
    alpha: np.ndarray
    extent: tuple[float, float, float, float]


@dataclass(frozen=True)
class AngleOverlayData:
    raw_samples: tuple[AngleOverlaySample, ...]
    display_samples: tuple[AngleOverlaySample, ...]
    representative_sample: AngleOverlaySample
    stats: AngleOverlayStats
    heatmap: AngleHeatmap | None
    removed_fraction: float
    display_bin_px: float


@dataclass(frozen=True)
class AngleDiagnosticGroup:
    origin_label: str
    direction_x_col: str
    direction_y_col: str
    min_angle_col: str
    margin_col: str | None
    target_x_col: str
    target_y_col: str
    priority: int = 1


@dataclass(frozen=True)
class FallbackRayCandidate:
    origin_label: str
    direction_x_col: str
    direction_y_col: str
    priority: int = 1


def angle_quiver_kwargs(theme: PlotTheme) -> dict[str, object]:
    """Return the shared kwargs used for angle quiver overlays."""
    return angle_quiver_kwargs_with_width(theme, line_width=theme.linewidth)


def angle_quiver_kwargs_with_width(
    theme: PlotTheme,
    *,
    line_width: float,
) -> dict[str, object]:
    """Return shared quiver kwargs with width scaled from line width.

    Vector geometry stays in data space through ``angles='xy'`` and
    ``scale_units='xy'``, but shaft width stays in screen/axes space so
    dense overlays remain visible in rendered figures.
    """
    base_width = max(theme.linewidth, 0.1)
    width_scale = max(line_width, 0.1) / base_width
    return {
        **theme.quiver_kwargs(),
        "pivot": "tail",
        "scale": 1,
        "scale_units": "xy",
        "width": 0.002 * width_scale,
        "headwidth": 3 * width_scale,
        "headlength": 4 * width_scale,
        "headaxislength": 3.5 * width_scale,
        "linewidths": max(0.5, line_width * 0.25),
    }


def draw_angle_samples(
    ax: Axes,
    samples: Sequence[AngleOverlaySample],
    *,
    theme: PlotTheme,
    comparison_color: str | None = None,
    line_width: float | None = None,
    zorder: int = 4,
) -> None:
    """Draw one batched quiver collection per vector role."""
    from ._select_display_bin_px import _display_origin

    if not samples:
        return

    qkw = angle_quiver_kwargs_with_width(
        theme,
        line_width=line_width or theme.linewidth,
    )
    origins = np.array([_display_origin(sample) for sample in samples], dtype=float)
    origins_x = origins[:, 0]
    origins_y = origins[:, 1]
    primary_u = np.array([sample.primary_vector[0] for sample in samples], dtype=float)
    primary_v = np.array([sample.primary_vector[1] for sample in samples], dtype=float)
    primary_colors = [sample.display_color for sample in samples]
    ax.quiver(
        origins_x,
        origins_y,
        primary_u,
        primary_v,
        color=primary_colors,
        zorder=zorder,
        **qkw,
    )

    compare_samples = [sample for sample in samples if sample.comparison_vector is not None]
    if not compare_samples:
        return

    compare_u = np.array(
        [sample.comparison_vector[0] for sample in compare_samples if sample.comparison_vector is not None],
        dtype=float,
    )
    compare_v = np.array(
        [sample.comparison_vector[1] for sample in compare_samples if sample.comparison_vector is not None],
        dtype=float,
    )
    compare_origins = np.array(
        [_display_origin(sample) for sample in compare_samples],
        dtype=float,
    )
    compare_x = compare_origins[:, 0]
    compare_y = compare_origins[:, 1]
    ax.quiver(
        compare_x,
        compare_y,
        compare_u,
        compare_v,
        color=comparison_color or theme.video_point_color,
        zorder=zorder,
        **qkw,
    )


def annotate_angle_sample(
    ax: Axes,
    sample: AngleOverlaySample,
    *,
    theme: PlotTheme,
    color: str | None = None,
) -> None:
    """Annotate a single representative angle label."""
    from ._best_angle_diag_for_frame import _label_anchor

    label_x, label_y = sample.label_anchor_xy or _label_anchor(sample)
    ax.text(
        label_x,
        label_y,
        f"{sample.rendered_angle_deg:.1f}°",
        color=color or sample.display_color,
        fontsize=max(theme.annotation_fontsize - 6, 4.5),
        fontweight="bold",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.04",
            "facecolor": "black",
            "alpha": theme.bbox_alpha,
            "edgecolor": "none",
        },
        zorder=6,
    )


def draw_angle_pair_labels(
    ax: Axes,
    samples: Sequence[AngleOverlaySample],
    *,
    theme: PlotTheme,
    line_width: float | None = None,
    zorder: int = 6,
) -> tuple[AngleOverlaySample, ...]:
    """Render one degree label per displayed angle pair."""
    from ._select_display_bin_px import _layout_angle_pair_label

    if not samples:
        return ()

    ax.figure.canvas.draw()
    renderer = cast(Any, ax.figure.canvas).get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    occupied_bboxes: list[Bbox] = [
        text.get_window_extent(renderer=renderer).expanded(1.08, 1.2)
        for text in ax.texts
        if _is_pair_angle_label_text(text.get_text())
    ]
    laid_out: list[AngleOverlaySample] = []
    tick_width = max((line_width or theme.linewidth) * 0.45, 0.6)

    for sample in samples:
        placed_sample, placed_bbox = _layout_angle_pair_label(
            ax,
            sample,
            theme=theme,
            renderer=renderer,
            axes_bbox=axes_bbox,
            occupied_bboxes=occupied_bboxes,
        )
        laid_out.append(placed_sample)
        occupied_bboxes.append(placed_bbox)
        if placed_sample.show_association_tick:
            from ._select_display_bin_px import _draw_angle_label_tick

            _draw_angle_label_tick(
                ax,
                placed_sample,
                bbox=placed_bbox,
                line_width=tick_width,
                zorder=zorder - 0.1,
            )
        annotate_angle_sample(
            ax,
            placed_sample,
            theme=theme,
            color=placed_sample.display_color,
        )

    return tuple(laid_out)


def _is_pair_angle_label_text(text: str) -> bool:
    return text.endswith(" deg") or text.endswith("°")


def annotate_angle_summary(
    ax: Axes,
    data: AngleOverlayData,
    *,
    theme: PlotTheme,
) -> None:
    """Draw a compact numeric summary for the current angle window."""
    annotate_angle_summary_rows(
        ax,
        angle_summary_rows(data),
        theme=theme,
    )


def annotate_angle_summary_rows(
    ax: Axes,
    rows: Sequence[str],
    *,
    theme: PlotTheme,
) -> None:
    """Draw a compact numeric summary box from preformatted rows."""
    if not rows:
        return
    ax.text(
        0.02,
        0.02,
        "\n".join(rows),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=max(theme.fontsize - 1, 6),
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="black",
            alpha=min(1.0, theme.bbox_alpha + 0.1),
            edgecolor="white",
        ),
        zorder=6,
    )


def draw_angle_metric_footer(
    ax: Axes,
    *,
    left_rows: Sequence[str],
    right_rows: Sequence[str],
    theme: PlotTheme,
) -> None:
    """Render footer summaries below an angle panel."""
    ax.axis("off")
    ax.set_facecolor("none")
    if left_rows:
        ax.text(
            0.0,
            0.9,
            "\n".join(left_rows),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=theme.colors()["ink"],
            fontsize=max(theme.fontsize - 1, 6),
            family="monospace",
        )
    if right_rows:
        ax.text(
            1.0,
            0.9,
            "\n".join(right_rows),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=theme.colors()["ink"],
            fontsize=max(theme.fontsize - 1, 6),
            family="monospace",
        )


def angle_summary_rows(data: AngleOverlayData) -> list[str]:
    """Return the default per-overlay summary lines."""
    rows = [
        f"rep: {data.stats.representative_angle_deg:.1f}°",
        f"median: {data.stats.median_angle_deg:.1f}°",
        f"p10-p90: {data.stats.p10_angle_deg:.1f}°-{data.stats.p90_angle_deg:.1f}°",
    ]
    if data.stats.threshold_angle_deg is not None:
        rows.append(f"threshold: {data.stats.threshold_angle_deg:.1f}°")
    if data.stats.median_margin_deg is not None:
        if data.stats.median_margin_deg >= 0:
            rows.append(f"median margin: +{data.stats.median_margin_deg:.1f}°")
        else:
            rows.append(f"median miss: {abs(data.stats.median_margin_deg):.1f}°")
    return rows


def angle_summary_row(data: AngleOverlayData, *, label: str) -> str:
    """Return a single compact row for multi-object angle summaries."""
    return (
        f"{label}: rep {data.stats.representative_angle_deg:.1f}°, "
        f"med {data.stats.median_angle_deg:.1f}°, "
        f"p10-p90 {data.stats.p10_angle_deg:.1f}°-{data.stats.p90_angle_deg:.1f}°"
    )


def draw_angle_heatmap(
    ax: Axes,
    data: AngleOverlayData,
    *,
    theme: PlotTheme,
    title: str = "Median angle (°)",
    overlay_samples: Sequence[AngleOverlaySample] | None = None,
) -> None:
    """Draw the adaptive median-angle sidecar heatmap."""
    from ._render_angle_heatmap import _render_angle_heatmap

    if data.heatmap is None:
        return

    _render_angle_heatmap(
        ax,
        data.heatmap,
        theme=theme,
        title=title,
        overlay_samples=overlay_samples or data.display_samples,
    )


def draw_angle_heatmap_from_samples(
    ax: Axes,
    samples: Sequence[AngleOverlaySample],
    *,
    theme: PlotTheme,
    title: str = "Median angle (°)",
    overlay_samples: Sequence[AngleOverlaySample] | None = None,
) -> None:
    """Draw a heatmap directly from raw angle samples."""
    from ._best_angle_diag_for_frame import build_angle_heatmap, build_display_angle_samples
    from ._render_angle_heatmap import _render_angle_heatmap

    heatmap = build_angle_heatmap(samples)
    if heatmap is None:
        return
    display_samples, _ = build_display_angle_samples(samples)
    _render_angle_heatmap(
        ax,
        heatmap,
        theme=theme,
        title=title,
        overlay_samples=overlay_samples or display_samples,
    )


__all__ = [
    "ANGLE_DENSE_MIN_SAMPLES",
    "ANGLE_DENSE_REMOVED_FRACTION",
    "ANGLE_HEATMAP_BINS",
    "ANGLE_HEURISTIC_VECTOR_PX",
    "ANGLE_LABEL_ESCAPE_OFFSET_PX",
    "ANGLE_LABEL_FARTHER_OFFSET_PX",
    "ANGLE_LABEL_GAP_PX",
    "ANGLE_LABEL_SAFE_BIN_PX",
    "ANGLE_LABEL_SWEEP_DEG",
    "ANGLE_LABEL_TICK_COLOR",
    "ANGLE_LABEL_TICK_GAP_PX",
    "ANGLE_LABEL_TICK_PX",
    "ANGLE_MAX_DISPLAY_BIN_PX",
    "ANGLE_MIN_DISPLAY_BIN_PX",
    "ANGLE_TARGET_DISPLAY_BINS",
    "DIAGNOSTIC_COLUMN_PREFIX",
    "GEOMETRY_EPSILON",
    "AngleDiagnosticGroup",
    "AngleHeatmap",
    "AngleOverlayData",
    "AngleOverlaySample",
    "AngleOverlayStats",
    "Axes",
    "Bbox",
    "FallbackRayCandidate",
    "FeatureAngle",
    "FeatureHeuristic",
    "InspectionManifest",
    "PerimeterSpec",
    "PlotTheme",
    "Sequence",
    "_metric_to_frame_pixels",
    "_resolve_metric_perimeter",
    "angle_quiver_kwargs",
    "angle_quiver_kwargs_with_width",
    "angle_summary_row",
    "angle_summary_rows",
    "annotate_angle_sample",
    "annotate_angle_summary",
    "annotate_angle_summary_rows",
    "draw_angle_heatmap",
    "draw_angle_heatmap_from_samples",
    "draw_angle_metric_footer",
    "draw_angle_pair_labels",
    "draw_angle_samples",
    "np",
    "replace",
]
