"""Per-kind renderer functions for feature inspection figures."""

from __future__ import annotations
from bekiper_inspect.plot.feature._helpers import _calibration_resolution

import math
from collections.abc import Sequence

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from bekiper_inspect.manifest import (
    FeatureAngle,
    FeatureArmSequence,
    FeatureCenterPeriphery,
    FeatureHeuristic,
    FeatureHeuristicCloud,
    FeatureHeuristicDecomposition,
    FeatureNortDiscrimination,
    FeaturePerimeter,
    FeatureProximity,
    FeatureQuadrant,
    FeatureRayCast,
    FeatureScaleDerivation,
    FeatureTimeSeries,
    FeatureTrajectory,
    FeatureYMazeArmSections,
    FeatureZone,
    InspectionManifest,
    PerimeterSpec,
)
from bekiper_inspect.plot.generic import (
    ax_hue_plot_coordinate_with_boolean_index,
    boolean_index_truth_sequences,
    filter_coordinates_by_speed,
    plot_coordinates,
    plot_gradient_line,
    plot_occupancy_heatmap,
    plot_state_colored_trajectory,
    plot_temporal_shaded_line,
    plot_velocity_field,
)
from bekiper_inspect.plot.heuristic import draw_timeline_on_ax
from bekiper_inspect.plot.perimeter import plot_perimeter_on_ax, plot_pixel_perimeter_on_ax
from bekiper_inspect.plot.scale import (
    _annotate_arena_side,
    _annotate_arm_length,
    _find_perimeter,
    _scale_nested,
)
from bekiper_inspect.plot.theme import PlotTheme, get_active_theme

from .._angle import (
    AngleOverlayData,
    angle_summary_row,
    angle_summary_rows,
    annotate_angle_sample,
    annotate_angle_summary_rows,
    build_feature_angle_sample,
    build_heuristic_angle_overlay,
    draw_angle_heatmap,
    draw_angle_metric_footer,
    draw_angle_pair_labels,
    draw_angle_samples,
)
from .._helpers import (
    _add_boolean_summary,
    _base_perimeter_label,
    _boolean_summary_rows,
    _figsize_from_frame,
    _find_heuristic_meta,
    _frame_scale,
    _metric_to_frame_pixels,
    _metric_to_pixels,
    _primary_coordinate_label,
    _resolve_color,
    _resolve_linewidth,
    _resolve_metric_perimeter,
    _resolve_plot_perimeter,
)

_INTERACTION_CONTEXT_TAGS = (
    "body_proximity",
    "olfaction",
    "whisker_interaction",
)


def _is_physical_interaction_target_label(perimeter_label: str) -> bool:
    base_label = _base_perimeter_label(perimeter_label)
    return base_label.startswith("object_") or base_label in {"novel", "familiar"}


def _draw_perimeter_overlay(
    ax: Axes,
    manifest: InspectionManifest,
    perimeter_label: str,
    color: str,
    frame_shape: tuple[int, ...] | None = None,
    linewidth: float | None = None,
) -> None:
    """Draw a perimeter on an axis, preferring pixel-space geometry.

    Uses the manifest's ``pixel_perimeters`` when available (pre-computed
    by the Rust ``ToVideoPixels`` trait), falling back to the legacy
    metric-space conversion path for old manifests. This ensures every
    element on the inspection figure goes through the same scaling.
    """
    perimeter, is_pixel = _resolve_plot_perimeter(manifest, perimeter_label)
    if perimeter is None:
        msg = f"Unknown perimeter label {perimeter_label!r}"
        raise StopIteration(msg)
    if is_pixel:
        plot_pixel_perimeter_on_ax(
            ax,
            perimeter,
            color=color,
            video_resolution=manifest.video.resolution,
            frame_shape=frame_shape,
            linewidth=linewidth,
        )
    else:
        plot_perimeter_on_ax(
            ax,
            perimeter,
            color=color,
            meters_per_pixel=manifest.settings.meters_per_pixel,
            as_pixels=True,
            video_resolution=_calibration_resolution(manifest),
            frame_shape=frame_shape,
            linewidth=linewidth,
        )


def _radial_maze_display_geometry(
    manifest: InspectionManifest,
    perimeter_label: str,
    frame_shape: tuple[int, ...] | None,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    perimeter, is_pixel = _resolve_plot_perimeter(manifest, perimeter_label)
    if perimeter is None or perimeter.shape != "radial_maze":
        return None

    scale = (
        1.0
        if is_pixel
        else (1.0 / manifest.settings.meters_per_pixel if manifest.settings.meters_per_pixel > 0 else 1.0)
    )
    sx, sy = _frame_scale(_calibration_resolution(manifest), frame_shape or (0, 0))

    def transform(points: list[list[float]] | list[tuple[float, float]]) -> np.ndarray:
        arr = np.asarray(points, dtype=float) * scale
        if arr.ndim != 2 or arr.shape[1] != 2:
            return np.empty((0, 2), dtype=float)
        arr[:, 0] *= sx
        arr[:, 1] *= sy
        return arr

    center_vertices = transform(perimeter.params.get("center_vertices", []))
    arms = [transform(arm) for arm in perimeter.params.get("arms", [])]
    return center_vertices, [arm for arm in arms if arm.size]


def _draw_threshold_segment(ax: Axes, p1: np.ndarray, p2: np.ndarray, label: str, theme: PlotTheme) -> None:
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color=theme.video_false_color,
        linewidth=max(theme.linewidth, 1.5),
        linestyle="-",
        alpha=0.95,
        zorder=5,
    )
    midpoint = (np.asarray(p1) + np.asarray(p2)) / 2.0
    ax.text(
        midpoint[0],
        midpoint[1],
        label,
        color="white",
        fontsize=max(6, theme.annotation_fontsize - 3),
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor="black",
            alpha=theme.bbox_alpha,
            edgecolor="none",
        ),
        zorder=6,
    )


def _draw_ymaze_arm_sections(
    ax: Axes,
    manifest: InspectionManifest,
    perimeter_label: str,
    proximal_threshold: float,
    frame_shape: tuple[int, ...] | None = None,
    threshold_lines: Sequence[tuple[str, tuple[float, float], tuple[float, float]]] = (),
) -> None:
    if threshold_lines:
        # Scored arm-start cross-lines (metres): draw exactly those segments.
        theme = get_active_theme()
        for _arm, p1, p2 in threshold_lines:
            ends = _metric_to_frame_pixels(
                np.array([p1, p2], dtype=float),
                manifest.settings.meters_per_pixel,
                _calibration_resolution(manifest),
                frame_shape or (0, 0),
            )
            _draw_threshold_segment(ax, ends[0], ends[1], f"{proximal_threshold * 100:.0f}%", theme)
        return
    geometry = _radial_maze_display_geometry(manifest, perimeter_label, frame_shape)
    if geometry is None:
        return
    center_vertices, arms = geometry
    if center_vertices.size == 0 or not arms:
        return

    theme = get_active_theme()
    center = center_vertices.mean(axis=0)
    colors = [theme.video_false_color, theme.video_true_color]
    labels = ["center", "arm"]

    for arm_index, arm in enumerate(arms):
        centroid = arm.mean(axis=0)
        axis = centroid - center
        norm = float(np.linalg.norm(axis))
        if norm <= 0:
            continue
        axis = axis / norm
        perp = np.array([-axis[1], axis[0]])
        rel = arm - center
        depths = rel @ axis
        widths = rel @ perp
        near = float(np.min(depths))
        far = float(np.max(depths))
        if not np.isfinite(near) or not np.isfinite(far) or far <= near:
            continue
        left = float(np.min(widths))
        right = float(np.max(widths))

        prox = near + proximal_threshold * (far - near)
        bounds = [near, prox, far]

        for lo, hi, color in zip(bounds[:-1], bounds[1:], colors):
            corners = np.array(
                [
                    center + axis * lo + perp * left,
                    center + axis * lo + perp * right,
                    center + axis * hi + perp * right,
                    center + axis * hi + perp * left,
                ]
            )
            ax.add_patch(
                patches.Polygon(
                    corners,
                    closed=True,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.11,
                    zorder=2,
                )
            )

        for threshold, label in ((prox, f"{proximal_threshold * 100:.0f}%"),):
            p1 = center + axis * threshold + perp * left
            p2 = center + axis * threshold + perp * right
            _draw_threshold_segment(ax, p1, p2, label, theme)

        label_depth = [(bounds[i] + bounds[i + 1]) / 2.0 for i in range(len(bounds) - 1)]
        for depth, label, color in zip(label_depth, labels, colors):
            pos = center + axis * depth + perp * ((left + right) / 2.0)
            ax.text(
                pos[0],
                pos[1] + (arm_index % 2) * 8,
                label,
                color=color,
                fontsize=max(6, theme.annotation_fontsize - 4),
                ha="center",
                va="center",
                alpha=0.9,
                zorder=6,
            )


def _angle_object_perimeter_linewidth(
    perimeter_label: str | None,
    *,
    theme: PlotTheme,
) -> float | None:
    if not perimeter_label or not _is_physical_interaction_target_label(perimeter_label):
        return None
    return max(theme.linewidth * 0.6, 0.9)


def _plot_perimeter_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeaturePerimeter,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    _draw_perimeter_overlay(
        ax,
        manifest,
        feature.perimeter_label,
        color=_resolve_color(
            feature, manifest.label_colors.get(feature.perimeter_label, get_active_theme().video_perimeter_color)
        ),
        frame_shape=frame_shape,
    )


def _plot_zone_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureZone,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    _draw_perimeter_overlay(
        ax,
        manifest,
        feature.perimeter_label,
        color=get_active_theme().video_perimeter_color,
        frame_shape=frame_shape,
    )
    coord_label = _primary_coordinate_label(feature)
    if not coord_label or coord_label not in manifest.coordinate_columns:
        return
    coords_metric = manifest.get_coordinates(coord_label)
    coords_pixels = _metric_to_frame_pixels(
        coords_metric,
        manifest.settings.meters_per_pixel,
        _calibration_resolution(manifest),
        frame_shape or (0, 0),
    )
    if feature.result_column not in manifest.df.columns:
        return
    boolean_index = manifest.df[feature.result_column].to_numpy()
    ax_hue_plot_coordinate_with_boolean_index(
        ax,
        boolean_index,
        coords_pixels,
        plot_non_confinement=True,
        plot_line=True,
    )
    _add_boolean_summary(ax, manifest, feature.result_column)


def _plot_quadrant_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureQuadrant,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw an NxN grid on the frame, color cells by per-cell occupancy.

    Per-cell occupancy is computed on the fly from the primary
    coordinate trajectory: the fraction of frames whose coordinates
    fall inside that cell. Cells are filled with `bar_primary_color`
    at alpha proportional to occupancy.
    """
    try:
        spec = manifest.get_quadrant(feature.quadrant_label)
    except StopIteration:
        return
    theme = get_active_theme()
    mpp = manifest.settings.meters_per_pixel
    scale = 1.0 / mpp if mpp > 0 else 1.0
    sx, sy = _frame_scale(_calibration_resolution(manifest), frame_shape or (0, 0))

    cell_w = (spec.x_max - spec.x_min) / spec.cols
    cell_h = (spec.y_max - spec.y_min) / spec.rows

    quad_coord_label = _primary_coordinate_label(feature)
    coords_metric = (
        manifest.get_coordinates(quad_coord_label)
        if quad_coord_label and quad_coord_label in manifest.coordinate_columns
        else None
    )

    # Prefer pre-computed occupancy grid from Rust when available.
    if feature.occupancy_grid is not None:
        occupancy = np.array(feature.occupancy_grid, dtype=float)
    else:
        occupancy = np.zeros((spec.rows, spec.cols), dtype=float)
        if coords_metric is not None and coords_metric.size:
            for r in range(spec.rows):
                y0 = spec.y_min + r * cell_h
                y1 = y0 + cell_h
                for c in range(spec.cols):
                    x0 = spec.x_min + c * cell_w
                    x1 = x0 + cell_w
                    inside = (
                        (coords_metric[:, 0] >= x0)
                        & (coords_metric[:, 0] < x1)
                        & (coords_metric[:, 1] >= y0)
                        & (coords_metric[:, 1] < y1)
                    )
                    occupancy[r, c] = inside.mean() if inside.size else 0.0
            max_occ = float(occupancy.max())
            if max_occ > 0:
                occupancy = occupancy / max_occ

    resolved_color = _resolve_color(feature, theme.bar_primary_color)

    for r in range(spec.rows):
        for c in range(spec.cols):
            x0 = (spec.x_min + c * cell_w) * scale * sx
            y0 = (spec.y_min + r * cell_h) * scale * sy
            rect = patches.Rectangle(
                (x0, y0),
                cell_w * scale * sx,
                cell_h * scale * sy,
                facecolor=resolved_color,
                alpha=float(occupancy[r, c]) * 0.7,
                edgecolor=theme.video_perimeter_color,
                linewidth=1.0,
            )
            ax.add_patch(rect)

    # Trajectory overlay so the user can sanity-check the occupancy values.
    if coords_metric is not None and coords_metric.size:
        coords_pixels = _metric_to_frame_pixels(
            coords_metric,
            mpp,
            _calibration_resolution(manifest),
            frame_shape or (0, 0),
        )
        plot_coordinates(
            ax,
            coords_pixels,
            color=theme.video_point_color,
            linewidth=theme.linewidth,
            alpha=0.6,
        )


__all__ = [
    "_INTERACTION_CONTEXT_TAGS",
    "AngleOverlayData",
    "Axes",
    "FeatureAngle",
    "FeatureArmSequence",
    "FeatureCenterPeriphery",
    "FeatureHeuristic",
    "FeatureHeuristicCloud",
    "FeatureHeuristicDecomposition",
    "FeatureNortDiscrimination",
    "FeaturePerimeter",
    "FeatureProximity",
    "FeatureQuadrant",
    "FeatureRayCast",
    "FeatureScaleDerivation",
    "FeatureTimeSeries",
    "FeatureTrajectory",
    "FeatureYMazeArmSections",
    "FeatureZone",
    "Figure",
    "GridSpec",
    "InspectionManifest",
    "PerimeterSpec",
    "PlotTheme",
    "_add_boolean_summary",
    "_angle_object_perimeter_linewidth",
    "_annotate_arena_side",
    "_annotate_arm_length",
    "_base_perimeter_label",
    "_boolean_summary_rows",
    "_draw_perimeter_overlay",
    "_draw_ymaze_arm_sections",
    "_figsize_from_frame",
    "_find_heuristic_meta",
    "_find_perimeter",
    "_frame_scale",
    "_metric_to_frame_pixels",
    "_metric_to_pixels",
    "_plot_perimeter_feature",
    "_plot_quadrant_feature",
    "_plot_zone_feature",
    "_primary_coordinate_label",
    "_radial_maze_display_geometry",
    "_resolve_color",
    "_resolve_linewidth",
    "_resolve_metric_perimeter",
    "_resolve_plot_perimeter",
    "_scale_nested",
    "angle_summary_row",
    "angle_summary_rows",
    "annotate_angle_sample",
    "annotate_angle_summary_rows",
    "ax_hue_plot_coordinate_with_boolean_index",
    "boolean_index_truth_sequences",
    "build_feature_angle_sample",
    "build_heuristic_angle_overlay",
    "draw_angle_heatmap",
    "draw_angle_metric_footer",
    "draw_angle_pair_labels",
    "draw_angle_samples",
    "draw_timeline_on_ax",
    "filter_coordinates_by_speed",
    "get_active_theme",
    "math",
    "np",
    "patches",
    "plot_coordinates",
    "plot_gradient_line",
    "plot_occupancy_heatmap",
    "plot_perimeter_on_ax",
    "plot_pixel_perimeter_on_ax",
    "plot_state_colored_trajectory",
    "plot_temporal_shaded_line",
    "plot_velocity_field",
    "plt",
]
