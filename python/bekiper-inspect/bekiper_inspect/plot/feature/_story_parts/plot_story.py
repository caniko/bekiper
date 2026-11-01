"""Multi-panel story renderer for grouped FeatureInspection entries.

Each :class:`StoryGroup` is rendered as a single figure with subplots,
typically one panel per heuristic-on-frame plus a shared timeline strip.
"""

from __future__ import annotations
from .._diagnostic import bound_ray_components, metric_axes

import math
from collections.abc import Sequence

import cv2
import numpy as np
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
    FeatureInspection,
    FeatureNortDiscrimination,
    FeaturePerimeter,
    FeatureProximity,
    FeatureYMazeArmSections,
    FeatureZone,
    InspectionManifest,
)
from bekiper_inspect.plot.frame import read_frame_at
from bekiper_inspect.plot.generic import plot_gradient_line
from bekiper_inspect.plot.heuristic import draw_timeline_on_ax
from bekiper_inspect.plot.perimeter import plot_perimeter_on_ax
from bekiper_inspect.plot.theme import active_theme_for_figure, get_active_theme

from .._angle import (
    AngleOverlayData,
    AngleOverlaySample,
    angle_summary_row,
    build_display_angle_samples,
    draw_angle_heatmap_from_samples,
    draw_angle_metric_footer,
)
from .._grouping import TASK_STORY_KEYS, StoryGroup
from .._helpers import (
    _add_boolean_summary,
    _figsize_from_frame,
    _metric_to_frame_pixels,
    _pick_frame_for_feature,
    _resolve_linewidth,
)
from .._renderers import (
    _angle_object_perimeter_linewidth,
    _build_angle_heuristic_data,
    _build_heuristic_decomposition_panels,
    _draw_angle_footer,
    _draw_angle_overlay,
    _is_angle_heuristic,
    _plot_angle_feature,
    _plot_angle_heuristic_feature,
    _plot_angle_heuristic_sidecar,
    _plot_arm_sequence_feature,
    _plot_center_periphery_feature,
    _plot_heuristic_cloud_panel,
    _plot_heuristic_feature,
    _plot_heuristic_trajectory_only,
    _plot_nort_discrimination_feature,
    _plot_perimeter_feature,
    _plot_proximity_feature,
    _plot_ymaze_arm_sections_feature,
    _plot_zone_feature,
)

# Distinct colors for stacked timeline rows and distinguishing heuristics.
_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def plot_story(
    manifest: InspectionManifest,
    group: StoryGroup,
) -> Figure | None:
    """Render a multi-panel figure for a :class:`StoryGroup`.

    Layout depends on the story type:

    * **Perimeter-anchored** (multiple heuristics about one target):
      top row = one panel per heuristic on frame, bottom row = stacked
      timeline.
    * **Standalone heuristic** (freezing, in_center): frame panels +
      timeline.
    * **Geometry**: single frame with all perimeters + angle overlaid.
    * **Task-specific** (NORT, arm_sequence, center_periphery): delegate
      to existing single-panel renderer.

    Story figures route through the profile system with synthetic
    ``kind="story"`` and ``label=group.key`` so users can target e.g.
    ``label = "in_distal_arm_*"`` or ``kind = "story"`` in
    ``[[inspect.plot.figure_profiles]]``.
    """
    with active_theme_for_figure("story", group.key) as theme:
        fig = _plot_story_inner(manifest, group)
        if fig is not None:
            fig._bekiper_plot_theme = theme
        return fig


def _plot_story_inner(
    manifest: InspectionManifest,
    group: StoryGroup,
) -> Figure | None:
    # Resolve a video frame for spatial panels.
    if group.features and all(feature.kind == "state_distribution" for feature in group.features):
        from .._state_distribution import plot_state_distributions

        return plot_state_distributions(group.features, manifest.label_colors)
    if group.key.startswith("geometry_vectors_"):
        from .._vector_examples import plot_vector_examples

        return plot_vector_examples(manifest, group.features)
    if group.features and all(feature.kind == "time_series" for feature in group.features):
        from ._plot_objects_story import _plot_fallback

        return _plot_fallback(manifest, group, None)
    frame = _resolve_frame(manifest, group)

    if group.key == "geometry":
        from ._plot_objects_story import _plot_geometry_story

        return _plot_geometry_story(manifest, group, frame)
    if group.key == "preference":
        from .._preference import plot_preference

        return plot_preference(group.features[0], manifest.label_colors)
    if group.key == "geometry_examples" or group.key.startswith("tracking_") or group.key == "pose_quality":
        from ._plot_objects_story import _plot_fallback

        return _plot_fallback(manifest, group, frame)

    if group.key == "objects":
        from ._plot_objects_story import _plot_objects_story

        return _plot_objects_story(manifest, group, frame)

    if group.key.startswith("object_decomposition_"):
        from ._plot_object_decomposition_story import _plot_object_decomposition_story

        return _plot_object_decomposition_story(manifest, group)

    if group.key in TASK_STORY_KEYS.values():
        from ._plot_objects_story import _plot_task_story

        return _plot_task_story(manifest, group, frame)

    # Count heuristic features (Heuristic, HeuristicCloud, Zone, Proximity) — these get
    # one spatial panel each.  Non-spatial features (Perimeter, Angle)
    # are overlaid on the first spatial panel.
    spatial_features = [
        f
        for f in group.features
        if isinstance(f, (FeatureHeuristic, FeatureHeuristicCloud, FeatureZone, FeatureProximity))
    ]

    if not spatial_features:
        # No spatial panels — fall back to single-panel for each feature.
        from ._plot_objects_story import _plot_fallback

        return _plot_fallback(manifest, group, frame)

    has_timelines = bool(group.heuristic_names)
    n_spatial = len(spatial_features)

    return _plot_heuristic_story(manifest, group, frame, spatial_features, n_spatial, has_timelines)


def _resolve_frame(manifest: InspectionManifest, group: StoryGroup) -> np.ndarray | None:
    """Load a video frame for the story, preferring the first feature's pick."""
    if group.features:
        frame, _ = _resolve_frame_for_feature(manifest, group.features[0])
        return frame
    return None


def _resolve_frame_for_feature(
    manifest: InspectionManifest,
    feature: FeatureInspection,
) -> tuple[np.ndarray | None, int]:
    """Load the preferred frame for a single feature."""
    frame_idx = _pick_frame_for_feature(manifest, feature)
    frame: np.ndarray | None = None

    frame = read_frame_at(manifest.video.path, manifest.source_frame_index(frame_idx), frames_dir=manifest.frames_dir)

    if frame is None and manifest.frame_image:
        img = cv2.imread(manifest.frame_image, cv2.IMREAD_COLOR)
        if img is not None:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if frame is None:
        w, h = manifest.video.resolution
        if w > 0 and h > 0:
            frame = np.full((h, w, 3), 40, dtype=np.uint8)

    return frame, frame_idx


# ── Layout: heuristic story (per-object / per-arena / standalone) ─────


def _plot_heuristic_story(
    manifest: InspectionManifest,
    group: StoryGroup,
    frame: np.ndarray | None,
    spatial_features: Sequence[FeatureInspection],
    n_spatial: int,
    has_timelines: bool,
) -> Figure | None:
    from ._plot_objects_story import _set_panel_title, _short_heuristic_label

    if frame is None:
        return None

    theme = get_active_theme()

    # Layout: classic trajectory heuristics get valid/rejected pairs;
    # cloud heuristics consume a single wider panel. Bottom row is the
    # shared timeline strip.
    families_per_row = min(2, n_spatial)
    spatial_rows = math.ceil(n_spatial / families_per_row)
    timeline_height = max(1.0, len(group.heuristic_names) * 0.22)
    panel_height = 4.7 * frame.shape[0] / frame.shape[1] + 1.6
    height_ratios = [panel_height] * spatial_rows + ([timeline_height] if has_timelines else [])
    n_rows = spatial_rows + int(has_timelines)
    fig_w = families_per_row * 10.0
    fig_h = sum(height_ratios) + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
    gs = GridSpec(
        n_rows,
        families_per_row,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.15,
        wspace=0.05,
    )

    # ── Spatial panels ───────────────────────────────────────────────
    fs = frame.shape
    obj_idx = _object_index_for_group(manifest, group)
    for index, feat in enumerate(spatial_features):
        row, column = divmod(index, families_per_row)
        slots = gs[row, column].subgridspec(
            1, len(_story_feature_width_ratios(feat)), width_ratios=_story_feature_width_ratios(feat)
        )
        fidx = _pick_frame_for_feature(manifest, feat)
        if isinstance(feat, FeatureHeuristicCloud):
            ax_cloud = fig.add_subplot(slots[0, 0])
            ax_cloud.imshow(frame, aspect="equal")
            _plot_heuristic_cloud_panel(
                ax_cloud,
                manifest,
                feat,
                frame_idx=fidx,
                frame_shape=fs,
            )
            _set_panel_title(ax_cloud, feat, theme)
            ax_cloud.set_xticks([])
            ax_cloud.set_yticks([])
            continue

        angle_data = None
        native_rays = bound_ray_components(manifest, feat.result_column) if isinstance(feat, FeatureHeuristic) else []
        if isinstance(feat, FeatureHeuristic) and _is_angle_heuristic(feat) and not native_rays:
            angle_data = _build_angle_heuristic_data(
                manifest,
                feat,
                fidx,
                fs,
            )

        valid_slot = slots[0, 0]
        ax_sidecar = None
        footer_ax = None
        if angle_data is not None:
            valid_grid = valid_slot.subgridspec(
                2,
                1,
                height_ratios=[4.0, 0.72],
                hspace=0.02,
            )
            valid_visual = valid_grid[0, 0]
            footer_ax = fig.add_subplot(valid_grid[1, 0])
            if angle_data.heatmap is not None:
                visual_grid = valid_visual.subgridspec(
                    1,
                    2,
                    width_ratios=[3, 2],
                    wspace=0.05,
                )
                ax_valid = fig.add_subplot(visual_grid[0, 0])
                ax_sidecar = fig.add_subplot(visual_grid[0, 1])
            else:
                ax_valid = fig.add_subplot(valid_visual)
        else:
            ax_valid = fig.add_subplot(valid_slot)

        ax_valid.imshow(frame, aspect="equal")
        if isinstance(feat, FeatureHeuristic):
            if angle_data is not None:
                _plot_angle_heuristic_feature(
                    ax_valid,
                    manifest,
                    feat,
                    fidx,
                    frame_shape=fs,
                    data=angle_data,
                )
                if ax_sidecar is not None:
                    _plot_angle_heuristic_sidecar(
                        ax_sidecar,
                        manifest,
                        feat,
                        angle_data,
                        frame_shape=fs,
                    )
                if footer_ax is not None:
                    _draw_angle_footer(
                        footer_ax,
                        manifest,
                        feat.result_column,
                        angle_data,
                    )
            else:
                _plot_heuristic_feature(
                    ax_valid,
                    manifest,
                    feat,
                    frame_shape=fs,
                    object_index=obj_idx,
                    n_objects=1,
                    frame_idx=fidx,
                    valid_only=True,
                )
        elif isinstance(feat, FeatureZone):
            _plot_zone_feature(ax_valid, manifest, feat, frame_shape=fs)
        elif isinstance(feat, FeatureProximity):
            _plot_proximity_feature(ax_valid, manifest, feat, frame_shape=fs)
        _set_panel_title(ax_valid, feat, theme)
        if angle_data is None:
            metric_axes(ax_valid, manifest, fs)
        else:
            ax_valid.set_xticks([])
            ax_valid.set_yticks([])
        if ax_sidecar is not None:
            ax_sidecar.set_xticks([])
            ax_sidecar.set_yticks([])

        # Rejected panel (right of pair).
        ax_reject = fig.add_subplot(slots[0, 1])
        ax_reject.imshow(frame, aspect="equal")
        if isinstance(feat, FeatureHeuristic):
            _plot_heuristic_feature(
                ax_reject,
                manifest,
                feat,
                frame_shape=fs,
                object_index=obj_idx,
                n_objects=1,
                frame_idx=fidx,
                valid_only=False,
            )
        elif isinstance(feat, FeatureZone):
            _plot_zone_feature(ax_reject, manifest, feat, frame_shape=fs)
        elif isinstance(feat, FeatureProximity):
            _plot_proximity_feature(ax_reject, manifest, feat, frame_shape=fs)
        ax_reject.set_title("rejected", fontsize=getattr(theme, "annotation_fontsize", 10), pad=4)
        if angle_data is None:
            metric_axes(ax_reject, manifest, fs)
        else:
            ax_reject.set_xticks([])
            ax_reject.set_yticks([])

    # ── Timeline strip ───────────────────────────────────────────────
    if has_timelines:
        ax_tl = fig.add_subplot(gs[-1, :])
        n_heur = len(group.heuristic_names)
        for j, hname in enumerate(group.heuristic_names):
            draw_timeline_on_ax(
                ax_tl,
                manifest,
                hname,
                label=_short_heuristic_label(hname, group.key),
                y_offset=j,
                row_height=1.0,
                show_stats=True,
            )
        ax_tl.set_ylim(0, max(n_heur, 1))
        ax_tl.set_yticks([])
        ax_tl.set_xlabel("Time (s)", fontsize=theme.fontsize)

    fig.suptitle(group.title, fontsize=theme.title_fontsize)
    return fig


def _story_feature_width_ratios(feature: FeatureInspection) -> list[int]:
    """Return GridSpec width ratios for one spatial feature."""
    if isinstance(feature, FeatureHeuristicCloud):
        return [2]
    return [1, 1]


def _object_index_for_group(manifest: InspectionManifest, group: StoryGroup) -> int:
    """Return a stable integer index for the group's target perimeter.

    Used to pick per-object gradient colors. Returns 0 for groups
    without a perimeter (standalone heuristics).
    """
    if not group.perimeter_label:
        return 0
    non_arena = [p.label for p in manifest.perimeters if p.label != "arena"]
    try:
        return non_arena.index(group.perimeter_label)
    except ValueError:
        return 0


def _heuristic_target_label(name: str, perim_labels: list[str]) -> str | None:
    """Extract the target suffix from a heuristic name."""
    for plabel in sorted(perim_labels, key=len, reverse=True):
        if name.endswith(f"_{plabel}"):
            return plabel
        if name == f"in_{plabel}":
            return plabel
    return None


def _humanize_story_label(label: str) -> str:
    return label.replace("_", " ").title()


__all__ = [
    "_PALETTE",
    "AngleOverlayData",
    "AngleOverlaySample",
    "Axes",
    "FeatureAngle",
    "FeatureArmSequence",
    "FeatureCenterPeriphery",
    "FeatureHeuristic",
    "FeatureHeuristicCloud",
    "FeatureHeuristicDecomposition",
    "FeatureInspection",
    "FeatureNortDiscrimination",
    "FeaturePerimeter",
    "FeatureProximity",
    "FeatureYMazeArmSections",
    "FeatureZone",
    "Figure",
    "GridSpec",
    "InspectionManifest",
    "Sequence",
    "StoryGroup",
    "_add_boolean_summary",
    "_angle_object_perimeter_linewidth",
    "_build_angle_heuristic_data",
    "_build_heuristic_decomposition_panels",
    "_draw_angle_footer",
    "_draw_angle_overlay",
    "_figsize_from_frame",
    "_heuristic_target_label",
    "_humanize_story_label",
    "_is_angle_heuristic",
    "_metric_to_frame_pixels",
    "_object_index_for_group",
    "_pick_frame_for_feature",
    "_plot_angle_feature",
    "_plot_angle_heuristic_feature",
    "_plot_angle_heuristic_sidecar",
    "_plot_arm_sequence_feature",
    "_plot_center_periphery_feature",
    "_plot_heuristic_cloud_panel",
    "_plot_heuristic_feature",
    "_plot_heuristic_story",
    "_plot_heuristic_trajectory_only",
    "_plot_nort_discrimination_feature",
    "_plot_perimeter_feature",
    "_plot_proximity_feature",
    "_plot_ymaze_arm_sections_feature",
    "_plot_zone_feature",
    "_resolve_frame",
    "_resolve_frame_for_feature",
    "_resolve_linewidth",
    "_story_feature_width_ratios",
    "angle_summary_row",
    "build_display_angle_samples",
    "draw_angle_heatmap_from_samples",
    "draw_angle_metric_footer",
    "draw_timeline_on_ax",
    "get_active_theme",
    "math",
    "np",
    "plot_gradient_line",
    "plot_perimeter_on_ax",
    "plot_story",
    "plt",
]
