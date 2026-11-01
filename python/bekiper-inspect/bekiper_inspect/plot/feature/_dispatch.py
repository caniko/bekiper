"""Top-level dispatcher for feature inspection figures."""

from __future__ import annotations

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bekiper_inspect.manifest import (
    FeatureAngle,
    FeatureArmSequence,
    FeatureCenterPeriphery,
    FeatureDetectionFields,
    FeatureHeuristic,
    FeatureHeuristicCloud,
    FeatureHeuristicDecomposition,
    FeatureInspection,
    FeatureNortDiscrimination,
    FeaturePerimeter,
    FeatureProximity,
    FeatureQuadrant,
    FeatureRayCast,
    FeatureScaleDerivation,
    FeatureStateDistribution,
    FeatureTimeSeries,
    FeatureTrajectory,
    FeatureYMazeArmSections,
    FeatureZone,
    InspectionManifest,
)
from bekiper_inspect.plot.frame import read_frame_at
from bekiper_inspect.plot.theme import active_theme_for_figure, get_active_theme

from ._diagnostic import bound_ray_components
from ._helpers import _figsize_from_frame, _pick_frame_for_feature
from ._renderers import (
    _build_angle_heuristic_data,
    _draw_angle_footer,
    _is_angle_heuristic,
    _plot_angle_feature,
    _plot_angle_heuristic_feature,
    _plot_angle_heuristic_sidecar,
    _plot_arm_sequence_feature,
    _plot_center_periphery_feature,
    _plot_detection_fields_feature,
    _plot_heuristic_cloud_feature,
    _plot_heuristic_decomposition_feature,
    _plot_heuristic_feature,
    _plot_nort_discrimination_feature,
    _plot_perimeter_feature,
    _plot_proximity_feature,
    _plot_quadrant_feature,
    _plot_raycast_feature,
    _plot_scale_derivation_feature,
    _plot_timeseries_feature,
    _plot_trajectory_feature,
    _plot_ymaze_arm_sections_feature,
    _plot_ymaze_arm_sections_multi,
    _plot_zone_feature,
)


def _annotate_feature(ax: Axes, feature: FeatureInspection, frame_idx: int) -> None:
    theme = get_active_theme()
    description = getattr(feature, "description", feature.label)
    ax.text(
        0.02,
        0.98,
        f"{description}\nframe {frame_idx}",
        transform=ax.transAxes,
        fontsize=theme.annotation_fontsize,
        verticalalignment="top",
        color="white",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "black",
            "alpha": theme.bbox_alpha,
            "edgecolor": "none",
        },
    )
    title = feature.title_override if feature.title_override else feature.label
    ax.set_title(title, fontsize=theme.title_fontsize)


def plot_feature_on_frame(
    manifest: InspectionManifest,
    feature: FeatureInspection,
) -> Figure | None:
    """Render a single feature inspection figure on a representative video frame.

    Frame resolution order:
    1. Live video decode / pre-extracted PNG (when ``video.path`` is set)
    2. Arena screenshot (``manifest.frame_image``)
    3. Dark grey blank canvas (when ``video.resolution`` is known)
    4. ``None`` — no figure produced

    Per-feature plot profiles configured under ``[inspect.plot.profiles]``
    and ``[[inspect.plot.figure_profiles]]`` are applied around the entire
    rendering call, so all downstream :func:`get_active_theme` calls see
    the resolved theme for this feature's ``kind``/``label``.
    """
    with active_theme_for_figure(feature.kind, feature.label) as theme:
        fig = _plot_feature_on_frame_inner(manifest, feature)
        if fig is not None:
            fig._bekiper_plot_theme = theme
        return fig


def _plot_feature_on_frame_inner(
    manifest: InspectionManifest,
    feature: FeatureInspection,
) -> Figure | None:
    if isinstance(feature, FeatureStateDistribution):
        from ._state_distribution import plot_state_distributions

        return plot_state_distributions([feature], manifest.label_colors)
    if feature.label.startswith("geometry_vectors_") and isinstance(feature, FeatureHeuristicDecomposition):
        from ._vector_examples import plot_vector_examples

        return plot_vector_examples(manifest, [feature])
    frame_idx = _pick_frame_for_feature(manifest, feature)
    frame: np.ndarray | None = None

    source_frame_idx = manifest.source_frame_index(frame_idx)
    frame = read_frame_at(manifest.video.path, source_frame_idx, frames_dir=manifest.frames_dir)

    if frame is None and manifest.frame_image:
        img = cv2.imread(manifest.frame_image, cv2.IMREAD_COLOR)
        if img is not None:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if frame is None:
        w, h = manifest.video.resolution
        if w > 0 and h > 0:
            frame = np.full((h, w, 3), 40, dtype=np.uint8)
        else:
            return None

    # Multi-panel features create their own figures, so handle them before
    # the default single-panel path.
    if isinstance(feature, FeatureYMazeArmSections) and get_active_theme().small_multiples_n > 1:
        return _plot_ymaze_arm_sections_multi(manifest, feature, frame, frame_idx)
    if isinstance(feature, FeatureHeuristicDecomposition):
        return _plot_heuristic_decomposition_feature(
            manifest,
            feature,
            frame,
            frame_idx,
            frame_shape=frame.shape,
        )
    if isinstance(feature, FeatureTimeSeries):
        return _plot_timeseries_feature(manifest, feature, frame, frame_idx, frame_shape=frame.shape)
    if isinstance(feature, FeatureNortDiscrimination) and feature.label.startswith("preference_"):
        from ._preference import plot_preference

        return plot_preference(feature, manifest.label_colors)
    if isinstance(feature, FeatureHeuristicCloud):
        return _plot_heuristic_cloud_feature(
            manifest,
            feature,
            frame,
            frame_idx,
            frame_shape=frame.shape,
        )
    if isinstance(feature, FeatureHeuristic) and bound_ray_components(manifest, feature.result_column):
        base_w, base_h = _figsize_from_frame(frame)
        fig, ax = plt.subplots(figsize=(base_w, base_h + 1.4), layout="constrained")
        ax.imshow(frame.mean(axis=2) if frame.ndim == 3 else frame, cmap="gray", vmin=0, vmax=255, aspect="equal")
        _plot_heuristic_feature(ax, manifest, feature, frame_shape=frame.shape, frame_idx=frame_idx)
        ax.set_title(feature.title_override or feature.description, fontsize=get_active_theme().title_fontsize)
        return fig
    if isinstance(feature, FeatureHeuristic) and _is_angle_heuristic(feature):
        angle_data = _build_angle_heuristic_data(
            manifest,
            feature,
            frame_idx,
            frame.shape,
        )
        base_w, base_h = _figsize_from_frame(frame)
        if angle_data is not None:
            fig = plt.figure(
                figsize=(
                    (base_w * 1.45) if angle_data.heatmap is not None else base_w,
                    base_h * 1.16,
                ),
                layout="constrained",
            )
            outer = fig.add_gridspec(
                2,
                1,
                height_ratios=[4.2, 0.75],
                hspace=0.02,
            )
            top_slot = outer[0, 0]
            ax_footer = fig.add_subplot(outer[1, 0])
            ax_sidecar = None
            if angle_data.heatmap is not None:
                top = top_slot.subgridspec(1, 2, width_ratios=[3, 2], wspace=0.05)
                ax_main = fig.add_subplot(top[0, 0])
                ax_sidecar = fig.add_subplot(top[0, 1])
            else:
                ax_main = fig.add_subplot(top_slot)
            ax_main.imshow(frame, aspect="equal")
            _plot_angle_heuristic_feature(
                ax_main,
                manifest,
                feature,
                frame_idx,
                frame_shape=frame.shape,
                data=angle_data,
            )
            if ax_sidecar is not None:
                _plot_angle_heuristic_sidecar(
                    ax_sidecar,
                    manifest,
                    feature,
                    angle_data,
                    frame_shape=frame.shape,
                )
            _draw_angle_footer(
                ax_footer,
                manifest,
                feature.result_column,
                angle_data,
            )
            _annotate_feature(ax_main, feature, source_frame_idx)
            return fig

        fig, ax = plt.subplots(figsize=(base_w, base_h))
        ax.imshow(frame, aspect="equal")
        _plot_heuristic_feature(
            ax,
            manifest,
            feature,
            frame_shape=frame.shape,
        )
        _annotate_feature(ax, feature, source_frame_idx)
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=_figsize_from_frame(frame))
    ax.imshow(frame, aspect="equal")

    if not draw_feature_on_ax(ax, manifest, feature, frame.shape, frame_idx):
        plt.close(fig)
        return None

    _annotate_feature(ax, feature, source_frame_idx)
    fig.tight_layout()
    return fig


def draw_feature_on_ax(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureInspection,
    frame_shape: tuple[int, ...],
    frame_idx: int,
) -> bool:
    """Draw a single-axis feature over a frame already shown on ``ax``.

    Returns ``False`` for kinds that need their own multi-panel figure.
    """
    fs = frame_shape
    if isinstance(feature, FeaturePerimeter):
        _plot_perimeter_feature(ax, manifest, feature, frame_shape=fs)
    elif isinstance(feature, FeatureZone):
        _plot_zone_feature(ax, manifest, feature, frame_shape=fs)
    elif isinstance(feature, FeatureQuadrant):
        _plot_quadrant_feature(ax, manifest, feature, frame_shape=fs)
    elif isinstance(feature, FeatureHeuristic):
        _plot_heuristic_feature(ax, manifest, feature, frame_shape=fs, frame_idx=frame_idx)
    elif isinstance(feature, FeatureAngle):
        _plot_angle_feature(ax, manifest, feature, frame_idx, frame_shape=fs)
    elif isinstance(feature, FeatureTrajectory):
        _plot_trajectory_feature(ax, manifest, feature, frame_shape=fs, frame_idx=frame_idx)
    elif isinstance(feature, FeatureRayCast):
        _plot_raycast_feature(ax, manifest, feature, frame_idx, frame_shape=fs)
    elif isinstance(feature, FeatureProximity):
        _plot_proximity_feature(ax, manifest, feature, frame_shape=fs)
    elif isinstance(feature, FeatureDetectionFields):
        _plot_detection_fields_feature(ax, manifest, feature, frame_idx, frame_shape=fs)
    elif isinstance(feature, FeatureScaleDerivation):
        _plot_scale_derivation_feature(ax, manifest, frame_shape=fs)
    elif isinstance(feature, FeatureNortDiscrimination):
        _plot_nort_discrimination_feature(ax, manifest, feature, frame_shape=fs)
    elif isinstance(feature, FeatureArmSequence):
        _plot_arm_sequence_feature(ax, manifest, feature, frame_shape=fs)
    elif isinstance(feature, FeatureYMazeArmSections):
        _plot_ymaze_arm_sections_feature(ax, manifest, feature, frame_shape=fs)
    elif isinstance(feature, FeatureCenterPeriphery):
        _plot_center_periphery_feature(ax, manifest, feature, frame_shape=fs)
    else:
        return False
    return True
