from bekiper_inspect.plot.feature._helpers import _calibration_resolution
from matplotlib.gridspec import SubplotSpec
from .._diagnostic import (
    bound_ray_components,
    draw_ray_field,
    draw_scalar_histogram,
    draw_state_trajectory,
    metric_axes,
)

from ._draw_perimeter_overlay import *
from ._plot_angle_heuristic_sidecar import *
from ._plot_ymaze_arm_sections_feature import *


def _plot_heuristic_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureHeuristic,
    frame_shape: tuple[int, ...] | None = None,
    object_index: int = 0,
    n_objects: int = 1,
    frame_idx: int | None = None,
    valid_only: bool | None = None,
    angle_context: str | None = None,
) -> None:
    """Render a heuristic feature on an axis.

    Parameters
    ----------
    valid_only:
        ``True``  — plot only valid (boolean-true) trajectory segments.
        ``False`` — plot only rejected (boolean-false) segments.
        ``None``  — legacy mode: plot both valid and rejected together.
    """
    theme = get_active_theme()
    _fs = frame_shape or (0, 0)

    ray_components = bound_ray_components(manifest, feature.result_column) if frame_idx is not None else []
    if ray_components:
        for component in ray_components:
            draw_ray_field(ax, manifest, component, _fs, result_column=feature.result_column, valid_only=valid_only)
        _add_boolean_summary(ax, manifest, feature.result_column)
        return

    if frame_idx is not None and _is_angle_heuristic(feature, heuristic_name=angle_context):
        truth_filter = "false" if valid_only is False else "true"
        angle_data = _plot_angle_heuristic_feature(
            ax,
            manifest,
            feature,
            frame_idx,
            frame_shape=_fs,
            heuristic_name=angle_context,
            truth_filter=truth_filter,
        )
        if angle_data is not None:
            if valid_only is False:
                annotate_angle_summary_rows(
                    ax,
                    angle_summary_rows(angle_data),
                    theme=theme,
                )
                _add_boolean_summary(ax, manifest, feature.result_column)
            return

    # Only draw perimeter context when it is semantically relevant.
    if feature.perimeter_label and _should_draw_heuristic_perimeter_context(
        feature,
        heuristic_name=angle_context,
    ):
        try:
            _draw_perimeter_overlay(
                ax,
                manifest,
                feature.perimeter_label,
                color=theme.video_perimeter_color,
                frame_shape=frame_shape,
            )
        except StopIteration:
            pass

    heur_coord_label = _primary_coordinate_label(feature)
    if not heur_coord_label or heur_coord_label not in manifest.coordinate_columns:
        return
    coords_metric = manifest.get_coordinates(heur_coord_label)
    coords_pixels = _metric_to_frame_pixels(
        coords_metric,
        manifest.settings.meters_per_pixel,
        _calibration_resolution(manifest),
        _fs,
    )
    if feature.result_column not in manifest.df.columns:
        return
    draw_state_trajectory(
        ax,
        coords_pixels,
        manifest.df[feature.result_column],
        valid_only=valid_only,
        object_index=object_index,
        n_objects=n_objects,
    )

    _add_boolean_summary(ax, manifest, feature.result_column)


def _build_heuristic_decomposition_panels(
    feature: FeatureHeuristicDecomposition,
) -> list[tuple[str, FeatureHeuristic | FeatureHeuristicCloud, str | None]]:
    """Return the ordered component panels for a decomposition figure."""
    use_cloud_panels = feature.heuristic_name == "freezing"
    panels: list[tuple[str, FeatureHeuristic | FeatureHeuristicCloud, str | None]] = []

    components = list(feature.components)
    order = [
        "left_ear_near",
        "left_ray_hits_object",
        "left_combined",
        "right_ear_near",
        "right_ray_hits_object",
        "right_combined",
        "raw_combined",
        "outside_object",
    ]
    if any(c.label == "left_combined" for c in components):
        components.sort(key=lambda c: order.index(c.label) if c.label in order else len(order))
    for component in components:
        panel_cls = FeatureHeuristicCloud if use_cloud_panels else FeatureHeuristic
        panels.append(
            (
                component.label,
                panel_cls(
                    label=f"{feature.label}_{component.label}",
                    result_column=component.result_column,
                    coordinate_labels=component.coordinate_labels or feature.coordinate_labels,
                    perimeter_label=component.perimeter_label or feature.perimeter_label,
                    description=component.label,
                ),
                None if use_cloud_panels else feature.heuristic_name,
            )
        )

    if feature.show_final_result:
        panel_cls = FeatureHeuristicCloud if use_cloud_panels else FeatureHeuristic
        panels.append(
            (
                "final",
                panel_cls(
                    label=f"{feature.label}_final",
                    result_column=feature.result_column,
                    coordinate_labels=feature.coordinate_labels,
                    perimeter_label=feature.perimeter_label,
                    description="final",
                ),
                None if use_cloud_panels else feature.heuristic_name,
            )
        )

    return panels


def _draw_heuristic_decomposition_panels(
    fig: Figure,
    slots: list[SubplotSpec],
    panels: list[tuple[str, FeatureHeuristic | FeatureHeuristicCloud, str | None]],
    manifest: InspectionManifest,
    frame: np.ndarray,
    frame_idx: int,
    frame_shape: tuple[int, ...] | None = None,
    diagnostics: list[object] | None = None,
) -> None:
    """Draw a decomposition grid with uniform footer and sidecar geometry."""
    modern = any(
        getattr(c, "ray_origin_label", None)
        or getattr(c, "buffer_meters", None) is not None
        or getattr(c, "evaluated_perimeter", None) is not None
        for c in diagnostics or []
    )
    bindings = {getattr(c, "label"): c for c in diagnostics or []} if modern else {}
    # Reserve the same sub-slots even for panels without angle data; otherwise
    # equal-aspect video images acquire different sizes within the same row.
    panel_angle_data: list[AngleOverlayData | None] = []
    for _title, panel_feature, angle_context in panels:
        data = None
        if (
            isinstance(panel_feature, FeatureHeuristic)
            and not modern
            and angle_context is not None
            and _is_angle_heuristic(panel_feature, heuristic_name=angle_context)
        ):
            data = _build_angle_heuristic_data(
                manifest,
                panel_feature,
                frame_idx,
                frame_shape or frame.shape,
                heuristic_name=angle_context,
            )
        panel_angle_data.append(data)
    column_has_footer = any(d is not None for d in panel_angle_data)
    column_has_sidecar = any(d is not None and d.heatmap is not None for d in panel_angle_data)

    for slot, (title, panel_feature, angle_context), angle_data in zip(slots, panels, panel_angle_data, strict=True):
        footer_ax = None
        ax_sidecar = None

        if column_has_footer:
            slot_grid = slot.subgridspec(
                2,
                1,
                height_ratios=[4.0, 0.72],
                hspace=0.02,
            )
            visual_slot = slot_grid[0, 0]
            footer_ax = fig.add_subplot(slot_grid[1, 0])
        else:
            visual_slot = slot

        if column_has_sidecar:
            visual_grid = visual_slot.subgridspec(
                1,
                2,
                width_ratios=[3, 2],
                wspace=0.05,
            )
            ax = fig.add_subplot(visual_grid[0, 0])
            ax_sidecar = fig.add_subplot(visual_grid[0, 1])
        else:
            ax = fig.add_subplot(visual_slot)

        ax.imshow(
            frame.mean(axis=2) if bindings and frame.ndim == 3 else frame,
            cmap="gray" if bindings else None,
            vmin=0,
            vmax=255,
            aspect="equal",
        )
        component = bindings.get(title)
        if component is not None and getattr(component, "ray_origin_label", None):
            draw_ray_field(ax, manifest, component, frame_shape or frame.shape)
        elif component is not None and getattr(component, "buffer_perimeter", None) is not None:
            from ...perimeter import plot_perimeter_on_ax

            for geometry, color, buffered in [
                (component.evaluated_perimeter, get_active_theme().video_perimeter_color, False),
                (component.buffer_perimeter, get_active_theme().video_true_color, True),
            ]:
                if geometry is not None:
                    before = len(ax.patches)
                    plot_perimeter_on_ax(
                        ax,
                        geometry,
                        color=color,
                        meters_per_pixel=manifest.settings.meters_per_pixel,
                        as_pixels=True,
                        video_resolution=_calibration_resolution(manifest),
                        frame_shape=frame.shape,
                    )
                    for patch in ax.patches[before:]:
                        if buffered:
                            patch.set_linestyle("--")
            trajectory = FeatureHeuristic(
                label=panel_feature.label,
                result_column=panel_feature.result_column,
                coordinate_labels=panel_feature.coordinate_labels,
                description=title,
            )
            _plot_heuristic_feature(ax, manifest, trajectory, frame_shape=frame_shape, frame_idx=None)
            metric_axes(ax, manifest, frame.shape)
        elif component is not None and getattr(component, "buffer_meters", None) is not None:
            from ._plot_raycast_feature import _plot_proximity_feature

            proximity = FeatureProximity(
                label=panel_feature.label,
                result_column=panel_feature.result_column,
                coordinate_labels=panel_feature.coordinate_labels,
                perimeter_label=panel_feature.perimeter_label,
                buffer_meters=component.buffer_meters,
                description=title,
            )
            _plot_proximity_feature(ax, manifest, proximity, frame_shape=frame_shape)
            metric_axes(ax, manifest, frame_shape or frame.shape)
        elif isinstance(panel_feature, FeatureHeuristicCloud):
            _plot_heuristic_cloud_panel(
                ax,
                manifest,
                panel_feature,
                frame_idx=frame_idx,
                frame_shape=frame_shape,
            )
        elif angle_data is not None:
            _plot_angle_heuristic_feature(
                ax,
                manifest,
                panel_feature,
                frame_idx,
                frame_shape=frame_shape,
                data=angle_data,
                heuristic_name=angle_context,
            )
            if ax_sidecar is not None:
                _plot_angle_heuristic_sidecar(
                    ax_sidecar,
                    manifest,
                    panel_feature,
                    angle_data,
                    frame_shape=frame_shape,
                )
            if footer_ax is not None:
                _draw_angle_footer(
                    footer_ax,
                    manifest,
                    panel_feature.result_column,
                    angle_data,
                )
        else:
            if modern:
                from ...perimeter import plot_perimeter_on_ax

                geometry = getattr(component, "evaluated_perimeter", None)
                if geometry is None and diagnostics:
                    geometry = getattr(diagnostics[0], "evaluated_perimeter", None)
                if geometry is not None:
                    plot_perimeter_on_ax(
                        ax,
                        geometry,
                        color=get_active_theme().video_perimeter_color,
                        meters_per_pixel=manifest.settings.meters_per_pixel,
                        as_pixels=True,
                        video_resolution=_calibration_resolution(manifest),
                        frame_shape=frame.shape,
                    )
                    panel_feature = FeatureHeuristic(
                        label=panel_feature.label,
                        result_column=panel_feature.result_column,
                        coordinate_labels=panel_feature.coordinate_labels,
                        description=title,
                    )
            _plot_heuristic_feature(
                ax,
                manifest,
                panel_feature,
                frame_shape=frame_shape,
                frame_idx=None if bindings else frame_idx,
                valid_only=None if bindings else True,
                angle_context=None if component is not None else angle_context,
            )
        # Keep the recorded bout policy below the comparison name rather than
        # spilling its caption into the neighboring object.
        name, separator, policy = title.partition(" (bout ")
        caption = f"{name.replace('_', ' ')}\n(bout {policy}" if separator else title
        ax.set_title(caption, fontsize=get_active_theme().title_fontsize)
        if modern:
            metric_axes(ax, manifest, frame.shape)
            ax.set_xlim(-0.5, frame.shape[1] - 0.5)
            ax.set_ylim(frame.shape[0] - 0.5, -0.5)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        if ax_sidecar is not None:
            ax_sidecar.set_xticks([])
            ax_sidecar.set_yticks([])
            if angle_data is None or angle_data.heatmap is None:
                ax_sidecar.axis("off")
        if footer_ax is not None and angle_data is None:
            footer_ax.axis("off")


def _plot_heuristic_decomposition_feature(
    manifest: InspectionManifest,
    feature: FeatureHeuristicDecomposition,
    frame: np.ndarray,
    frame_idx: int,
    frame_shape: tuple[int, ...] | None = None,
) -> Figure:
    panels = _build_heuristic_decomposition_panels(feature)
    n_cols = _decomposition_columns(feature, len(panels))
    has_scalar = len(feature.components) == 1 and feature.components[0].value_column is not None
    if has_scalar:
        n_cols = 2
    panel_count = len(panels) + int(has_scalar)
    n_rows = int(math.ceil(max(1, panel_count) / n_cols))
    fig = plt.figure(
        figsize=(n_cols * 5.5, n_rows * (4.8 * frame.shape[0] / frame.shape[1] + 1.6)),
        layout="constrained",
    )
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.12, wspace=0.08)
    slots = [gs[index // n_cols, index % n_cols] for index in range(len(panels))]
    _draw_heuristic_decomposition_panels(
        fig, slots, panels, manifest, frame, frame_idx, frame_shape, diagnostics=feature.components
    )

    if has_scalar:
        scalar_ax = fig.add_subplot(gs[len(panels) // n_cols, len(panels) % n_cols])
        draw_scalar_histogram(scalar_ax, manifest, feature.components[0])

    for index in range(panel_count, n_rows * n_cols):
        row = index // n_cols
        col = index % n_cols
        ax_empty = fig.add_subplot(gs[row, col])
        ax_empty.axis("off")

    fig.suptitle(
        feature.title_override or feature.heuristic_name,
        fontsize=get_active_theme().title_fontsize + 1,
    )
    return fig


def _decomposition_columns(feature: FeatureHeuristicDecomposition, panel_count: int) -> int:
    if feature.heuristic_name.startswith("recipe_olfaction_"):
        return 1
    if feature.heuristic_name.startswith("recipe_body_proximity_"):
        return 1
    if feature.heuristic_name.startswith(("cumulative_observation_", "fusion_comparison_")):
        return 2
    return min(3, max(1, panel_count))


def _build_angle_heuristic_data(
    manifest: InspectionManifest,
    feature: FeatureHeuristic,
    frame_idx: int,
    frame_shape: tuple[int, ...],
    *,
    heuristic_name: str | None = None,
    truth_filter: str = "true",
) -> AngleOverlayData | None:
    theme = get_active_theme()
    default_color = theme.video_false_color if truth_filter == "false" else theme.video_true_color
    return build_heuristic_angle_overlay(
        manifest,
        feature,
        frame_idx,
        frame_shape,
        theme=theme,
        heuristic_name=heuristic_name,
        truth_filter=truth_filter,
        primary_color=_resolve_color(feature, default_color),
    )


def _draw_angle_overlay(
    ax: Axes,
    data: AngleOverlayData,
    *,
    line_width: float | None = None,
    show_pair_labels: bool = True,
) -> None:
    theme = get_active_theme()
    draw_angle_samples(
        ax,
        data.display_samples,
        theme=theme,
        comparison_color=theme.video_point_color,
        line_width=line_width,
    )
    if show_pair_labels:
        draw_angle_pair_labels(
            ax,
            data.display_samples,
            theme=theme,
            line_width=line_width,
        )


def _draw_angle_footer(
    ax: Axes,
    manifest: InspectionManifest,
    result_column: str,
    data: AngleOverlayData | None,
    *,
    left_rows: list[str] | None = None,
    right_rows: list[str] | None = None,
) -> None:
    theme = get_active_theme()
    draw_angle_metric_footer(
        ax,
        left_rows=left_rows if left_rows is not None else (angle_summary_rows(data) if data is not None else []),
        right_rows=right_rows if right_rows is not None else (_boolean_summary_rows(manifest, result_column)),
        theme=theme,
    )


def _plot_angle_heuristic_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureHeuristic,
    frame_idx: int,
    frame_shape: tuple[int, ...] | None = None,
    data: AngleOverlayData | None = None,
    *,
    heuristic_name: str | None = None,
    truth_filter: str = "true",
) -> AngleOverlayData | None:
    theme = get_active_theme()
    _fs = frame_shape or (0, 0)

    angle_data = data or _build_angle_heuristic_data(
        manifest,
        feature,
        frame_idx,
        _fs,
        heuristic_name=heuristic_name,
        truth_filter=truth_filter,
    )
    if angle_data is None:
        return None

    if feature.perimeter_label and _should_draw_heuristic_perimeter_context(feature):
        try:
            _draw_perimeter_overlay(
                ax,
                manifest,
                feature.perimeter_label,
                color=theme.video_perimeter_color,
                frame_shape=_fs,
                linewidth=_angle_object_perimeter_linewidth(
                    feature.perimeter_label,
                    theme=theme,
                ),
            )
        except StopIteration:
            pass

    _draw_angle_overlay(
        ax,
        angle_data,
        line_width=_resolve_linewidth(feature, theme.linewidth),
    )
    return angle_data


__all__ = [
    "_build_angle_heuristic_data",
    "_build_heuristic_decomposition_panels",
    "_decomposition_columns",
    "_draw_angle_footer",
    "_draw_angle_overlay",
    "_draw_heuristic_decomposition_panels",
    "_plot_angle_heuristic_feature",
    "_plot_heuristic_decomposition_feature",
    "_plot_heuristic_feature",
]
