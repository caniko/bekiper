from bekiper_inspect.plot.feature._helpers import _calibration_resolution
from ._draw_perimeter_overlay import *
from ._plot_ymaze_arm_sections_feature import *


def _plot_angle_heuristic_sidecar(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureHeuristic,
    data: AngleOverlayData,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    if data.heatmap is None:
        return

    if feature.perimeter_label and _should_draw_heuristic_perimeter_context(feature):
        try:
            _draw_perimeter_overlay(
                ax,
                manifest,
                feature.perimeter_label,
                color=get_active_theme().video_perimeter_color,
                frame_shape=frame_shape,
                linewidth=_angle_object_perimeter_linewidth(
                    feature.perimeter_label,
                    theme=get_active_theme(),
                ),
            )
        except StopIteration:
            pass
    draw_angle_heatmap(ax, data, theme=get_active_theme())


def _plot_heuristic_cloud_panel(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureHeuristicCloud,
    frame_idx: int,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Render a heuristic as a point cloud with no linked trajectory."""
    theme = get_active_theme()
    _fs = frame_shape or (0, 0)

    if feature.perimeter_label and _should_draw_heuristic_perimeter_context(feature):
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

    coord_label = _primary_coordinate_label(feature)
    if not coord_label or coord_label not in manifest.coordinate_columns:
        return

    coords_metric = manifest.get_coordinates(coord_label)
    coords_pixels = _metric_to_frame_pixels(
        coords_metric,
        manifest.settings.meters_per_pixel,
        _calibration_resolution(manifest),
        _fs,
    )

    valid_points = (
        np.isfinite(coords_pixels[:, 0]) & np.isfinite(coords_pixels[:, 1])
        if coords_pixels.size
        else np.zeros(0, dtype=bool)
    )
    if valid_points.any():
        pts = coords_pixels[valid_points]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=8,
            color=theme.hint_color,
            alpha=0.25,
            linewidths=0,
            zorder=2,
        )

    boolean_index = None
    if feature.result_column in manifest.df.columns:
        boolean_index = manifest.df[feature.result_column].to_numpy().astype(bool)
        active = boolean_index & valid_points
        if active.any():
            pts = coords_pixels[active]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=24,
                color=_resolve_color(feature, theme.video_true_color),
                alpha=0.85,
                linewidths=0,
                zorder=3,
            )

    if 0 <= frame_idx < len(coords_pixels):
        rep = coords_pixels[frame_idx]
        if np.isfinite(rep[0]) and np.isfinite(rep[1]):
            ax.scatter(
                [rep[0]],
                [rep[1]],
                s=120,
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
                zorder=4,
            )

    if boolean_index is not None:
        _add_boolean_summary(ax, manifest, feature.result_column)


def _plot_result_timeline_on_ax(
    ax: Axes,
    manifest: InspectionManifest,
    result_column: str,
    *,
    label: str | None = None,
) -> None:
    """Draw a single-row timeline for a result column."""
    theme = get_active_theme()
    meta = _find_heuristic_meta(manifest, result_column)
    if meta is not None:
        draw_timeline_on_ax(
            ax,
            manifest,
            meta.name,
            label=label or meta.name,
            y_offset=0.0,
            row_height=1.0,
            show_stats=True,
        )
    elif result_column in manifest.df.columns:
        boolean_index = manifest.df[result_column].to_numpy().astype(bool)
        fps = manifest.video.fps
        sequences = boolean_index_truth_sequences(boolean_index)
        for start, end in sequences:
            t_start = start / fps if fps else 0
            t_end = end / fps if fps else 0
            ax.axhspan(
                0.0,
                1.0,
                xmin=t_start / (len(boolean_index) / fps) if fps and len(boolean_index) else 0,
                xmax=t_end / (len(boolean_index) / fps) if fps and len(boolean_index) else 0,
                alpha=0.2,
                color=theme.bar_primary_color,
            )
        if fps:
            ax.set_xlim(0, len(boolean_index) / fps)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=theme.fontsize)


def _plot_heuristic_cloud_feature(
    manifest: InspectionManifest,
    feature: FeatureHeuristicCloud,
    frame: np.ndarray,
    frame_idx: int,
    frame_shape: tuple[int, ...] | None = None,
) -> Figure:
    """Render a cloud-style heuristic as frame + timeline."""
    theme = get_active_theme()
    base_w, base_h = _figsize_from_frame(frame)
    fig = plt.figure(
        figsize=(base_w, base_h + (base_h * 0.3)),
        layout="constrained",
    )
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[4, 1],
        hspace=0.15,
    )

    ax_cloud = fig.add_subplot(gs[0, 0])
    ax_cloud.imshow(frame, aspect="equal")
    _plot_heuristic_cloud_panel(
        ax_cloud,
        manifest,
        feature,
        frame_idx=frame_idx,
        frame_shape=frame_shape or frame.shape,
    )
    ax_cloud.text(
        0.02,
        0.98,
        f"{feature.description}\nframe {manifest.source_frame_index(frame_idx)}",
        transform=ax_cloud.transAxes,
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
    ax_cloud.set_title(
        feature.title_override if feature.title_override else feature.label,
        fontsize=theme.title_fontsize,
    )
    ax_cloud.set_xticks([])
    ax_cloud.set_yticks([])

    ax_tl = fig.add_subplot(gs[1, 0])
    timeline_label = feature.label.removeprefix("heuristic_").replace("_", " ")
    _plot_result_timeline_on_ax(
        ax_tl,
        manifest,
        feature.result_column,
        label=timeline_label,
    )

    return fig


def _plot_angle_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureAngle,
    frame_idx: int,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw a body-axis vector at the chosen frame.

    Two coordinate labels are required (`from`, `to`); a vector arrow
    is drawn between them and the heading is annotated in degrees.
    """
    theme = get_active_theme()
    resolved_color = _resolve_color(feature, theme.video_true_color)
    resolved_lw = _resolve_linewidth(feature, theme.linewidth)
    sample = build_feature_angle_sample(
        manifest,
        feature,
        frame_idx,
        frame_shape or (0, 0),
        color=resolved_color,
    )
    if sample is None:
        return

    draw_angle_samples(
        ax,
        [sample],
        theme=theme,
        line_width=resolved_lw,
    )
    ax.scatter(
        [sample.origin_xy[0]],
        [sample.origin_xy[1]],
        color=theme.video_point_color,
        s=50,
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )
    annotate_angle_sample(ax, sample, theme=theme, color=resolved_color)


def _plot_scale_derivation_feature(
    ax: Axes,
    manifest: InspectionManifest,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw the scale derivation overlay on the existing ax.

    Mirrors the standalone ``plot_scale_derivation`` from ``scale.py``
    but draws on the already-created ax (which already has the frame
    imshow underlay) instead of creating its own figure.
    """
    derivation = manifest.scale_derivation
    if derivation is None or derivation.method == "manual":
        return

    ref_label = derivation.reference_label
    mpp = manifest.settings.meters_per_pixel

    # Draw all perimeters; highlight the reference one.
    for p in manifest.perimeters:
        color = "lime" if p.label == ref_label else "gray"
        _draw_perimeter_overlay(
            ax,
            manifest,
            p.label,
            color=color,
            frame_shape=frame_shape,
        )

    # Build a pixel-space copy of the reference perimeter for dimension
    # annotations (the manifest stores metric-space coordinates).
    sx, sy = _frame_scale(_calibration_resolution(manifest), frame_shape or (0, 0))
    # Combined metric→frame-pixel scale (uniform sx used for simplicity;
    # frame scaling is almost always isotropic).
    combined = (1.0 / mpp * sx) if mpp > 0 else 1.0
    ref_perimeter = _find_perimeter(manifest.perimeters, ref_label) if ref_label else None
    if ref_perimeter is not None:
        pixel_params = {}
        for key, val in ref_perimeter.params.items():
            if isinstance(val, (int, float)) and key not in ("label",):
                pixel_params[key] = val * combined
            elif isinstance(val, list):
                pixel_params[key] = _scale_nested(val, combined)
            else:
                pixel_params[key] = val

        pixel_perim = PerimeterSpec(
            label=ref_perimeter.label,
            shape=ref_perimeter.shape,
            params=pixel_params,
        )

        if derivation.method == "arena_side":
            _annotate_arena_side(ax, pixel_perim, derivation)
        elif derivation.method == "arm_length":
            _annotate_arm_length(ax, pixel_perim, derivation)

    # Summary text box.
    theme = get_active_theme()
    lines = [f"Method: {derivation.method}"]
    if derivation.physical_meters is not None:
        lines.append(f"Known dimension: {derivation.physical_meters:.4f} m ({derivation.physical_meters * 100:.1f} cm)")
    if derivation.pixel_measurement is not None:
        lines.append(f"Measured: {derivation.pixel_measurement:.1f} px")
    if derivation.arm_pixel_spans:
        spans_str = ", ".join(f"{s:.1f}" for s in derivation.arm_pixel_spans)
        lines.append(f"Arm spans: [{spans_str}] px")
    lines.append(f"Result: {derivation.meters_per_pixel:.6f} m/px")

    ax.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=theme.fontsize,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "black",
            "alpha": min(1.0, theme.bbox_alpha + 0.1),
            "edgecolor": "white",
        },
        color="white",
    )


__all__ = [
    "_plot_angle_feature",
    "_plot_angle_heuristic_sidecar",
    "_plot_heuristic_cloud_feature",
    "_plot_heuristic_cloud_panel",
    "_plot_result_timeline_on_ax",
    "_plot_scale_derivation_feature",
]
