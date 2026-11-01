from bekiper_inspect.plot.feature._helpers import _calibration_resolution
from ._draw_perimeter_overlay import *
from ._plot_angle_heuristic_sidecar import *
from ._plot_heuristic_feature import *
from ._plot_ymaze_arm_sections_feature import *
from bekiper_inspect.plot.color import identity_color
from matplotlib.transforms import Affine2D
from .._diagnostic import draw_state_trajectory


def _plot_raycast_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureRayCast,
    frame_idx: int,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw an olfactory/sensing cone at the origin coordinate.

    The cone is a matplotlib ``Wedge`` patch centred on the origin label's
    position at *frame_idx*, oriented along the heading angle read from
    ``feature.angle_column``, and spanning ``feature.cone_width_deg``
    degrees.
    """
    if feature.origin_label not in manifest.coordinate_columns:
        return
    if feature.angle_column not in manifest.df.columns:
        return

    origin_metric = manifest.get_coordinates(feature.origin_label)
    if frame_idx >= len(origin_metric):
        return

    heading_angle = manifest.df[feature.angle_column]
    if frame_idx >= len(heading_angle):
        return

    mpp = manifest.settings.meters_per_pixel
    _fs = frame_shape or (0, 0)
    origin_px = _metric_to_frame_pixels(
        origin_metric[frame_idx : frame_idx + 1], mpp, _calibration_resolution(manifest), _fs
    )[0]
    ox, oy = float(origin_px[0]), float(origin_px[1])
    if not (np.isfinite(ox) and np.isfinite(oy)):
        return

    angle_rad = float(heading_angle[frame_idx])
    angle_deg = float(np.degrees(angle_rad))
    half_width = feature.cone_width_deg / 2.0
    if not np.isfinite(angle_rad):
        return
    sx, sy = _frame_scale(_calibration_resolution(manifest), _fs)
    length_px = feature.length_meters / mpp if mpp > 0 else feature.length_meters

    theme = get_active_theme()
    resolved_color = _resolve_color(feature, theme.video_true_color)
    resolved_lw = _resolve_linewidth(feature, 1.5)

    # Sensing cone wedge.
    wedge = patches.Wedge(
        (0, 0),
        length_px,
        angle_deg - half_width,
        angle_deg + half_width,
        facecolor=resolved_color,
        alpha=0.3,
        edgecolor=resolved_color,
        linewidth=resolved_lw,
    )
    wedge.set_transform(Affine2D().scale(sx, sy).translate(ox, oy) + ax.transData)
    ax.add_patch(wedge)

    # Origin dot.
    ax.scatter([ox], [oy], color=theme.video_point_color, s=40, zorder=5)

    # Center ray line.
    end_x = ox + length_px * np.cos(angle_rad) * sx
    end_y = oy + length_px * np.sin(angle_rad) * sy
    ax.plot(
        [ox, end_x],
        [oy, end_y],
        color=resolved_color,
        linewidth=resolved_lw,
        zorder=4,
    )

    # Angle annotation.
    ax.text(
        (ox + end_x) / 2,
        (oy + end_y) / 2,
        f"{angle_deg:.1f}\u00b0",
        color=resolved_color,
        fontsize=theme.annotation_fontsize,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "black",
            "alpha": theme.bbox_alpha,
            "edgecolor": "none",
        },
    )


def _plot_proximity_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureProximity,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw the primary perimeter, an expanded buffer ring, and color the
    trajectory by the boolean result column."""
    theme = get_active_theme()
    mpp = manifest.settings.meters_per_pixel
    _fs = frame_shape or (0, 0)
    sx, sy = _frame_scale(_calibration_resolution(manifest), _fs)

    # Draw the primary perimeter.
    try:
        perimeter = _resolve_metric_perimeter(manifest, feature.perimeter_label)
        if perimeter is None:
            return
        _draw_perimeter_overlay(
            ax,
            manifest,
            feature.perimeter_label,
            color=theme.video_perimeter_color,
            frame_shape=frame_shape,
        )

        # Draw the expanded buffer ring.
        buffer_px = feature.buffer_meters / mpp if mpp > 0 else feature.buffer_meters
        buffer_x, buffer_y = buffer_px * sx, buffer_px * sy
        if perimeter.shape == "circle":
            cx = (perimeter.params["center_x"] / mpp if mpp > 0 else perimeter.params["center_x"]) * sx
            cy = (perimeter.params["center_y"] / mpp if mpp > 0 else perimeter.params["center_y"]) * sy
            r = perimeter.params["radius"] / mpp if mpp > 0 else perimeter.params["radius"]
            outer: patches.Patch = patches.Ellipse(
                (cx, cy),
                width=2 * (r + buffer_px) * sx,
                height=2 * (r + buffer_px) * sy,
                fill=False,
                edgecolor=theme.video_true_color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
            )
            ax.add_patch(outer)
        elif perimeter.shape == "rectangle":
            cx = (perimeter.params["center_x"] / mpp if mpp > 0 else perimeter.params["center_x"]) * sx
            cy = (perimeter.params["center_y"] / mpp if mpp > 0 else perimeter.params["center_y"]) * sy
            w = (perimeter.params["width"] / mpp if mpp > 0 else perimeter.params["width"]) * sx
            h = (perimeter.params["height"] / mpp if mpp > 0 else perimeter.params["height"]) * sy
            outer = patches.Rectangle(
                (cx - w / 2 - buffer_x, cy - h / 2 - buffer_y),
                w + 2 * buffer_x,
                h + 2 * buffer_y,
                fill=False,
                edgecolor=theme.video_true_color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
            )
            ax.add_patch(outer)
    except StopIteration:
        pass

    # Color trajectory by boolean result column.
    prox_coord_label = _primary_coordinate_label(feature)
    if prox_coord_label and prox_coord_label in manifest.coordinate_columns:
        coords_metric = manifest.get_coordinates(prox_coord_label)
        coords_pixels = _metric_to_frame_pixels(coords_metric, mpp, _calibration_resolution(manifest), _fs)
        if feature.result_column in manifest.df.columns:
            draw_state_trajectory(ax, coords_pixels, manifest.df[feature.result_column])
            _add_boolean_summary(ax, manifest, feature.result_column)


def _plot_trajectory_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureTrajectory,
    frame_shape: tuple[int, ...] | None = None,
    frame_idx: int | None = None,
) -> None:
    if feature.label.startswith("skeleton_") and len(feature.coordinate_labels) > 1:
        _plot_skeleton_zoom_feature(ax, manifest, feature, frame_shape=frame_shape, frame_idx=frame_idx)
        return

    traj_coord_label = _primary_coordinate_label(feature)
    if not traj_coord_label or traj_coord_label not in manifest.coordinate_columns:
        return
    coords_metric = manifest.get_coordinates(traj_coord_label)
    coords_pixels = _metric_to_frame_pixels(
        coords_metric,
        manifest.settings.meters_per_pixel,
        _calibration_resolution(manifest),
        frame_shape or (0, 0),
    )
    theme = get_active_theme()
    if theme.trajectory_gradient:
        plot_gradient_line(ax, coords_pixels, linewidth=_resolve_linewidth(feature, theme.linewidth), alpha=0.8)
        return
    plot_coordinates(
        ax,
        coords_pixels,
        color=_resolve_color(feature, theme.video_true_color),
        linewidth=_resolve_linewidth(feature, theme.linewidth),
    )


def _skeleton_edges(labels: list[str]) -> list[tuple[str, str]]:
    present = set(labels)
    candidate_edges = [
        ("nose", "left_ear"),
        ("nose", "right_ear"),
        ("left_ear", "center_ear"),
        ("right_ear", "center_ear"),
        ("center_ear", "tail"),
        ("center_ear", "base_tail"),
        ("center_ear", "tail_base"),
    ]
    edges = [(a, b) for a, b in candidate_edges if a in present and b in present]
    if edges:
        return edges
    return list(zip(labels, labels[1:]))


def _plot_skeleton_zoom_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureTrajectory,
    frame_shape: tuple[int, ...] | None = None,
    frame_idx: int | None = None,
) -> None:
    frame_idx = int(feature.representative_frame_index or 0) if frame_idx is None else int(frame_idx)
    theme = get_active_theme()
    _fs = frame_shape or (0, 0)
    points: dict[str, tuple[float, float]] = {}

    for label in feature.coordinate_labels:
        if label not in manifest.coordinate_columns:
            continue
        coords_metric = manifest.get_coordinates(label)
        if frame_idx >= len(coords_metric):
            continue
        coords_pixels = _metric_to_frame_pixels(
            coords_metric[frame_idx : frame_idx + 1],
            manifest.settings.meters_per_pixel,
            _calibration_resolution(manifest),
            _fs,
        )
        if len(coords_pixels) == 0:
            continue
        x, y = float(coords_pixels[0][0]), float(coords_pixels[0][1])
        if np.isfinite(x) and np.isfinite(y):
            points[label] = (x, y)

    if len(points) < 2:
        return

    color = _resolve_color(feature, theme.video_true_color)
    linewidth = _resolve_linewidth(feature, theme.linewidth)
    for a, b in _skeleton_edges(feature.coordinate_labels):
        if a in points and b in points:
            ax.plot(
                [points[a][0], points[b][0]],
                [points[a][1], points[b][1]],
                color=color,
                linewidth=linewidth,
                alpha=0.95,
                zorder=4,
            )

    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    ax.scatter(xs, ys, color=[identity_color(label, manifest.label_colors) for label in points], s=44, zorder=5)
    for label, (x, y) in points.items():
        ax.text(
            x + 4,
            y + 4,
            label,
            color="white",
            fontsize=max(7, theme.annotation_fontsize - 1),
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "black",
                "alpha": theme.bbox_alpha,
                "edgecolor": "none",
            },
            zorder=6,
        )

    pad = max(24.0, max(max(xs) - min(xs), max(ys) - min(ys)) * 0.45)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(max(ys) + pad, min(ys) - pad)


def _draw_timeseries_on_ax(ax: Axes, manifest: InspectionManifest, feature: FeatureTimeSeries, frame_idx: int) -> None:
    """Draw native scalar values on their recorded timeline, retaining gaps."""
    theme = get_active_theme()
    ax.set_title(feature.title_override or feature.description, fontsize=theme.title_fontsize)
    ax.set_ylabel(f"{feature.title_override or feature.value_column} ({feature.unit})")
    if feature.value_column not in manifest.df.columns:
        ax.text(0.5, 0.5, "Native scalar unavailable", transform=ax.transAxes, ha="center")
        return
    values = manifest.df[feature.value_column].to_numpy().astype(float)
    times = np.arange(len(values), dtype=float)
    x_label = "Frame"
    for column in ("elapsed_s", "elapsed"):
        if column in manifest.df.columns:
            times = manifest.df[column].to_numpy().astype(float)
            x_label = "Time (s)"
            break
    else:
        if manifest.video.fps > 0:
            times /= manifest.video.fps
            x_label = "Time (s)"
    ax.plot(
        times,
        np.where(np.isfinite(values), values, np.nan),
        color=_resolve_color(feature, theme.bar_primary_color),
        linewidth=_resolve_linewidth(feature, 1.0),
    )
    if 0 <= frame_idx < len(times) and np.isfinite(times[frame_idx]):
        ax.axvline(
            times[frame_idx],
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"frame {manifest.source_frame_index(frame_idx)}",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel(x_label)


def _plot_timeseries_feature(
    manifest: InspectionManifest,
    feature: FeatureTimeSeries,
    frame: np.ndarray,
    frame_idx: int,
    frame_shape: tuple[int, ...] | None = None,
) -> Figure:
    """Dual-panel figure: video frame + trajectory on the left, scalar
    line plot over frames on the right.
    """
    theme = get_active_theme()
    fig, (ax_frame, ax_ts) = plt.subplots(
        1,
        2,
        figsize=(20, frame.shape[0] / frame.shape[1] * 10),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    # Left panel: video frame with trajectory overlay.
    ax_frame.imshow(frame, aspect="equal")
    coord_label = None
    for lbl in manifest.labels:
        if lbl in manifest.coordinate_columns:
            coord_label = lbl
            break
    if coord_label:
        coords = manifest.get_coordinates(coord_label)
        coords_px = _metric_to_frame_pixels(
            coords,
            manifest.settings.meters_per_pixel,
            _calibration_resolution(manifest),
            frame_shape or frame.shape,
        )
        plot_coordinates(
            ax_frame,
            coords_px,
            color=theme.video_true_color,
            linewidth=theme.linewidth,
        )
    title = feature.title_override if feature.title_override else feature.label
    ax_frame.set_title(title, fontsize=theme.title_fontsize)

    _draw_timeseries_on_ax(ax_ts, manifest, feature, frame_idx)

    fig.tight_layout()
    return fig


def _plot_nort_discrimination_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureNortDiscrimination,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw object perimeters on the arena and overlay per-object exploration
    time bars + discrimination indices."""
    theme = get_active_theme()
    mpp = manifest.settings.meters_per_pixel

    # Highlight each object perimeter with a distinct color.
    _fs = frame_shape or (0, 0)
    for i, obj_label in enumerate(feature.object_labels):
        try:
            color = identity_color(obj_label, manifest.label_colors)
            _draw_perimeter_overlay(
                ax,
                manifest,
                obj_label,
                color=color,
                frame_shape=frame_shape,
            )
            perimeter = manifest.get_perimeter(obj_label)
            # Label the object on the perimeter center.
            cx, cy = _perimeter_center_px(perimeter, mpp, _calibration_resolution(manifest), _fs)
            if cx is not None and cy is not None:
                ax.text(
                    cx,
                    cy,
                    obj_label,
                    color=color,
                    fontsize=theme.annotation_fontsize,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=theme.bbox_alpha, edgecolor="none"),
                )
        except StopIteration:
            pass

    # Trajectory overlay.
    nort_coord = _primary_coordinate_label(feature)
    if nort_coord and nort_coord in manifest.coordinate_columns:
        coords = manifest.get_coordinates(nort_coord)
        coords_px = _metric_to_frame_pixels(coords, mpp, _calibration_resolution(manifest), _fs)
        plot_coordinates(
            ax,
            coords_px,
            color=theme.video_point_color,
            linewidth=_resolve_linewidth(feature, theme.linewidth),
            alpha=0.5,
        )

    # Bar chart of exploration times in bottom-right.
    bar_lines = []
    if feature.protocol:
        bar_lines.append(f"protocol: {feature.protocol}")
    for obj_label, t in zip(feature.object_labels, feature.exploration_times):
        bar_lines.append(f"{obj_label}: {t:.1f}s")
    if feature.discrimination_index is not None:
        bar_lines.append(f"DI: {feature.discrimination_index:+.3f}")
    if feature.recognition_index is not None:
        bar_lines.append(f"RI: {feature.recognition_index:.3f}")
    ax.text(
        0.98,
        0.02,
        "\n".join(bar_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=theme.fontsize,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="black", alpha=min(1.0, theme.bbox_alpha + 0.1), edgecolor="white"
        ),
    )

    # Qualia decomposition panel in bottom-left (if available).
    if feature.qualia_channels:
        qualia_lines = ["Qualia:"]
        for ch in feature.qualia_channels:
            label = ch.heuristic_type.replace("_", " ")
            parts = []
            for obj_label, secs, bias in zip(feature.object_labels, ch.per_object_seconds, ch.relative_bias):
                parts.append(f"{obj_label}={secs:.1f}s ({bias:.0f}%)")
            qualia_lines.append(f"  {label}: {', '.join(parts)}")
        if feature.alternation_count is not None:
            qualia_lines.append(f"  alternations: {feature.alternation_count}")
        if feature.merged_channels:
            for merge_name, per_obj in feature.merged_channels.items():
                label = merge_name.replace("_", " ")
                parts = [f"{obj}={s:.1f}s" for obj, s in zip(feature.object_labels, per_obj)]
                qualia_lines.append(f"  {label}: {', '.join(parts)}")
        ax.text(
            0.02,
            0.02,
            "\n".join(qualia_lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="white",
            fontsize=max(theme.fontsize - 1, 6),
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="black", alpha=min(1.0, theme.bbox_alpha + 0.1), edgecolor="white"
            ),
        )


def _plot_arm_sequence_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureArmSequence,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw the trajectory with numbered arm-entry annotations."""
    theme = get_active_theme()
    mpp = manifest.settings.meters_per_pixel
    _fs = frame_shape or (0, 0)

    # Draw all perimeters (arms + center) for context.
    section_feature = next(
        (fi for fi in manifest.feature_inspections if isinstance(fi, FeatureYMazeArmSections)),
        None,
    )
    for p in manifest.perimeters:
        _draw_perimeter_overlay(
            ax,
            manifest,
            p.label,
            color=theme.video_perimeter_color,
            frame_shape=frame_shape,
        )
        if p.shape == "radial_maze":
            _draw_ymaze_arm_sections(
                ax,
                manifest,
                p.label,
                section_feature.proximal_threshold if section_feature else 0.25,
                frame_shape=frame_shape,
                threshold_lines=section_feature.threshold_lines if section_feature else (),
            )

    # Trajectory overlay.
    arm_coord = _primary_coordinate_label(feature)
    if not arm_coord or arm_coord not in manifest.coordinate_columns:
        return
    coords = manifest.get_coordinates(arm_coord)
    coords_px = _metric_to_frame_pixels(coords, mpp, _calibration_resolution(manifest), _fs)
    plot_coordinates(
        ax,
        coords_px,
        color=theme.video_point_color,
        linewidth=_resolve_linewidth(feature, theme.linewidth),
        alpha=0.5,
    )

    # Annotate arm entries with numbered labels at the perimeter centers.
    # Track visit counts for error/rerun markers (radial arm: revisits are errors).
    visit_counts: dict[str, int] = {}
    for entry_idx, arm in enumerate(feature.arm_sequence, start=1):
        visit_counts[arm] = visit_counts.get(arm, 0) + 1
        is_revisit = visit_counts[arm] > 1
        try:
            perimeter = manifest.get_perimeter(arm)
            cx, cy = _perimeter_center_px(perimeter, mpp, _calibration_resolution(manifest), _fs)
            if cx is None or cy is None:
                continue
            # Offset successive annotations vertically to avoid overlap.
            offset = (visit_counts[arm] - 1) * 12
            color = "red" if is_revisit else theme.video_true_color
            marker = f"{entry_idx}{'*' if is_revisit else ''}"
            ax.text(
                cx,
                cy + offset,
                marker,
                color=color,
                fontsize=max(6, theme.annotation_fontsize - 2),
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=theme.bbox_alpha, edgecolor="none"),
            )
        except StopIteration:
            pass

    # Summary box.
    lines = [f"Entries: {len(feature.arm_sequence)}"]
    if feature.alternation_pct is not None:
        lines.append(f"Alternation: {feature.alternation_pct:.1f}%")
    ax.text(
        0.98,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=theme.fontsize,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="black", alpha=min(1.0, theme.bbox_alpha + 0.1), edgecolor="white"
        ),
    )


__all__ = [
    "_draw_timeseries_on_ax",
    "_plot_arm_sequence_feature",
    "_plot_nort_discrimination_feature",
    "_plot_proximity_feature",
    "_plot_raycast_feature",
    "_plot_timeseries_feature",
    "_plot_trajectory_feature",
]
