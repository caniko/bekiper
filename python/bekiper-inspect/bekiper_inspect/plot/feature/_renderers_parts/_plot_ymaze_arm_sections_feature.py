from bekiper_inspect.plot.feature._helpers import _calibration_resolution
from ._draw_perimeter_overlay import *


def _plot_ymaze_arm_sections_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureYMazeArmSections,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw Y-maze center/arm sections on a single axes."""
    coords_px, _state_masks = _prepare_ymaze_coords(manifest, feature, frame_shape)
    _render_ymaze_arm_sections_on_ax(
        ax,
        manifest,
        feature,
        frame_shape,
        coords_px=coords_px,
        state_masks=_state_masks,
        draw_info_box=True,
    )


def _resolve_state_masks(
    manifest: InspectionManifest,
    state_color_columns: dict[str, str],
) -> dict[str, np.ndarray]:
    """Return ``{label: bool_array}`` for each configured state column.

    Columns that are missing from ``manifest.df`` are silently skipped
    so a single project-wide profile can target columns that only exist
    on some recipes (e.g. ``freezing`` is OFT-specific).
    """
    masks: dict[str, np.ndarray] = {}
    for column in state_color_columns:
        if column in manifest.df.columns:
            masks[column] = manifest.df[column].to_numpy().astype(bool)
    return masks


def _prepare_ymaze_coords(
    manifest: InspectionManifest,
    feature: FeatureYMazeArmSections,
    frame_shape: tuple[int, ...] | None,
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Compute pixel-space coordinates and state masks for a Y-maze panel.

    Pulled out so the multi-panel renderer can slice the same arrays into
    temporal chunks without redoing the metric→pixel conversion or the
    speed-filter pass.
    """
    theme = get_active_theme()
    mpp = manifest.settings.meters_per_pixel
    _fs = frame_shape or (0, 0)
    coord_label = _primary_coordinate_label(feature)
    if not coord_label or coord_label not in manifest.coordinate_columns:
        return None, {}
    coords = manifest.get_coordinates(coord_label)
    coords_px = _metric_to_frame_pixels(coords, mpp, _calibration_resolution(manifest), _fs)
    fps = manifest.video.fps
    if theme.min_speed_filter > 0 and mpp > 0 and fps > 0:
        min_step_px = theme.min_speed_filter / (mpp * fps)
        coords_px = filter_coordinates_by_speed(
            coords_px,
            min_speed_units_per_frame=min_step_px,
        )
    state_masks = _resolve_state_masks(manifest, theme.state_color_columns)
    return coords_px, state_masks


def _render_ymaze_arm_sections_on_ax(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureYMazeArmSections,
    frame_shape: tuple[int, ...] | None,
    *,
    coords_px: np.ndarray | None,
    state_masks: dict[str, np.ndarray],
    draw_info_box: bool,
) -> None:
    """Per-axes drawing routine — perimeter, sections, trajectory, info box.

    Shared by :func:`_plot_ymaze_arm_sections_feature` and
    :func:`_plot_ymaze_arm_sections_multi`. The caller decides which
    coordinate slice to render and whether the info box should be shown
    (only the leftmost panel of a multi-figure draws it).
    """
    theme = get_active_theme()
    _draw_perimeter_overlay(
        ax,
        manifest,
        feature.perimeter_label,
        color=theme.video_perimeter_color,
        frame_shape=frame_shape,
    )
    _draw_ymaze_arm_sections(
        ax,
        manifest,
        feature.perimeter_label,
        feature.proximal_threshold,
        frame_shape=frame_shape,
        threshold_lines=feature.threshold_lines,
    )

    if coords_px is not None and len(coords_px) > 0:
        mode = theme.trajectory_render_mode
        if mode in ("density", "both"):
            plot_occupancy_heatmap(ax, coords_px)
        if mode in ("line", "both"):
            if state_masks:
                plot_state_colored_trajectory(
                    ax,
                    coords_px,
                    state_masks,
                    colors=dict(theme.state_color_columns),
                    linewidth=_resolve_linewidth(feature, theme.linewidth),
                    alpha=0.7,
                )
            else:
                plot_coordinates(
                    ax,
                    coords_px,
                    color=theme.video_point_color,
                    linewidth=_resolve_linewidth(feature, theme.linewidth),
                    alpha=0.45,
                )
        if theme.show_velocity_field:
            plot_velocity_field(ax, coords_px)

    if draw_info_box:
        ax.text(
            0.98,
            0.02,
            f"proximal: {feature.proximal_threshold * 100:.0f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="white",
            fontsize=theme.fontsize,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="black",
                alpha=min(1.0, theme.bbox_alpha + 0.1),
                edgecolor="white",
            ),
        )


def _slice_state_masks(
    state_masks: dict[str, np.ndarray],
    start: int,
    stop: int,
) -> dict[str, np.ndarray]:
    return {label: mask[start:stop] for label, mask in state_masks.items()}


def _plot_ymaze_arm_sections_multi(
    manifest: InspectionManifest,
    feature: FeatureYMazeArmSections,
    frame: np.ndarray,
    frame_idx: int,
) -> "Figure":
    """Build a side-by-side multi-panel Y-maze figure (one panel per chunk).

    The session is split into ``theme.small_multiples_n`` equal temporal
    chunks; each chunk is drawn with the full arena perimeter and section
    overlays so panels are spatially comparable. The info box is shown
    only in the leftmost panel to avoid repeated text noise.
    """
    theme = get_active_theme()
    n = max(1, theme.small_multiples_n)
    coords_px, state_masks = _prepare_ymaze_coords(manifest, feature, frame.shape)
    n_frames = len(coords_px) if coords_px is not None else 0
    chunk = max(1, math.ceil(n_frames / n)) if n_frames else 0

    base_w, base_h = _figsize_from_frame(frame)
    # Panels are narrow individually; widen the figure proportionally so
    # each panel still resembles the single-panel aspect, and add headroom
    # for the per-panel time-range titles.
    fig = plt.figure(figsize=(base_w * n, base_h * 1.06), layout="constrained")
    for i in range(n):
        ax_i = fig.add_subplot(1, n, i + 1)
        ax_i.imshow(frame, aspect="equal")
        if coords_px is not None and chunk > 0:
            lo = i * chunk
            hi = min(n_frames, (i + 1) * chunk)
            sub_coords = coords_px[lo:hi]
            sub_states = _slice_state_masks(state_masks, lo, hi)
        else:
            sub_coords = None
            sub_states = {}
        _render_ymaze_arm_sections_on_ax(
            ax_i,
            manifest,
            feature,
            frame.shape,
            coords_px=sub_coords,
            state_masks=sub_states,
            draw_info_box=(i == 0),
        )
        # Per-panel chunk label — readable from a thumbnail.
        if n_frames and manifest.video.fps > 0:
            t0 = lo / manifest.video.fps
            t1 = hi / manifest.video.fps
            title = f"chunk {i + 1}/{n}: {t0:.0f}–{t1:.0f}s"
        else:
            title = f"chunk {i + 1}/{n}"
        # Scale the title up with N — wider figures need bigger fonts to
        # stay readable when the image is rendered at thumbnail size.
        title_size = theme.title_fontsize + 4 * n
        ax_i.set_title(title, fontsize=title_size, fontweight="bold")
        ax_i.set_xticks([])
        ax_i.set_yticks([])
    return fig


def _plot_center_periphery_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureCenterPeriphery,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw center/periphery zones with occupancy heatmap and thigmotaxis stats."""
    theme = get_active_theme()
    mpp = manifest.settings.meters_per_pixel
    _fs = frame_shape or (0, 0)

    # Draw arena and center perimeters.
    for label in (feature.arena_perimeter_label, feature.center_perimeter_label):
        try:
            perimeter = manifest.get_perimeter(label)
            color = "lime" if label == feature.center_perimeter_label else theme.video_perimeter_color
            _draw_perimeter_overlay(
                ax,
                manifest,
                label,
                color=color,
                frame_shape=frame_shape,
            )
        except StopIteration:
            pass

    # Occupancy heatmap from trajectory coordinates.
    cp_coord = _primary_coordinate_label(feature)
    if cp_coord and cp_coord in manifest.coordinate_columns:
        coords = manifest.get_coordinates(cp_coord)
        coords_px = _metric_to_frame_pixels(coords, mpp, _calibration_resolution(manifest), _fs)
        fps = manifest.video.fps
        if theme.min_speed_filter > 0 and mpp > 0 and fps > 0:
            min_step_px = theme.min_speed_filter / (mpp * fps)
            coords_px = filter_coordinates_by_speed(
                coords_px,
                min_speed_units_per_frame=min_step_px,
            )
        plot_occupancy_heatmap(ax, coords_px)
        if theme.show_velocity_field:
            plot_velocity_field(ax, coords_px)

    # Summary textbox.
    lines = [
        f"Center: {feature.center_time_pct:.1f}%",
        f"Periphery: {feature.periphery_time_pct:.1f}%",
        f"Thigmotaxis: {feature.periphery_time_pct:.1f}%",
    ]
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
            boxstyle="round,pad=0.4", facecolor="black", alpha=min(1.0, theme.bbox_alpha + 0.1), edgecolor="white"
        ),
    )


def _perimeter_center_px(
    perimeter: PerimeterSpec,
    mpp: float,
    video_resolution: tuple[int, int] = (0, 0),
    frame_shape: tuple[int, ...] = (0, 0),
) -> tuple[float | None, float | None]:
    """Return the pixel-space center of a perimeter, or (None, None)."""
    scale = 1.0 / mpp if mpp > 0 else 1.0
    sx, sy = _frame_scale(video_resolution, frame_shape)
    if perimeter.shape in ("circle", "rectangle"):
        cx = perimeter.params.get("center_x")
        cy = perimeter.params.get("center_y")
        if cx is not None and cy is not None:
            return float(cx) * scale * sx, float(cy) * scale * sy
    elif perimeter.shape == "polygon":
        verts = perimeter.params.get("vertices", [])
        if verts:
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            return float(np.mean(xs)) * scale * sx, float(np.mean(ys)) * scale * sy
    return None, None


# ── Multi-object trajectory overlay (used by _plot_objects_story) ─────


def _plot_heuristic_trajectory_only(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureHeuristic,
    frame_shape: tuple[int, ...],
    object_index: int = 0,
    n_objects: int = 1,
    valid_only: bool = True,
) -> None:
    """Plot just the trajectory for one feature on an existing axes.

    No perimeter drawing, no summary text — the caller handles those.
    Splits by boolean index: *valid_only=True* draws only true-frames,
    *valid_only=False* draws only false-frames.
    """
    coord_label = _primary_coordinate_label(feature)
    if not coord_label or coord_label not in manifest.coordinate_columns:
        return
    if feature.result_column not in manifest.df.columns:
        return

    coords_metric = manifest.get_coordinates(coord_label)
    coords_pixels = _metric_to_frame_pixels(
        coords_metric,
        manifest.settings.meters_per_pixel,
        _calibration_resolution(manifest),
        frame_shape,
    )
    from .._diagnostic import draw_state_trajectory

    draw_state_trajectory(
        ax,
        coords_pixels,
        manifest.df[feature.result_column],
        valid_only=valid_only,
        object_index=object_index,
        n_objects=n_objects,
    )


# ── Angle-based heuristic vector overlay ──────────────────────────────


# Substring tags used to recognise heuristics that draw ray-offset
# (angle) overlays. Adding a new angle-kind heuristic on the Rust
# side requires extending this tuple. A future cross-stack improvement
# would carry an explicit `angle_kind` on HeuristicMeta so this
# Python-side substring match becomes redundant.
_ANGLE_HEURISTIC_NAME_TAGS: tuple[str, ...] = (
    "olfaction",
    "whisker",
    "exploring",
)


def _is_angle_heuristic(
    feature: FeatureHeuristic,
    *,
    heuristic_name: str | None = None,
) -> bool:
    """Return True for heuristics that use ray-offset (angle) filtering."""
    haystacks = (
        heuristic_name,
        feature.label,
        feature.result_column,
    )
    return any(tag in text for text in haystacks if text for tag in _ANGLE_HEURISTIC_NAME_TAGS)


def _is_interaction_context_heuristic(
    feature: FeatureHeuristic | FeatureHeuristicCloud,
    *,
    heuristic_name: str | None = None,
) -> bool:
    """Return True for interaction heuristics that should hide abstract context."""
    haystacks = (
        heuristic_name,
        feature.label,
        feature.result_column,
    )
    return any(tag in text for text in haystacks if text for tag in _INTERACTION_CONTEXT_TAGS)


def _is_physical_interaction_target_label(perimeter_label: str) -> bool:
    """Return True when the perimeter is the physical target of an interaction."""
    base_label = _base_perimeter_label(perimeter_label)
    return base_label.startswith("object_") or base_label in {"novel", "familiar"}


def _should_draw_heuristic_perimeter_context(
    feature: FeatureHeuristic | FeatureHeuristicCloud,
    *,
    heuristic_name: str | None = None,
) -> bool:
    """Return True when perimeter context is relevant for a heuristic-style panel."""
    perimeter_label = feature.perimeter_label
    if not perimeter_label:
        return False
    if not _is_interaction_context_heuristic(
        feature,
        heuristic_name=heuristic_name,
    ):
        return True
    return _is_physical_interaction_target_label(perimeter_label)


def _draw_angle_vectors(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureHeuristic,
    frame_idx: int,
    frame_shape: tuple[int, ...],
) -> None:
    """Draw the shared angle overlay without sidecar or summary boxes."""
    from ._plot_heuristic_feature import _build_angle_heuristic_data, _draw_angle_overlay

    data = _build_angle_heuristic_data(
        manifest,
        feature,
        frame_idx,
        frame_shape,
    )
    if data is None:
        return
    _draw_angle_overlay(ax, data)


__all__ = [
    "_ANGLE_HEURISTIC_NAME_TAGS",
    "_draw_angle_vectors",
    "_is_angle_heuristic",
    "_is_interaction_context_heuristic",
    "_is_physical_interaction_target_label",
    "_perimeter_center_px",
    "_plot_center_periphery_feature",
    "_plot_heuristic_trajectory_only",
    "_plot_ymaze_arm_sections_feature",
    "_plot_ymaze_arm_sections_multi",
    "_should_draw_heuristic_perimeter_context",
]
