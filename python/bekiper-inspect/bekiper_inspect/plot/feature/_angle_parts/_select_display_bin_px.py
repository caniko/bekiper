from typing import Any, cast

from ._first_ray_circle_hit import *
from .angleoverlaysample import *


def _normalize(x: float, y: float) -> tuple[float, float]:
    norm = float(np.hypot(x, y))
    if norm <= GEOMETRY_EPSILON:
        return float("nan"), float("nan")
    return x / norm, y / norm


def _select_display_bin_px(
    samples: Sequence[AngleOverlaySample],
) -> float:
    """Choose one display bin size from the raw sample cloud span.

    Applies the safe-bin floor regardless of sample count: previously the
    floor was conditional on `len(samples) >= ANGLE_DENSE_MIN_SAMPLES`, so
    sparse clusters got tiny bins that left no room for label glyphs and
    rendered as illegible specks. The dense check now only governs whether
    the bin should be even larger than the unconditional safe floor.
    """
    if len(samples) <= 1:
        return ANGLE_MIN_DISPLAY_BIN_PX

    origins = np.asarray([sample.origin_xy for sample in samples], dtype=float)
    x_span = max(float(np.ptp(origins[:, 0])), 1.0)
    y_span = max(float(np.ptp(origins[:, 1])), 1.0)
    area = x_span * y_span
    bin_px = float(np.sqrt(area / ANGLE_TARGET_DISPLAY_BINS))
    bin_px = max(bin_px, ANGLE_LABEL_SAFE_BIN_PX)
    return float(
        np.clip(
            bin_px,
            ANGLE_MIN_DISPLAY_BIN_PX,
            ANGLE_MAX_DISPLAY_BIN_PX,
        )
    )


def _display_origin(sample: AngleOverlaySample) -> tuple[float, float]:
    return sample.display_origin_xy or sample.origin_xy


def _layout_angle_pair_label(
    ax: Axes,
    sample: AngleOverlaySample,
    *,
    theme: PlotTheme,
    renderer: Any,
    axes_bbox: Bbox,
    occupied_bboxes: Sequence[Bbox],
) -> tuple[AngleOverlaySample, Bbox]:
    association_anchor_data = _pair_association_anchor(sample)
    best_sample: AngleOverlaySample | None = None
    best_bbox: Bbox | None = None
    best_score: float | None = None

    for label_anchor_disp in _label_candidate_positions(ax, sample):
        label_anchor_data = cast(
            tuple[float, float],
            tuple(float(value) for value in ax.transData.inverted().transform(label_anchor_disp)),
        )
        candidate = replace(
            sample,
            label_anchor_xy=label_anchor_data,
            association_anchor_xy=association_anchor_data,
        )
        bbox = _measure_angle_label_bbox(
            ax,
            candidate,
            theme=theme,
            renderer=renderer,
        )
        overlap_area = sum(_bbox_overlap_area(bbox, existing_bbox) for existing_bbox in occupied_bboxes)
        outside_area = _bbox_outside_area(bbox, axes_bbox)
        bbox_area = max(float(bbox.width), 1.0) * max(float(bbox.height), 1.0)
        score = overlap_area * 1000.0 + outside_area
        if outside_area <= 1e-6 and overlap_area <= 1e-6:
            return candidate, bbox
        if overlap_area <= 1e-6 and outside_area <= bbox_area * 0.15:
            return candidate, bbox
        if best_score is None or score < best_score:
            best_sample = candidate
            best_bbox = bbox
            best_score = score

    assert best_sample is not None
    assert best_bbox is not None
    # If the chosen label sits far away from the axes (not merely past
    # the edge but clearly disconnected), the caller has set up the
    # axes too small for these overlays. Silently returning would let
    # labels escape and bbox_inches="tight" would balloon the saved
    # figure (root cause of the whisker-decomposition story artifact).
    # The legitimate fallback path lays labels just past the edge with
    # a short connector tick, so allow small overshoots.
    gap_px = _bbox_outside_gap(best_bbox, axes_bbox)
    if gap_px > _MAX_LABEL_OUTSIDE_GAP_PX:
        raise RuntimeError(
            "angle pair label could not be placed inside its axes "
            f"(label bbox={tuple(best_bbox.extents)}, "
            f"axes bbox={tuple(axes_bbox.extents)}, "
            f"gap from axes={gap_px:.1f}px > {_MAX_LABEL_OUTSIDE_GAP_PX}px) — "
            "axes data limits are too tight for vector + label radii"
        )
    return replace(best_sample, show_association_tick=True), best_bbox


# Maximum allowed display-pixel gap between a label bbox and its axes
# bbox before we consider the placement broken. Sized to comfortably
# admit the legitimate fallback (labels nudged a few pixels past the
# axes edge with a short connector tick) but reject runaway escapes
# (labels hundreds of pixels away because the axes was tiny relative
# to the vector length).
_MAX_LABEL_OUTSIDE_GAP_PX = 30.0


def _bbox_outside_gap(bbox: Bbox, container: Bbox) -> float:
    """Display-pixel distance from *bbox* to *container*; 0 when they overlap."""
    dx = max(container.x0 - bbox.x1, bbox.x0 - container.x1, 0.0)
    dy = max(container.y0 - bbox.y1, bbox.y0 - container.y1, 0.0)
    return float(np.hypot(dx, dy))


def _label_candidate_positions(
    ax: Axes,
    sample: AngleOverlaySample,
) -> tuple[np.ndarray, ...]:
    origin = np.asarray(_display_origin(sample), dtype=float)
    origin_disp = np.asarray(ax.transData.transform(origin), dtype=float)
    primary_tip_disp = np.asarray(
        ax.transData.transform(origin + np.asarray(sample.primary_vector, dtype=float)),
        dtype=float,
    )
    primary_dir = _normalize_display_vector(primary_tip_disp - origin_disp)

    if sample.comparison_vector is None:
        bisector_dir = primary_dir
    else:
        comparison_tip_disp = np.asarray(
            ax.transData.transform(origin + np.asarray(sample.comparison_vector, dtype=float)),
            dtype=float,
        )
        comparison_dir = _normalize_display_vector(comparison_tip_disp - origin_disp)
        bisector_dir = _normalize_display_vector(primary_dir + comparison_dir)
        if np.hypot(*bisector_dir) <= 1e-9:
            bisector_dir = primary_dir

    clockwise_dir = _rotate_display_vector(bisector_dir, -ANGLE_LABEL_SWEEP_DEG)
    counter_clockwise_dir = _rotate_display_vector(
        bisector_dir,
        ANGLE_LABEL_SWEEP_DEG,
    )
    left_perp = np.asarray([-bisector_dir[1], bisector_dir[0]], dtype=float)
    right_perp = -left_perp
    escape_left = _normalize_display_vector(left_perp + bisector_dir * 0.25)
    escape_right = _normalize_display_vector(right_perp + bisector_dir * 0.25)
    tip_projections = [
        float(np.dot(primary_tip_disp - origin_disp, bisector_dir)),
    ]
    tip_distances = [
        float(np.hypot(*(primary_tip_disp - origin_disp))),
    ]
    if sample.comparison_vector is not None:
        tip_projections.append(float(np.dot(comparison_tip_disp - origin_disp, bisector_dir)))
        tip_distances.append(float(np.hypot(*(comparison_tip_disp - origin_disp))))

    base_radius = (
        max(
            max(tip_projections, default=0.0),
            max(tip_distances, default=0.0),
        )
        + ANGLE_LABEL_GAP_PX
    )
    farther_radius = base_radius + ANGLE_LABEL_FARTHER_OFFSET_PX
    escape_radius = base_radius + ANGLE_LABEL_ESCAPE_OFFSET_PX
    second_farther_radius = farther_radius + ANGLE_LABEL_FARTHER_OFFSET_PX
    second_escape_radius = escape_radius + ANGLE_LABEL_FARTHER_OFFSET_PX

    return tuple(
        origin_disp + direction * radius
        for direction, radius in (
            (bisector_dir, base_radius),
            (clockwise_dir, base_radius),
            (counter_clockwise_dir, base_radius),
            (bisector_dir, farther_radius),
            (clockwise_dir, farther_radius),
            (counter_clockwise_dir, farther_radius),
            (escape_left, escape_radius),
            (escape_right, escape_radius),
            (bisector_dir, second_farther_radius),
            (clockwise_dir, second_farther_radius),
            (counter_clockwise_dir, second_farther_radius),
            (escape_left, second_escape_radius),
            (escape_right, second_escape_radius),
        )
    )


def _pair_association_anchor(sample: AngleOverlaySample) -> tuple[float, float]:
    ox, oy = _display_origin(sample)
    if sample.comparison_vector is None:
        dx, dy = sample.primary_vector
        return (ox + dx * 0.6, oy + dy * 0.6)
    pdx, pdy = sample.primary_vector
    cdx, cdy = sample.comparison_vector
    return (
        ox + (pdx + cdx) * 0.3,
        oy + (pdy + cdy) * 0.3,
    )


def _measure_angle_label_bbox(
    ax: Axes,
    sample: AngleOverlaySample,
    *,
    theme: PlotTheme,
    renderer: Any,
) -> Bbox:
    assert sample.label_anchor_xy is not None
    text = ax.text(
        sample.label_anchor_xy[0],
        sample.label_anchor_xy[1],
        f"{sample.rendered_angle_deg:.1f}°",
        color=sample.display_color,
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
        alpha=0.0,
    )
    bbox = text.get_window_extent(renderer=renderer).expanded(1.08, 1.2)
    text.remove()
    return bbox


def _is_pair_angle_label_text(text: str) -> bool:
    if not text.endswith("°"):
        return False
    try:
        float(text[:-1])
    except ValueError:
        return False
    return True


def _bbox_overlap_area(a: Bbox, b: Bbox) -> float:
    overlap = Bbox.intersection(a, b)
    if overlap is None:
        return 0.0
    return max(float(overlap.width), 0.0) * max(float(overlap.height), 0.0)


def _bbox_outside_area(bbox: Bbox, container: Bbox) -> float:
    inside = Bbox.intersection(bbox, container)
    bbox_area = max(float(bbox.width), 0.0) * max(float(bbox.height), 0.0)
    if inside is None:
        return bbox_area
    inside_area = max(float(inside.width), 0.0) * max(float(inside.height), 0.0)
    return max(bbox_area - inside_area, 0.0)


def _normalize_display_vector(vector: np.ndarray) -> np.ndarray:
    length = float(np.hypot(vector[0], vector[1]))
    if length <= 1e-9 or not np.isfinite(length):
        return np.asarray([0.0, -1.0], dtype=float)
    return vector / length


def _rotate_display_vector(vector: np.ndarray, degrees: float) -> np.ndarray:
    radians = float(np.deg2rad(degrees))
    cos_theta = float(np.cos(radians))
    sin_theta = float(np.sin(radians))
    rotated = np.asarray(
        [
            vector[0] * cos_theta - vector[1] * sin_theta,
            vector[0] * sin_theta + vector[1] * cos_theta,
        ],
        dtype=float,
    )
    return _normalize_display_vector(rotated)


def _draw_angle_label_tick(
    ax: Axes,
    sample: AngleOverlaySample,
    *,
    bbox: Bbox,
    line_width: float,
    zorder: float,
) -> None:
    if sample.label_anchor_xy is None or sample.association_anchor_xy is None:
        return

    label_anchor_disp = np.asarray(
        ax.transData.transform(sample.label_anchor_xy),
        dtype=float,
    )
    association_anchor_disp = np.asarray(
        ax.transData.transform(sample.association_anchor_xy),
        dtype=float,
    )
    tick_dir = _normalize_display_vector(association_anchor_disp - label_anchor_disp)
    start_disp = (
        _bbox_edge_point_from_center(
            bbox,
            label_anchor_disp,
            tick_dir,
        )
        + tick_dir * ANGLE_LABEL_TICK_GAP_PX
    )
    end_disp = start_disp + tick_dir * ANGLE_LABEL_TICK_PX
    tick_points = ax.transData.inverted().transform(np.vstack([start_disp, end_disp]))
    ax.plot(
        tick_points[:, 0],
        tick_points[:, 1],
        color=ANGLE_LABEL_TICK_COLOR,
        linewidth=line_width,
        alpha=0.95,
        solid_capstyle="round",
        zorder=zorder,
    )


def _bbox_edge_point_from_center(
    bbox: Bbox,
    center: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    distances: list[float] = []
    dx = float(direction[0])
    dy = float(direction[1])

    if abs(dx) > 1e-9:
        x_edge = float(bbox.x1 if dx > 0 else bbox.x0)
        distances.append((x_edge - float(center[0])) / dx)
    if abs(dy) > 1e-9:
        y_edge = float(bbox.y1 if dy > 0 else bbox.y0)
        distances.append((y_edge - float(center[1])) / dy)

    positive = [distance for distance in distances if distance >= 0.0]
    if not positive:
        return center.copy()
    return center + direction * min(positive)


def _perimeter_visibility_metric(
    perimeter: PerimeterSpec,
    *,
    origin: tuple[float, float],
    heading: tuple[float, float],
) -> tuple[float, tuple[float, float]] | None:
    if perimeter.shape == "circle":
        center_x = perimeter.params.get("center_x")
        center_y = perimeter.params.get("center_y")
        radius = perimeter.params.get("radius")
        if center_x is None or center_y is None or radius is None:
            return None
        return _circle_visibility_metric(
            origin=origin,
            heading=heading,
            center=(float(center_x), float(center_y)),
            radius=float(radius),
        )

    polygons: list[np.ndarray] = []
    if perimeter.shape == "rectangle":
        rect = _rectangle_vertices_metric(perimeter)
        if rect is not None:
            polygons.append(rect)
    elif perimeter.shape in ("polygon", "triangle"):
        vertices = np.asarray(perimeter.params.get("vertices", []), dtype=float)
        if vertices.size:
            polygons.append(vertices)
    elif perimeter.shape == "radial_maze":
        center_vertices = np.asarray(
            perimeter.params.get("center_vertices", []),
            dtype=float,
        )
        if center_vertices.size:
            polygons.append(center_vertices)
        for arm in perimeter.params.get("arms", []):
            arm_vertices = np.asarray(arm, dtype=float)
            if arm_vertices.size:
                polygons.append(arm_vertices)

    best: tuple[float, tuple[float, float]] | None = None
    for vertices in polygons:
        visibility = _polygon_visibility_metric(
            origin=origin,
            heading=heading,
            vertices=vertices,
        )
        best = _better_visibility(origin, best, visibility)
    return best


def _rectangle_vertices_metric(perimeter: PerimeterSpec) -> np.ndarray | None:
    center_x = perimeter.params.get("center_x")
    center_y = perimeter.params.get("center_y")
    width = perimeter.params.get("width")
    height = perimeter.params.get("height")
    if center_x is None or center_y is None or width is None or height is None:
        return None
    cx = float(center_x)
    cy = float(center_y)
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    return np.asarray(
        [
            [cx - half_w, cy - half_h],
            [cx + half_w, cy - half_h],
            [cx + half_w, cy + half_h],
            [cx - half_w, cy + half_h],
        ],
        dtype=float,
    )


def _circle_visibility_metric(
    *,
    origin: tuple[float, float],
    heading: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> tuple[float, tuple[float, float]] | None:
    hdx, hdy = _normalize(*heading)
    if not all(np.isfinite([hdx, hdy])):
        return None

    hit = _first_ray_circle_hit(origin, (hdx, hdy), center, radius)
    if hit is not None:
        return 0.0, hit

    to_center = (center[0] - origin[0], center[1] - origin[1])
    center_dist = float(np.hypot(*to_center))
    if center_dist <= radius + GEOMETRY_EPSILON:
        hit = _first_ray_circle_hit(origin, (hdx, hdy), center, radius)
        return (0.0, hit) if hit is not None else None

    to_center_unit = (to_center[0] / center_dist, to_center[1] / center_dist)
    center_angle = _angle_between(heading=(hdx, hdy), vector=to_center_unit)
    tangent_half_angle = float(np.arcsin(np.clip(radius / center_dist, -1.0, 1.0)))
    angle_offset = max(center_angle - tangent_half_angle, 0.0)

    center_bearing = float(np.arctan2(to_center[1], to_center[0]))
    heading_bearing = float(np.arctan2(hdy, hdx))
    signed_delta = _wrap_angle_radians(heading_bearing - center_bearing)
    tangent_bearing = center_bearing + (tangent_half_angle if signed_delta >= 0.0 else -tangent_half_angle)
    tangent_direction = (float(np.cos(tangent_bearing)), float(np.sin(tangent_bearing)))
    tangent_hit = _first_ray_circle_hit(origin, tangent_direction, center, radius)
    if tangent_hit is None:
        return None
    return float(np.degrees(angle_offset)), tangent_hit


__all__ = [
    "_bbox_edge_point_from_center",
    "_bbox_outside_area",
    "_bbox_overlap_area",
    "_circle_visibility_metric",
    "_display_origin",
    "_draw_angle_label_tick",
    "_is_pair_angle_label_text",
    "_label_candidate_positions",
    "_layout_angle_pair_label",
    "_measure_angle_label_bbox",
    "_normalize_display_vector",
    "_pair_association_anchor",
    "_perimeter_visibility_metric",
    "_rectangle_vertices_metric",
    "_rotate_display_vector",
    "_select_display_bin_px",
]
