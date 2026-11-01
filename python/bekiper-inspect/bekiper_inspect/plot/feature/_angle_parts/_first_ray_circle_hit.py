from .angleoverlaysample import *


def _normalize(x: float, y: float) -> tuple[float, float]:
    norm = float(np.hypot(x, y))
    if norm <= GEOMETRY_EPSILON:
        return float("nan"), float("nan")
    return x / norm, y / norm


def _first_ray_circle_hit(
    origin: tuple[float, float],
    direction: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float] | None:
    ox, oy = origin
    dx, dy = direction
    fx = ox - center[0]
    fy = oy - center[1]

    a = dx * dx + dy * dy
    if a <= GEOMETRY_EPSILON:
        return None
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius

    discriminant = b * b - 4.0 * a * c
    if discriminant < -GEOMETRY_EPSILON:
        return None

    sqrt_disc = float(np.sqrt(max(discriminant, 0.0)))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    positive = [t for t in (t1, t2) if t >= -GEOMETRY_EPSILON]
    if not positive:
        return None
    t = max(min(positive), 0.0)
    return (ox + t * dx, oy + t * dy)


def _polygon_visibility_metric(
    *,
    origin: tuple[float, float],
    heading: tuple[float, float],
    vertices: np.ndarray,
) -> tuple[float, tuple[float, float]] | None:
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return None
    if points.shape[0] > 1 and np.allclose(points[0], points[-1], atol=GEOMETRY_EPSILON):
        points = points[:-1]
    if points.shape[0] < 2:
        return None

    best: tuple[float, tuple[float, float]] | None = None
    for idx in range(points.shape[0]):
        start = (float(points[idx, 0]), float(points[idx, 1]))
        next_idx = (idx + 1) % points.shape[0]
        end = (float(points[next_idx, 0]), float(points[next_idx, 1]))
        visibility = _segment_visibility_metric(origin, heading, start, end)
        best = _better_visibility(origin, best, visibility)
    return best


def _segment_visibility_metric(
    origin: tuple[float, float],
    heading: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, tuple[float, float]] | None:
    point = _ray_segment_intersection_point(origin, heading, start, end)
    if point is not None:
        return 0.0, point

    best: tuple[float, tuple[float, float]] | None = None
    for point in (start, end):
        unit = _normalize(point[0] - origin[0], point[1] - origin[1])
        if not all(np.isfinite(unit)):
            continue
        candidate = (float(np.degrees(_angle_between(heading=heading, vector=unit))), point)
        best = _better_visibility(origin, best, candidate)
    return best


def _ray_segment_intersection_point(
    origin: tuple[float, float],
    direction: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> tuple[float, float] | None:
    ox, oy = origin
    dx, dy = direction
    sx, sy = seg_start
    ex, ey = seg_end

    ray_len_sq = dx * dx + dy * dy
    if ray_len_sq <= GEOMETRY_EPSILON:
        return None

    seg_dx = ex - sx
    seg_dy = ey - sy
    offset_x = sx - ox
    offset_y = sy - oy
    denom = dx * seg_dy - dy * seg_dx
    if abs(denom) <= GEOMETRY_EPSILON:
        cross = offset_x * dy - offset_y * dx
        if abs(cross) > GEOMETRY_EPSILON:
            return None

        t0 = (offset_x * dx + offset_y * dy) / ray_len_sq
        t1 = ((ex - ox) * dx + (ey - oy) * dy) / ray_len_sq
        t_min = min(t0, t1)
        t_max = max(t0, t1)
        if t_max < -GEOMETRY_EPSILON:
            return None
        t = max(t_min, 0.0) if t_min >= -GEOMETRY_EPSILON else 0.0
        return (ox + t * dx, oy + t * dy)

    t = (offset_x * seg_dy - offset_y * seg_dx) / denom
    u = (offset_x * dy - offset_y * dx) / denom
    if t >= -GEOMETRY_EPSILON and -GEOMETRY_EPSILON <= u <= 1.0 + GEOMETRY_EPSILON:
        t = max(t, 0.0)
        return (ox + t * dx, oy + t * dy)
    return None


def _better_visibility(
    origin: tuple[float, float],
    current: tuple[float, tuple[float, float]] | None,
    candidate: tuple[float, tuple[float, float]] | None,
) -> tuple[float, tuple[float, float]] | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    current_key = (
        current[0],
        float(np.hypot(current[1][0] - origin[0], current[1][1] - origin[1])),
    )
    candidate_key = (
        candidate[0],
        float(np.hypot(candidate[1][0] - origin[0], candidate[1][1] - origin[1])),
    )
    return candidate if candidate_key < current_key else current


def _angle_between(*, heading: tuple[float, float], vector: tuple[float, float]) -> float:
    dot = heading[0] * vector[0] + heading[1] * vector[1]
    return float(np.arccos(np.clip(dot, -1.0, 1.0)))


def _wrap_angle_radians(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


__all__ = [
    "_angle_between",
    "_better_visibility",
    "_first_ray_circle_hit",
    "_polygon_visibility_metric",
    "_ray_segment_intersection_point",
    "_segment_visibility_metric",
    "_wrap_angle_radians",
]
