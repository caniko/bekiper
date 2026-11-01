from bekiper_inspect.plot.feature._helpers import _calibration_resolution
from typing import Any, cast

from .._helpers import _metric_to_frame_pixels
from ._first_ray_circle_hit import *
from ._select_display_bin_px import *
from .angleoverlaysample import *
from .diagnostic_values import _float_or_nan, _float_or_none, _heuristic_name_from_feature, _parse_diag_result_column


def _best_angle_diag_for_frame(
    manifest: InspectionManifest,
    frame_index: int,
    groups: Sequence[AngleDiagnosticGroup],
    *,
    frame_shape: tuple[int, ...],
) -> dict[str, object] | None:
    best: dict[str, object] | None = None

    for group in groups:
        if group.origin_label not in manifest.coordinate_columns:
            continue
        needed_columns = [
            group.direction_x_col,
            group.direction_y_col,
            group.min_angle_col,
            group.target_x_col,
            group.target_y_col,
        ]
        if any(column not in manifest.df.columns for column in needed_columns):
            continue

        origin_metric = manifest.get_coordinates(group.origin_label)
        if frame_index >= len(origin_metric):
            continue
        ox_m, oy_m = map(float, origin_metric[frame_index])
        hdx, hdy = _normalize(
            _float_or_nan(manifest.df[group.direction_x_col][frame_index]),
            _float_or_nan(manifest.df[group.direction_y_col][frame_index]),
        )
        angle_deg = _float_or_none(manifest.df[group.min_angle_col][frame_index])
        tx_m = _float_or_none(manifest.df[group.target_x_col][frame_index])
        ty_m = _float_or_none(manifest.df[group.target_y_col][frame_index])
        if angle_deg is None or tx_m is None or ty_m is None:
            continue
        if not all(np.isfinite([ox_m, oy_m, hdx, hdy, angle_deg, tx_m, ty_m])):
            continue

        pts_px = _metric_to_frame_pixels(
            np.array([[ox_m, oy_m], [tx_m, ty_m]], dtype=float),
            manifest.settings.meters_per_pixel,
            _calibration_resolution(manifest),
            frame_shape,
        )
        ox_px, oy_px = map(float, pts_px[0])
        tx_px, ty_px = map(float, pts_px[1])
        target_dx, target_dy = _normalize(tx_px - ox_px, ty_px - oy_px)
        if not all(np.isfinite([ox_px, oy_px, tx_px, ty_px, target_dx, target_dy])):
            continue

        margin_deg: float | None = None
        threshold_deg: float | None = None
        if group.margin_col is not None and group.margin_col in manifest.df.columns:
            margin_value = _float_or_none(manifest.df[group.margin_col][frame_index])
            if margin_value is not None:
                margin_deg = margin_value
                threshold_deg = angle_deg + margin_value

        candidate = {
            "origin_px": (ox_px, oy_px),
            "heading_unit": (hdx, hdy),
            "target_unit": (target_dx, target_dy),
            "angle_deg": angle_deg,
            "margin_deg": margin_deg,
            "threshold_deg": threshold_deg,
        }
        if best is None or (
            group.priority,
            float(angle_deg),
        ) < (
            int(cast(Any, best).get("priority", 1)),
            float(cast(Any, best)["angle_deg"]),
        ):
            candidate["priority"] = group.priority
            best = candidate

    return best


def _fallback_ray_candidates(
    feature: FeatureHeuristic,
    *,
    heuristic_name: str | None = None,
) -> tuple[FallbackRayCandidate, ...]:
    parsed = _parse_diag_result_column(feature.result_column)
    if parsed is not None:
        _, component = parsed
        if component == "ray_hits_object":
            return (FallbackRayCandidate("nose", "snout_dx", "snout_dy"),)
        if component == "left_ray_hits_object":
            return (FallbackRayCandidate("center_ear", "left_ear_dx", "left_ear_dy"),)
        if component == "right_ray_hits_object":
            return (FallbackRayCandidate("center_ear", "right_ear_dx", "right_ear_dy"),)
        if component == "within_angle":
            return (FallbackRayCandidate("nose", "snout_dx", "snout_dy"),)

    inferred_name = _heuristic_name_from_feature(feature, heuristic_name=heuristic_name)
    haystacks = (
        inferred_name,
        feature.label,
        feature.result_column,
    )
    if any("whisker" in text for text in haystacks if text):
        return (
            FallbackRayCandidate("center_ear", "left_ear_dx", "left_ear_dy"),
            FallbackRayCandidate("center_ear", "right_ear_dx", "right_ear_dy"),
        )
    if any("olfaction" in text for text in haystacks if text):
        return (FallbackRayCandidate("nose", "snout_dx", "snout_dy"),)
    if any("exploring" in text for text in haystacks if text):
        return (FallbackRayCandidate("nose", "snout_dx", "snout_dy"),)
    return ()


def _fallback_angle_diag_for_frame(
    manifest: InspectionManifest,
    frame_index: int,
    candidates: Sequence[FallbackRayCandidate],
    *,
    perimeter: PerimeterSpec | None,
    frame_shape: tuple[int, ...],
) -> dict[str, object] | None:
    if perimeter is None:
        return None

    best: dict[str, object] | None = None
    for candidate in candidates:
        if candidate.origin_label not in manifest.coordinate_columns:
            continue
        if candidate.direction_x_col not in manifest.df.columns or candidate.direction_y_col not in manifest.df.columns:
            continue
        origin_metric = manifest.get_coordinates(candidate.origin_label)
        if frame_index >= len(origin_metric):
            continue
        ox_m, oy_m = map(float, origin_metric[frame_index])
        hdx, hdy = _normalize(
            _float_or_nan(manifest.df[candidate.direction_x_col][frame_index]),
            _float_or_nan(manifest.df[candidate.direction_y_col][frame_index]),
        )
        if not all(np.isfinite([ox_m, oy_m, hdx, hdy])):
            continue

        visibility = _perimeter_visibility_metric(
            perimeter,
            origin=(ox_m, oy_m),
            heading=(hdx, hdy),
        )
        if visibility is None:
            continue
        angle_deg, target_metric = visibility

        pts_px = _metric_to_frame_pixels(
            np.array([[ox_m, oy_m], [target_metric[0], target_metric[1]]], dtype=float),
            manifest.settings.meters_per_pixel,
            _calibration_resolution(manifest),
            frame_shape,
        )
        ox_px, oy_px = map(float, pts_px[0])
        tx_px, ty_px = map(float, pts_px[1])
        target_dx, target_dy = _normalize(tx_px - ox_px, ty_px - oy_px)
        if not all(np.isfinite([ox_px, oy_px, tx_px, ty_px, target_dx, target_dy])):
            continue

        candidate_diag = {
            "origin_px": (ox_px, oy_px),
            "heading_unit": (hdx, hdy),
            "target_unit": (target_dx, target_dy),
            "angle_deg": angle_deg,
            "priority": candidate.priority,
        }
        if best is None or (
            candidate.priority,
            angle_deg,
        ) < (
            int(cast(Any, best).get("priority", 1)),
            float(cast(Any, best)["angle_deg"]),
        ):
            best = candidate_diag
    return best


def build_display_angle_samples(
    samples: Sequence[AngleOverlaySample],
) -> tuple[tuple[AngleOverlaySample, ...], float]:
    """Prepare label-ready display samples from raw angle samples."""
    if not samples:
        return (), ANGLE_MIN_DISPLAY_BIN_PX

    display_bin_px = _select_display_bin_px(samples)
    display_samples = _display_angle_samples(
        samples,
        display_bin_px=display_bin_px,
    )
    return tuple(display_samples), display_bin_px


def _heuristic_window_indices(total_frames: int, representative_frame: int) -> range:
    window = max(1, total_frames // 20)
    lo = max(0, representative_frame - window)
    hi = min(total_frames, representative_frame + window + 1)
    return range(lo, hi)


def _display_angle_samples(
    raw_samples: Sequence[AngleOverlaySample],
    *,
    display_bin_px: float,
) -> list[AngleOverlaySample]:
    ordered = sorted(raw_samples, key=lambda sample: sample.frame_index)
    if len(ordered) < ANGLE_DENSE_MIN_SAMPLES:
        return [replace(sample, display_origin_xy=sample.origin_xy) for sample in ordered]
    return _bin_angle_display_samples(ordered, display_bin_px=display_bin_px)


def _bin_angle_display_samples(
    samples: Sequence[AngleOverlaySample],
    *,
    display_bin_px: float,
) -> list[AngleOverlaySample]:
    buckets: dict[tuple[int, int], list[AngleOverlaySample]] = {}
    for sample in samples:
        buckets.setdefault(
            _display_bin_key(sample.origin_xy, display_bin_px=display_bin_px),
            [],
        ).append(sample)

    display_samples = [
        _aggregate_angle_bucket(bucket)
        for _, bucket in sorted(
            buckets.items(),
            key=lambda item: np.median([sample.frame_index for sample in item[1]]),
        )
    ]
    return sorted(display_samples, key=lambda sample: sample.frame_index)


def _display_bin_key(
    origin_xy: tuple[float, float],
    *,
    display_bin_px: float,
) -> tuple[int, int]:
    x, y = origin_xy
    return (
        int(np.floor(x / display_bin_px)),
        int(np.floor(y / display_bin_px)),
    )


def _aggregate_angle_bucket(
    bucket: Sequence[AngleOverlaySample],
) -> AngleOverlaySample:
    origins = np.array([sample.origin_xy for sample in bucket], dtype=float)
    frame_indices = np.array([sample.frame_index for sample in bucket], dtype=float)
    median_frame = float(np.median(frame_indices))
    representative = min(
        bucket,
        key=lambda sample: abs(sample.frame_index - median_frame),
    )
    comparison_vectors = [sample.comparison_vector for sample in bucket if sample.comparison_vector is not None]
    return AngleOverlaySample(
        frame_index=representative.frame_index,
        origin_xy=(
            float(np.median(origins[:, 0])),
            float(np.median(origins[:, 1])),
        ),
        primary_vector=_aggregate_angle_vectors([sample.primary_vector for sample in bucket]),
        comparison_vector=(_aggregate_angle_vectors(comparison_vectors) if comparison_vectors else None),
        rendered_angle_deg=_circular_median_deg([sample.rendered_angle_deg for sample in bucket]),
        display_color=representative.display_color,
        display_origin_xy=(
            float(np.median(origins[:, 0])),
            float(np.median(origins[:, 1])),
        ),
    )


def _aggregate_angle_vectors(
    vectors: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    arr = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1)
    valid = np.isfinite(arr).all(axis=1) & (norms > 1e-9)
    if not valid.any():
        return (0.0, 0.0)

    angles = np.arctan2(arr[valid, 1], arr[valid, 0])
    median_angle = _circular_median_rad(angles)
    median_length = float(np.median(norms[valid]))
    return (
        float(np.cos(median_angle) * median_length),
        float(np.sin(median_angle) * median_length),
    )


def build_angle_heatmap(samples: Sequence[AngleOverlaySample]) -> AngleHeatmap | None:
    """Build the median-angle heatmap for a set of raw samples."""
    return _build_angle_heatmap(samples, pad_px=_select_display_bin_px(samples))


def _build_angle_heatmap(
    samples: Sequence[AngleOverlaySample],
    *,
    pad_px: float,
) -> AngleHeatmap | None:
    xs = np.array([sample.origin_xy[0] for sample in samples], dtype=float)
    ys = np.array([sample.origin_xy[1] for sample in samples], dtype=float)
    if not len(xs):
        return None

    # Pad the extent so vector tips and the angle pair labels drawn at
    # those tips have room inside the axes. Vectors are
    # ANGLE_HEURISTIC_VECTOR_PX data units long; labels sit slightly
    # beyond the tip. Without this, a tightly-clustered sample set
    # (e.g. a sparse whisker_interaction window) produces a heatmap
    # extent only a few data units wide, the overlays fall outside the
    # axes, and bbox_inches="tight" balloons the saved figure.
    overlay_pad_px = ANGLE_HEURISTIC_VECTOR_PX * 1.5
    effective_pad = max(pad_px, overlay_pad_px)
    x_min, x_max = _padded_extent(xs, pad_px=effective_pad)
    y_min, y_max = _padded_extent(ys, pad_px=effective_pad)
    x_edges = np.linspace(x_min, x_max, ANGLE_HEATMAP_BINS + 1)
    y_edges = np.linspace(y_min, y_max, ANGLE_HEATMAP_BINS + 1)

    buckets: dict[tuple[int, int], list[float]] = {}
    for sample in samples:
        x_idx = int(np.clip(np.searchsorted(x_edges, sample.origin_xy[0], side="right") - 1, 0, ANGLE_HEATMAP_BINS - 1))
        y_idx = int(np.clip(np.searchsorted(y_edges, sample.origin_xy[1], side="right") - 1, 0, ANGLE_HEATMAP_BINS - 1))
        buckets.setdefault((y_idx, x_idx), []).append(sample.rendered_angle_deg)

    values = np.full((ANGLE_HEATMAP_BINS, ANGLE_HEATMAP_BINS), np.nan, dtype=float)
    counts = np.zeros((ANGLE_HEATMAP_BINS, ANGLE_HEATMAP_BINS), dtype=float)
    for (y_idx, x_idx), bucket in buckets.items():
        values[y_idx, x_idx] = _circular_median_deg(bucket)
        counts[y_idx, x_idx] = float(len(bucket))

    max_count = float(counts.max())
    alpha = counts / max_count if max_count > 0 else counts
    return AngleHeatmap(
        values=values,
        alpha=alpha,
        extent=(x_min, x_max, y_max, y_min),
    )


def _circular_median_deg(values: Sequence[float]) -> float:
    if len(values) == 1:
        return float(values[0])

    radians = np.deg2rad(np.asarray(values, dtype=float))
    diffs = np.abs(np.angle(np.exp(1j * (radians[:, None] - radians[None, :]))))
    idx = int(np.argmin(diffs.sum(axis=1)))
    return float(values[idx])


def _circular_median_rad(values: np.ndarray) -> float:
    if values.size == 1:
        return float(values[0])

    diffs = np.abs(np.angle(np.exp(1j * (values[:, None] - values[None, :]))))
    idx = int(np.argmin(diffs.sum(axis=1)))
    return float(values[idx])


def _label_anchor(sample: AngleOverlaySample) -> tuple[float, float]:
    ox, oy = _display_origin(sample)
    if sample.comparison_vector is None:
        dx, dy = sample.primary_vector
        length = np.hypot(dx, dy)
        if length > 1e-9:
            perp_x, perp_y = -dy / length * 12.0, dx / length * 12.0
        else:
            perp_x, perp_y = 0.0, -12.0
        return ox + dx * 0.5 + perp_x, oy + dy * 0.5 + perp_y

    vectors = [sample.primary_vector]
    if sample.comparison_vector is not None:
        vectors.append(sample.comparison_vector)
    mean_x = 0.0
    mean_y = 0.0
    for dx, dy in vectors:
        ux, uy = _normalize(dx, dy)
        mean_x += ux
        mean_y += uy
    norm = np.hypot(mean_x, mean_y)
    if norm > 1e-9:
        return ox + mean_x / norm * 20.0, oy + mean_y / norm * 20.0
    return ox, oy - 14.0


def _normalize(x: float, y: float) -> tuple[float, float]:
    norm = float(np.hypot(x, y))
    if norm <= 1e-9:
        return float("nan"), float("nan")
    return x / norm, y / norm


def _padded_extent(values: np.ndarray, *, pad_px: float) -> tuple[float, float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    pad = max(pad_px, 1.0) if hi <= lo else float(pad_px)
    return lo - pad, hi + pad


__all__ = [
    "_aggregate_angle_bucket",
    "_aggregate_angle_vectors",
    "_best_angle_diag_for_frame",
    "_bin_angle_display_samples",
    "_build_angle_heatmap",
    "_circular_median_deg",
    "_circular_median_rad",
    "_display_angle_samples",
    "_fallback_angle_diag_for_frame",
    "_fallback_ray_candidates",
    "_heuristic_window_indices",
    "_label_anchor",
    "_padded_extent",
    "build_angle_heatmap",
    "build_display_angle_samples",
]
