"""Per-feature inspection figures: frame selection, geometry, and annotation helpers."""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
from matplotlib.axes import Axes

from bekiper_inspect.manifest import (
    FeatureDetectionFields,
    FeatureHeuristic,
    FeatureHeuristicCloud,
    FeatureHeuristicDecomposition,
    FeatureInspection,
    FeatureProximity,
    FeatureZone,
    HeuristicMeta,
    InspectionManifest,
    PerimeterSpec,
    find_representative_frame,
)

_EXPANDED_PERIMETER_RE = re.compile(r"^(?P<base>.+)_expanded_-?\d+(?:\.\d+)?$")

# ── Frame selection ──────────────────────────────────────────────────────


def _result_column_for_feature(feature: FeatureInspection) -> str | None:
    """Return the boolean result column attached to a feature, if any.

    Used by the frame picker to find a representative active frame.
    """
    if isinstance(feature, FeatureZone):
        return cast(str, feature.result_column)
    if isinstance(feature, FeatureHeuristic):
        return cast(str, feature.result_column)
    if isinstance(feature, FeatureHeuristicDecomposition):
        return cast(str, feature.result_column)
    if isinstance(feature, FeatureHeuristicCloud):
        return cast(str, feature.result_column)
    if isinstance(feature, FeatureProximity):
        return cast(str, feature.result_column)
    return None


def _detection_hits(manifest: InspectionManifest, feature: FeatureDetectionFields) -> tuple[np.ndarray, np.ndarray]:
    """Per-object (nose_hit, whisker_hit) boolean matrices, shape (objects, frames).

    Missing diagnostic columns read as all-False so a nose-only run still draws.
    """
    n = len(manifest.df)
    df = manifest.df

    def col(name: str | None) -> np.ndarray:
        if name is None or name not in df.columns:
            return np.zeros(n, dtype=bool)
        return df[name].fill_null(False).to_numpy().astype(bool)

    objects = range(len(feature.object_labels))
    near, gate, fan = feature.nose_near_columns, feature.nose_gate_columns, feature.whisker_hit_columns
    nose = np.array([col(near[i]) & col(gate[i]) if i < len(near) and i < len(gate) else col(None) for i in objects])
    whisker = np.array([col(fan[i]) if i < len(fan) else col(None) for i in objects])
    return nose.reshape(len(objects), n), whisker.reshape(len(objects), n)


def _pick_detection_frame(manifest: InspectionManifest, feature: FeatureDetectionFields) -> int | None:
    """Prefer a whisker-only hit frame, then a nose hit frame."""
    nose, whisker = _detection_hits(manifest, feature)
    nose_any, whisker_any = nose.any(axis=0), whisker.any(axis=0)
    for mask in (whisker_any & ~nose_any, nose_any):
        idx = find_representative_frame(mask)
        if idx is not None:
            return idx
    return None


def _primary_coordinate_label(feature: FeatureInspection) -> str | None:
    """Return the first coordinate label from a feature, or None.

    Handles both the new ``coordinate_labels`` list and the legacy
    ``coordinate_label`` string for variants that haven't migrated
    (e.g. Angle's ``from_coordinate_label``).
    """
    labels = getattr(feature, "coordinate_labels", None)
    if labels:
        return str(labels[0])
    label = getattr(feature, "coordinate_label", None) or getattr(feature, "from_coordinate_label", None)
    return str(label) if label is not None else None


def _coordinate_label_for_feature(feature: FeatureInspection, manifest: InspectionManifest) -> str | None:
    """Pick a coordinate label associated with this feature for fallback frame
    selection (median trajectory frame). Falls back to the manifest's first
    coordinate column if the feature carries no explicit label.
    """
    label = _primary_coordinate_label(feature)
    if label and label in manifest.coordinate_columns:
        return label
    if manifest.labels:
        for candidate in manifest.labels:
            if candidate in manifest.coordinate_columns:
                return candidate
    return None


def _pick_frame_for_feature(manifest: InspectionManifest, feature: FeatureInspection) -> int:
    """Heuristic-driven frame selection.

    0. If the feature carries an explicit ``representative_frame_index``
       (set by the Rust producer), use it directly.
    1. If the feature has a boolean result column, return the middle of
       its longest True bout.
    2. Otherwise return the median index of the primary coordinate's
       non-null trajectory.
    3. As a last resort, return frame 0.
    """
    rep = getattr(feature, "representative_frame_index", None)
    if rep is not None:
        return _clamp_frame(manifest, int(rep))

    if isinstance(feature, FeatureDetectionFields):
        idx = _pick_detection_frame(manifest, feature)
        if idx is not None:
            return _clamp_frame(manifest, idx)
        feature_label = feature.nose_label
        if feature_label in manifest.coordinate_columns:
            non_null = ~manifest.df[manifest.coordinate_columns[feature_label].x].is_null().to_numpy()
            if non_null.any():
                return _clamp_frame(manifest, int(np.median(np.flatnonzero(non_null))))
        return 0

    column = _result_column_for_feature(feature)
    if column is not None and column in manifest.df.columns:
        boolean_index = manifest.df[column].to_numpy()
        idx = find_representative_frame(boolean_index)
        if idx is not None:
            return _clamp_frame(manifest, idx)

    coord_label = _coordinate_label_for_feature(feature, manifest)
    if coord_label is not None:
        spec = manifest.coordinate_columns[coord_label]
        x_col = manifest.df[spec.x]
        # Median index of non-null entries.
        non_null = ~x_col.is_null().to_numpy()
        if non_null.any():
            indices = np.flatnonzero(non_null)
            return _clamp_frame(manifest, int(np.median(indices)))

    return 0


def _clamp_frame(manifest: InspectionManifest, index: int) -> int:
    total = manifest.video.total_frames
    if total > 0:
        return int(max(0, min(index, total - 1)))
    return int(max(0, index))


# ── Per-feature styling resolution ───────────────────────────────────────


def _resolve_color(feature: FeatureInspection, default: str) -> str:
    """Return ``feature.color`` if set, otherwise *default*."""
    c = getattr(feature, "color", None)
    return c if c else default


def _resolve_linewidth(feature: FeatureInspection, default: float) -> float:
    """Return ``feature.linewidth`` if set, otherwise *default*."""
    lw = getattr(feature, "linewidth", None)
    return lw if lw is not None else default


def _fallback_perimeter_labels(perimeter_label: str) -> list[str]:
    """Return plot-side fallback labels for synthetic perimeter names."""
    labels = [perimeter_label]
    match = _EXPANDED_PERIMETER_RE.match(perimeter_label)
    if match is not None:
        labels.append(match.group("base"))
    return labels


def _base_perimeter_label(perimeter_label: str) -> str:
    """Return the concrete base label for synthetic perimeter names."""
    return _fallback_perimeter_labels(perimeter_label)[-1]


def _resolve_metric_perimeter(
    manifest: InspectionManifest,
    perimeter_label: str,
) -> PerimeterSpec | None:
    """Resolve a metric-space perimeter, falling back from synthetic labels."""
    for candidate in _fallback_perimeter_labels(perimeter_label):
        try:
            return manifest.get_perimeter(candidate)
        except StopIteration:
            continue
    return None


def _resolve_plot_perimeter(
    manifest: InspectionManifest,
    perimeter_label: str,
) -> tuple[PerimeterSpec | None, bool]:
    """Resolve a perimeter for drawing, preferring pixel geometry when present."""
    pixel_perim = manifest.get_pixel_perimeter(perimeter_label)
    if pixel_perim is not None:
        return pixel_perim, True

    metric_perim = _resolve_metric_perimeter(manifest, perimeter_label)
    if metric_perim is not None and metric_perim.label == perimeter_label:
        return metric_perim, False

    for candidate in _fallback_perimeter_labels(perimeter_label)[1:]:
        pixel_perim = manifest.get_pixel_perimeter(candidate)
        if pixel_perim is not None:
            return pixel_perim, True

    return metric_perim, False


# ── Geometry helpers (metric -> pixel) ────────────────────────────────────


def _metric_to_pixels(coords: np.ndarray, meters_per_pixel: float) -> np.ndarray:
    """Convert (N, 2) metric coords to pixel coords.

    Returns the input unchanged if `meters_per_pixel` is zero or
    negative -- the manifest may carry coordinates that are already in
    pixels (legacy data) or unitless.
    """
    if meters_per_pixel and meters_per_pixel > 0:
        return coords / meters_per_pixel
    return coords


def _y_aspect(manifest: Any) -> float:
    """Vertical/horizontal metres-per-pixel ratio (1.0 for square pixels).

    Metric coordinates divided by ``settings.meters_per_pixel`` (the
    horizontal scale) land in square-pixel space; non-square sources
    (e.g. a detected 40 cm box of 204 x 210 px) need their vertical axis
    divided by this ratio to line up with the video frame.
    """
    derivation = getattr(manifest, "scale_derivation", None)
    mpp_y = getattr(derivation, "meters_per_pixel_y", None)
    mpp_x = getattr(derivation, "meters_per_pixel", 0.0)
    if mpp_y and mpp_x and mpp_x > 0:
        return float(mpp_y) / float(mpp_x)
    return 1.0


def _calibration_resolution(manifest: Any) -> tuple[float, float]:
    """Square-pixel space size: the frame-scale reference for metric geometry.

    Pass this (not ``manifest.video.resolution``) wherever metric coordinates
    divided by ``meters_per_pixel`` are mapped onto a displayed frame.
    """
    width, height = manifest.video.resolution
    return width, height * _y_aspect(manifest)


def _frame_scale(
    video_resolution: tuple[float, float],
    frame_shape: tuple[int, ...],
) -> tuple[float, float]:
    """Return (sx, sy) to map calibration-pixel coords to actual frame pixels.

    ``video_resolution`` is (width, height) from the manifest.
    ``frame_shape`` is the numpy array shape (H, W, ...) of the displayed frame.
    Returns (1.0, 1.0) when they match or when the manifest resolution is unknown.
    """
    vid_w, vid_h = video_resolution
    frame_h, frame_w = frame_shape[0], frame_shape[1]
    if vid_w > 0 and vid_h > 0:
        return frame_w / vid_w, frame_h / vid_h
    return 1.0, 1.0


def _metric_to_frame_pixels(
    coords: np.ndarray,
    meters_per_pixel: float,
    video_resolution: tuple[int, int],
    frame_shape: tuple[int, ...],
) -> np.ndarray:
    """Convert (N, 2) metric coords to the pixel space of the displayed frame.

    Two-step conversion:
    1. metric → calibration pixels  (divide by meters_per_pixel)
    2. calibration pixels → frame pixels  (scale by frame/video ratio)
    """
    px = _metric_to_pixels(coords, meters_per_pixel)
    sx, sy = _frame_scale(video_resolution, frame_shape)
    if sx != 1.0 or sy != 1.0:
        px = px.copy()
        px[:, 0] *= sx
        px[:, 1] *= sy
    return px


def _figsize_from_frame(frame: np.ndarray, base_width: float = 12.0) -> tuple[float, float]:
    h, w = frame.shape[:2]
    if w <= 0:
        return (base_width, base_width * 0.8)
    aspect = h / w
    return (base_width, base_width * aspect)


# ── Summary annotation helpers ──────────────────────────────────────────


def _find_heuristic_meta(manifest: InspectionManifest, result_column: str) -> HeuristicMeta | None:
    """Find HeuristicMeta matching *result_column*, or None."""
    for h in manifest.heuristics:
        if h.result_column == result_column:
            return h
    return None


def _summary_rows(
    seconds: float,
    percentage: float,
    true_frames: int,
    total_frames: int,
) -> list[str]:
    """Return the shared text rows used by boolean summaries."""
    return [
        f"{seconds:.1f}s / {percentage:.0f}%",
        f"({true_frames} / {total_frames} frames)",
    ]


def _boolean_summary_rows(
    manifest: InspectionManifest,
    result_column: str,
) -> list[str]:
    """Return summary rows for a boolean feature without drawing them."""
    meta = _find_heuristic_meta(manifest, result_column)
    if meta is not None:
        return _summary_rows(
            meta.seconds,
            meta.percentage,
            meta.true_frames,
            meta.total_frames,
        )

    if result_column not in manifest.df.columns:
        return []

    bool_idx = manifest.df[result_column].to_numpy().astype(bool)
    true_frames = int(bool_idx.sum())
    total_frames = len(bool_idx)
    seconds = true_frames / manifest.video.fps if manifest.video.fps else 0.0
    percentage = (true_frames / total_frames * 100) if total_frames else 0.0
    return _summary_rows(seconds, percentage, true_frames, total_frames)


def _add_boolean_summary(ax: Axes, manifest: InspectionManifest, result_column: str) -> None:
    """Compute summary stats for a boolean feature and annotate *ax*.

    Looks up pre-computed ``HeuristicMeta`` first; falls back to on-the-fly
    computation from the parquet column when no metadata match exists.
    """
    rows = _boolean_summary_rows(manifest, result_column)
    if not rows:
        return
    ax.text(
        0.98,
        0.02,
        "\n".join(rows),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )
