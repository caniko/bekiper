"""Convert detected masks to geometric perimeter representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bekiper_auto_detect.model import DetectedMask

CIRCULARITY_THRESHOLD = 0.85


@dataclass(frozen=True)
class CirclePerimeter:
    """Circle defined by center and radius in pixels."""

    center_pixels: tuple[int, int]
    radius_pixels: float
    label: str = ""


@dataclass(frozen=True)
class PolygonPerimeter:
    """Polygon defined by vertices in pixels."""

    vertices_pixels: np.ndarray  # (N, 2) float64
    label: str = ""


Perimeter = CirclePerimeter | PolygonPerimeter


def mask_to_perimeter(mask: DetectedMask, *, circularity_threshold: float = CIRCULARITY_THRESHOLD) -> Perimeter:
    """Convert a DetectedMask to a CirclePerimeter or PolygonPerimeter based on circularity."""
    if mask.circularity >= circularity_threshold:
        center, radius = mask.fitted_circle
        return CirclePerimeter(center_pixels=center, radius_pixels=radius, label=mask.label)

    vertices = mask.simplified_contour.reshape(-1, 2).astype(np.float64)
    return PolygonPerimeter(vertices_pixels=vertices, label=mask.label)


def masks_to_perimeter_set(
    masks: list[DetectedMask],
    *,
    circularity_threshold: float = CIRCULARITY_THRESHOLD,
    restricted_labels: set[str] | None = None,
) -> dict[str, Perimeter]:
    """Convert multiple masks to a label->perimeter dict."""
    result: dict[str, Perimeter] = {}
    for mask in masks:
        if restricted_labels and mask.label not in restricted_labels:
            continue
        perimeter = mask_to_perimeter(mask, circularity_threshold=circularity_threshold)
        label = mask.label or f"object_{len(result)}"
        result[label] = perimeter
    return result
