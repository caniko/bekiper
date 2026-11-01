"""Pydantic data models for detection results."""

from __future__ import annotations

from functools import cached_property

import cv2
import numpy as np
from pydantic import BaseModel, computed_field


class DetectedMask(BaseModel):
    """A single detected object mask with metadata."""

    model_config = {"arbitrary_types_allowed": True}

    mask: np.ndarray  # (H, W) bool
    score: float
    label: str = ""
    area: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def contour(self) -> np.ndarray:
        """Largest contour extracted from the binary mask."""
        contours, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.empty((0, 1, 2), dtype=np.int32)
        return max(contours, key=cv2.contourArea)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def simplified_contour(self) -> np.ndarray:
        """Douglas-Peucker simplified contour."""
        epsilon = 0.02 * cv2.arcLength(self.contour, closed=True)
        result: np.ndarray = cv2.approxPolyDP(self.contour, epsilon, closed=True)
        return result

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def circularity(self) -> float:
        """4*pi*area / perimeter^2. Perfect circle = 1.0."""
        perimeter = cv2.arcLength(self.contour, closed=True)
        if perimeter == 0:
            return 0.0
        contour_area = cv2.contourArea(self.contour)
        return float((4 * np.pi * contour_area) / (perimeter**2))

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def fitted_circle(self) -> tuple[tuple[int, int], float]:
        """Best-fit circle (center, radius) in pixels.

        Uses the contour centroid (center of mass) as center and the
        median distance from contour points to that centroid as radius.
        This is more robust than ``cv2.minEnclosingCircle`` which shifts
        the center toward contour asymmetries — common with imperfect
        segmentation masks.
        """
        if len(self.contour) < 3:
            return (0, 0), 0.0
        cx, cy = self.centroid
        # Median radial distance from centroid to contour points.
        pts = self.contour.reshape(-1, 2).astype(np.float64)
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        radius = float(np.median(dists))
        return (cx, cy), radius

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def centroid(self) -> tuple[int, int]:
        """Centroid of the contour."""
        m = cv2.moments(self.contour)
        if m["m00"] == 0:
            return (0, 0)
        return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))


class DetectionResult(BaseModel):
    """Collection of detected masks with filtering helpers."""

    masks: list[DetectedMask]

    def filter_by_area(self, min_area: int = 0, max_area: int = int(1e9)) -> DetectionResult:
        return DetectionResult(masks=[m for m in self.masks if min_area <= m.area <= max_area])

    def filter_by_score(self, min_score: float = 0.0) -> DetectionResult:
        return DetectionResult(masks=[m for m in self.masks if m.score >= min_score])

    def top_k(self, k: int) -> DetectionResult:
        sorted_masks = sorted(self.masks, key=lambda m: m.area, reverse=True)
        return DetectionResult(masks=sorted_masks[:k])

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, idx: int) -> DetectedMask:
        return self.masks[idx]
