"""Experiment-specific detection recipes for behavioral neuroscience arenas.

Each recipe defines:
- How many objects to detect
- What backend/prompt to use
- How to assign semantic labels based on spatial rules
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bekiper_auto_detect.config import DetectorConfig
from bekiper_auto_detect.detect import auto_detect_perimeters, auto_detect_perimeters_with_text
from bekiper_auto_detect.mask_to_perimeter import Perimeter, mask_to_perimeter
from bekiper_auto_detect.model import DetectedMask, DetectionResult

logger = logging.getLogger(__name__)


class DetectionCountError(ValueError):
    """Raised when detection finds fewer/more objects than the recipe expects."""


class ExperimentDetectionRecipe(ABC):
    """Base class for experiment-specific detection flows.

    Recipes pull all model IDs, thresholds, and morphology parameters from
    a :class:`DetectorConfig`. Construct the recipe with a custom config to
    override defaults, or rely on :meth:`DetectorConfig.load` to pick up
    env/TOML values automatically.
    """

    expected_object_count: int
    text_prompt: str = "object"
    backend: str = "sam2_text"  # "sam2_auto" | "sam2_text" | "yoloe"
    min_area: int = 500

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config: DetectorConfig = config or DetectorConfig()

    def detect(self, image: np.ndarray[Any, Any]) -> DetectionResult:
        if self.backend == "sam2_auto":
            result = auto_detect_perimeters(image, config=self.config)
        elif self.backend == "sam2_text":
            result = auto_detect_perimeters_with_text(image, self.text_prompt, config=self.config)
        else:
            from bekiper_auto_detect.detect import auto_detect_perimeters_yoloe

            result = auto_detect_perimeters_yoloe(image, [self.text_prompt], config=self.config)

        result = result.filter_by_area(min_area=self.min_area)
        result = result.top_k(self.expected_object_count)

        if len(result) != self.expected_object_count:
            raise DetectionCountError(
                f"Expected {self.expected_object_count} objects, got {len(result)}. "
                f"Try a different frame (--frame), lower confidence (--confidence), "
                f"or check that objects are visible in the video."
            )

        return result

    def run(self, image_path: str | Path) -> dict[str, Perimeter]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self.detect(image)
        labeled_masks = self.assign_labels(result.masks)

        perimeters: dict[str, Perimeter] = {}
        for mask in labeled_masks:
            perimeters[mask.label] = mask_to_perimeter(mask)

        return perimeters

    @abstractmethod
    def assign_labels(self, masks: list[DetectedMask]) -> list[DetectedMask]:
        """Assign semantic labels to detected masks based on spatial rules."""
        ...


def _detect_dark_objects(
    image: np.ndarray,
    *,
    config: DetectorConfig | None = None,
) -> DetectionResult:
    """Detect dark objects on a light background.

    All spatial parameters scale with image dimensions so the algorithm
    works identically on a 224×212 thumbnail and a 1920×1080 full frame.

    Strategy:
    1. Percentile-based threshold (:attr:`DetectorConfig.floor_percentile`)
       to capture anything darker than the arena floor.
    2. Flood-fill border removal (black arena surround).
    3. Remove the mouse (largest blob > ``mouse_min_frac`` of image).
    4. Filter remaining contours by fractional area bounds.
    """
    cfg = config or DetectorConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    img_area = h * w
    diag = int(np.sqrt(img_area))  # characteristic length for kernel scaling

    min_area = int(img_area * cfg.min_area_frac)
    max_area = int(img_area * cfg.max_area_frac)
    mouse_min_area = int(img_area * cfg.mouse_min_frac)
    logger.info(f"Detection image: {w}x{h}, area bounds: {min_area}–{max_area}px, mouse>{mouse_min_area}px")

    # Threshold: floor is the brightest large region (floor_percentile);
    # objects are `threshold_factor` times darker.
    floor_val = float(np.percentile(gray, cfg.floor_percentile))
    threshold_val = floor_val * cfg.threshold_factor
    logger.info(f"Floor estimate (P{cfg.floor_percentile:g}): {floor_val:.0f}, threshold: {threshold_val:.0f}")
    _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # Remove black border by flood-filling from all four corners
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for corner in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(thresh, flood_mask, corner, 0)

    # Morphological close: kernel ≈ close_kernel_frac × diagonal.
    close_k = max(3, int(diag * cfg.close_kernel_frac)) | 1  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find and remove the mouse
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_areas = sorted([int(cv2.contourArea(c)) for c in contours], reverse=True)
    logger.info(f"Threshold found {len(contours)} contours, areas (desc): {all_areas[:20]}")

    mouse_contours = [c for c in contours if cv2.contourArea(c) >= mouse_min_area]
    if mouse_contours:
        mouse_mask = np.zeros_like(thresh)
        cv2.drawContours(mouse_mask, mouse_contours, -1, 255, cv2.FILLED)
        # Dilate by dilate_frac × diagonal to cover tail/shadow
        dilate_r = max(3, int(diag * cfg.dilate_frac)) | 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_r, dilate_r))
        mouse_mask = cv2.dilate(mouse_mask, dilate_kernel)
        thresh[mouse_mask > 0] = 0
        logger.info(f"Excluded {len(mouse_contours)} mouse region(s), dilate radius={dilate_r}px")

    # Remaining contours are candidate objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    remaining_areas = sorted([int(cv2.contourArea(c)) for c in contours], reverse=True)
    logger.info(f"After mouse removal: {len(contours)} contours, areas: {remaining_areas[:20]}")

    masks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        mask = np.zeros(gray.shape, dtype=bool)
        cv2.drawContours(mask.view(np.uint8), [cnt], -1, 1, cv2.FILLED)
        x, y, bw, bh = cv2.boundingRect(cnt)
        masks.append(DetectedMask(mask=mask, score=1.0, area=int(area), bbox=(x, y, bw, bh)))

    return DetectionResult(masks=masks)


def _diagonal_pair_assignment(masks: list[DetectedMask], labels: tuple[str, str]) -> list[DetectedMask]:
    """Assign labels based on diagonal position (sum of x+y coordinates)."""
    if len(masks) != 2:
        raise ValueError(f"Diagonal assignment requires exactly 2 masks, got {len(masks)}")

    scored = [(m, m.centroid[0] + m.centroid[1]) for m in masks]
    scored.sort(key=lambda x: x[1])

    result = []
    for (mask, _), label in zip(scored, labels):
        result.append(DetectedMask(mask=mask.mask, score=mask.score, label=label, area=mask.area, bbox=mask.bbox))
    return result


def _positional_assignment(masks: list[DetectedMask], labels: list[str]) -> list[DetectedMask]:
    """Assign labels sorted by position: top-to-bottom, then left-to-right."""
    if len(masks) != len(labels):
        raise ValueError(f"Got {len(masks)} masks but {len(labels)} labels")

    scored = [(m, (m.centroid[1], m.centroid[0])) for m in masks]  # (y, x)
    scored.sort(key=lambda x: x[1])

    result = []
    for (mask, _), label in zip(scored, labels):
        result.append(DetectedMask(mask=mask.mask, score=mask.score, label=label, area=mask.area, bbox=mask.bbox))
    return result


# ---- NORT recipes ----


class _NORTBase(ExperimentDetectionRecipe):
    """NORT base: classical CV dark-object detection with SAM2 auto fallback.

    Area bounds come from :class:`DetectorConfig` (``min_area_frac`` /
    ``max_area_frac``) so they scale with image resolution and are
    overridable via TOML / env / CLI.
    """

    expected_object_count = 2

    def detect(self, image: np.ndarray[Any, Any]) -> DetectionResult:
        result = _detect_dark_objects(image, config=self.config)
        logger.info(f"Dark-object detector found {len(result)} masks")

        # Take the smallest N masks — objects are smaller than stray detections
        sorted_masks = sorted(result.masks, key=lambda m: m.area)
        result = DetectionResult(masks=sorted_masks[: self.expected_object_count])

        if len(result) != self.expected_object_count:
            raise DetectionCountError(
                f"Expected {self.expected_object_count} objects, got {len(result)} "
                f"(area frac: {self.config.min_area_frac}–{self.config.max_area_frac}). "
                f"Try a different frame (--frame) or check that objects are visible."
            )
        return result


class NORTTrainingRecipe(_NORTBase):
    def assign_labels(self, masks: list[DetectedMask]) -> list[DetectedMask]:
        return _diagonal_pair_assignment(masks, ("variable", "familiar"))


class NORTNoveltyRecipe(_NORTBase):
    def assign_labels(self, masks: list[DetectedMask]) -> list[DetectedMask]:
        return _diagonal_pair_assignment(masks, ("novel", "familiar"))


# ---- OUL recipes ----


class _OULBase(_NORTBase):
    """OUL base: same SAM2 auto strategy as NORT."""


class OULTrainingRecipe(_OULBase):
    expected_object_count = 2

    def assign_labels(self, masks: list[DetectedMask]) -> list[DetectedMask]:
        return _positional_assignment(masks, ["object_1", "object_2"])


class OULUpdateRecipe(_OULBase):
    expected_object_count = 2

    def assign_labels(self, masks: list[DetectedMask]) -> list[DetectedMask]:
        return _positional_assignment(masks, ["object_1", "object_3"])


class OULTestRecipe(_OULBase):
    expected_object_count = 4

    def assign_labels(self, masks: list[DetectedMask]) -> list[DetectedMask]:
        return _positional_assignment(masks, ["object_1", "object_2", "object_3", "object_4"])


# ---- OFT recipe ----


class OFTRecipe(ExperimentDetectionRecipe):
    """Open Field Test: detect the square arena boundary.

    Detects 1 object (the arena floor/boundary) and labels it 'arena'.
    The Rust pipeline derives center and peripheral zones from the arena rectangle.
    """

    expected_object_count = 1
    text_prompt = "square arena floor"
    backend = "sam2_text"
    min_area = 10000  # Arena is large

    def assign_labels(self, masks: list[DetectedMask]) -> list[DetectedMask]:
        if not masks:
            raise ValueError("No arena detected for OFT")
        largest = max(masks, key=lambda m: m.area)
        return [
            DetectedMask(mask=largest.mask, score=largest.score, label="arena", area=largest.area, bbox=largest.bbox)
        ]

    def detect(self, image: np.ndarray[Any, Any]) -> DetectionResult:
        result = super().detect(image)
        if len(result) == 0:
            logger.warning("No arena detected, trying automatic detection")
            result = auto_detect_perimeters(image)
            result = result.filter_by_area(min_area=self.min_area)
        return result


# ---- Y-maze recipe ----


class YMazeRecipe(ExperimentDetectionRecipe):
    """Y-maze: detect the maze boundary.

    Detects 1 object (the Y-shaped maze) and labels it 'maze'.
    The Rust pipeline needs a RadialMaze perimeter with center + 3 arms,
    but auto-detection provides the raw mask. Post-processing in the pipeline
    or a separate step must decompose it into arms.
    """

    expected_object_count = 1
    text_prompt = "Y-shaped maze"
    backend = "sam2_text"
    min_area = 10000

    def assign_labels(self, masks: list[DetectedMask]) -> list[DetectedMask]:
        if not masks:
            raise ValueError("No maze detected for Y-maze")
        largest = max(masks, key=lambda m: m.area)
        return [
            DetectedMask(mask=largest.mask, score=largest.score, label="maze", area=largest.area, bbox=largest.bbox)
        ]

    def detect(self, image: np.ndarray[Any, Any]) -> DetectionResult:
        result = super().detect(image)
        if len(result) == 0:
            logger.warning("No maze detected, trying automatic detection")
            result = auto_detect_perimeters(image)
            result = result.filter_by_area(min_area=self.min_area)
        return result


RECIPE_REGISTRY: dict[str, type[ExperimentDetectionRecipe]] = {
    "nort_training": NORTTrainingRecipe,
    "nort_novelty": NORTNoveltyRecipe,
    "nort": NORTTrainingRecipe,
    "oul_training": OULTrainingRecipe,
    "oul_update": OULUpdateRecipe,
    "oul_test": OULTestRecipe,
    "oft": OFTRecipe,
    "open_field": OFTRecipe,
    "ymaze": YMazeRecipe,
    "y_maze": YMazeRecipe,
}
