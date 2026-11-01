"""Tests for mask-to-perimeter conversion."""

import numpy as np

from bekiper_auto_detect.mask_to_perimeter import (
    CirclePerimeter,
    PolygonPerimeter,
    mask_to_perimeter,
    masks_to_perimeter_set,
)
from bekiper_auto_detect.model import DetectedMask


class TestMaskToPerimeter:
    def test_circle_mask_becomes_circle_perimeter(self, synthetic_circle_mask):
        mask = DetectedMask(mask=synthetic_circle_mask, score=0.95, area=int(synthetic_circle_mask.sum()))
        perimeter = mask_to_perimeter(mask)
        assert isinstance(perimeter, CirclePerimeter)
        assert abs(perimeter.center_pixels[0] - 100) < 3
        assert abs(perimeter.center_pixels[1] - 100) < 3
        assert abs(perimeter.radius_pixels - 40) < 3

    def test_rectangle_mask_becomes_polygon_perimeter(self, synthetic_rectangle_mask):
        mask = DetectedMask(mask=synthetic_rectangle_mask, score=0.95, area=int(synthetic_rectangle_mask.sum()))
        perimeter = mask_to_perimeter(mask)
        assert isinstance(perimeter, PolygonPerimeter)
        assert perimeter.vertices_pixels.shape[0] >= 4

    def test_custom_circularity_threshold(self, synthetic_circle_mask):
        mask = DetectedMask(mask=synthetic_circle_mask, score=0.95, area=int(synthetic_circle_mask.sum()))
        # With impossibly high threshold, circle becomes polygon
        perimeter = mask_to_perimeter(mask, circularity_threshold=0.999)
        assert isinstance(perimeter, PolygonPerimeter)


class TestMasksToPerimeterSet:
    def test_basic_conversion(self, two_diagonal_masks):
        for i, m in enumerate(two_diagonal_masks):
            two_diagonal_masks[i] = DetectedMask(mask=m.mask, score=m.score, label=f"obj_{i}", area=m.area, bbox=m.bbox)
        result = masks_to_perimeter_set(two_diagonal_masks)
        assert len(result) == 2

    def test_restricted_labels(self, two_diagonal_masks):
        two_diagonal_masks[0] = DetectedMask(
            mask=two_diagonal_masks[0].mask,
            score=0.95,
            label="keep",
            area=two_diagonal_masks[0].area,
            bbox=two_diagonal_masks[0].bbox,
        )
        two_diagonal_masks[1] = DetectedMask(
            mask=two_diagonal_masks[1].mask,
            score=0.92,
            label="drop",
            area=two_diagonal_masks[1].area,
            bbox=two_diagonal_masks[1].bbox,
        )
        result = masks_to_perimeter_set(two_diagonal_masks, restricted_labels={"keep"})
        assert len(result) == 1
        assert "keep" in result

    def test_auto_labeling_when_no_label(self):
        import cv2

        masks = []
        for i in range(3):
            m = np.zeros((100, 100), dtype=bool)
            cv2.circle(m.view(np.uint8), (30 + i * 20, 50), 10, 1, thickness=-1)
            masks.append(DetectedMask(mask=m, score=0.9, label="", area=int(m.sum())))
        result = masks_to_perimeter_set(masks)
        assert len(result) == 3
        assert "object_0" in result
