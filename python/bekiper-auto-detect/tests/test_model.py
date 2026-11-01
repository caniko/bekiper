"""Tests for DetectedMask and DetectionResult."""

import numpy as np

from bekiper_auto_detect.model import DetectedMask, DetectionResult


class TestDetectedMask:
    def test_contour_extraction(self, synthetic_circle_mask):
        mask = DetectedMask(mask=synthetic_circle_mask, score=0.95, area=int(synthetic_circle_mask.sum()))
        assert len(mask.contour) > 0

    def test_circularity_of_circle(self, synthetic_circle_mask):
        mask = DetectedMask(mask=synthetic_circle_mask, score=0.95, area=int(synthetic_circle_mask.sum()))
        assert mask.circularity > 0.85, f"Circle circularity {mask.circularity} should be > 0.85"

    def test_circularity_of_rectangle(self, synthetic_rectangle_mask):
        mask = DetectedMask(mask=synthetic_rectangle_mask, score=0.95, area=int(synthetic_rectangle_mask.sum()))
        assert mask.circularity < 0.85, f"Rectangle circularity {mask.circularity} should be < 0.85"

    def test_fitted_circle(self, synthetic_circle_mask):
        mask = DetectedMask(mask=synthetic_circle_mask, score=0.95, area=int(synthetic_circle_mask.sum()))
        center, radius = mask.fitted_circle
        assert abs(center[0] - 100) < 3
        assert abs(center[1] - 100) < 3
        assert abs(radius - 40) < 3

    def test_centroid(self, synthetic_circle_mask):
        mask = DetectedMask(mask=synthetic_circle_mask, score=0.95, area=int(synthetic_circle_mask.sum()))
        cx, cy = mask.centroid
        assert abs(cx - 100) < 3
        assert abs(cy - 100) < 3

    def test_simplified_contour_has_fewer_points(self, synthetic_circle_mask):
        mask = DetectedMask(mask=synthetic_circle_mask, score=0.95, area=int(synthetic_circle_mask.sum()))
        assert len(mask.simplified_contour) < len(mask.contour)

    def test_empty_mask(self):
        mask = DetectedMask(mask=np.zeros((100, 100), dtype=bool), score=0.0, area=0)
        assert mask.circularity == 0.0
        assert mask.centroid == (0, 0)


class TestDetectionResult:
    def test_filter_by_area(self):
        masks = [
            DetectedMask(mask=np.zeros((10, 10), dtype=bool), score=0.9, area=100),
            DetectedMask(mask=np.zeros((10, 10), dtype=bool), score=0.8, area=5000),
            DetectedMask(mask=np.zeros((10, 10), dtype=bool), score=0.7, area=50),
        ]
        result = DetectionResult(masks=masks)
        filtered = result.filter_by_area(min_area=80)
        assert len(filtered) == 2

    def test_filter_by_score(self):
        masks = [
            DetectedMask(mask=np.zeros((10, 10), dtype=bool), score=0.9, area=100),
            DetectedMask(mask=np.zeros((10, 10), dtype=bool), score=0.5, area=100),
            DetectedMask(mask=np.zeros((10, 10), dtype=bool), score=0.3, area=100),
        ]
        result = DetectionResult(masks=masks)
        filtered = result.filter_by_score(min_score=0.6)
        assert len(filtered) == 1

    def test_top_k(self):
        masks = [
            DetectedMask(mask=np.zeros((10, 10), dtype=bool), score=0.9, area=100),
            DetectedMask(mask=np.zeros((10, 10), dtype=bool), score=0.8, area=5000),
            DetectedMask(mask=np.zeros((10, 10), dtype=bool), score=0.7, area=300),
        ]
        result = DetectionResult(masks=masks)
        top2 = result.top_k(2)
        assert len(top2) == 2
        # top_k sorts by area descending
        assert top2[0].area == 5000
        assert top2[1].area == 300

    def test_len_and_getitem(self):
        masks = [
            DetectedMask(mask=np.zeros((10, 10), dtype=bool), score=0.9, area=100),
        ]
        result = DetectionResult(masks=masks)
        assert len(result) == 1
        assert result[0].score == 0.9
