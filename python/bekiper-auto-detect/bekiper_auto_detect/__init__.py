"""AI-powered object detection for behavioral neuroscience arenas."""

from bekiper_auto_detect.detect import (
    auto_detect_perimeters,
    auto_detect_perimeters_with_text,
    auto_detect_perimeters_yoloe,
)
from bekiper_auto_detect.mask_to_perimeter import mask_to_perimeter, masks_to_perimeter_set
from bekiper_auto_detect.model import DetectedMask, DetectionResult

__all__ = [
    "DetectedMask",
    "DetectionResult",
    "auto_detect_perimeters",
    "auto_detect_perimeters_with_text",
    "auto_detect_perimeters_yoloe",
    "mask_to_perimeter",
    "masks_to_perimeter_set",
]
