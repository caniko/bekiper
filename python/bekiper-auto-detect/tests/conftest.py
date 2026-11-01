"""Shared fixtures for auto-detect tests."""

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_circle_mask() -> np.ndarray:
    """A 200x200 binary mask with a filled circle at (100, 100) radius 40."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 1, thickness=-1)
    return mask.astype(bool)


@pytest.fixture
def synthetic_rectangle_mask() -> np.ndarray:
    """A 200x200 binary mask with a filled rectangle."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (30, 50), (170, 150), 1, thickness=-1)
    return mask.astype(bool)


@pytest.fixture
def synthetic_arena_image() -> np.ndarray:
    """A 480x640 RGB image with two circular objects on a grey background.

    Object A at (160, 120) — top-left diagonal.
    Object B at (480, 360) — bottom-right diagonal.
    """
    img = np.full((480, 640, 3), 180, dtype=np.uint8)  # grey background
    cv2.circle(img, (160, 120), 30, (50, 50, 200), thickness=-1)  # red-ish
    cv2.circle(img, (480, 360), 30, (200, 50, 50), thickness=-1)  # blue-ish
    return img


@pytest.fixture
def two_diagonal_masks() -> list:
    """Two DetectedMask objects positioned diagonally."""
    from bekiper_auto_detect.model import DetectedMask

    mask_a = np.zeros((480, 640), dtype=bool)
    cv2.circle(mask_a.view(np.uint8), (160, 120), 30, 1, thickness=-1)

    mask_b = np.zeros((480, 640), dtype=bool)
    cv2.circle(mask_b.view(np.uint8), (480, 360), 30, 1, thickness=-1)

    return [
        DetectedMask(mask=mask_a, score=0.95, area=int(mask_a.sum()), bbox=(130, 90, 60, 60)),
        DetectedMask(mask=mask_b, score=0.92, area=int(mask_b.sum()), bbox=(450, 330, 60, 60)),
    ]


@pytest.fixture
def four_positional_masks() -> list:
    """Four DetectedMask objects in a grid pattern for OUL test."""
    from bekiper_auto_detect.model import DetectedMask

    positions = [(160, 120), (480, 120), (160, 360), (480, 360)]
    masks = []
    for x, y in positions:
        m = np.zeros((480, 640), dtype=bool)
        cv2.circle(m.view(np.uint8), (x, y), 25, 1, thickness=-1)
        masks.append(DetectedMask(mask=m, score=0.9, area=int(m.sum()), bbox=(x - 25, y - 25, 50, 50)))
    return masks
