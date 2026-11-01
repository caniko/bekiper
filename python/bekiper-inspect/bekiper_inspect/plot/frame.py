"""Video-frame loading helpers shared by inspection plot modules.

Centralizes the OpenCV `VideoCapture` calls (formerly hidden inside
`scale.py`) so that `plot/scale.py` and `plot/feature.py` can both ask
for an arbitrary frame index without re-decoding the same frame twice.
A small LRU cache absorbs the common case where many feature
inspections target the same representative frame.

When the Rust ingest pipeline has pre-extracted representative frames
as PNGs (``{output}/frames/frame_XXXXXX.png``), the functions here
load those directly, avoiding the need for the source MP4 and the
overhead of video decoding.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def _find_pre_extracted_frame(video_path: str, index: int) -> np.ndarray | None:
    """Check for a pre-extracted PNG produced by the Rust ingest pipeline.

    The Rust side writes frames to ``{output}/frames/frame_{index:06}.png``
    where ``{output}`` is the experiment directory (the parent of the video
    file's directory, or — more reliably — the sibling ``frames/`` directory
    next to the inspection TOML). We derive the frames directory from the
    video path: ``video_dir/../frames/``.

    Returns the frame as an RGB numpy array, or ``None`` if the PNG does
    not exist.
    """
    video_dir = Path(video_path).resolve().parent
    # The output directory layout is: {output}/{experiment}/video.mp4
    # and frames live at {output}/{experiment}/frames/frame_XXXXXX.png
    # But video_path may also be *outside* the output tree. Try a few
    # plausible locations.
    candidates = [
        video_dir / "frames" / f"frame_{index:06d}.png",
        video_dir.parent / "frames" / f"frame_{index:06d}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            img = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def _find_pre_extracted_frame_in_dir(frames_dir: str, index: int) -> np.ndarray | None:
    """Load a pre-extracted PNG from an explicit frames directory."""
    png_path = Path(frames_dir) / f"frame_{index:06d}.png"
    if png_path.is_file():
        img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


@lru_cache(maxsize=8)
def _read_frame_cached(video_path: str, index: int) -> np.ndarray | None:
    """LRU-cached frame read keyed by (path, index).

    Checks for a pre-extracted PNG first, falling back to live video
    decode via OpenCV.

    Cache size 8 is intentionally small: a typical inspection run picks
    a handful of representative frames (one per feature) and most
    features collapse to the same frame, so even tiny cache pays off.
    """
    # Prefer pre-extracted PNG from the Rust ingest pipeline.
    pre = _find_pre_extracted_frame(video_path, index)
    if pre is not None:
        return pre

    cap = cv2.VideoCapture(video_path)
    try:
        if index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def read_frame_at(
    video_path: str | Path | None,
    index: int = 0,
    *,
    frames_dir: str | Path | None = None,
) -> np.ndarray | None:
    """Load a video frame, preferring pre-extracted PNGs.

    Parameters
    ----------
    video_path:
        Path to the source video file. May be ``None`` when the video is
        unavailable — in that case we still try *frames_dir*.
    index:
        Zero-based frame index.
    frames_dir:
        Explicit directory containing pre-extracted ``frame_XXXXXX.png``
        files. When provided, this takes priority over both the implicit
        PNG search and live video decode.

    Returns ``None`` when the frame cannot be loaded by any method.
    """
    idx = int(max(index, 0))

    # 1. Explicit frames directory (highest priority).
    if frames_dir is not None:
        frame = _find_pre_extracted_frame_in_dir(str(frames_dir), idx)
        if frame is not None:
            return frame

    # 2. Delegate to the cached loader (PNG search + video decode).
    if video_path is not None:
        return _read_frame_cached(str(video_path), idx)

    return None


def read_first_frame(video_path: str | Path | None) -> np.ndarray | None:
    """Convenience wrapper for the most common request — frame zero."""
    return read_frame_at(video_path, 0)
