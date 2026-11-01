"""Tests for the multi-panel story renderer.

Covers:
- Geometry story renders a single-panel figure.
- Heuristic story (multi-panel) renders expected number of axes.
- Standalone heuristic story with timeline strip.
- Task-specific stories delegate to existing renderers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np
import polars as pl
import pytest
from matplotlib.figure import Figure

from bekiper_inspect.manifest import (
    CoordinateColumnSpec,
    HeuristicMeta,
    InspectionManifest,
    InspectionSettings,
    PerimeterSpec,
    VideoMeta,
)
from bekiper_inspect.plot.theme import PlotTheme, set_active_theme


def _quiver_positions(quiver: object) -> np.ndarray:
    x = getattr(quiver, "X", None)
    y = getattr(quiver, "Y", None)
    if x is not None and y is not None:
        return np.column_stack(
            [
                np.asarray(x, dtype=float).ravel(),
                np.asarray(y, dtype=float).ravel(),
            ]
        )
    offsets = np.asarray(quiver.get_offsets(), dtype=float)
    return offsets.reshape(-1, 2)


def _unique_position_count(points: np.ndarray) -> int:
    if points.size == 0:
        return 0
    rounded = np.round(points.astype(float), 3)
    return int(np.unique(rounded, axis=0).shape[0])


def _quiver_rgba(quiver: object) -> np.ndarray:
    colors = np.asarray(quiver.get_facecolors(), dtype=float)
    if colors.size:
        return colors[0]
    colors = np.asarray(quiver.get_edgecolors(), dtype=float)
    if colors.size:
        return colors[0]
    return np.asarray(matplotlib.colors.to_rgba("black"), dtype=float)


def _quiver_lengths(quiver: object) -> np.ndarray:
    return np.hypot(
        np.asarray(quiver.U, dtype=float),
        np.asarray(quiver.V, dtype=float),
    )


def _is_pair_angle_label(text: str) -> bool:
    if not text.endswith("°"):
        return False
    try:
        float(text[:-1])
    except ValueError:
        return False
    return True


def _pair_angle_texts(ax: object) -> list[object]:
    return [text for text in getattr(ax, "texts", []) if _is_pair_angle_label(text.get_text())]


def _angle_label_bboxes(fig: Figure, ax: object) -> list[object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [text.get_window_extent(renderer=renderer) for text in _pair_angle_texts(ax)]


def _assert_no_angle_label_overlap(fig: Figure, ax: object) -> None:
    bboxes = _angle_label_bboxes(fig, ax)
    for idx, bbox in enumerate(bboxes):
        for other in bboxes[idx + 1 :]:
            overlap = type(bbox).intersection(bbox, other)
            assert overlap is None or overlap.width * overlap.height <= 1e-6


def _axis_line_display_lengths(ax: object) -> list[float]:
    lengths: list[float] = []
    for line in getattr(ax, "lines", []):
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size < 2 or y.size < 2:
            continue
        points = np.column_stack([x, y])
        display_points = ax.transData.transform(points)
        segment_lengths = np.linalg.norm(np.diff(display_points, axis=0), axis=1)
        lengths.append(float(segment_lengths.sum()))
    return lengths


def _long_axis_lines(ax: object, *, min_length_px: float = 14.0) -> list[object]:
    return [
        line
        for line, length in zip(getattr(ax, "lines", []), _axis_line_display_lengths(ax))
        if length >= min_length_px
    ]


def _has_representative_angle_marker(ax: object) -> bool:
    from matplotlib.collections import PathCollection

    for coll in getattr(ax, "collections", []):
        if not isinstance(coll, PathCollection):
            continue
        offsets = np.asarray(coll.get_offsets(), dtype=float)
        if offsets.shape[0] != 1:
            continue
        edgecolors = np.asarray(coll.get_edgecolors(), dtype=float)
        linewidths = np.asarray(coll.get_linewidths(), dtype=float)
        if not edgecolors.size:
            continue
        if np.allclose(edgecolors[0][:3], [1.0, 1.0, 1.0], atol=0.03) and (
            linewidths.size == 0 or float(np.max(linewidths)) >= 0.7
        ):
            return True
    return False


def _circle_linewidths(ax: object) -> list[float]:
    return [
        float(patch.get_linewidth())
        for patch in getattr(ax, "patches", [])
        if isinstance(patch, matplotlib.patches.Circle)
    ]


@pytest.fixture(scope="module")
def fixture_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "fixture.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    try:
        for i in range(30):
            frame = np.full((240, 320, 3), (i * 8) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def manifest(fixture_video: Path) -> InspectionManifest:
    n = 30
    nose_x = np.linspace(0.05, 0.25, n)
    nose_y = np.full(n, 0.15)
    tail_x = nose_x - 0.04
    tail_y = nose_y + 0.005
    body_prox = np.zeros(n, dtype=bool)
    body_prox[5:15] = True
    olfaction = np.zeros(n, dtype=bool)
    olfaction[8:18] = True

    df = pl.DataFrame(
        {
            "nose_x": nose_x,
            "nose_y": nose_y,
            "tail_base_x": tail_x,
            "tail_base_y": tail_y,
            "body_proximity_object_1": body_prox,
            "olfaction_object_1": olfaction,
            "freezing": np.zeros(n, dtype=bool),
            "heading_rad": np.linspace(0.0, 2 * np.pi, n),
        }
    )

    set_active_theme(PlotTheme())

    return InspectionManifest(
        video=VideoMeta(
            fps=30.0,
            total_frames=n,
            resolution=(320, 240),
            meters_per_pixel=0.001,
            path=str(fixture_video),
        ),
        labels=["nose", "tail_base"],
        label_colors={},
        coordinate_columns={
            "nose": CoordinateColumnSpec(x="nose_x", y="nose_y"),
            "tail_base": CoordinateColumnSpec(x="tail_base_x", y="tail_base_y"),
        },
        perimeters=[
            PerimeterSpec(
                label="object_1", shape="circle", params={"center_x": 0.15, "center_y": 0.15, "radius": 0.02}
            ),
            PerimeterSpec(label="arena", shape="circle", params={"center_x": 0.15, "center_y": 0.15, "radius": 0.14}),
        ],
        heuristics=[
            HeuristicMeta(
                name="body_proximity_object_1",
                result_column="body_proximity_object_1",
                true_frames=10,
                total_frames=n,
                seconds=10 / 30,
            ),
            HeuristicMeta(
                name="olfaction_object_1",
                result_column="olfaction_object_1",
                true_frames=10,
                total_frames=n,
                seconds=10 / 30,
            ),
            HeuristicMeta(name="freezing", result_column="freezing", true_frames=0, total_frames=n, seconds=0.0),
        ],
        settings=InspectionSettings(
            minimum_seconds_tolerance=0.5, maximum_seconds_distraction=1 / 3, meters_per_pixel=0.001
        ),
        scale_derivation=None,
        df=df,
        quadrants=[],
        feature_inspections=[],
    )


def _make_dense_angle_manifest(*, two_objects: bool = False) -> InspectionManifest:
    n = 240
    angles = np.full(n, 60.0)
    angles[119] = 3.0
    coords_x = 0.10 + (np.arange(n) % 25) * 0.03
    coords_y = 0.10 + ((np.arange(n) // 5) % 5) * 0.03
    cols: dict[str, object] = {
        "nose_x": coords_x,
        "nose_y": coords_y,
        "snout_dx": np.cos(np.deg2rad(angles)),
        "snout_dy": np.sin(np.deg2rad(angles)),
        "olfaction_object_1": np.ones(n, dtype=bool),
    }
    heuristics = [
        HeuristicMeta(
            name="olfaction_object_1",
            result_column="olfaction_object_1",
            true_frames=n,
            total_frames=n,
            seconds=n / 30.0,
        )
    ]
    perimeters = [
        PerimeterSpec(
            label="object_1",
            shape="circle",
            params={"center_x": 0.20, "center_y": 0.15, "radius": 0.02},
        ),
        PerimeterSpec(
            label="arena",
            shape="circle",
            params={"center_x": 0.20, "center_y": 0.15, "radius": 0.14},
        ),
    ]
    if two_objects:
        cols["olfaction_object_2"] = np.ones(n, dtype=bool)
        heuristics.append(
            HeuristicMeta(
                name="olfaction_object_2",
                result_column="olfaction_object_2",
                true_frames=n,
                total_frames=n,
                seconds=n / 30.0,
            )
        )
        perimeters.insert(
            1,
            PerimeterSpec(
                label="object_2",
                shape="circle",
                params={"center_x": 0.28, "center_y": 0.15, "radius": 0.02},
            ),
        )

    return InspectionManifest(
        video=VideoMeta(
            fps=30.0,
            total_frames=n,
            resolution=(640, 480),
            meters_per_pixel=0.001,
            path=None,
        ),
        labels=["nose"],
        label_colors={},
        coordinate_columns={"nose": CoordinateColumnSpec(x="nose_x", y="nose_y")},
        perimeters=perimeters,
        heuristics=heuristics,
        settings=InspectionSettings(
            minimum_seconds_tolerance=0.5,
            maximum_seconds_distraction=1 / 3,
            meters_per_pixel=0.001,
        ),
        scale_derivation=None,
        df=pl.DataFrame(cols),
        quadrants=[],
        feature_inspections=[],
    )


__all__ = [name for name in globals() if not name.startswith("__")]
