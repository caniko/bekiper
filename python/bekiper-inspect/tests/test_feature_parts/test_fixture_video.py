"""Tests for the per-feature inspection figure pipeline.

Covers:

- ``find_representative_frame`` — heuristic-driven middle-frame picker.
- ``_pick_frame_for_feature`` — fallback chain (rep-frame hint → boolean
  → median traj → 0).
- ``plot_feature_on_frame`` dispatcher for every variant of
  ``FeatureInspection``, asserting the figure was rendered with a video
  frame underlay (an `imshow` AxesImage on the axis) plus the expected
  number of overlay artists.
- Golden-fixture regression: every variant (Perimeter, Zone, Quadrant,
  Heuristic, Angle, Trajectory) round-trips through the Rust→TOML→Python
  path and decodes to the correct dataclass.
- Frame caching reuse across multiple features that target the same frame.
"""

from __future__ import annotations

# This module intentionally re-exports shared test fixtures and helpers.
# ruff: noqa: F401
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np
import polars as pl
import pytest
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from bekiper_inspect.manifest import (
    CoordinateColumnSpec,
    FeatureAngle,
    FeatureArmSequence,
    FeatureCenterPeriphery,
    FeatureHeuristic,
    FeatureHeuristicCloud,
    FeatureHeuristicDecomposition,
    FeatureNortDiscrimination,
    FeaturePerimeter,
    FeatureProximity,
    FeatureQuadrant,
    FeatureRayCast,
    FeatureScaleDerivation,
    FeatureTimeSeries,
    FeatureTrajectory,
    FeatureYMazeArmSections,
    FeatureZone,
    HeuristicDecompositionComponent,
    HeuristicMeta,
    InspectionManifest,
    InspectionSettings,
    PerimeterSpec,
    QuadrantSpec,
    ScaleDerivation,
    VideoMeta,
    find_representative_frame,
)
from bekiper_inspect.plot.feature import (
    _pick_frame_for_feature,
    plot_feature_on_frame,
)
from bekiper_inspect.plot.feature._angle import (
    AngleOverlaySample,
    build_heuristic_angle_overlay,
    draw_angle_pair_labels,
)
from bekiper_inspect.plot.feature._renderers import _plot_heuristic_feature
from bekiper_inspect.plot.frame import _read_frame_cached, read_frame_at
from bekiper_inspect.plot.theme import PlotTheme

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fixture_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a tiny synthetic video so frame-reading code paths can run.

    The video is 30 frames at 320x240, each frame a unique solid color so
    `read_frame_at(idx)` can be sanity-checked by inspecting pixel value.
    """
    path = tmp_path_factory.mktemp("video") / "fixture.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
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
    """A small in-memory manifest with one perimeter, one quadrant, one
    boolean column, and a snout/tail_base body axis.
    """
    n = 30
    # Trajectory walks across the arena from left to right.
    nose_x = np.linspace(0.05, 0.25, n)
    nose_y = np.full(n, 0.15)
    tail_x = nose_x - 0.04
    tail_y = nose_y + 0.005
    # in_center: True for the middle 10 frames (frame indices 10..19).
    in_center = np.zeros(n, dtype=bool)
    in_center[10:20] = True
    freezing = np.zeros(n, dtype=bool)
    freezing[12:18] = True

    df = pl.DataFrame(
        {
            "nose_x": nose_x,
            "nose_y": nose_y,
            "tail_base_x": tail_x,
            "tail_base_y": tail_y,
            "in_center": in_center,
            "freezing": freezing,
            "heading_rad": np.linspace(0.0, 2 * np.pi, n),
            "near_center": in_center,
        }
    )

    return InspectionManifest(
        video=VideoMeta(
            fps=30.0,
            total_frames=n,
            resolution=(320, 240),
            meters_per_pixel=0.001,
            path=str(fixture_video),
        ),
        labels=["nose", "tail_base"],
        label_colors={"nose": "#1f77b4"},
        coordinate_columns={
            "nose": CoordinateColumnSpec(x="nose_x", y="nose_y"),
            "tail_base": CoordinateColumnSpec(x="tail_base_x", y="tail_base_y"),
        },
        perimeters=[
            PerimeterSpec(
                label="center",
                shape="circle",
                params={"center_x": 0.15, "center_y": 0.15, "radius": 0.05},
            )
        ],
        heuristics=[
            HeuristicMeta(
                name="in_center",
                result_column="in_center",
                true_frames=10,
                total_frames=n,
                seconds=10 / 30,
            ),
            HeuristicMeta(
                name="freezing",
                result_column="freezing",
                true_frames=6,
                total_frames=n,
                seconds=6 / 30,
            ),
        ],
        settings=InspectionSettings(
            minimum_seconds_tolerance=0.5,
            maximum_seconds_distraction=1.0 / 3.0,
            meters_per_pixel=0.001,
        ),
        scale_derivation=None,
        df=df,
        quadrants=[
            QuadrantSpec(
                label="grid",
                x_min=0.0,
                y_min=0.0,
                x_max=0.3,
                y_max=0.3,
                rows=2,
                cols=2,
            )
        ],
        feature_inspections=[],
    )


# ── find_representative_frame ────────────────────────────────────────────


class TestFindRepresentativeFrame:
    def test_returns_none_for_all_false(self):
        assert find_representative_frame(np.zeros(20, dtype=bool)) is None

    def test_returns_none_for_empty(self):
        assert find_representative_frame(np.array([], dtype=bool)) is None

    def test_picks_middle_of_only_true_span(self):
        arr = np.zeros(10, dtype=bool)
        arr[2:8] = True  # length-6 span at [2, 8)
        # middle of [2, 8) → indices (2 + 7) // 2 = 4
        assert find_representative_frame(arr) == 4

    def test_picks_longest_of_multiple_spans(self):
        arr = np.array(
            [True, True, False, True, True, True, True, False, True, False],
            dtype=bool,
        )
        # Longest span is [3, 7), length 4 → middle (3 + 6) // 2 = 4
        assert find_representative_frame(arr) == 4

    def test_single_true_frame(self):
        arr = np.zeros(5, dtype=bool)
        arr[3] = True
        assert find_representative_frame(arr) == 3


# ── _pick_frame_for_feature fallbacks ───────────────────────────────────


class TestPickFrameForFeature:
    def test_heuristic_with_boolean_picks_middle_of_true_span(self, manifest: InspectionManifest):
        feature = FeatureHeuristic(
            label="heuristic_in_center",
            result_column="in_center",
            coordinate_labels=["nose"],
            description="x",
        )
        # in_center True for [10, 20) → middle (10 + 19) // 2 = 14
        assert _pick_frame_for_feature(manifest, feature) == 14

    def test_perimeter_falls_back_to_median_trajectory_frame(self, manifest: InspectionManifest):
        feature = FeaturePerimeter(
            label="perimeter_center",
            perimeter_label="center",
            description="x",
        )
        # No boolean column, all 30 frames non-null → median = 14
        assert _pick_frame_for_feature(manifest, feature) == 14

    def test_clamps_to_total_frames(self, manifest: InspectionManifest):
        # Build a manifest where total_frames is small but coordinates are long
        manifest.video.total_frames = 5
        feature = FeaturePerimeter(
            label="perimeter_center",
            perimeter_label="center",
            description="x",
        )
        idx = _pick_frame_for_feature(manifest, feature)
        assert 0 <= idx < 5

    def test_explicit_representative_frame_overrides_heuristic(self, manifest: InspectionManifest):
        # The heuristic's longest-True-span midpoint would pick frame 14,
        # but an explicit representative_frame_index wins over it.
        feature = FeatureHeuristic(
            label="heuristic_in_center",
            result_column="in_center",
            coordinate_labels=["nose"],
            description="x",
            representative_frame_index=7,
        )
        assert _pick_frame_for_feature(manifest, feature) == 7

    def test_explicit_representative_frame_is_clamped(self, manifest: InspectionManifest):
        # Out-of-range hint is clamped to total_frames - 1 rather than
        # raising, so bad Rust-side inputs don't crash the inspector.
        feature = FeaturePerimeter(
            label="perimeter_center",
            perimeter_label="center",
            description="x",
            representative_frame_index=9999,
        )
        total = manifest.video.total_frames
        assert _pick_frame_for_feature(manifest, feature) == total - 1


# ── plot_feature_on_frame dispatcher ─────────────────────────────────────


def _has_imshow(fig: Figure) -> bool:
    return any(isinstance(art, AxesImage) for ax in fig.axes for art in ax.images)


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


def _line_rgba(line: object) -> np.ndarray:
    return np.asarray(matplotlib.colors.to_rgba(line.get_color()), dtype=float)


def _is_neutral_gray(rgba: np.ndarray, *, atol: float = 0.03) -> bool:
    return bool(np.allclose(rgba[0], rgba[1], atol=atol) and np.allclose(rgba[1], rgba[2], atol=atol))


def _has_representative_angle_marker(ax: object) -> bool:
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


__all__ = [name for name in globals() if not name.startswith(("__", "Test"))]
