from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from matplotlib import pyplot as plt
from matplotlib.quiver import Quiver

from bekiper_inspect.manifest import FeatureHeuristicDecomposition, HeuristicDecompositionComponent
from bekiper_inspect.plot.feature import plot_feature_on_frame
from bekiper_inspect.plot.feature._diagnostic import draw_ray_field, sample_indices
from bekiper_inspect.manifest import FeatureNortDiscrimination
from bekiper_inspect.plot.feature._preference import plot_preference


def test_zero_exploration_is_unavailable_instead_of_equal_preference():
    feature = FeatureNortDiscrimination(
        label="preference_summary",
        object_labels=["A", "B"],
        exploration_times=[0.0, 0.0],
        coordinate_labels=["nose"],
        description="preference",
    )
    fig = plot_preference(feature)
    assert not fig.axes[0].patches
    assert fig.axes[0].texts[0].get_text() == "No measured exploration"


def test_sampling_keeps_both_trial_endpoints_and_is_deterministic():
    indices = np.arange(10_000)
    sampled = sample_indices(indices)
    assert len(sampled) == 120
    assert sampled[0] == 0 and sampled[-1] == 9999
    np.testing.assert_array_equal(sampled, sample_indices(indices))


def test_ray_field_uses_all_states_and_metric_frame_alignment(sample_manifest):
    manifest = sample_manifest
    manifest.video.resolution = (400, 200)
    manifest.df = pl.DataFrame(
        {
            "nose_x": [0.01, 0.02, 0.03, 0.04, 0.05],
            "nose_y": [0.02] * 5,
            "dx": [1.0, -1.0, 0.0, 1.0, float("nan")],
            "dy": [0.0, 0.0, 1.0, 0.0, 0.0],
            "accepted": [True, False, True, None, True],
        }
    )
    component = HeuristicDecompositionComponent(
        label="heading",
        result_column="accepted",
        coordinate_labels=["nose"],
        ray_origin_label="nose",
        ray_direction_x_column="dx",
        ray_direction_y_column="dy",
    )
    fig, ax = plt.subplots()
    draw_ray_field(ax, manifest, component, (100, 100, 3))
    arrows = [c for c in ax.collections if isinstance(c, Quiver)]
    assert len(arrows) == 3  # accepted, rejected, unknown
    mpp = manifest.settings.meters_per_pixel
    np.testing.assert_allclose(arrows[0].X, np.array([0.01, 0.03]) / mpp * 0.25)
    np.testing.assert_allclose(arrows[0].Y, np.array([0.02, 0.02]) / mpp * 0.5)
    assert arrows[1].U[0] < 0
    assert arrows[0].V[1] > 0
    assert arrows[2].X[0] == 0.04 / mpp * 0.25
    assert "1 missing/zero" in ax.texts[0].get_text()


def test_standalone_diagnostic_renders_native_heading_histogram(sample_manifest):
    manifest = sample_manifest
    manifest.df = manifest.df.with_columns(
        pl.lit(1.0).alias("dx"), pl.lit(0.0).alias("dy"), pl.lit(23.0).alias("min_angle")
    )
    feature = FeatureHeuristicDecomposition(
        label="gaze_field",
        heuristic_name="diagnostic",
        result_column="olfaction",
        coordinate_labels=["nose"],
        description="Full trial",
        show_final_result=False,
        components=[
            HeuristicDecompositionComponent(
                label="direction",
                result_column="olfaction",
                coordinate_labels=["nose"],
                ray_origin_label="nose",
                ray_direction_x_column="dx",
                ray_direction_y_column="dy",
                value_column="min_angle",
                value_unit="degrees",
            )
        ],
    )
    fig = plot_feature_on_frame(manifest, feature)
    assert fig is not None
    assert len(fig.axes) == 2
    assert any(isinstance(c, Quiver) for c in fig.axes[0].collections)
    assert fig.axes[1].get_xlabel() == "degrees"
    assert "100 finite" in fig.axes[1].get_title()


@pytest.mark.parametrize("show_final", [False, True])
@pytest.mark.parametrize("scalar", [23.0, None])
def test_scalar_and_final_result_have_distinct_panel_slots(sample_manifest, show_final, scalar):
    manifest = sample_manifest
    manifest.df = manifest.df.with_columns(
        pl.lit(1.0).alias("dx"), pl.lit(0.0).alias("dy"), pl.lit(scalar, dtype=pl.Float64).alias("min_angle")
    )
    feature = FeatureHeuristicDecomposition(
        label="gaze_field",
        heuristic_name="diagnostic",
        result_column="olfaction",
        coordinate_labels=["nose"],
        description="Full trial",
        show_final_result=show_final,
        components=[
            HeuristicDecompositionComponent(
                label="direction",
                result_column="olfaction",
                coordinate_labels=["nose"],
                ray_origin_label="nose",
                ray_direction_x_column="dx",
                ray_direction_y_column="dy",
                value_column="min_angle",
                value_unit="degrees",
            )
        ],
    )
    fig = plot_feature_on_frame(manifest, feature)
    assert fig is not None
    panels = [ax for ax in fig.axes if ax.axison]
    assert len(panels) == 2 + int(show_final)
    assert len({ax.get_subplotspec() for ax in panels}) == len(panels)
    histogram = next(ax for ax in panels if ax.get_xlabel() == "degrees")
    assert histogram.get_title() == f"{100 if scalar is not None else 0} finite / 100 frames"
    fig.canvas.draw()
    for index, ax in enumerate(panels):
        assert not any(ax.get_position().overlaps(other.get_position()) for other in panels[index + 1 :])


def test_temporal_shading_does_not_join_excluded_frames():
    from bekiper_inspect.plot.generic import plot_temporal_shaded_line

    fig, ax = plt.subplots()
    plot_temporal_shaded_line(
        ax, np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]), np.array([0, 1, 9, 10]), 11
    )
    segments = ax.collections[0].get_segments()
    assert len(segments) == 2
    np.testing.assert_array_equal(segments[0], [[1.0, 1.0], [2.0, 1.0]])
    np.testing.assert_array_equal(segments[1], [[3.0, 1.0], [4.0, 1.0]])


def test_standalone_parent_uses_full_trial_bindings(sample_manifest):
    from bekiper_inspect.manifest import FeatureHeuristic

    manifest = sample_manifest
    manifest.df = manifest.df.with_columns(pl.lit(1.0).alias("dx"), pl.lit(0.0).alias("dy"))
    manifest.feature_inspections = [
        FeatureHeuristicDecomposition(
            label="heuristic_decomposition_olfaction",
            heuristic_name="olfaction",
            result_column="olfaction",
            coordinate_labels=["nose"],
            description="native",
            components=[
                HeuristicDecompositionComponent(
                    label="direction",
                    result_column="olfaction",
                    coordinate_labels=["nose"],
                    ray_origin_label="nose",
                    ray_direction_x_column="dx",
                    ray_direction_y_column="dy",
                )
            ],
        )
    ]
    parent = FeatureHeuristic(
        label="heuristic_olfaction", result_column="olfaction", coordinate_labels=["nose"], description="Olfaction"
    )
    fig = plot_feature_on_frame(manifest, parent)
    arrows = [c for c in fig.axes[0].collections if isinstance(c, Quiver)]
    assert sum(len(arrow.X) for arrow in arrows) == manifest.df.height
    assert len(fig.axes) == 1
    assert "Full trial" in fig.axes[0].texts[0].get_text()


def test_story_uses_cached_frame_without_source_video(sample_manifest, tmp_path):
    import cv2
    from bekiper_inspect.manifest import FeatureTrajectory
    from bekiper_inspect.plot.feature._story_parts.plot_story import _resolve_frame_for_feature

    manifest = sample_manifest
    manifest.frames_dir = str(tmp_path)
    manifest.video.path = None
    cv2.imwrite(str(tmp_path / "frame_000009.png"), np.full((23, 41, 3), 127, dtype=np.uint8))
    feature = FeatureTrajectory(
        label="tracking_test", coordinate_labels=["nose"], description="tracking", representative_frame_index=9
    )
    frame, index = _resolve_frame_for_feature(manifest, feature)
    assert index == 9
    assert frame.shape == (23, 41, 3)
    assert np.all(frame == 127)


def test_pose_quality_story_draws_scalar_traces_with_missing_gaps(sample_manifest):
    from bekiper_inspect.manifest import FeatureTimeSeries
    from bekiper_inspect.plot.feature._grouping import StoryGroup
    from bekiper_inspect.plot.feature._story import plot_story

    manifest = sample_manifest
    manifest.df = pl.DataFrame({"elapsed_s": [0.0, 0.2, 0.7, 1.5], "speed": [0.1, None, 0.2, 0.3]})
    feature = FeatureTimeSeries(
        label="speed", value_column="speed", unit="m/s", description="Native speed", representative_frame_index=2
    )
    fig = plot_story(manifest, StoryGroup(key="pose_quality", title="Pose Quality", features=[feature]))
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "speed (m/s)"
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.0, 0.2, 0.7, 1.5])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.1, np.nan, 0.2, 0.3], equal_nan=True)
    assert ax.lines[1].get_xdata() == [0.7, 0.7]
    assert not ax.images


@pytest.mark.parametrize("gradient", [False, True])
@pytest.mark.parametrize("valid_only", [False, True])
def test_shared_composite_preserves_isolated_observations_and_unknowns(sample_manifest, gradient, valid_only):
    from matplotlib.collections import LineCollection, PathCollection
    from bekiper_inspect.manifest import FeatureHeuristic
    from bekiper_inspect.plot.feature._renderers_parts._plot_heuristic_feature import _plot_heuristic_feature
    from bekiper_inspect.plot.feature._renderers_parts._plot_ymaze_arm_sections_feature import (
        _plot_heuristic_trajectory_only,
    )
    from bekiper_inspect.plot.theme import get_active_theme, set_active_theme

    manifest = sample_manifest
    manifest.df = pl.DataFrame(
        {
            "nose_x": [0.01, 0.02, 0.03, 0.04, 0.05, float("nan")],
            "nose_y": [0.02] * 6,
            "gate": [True, False, None, True, True, True],
        }
    )
    feature = FeatureHeuristic(label="gate", result_column="gate", coordinate_labels=["nose"], description="gate")
    original = get_active_theme()
    set_active_theme(original.with_overrides(trajectory_gradient=gradient))
    try:
        fig, axes = plt.subplots(1, 2)
        _plot_heuristic_feature(axes[0], manifest, feature, frame_shape=(480, 640, 3), valid_only=valid_only)
        _plot_heuristic_trajectory_only(axes[1], manifest, feature, (480, 640, 3), valid_only=valid_only)
        expected_indices = [0, 3, 4, 2] if valid_only else [1]
        expected = manifest.get_coordinates("nose")[expected_indices] / manifest.settings.meters_per_pixel
        for ax in axes:
            markers = [c.get_offsets() for c in ax.collections if isinstance(c, PathCollection)]
            assert markers, "isolated observations must remain visible"
            np.testing.assert_allclose(np.concatenate(markers), expected)
            segments = [
                segment for c in ax.collections if isinstance(c, LineCollection) for segment in c.get_segments()
            ]
            segments.extend(np.column_stack([line.get_xdata(), line.get_ydata()]) for line in ax.lines)
            assert len(segments) == int(valid_only)
            if valid_only:
                np.testing.assert_allclose(segments[0], expected[1:3])
    finally:
        set_active_theme(original)


@pytest.mark.parametrize("pixel_geometry", [False, True])
@pytest.mark.parametrize(
    "frame_shape, expected_center, expected_size",
    [
        ((100, 100, 3), (25.0, 50.0), (10.0, 20.0)),
        ((100, 200, 3), (50.0, 50.0), (20.0, 20.0)),
    ],
)
def test_circle_geometry_uses_full_frame_transform(pixel_geometry, frame_shape, expected_center, expected_size):
    from bekiper_inspect.manifest import PerimeterSpec
    from bekiper_inspect.plot.perimeter import plot_perimeter_on_ax, plot_pixel_perimeter_on_ax

    factor = 1000.0 if pixel_geometry else 1.0
    perimeter = PerimeterSpec(
        label="native",
        shape="circle",
        params={"center_x": 0.1 * factor, "center_y": 0.1 * factor, "radius": 0.02 * factor},
    )
    fig, ax = plt.subplots()
    if pixel_geometry:
        plot_pixel_perimeter_on_ax(ax, perimeter, video_resolution=(400, 200), frame_shape=frame_shape)
    else:
        plot_perimeter_on_ax(
            ax, perimeter, meters_per_pixel=0.001, as_pixels=True, video_resolution=(400, 200), frame_shape=frame_shape
        )
    np.testing.assert_allclose(ax.patches[0].center, expected_center)
    np.testing.assert_allclose([ax.patches[0].width, ax.patches[0].height], expected_size)


@pytest.mark.parametrize(
    "shape, params",
    [
        ("circle", {"center_x": 0.1, "center_y": 0.1, "radius": 0.02}),
        ("rectangle", {"center_x": 0.1, "center_y": 0.1, "width": 0.04, "height": 0.04}),
    ],
)
def test_proximity_buffer_uses_both_frame_scales(sample_manifest, shape, params):
    from bekiper_inspect.manifest import FeatureProximity, PerimeterSpec
    from bekiper_inspect.plot.feature._renderers_parts._plot_raycast_feature import _plot_proximity_feature

    manifest = sample_manifest
    manifest.video.resolution = (400, 200)
    manifest.perimeters = [PerimeterSpec(label="object", shape=shape, params=params)]
    manifest.pixel_perimeters = []
    feature = FeatureProximity(
        label="buffer",
        perimeter_label="object",
        buffer_meters=0.02,
        result_column="olfaction",
        coordinate_labels=["nose"],
        description="buffer",
    )
    fig, ax = plt.subplots()
    _plot_proximity_feature(ax, manifest, feature, frame_shape=(100, 100, 3))
    assert len(ax.patches) == 2
    np.testing.assert_allclose([ax.patches[0].get_width(), ax.patches[0].get_height()], [10.0, 20.0])
    np.testing.assert_allclose([ax.patches[1].get_width(), ax.patches[1].get_height()], [20.0, 40.0])
