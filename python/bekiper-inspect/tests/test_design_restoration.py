import numpy as np
import polars as pl
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from bekiper_inspect.manifest import (
    FeatureHeuristicDecomposition,
    FeatureNortDiscrimination,
    FeatureRayCast,
    FeatureTimeSeries,
    HeuristicDecompositionComponent,
)
from bekiper_inspect.plot.feature import plot_feature_on_frame
from bekiper_inspect.plot.generic import ax_hue_plot_coordinate_with_boolean_index, save_figure
from bekiper_inspect.plot.theme import FigureProfileRule, PlotTheme, ProfileConfig, get_active_theme, set_active_theme


def test_numeric_object_identity_matches_native_for_unbounded_and_non_ascii_labels():
    from bekiper_inspect.plot.color import identity_color

    for label, expected in [
        ("object_2", "#E69F00"),
        ("object_0002", "#E69F00"),
        ("object_18446744073709551616", "#332288"),
        ("object_²", "#332288"),
        ("object_+2", "#009E73"),
    ]:
        assert identity_color(label) == expected


def test_routed_theme_survives_save_outside_render_context(sample_manifest, tmp_path):
    from PIL import Image

    previous = get_active_theme()
    theme = PlotTheme(
        min_dpi=300,
        profiles=ProfileConfig(
            presets={"small": {"min_dpi": 90}}, rules=[FigureProfileRule(profile="small", kind="ray_cast")]
        ),
    )
    set_active_theme(theme)
    try:
        feature = FeatureRayCast(
            label="cone",
            origin_label="nose",
            angle_column="heading_rad",
            length_meters=0.03,
            cone_width_deg=60,
            description="cone",
        )
        fig = plot_feature_on_frame(sample_manifest, feature)
        assert get_active_theme() is theme
        path = save_figure(fig, tmp_path / "routed", fmt=".png")
        with Image.open(path) as image:
            assert image.info["dpi"][0] == pytest.approx(90, abs=0.02)
    finally:
        set_active_theme(previous)


def test_legacy_nullable_boolean_helper_keeps_isolated_and_unknown_observations():
    coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _fig, ax = plt.subplots()
    ax_hue_plot_coordinate_with_boolean_index(
        ax, np.array([False, True, None], dtype=object), coords, plot_line=True, plot_non_confinement=True
    )
    markers = np.concatenate([c.get_offsets() for c in ax.collections])
    assert {tuple(p) for p in markers} == {tuple(p) for p in coords}
    assert not ax.lines


def test_affine_cone_transforms_heading_and_boundary_together(sample_manifest):
    from bekiper_inspect.plot.feature._renderers_parts._plot_raycast_feature import _plot_raycast_feature

    sample_manifest.video.resolution = (400, 200)
    sample_manifest.df = pl.DataFrame({"nose_x": [0.1], "nose_y": [0.1], "heading_rad": [np.pi / 4]})
    feature = FeatureRayCast(
        label="cone",
        origin_label="nose",
        angle_column="heading_rad",
        length_meters=0.02,
        cone_width_deg=60,
        description="cone",
    )
    _fig, ax = plt.subplots()
    _plot_raycast_feature(ax, sample_manifest, feature, 0, (100, 100, 3))
    origin = np.array([0.1, 0.1]) / sample_manifest.settings.meters_per_pixel * [0.25, 0.5]
    delta = 0.02 / sample_manifest.settings.meters_per_pixel * np.array([0.25, 0.5]) / np.sqrt(2)
    np.testing.assert_allclose([ax.lines[0].get_xdata()[-1], ax.lines[0].get_ydata()[-1]], origin + delta)
    transformed = ax.transData.inverted().transform(ax.patches[0].get_transform().transform([[0, 0]]))
    np.testing.assert_allclose(transformed[0], origin)


def test_conditional_pie_and_geometry_share_identity_styling(sample_manifest):
    from bekiper_inspect.plot.color import identity_color
    from bekiper_inspect.plot.feature._preference import plot_preference
    from bekiper_inspect.plot.feature._renderers_parts._plot_raycast_feature import _plot_nort_discrimination_feature

    feature = FeatureNortDiscrimination(
        label="test",
        object_labels=["zone_a", "arena"],
        exploration_times=[2.0, 3.0],
        coordinate_labels=[],
        description="test",
    )
    sample_manifest.label_colors["zone_a"] = "#abcdef"
    pie = plot_preference(feature, sample_manifest.label_colors)
    _fig, ax = plt.subplots()
    _plot_nort_discrimination_feature(ax, sample_manifest, feature, (480, 640, 3))
    assert pie.axes[0].patches[0].get_facecolor() == to_rgba(identity_color("zone_a", sample_manifest.label_colors))
    assert ax.patches[0].get_edgecolor() == pie.axes[0].patches[0].get_facecolor()


def test_native_composition_renders_without_video_and_keeps_missing_category(sample_manifest):
    from bekiper_inspect.plot.feature import plot_story
    from bekiper_inspect.plot.feature._grouping import StoryGroup

    feature = next(f for f in sample_manifest.feature_inspections if f.kind == "state_distribution")
    sample_manifest.video.resolution = (0, 0)
    sample_manifest.video.path = None
    fig = plot_feature_on_frame(sample_manifest, feature)
    assert len(fig.axes[0].patches) == 3
    assert "Denominator: 10.00 s" in fig.axes[0].get_title()
    assert any("Missing: 1.00 s (10.0%)" == t.get_text() for t in fig.axes[0].get_legend().texts)
    story = plot_story(sample_manifest, StoryGroup(key="session_composition", title="composition", features=[feature]))
    assert len(story.axes[0].patches) == 3


def test_tracking_traces_remain_aligned_and_gap_preserving_without_video(sample_manifest):
    from bekiper_inspect.plot.feature import plot_story
    from bekiper_inspect.plot.feature._grouping import StoryGroup

    sample_manifest.video.resolution = (0, 0)
    sample_manifest.df = pl.DataFrame(
        {"elapsed_s": [0.0, 0.2, 0.7], "raw_x": [1.0, None, 3.0], "clean_x": [1.0, 2.0, 3.0]}
    )
    features = [
        FeatureTimeSeries(label=column, value_column=column, unit="m", description=column, representative_frame_index=1)
        for column in ("raw_x", "clean_x")
    ]
    fig = plot_story(sample_manifest, StoryGroup(key="tracking_timeseries_nose", title="alignment", features=features))
    for ax in fig.axes:
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.0, 0.2, 0.7])
    assert np.isnan(fig.axes[0].lines[0].get_ydata()[1])
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [1.0, 2.0, 3.0])
    assert fig.axes[0].get_shared_x_axes().joined(*fig.axes)


def test_native_vector_example_annotates_recorded_angle_without_recomputation(sample_manifest):
    from matplotlib.quiver import Quiver

    from bekiper_inspect.plot.feature._vector_examples import draw_vector_example

    sample_manifest.video.resolution = (400, 200)
    sample_manifest.df = pl.DataFrame(
        {
            "nose_x": [0.1],
            "nose_y": [0.1],
            "dx": [1.0],
            "dy": [0.0],
            "tx": [0.2],
            "ty": [0.2],
            "native_angle": [123.456],
            "hit": [False],
        }
    )
    component = HeuristicDecompositionComponent(
        label="ray",
        result_column="hit",
        coordinate_labels=["nose"],
        ray_origin_label="nose",
        ray_direction_x_column="dx",
        ray_direction_y_column="dy",
        ray_target_x_column="tx",
        ray_target_y_column="ty",
        value_column="native_angle",
        value_unit="degrees",
    )
    feature = FeatureHeuristicDecomposition(
        label="geometry_vectors_A",
        heuristic_name="olfaction_A",
        result_column="hit",
        coordinate_labels=["nose"],
        components=[component],
        show_final_result=False,
        description="example",
    )
    _fig, ax = plt.subplots()
    draw_vector_example(ax, sample_manifest, feature, np.zeros((100, 100, 3)), 0)
    rays = [c for c in ax.collections if isinstance(c, Quiver)]
    assert len(rays) == 2
    np.testing.assert_allclose(
        [rays[1].U[0], rays[1].V[0]], np.array([0.1, 0.1]) / sample_manifest.settings.meters_per_pixel * [0.25, 0.5]
    )
    assert "Native angle: 123.46°" in ax.get_title() and "Frame 0" in ax.get_title()


def test_geometric_examples_do_not_expand_all_object_decomposition_layout(sample_manifest):
    from bekiper_inspect.plot.feature._grouping import group_into_stories

    example = FeatureHeuristicDecomposition(
        label="geometry_vectors_olfaction_object_1_ray_0",
        heuristic_name="olfaction_object_1",
        result_column="hit",
        coordinate_labels=["nose"],
        components=[],
        show_final_result=False,
        description="example",
        story_key="geometry_vectors_olfaction_object_1_ray",
    )
    from bekiper_inspect.manifest import PerimeterSpec

    perimeter = PerimeterSpec(
        label="object_1", shape="circle", params={"center_x": 0.1, "center_y": 0.1, "radius": 0.02}
    )
    stories, _ = group_into_stories([example, example], [], [perimeter])
    assert [story.key for story in stories] == ["geometry_vectors_olfaction_object_1_ray"]
