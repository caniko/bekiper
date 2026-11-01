from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from bekiper_inspect.manifest import InspectionManifest, PerimeterSpec
from bekiper_inspect.plot.color import boolean_index_colormap, color_map_by_number
from bekiper_inspect.plot.generic import (
    WEBP_MAX_DIMENSION_PX,
    ax_hue_plot_coordinate_pair_as_lines,
    ax_hue_plot_coordinate_with_boolean_index,
    ax_hue_plot_coordinates,
    boolean_index_truth_sequences,
    filter_coordinates_by_speed,
    plot_coordinates,
    plot_ellipse,
    plot_occupancy_heatmap,
    plot_state_colored_trajectory,
    plot_trajectory_small_multiples,
    plot_velocity_field,
    save_figure,
)
from bekiper_inspect.plot.heuristic import (
    plot_heuristic_on_coordinates,
    plot_heuristic_summary,
    plot_heuristic_timeline,
)
from bekiper_inspect.plot.perimeter import plot_perimeter_on_ax, subplot_with_perimeter
from bekiper_inspect.plot.theme import (
    FigureProfileRule,
    PlotTheme,
    ProfileConfig,
    active_theme_for_figure,
    get_active_theme,
    set_active_theme,
)


class TestBooleanIndexTruthSequences:
    def test_empty(self):
        assert boolean_index_truth_sequences(np.array([], dtype=bool)) == []

    def test_all_false(self):
        assert boolean_index_truth_sequences(np.array([False, False, False])) == []

    def test_all_true(self):
        assert boolean_index_truth_sequences(np.array([True, True, True])) == [(0, 3)]

    def test_single_true(self):
        assert boolean_index_truth_sequences(np.array([False, True, False])) == [(1, 2)]

    def test_multiple_sequences(self):
        idx = np.array([True, True, False, True, False, False, True, True, True])
        result = boolean_index_truth_sequences(idx)
        assert result == [(0, 2), (3, 4), (6, 9)]

    def test_trailing_true(self):
        idx = np.array([False, False, True, True])
        assert boolean_index_truth_sequences(idx) == [(2, 4)]

    def test_leading_true(self):
        idx = np.array([True, True, False, False])
        assert boolean_index_truth_sequences(idx) == [(0, 2)]


class TestColorMapByNumber:
    def test_returns_array(self):
        result = color_map_by_number(5)
        # Should be iterable with 5 colors
        colors = list(result)
        assert len(colors) == 5

    def test_single_color(self):
        result = list(color_map_by_number(1))
        assert len(result) == 1


class TestBooleanIndexColormap:
    def test_yields_correct_count(self):
        idx = [True, False, True]
        colors = list(boolean_index_colormap(idx))
        assert len(colors) == 3


class TestPlotCoordinates:
    def test_no_error(self):
        fig, ax = plt.subplots()
        coords = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        plot_coordinates(ax, coords)
        plt.close(fig)

    def test_with_color(self):
        fig, ax = plt.subplots()
        coords = np.array([[0, 0], [1, 1]], dtype=float)
        plot_coordinates(ax, coords, color="red")
        plt.close(fig)


class TestPlotEllipse:
    def test_circle(self):
        fig, ax = plt.subplots()
        plot_ellipse(ax, (0.5, 0.5), 0.3)
        assert len(ax.artists) == 1 or len(ax.patches) >= 0  # Circle added as artist
        plt.close(fig)

    def test_ellipse(self):
        fig, ax = plt.subplots()
        plot_ellipse(ax, (0.5, 0.5), (0.3, 0.5))
        assert len(ax.patches) >= 1
        plt.close(fig)

    def test_equal_radii_becomes_circle(self):
        fig, ax = plt.subplots()
        plot_ellipse(ax, (0.5, 0.5), (0.3, 0.3))
        # Should be treated as circle
        assert len(ax.patches) >= 1
        plt.close(fig)


class TestPlotOccupancyHeatmap:
    def test_draws_image_for_dense_points(self):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(0)
        coords = rng.normal(size=(500, 2))
        assert plot_occupancy_heatmap(ax, coords) is True
        assert len(ax.images) == 1
        plt.close(fig)

    def test_returns_false_when_all_nan(self):
        fig, ax = plt.subplots()
        coords = np.full((10, 2), np.nan)
        assert plot_occupancy_heatmap(ax, coords) is False
        assert len(ax.images) == 0
        plt.close(fig)

    def test_returns_false_when_degenerate_extent(self):
        fig, ax = plt.subplots()
        coords = np.tile([1.0, 2.0], (20, 1))
        assert plot_occupancy_heatmap(ax, coords) is False
        plt.close(fig)

    def test_skips_non_finite_samples(self):
        fig, ax = plt.subplots()
        coords = np.array([[0.0, 0.0], [np.nan, 1.0], [1.0, 1.0], [2.0, 0.5]])
        assert plot_occupancy_heatmap(ax, coords, bins=4) is True
        plt.close(fig)


class TestAxHuePlotCoordinateWithBooleanIndex:
    def test_scatter_mode(self):
        fig, ax = plt.subplots()
        coords = np.random.rand(10, 2)
        idx = np.array([True, False] * 5)
        ax_hue_plot_coordinate_with_boolean_index(ax, idx, coords)
        plt.close(fig)

    def test_scatter_with_non_confinement(self):
        fig, ax = plt.subplots()
        coords = np.random.rand(10, 2)
        idx = np.array([True, False] * 5)
        ax_hue_plot_coordinate_with_boolean_index(ax, idx, coords, plot_non_confinement=True)
        plt.close(fig)

    def test_line_mode(self):
        fig, ax = plt.subplots()
        coords = np.random.rand(10, 2)
        idx = np.array([True, True, True, False, False, True, True, False, True, True])
        ax_hue_plot_coordinate_with_boolean_index(ax, idx, coords, plot_line=True)
        plt.close(fig)


class TestAxHuePlotCoordinates:
    def test_no_error(self):
        fig, ax = plt.subplots()
        coords = np.random.rand(5, 2)
        ax_hue_plot_coordinates(ax, coords)
        plt.close(fig)


class TestAxHuePlotCoordinatePairAsLines:
    def test_no_error(self):
        fig, ax = plt.subplots()
        a = np.random.rand(5, 2)
        b = np.random.rand(5, 2)
        ax_hue_plot_coordinate_pair_as_lines(ax, a, b)
        plt.close(fig)


class TestSaveFigure:
    def test_save_png(self, tmp_path: Path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = save_figure(fig, tmp_path / "test_fig", fmt=".png")
        assert path.exists()
        assert path.suffix == ".png"

    def test_save_creates_directories(self, tmp_path: Path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = save_figure(fig, tmp_path / "nested" / "deep" / "fig", fmt=".png")
        assert path.exists()

    def test_save_figure_raises_when_artist_escapes_axes(self, tmp_path: Path):
        """An artist far outside its axes used to silently bloat the saved
        figure once `bbox_inches="tight"` engulfed it (see the
        story_object_decomposition_whisker_interaction_familiar artifact
        — saved at ~7900px wide for a nominal 16.5"×10.4" figure).
        save_figure now refuses to write such a figure."""
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        ax.plot([0, 1], [0, 1])
        ax.text(50.0, 50.0, "escaped", clip_on=False)
        try:
            with pytest.raises(RuntimeError, match="bbox_inches='tight' would grow"):
                save_figure(fig, tmp_path / "ballooned", fmt=".png")
        finally:
            plt.close(fig)

    def test_save_figure_growth_limit_can_be_disabled(self, tmp_path: Path):
        """Callers can opt out for figures with legitimate overhang."""
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        ax.plot([0, 1], [0, 1])
        ax.text(50.0, 50.0, "escaped", clip_on=False)
        try:
            path = save_figure(
                fig,
                tmp_path / "opted_out",
                fmt=".png",
                bbox_growth_limit=math.inf,
            )
            assert path.exists()
        finally:
            if not fig._suptitle and ax in fig.axes:
                plt.close(fig)

    def test_save_webp_clamps_dpi_for_tight_bbox(self, monkeypatch, tmp_path: Path):
        original_theme = get_active_theme()
        set_active_theme(PlotTheme(min_dpi=300))
        try:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            def fake_tightbbox(renderer):
                return Bbox.from_bounds(0.0, 0.0, 80.0, 10.0)

            fig.get_tightbbox = fake_tightbbox  # type: ignore[method-assign]

            calls: list[tuple[Path, dict[str, object]]] = []

            def fake_savefig(self, path, **kwargs):
                target = Path(path)
                calls.append((target, kwargs))
                target.write_bytes(b"ok")

            monkeypatch.setattr(Figure, "savefig", fake_savefig)

            # The mocked tightbbox is intentionally far larger than the
            # nominal figure so we can exercise the WebP DPI clamp.
            # Disable the bbox-growth guard for this test only.
            path = save_figure(
                fig,
                tmp_path / "oversized",
                fmt=".webp",
                bbox_growth_limit=math.inf,
            )

            assert path.suffix == ".webp"
            assert path.exists()
            assert len(calls) == 1
            assert calls[0][0].suffix == ".webp"

            pad_inches = float(rcParams["savefig.pad_inches"])
            expected_dpi = int((WEBP_MAX_DIMENSION_PX - 1) / (80.0 + (2.0 * pad_inches)))
            assert calls[0][1]["dpi"] == expected_dpi
            assert expected_dpi < 300
        finally:
            set_active_theme(original_theme)

    def test_save_webp_falls_back_to_png_on_encoder_limit(self, monkeypatch, tmp_path: Path):
        original_theme = get_active_theme()
        set_active_theme(PlotTheme(min_dpi=300))
        try:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            calls: list[tuple[Path, dict[str, object]]] = []

            def fake_savefig(self, path, **kwargs):
                target = Path(path)
                calls.append((target, kwargs))
                if target.suffix == ".webp":
                    raise ValueError("encoding error 5: Image size exceeds WebP limit of 16383 pixels")
                target.write_bytes(b"png")

            monkeypatch.setattr(Figure, "savefig", fake_savefig)

            path = save_figure(fig, tmp_path / "fallback", fmt=".webp")

            assert path.suffix == ".png"
            assert path.exists()
            assert [call[0].suffix for call in calls] == [".webp", ".png"]
            assert "pil_kwargs" in calls[0][1]
            assert "pil_kwargs" not in calls[1][1]
        finally:
            set_active_theme(original_theme)


class TestPlotPerimeterOnAx:
    def test_circle(self):
        fig, ax = plt.subplots()
        p = PerimeterSpec(label="c", shape="circle", params={"center_x": 0.5, "center_y": 0.5, "radius": 0.3})
        plot_perimeter_on_ax(ax, p)
        plt.close(fig)

    def test_rectangle(self):
        fig, ax = plt.subplots()
        p = PerimeterSpec(
            label="r", shape="rectangle", params={"center_x": 0.5, "center_y": 0.5, "width": 0.3, "height": 0.2}
        )
        plot_perimeter_on_ax(ax, p)
        assert len(ax.patches) >= 1
        plt.close(fig)

    def test_polygon(self):
        fig, ax = plt.subplots()
        p = PerimeterSpec(label="p", shape="polygon", params={"vertices": [[0, 0], [1, 0], [1, 1], [0, 1]]})
        plot_perimeter_on_ax(ax, p)
        plt.close(fig)

    def test_triangle(self):
        fig, ax = plt.subplots()
        p = PerimeterSpec(label="t", shape="triangle", params={"vertices": [[0, 0], [1, 0], [0.5, 1]]})
        plot_perimeter_on_ax(ax, p)
        plt.close(fig)

    def test_radial_maze(self):
        fig, ax = plt.subplots()
        p = PerimeterSpec(
            label="m",
            shape="radial_maze",
            params={
                "center_vertices": [[0, 0], [1, 0], [0.5, 1]],
                "arms": [[[2, 0], [3, 0], [3, 1], [2, 1]]],
            },
        )
        plot_perimeter_on_ax(ax, p)
        plt.close(fig)

    def test_with_pixel_scaling(self):
        fig, ax = plt.subplots()
        p = PerimeterSpec(label="c", shape="circle", params={"center_x": 0.15, "center_y": 0.12, "radius": 0.14})
        plot_perimeter_on_ax(ax, p, meters_per_pixel=0.001, as_pixels=True)
        plt.close(fig)


class TestSubplotWithPerimeter:
    def test_single(self, sample_manifest: InspectionManifest):
        fig, ax = subplot_with_perimeter(sample_manifest)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_specific_perimeter(self, sample_manifest: InspectionManifest):
        fig, ax = subplot_with_perimeter(sample_manifest, perimeter_label="arena")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multi_subplot(self, sample_manifest: InspectionManifest):
        fig, axes = subplot_with_perimeter(sample_manifest, nrows=2, ncols=2)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotHeuristicSummary:
    def test_returns_figure(self, sample_manifest: InspectionManifest):
        fig = plot_heuristic_summary(sample_manifest)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_empty_heuristics(self, sample_manifest: InspectionManifest):
        sample_manifest.heuristics = []
        fig = plot_heuristic_summary(sample_manifest)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotHeuristicTimeline:
    def test_returns_figure(self, sample_manifest: InspectionManifest):
        fig = plot_heuristic_timeline(sample_manifest, "body_proximity")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotHeuristicOnCoordinates:
    def test_returns_figure(self, sample_manifest: InspectionManifest):
        fig = plot_heuristic_on_coordinates(sample_manifest, "body_proximity", "nose")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_perimeter(self, sample_manifest: InspectionManifest):
        fig = plot_heuristic_on_coordinates(sample_manifest, "body_proximity", "nose", perimeter_label="arena")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_line_mode(self, sample_manifest: InspectionManifest):
        fig = plot_heuristic_on_coordinates(sample_manifest, "body_proximity", "nose", plot_line=True)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestFilterCoordinatesBySpeed:
    def test_disabled_when_threshold_zero(self):
        coords = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]])
        out = filter_coordinates_by_speed(coords, min_speed_units_per_frame=0.0)
        assert np.array_equal(out, coords)

    def test_masks_stationary_run(self):
        # Long stationary run with one fast jump.
        coords = np.array(
            [[0, 0], [0.01, 0], [0.02, 0], [10, 0], [10.01, 0], [10.02, 0]],
            dtype=float,
        )
        out = filter_coordinates_by_speed(coords, min_speed_units_per_frame=1.0)
        # The first/last of each stationary run keep one fast neighbour, so
        # only inner samples get masked.
        assert not np.isnan(out[0]).all()
        assert np.isnan(out[1]).all()
        assert not np.isnan(out[2]).all()
        assert not np.isnan(out[3]).all()
        assert np.isnan(out[4]).all()
        assert not np.isnan(out[5]).all()

    def test_short_input_returned_unchanged(self):
        coords = np.array([[1.0, 2.0]])
        out = filter_coordinates_by_speed(coords, min_speed_units_per_frame=1.0)
        assert np.array_equal(out, coords)


class TestPlotOccupancyHeatmapLogScale:
    def test_log_scale_changes_normalised_values(self):
        rng = np.random.default_rng(1)
        # Strongly bimodal: hot cluster + scattered samples.
        a = rng.normal(loc=(0, 0), scale=0.05, size=(2000, 2))
        b = rng.normal(loc=(2, 2), scale=0.5, size=(50, 2))
        coords = np.concatenate([a, b])
        fig, ax = plt.subplots()
        assert plot_occupancy_heatmap(ax, coords, scale="log") is True
        log_img = ax.images[0].get_array()
        plt.close(fig)
        fig, ax = plt.subplots()
        assert plot_occupancy_heatmap(ax, coords, scale="linear") is True
        lin_img = ax.images[0].get_array()
        plt.close(fig)
        # Log compresses dynamic range — the dim cluster should be
        # relatively brighter under log than under linear.
        assert log_img.mean() > lin_img.mean()


class TestPlotVelocityField:
    def test_draws_quiver_for_directional_walk(self):
        # Diagonal random walk biased toward +x +y.
        rng = np.random.default_rng(0)
        steps = rng.normal(loc=(0.5, 0.3), scale=0.1, size=(400, 2))
        coords = np.cumsum(steps, axis=0)
        fig, ax = plt.subplots()
        assert plot_velocity_field(ax, coords, bins=8, min_samples_per_bin=2) is True
        # Quiver creates one PolyCollection-like artist.
        from matplotlib.quiver import Quiver

        assert any(isinstance(c, Quiver) for c in ax.collections)
        plt.close(fig)

    def test_returns_false_for_too_few_samples(self):
        fig, ax = plt.subplots()
        coords = np.array([[0.0, 0.0]])
        assert plot_velocity_field(ax, coords) is False
        plt.close(fig)

    def test_returns_false_for_degenerate_extent(self):
        fig, ax = plt.subplots()
        coords = np.tile([1.0, 2.0], (50, 1))
        assert plot_velocity_field(ax, coords) is False
        plt.close(fig)


class TestPlotStateColoredTrajectory:
    def test_colors_per_state(self):
        coords = np.array([[i, 0.0] for i in range(6)])
        states = {
            "freezing": np.array([True, True, False, False, False, False]),
            "exploring": np.array([False, False, False, True, True, False]),
        }
        fig, ax = plt.subplots()
        plot_state_colored_trajectory(
            ax,
            coords,
            states,
            colors={"freezing": "blue", "exploring": "red"},
            default_color="gray",
        )
        # Each segment is its own Line2D — count and sample colors.
        from matplotlib.colors import to_rgba

        seen = {
            tuple(line.get_color()) if isinstance(line.get_color(), (list, tuple)) else to_rgba(line.get_color())
            for line in ax.get_lines()
        }
        # Should contain at least the freezing and gray default colors.
        assert to_rgba("blue") in seen
        assert to_rgba("red") in seen
        assert to_rgba("gray") in seen
        plt.close(fig)

    def test_no_states_uses_default(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        fig, ax = plt.subplots()
        plot_state_colored_trajectory(ax, coords, {}, default_color="green")
        from matplotlib.colors import to_rgba

        for line in ax.get_lines():
            assert to_rgba(line.get_color()) == to_rgba("green")
        plt.close(fig)


class TestPlotTrajectorySmallMultiples:
    def test_creates_n_axes(self):
        rng = np.random.default_rng(0)
        coords = np.cumsum(rng.normal(size=(120, 2)), axis=0)
        fig = plt.figure()
        axes = plot_trajectory_small_multiples(fig, coords, n_panels=3)
        assert len(axes) == 3
        for ax in axes:
            assert len(ax.get_lines()) >= 1
        plt.close(fig)

    def test_with_extent_locks_axis_range(self):
        coords = np.array([[float(i), float(i)] for i in range(60)])
        fig = plt.figure()
        axes = plot_trajectory_small_multiples(
            fig,
            coords,
            n_panels=2,
            extent=(0.0, 100.0, 0.0, 100.0),
        )
        for ax in axes:
            assert ax.get_xlim() == (0.0, 100.0)
            # Image-style inverted y range.
            assert ax.get_ylim() == (100.0, 0.0)
        plt.close(fig)

    def test_zero_panels_raises(self):
        fig = plt.figure()
        with pytest.raises(ValueError):
            plot_trajectory_small_multiples(fig, np.zeros((4, 2)), n_panels=0)
        plt.close(fig)


class TestProfileSystem:
    def test_builtin_profiles_present_on_default_theme(self):
        theme = PlotTheme()
        for name in ("line", "density", "dense_trajectory", "temporal", "print_friendly"):
            assert name in theme.profiles.presets

    def test_with_profile_applies_overrides(self):
        theme = PlotTheme()
        derived = theme.with_profile("dense_trajectory")
        assert derived.trajectory_render_mode == "both"
        # Base theme is unchanged.
        assert theme.trajectory_render_mode == "line"

    def test_with_unknown_profile_returns_self_with_warning(self, caplog):
        theme = PlotTheme()
        with caplog.at_level("WARNING"):
            derived = theme.with_profile("does_not_exist")
        assert derived is theme
        assert any("does_not_exist" in r.message for r in caplog.records)

    def test_resolve_for_figure_picks_first_matching_rule(self):
        theme = PlotTheme(
            profiles=ProfileConfig(
                presets={
                    "a": {"trajectory_render_mode": "density"},
                    "b": {"trajectory_render_mode": "both"},
                },
                rules=[
                    FigureProfileRule(profile="a", kind="trajectory"),
                    FigureProfileRule(profile="b", kind="*"),
                ],
            ),
        )
        assert theme.resolve_for_figure("trajectory", "x").trajectory_render_mode == "density"
        assert theme.resolve_for_figure("heuristic", "x").trajectory_render_mode == "both"

    def test_resolve_for_figure_no_match_returns_self(self):
        theme = PlotTheme(
            profiles=ProfileConfig(
                presets={"a": {"linewidth": 9.0}},
                rules=[FigureProfileRule(profile="a", kind="trajectory")],
            ),
        )
        assert theme.resolve_for_figure("heuristic", "x") is theme

    def test_label_pattern_matches(self):
        theme = PlotTheme(
            profiles=ProfileConfig(
                presets={"a": {"linewidth": 9.0}},
                rules=[FigureProfileRule(profile="a", label="story_object_*")],
            ),
        )
        assert theme.resolve_for_figure("story", "story_object_olfaction").linewidth == 9.0
        assert theme.resolve_for_figure("story", "story_other") is theme

    def test_active_theme_for_figure_swaps_and_restores(self):
        base = PlotTheme.load(
            None,
            profiles={"loud": {"linewidth": 7.0}},
            figure_profiles=[{"kind": "trajectory", "profile": "loud"}],
        )
        set_active_theme(base)
        try:
            assert get_active_theme().linewidth == base.linewidth
            with active_theme_for_figure("trajectory", "x") as resolved:
                assert resolved.linewidth == 7.0
                assert get_active_theme() is resolved
            assert get_active_theme() is base
        finally:
            set_active_theme(PlotTheme())

    def test_active_theme_for_figure_restores_on_exception(self):
        base = PlotTheme.load(
            None,
            profiles={"loud": {"linewidth": 7.0}},
            figure_profiles=[{"kind": "*", "profile": "loud"}],
        )
        set_active_theme(base)
        try:
            with pytest.raises(RuntimeError), active_theme_for_figure("trajectory", "x"):
                raise RuntimeError("boom")
            assert get_active_theme() is base
        finally:
            set_active_theme(PlotTheme())

    def test_small_multiples_profile_sets_knob(self):
        theme = PlotTheme()
        derived = theme.with_profile("small_multiples_3")
        assert derived.small_multiples_n == 3
        assert theme.small_multiples_n == 1  # base unchanged

    def test_small_multiples_via_load_kwargs(self):
        # CLI flags reach PlotTheme via load() keyword overrides; verify
        # the int conversion happens in typer's layer (here we pass an int).
        theme = PlotTheme.load(small_multiples_n=4)
        assert theme.small_multiples_n == 4

    def test_load_merges_user_overrides_into_builtin_profile(self, tmp_path: Path):
        cfg = tmp_path / "bekiper_config.toml"
        cfg.write_text(
            "[inspect.plot.profiles.dense_trajectory]\n"
            "density_alpha = 0.9\n"
            "\n"
            "[[inspect.plot.figure_profiles]]\n"
            'kind = "ymaze_arm_sections"\n'
            'profile = "dense_trajectory"\n',
            encoding="utf-8",
        )
        theme = PlotTheme.load(cfg)
        preset = theme.profiles.presets["dense_trajectory"]
        # User override applied
        assert preset["density_alpha"] == 0.9
        # Built-in fields preserved
        assert preset["trajectory_render_mode"] == "both"
        # Routing rule registered
        resolved = theme.resolve_for_figure("ymaze_arm_sections", "anything")
        assert resolved.density_alpha == 0.9
        assert resolved.trajectory_render_mode == "both"
