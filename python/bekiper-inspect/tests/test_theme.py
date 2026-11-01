"""Tests for the PlotTheme precedence chain.

Verifies that ``PlotTheme.load`` layers values in the documented order:
constructor / CLI override > environment > ``[inspect.plot]`` TOML file >
built-in defaults. Captured settings can be replayed with an empty injected
environment so live overrides cannot alter a sealed presentation.
"""

from __future__ import annotations

import pytest

from bekiper_inspect.plot.theme import (
    DEFAULT_ANNOTATION_FONTSIZE,
    DEFAULT_BAR_PRIMARY_COLOR,
    DEFAULT_BAR_SECONDARY_COLOR,
    DEFAULT_BBOX_ALPHA,
    DEFAULT_FFMPEG_PRESET,
    DEFAULT_FIG_FORMAT,
    DEFAULT_FONTSIZE,
    DEFAULT_HINT_TEXT_COLOR,
    DEFAULT_LINEWIDTH,
    DEFAULT_MIN_DPI,
    DEFAULT_QUIVER_ALPHA,
    DEFAULT_TIMELINE_ALPHA,
    DEFAULT_TITLE_FONTSIZE,
    PlotTheme,
    SkipConfig,
    get_active_theme,
    set_active_theme,
)

# ── Built-in defaults ─────────────────────────────────────────────────────


class TestDefaults:
    def test_field_defaults_match_module_constants(self):
        t = PlotTheme()
        assert t.min_dpi == DEFAULT_MIN_DPI
        assert t.fig_format == DEFAULT_FIG_FORMAT
        assert t.ffmpeg_preset == DEFAULT_FFMPEG_PRESET
        assert t.linewidth == DEFAULT_LINEWIDTH
        assert t.fontsize == DEFAULT_FONTSIZE
        assert t.title_fontsize == DEFAULT_TITLE_FONTSIZE
        assert t.annotation_fontsize == DEFAULT_ANNOTATION_FONTSIZE
        assert t.bbox_alpha == DEFAULT_BBOX_ALPHA
        assert t.quiver_alpha == DEFAULT_QUIVER_ALPHA
        assert t.timeline_alpha == DEFAULT_TIMELINE_ALPHA
        assert t.bar_primary_color == DEFAULT_BAR_PRIMARY_COLOR
        assert t.bar_secondary_color == DEFAULT_BAR_SECONDARY_COLOR
        assert t.hint_text_color == DEFAULT_HINT_TEXT_COLOR

    def test_load_with_no_inputs_yields_defaults(self):
        t = PlotTheme.load()
        assert t.min_dpi == DEFAULT_MIN_DPI
        assert t.fig_format == DEFAULT_FIG_FORMAT

    def test_quiver_kwargs_projects_alpha(self):
        t = PlotTheme(quiver_alpha=0.42)
        assert t.quiver_kwargs() == {
            "alpha": 0.42,
            "angles": "xy",
        }

    def test_with_overrides_returns_copy(self):
        t = PlotTheme()
        t2 = t.with_overrides(min_dpi=150)
        assert t.min_dpi == DEFAULT_MIN_DPI
        assert t2.min_dpi == 150
        assert t is not t2


# ── Capturable environment layer ─────────────────────────────────────────


class TestCapturedEnvLayer:
    def test_precedence_and_replay(self, tmp_path):
        config = tmp_path / "bekiper_config.toml"
        config.write_text('[inspect.plot]\nmin_dpi=150\nfig_format=".pdf"\n')
        env = {"BEKIPER_PLOT_DPI": "300", "BEKIPER_PLOT_FORMAT": ".png", "BEKIPER_PLOT_TRAJECTORY_GRADIENT": "true"}
        theme = PlotTheme.load(config, env=env)
        assert theme.min_dpi == 300
        assert theme.fig_format == ".png"
        assert theme.trajectory_gradient is True
        assert PlotTheme.load(config, env=env, min_dpi=72).min_dpi == 72
        assert PlotTheme.load(config, env={}).min_dpi == 150

    @pytest.mark.parametrize("raw_dpi", ["0", "-1", "bad", "true"])
    def test_explicit_dpi_beats_invalid_environment(self, raw_dpi):
        theme = PlotTheme.load(env={"BEKIPER_PLOT_DPI": raw_dpi}, min_dpi=300)
        assert theme.min_dpi == 300

    def test_explicit_field_beats_unparseable_environment(self):
        theme = PlotTheme.load(env={"BEKIPER_PLOT_LINEWIDTH": "bad"}, linewidth=2.5)
        assert theme.linewidth == 2.5

    def test_none_override_keeps_environment_validation(self):
        with pytest.raises(ValueError):
            PlotTheme.load(env={"BEKIPER_PLOT_DPI": "bad"}, min_dpi=None)


# ── TOML file overlay ────────────────────────────────────────────────────


class TestTomlOverlay:
    def test_inspect_plot_section_applied(self, tmp_path):
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text(
            """
[project]
project_name = "test"

[inspect.plot]
min_dpi = 200
linewidth = 2.5
ffmpeg_preset = "veryfast"
bar_primary_color = "teal"
"""
        )
        t = PlotTheme.load(config_path=toml)
        assert t.min_dpi == 200
        assert t.linewidth == pytest.approx(2.5)
        assert t.ffmpeg_preset == "veryfast"
        assert t.bar_primary_color == "teal"

    def test_toml_without_inspect_plot_section_uses_defaults(self, tmp_path):
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text('[project]\nproject_name = "test"\n')
        t = PlotTheme.load(config_path=toml)
        assert t.min_dpi == DEFAULT_MIN_DPI

    def test_nonexistent_toml_file_is_silent(self, tmp_path):
        t = PlotTheme.load(config_path=tmp_path / "does_not_exist.toml")
        assert t.min_dpi == DEFAULT_MIN_DPI

    def test_malformed_toml_is_silent(self, tmp_path):
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text("this is [not valid toml")
        t = PlotTheme.load(config_path=toml)
        assert t.min_dpi == DEFAULT_MIN_DPI

    def test_extra_toml_keys_ignored(self, tmp_path):
        """Unknown keys in [inspect.plot] must not crash construction."""
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text(
            """
[inspect.plot]
min_dpi = 200
unknown_field = "oops"
"""
        )
        t = PlotTheme.load(config_path=toml)
        assert t.min_dpi == 200


# ── Full precedence chain ────────────────────────────────────────────────


class TestPrecedenceChain:
    def test_cli_beats_file(self, tmp_path):
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text("[inspect.plot]\nmin_dpi = 72\n")
        t = PlotTheme.load(config_path=toml, min_dpi=600)
        assert t.min_dpi == 600

    def test_file_beats_default(self, tmp_path):
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text("[inspect.plot]\nmin_dpi = 72\n")
        t = PlotTheme.load(config_path=toml)
        assert t.min_dpi == 72

    def test_none_override_does_not_clobber(self, tmp_path):
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text("[inspect.plot]\nmin_dpi = 72\n")
        # min_dpi=None must not overwrite the TOML value; explicit overrides
        # for other fields still win.
        t = PlotTheme.load(config_path=toml, min_dpi=None, fig_format=".pdf")
        assert t.min_dpi == 72
        assert t.fig_format == ".pdf"

    @pytest.mark.parametrize("dpi", [0, -1, 1.5, "300", True])
    def test_invalid_effective_dpi_is_rejected(self, dpi):
        with pytest.raises(ValueError, match="DPI"):
            PlotTheme.load(env={"BEKIPER_PLOT_DPI": "300"}, min_dpi=dpi)

    def test_explicit_dpi_beats_invalid_file_value(self, tmp_path):
        config = tmp_path / "bekiper_config.toml"
        config.write_text("[inspect.plot]\nmin_dpi = 0\n")
        assert PlotTheme.load(config, env={}, min_dpi=300).min_dpi == 300

    def test_profile_dpi_is_validated_when_applied(self):
        from bekiper_inspect.plot.theme import ProfileConfig

        theme = PlotTheme(profiles=ProfileConfig(presets={"invalid": {"min_dpi": -1}}))
        with pytest.raises(ValueError, match="DPI"):
            theme.with_profile("invalid")


# ── Active-theme registry ────────────────────────────────────────────────


class TestActiveTheme:
    def test_default_active_theme_is_builtin_defaults(self):
        assert get_active_theme().min_dpi == DEFAULT_MIN_DPI

    def test_set_active_theme_installs_new(self):
        original = get_active_theme()
        try:
            set_active_theme(PlotTheme(min_dpi=42))
            assert get_active_theme().min_dpi == 42
        finally:
            set_active_theme(original)

    def test_set_active_theme_isolated(self):
        """Ensure tests that install a theme don't leak across modules."""
        original = get_active_theme()
        try:
            set_active_theme(PlotTheme(fig_format=".png"))
            assert get_active_theme().fig_format == ".png"
        finally:
            set_active_theme(original)
        assert get_active_theme().fig_format == DEFAULT_FIG_FORMAT


# ── Per-feature skip configuration ──────────────────────────────────────


class TestSkipConfig:
    def test_default_skip_is_empty(self):
        t = PlotTheme()
        assert t.skip.kinds == []
        assert t.skip.labels == []
        assert not t.should_skip_feature("perimeter", "perimeter_arena")

    def test_skip_by_kind(self):
        t = PlotTheme(skip=SkipConfig(kinds=["scale_derivation", "time_series"]))
        assert t.should_skip_feature("scale_derivation", "scale_derivation")
        assert t.should_skip_feature("time_series", "timeseries_speed_mps")
        assert not t.should_skip_feature("perimeter", "perimeter_arena")

    def test_skip_by_label_glob(self):
        t = PlotTheme(skip=SkipConfig(labels=["perimeter_*", "proximity_arena"]))
        assert t.should_skip_feature("perimeter", "perimeter_arena")
        assert t.should_skip_feature("perimeter", "perimeter_zone_a")
        assert t.should_skip_feature("proximity", "proximity_arena")
        assert not t.should_skip_feature("heuristic", "heuristic_freezing")

    def test_skip_from_toml(self, tmp_path):
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text(
            """
[inspect.plot.skip]
kinds = ["scale_derivation"]
labels = ["perimeter_*"]
"""
        )
        t = PlotTheme.load(config_path=toml)
        assert t.should_skip_feature("scale_derivation", "scale_derivation")
        assert t.should_skip_feature("perimeter", "perimeter_arena")
        assert not t.should_skip_feature("heuristic", "heuristic_freezing")

    def test_skip_cloud_kind(self):
        t = PlotTheme(skip=SkipConfig(kinds=["heuristic_cloud"]))
        assert t.should_skip_feature("heuristic_cloud", "heuristic_freezing")
        assert not t.should_skip_feature("heuristic", "heuristic_in_center")


def test_night_mode_is_the_default_and_light_is_one_setting_away(tmp_path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from bekiper_inspect.plot.theme import get_active_theme, set_active_theme

    assert PlotTheme().color_mode == "dark"
    config = tmp_path / "bekiper_config.toml"
    config.write_text('[inspect.plot]\ncolor_mode = "light"\n')
    assert PlotTheme.load(config_path=config, env={}).color_mode == "light"
    assert PlotTheme.load(config_path=config, env={"BEKIPER_PLOT_COLOR_MODE": "dark"}).color_mode == "dark"
    assert PlotTheme.load(config_path=config, env={}, color_mode="dark").color_mode == "dark"
    with pytest.raises(ValueError, match="colour mode"):
        PlotTheme(color_mode="sepia")  # type: ignore[arg-type]
    previous = get_active_theme()
    try:
        for mode in ("dark", "light"):
            theme = PlotTheme(color_mode=mode)
            set_active_theme(theme)
            fig = plt.figure()
            assert to_hex(fig.get_facecolor()) == theme.colors()["background"]
            assert theme.hint_color == theme.colors()["muted"]
            plt.close(fig)
        assert PlotTheme(color_mode="dark", hint_text_color="teal").hint_color == "teal"
    finally:
        set_active_theme(previous)
