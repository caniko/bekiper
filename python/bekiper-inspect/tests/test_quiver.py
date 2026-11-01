from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from matplotlib import colormaps
from matplotlib.quiver import Quiver

from bekiper_inspect.plot.quiver import CMAP_HIGH, CMAP_LOW, QuiverSpec, plot_quiver, quiver_specs
from bekiper_inspect.plot.theme import PlotTheme, set_active_theme

N = 120


def stub(**columns: np.ndarray):
    """A manifest stand-in: heading east-ish, one frame per 0.5 s, every column named by the caller."""
    rng = np.random.default_rng(0)
    data = {
        "elapsed_s": np.arange(N) * 0.5,
        "snout_dx": np.cos(np.linspace(0, 3, N)) * 0.01,
        "snout_dy": np.sin(np.linspace(0, 3, N)) * 0.01,
        "center_ear_x": rng.uniform(0.05, 0.4, N),
        "center_ear_y": rng.uniform(0.05, 0.4, N),
    } | columns
    heuristics = [SimpleNamespace(name=k, result_column=k) for k, v in data.items() if v.dtype == bool]
    return SimpleNamespace(
        df=pl.DataFrame(data),
        heuristics=heuristics,
        settings=SimpleNamespace(meters_per_pixel=0.002),
        video=SimpleNamespace(resolution=(224, 212), fps=2.0),
        frames_dir=None,
    )


def quivers(fig) -> list[Quiver]:
    found = [c for ax in fig.axes for c in ax.collections if isinstance(c, Quiver)]
    for q in found:
        q.update_scalarmappable()
    return found


def expected(cmap: str, t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return colormaps[cmap](CMAP_LOW + (CMAP_HIGH - CMAP_LOW) * (t - lo) / (hi - lo))[:, :3]


@pytest.fixture(autouse=True)
def theme():
    set_active_theme(PlotTheme(quiver_stride=1))
    yield
    set_active_theme(PlotTheme())


def test_binary_arrows_use_time_graded_reds_and_blues() -> None:
    novel = np.arange(N) < 40
    manifest = stub(exploring_novel=novel, exploring_familiar=(np.arange(N) >= 80))
    spec = next(s for s in quiver_specs(manifest) if s.stem == "quiver_novel_vs_familiar")
    reds, blues = quivers(plot_quiver(manifest, spec))
    t = np.arange(N) * 0.5
    # One time scale for the whole trial (0 .. last elapsed second), not the selected frames.
    lo, hi = 0.0, t.max()
    np.testing.assert_allclose(reds.get_facecolor()[:, :3], expected("Reds", t[novel], lo, hi), atol=0.01)
    np.testing.assert_allclose(blues.get_facecolor()[:, :3], expected("Blues", t[80:], lo, hi), atol=0.01)


def test_single_case_uses_rdpu_and_one_colourbar() -> None:
    manifest = stub()
    spec = next(s for s in quiver_specs(manifest) if s.stem == "quiver_heading")
    fig = plot_quiver(manifest, spec)
    (q,) = quivers(fig)
    t = np.arange(N) * 0.5
    np.testing.assert_allclose(q.get_facecolor()[:, :3], expected("RdPu", t, 0.0, t.max()), atol=0.01)
    labels = [ax.get_ylabel() for ax in fig.axes]
    assert sum("time (s)" in label for label in labels) == 1


def test_stride_subsamples_selected_frames() -> None:
    manifest = stub(in_center=np.arange(N) % 2 == 0, in_periphery=np.arange(N) % 2 == 1)
    spec = next(s for s in quiver_specs(manifest) if s.stem == "quiver_center_vs_periphery")
    set_active_theme(PlotTheme(quiver_stride=10))
    center, periphery = quivers(plot_quiver(manifest, spec))
    assert len(center.get_array()) == 6 and len(periphery.get_array()) == 6
    # stride 1 draws every selected frame
    set_active_theme(PlotTheme(quiver_stride=1))
    assert len(quivers(plot_quiver(manifest, spec))[0].get_array()) == 60


def test_nan_vectors_and_positions_are_dropped_and_arrows_are_unit_length() -> None:
    manifest = stub()
    dx = manifest.df["snout_dx"].to_numpy().copy()
    x = manifest.df["center_ear_x"].to_numpy().copy()
    dx[:10] = np.nan
    x[10:20] = np.nan
    manifest.df = manifest.df.with_columns(snout_dx=pl.Series(dx), center_ear_x=pl.Series(x))
    spec = quiver_specs(manifest)[0]
    (q,) = quivers(plot_quiver(manifest, spec))
    assert len(q.get_array()) == N - 20
    assert np.isfinite(q.U).all() and np.isfinite(q.X).all()
    np.testing.assert_allclose(np.hypot(q.U, q.V), 0.04 * 224)


def test_empty_selection_is_a_valid_figure_with_a_note() -> None:
    manifest = stub(in_center=np.zeros(N, bool), in_periphery=np.arange(N) > 60)
    spec = next(s for s in quiver_specs(manifest) if s.stem == "quiver_center_vs_periphery")
    fig = plot_quiver(manifest, spec)
    assert len(quivers(fig)) == 1
    assert any("no frames: center" in t.get_text() for ax in fig.axes for t in ax.texts)
    everything_empty = plot_quiver(manifest, QuiverSpec("q", "t", (("a", np.zeros(N, bool)),)))
    assert not quivers(everything_empty)
    assert any("no frames" in t.get_text() for ax in everything_empty.axes for t in ax.texts)


def test_specs_follow_the_columns_present() -> None:
    nort = stub(
        exploring_novel=np.arange(N) < 30,
        exploring_familiar=np.arange(N) > 90,
        whisker_fan_novel=np.arange(N) < 20,
        whisker_fan_familiar=np.arange(N) < 0,
        olfaction_novel=np.arange(N) < 40,
        olfaction_familiar=np.arange(N) < 0,
    )
    stems = {s.stem for s in quiver_specs(nort)}
    assert stems == {"quiver_heading", "quiver_novel_vs_familiar", "quiver_whisker_vs_nose"}
    whisker, nose_only = next(s for s in quiver_specs(nort) if s.stem == "quiver_whisker_vs_nose").classes
    assert whisker[1].sum() == 20 and nose_only[1].sum() == 20  # olfaction frames the fan also hit are not nose-only
    ymaze = stub(
        in_arm_A=np.arange(N) < 30, in_arm_B=(np.arange(N) >= 30) & (np.arange(N) < 60), in_center=np.arange(N) >= 60
    )
    assert {s.stem for s in quiver_specs(ymaze)} == {
        "quiver_heading",
        "quiver_arm_vs_center",
        "quiver_heading_arm_A",
        "quiver_heading_arm_B",
    }
    bare = stub()
    bare.df = bare.df.drop("snout_dx")
    assert quiver_specs(bare) == []


# ── theme layering: kwarg > env > [inspect.plot] TOML > default ──────────


def test_quiver_knob_defaults_and_layering(tmp_path) -> None:
    default = PlotTheme.load(env={})
    assert (default.quiver_binary_cmaps, default.quiver_single_cmap, default.quiver_stride) == (
        ("Reds", "Blues"),
        "RdPu",
        15,
    )
    config = tmp_path / "bekiper_config.toml"
    config.write_text(
        '[inspect.plot]\nquiver_stride = 7\nquiver_single_cmap = "PuRd"\nquiver_binary_cmaps = ["Oranges", "Greens"]\n'
    )
    from_file = PlotTheme.load(config, env={})
    assert (from_file.quiver_stride, from_file.quiver_single_cmap) == (7, "PuRd")
    assert from_file.quiver_binary_cmaps == ("Oranges", "Greens")
    env = {
        "BEKIPER_PLOT_QUIVER_STRIDE": "3",
        "BEKIPER_PLOT_QUIVER_SINGLE_CMAP": "BuPu",
        "BEKIPER_PLOT_QUIVER_BINARY_CMAPS": '["Purples", "Greys"]',
    }
    from_env = PlotTheme.load(config, env=env)
    assert (from_env.quiver_stride, from_env.quiver_single_cmap) == (3, "BuPu")
    assert from_env.quiver_binary_cmaps == ("Purples", "Greys")
    from_kwarg = PlotTheme.load(config, env=env, quiver_stride=2, quiver_single_cmap="RdPu")
    assert (from_kwarg.quiver_stride, from_kwarg.quiver_single_cmap) == (2, "RdPu")
    assert from_kwarg.quiver_binary_cmaps == ("Purples", "Greys")  # env still beats the file


def test_invalid_quiver_knobs_are_rejected() -> None:
    with pytest.raises(ValueError, match="stride"):
        PlotTheme(quiver_stride=0)
    with pytest.raises(ValueError, match="two"):
        PlotTheme(quiver_binary_cmaps=("Reds",))


def test_every_figure_of_a_trial_shares_one_time_scale_and_reports_arrow_counts() -> None:
    manifest = stub(in_arm_A=np.arange(N) < 30, in_arm_B=np.arange(N) >= 90)
    t_max = (N - 1) * 0.5
    for spec in quiver_specs(manifest):
        fig = plot_quiver(manifest, spec)
        assert {q.get_clim() for q in quivers(fig)} == {(0.0, t_max)}
        assert "arrows" in next(ax.get_title() for ax in fig.axes if ax.get_title())


def test_arrows_carry_a_halo_and_tall_cmaps_are_capped() -> None:
    manifest = stub()
    (q,) = quivers(plot_quiver(manifest, quiver_specs(manifest)[0]))
    assert q.get_linewidth()[0] > 0 and q.get_edgecolor()[0][3] < 1
    assert CMAP_HIGH < 1.0
