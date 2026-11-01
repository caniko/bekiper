from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from bekiper_cli.inspection_export import execute_plan, render_plan
from bekiper_inspect.plot.theme import PlotTheme

N = 60


def manifest():
    columns = {
        "elapsed_s": np.arange(N) / 2,
        "snout_dx": np.full(N, 0.01),
        "snout_dy": np.full(N, 0.01),
        "center_ear_x": np.linspace(0.05, 0.3, N),
        "center_ear_y": np.linspace(0.05, 0.3, N),
        "in_center": np.arange(N) < 30,
        "in_periphery": np.arange(N) >= 30,
    }
    return SimpleNamespace(
        df=pl.DataFrame(columns),
        heuristics=[SimpleNamespace(name=n, result_column=n) for n in ("in_center", "in_periphery")],
        feature_inspections=[],
        perimeters=[],
        labels=[],
        coordinate_columns={},
        settings=SimpleNamespace(meters_per_pixel=0.002),
        video=SimpleNamespace(resolution=(100, 100), fps=2.0),
        frames_dir=None,
    )


def stems(jobs, selected=True):
    return [j.stem for j in jobs if j.selected or not selected]


def test_quiver_figures_are_normal_plan_entries():
    jobs = render_plan(manifest(), PlotTheme(), True, False, False, ".png", None, None)
    assert stems(jobs) == ["quiver_heading", "quiver_center_vs_periphery"]
    # features-off plans carry no quiver figures
    assert render_plan(manifest(), PlotTheme(), False, False, False, ".png", None, None) == []


def test_quiver_figures_respect_only_and_skip():
    only = render_plan(manifest(), PlotTheme(), True, False, False, ".png", "quiver_center*", None)
    assert stems(only) == ["quiver_center_vs_periphery"]
    skipped = render_plan(manifest(), PlotTheme(), True, False, False, ".png", None, "quiver_*")
    assert stems(skipped) == [] and all(j.reason == "output_selector" for j in skipped)


def test_quiver_figures_appear_in_the_render_receipt(tmp_path):
    theme = PlotTheme(min_dpi=48, fig_format=".png", quiver_stride=5)
    jobs = render_plan(manifest(), theme, True, False, False, ".png", None, None)
    paths = list(execute_plan(jobs, theme, tmp_path, tmp_path / "figures", {}))
    assert [p.name for p in paths] == ["quiver_heading.png", "quiver_center_vs_periphery.png"]
    receipt = json.loads((tmp_path / "figures" / "render_receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert [a["path"].rsplit("/", 1)[-1] for a in receipt["artifacts"]] == [p.name for p in paths]
    assert {i["stem"] for i in receipt["inventory"]} == {"quiver_heading", "quiver_center_vs_periphery"}
    assert receipt["effective_theme"]["quiver_stride"] == 5
    assert receipt["effective_theme"]["quiver_binary_cmaps"] == ["Reds", "Blues"]


def test_cli_flags_reach_the_theme(inspection_dir, monkeypatch):
    from bekiper_cli import inspect as cli

    seen = {}
    monkeypatch.setattr("bekiper_cli.inspection_export.render_plan", lambda *a: seen.update(theme=a[1]) or [])
    result = CliRunner().invoke(
        cli.app,
        [
            "plot",
            str(inspection_dir),
            "--dry-run",
            "--quiver-stride",
            "4",
            "--quiver-single-cmap",
            "PuRd",
            "--quiver-binary-cmaps",
            "Oranges,Greens",
        ],
    )
    assert result.exit_code == 0, result.output
    theme = seen["theme"]
    assert (theme.quiver_stride, theme.quiver_single_cmap, theme.quiver_binary_cmaps) == (
        4,
        "PuRd",
        ("Oranges", "Greens"),
    )
