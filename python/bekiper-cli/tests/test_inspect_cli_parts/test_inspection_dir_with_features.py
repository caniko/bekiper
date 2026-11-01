from __future__ import annotations

# This module intentionally re-exports shared test fixtures and helpers.
# ruff: noqa: F401
import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
import polars as pl
import pytest
import tomli_w

matplotlib.use("Agg")

from conftest import SAMPLE_MANIFEST_DICT, make_sample_df  # type: ignore[import-not-found]
from typer.testing import CliRunner

import bekiper_cli.inspect as inspect_cli
from bekiper_cli.main import app

runner = CliRunner()


@pytest.fixture
def inspection_dir_with_features(tmp_path: Path) -> Path:
    """Inspection directory whose manifest carries `feature_inspections`
    and points at a small synthetic video. Exercises the new
    `bekiper inspect plot` per-feature overlay loop end-to-end.
    """
    # Build a tiny mp4 so the per-feature dispatcher can decode frames.
    # Frame count must match the manifest's `total_frames` (100) so the
    # heuristic-driven frame picker never lands past the end of the video.
    video_path = tmp_path / "fixture.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    try:
        for i in range(100):
            frame = np.full((240, 320, 3), (i * 2) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    manifest = copy.deepcopy(SAMPLE_MANIFEST_DICT)
    manifest["video_path"] = str(video_path)
    manifest["heuristics"].extend(
        [
            {
                "name": "olfaction_object_1",
                "result_column": "olfaction_object_1",
                "true_frames": 36,
                "total_frames": 100,
                "seconds": 1.2,
            },
            {
                "name": "olfaction_object_2",
                "result_column": "olfaction_object_2",
                "true_frames": 34,
                "total_frames": 100,
                "seconds": 1.13,
            },
        ]
    )
    manifest["perimeters"].extend(
        [
            {
                "shape": "circle",
                "label": "object_1",
                "center_x": 0.12,
                "center_y": 0.12,
                "radius": 0.025,
            },
            {
                "shape": "circle",
                "label": "object_2",
                "center_x": 0.18,
                "center_y": 0.12,
                "radius": 0.025,
            },
        ]
    )
    manifest["feature_inspections"] = [
        {
            "kind": "heuristic",
            "label": "heuristic_olfaction_object_1",
            "result_column": "olfaction_object_1",
            "coordinate_labels": ["nose"],
            "perimeter_label": "object_1",
            "description": "Olfaction object 1",
            "story_key": "objects",
        },
        {
            "kind": "heuristic",
            "label": "heuristic_olfaction_object_2",
            "result_column": "olfaction_object_2",
            "coordinate_labels": ["nose"],
            "perimeter_label": "object_2",
            "description": "Olfaction object 2",
            "story_key": "objects",
        },
        {
            "kind": "heuristic_decomposition",
            "label": "heuristic_decomposition_body_proximity_object_1",
            "heuristic_name": "body_proximity_object_1",
            "result_column": "body_proximity_object_1",
            "coordinate_labels": ["nose"],
            "perimeter_label": "object_1",
            "show_final_result": True,
            "components": [
                {
                    "label": "nose_near",
                    "result_column": "__bk_diag__body_proximity_object_1__nose_near",
                    "coordinate_labels": ["nose"],
                    "perimeter_label": "object_1",
                },
            ],
            "description": "Body proximity decomposition",
        },
        {
            "kind": "heuristic_decomposition",
            "label": "heuristic_decomposition_body_proximity_object_2",
            "heuristic_name": "body_proximity_object_2",
            "result_column": "body_proximity_object_2",
            "coordinate_labels": ["nose"],
            "perimeter_label": "object_2",
            "show_final_result": True,
            "components": [
                {
                    "label": "nose_near",
                    "result_column": "__bk_diag__body_proximity_object_2__nose_near",
                    "coordinate_labels": ["nose"],
                    "perimeter_label": "object_2",
                },
            ],
            "description": "Body proximity decomposition",
        },
        {
            "kind": "heuristic_decomposition",
            "label": "heuristic_decomposition_olfaction_object_1",
            "heuristic_name": "olfaction_object_1",
            "result_column": "olfaction_object_1",
            "coordinate_labels": ["nose"],
            "perimeter_label": "object_1",
            "show_final_result": True,
            "story_key": "object_decomposition_olfaction",
            "components": [
                {
                    "label": "nose_near",
                    "result_column": "__bk_diag__olfaction_object_1__nose_near",
                    "coordinate_labels": ["nose"],
                    "perimeter_label": "object_1",
                },
            ],
            "description": "Olfaction decomposition",
        },
        {
            "kind": "heuristic_decomposition",
            "label": "heuristic_decomposition_olfaction_object_2",
            "heuristic_name": "olfaction_object_2",
            "result_column": "olfaction_object_2",
            "coordinate_labels": ["nose"],
            "perimeter_label": "object_2",
            "show_final_result": True,
            "story_key": "object_decomposition_olfaction",
            "components": [
                {
                    "label": "nose_near",
                    "result_column": "__bk_diag__olfaction_object_2__nose_near",
                    "coordinate_labels": ["nose"],
                    "perimeter_label": "object_2",
                },
            ],
            "description": "Olfaction decomposition",
        },
        {
            "kind": "trajectory",
            "label": "trajectory_nose",
            "coordinate_labels": ["nose"],
            "description": "Nose trajectory",
        },
    ]

    toml_path = tmp_path / "experiment_inspection.toml"
    parquet_path = tmp_path / "experiment_evaluated.parquet"
    toml_path.write_text(tomli_w.dumps(manifest))
    df = make_sample_df()
    body_1 = df["body_prox"].to_numpy()
    body_2 = np.roll(body_1, 7)
    olf_1 = np.roll(body_1, 3)
    olf_2 = np.roll(body_1, 11)
    angles = np.linspace(25.0, 155.0, df.height, dtype=float)
    df = df.with_columns(
        [
            pl.Series("body_proximity_object_1", body_1),
            pl.Series("body_proximity_object_2", body_2),
            pl.Series("__bk_diag__body_proximity_object_1__nose_near", body_1),
            pl.Series("__bk_diag__body_proximity_object_2__nose_near", body_2),
            pl.Series("olfaction_object_1", olf_1),
            pl.Series("olfaction_object_2", olf_2),
            pl.Series("__bk_diag__olfaction_object_1__nose_near", olf_1),
            pl.Series("__bk_diag__olfaction_object_2__nose_near", olf_2),
            pl.Series("snout_dx", np.cos(np.deg2rad(angles))),
            pl.Series("snout_dy", np.sin(np.deg2rad(angles))),
        ]
    )
    df.write_parquet(parquet_path)
    return tmp_path


class TestInfoCommand:
    def test_info_output(self, inspection_dir: Path):
        result = runner.invoke(app, ["inspect", "info", str(inspection_dir)])
        assert result.exit_code == 0
        assert "FPS: 30.0" in result.output
        assert "Frames: 100" in result.output
        assert "body_proximity" in result.output
        assert "olfaction" in result.output
        assert "100 rows" in result.output

    def test_info_shows_perimeters(self, inspection_dir: Path):
        result = runner.invoke(app, ["inspect", "info", str(inspection_dir)])
        assert result.exit_code == 0
        assert "circle" in result.output
        assert "rectangle" in result.output
        assert "polygon" in result.output

    def test_info_shows_labels(self, inspection_dir: Path):
        result = runner.invoke(app, ["inspect", "info", str(inspection_dir)])
        assert result.exit_code == 0
        assert "nose" in result.output
        assert "tail_base" in result.output

    def test_info_shows_settings(self, inspection_dir: Path):
        result = runner.invoke(app, ["inspect", "info", str(inspection_dir)])
        assert result.exit_code == 0
        assert "Meters/pixel" in result.output
        assert "0.001" in result.output


class TestPlotCommand:
    def test_plot_generates_figures(self, inspection_dir: Path):
        output_dir = inspection_dir / "figures"
        result = runner.invoke(app, ["inspect", "plot", str(inspection_dir), "-o", str(output_dir), "-f", ".png"])
        assert result.exit_code == 0
        assert "Saved:" in result.output
        assert output_dir.exists()
        png_files = list(output_dir.glob("*.png"))
        assert len(png_files) >= 1

    def test_plot_svg_format(self, inspection_dir: Path):
        output_dir = inspection_dir / "figures_svg"
        result = runner.invoke(app, ["inspect", "plot", str(inspection_dir), "-o", str(output_dir), "-f", ".svgz"])
        assert result.exit_code == 0
        assert output_dir.exists()
        svgz_files = list(output_dir.glob("*.svgz"))
        assert len(svgz_files) >= 1

    def test_plot_custom_output_dir(self, inspection_dir: Path):
        custom_dir = inspection_dir / "my_custom_figures"
        result = runner.invoke(app, ["inspect", "plot", str(inspection_dir), "-o", str(custom_dir)])
        assert result.exit_code == 0
        assert custom_dir.exists()
        figure_files = list(custom_dir.iterdir())
        assert len(figure_files) >= 1
        assert f"All figures written to {custom_dir}" in result.output


__all__ = [name for name in globals() if not name.startswith(("__", "Test"))]
