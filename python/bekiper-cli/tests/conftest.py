from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tomli_w

SAMPLE_MANIFEST_DICT = {
    "video": {
        "fps": 30.0,
        "total_frames": 100,
        "resolution": [640, 480],
        "meters_per_pixel": 0.001,
    },
    "labels": ["nose", "tail_base"],
    "label_colors": {"nose": "#1f77b4", "tail_base": "#ff7f0e"},
    "coordinate_columns": {
        "nose": {"x": "nose_x", "y": "nose_y"},
        "tail_base": {"x": "tail_x", "y": "tail_y"},
    },
    "perimeters": [
        {
            "shape": "circle",
            "label": "arena",
            "center_x": 0.15,
            "center_y": 0.12,
            "radius": 0.14,
        },
        {
            "shape": "rectangle",
            "label": "zone_a",
            "center_x": 0.1,
            "center_y": 0.1,
            "width": 0.05,
            "height": 0.04,
        },
        {
            "shape": "polygon",
            "label": "pen",
            "vertices": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]],
        },
    ],
    "heuristics": [
        {
            "name": "body_proximity",
            "result_column": "body_prox",
            "true_frames": 40,
            "total_frames": 100,
            "seconds": 1.33,
        },
        {
            "name": "olfaction",
            "result_column": "olfaction",
            "true_frames": 20,
            "total_frames": 100,
            "seconds": 0.67,
        },
    ],
    "settings": {
        "minimum_seconds_tolerance": 0.5,
        "maximum_seconds_distraction": 0.333,
        "meters_per_pixel": 0.001,
    },
}


def make_sample_df() -> pl.DataFrame:
    np.random.seed(42)
    n = 100
    body_prox = np.random.choice([True, False], n, p=[0.4, 0.6])
    near_arena = np.random.choice([True, False], n, p=[0.3, 0.7])
    return pl.DataFrame(
        {
            "nose_x": np.random.uniform(0.0, 0.3, n),
            "nose_y": np.random.uniform(0.0, 0.3, n),
            "tail_x": np.random.uniform(0.0, 0.3, n),
            "tail_y": np.random.uniform(0.0, 0.3, n),
            "body_prox": body_prox,
            "body_prox_final": body_prox & near_arena,
            "olfaction": np.random.choice([True, False], n, p=[0.2, 0.8]),
            "near_arena": near_arena,
        }
    )


@pytest.fixture
def inspection_dir(tmp_path: Path) -> Path:
    toml_path = tmp_path / "experiment_inspection.toml"
    parquet_path = tmp_path / "experiment_evaluated.parquet"
    toml_path.write_text(tomli_w.dumps(SAMPLE_MANIFEST_DICT))
    make_sample_df().write_parquet(parquet_path)
    return tmp_path
