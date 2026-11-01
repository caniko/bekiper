from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

# Snapshot of the inspection manifest TOML that `bekiper-core` serializes.
# Regenerate via:
#   UPDATE_GOLDEN=1 cargo test -p bekiper-ingress --test test_golden_manifest \
#       golden_inspection_toml_matches_snapshot
GOLDEN_TOML = Path(__file__).parent / "fixtures" / "golden_inspection.toml"


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
            "freezing": np.random.choice([True, False], n, p=[0.3, 0.7]),
            "heading_rad": np.linspace(0.0, 2 * np.pi, n),
            "speed_mps": np.random.uniform(0.0, 0.5, n),
            "near_arena": near_arena,
        }
    )


@pytest.fixture
def inspection_dir(tmp_path: Path) -> Path:
    """Create a temporary inspection output directory with sample data.

    The manifest is the byte-for-byte output of the Rust
    ``bekiper_core::inspection::InspectionManifest`` serializer, so
    any schema drift on the Rust side is caught by the
    ``test_golden_manifest`` snapshot test before it reaches here.
    """
    (tmp_path / "experiment_inspection.toml").write_bytes(GOLDEN_TOML.read_bytes())
    make_sample_df().write_parquet(tmp_path / "experiment_evaluated.parquet")
    return tmp_path


@pytest.fixture
def sample_manifest(inspection_dir: Path):
    from bekiper_inspect.manifest import InspectionManifest

    return InspectionManifest.from_directory(inspection_dir)


@pytest.fixture(autouse=True)
def close_test_figures():
    """Keep Matplotlib figure lifetime isolated to each test."""
    plt.close("all")
    yield
    plt.close("all")
