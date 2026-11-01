"""Tests for the DetectorConfig precedence chain.

Verifies that ``DetectorConfig.load`` layers values in the documented
order: CLI / constructor override > env var > TOML file > built-in
default. Uses monkeypatch for env-var isolation so tests can run in
parallel without polluting each other.
"""

from __future__ import annotations

import pytest

from bekiper_auto_detect.config import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_CONFIDENCE,
    DEFAULT_DEVICE,
    DEFAULT_DILATE_FRAC,
    DEFAULT_FLOOR_PERCENTILE,
    DEFAULT_GROUNDING_DINO_MODEL,
    DEFAULT_MAX_AREA_FRAC,
    DEFAULT_MIN_AREA_FRAC,
    DEFAULT_MOUSE_MIN_FRAC,
    DEFAULT_SAM2_MODEL,
    DEFAULT_THRESHOLD_FACTOR,
    DEFAULT_YOLOE_MODEL,
    DetectorConfig,
)

# ── Built-in defaults ─────────────────────────────────────────────────────


class TestDefaults:
    def test_all_field_defaults_match_module_constants(self):
        cfg = DetectorConfig()
        assert cfg.sam2_model == DEFAULT_SAM2_MODEL
        assert cfg.grounding_dino_model == DEFAULT_GROUNDING_DINO_MODEL
        assert cfg.yoloe_model == DEFAULT_YOLOE_MODEL
        assert cfg.device == DEFAULT_DEVICE
        assert cfg.confidence == DEFAULT_CONFIDENCE
        assert cfg.box_threshold == DEFAULT_BOX_THRESHOLD
        assert cfg.floor_percentile == DEFAULT_FLOOR_PERCENTILE
        assert cfg.threshold_factor == DEFAULT_THRESHOLD_FACTOR
        assert cfg.dilate_frac == DEFAULT_DILATE_FRAC
        assert cfg.min_area_frac == DEFAULT_MIN_AREA_FRAC
        assert cfg.max_area_frac == DEFAULT_MAX_AREA_FRAC
        assert cfg.mouse_min_frac == DEFAULT_MOUSE_MIN_FRAC

    def test_load_with_no_inputs_yields_defaults(self, monkeypatch):
        # Scrub any BEKIPER_* env vars that happen to be set in the
        # surrounding shell.
        for key in list(os_environ_keys(monkeypatch)):
            if key.startswith("BEKIPER_"):
                monkeypatch.delenv(key, raising=False)
        cfg = DetectorConfig.load()
        assert cfg.sam2_model == DEFAULT_SAM2_MODEL
        assert cfg.device == DEFAULT_DEVICE

    def test_confidence_range_validated(self):
        with pytest.raises(ValueError):
            DetectorConfig(confidence=1.5)
        with pytest.raises(ValueError):
            DetectorConfig(confidence=-0.1)

    def test_device_literal_validated(self):
        with pytest.raises(ValueError):
            DetectorConfig(device="tpu")  # type: ignore[arg-type]


# ── Env-var overlay ───────────────────────────────────────────────────────


class TestEnvOverlay:
    def test_env_string_field(self, monkeypatch):
        _scrub_bekiper_env(monkeypatch)
        monkeypatch.setenv("BEKIPER_SAM2_MODEL", "sam2.1_b.pt")
        cfg = DetectorConfig.load()
        assert cfg.sam2_model == "sam2.1_b.pt"
        # Other fields untouched
        assert cfg.yoloe_model == DEFAULT_YOLOE_MODEL

    def test_env_float_field(self, monkeypatch):
        _scrub_bekiper_env(monkeypatch)
        monkeypatch.setenv("BEKIPER_DETECT_CONFIDENCE", "0.42")
        cfg = DetectorConfig.load()
        assert cfg.confidence == pytest.approx(0.42)

    def test_env_device_lowercased(self, monkeypatch):
        _scrub_bekiper_env(monkeypatch)
        monkeypatch.setenv("BEKIPER_DEVICE", "CPU")
        cfg = DetectorConfig.load()
        assert cfg.device == "cpu"

    def test_env_invalid_float_is_ignored(self, monkeypatch):
        _scrub_bekiper_env(monkeypatch)
        monkeypatch.setenv("BEKIPER_DETECT_CONFIDENCE", "not-a-number")
        # Invalid env var is silently dropped; default applies.
        cfg = DetectorConfig.load()
        assert cfg.confidence == DEFAULT_CONFIDENCE

    def test_env_empty_string_is_ignored(self, monkeypatch):
        _scrub_bekiper_env(monkeypatch)
        monkeypatch.setenv("BEKIPER_SAM2_MODEL", "")
        cfg = DetectorConfig.load()
        assert cfg.sam2_model == DEFAULT_SAM2_MODEL

    def test_multiple_env_vars(self, monkeypatch):
        _scrub_bekiper_env(monkeypatch)
        monkeypatch.setenv("BEKIPER_SAM2_MODEL", "custom.pt")
        monkeypatch.setenv("BEKIPER_DETECT_CONFIDENCE", "0.9")
        monkeypatch.setenv("BEKIPER_DETECT_MIN_AREA_FRAC", "0.001")
        cfg = DetectorConfig.load()
        assert cfg.sam2_model == "custom.pt"
        assert cfg.confidence == pytest.approx(0.9)
        assert cfg.min_area_frac == pytest.approx(0.001)


# ── TOML file overlay ─────────────────────────────────────────────────────


class TestTomlOverlay:
    def test_toml_detect_section_applied(self, monkeypatch, tmp_path):
        _scrub_bekiper_env(monkeypatch)
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text(
            """
[project]
project_name = "test"
data_directory = "."

[detect]
sam2_model = "from_file.pt"
confidence = 0.8
min_area_frac = 0.01
"""
        )
        cfg = DetectorConfig.load(config_path=toml)
        assert cfg.sam2_model == "from_file.pt"
        assert cfg.confidence == pytest.approx(0.8)
        assert cfg.min_area_frac == pytest.approx(0.01)

    def test_toml_without_detect_section_uses_defaults(self, monkeypatch, tmp_path):
        _scrub_bekiper_env(monkeypatch)
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text('[project]\nproject_name = "test"\n')
        cfg = DetectorConfig.load(config_path=toml)
        assert cfg.sam2_model == DEFAULT_SAM2_MODEL

    def test_nonexistent_toml_file_is_silent(self, monkeypatch, tmp_path):
        _scrub_bekiper_env(monkeypatch)
        cfg = DetectorConfig.load(config_path=tmp_path / "does_not_exist.toml")
        assert cfg.sam2_model == DEFAULT_SAM2_MODEL

    def test_malformed_toml_is_silent(self, monkeypatch, tmp_path, caplog):
        _scrub_bekiper_env(monkeypatch)
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text("this is [not valid toml")
        # Should not raise; should log a warning.
        cfg = DetectorConfig.load(config_path=toml)
        assert cfg.sam2_model == DEFAULT_SAM2_MODEL


# ── Full precedence chain ────────────────────────────────────────────────


class TestPrecedenceChain:
    def test_cli_beats_env(self, monkeypatch):
        _scrub_bekiper_env(monkeypatch)
        monkeypatch.setenv("BEKIPER_SAM2_MODEL", "from_env.pt")
        cfg = DetectorConfig.load(sam2_model="from_cli.pt")
        assert cfg.sam2_model == "from_cli.pt"

    def test_env_beats_file(self, monkeypatch, tmp_path):
        _scrub_bekiper_env(monkeypatch)
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text('[project]\nproject_name = "x"\n\n[detect]\nsam2_model = "from_file.pt"\n')
        monkeypatch.setenv("BEKIPER_SAM2_MODEL", "from_env.pt")
        cfg = DetectorConfig.load(config_path=toml)
        assert cfg.sam2_model == "from_env.pt"

    def test_file_beats_default(self, monkeypatch, tmp_path):
        _scrub_bekiper_env(monkeypatch)
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text('[project]\nproject_name = "x"\n\n[detect]\nsam2_model = "from_file.pt"\n')
        cfg = DetectorConfig.load(config_path=toml)
        assert cfg.sam2_model == "from_file.pt"

    def test_none_override_does_not_clobber_lower_layer(self, monkeypatch, tmp_path):
        """Passing ``None`` as an override must not wipe out the env/file value."""
        _scrub_bekiper_env(monkeypatch)
        monkeypatch.setenv("BEKIPER_SAM2_MODEL", "from_env.pt")
        cfg = DetectorConfig.load(sam2_model=None, confidence=0.1)
        # None override dropped → env wins for sam2_model.
        assert cfg.sam2_model == "from_env.pt"
        # Explicit override still applied.
        assert cfg.confidence == pytest.approx(0.1)

    def test_full_chain_all_layers_combined(self, monkeypatch, tmp_path):
        _scrub_bekiper_env(monkeypatch)
        toml = tmp_path / "bekiper_config.toml"
        toml.write_text(
            """
[detect]
sam2_model = "file.pt"
confidence = 0.2
min_area_frac = 0.001
yoloe_model = "file_yoloe.pt"
"""
        )
        monkeypatch.setenv("BEKIPER_DETECT_CONFIDENCE", "0.5")
        monkeypatch.setenv("BEKIPER_YOLOE_MODEL", "env_yoloe.pt")

        cfg = DetectorConfig.load(config_path=toml, sam2_model="cli.pt")

        # CLI wins for sam2_model
        assert cfg.sam2_model == "cli.pt"
        # Env wins for confidence and yoloe_model
        assert cfg.confidence == pytest.approx(0.5)
        assert cfg.yoloe_model == "env_yoloe.pt"
        # File wins for min_area_frac (no env, no CLI)
        assert cfg.min_area_frac == pytest.approx(0.001)
        # Default wins for grounding_dino_model (nothing set)
        assert cfg.grounding_dino_model == DEFAULT_GROUNDING_DINO_MODEL


# ── resolve_device() ──────────────────────────────────────────────────────


class TestResolveDevice:
    def test_explicit_device_returned_verbatim(self):
        assert DetectorConfig(device="cpu").resolve_device() == "cpu"

    def test_auto_picks_cuda_when_available(self, monkeypatch):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA/ROCm GPU available")
        assert DetectorConfig(device="auto").resolve_device() == "cuda"


# ── helpers ───────────────────────────────────────────────────────────────


def _scrub_bekiper_env(monkeypatch):
    """Delete every BEKIPER_* variable in the current env for this test."""
    import os

    for key in list(os.environ):
        if key.startswith("BEKIPER_"):
            monkeypatch.delenv(key, raising=False)


def os_environ_keys(monkeypatch):
    import os

    return list(os.environ)
