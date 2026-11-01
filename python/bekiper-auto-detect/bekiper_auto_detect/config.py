"""Detector configuration: model identifiers, device, thresholds, morphology.

All values flow through a single :class:`DetectorConfig` pydantic model with
a four-layer precedence chain: constructor arg > environment variable >
``[detect]`` section of ``bekiper_config.toml`` > built-in default.

The Rust ingress layer reads the same TOML file; Python parses its own
``[detect]`` section independently so the two languages share a physical
file without sharing a schema.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

Device = Literal["auto", "cuda", "mps", "cpu"]
DEFAULT_CONFIG_FILENAME = "bekiper_config.toml"
ENV_PREFIX = "BEKIPER"

# ── Built-in defaults ─────────────────────────────────────────────────────
# These constants are the single source of truth for Python-side detector
# defaults. Every field in :class:`DetectorConfig` references one of them
# so tests and downstream code can import the default without instantiating
# the model.

#: Ultralytics SAM2 checkpoint filename (auto-downloaded on first use).
DEFAULT_SAM2_MODEL: str = "sam2.1_l.pt"

#: HuggingFace model ID for the GroundingDINO base model used by the
#: text-prompted SAM2 pipeline.
DEFAULT_GROUNDING_DINO_MODEL: str = "IDEA-Research/grounding-dino-base"

#: Ultralytics YOLOE checkpoint for the lightweight open-vocabulary path.
DEFAULT_YOLOE_MODEL: str = "yoloe-11l-seg.pt"

#: Device-selection policy. ``"auto"`` picks cuda → mps → cpu based on
#: torch availability; the other values force a specific backend.
DEFAULT_DEVICE: Device = "auto"

#: Default detection confidence for YOLOE prediction.
DEFAULT_CONFIDENCE: float = 0.25

#: Default GroundingDINO box threshold (matches the upstream demo value).
DEFAULT_BOX_THRESHOLD: float = 0.25

#: Grayscale-histogram percentile used as the "floor" (background) estimate
#: in :func:`bekiper_auto_detect.recipe._detect_dark_objects`.
DEFAULT_FLOOR_PERCENTILE: float = 75.0

#: Multiplier applied to the floor value to derive the binary threshold.
DEFAULT_THRESHOLD_FACTOR: float = 0.85

#: Morphological close kernel size, as fraction of image diagonal.
DEFAULT_CLOSE_KERNEL_FRAC: float = 0.007

#: Dilation radius (fraction of image diagonal) around the mouse candidate.
DEFAULT_DILATE_FRAC: float = 0.03

#: Default minimum blob area (fraction of image area) for NORT object
#: candidates.
DEFAULT_MIN_AREA_FRAC: float = 0.005

#: Default maximum blob area (fraction of image area) for NORT object
#: candidates. Larger blobs are assumed to be mice or arena pixels.
DEFAULT_MAX_AREA_FRAC: float = 0.05

#: Minimum blob area (fraction of image area) for the mouse-removal pass.
DEFAULT_MOUSE_MIN_FRAC: float = 0.02


class DetectorConfig(BaseModel):
    """Python-side detector configuration.

    Layering order when constructed via :meth:`DetectorConfig.load`:

    1. Constructor kwargs (highest priority)
    2. ``BEKIPER_*`` environment variables
    3. ``[detect]`` section of ``bekiper_config.toml``
    4. Built-in defaults (lowest priority)
    """

    model_config = ConfigDict(extra="ignore")

    # Model identifiers ────────────────────────────────────────────────
    sam2_model: str = Field(
        default=DEFAULT_SAM2_MODEL,
        description="Ultralytics SAM2 checkpoint filename.",
    )
    grounding_dino_model: str = Field(
        default=DEFAULT_GROUNDING_DINO_MODEL,
        description="HuggingFace model ID for GroundingDINO.",
    )
    yoloe_model: str = Field(
        default=DEFAULT_YOLOE_MODEL,
        description="Ultralytics YOLOE checkpoint filename.",
    )

    # Compute backend ─────────────────────────────────────────────────
    device: Device = Field(
        default=DEFAULT_DEVICE,
        description='Device policy: "auto" (cuda→mps→cpu), "cuda", "mps", or "cpu".',
    )

    # Detection thresholds ────────────────────────────────────────────
    confidence: float = Field(
        default=DEFAULT_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="YOLOE / SAM2 detection confidence threshold.",
    )
    box_threshold: float = Field(
        default=DEFAULT_BOX_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="GroundingDINO box score threshold.",
    )

    # Classical-CV morphology knobs ───────────────────────────────────
    floor_percentile: float = Field(
        default=DEFAULT_FLOOR_PERCENTILE,
        ge=0.0,
        le=100.0,
        description="Grayscale percentile used as the background floor.",
    )
    threshold_factor: float = Field(
        default=DEFAULT_THRESHOLD_FACTOR,
        gt=0.0,
        description="Multiplier applied to the floor value for binary threshold.",
    )
    close_kernel_frac: float = Field(
        default=DEFAULT_CLOSE_KERNEL_FRAC,
        gt=0.0,
        description="Morphological close kernel size as fraction of image diagonal.",
    )
    dilate_frac: float = Field(
        default=DEFAULT_DILATE_FRAC,
        gt=0.0,
        description="Dilation radius around mouse blob as fraction of image diagonal.",
    )
    min_area_frac: float = Field(
        default=DEFAULT_MIN_AREA_FRAC,
        gt=0.0,
        description="Minimum candidate-object area as fraction of image area.",
    )
    max_area_frac: float = Field(
        default=DEFAULT_MAX_AREA_FRAC,
        gt=0.0,
        description="Maximum candidate-object area as fraction of image area.",
    )
    mouse_min_frac: float = Field(
        default=DEFAULT_MOUSE_MIN_FRAC,
        gt=0.0,
        description="Minimum blob area (fraction of image area) for mouse removal.",
    )

    @field_validator("device", mode="before")
    @classmethod
    def _lower_device(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v

    # ── Loaders ──────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        config_path: Path | str | None = None,
        **overrides: Any,
    ) -> DetectorConfig:
        """Build a :class:`DetectorConfig` with the full precedence chain.

        Parameters
        ----------
        config_path:
            Path to a ``bekiper_config.toml`` file. Its ``[detect]`` section,
            if present, is read as the lowest-priority file layer. ``None``
            skips the file layer entirely.
        **overrides:
            Highest-priority field overrides, typically sourced from CLI
            flags. Unknown keys are ignored (``model_config.extra="ignore"``).
        """
        values: dict[str, Any] = {}

        # Layer 1: TOML file
        if config_path is not None:
            file_values = _load_detect_toml(Path(config_path))
            values.update(file_values)

        # Layer 2: environment variables
        env_values = _load_env_overrides()
        values.update(env_values)

        # Layer 3: explicit overrides (CLI / constructor)
        values.update({k: v for k, v in overrides.items() if v is not None})

        return cls(**values)

    def resolve_device(self) -> str:
        """Resolve the ``"auto"`` policy into a concrete torch device string.

        Performs the import lazily so config construction itself stays
        dependency-free.
        """
        if self.device != "auto":
            return self.device

        try:
            import torch
        except ImportError:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ── Internal helpers ──────────────────────────────────────────────────────


_ENV_MAP: dict[str, str] = {
    f"{ENV_PREFIX}_SAM2_MODEL": "sam2_model",
    f"{ENV_PREFIX}_GROUNDING_DINO_MODEL": "grounding_dino_model",
    f"{ENV_PREFIX}_YOLOE_MODEL": "yoloe_model",
    f"{ENV_PREFIX}_DEVICE": "device",
    f"{ENV_PREFIX}_DETECT_CONFIDENCE": "confidence",
    f"{ENV_PREFIX}_DETECT_BOX_THRESHOLD": "box_threshold",
    f"{ENV_PREFIX}_DETECT_FLOOR_PERCENTILE": "floor_percentile",
    f"{ENV_PREFIX}_DETECT_THRESHOLD_FACTOR": "threshold_factor",
    f"{ENV_PREFIX}_DETECT_CLOSE_KERNEL_FRAC": "close_kernel_frac",
    f"{ENV_PREFIX}_DETECT_DILATE_FRAC": "dilate_frac",
    f"{ENV_PREFIX}_DETECT_MIN_AREA_FRAC": "min_area_frac",
    f"{ENV_PREFIX}_DETECT_MAX_AREA_FRAC": "max_area_frac",
    f"{ENV_PREFIX}_DETECT_MOUSE_MIN_FRAC": "mouse_min_frac",
}

_FLOAT_FIELDS = {
    "confidence",
    "box_threshold",
    "floor_percentile",
    "threshold_factor",
    "close_kernel_frac",
    "dilate_frac",
    "min_area_frac",
    "max_area_frac",
    "mouse_min_frac",
}


def _load_detect_toml(config_path: Path) -> dict[str, Any]:
    """Extract the ``[detect]`` section of a ``bekiper_config.toml`` file.

    Missing files or sections return an empty dict; malformed TOML is
    logged and skipped so Python never blocks on a file meant for Rust.
    """
    if not config_path.is_file():
        return {}
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        logger.warning("failed to parse %s: %s", config_path, e)
        return {}
    detect_section = data.get("detect")
    if not isinstance(detect_section, dict):
        return {}
    return detect_section


def _load_env_overrides() -> dict[str, Any]:
    """Read ``BEKIPER_*`` environment variables into a ``DetectorConfig`` dict."""
    values: dict[str, Any] = {}
    for env_name, field_name in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        if field_name in _FLOAT_FIELDS:
            try:
                values[field_name] = float(raw)
            except ValueError:
                logger.warning("ignoring %s=%r: not a valid float", env_name, raw)
                continue
        else:
            values[field_name] = raw
    return values
