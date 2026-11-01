"""Re-exports of the Rust-backed inspection types.

The classes themselves live in the ``bekiper_rs`` extension module
(built from ``rust/bekiper-inspect-py``); the Rust crate ``bekiper-core``
is the single source of truth for their schemas.

Only the ``DEFAULT_*`` fallback constants used by the loader for
older inspection outputs remain Python-side, since they are not
schema-bearing.
"""

from __future__ import annotations

from bekiper_rs import (
    CoordinateColumnSpec,
    HeuristicMeta,
    IngressSessionContext,
    InspectionSettings,
    PerimeterSpec,
    QuadrantSpec,
    ScaleDerivation,
    VideoMeta,
)

# ── InspectionSettings fallbacks ──────────────────────────────────────────
# Used when the inspection TOML omits the corresponding field. Kept on the
# Python side because they exist purely to keep old manifests loadable.

#: Default minimum-seconds-tolerance filter for heuristic bouts.
DEFAULT_MINIMUM_SECONDS_TOLERANCE: float = 0.5

#: Default maximum-seconds-distraction gap bridging value (1/3 second).
DEFAULT_MAXIMUM_SECONDS_DISTRACTION: float = 1.0 / 3.0

#: Default meters-per-pixel fallback (1 mm/px) when the manifest omits
#: scale — produces output in sensible units instead of raw pixels.
DEFAULT_METERS_PER_PIXEL: float = 0.001

__all__ = [
    "DEFAULT_MAXIMUM_SECONDS_DISTRACTION",
    "DEFAULT_METERS_PER_PIXEL",
    "DEFAULT_MINIMUM_SECONDS_TOLERANCE",
    "CoordinateColumnSpec",
    "HeuristicMeta",
    "IngressSessionContext",
    "InspectionSettings",
    "PerimeterSpec",
    "QuadrantSpec",
    "ScaleDerivation",
    "VideoMeta",
]
