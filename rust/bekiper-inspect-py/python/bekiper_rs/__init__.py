"""Rust-backed inspection types for bekiper.

This package re-exports the PyO3 bindings defined in `_bekiper_rs` so that
Python code can import a stable surface (``from bekiper_rs import ...``)
without depending on the private extension module name.

The Rust crate `bekiper-core` is the single source of truth for these
types; the dataclasses that previously lived in
``bekiper_inspect.manifest._types`` and ``_features_parts`` are now thin
re-exports of the classes defined here.
"""

from __future__ import annotations

from ._bekiper_rs import (
    CoordinateColumnSpec,
    EnvCheckReport,
    FeatureAngle,
    FeatureArmSequence,
    FeatureCenterPeriphery,
    FeatureDetectionFields,
    FeatureHeuristic,
    FeatureHeuristicCloud,
    FeatureHeuristicDecomposition,
    FeatureNortDiscrimination,
    FeaturePerimeter,
    FeatureProximity,
    FeatureQuadrant,
    FeatureRayCast,
    FeatureScaleDerivation,
    FeatureStateDistribution,
    FeatureTimeSeries,
    FeatureTrajectory,
    FeatureYMazeArmSections,
    FeatureZone,
    HeuristicDecompositionComponent,
    HeuristicMeta,
    IngressSessionContext,
    InspectionSettings,
    LoadedManifest,
    PerimeterSpec,
    QuadrantSpec,
    QualiaChannelSpec,
    ScaleDerivation,
    VideoMeta,
    check_env,
    experiment_descriptors,
    load_inspection_toml,
    theme_css,
    theme_palette,
    theme_rc,
    theme_resolve,
)

#: Tagged-union alias matching the prior Python dataclass layout. The
#: dispatcher in ``bekiper_inspect.plot.feature`` switches on
#: ``feature.kind`` to pick the right renderer.
FeatureInspection = (
    FeaturePerimeter
    | FeatureZone
    | FeatureQuadrant
    | FeatureHeuristic
    | FeatureHeuristicCloud
    | FeatureHeuristicDecomposition
    | FeatureAngle
    | FeatureTrajectory
    | FeatureScaleDerivation
    | FeatureRayCast
    | FeatureDetectionFields
    | FeatureTimeSeries
    | FeatureProximity
    | FeatureNortDiscrimination
    | FeatureArmSequence
    | FeatureYMazeArmSections
    | FeatureCenterPeriphery
    | FeatureStateDistribution
)

__all__ = [
    "CoordinateColumnSpec",
    "EnvCheckReport",
    "FeatureAngle",
    "FeatureArmSequence",
    "FeatureCenterPeriphery",
    "FeatureDetectionFields",
    "FeatureHeuristic",
    "FeatureHeuristicCloud",
    "FeatureHeuristicDecomposition",
    "FeatureInspection",
    "FeatureNortDiscrimination",
    "FeaturePerimeter",
    "FeatureProximity",
    "FeatureQuadrant",
    "FeatureRayCast",
    "FeatureScaleDerivation",
    "FeatureStateDistribution",
    "FeatureTimeSeries",
    "FeatureTrajectory",
    "FeatureYMazeArmSections",
    "FeatureZone",
    "HeuristicDecompositionComponent",
    "HeuristicMeta",
    "IngressSessionContext",
    "InspectionSettings",
    "LoadedManifest",
    "PerimeterSpec",
    "QuadrantSpec",
    "QualiaChannelSpec",
    "ScaleDerivation",
    "VideoMeta",
    "check_env",
    "experiment_descriptors",
    "load_inspection_toml",
    "theme_css",
    "theme_palette",
    "theme_rc",
    "theme_resolve",
]
