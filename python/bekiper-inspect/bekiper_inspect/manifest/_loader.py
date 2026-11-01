from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from bekiper_rs import load_inspection_toml

from ._features import FeatureInspection
from ._types import (
    CoordinateColumnSpec,
    HeuristicMeta,
    IngressSessionContext,
    InspectionSettings,
    PerimeterSpec,
    QuadrantSpec,
    ScaleDerivation,
    VideoMeta,
)


@dataclass
class InspectionManifest:
    video: VideoMeta
    labels: list[str]
    label_colors: dict[str, str]
    coordinate_columns: dict[str, CoordinateColumnSpec]
    perimeters: list[PerimeterSpec]
    heuristics: list[HeuristicMeta]
    settings: InspectionSettings
    scale_derivation: ScaleDerivation | None
    df: pl.DataFrame
    quadrants: list[QuadrantSpec] = field(default_factory=list)
    feature_inspections: list[FeatureInspection] = field(default_factory=list)
    session_context: IngressSessionContext | None = None
    frame_image: str | None = None
    pixel_perimeters: list[PerimeterSpec] = field(default_factory=list)
    pixel_quadrants: list[QuadrantSpec] = field(default_factory=list)
    frames_dir: str | None = None

    @classmethod
    def from_directory(cls, path: Path) -> InspectionManifest:
        """Load a manifest from a directory containing inspection TOML + Parquet files.

        Expects either:
        - A single ``*_inspection.toml`` and ``*_evaluated.parquet``, or
        - Specific filenames passed via :meth:`from_files`.
        """
        toml_files = list(path.glob("*_inspection.toml"))
        parquet_files = list(path.glob("*_evaluated.parquet"))

        if not toml_files:
            msg = f"No *_inspection.toml found in {path}"
            raise FileNotFoundError(msg)
        if not parquet_files:
            msg = f"No *_evaluated.parquet found in {path}"
            raise FileNotFoundError(msg)

        return cls.from_files(toml_files[0], parquet_files[0])

    @classmethod
    def from_files(cls, toml_path: Path, parquet_path: Path) -> InspectionManifest:
        """Load from explicit file paths.

        TOML parsing and the construction of every typed field is delegated
        to the Rust loader in ``bekiper_rs``. This Python wrapper only
        attaches the polars DataFrame loaded from the evaluation Parquet.
        """
        loaded = load_inspection_toml(str(toml_path))
        df = pl.read_parquet(parquet_path)
        if loaded.settings.invert_y:
            # Non-square sources reflect about the source height in metres
            # on the vertical scale (metres_per_pixel_y when present).
            mpp_y = getattr(loaded.scale_derivation, "meters_per_pixel_y", None)
            height = loaded.video.resolution[1] * (mpp_y or loaded.settings.meters_per_pixel)
            if height <= 0:
                raise ValueError("Inspection Y transform needs source resolution and positive scale")
            # Only drawing inputs are projected back to source-image space. The
            # recorded Parquet, measured endpoints and native masks are untouched.
            df = df.with_columns(
                [(height - pl.col(name)).alias(name) for name in loaded.overlay_y_columns]
                + [(-pl.col(name)).alias(name) for name in loaded.overlay_vector_y_columns]
            )

        return cls(
            video=loaded.video,
            labels=list(loaded.labels),
            label_colors=dict(loaded.label_colors),
            coordinate_columns=dict(loaded.coordinate_columns),
            perimeters=list(loaded.perimeters),
            heuristics=list(loaded.heuristics),
            settings=loaded.settings,
            scale_derivation=loaded.scale_derivation,
            df=df,
            quadrants=list(loaded.quadrants),
            feature_inspections=list(loaded.feature_inspections),
            session_context=loaded.session_context,
            frame_image=loaded.frame_image,
            pixel_perimeters=list(loaded.pixel_perimeters),
            pixel_quadrants=list(loaded.pixel_quadrants),
            frames_dir=str(toml_path.parent / "frames"),
        )

    def get_coordinates(self, label: str) -> np.ndarray:
        """Get (N, 2) coordinate array for a label."""
        spec = self.coordinate_columns[label]
        x = self.df[spec.x].to_numpy()
        y = self.df[spec.y].to_numpy()
        return np.column_stack([x, y])

    def source_frame_index(self, row_index: int) -> int:
        """Map a representative table row to its recorded source-video frame.

        Feature nominations and diagnostics use evaluated-row positions. A
        cropped or sparse table's ``frame`` column retains the video positions.
        Legacy tables without that column start at video frame zero.
        """
        if "frame" not in self.df.columns:
            return row_index
        frames = self.df["frame"]
        if not frames.dtype.is_integer():
            raise ValueError("Inspection frame keys must be nonnegative integers")
        frame_index = frames[row_index]
        if frame_index is None or frame_index < 0:
            raise ValueError("Inspection frame keys must be nonnegative integers")
        return int(frame_index)

    def get_boolean_index(self, heuristic_name: str) -> np.ndarray:
        """Get boolean array for a heuristic result column."""
        meta = next(h for h in self.heuristics if h.name == heuristic_name)
        return cast(np.ndarray, self.df[meta.result_column].to_numpy())

    def get_perimeter(self, label: str) -> PerimeterSpec:
        """Get perimeter spec by label (metric space)."""
        return next(p for p in self.perimeters if p.label == label)

    def get_pixel_perimeter(self, label: str) -> PerimeterSpec | None:
        """Get pre-computed pixel-space perimeter by label.

        Returns ``None`` for old manifests that don't carry pixel geometry.
        """
        return next((p for p in self.pixel_perimeters if p.label == label), None)

    def get_quadrant(self, label: str) -> QuadrantSpec:
        """Get quadrant spec by label."""
        return next(q for q in self.quadrants if q.label == label)
