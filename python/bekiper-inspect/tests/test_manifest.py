from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bekiper_inspect.manifest import (
    HeuristicMeta,
    InspectionManifest,
    PerimeterSpec,
    VideoMeta,
)


class TestVideoMeta:
    def test_duration_seconds(self):
        v = VideoMeta(fps=30.0, total_frames=9000, resolution=(1920, 1080), meters_per_pixel=0.001)
        assert abs(v.duration_seconds - 300.0) < 1e-9

    def test_duration_zero_fps(self):
        v = VideoMeta(fps=0.0, total_frames=100, resolution=(640, 480), meters_per_pixel=0.001)
        assert v.duration_seconds == 0.0

    def test_meters_to_pixels(self):
        v = VideoMeta(fps=30.0, total_frames=100, resolution=(640, 480), meters_per_pixel=0.001)
        assert abs(v.meters_to_pixels(0.1) - 100.0) < 1e-9

    def test_meters_to_pixels_zero_mpp(self):
        v = VideoMeta(fps=30.0, total_frames=100, resolution=(640, 480), meters_per_pixel=0.0)
        assert v.meters_to_pixels(0.1) == 0.1

    def test_pixel_coords_from_metric(self):
        v = VideoMeta(fps=30.0, total_frames=100, resolution=(640, 480), meters_per_pixel=0.002)
        coords = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = v.pixel_coords_from_metric(coords)
        np.testing.assert_allclose(result, [[50.0, 100.0], [150.0, 200.0]])


class TestHeuristicMeta:
    def test_percentage(self):
        h = HeuristicMeta(name="test", result_column="test", true_frames=40, total_frames=100, seconds=1.33)
        assert abs(h.percentage - 40.0) < 1e-9

    def test_percentage_zero_total(self):
        h = HeuristicMeta(name="test", result_column="test", true_frames=0, total_frames=0, seconds=0.0)
        assert h.percentage == 0.0


class TestPerimeterSpec:
    def test_center_from_circle_params(self):
        p = PerimeterSpec(label="arena", shape="circle", params={"center_x": 1.0, "center_y": 2.0, "radius": 3.0})
        assert p.center == (1.0, 2.0)

    def test_center_from_vertices(self):
        p = PerimeterSpec(label="pen", shape="polygon", params={"vertices": [[0, 0], [2, 0], [2, 2], [0, 2]]})
        center = p.center
        assert center is not None
        assert abs(center[0] - 1.0) < 1e-9
        assert abs(center[1] - 1.0) < 1e-9

    def test_center_none(self):
        p = PerimeterSpec(label="x", shape="unknown", params={})
        assert p.center is None

    def test_vertices_present(self):
        p = PerimeterSpec(label="pen", shape="polygon", params={"vertices": [[0, 0], [1, 0], [1, 1]]})
        verts = p.vertices
        assert verts is not None
        assert verts.shape == (3, 2)

    def test_vertices_absent(self):
        p = PerimeterSpec(label="c", shape="circle", params={"center_x": 0, "center_y": 0, "radius": 1})
        assert p.vertices is None


class TestInspectionManifest:
    def test_from_directory(self, inspection_dir: Path):
        manifest = InspectionManifest.from_directory(inspection_dir)
        assert manifest.video.fps == 30.0
        assert manifest.video.total_frames == 100
        assert manifest.video.resolution == (640, 480)
        assert len(manifest.labels) == 2
        assert "nose" in manifest.labels

    def test_from_directory_no_toml(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="inspection.toml"):
            InspectionManifest.from_directory(tmp_path)

    def test_from_directory_no_parquet(self, tmp_path: Path):
        (tmp_path / "x_inspection.toml").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="evaluated.parquet"):
            InspectionManifest.from_directory(tmp_path)

    def test_coordinate_columns(self, sample_manifest: InspectionManifest):
        assert "nose" in sample_manifest.coordinate_columns
        assert sample_manifest.coordinate_columns["nose"].x == "nose_x"
        assert sample_manifest.coordinate_columns["nose"].y == "nose_y"

    def test_perimeters_loaded(self, sample_manifest: InspectionManifest):
        assert len(sample_manifest.perimeters) == 4
        labels = {p.label for p in sample_manifest.perimeters}
        assert labels == {"arena", "zone_a", "pen", "ymaze"}

    def test_perimeter_shapes(self, sample_manifest: InspectionManifest):
        shapes = {p.label: p.shape for p in sample_manifest.perimeters}
        assert shapes["arena"] == "circle"
        assert shapes["zone_a"] == "rectangle"
        assert shapes["pen"] == "polygon"
        assert shapes["ymaze"] == "radial_maze"

    def test_heuristics_loaded(self, sample_manifest: InspectionManifest):
        assert len(sample_manifest.heuristics) == 3
        names = {h.name for h in sample_manifest.heuristics}
        assert names == {"body_proximity", "olfaction", "freezing"}

    def test_from_files_resolves_relative_frame_and_video_paths(
        self,
        tmp_path: Path,
    ):
        repo_like_root = tmp_path / "repo"
        assets_dir = repo_like_root / "assets"
        manifest_dir = repo_like_root / "nested" / "output" / "run"
        assets_dir.mkdir(parents=True)
        manifest_dir.mkdir(parents=True)

        frame_path = assets_dir / "frame.jpeg"
        video_path = assets_dir / "video.mp4"
        frame_path.write_bytes(b"jpeg")
        video_path.write_bytes(b"mp4")

        toml_path = manifest_dir / "example_inspection.toml"
        toml_path.write_text(
            """
frame_image = "assets/frame.jpeg"
video_path = "assets/video.mp4"
labels = ["nose"]

[video]
fps = 30.0
total_frames = 2
resolution = [10, 10]
meters_per_pixel = 0.001

[label_colors]
nose = "#ffffff"

[coordinate_columns.nose]
x = "nose_x"
y = "nose_y"

[[perimeters]]
label = "arena"
shape = "circle"
center_x = 0.1
center_y = 0.1
radius = 0.05

[[heuristics]]
name = "olfaction"
result_column = "olfaction"
true_frames = 1
total_frames = 2
seconds = 0.033
""".strip()
        )
        parquet_path = manifest_dir / "example_evaluated.parquet"
        pl.DataFrame(
            {
                "nose_x": [0.1, 0.2],
                "nose_y": [0.1, 0.2],
                "olfaction": [True, False],
            }
        ).write_parquet(parquet_path)

        manifest = InspectionManifest.from_files(toml_path, parquet_path)
        assert manifest.frame_image == str(frame_path.resolve())
        assert manifest.video.path == str(video_path.resolve())

    def test_settings(self, sample_manifest: InspectionManifest):
        assert abs(sample_manifest.settings.minimum_seconds_tolerance - 0.5) < 1e-9
        assert abs(sample_manifest.settings.meters_per_pixel - 0.001) < 1e-9

    def test_get_coordinates(self, sample_manifest: InspectionManifest):
        coords = sample_manifest.get_coordinates("nose")
        assert coords.shape == (100, 2)
        assert coords.dtype == np.float64

    def test_get_coordinates_missing_label(self, sample_manifest: InspectionManifest):
        with pytest.raises(KeyError):
            sample_manifest.get_coordinates("nonexistent")

    def test_get_boolean_index(self, sample_manifest: InspectionManifest):
        idx = sample_manifest.get_boolean_index("body_proximity")
        assert idx.shape == (100,)
        assert idx.dtype == bool

    def test_get_boolean_index_missing(self, sample_manifest: InspectionManifest):
        with pytest.raises(StopIteration):
            sample_manifest.get_boolean_index("nonexistent")

    def test_get_perimeter(self, sample_manifest: InspectionManifest):
        p = sample_manifest.get_perimeter("arena")
        assert p.shape == "circle"
        assert abs(p.params["radius"] - 0.14) < 1e-9

    def test_get_perimeter_missing(self, sample_manifest: InspectionManifest):
        with pytest.raises(StopIteration):
            sample_manifest.get_perimeter("nonexistent")

    def test_label_colors(self, sample_manifest: InspectionManifest):
        assert sample_manifest.label_colors["nose"] == "#1f77b4"

    def test_df_shape(self, sample_manifest: InspectionManifest):
        assert sample_manifest.df.shape[0] == 100
