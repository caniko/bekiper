from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from bekiper_inspect.manifest import FeaturePerimeter, FeatureTrajectory
from bekiper_inspect.plot.feature import plot_feature_on_frame
from bekiper_inspect.plot.feature._grouping import StoryGroup
from bekiper_inspect.plot.feature._story_parts.plot_story import _resolve_frame_for_feature, plot_story


def test_cropped_table_feature_and_story_use_source_frame_cache(sample_manifest, tmp_path: Path):
    manifest = sample_manifest
    manifest.df = manifest.df.with_columns(pl.Series("frame", range(76, 76 + manifest.df.height)))
    manifest.frames_dir = str(tmp_path)
    manifest.video.path = None
    feature = FeaturePerimeter(
        label="arena", perimeter_label="arena", description="Recorded arena", representative_frame_index=0
    )
    wrong_frame = np.full((240, 320, 3), 20, dtype=np.uint8)
    source_frame = np.full((240, 320, 3), 180, dtype=np.uint8)
    Image.fromarray(wrong_frame).save(tmp_path / "frame_000000.png")
    Image.fromarray(source_frame).save(tmp_path / "frame_000076.png")

    fig = plot_feature_on_frame(manifest, feature)
    assert fig is not None
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), source_frame)
    assert any("frame 76" in text.get_text() for text in fig.axes[0].texts)
    background, row = _resolve_frame_for_feature(manifest, feature)
    np.testing.assert_array_equal(background, source_frame)
    assert row == 0  # Diagnostics still index the evaluated table's first row.


def test_geometry_examples_label_the_selected_source_frame(sample_manifest):
    manifest = sample_manifest
    manifest.df = manifest.df.with_columns(pl.Series("frame", range(76, 76 + manifest.df.height)))
    manifest.video.path = None
    feature = FeatureTrajectory(
        label="skeleton_geometry_0",
        coordinate_labels=["nose"],
        description="Recorded geometry",
        representative_frame_index=0,
    )
    fig = plot_story(manifest, StoryGroup(key="geometry_examples", title="Geometry examples", features=[feature]))
    assert fig is not None
    assert fig.axes[0].get_title() == "Frame 76"


def test_inverse_display_projection_preserves_native_masks_scalars_and_bytes(inspection_dir):
    import tomllib
    import tomli_w
    from bekiper_inspect.manifest import InspectionManifest

    toml_path = inspection_dir / "experiment_inspection.toml"
    parquet_path = inspection_dir / "experiment_evaluated.parquet"
    native = pl.read_parquet(parquet_path)
    spec = tomllib.loads(toml_path.read_text())
    spec["settings"]["invert_y"] = True
    spec["feature_inspections"].append(
        {
            "kind": "heuristic_decomposition",
            "label": "ray_binding",
            "heuristic_name": "olfaction",
            "result_column": "olfaction",
            "description": "Native ray",
            "coordinate_labels": ["nose"],
            "components": [
                {
                    "label": "ray",
                    "result_column": "olfaction",
                    "coordinate_labels": ["nose"],
                    "ray_origin_label": "nose",
                    "ray_direction_x_column": "dx",
                    "ray_direction_y_column": "dy",
                    "ray_target_x_column": "target_x",
                    "ray_target_y_column": "target_y",
                    "value_column": "recorded_angle",
                    "value_unit": "deg",
                }
            ],
        }
    )
    native = native.with_columns(
        pl.lit(0.02).alias("dx"),
        pl.lit(-0.01).alias("dy"),
        pl.lit(0.1).alias("target_x"),
        pl.lit(0.08).alias("target_y"),
        pl.lit(17.0).alias("recorded_angle"),
    )
    native.write_parquet(parquet_path)
    toml_path.write_text(tomli_w.dumps(spec))
    before = (toml_path.read_bytes(), parquet_path.read_bytes())
    display = InspectionManifest.from_files(toml_path, parquet_path)
    height = display.video.resolution[1] * display.settings.meters_per_pixel
    np.testing.assert_allclose(display.df["nose_y"], height - native["nose_y"])
    np.testing.assert_allclose(display.df["target_y"], height - native["target_y"])
    np.testing.assert_allclose(display.df["dy"], -native["dy"])
    np.testing.assert_allclose(display.df["heading_rad"], -native["heading_rad"])
    assert display.df["recorded_angle"].equals(native["recorded_angle"])
    assert display.df["olfaction"].equals(native["olfaction"])
    expected = spec["perimeters"][0]["center_y"]
    assert np.isclose(display.perimeters[0].params["center_y"], height - expected)
    assert before == (toml_path.read_bytes(), parquet_path.read_bytes())


def test_source_pixel_geometry_and_heading_match_native_video_fixture(inspection_dir):
    import tomllib
    import tomli_w
    from bekiper_inspect.manifest import InspectionManifest

    toml_path = inspection_dir / "experiment_inspection.toml"
    parquet_path = inspection_dir / "experiment_evaluated.parquet"
    spec = tomllib.loads(toml_path.read_text())
    spec["video"].update(resolution=[320, 240], meters_per_pixel=0.001)
    spec["settings"].update(invert_y=True, meters_per_pixel=0.001)
    spec["perimeters"] = [{"shape": "circle", "label": "arena", "center_x": 0.16, "center_y": 0.18, "radius": 0.01}]
    spec["pixel_perimeters"] = []
    spec["feature_inspections"] = [
        {
            "kind": "perimeter",
            "label": "arena",
            "perimeter_label": "arena",
            "description": "Native geometry",
            "representative_frame_index": 0,
        },
        {
            "kind": "ray_cast",
            "label": "heading",
            "origin_label": "nose",
            "angle_column": "heading_rad",
            "length_meters": 0.03,
            "cone_width_deg": 20.0,
            "description": "Native heading",
            "representative_frame_index": 0,
        },
    ]
    df = pl.read_parquet(parquet_path)
    df = df.with_columns(
        pl.Series("frame", range(76, 76 + df.height)),
        pl.lit(0.10).alias("nose_x"),
        pl.lit(0.20).alias("nose_y"),
        pl.lit(-np.pi / 2).alias("heading_rad"),
    )
    df.write_parquet(parquet_path)
    toml_path.write_text(tomli_w.dumps(spec))
    frames = inspection_dir / "frames"
    frames.mkdir(exist_ok=True)
    background = np.full((240, 320, 3), 180, dtype=np.uint8)
    Image.fromarray(background).save(frames / "frame_000076.png")
    display = InspectionManifest.from_files(toml_path, parquet_path)
    geometry = plot_feature_on_frame(display, display.feature_inspections[0])
    heading = plot_feature_on_frame(display, display.feature_inspections[1])
    np.testing.assert_allclose(geometry.axes[0].patches[0].center, [160, 60])
    np.testing.assert_allclose(heading.axes[0].lines[0].get_xydata(), [[100, 40], [100, 70]])
    for figure in [geometry, heading]:
        np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), background)
        assert any("frame 76" in text.get_text() for text in figure.axes[0].texts)
