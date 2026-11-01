"""Metric geometry from non-square source pixels must land on the true frame pixels."""

from types import SimpleNamespace

import numpy as np

from bekiper_inspect.plot.feature._helpers import _calibration_resolution, _metric_to_frame_pixels


def _manifest(mpp_y):
    derivation = SimpleNamespace(meters_per_pixel=0.40 / 204, meters_per_pixel_y=mpp_y)
    return SimpleNamespace(video=SimpleNamespace(resolution=(224, 224)), scale_derivation=derivation)


def test_non_square_box_corner_maps_to_its_video_pixel():
    manifest = _manifest(0.40 / 210)
    # The 40 x 40 cm floor corner is (204, 210) px in a 204 x 210 px box.
    corner = np.array([[0.40, 0.40]])
    px = _metric_to_frame_pixels(corner, 0.40 / 204, _calibration_resolution(manifest), (224, 224))
    np.testing.assert_allclose(px, [[204.0, 210.0]])
    # A 2x display frame scales both axes from the same true pixels.
    px2 = _metric_to_frame_pixels(corner, 0.40 / 204, _calibration_resolution(manifest), (448, 448))
    np.testing.assert_allclose(px2, [[408.0, 420.0]])


def test_square_pixels_and_older_manifests_are_unchanged():
    for manifest in (_manifest(None), SimpleNamespace(video=SimpleNamespace(resolution=(224, 224)))):
        assert _calibration_resolution(manifest) == (224, 224)
