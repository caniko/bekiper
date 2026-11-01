from .test_fixture_video import *


class TestFrameCache:
    def test_cache_reuses_same_index(self, fixture_video: Path):
        _read_frame_cached.cache_clear()
        a = read_frame_at(fixture_video, 5)
        b = read_frame_at(fixture_video, 5)
        info = _read_frame_cached.cache_info()
        # Two reads, one decode (the second is a cache hit)
        assert info.hits >= 1
        assert a is not None
        np.testing.assert_array_equal(a, b)

    def test_returns_none_for_missing_path(self):
        assert read_frame_at(None, 0) is None

    def test_reads_distinct_indices(self, fixture_video: Path):
        _read_frame_cached.cache_clear()
        f0 = read_frame_at(fixture_video, 0)
        f10 = read_frame_at(fixture_video, 10)
        assert f0 is not None and f10 is not None
        # The synthetic video uses different colors per frame, so frame 0
        # and frame 10 must differ.
        assert not np.array_equal(f0, f10)

    def test_prefers_pre_extracted_png(self, tmp_path: Path):
        """When a pre-extracted PNG exists in a frames_dir, read_frame_at
        loads it instead of attempting video decode."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        # Create a recognizable 4x4 red PNG.
        red = np.full((4, 4, 3), [0, 0, 255], dtype=np.uint8)  # BGR
        png_path = frames_dir / "frame_000007.png"
        cv2.imwrite(str(png_path), red)

        # Pass video_path=None so any video decode would fail — the PNG
        # must be found via frames_dir alone.
        frame = read_frame_at(None, 7, frames_dir=frames_dir)
        assert frame is not None
        # cv2.imread → cvtColor BGR→RGB, so we expect RGB red.
        np.testing.assert_array_equal(frame[0, 0], [255, 0, 0])

    def test_frames_dir_fallback_to_video(self, fixture_video: Path, tmp_path: Path):
        """When frames_dir has no matching PNG, falls back to video decode."""
        _read_frame_cached.cache_clear()
        empty_dir = tmp_path / "empty_frames"
        empty_dir.mkdir()

        frame = read_frame_at(fixture_video, 0, frames_dir=empty_dir)
        assert frame is not None  # decoded from the video

    def test_frames_dir_none_video_none_returns_none(self):
        """Both sources missing → None."""
        assert read_frame_at(None, 0, frames_dir=None) is None


# ── Per-feature styling overrides (P4.2) ──────────────────────────────


class TestStylingOverrides:
    def test_title_override_used_in_figure(self, manifest: InspectionManifest):
        feature = FeatureTrajectory(
            label="trajectory_nose",
            coordinate_labels=["nose"],
            description="Nose trajectory",
            title_override="My Custom Title",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert fig.axes[0].get_title() == "My Custom Title"

    def test_color_override_defaults_to_none(self):
        feature = FeatureTrajectory(
            label="trajectory_nose",
            coordinate_labels=["nose"],
            description="Nose trajectory",
        )
        assert feature.color is None
        assert feature.linewidth is None
        assert feature.title_override is None

    def test_styling_fields_parsed_from_toml(self):
        from bekiper_inspect.manifest._features import _parse_feature_inspection

        raw = {
            "kind": "trajectory",
            "label": "trajectory_nose",
            "coordinate_label": "nose",
            "description": "Nose trajectory",
            "color": "red",
            "linewidth": 5.0,
            "title_override": "Custom",
        }
        feature = _parse_feature_inspection(raw)
        assert feature is not None
        assert feature.color == "red"
        assert feature.linewidth == 5.0
        assert feature.title_override == "Custom"
        # Backwards-compat: old coordinate_label wrapped into coordinate_labels.
        assert feature.coordinate_labels == ["nose"]

    def test_styling_fields_absent_parse_to_none(self):
        from bekiper_inspect.manifest._features import _parse_feature_inspection

        raw = {
            "kind": "perimeter",
            "label": "perimeter_arena",
            "perimeter_label": "arena",
            "description": "Arena perimeter",
        }
        feature = _parse_feature_inspection(raw)
        assert feature is not None
        assert feature.color is None
        assert feature.linewidth is None
        assert feature.title_override is None

    def test_ymaze_arm_sections_parsed_from_toml(self):
        from bekiper_inspect.manifest._features import _parse_feature_inspection

        raw = {
            "kind": "ymaze_arm_sections",
            "label": "ymaze_arm_sections",
            "perimeter_label": "ymaze",
            "proximal_threshold": 0.25,
            "coordinate_labels": ["nose"],
            "description": "Y-maze arm section thresholds",
        }
        feature = _parse_feature_inspection(raw)
        assert isinstance(feature, FeatureYMazeArmSections)
        assert feature.perimeter_label == "ymaze"
        assert feature.proximal_threshold == 0.25
        assert feature.coordinate_labels == ["nose"]

    def test_golden_variants_have_no_styling_by_default(self, sample_manifest: InspectionManifest):
        for fi in sample_manifest.feature_inspections:
            assert fi.color is None
            assert fi.linewidth is None
            if fi.kind == "object_discrimination":
                assert fi.title_override == "Object Discrimination: NORT Novelty"
            else:
                assert fi.title_override is None
