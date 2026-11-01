from .test_fixture_video import *


class TestPlotFeatureOnFrameNortAndManifest:
    def test_nort_discrimination_with_qualia(self, manifest: InspectionManifest):
        """NORT figure with qualia decomposition renders qualia text overlay."""
        from bekiper_inspect.manifest import QualiaChannelSpec

        manifest.perimeters.append(
            PerimeterSpec(
                label="novel",
                shape="circle",
                params={"center_x": 0.08, "center_y": 0.15, "radius": 0.02},
            )
        )
        feature = FeatureNortDiscrimination(
            label="object_discrimination",
            object_labels=["center", "novel"],
            exploration_times=[7.5, 12.5],
            coordinate_labels=["nose"],
            description="NORT discrimination with qualia",
            discrimination_index=0.25,
            recognition_index=0.625,
            qualia_channels=[
                QualiaChannelSpec(
                    heuristic_type="body_proximity",
                    per_object_seconds=[5.0, 8.0],
                    per_object_bouts=[2, 3],
                    relative_bias=[38.5, 61.5],
                ),
                QualiaChannelSpec(
                    heuristic_type="olfaction",
                    per_object_seconds=[6.0, 10.0],
                    per_object_bouts=[3, 4],
                    relative_bias=[37.5, 62.5],
                ),
                QualiaChannelSpec(
                    heuristic_type="whisker_interaction",
                    per_object_seconds=[3.5, 6.0],
                    per_object_bouts=[2, 2],
                    relative_bias=[36.8, 63.2],
                ),
            ],
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        texts = [t.get_text() for t in fig.axes[0].texts]
        # Should have both DI text and Qualia text.
        assert any("DI" in t for t in texts)
        assert any("Qualia" in t for t in texts)

    def test_nort_with_alternation_count_and_merged(self, manifest: InspectionManifest):
        """NORT figure with alternation count and merged channels."""
        from bekiper_inspect.manifest import QualiaChannelSpec

        manifest.perimeters.append(
            PerimeterSpec(
                label="novel",
                shape="circle",
                params={"center_x": 0.08, "center_y": 0.15, "radius": 0.02},
            )
        )
        feature = FeatureNortDiscrimination(
            label="object_discrimination",
            object_labels=["center", "novel"],
            exploration_times=[7.5, 12.5],
            coordinate_labels=["nose"],
            description="NORT with alternation + merged",
            discrimination_index=0.25,
            recognition_index=0.625,
            qualia_channels=[
                QualiaChannelSpec(
                    heuristic_type="body_proximity",
                    per_object_seconds=[5.0, 8.0],
                    per_object_bouts=[2, 3],
                    relative_bias=[38.5, 61.5],
                ),
            ],
            alternation_count=7,
            merged_channels={
                "close_investigation": [3.0, 5.0],
            },
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("alternation" in t for t in texts)
        assert any("close investigation" in t for t in texts)

    def test_nort_qualia_without_optional_fields(self, manifest: InspectionManifest):
        """NORT figure with qualia but no alternation or merged channels."""
        from bekiper_inspect.manifest import QualiaChannelSpec

        manifest.perimeters.append(
            PerimeterSpec(
                label="novel",
                shape="circle",
                params={"center_x": 0.08, "center_y": 0.15, "radius": 0.02},
            )
        )
        feature = FeatureNortDiscrimination(
            label="object_discrimination",
            object_labels=["center", "novel"],
            exploration_times=[7.5, 12.5],
            coordinate_labels=["nose"],
            description="NORT without alternation",
            qualia_channels=[
                QualiaChannelSpec(
                    heuristic_type="olfaction",
                    per_object_seconds=[6.0, 10.0],
                    per_object_bouts=[3, 4],
                    relative_bias=[37.5, 62.5],
                ),
            ],
            alternation_count=None,
            merged_channels=None,
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("Qualia" in t for t in texts)
        # No alternation count or merged text when not provided.
        assert not any("alternations:" in t for t in texts)
        assert not any("close investigation" in t for t in texts)

    def test_nort_no_qualia_renders_clean(self, manifest: InspectionManifest):
        """NORT figure without qualia should still render DI/RI."""
        manifest.perimeters.append(
            PerimeterSpec(
                label="novel",
                shape="circle",
                params={"center_x": 0.08, "center_y": 0.15, "radius": 0.02},
            )
        )
        feature = FeatureNortDiscrimination(
            label="object_discrimination",
            object_labels=["center", "novel"],
            exploration_times=[7.5, 12.5],
            coordinate_labels=["nose"],
            description="NORT clean",
            discrimination_index=0.25,
            recognition_index=0.625,
            qualia_channels=None,
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("DI" in t for t in texts)
        assert not any("Qualia" in t for t in texts)

    def test_arm_sequence_renders_numbered_entries(self, manifest: InspectionManifest):
        feature = FeatureArmSequence(
            label="arm_sequence",
            arm_sequence=["center", "center"],
            coordinate_labels=["nose"],
            description="Arm entry sequence",
            alternation_pct=66.7,
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("Alternation" in t for t in texts)

    def test_ymaze_arm_sections_draws_given_crossline_segment(self, manifest: InspectionManifest):
        manifest.perimeters.append(
            PerimeterSpec(
                label="ymaze",
                shape="radial_maze",
                params={
                    "center_vertices": [[0.14, 0.14], [0.16, 0.14], [0.15, 0.16]],
                    "arms": [[[0.14, 0.16], [0.16, 0.16], [0.16, 0.28], [0.14, 0.28]]],
                },
            )
        )
        p1, p2 = (0.14, 0.2), (0.16, 0.19)
        feature = FeatureYMazeArmSections(
            label="ymaze_arm_sections",
            perimeter_label="ymaze",
            proximal_threshold=0.5,
            coordinate_labels=["nose"],
            description="Y-maze arm section thresholds",
            threshold_lines=[("A", p1, p2)],
        )
        import numpy as np
        from bekiper_inspect.plot.feature._helpers import _calibration_resolution, _metric_to_frame_pixels

        fig = plot_feature_on_frame(manifest, feature)
        ax = fig.axes[0]
        expected = _metric_to_frame_pixels(
            np.array([p1, p2]),
            manifest.settings.meters_per_pixel,
            _calibration_resolution(manifest),
            ax.images[0].get_array().shape,
        )
        drawn = [np.column_stack(line.get_data()) for line in ax.lines if len(line.get_xdata()) == 2]
        assert any(np.allclose(d, expected) for d in drawn), (drawn, expected)
        assert "50%" in [t.get_text() for t in ax.texts]

    def test_ymaze_arm_sections_renders_threshold_lines(self, manifest: InspectionManifest):
        manifest.perimeters.append(
            PerimeterSpec(
                label="ymaze",
                shape="radial_maze",
                params={
                    "center_vertices": [[0.14, 0.14], [0.16, 0.14], [0.15, 0.16]],
                    "arms": [
                        [[0.14, 0.16], [0.16, 0.16], [0.16, 0.28], [0.14, 0.28]],
                        [[0.14, 0.14], [0.04, 0.08], [0.05, 0.06], [0.15, 0.12]],
                        [[0.16, 0.14], [0.26, 0.08], [0.25, 0.06], [0.15, 0.12]],
                    ],
                },
            )
        )
        feature = FeatureYMazeArmSections(
            label="ymaze_arm_sections",
            perimeter_label="ymaze",
            proximal_threshold=0.25,
            coordinate_labels=["nose"],
            description="Y-maze arm section thresholds",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        ax = fig.axes[0]
        assert len(ax.lines) >= 4  # perimeter context plus one threshold per arm
        assert len(ax.patches) >= 6  # center/arm sections per arm
        texts = [t.get_text() for t in ax.texts]
        assert "25%" in texts
        assert texts.count("center") == 3
        assert texts.count("arm") == 3
        assert "proximal" not in texts

    def test_center_periphery_renders_zones_and_heatmap(self, manifest: InspectionManifest):
        # Add an arena perimeter.
        manifest.perimeters.append(
            PerimeterSpec(
                label="arena",
                shape="circle",
                params={"center_x": 0.15, "center_y": 0.15, "radius": 0.14},
            )
        )
        feature = FeatureCenterPeriphery(
            label="center_periphery",
            center_perimeter_label="center",
            arena_perimeter_label="arena",
            coordinate_labels=["nose"],
            center_time_pct=35.2,
            periphery_time_pct=64.8,
            description="Center vs periphery",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("Center" in t for t in texts)
        assert any("Periphery" in t for t in texts)


# ── Auto-summary annotation ────────────────────────────────────────────


class TestAutoSummaryAnnotation:
    """Summary stats textbox on Heuristic / Zone figures."""

    def test_heuristic_has_summary_annotation(self, manifest: InspectionManifest):
        feature = FeatureHeuristic(
            label="heuristic_in_center",
            result_column="in_center",
            coordinate_labels=["nose"],
            description="In center heuristic",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("33%" in t for t in texts)
        assert any("10 / 30" in t for t in texts)

    def test_zone_has_summary_annotation(self, manifest: InspectionManifest):
        feature = FeatureZone(
            label="zone_in_center",
            perimeter_label="center",
            result_column="in_center",
            coordinate_labels=["nose"],
            description="In center zone",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("33%" in t for t in texts)
        assert any("10 / 30" in t for t in texts)

    def test_summary_values_correct(self, manifest: InspectionManifest):
        """10 true frames out of 30 at 30 fps => 0.3s, 33%."""
        feature = FeatureHeuristic(
            label="heuristic_in_center",
            result_column="in_center",
            coordinate_labels=["nose"],
            description="In center heuristic",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        texts = [t.get_text() for t in fig.axes[0].texts]
        # The HeuristicMeta has seconds = 10/30 ≈ 0.333 → "0.3s"
        summary = [t for t in texts if "frames" in t]
        assert len(summary) == 1
        assert "0.3s" in summary[0]
        assert "33%" in summary[0]
        assert "10 / 30 frames" in summary[0]

    def test_no_crash_when_heuristic_meta_missing(self, manifest: InspectionManifest):
        """When result_column has no matching HeuristicMeta, fall back to
        on-the-fly computation instead of crashing."""
        # Clear the heuristics list so lookup falls through to the fallback.
        manifest.heuristics = []
        feature = FeatureHeuristic(
            label="heuristic_in_center",
            result_column="in_center",
            coordinate_labels=["nose"],
            description="In center heuristic",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        texts = [t.get_text() for t in fig.axes[0].texts]
        # Fallback computes the same values from the boolean column.
        assert any("33%" in t for t in texts)
        assert any("10 / 30" in t for t in texts)


# ── Golden fixture regression: every variant must decode correctly ──────


class TestGoldenFeatureInspections:
    """The `golden_inspection.toml` fixture carries one of every
    FeatureInspection variant. These tests assert the Python parser
    matches the Rust serializer — schema drift here means the Rust
    snapshot test should also be failing.
    """

    def test_all_variants_present(self, sample_manifest: InspectionManifest):
        kinds = {fi.kind for fi in sample_manifest.feature_inspections}
        assert kinds == {
            "state_distribution",
            "perimeter",
            "zone",
            "quadrant",
            "heuristic",
            "heuristic_decomposition",
            "heuristic_cloud",
            "angle",
            "trajectory",
            "ray_cast",
            "detection_fields",
            "scale_derivation",
            "time_series",
            "proximity",
            "object_discrimination",
            "arm_sequence",
            "ymaze_arm_sections",
            "center_periphery",
        }

    def test_quadrant_spec_loaded(self, sample_manifest: InspectionManifest):
        # Golden carries a single 4x4 grid; assert it round-trips.
        assert len(sample_manifest.quadrants) == 1
        q = sample_manifest.quadrants[0]
        assert q.label == "open_field_grid"
        assert (q.rows, q.cols) == (4, 4)

    def test_zone_carries_representative_frame_index(self, sample_manifest: InspectionManifest):
        zone = next(fi for fi in sample_manifest.feature_inspections if fi.kind == "zone")
        assert isinstance(zone, FeatureZone)
        assert zone.representative_frame_index == 42

    def test_other_variants_default_rep_frame_to_none(self, sample_manifest: InspectionManifest):
        for fi in sample_manifest.feature_inspections:
            if fi.kind in ("zone", "scale_derivation"):
                continue
            assert fi.representative_frame_index is None

    def test_nort_qualia_channels_parsed(self, sample_manifest: InspectionManifest):
        nort = next(fi for fi in sample_manifest.feature_inspections if fi.kind == "object_discrimination")
        assert isinstance(nort, FeatureNortDiscrimination)
        assert nort.qualia_channels is not None
        assert len(nort.qualia_channels) == 3
        types = [ch.heuristic_type for ch in nort.qualia_channels]
        assert "body_proximity" in types
        assert "olfaction" in types
        assert "whisker_interaction" in types
        # Verify data integrity.
        bp = next(ch for ch in nort.qualia_channels if ch.heuristic_type == "body_proximity")
        assert bp.per_object_seconds == [8.0, 5.0]
        assert bp.per_object_bouts == [3, 2]
        assert bp.relative_bias == [61.5, 38.5]

    def test_nort_alternation_count_parsed(self, sample_manifest: InspectionManifest):
        nort = next(fi for fi in sample_manifest.feature_inspections if fi.kind == "object_discrimination")
        assert isinstance(nort, FeatureNortDiscrimination)
        assert nort.alternation_count == 5
        assert nort.protocol == "nort_novelty"
        assert nort.title_override == "Object Discrimination: NORT Novelty"

    def test_nort_merged_channels_default_none(self, sample_manifest: InspectionManifest):
        nort = next(fi for fi in sample_manifest.feature_inspections if fi.kind == "object_discrimination")
        assert isinstance(nort, FeatureNortDiscrimination)
        # Golden fixture has no merged_channels — defaults to None.
        assert nort.merged_channels is None
