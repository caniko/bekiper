from ._make_angle_manifest import _make_angle_manifest, _make_whisker_manifest
from .test_fixture_video import *


class TestPlotFeatureOnFrame:
    def test_returns_none_when_no_video_and_no_resolution(self, manifest: InspectionManifest):
        manifest.video.path = None
        manifest.video.resolution = (0, 0)
        feature = FeaturePerimeter(
            label="perimeter_center",
            perimeter_label="center",
            description="x",
        )
        assert plot_feature_on_frame(manifest, feature) is None

    def test_blank_canvas_when_no_video_but_valid_resolution(self, manifest: InspectionManifest):
        manifest.video.path = None
        manifest.video.resolution = (320, 240)
        feature = FeaturePerimeter(
            label="perimeter_center",
            perimeter_label="center",
            description="Center perimeter",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)

    def test_frame_image_used_when_no_video(self, manifest: InspectionManifest, tmp_path: Path):
        # Write a real JPEG to disk so cv2.imread can load it.
        img = np.full((240, 320, 3), 100, dtype=np.uint8)
        jpeg_path = tmp_path / "0.frame-999.jpeg"
        cv2.imwrite(str(jpeg_path), img)

        manifest.video.path = None
        manifest.video.resolution = (0, 0)
        manifest.frame_image = str(jpeg_path)

        feature = FeaturePerimeter(
            label="perimeter_center",
            perimeter_label="center",
            description="Arena screenshot fallback",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)

    def test_perimeter_renders_with_frame(self, manifest: InspectionManifest):
        feature = FeaturePerimeter(
            label="perimeter_center",
            perimeter_label="center",
            description="Center perimeter",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        assert any("center" in p.get_label() or True for p in fig.axes[0].patches)
        assert _circle_linewidths(fig.axes[0]) == [2.0]

    def test_zone_renders_perimeter_and_trajectory(self, manifest: InspectionManifest):
        feature = FeatureZone(
            label="zone_in_center",
            perimeter_label="center",
            result_column="in_center",
            coordinate_labels=["nose"],
            description="In center zone",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        # The zone renderer draws the trajectory as colored line segments.
        ax = fig.axes[0]
        from matplotlib.collections import LineCollection

        segments = [
            s
            for collection in ax.collections
            if isinstance(collection, LineCollection)
            for s in collection.get_segments()
        ]
        # [0,10), [10,20), [20,30): each state run contributes 9 edges.
        assert len(segments) == 27
        coordinates = manifest.get_coordinates("nose") / manifest.settings.meters_per_pixel
        expected = [coordinates[i : i + 2] for i in range(29) if i not in (9, 19)]
        assert {tuple(s.ravel()) for s in segments} == {tuple(s.ravel()) for s in expected}

    def test_quadrant_renders_grid(self, manifest: InspectionManifest):
        feature = FeatureQuadrant(
            label="quadrant_grid",
            quadrant_label="grid",
            coordinate_labels=["nose"],
            description="2x2 grid",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        # 2x2 grid → 4 Rectangle patches
        rect_count = sum(1 for p in fig.axes[0].patches if type(p).__name__ == "Rectangle")
        assert rect_count == 4

    def test_heuristic_without_perimeter_renders_trajectory(self, manifest: InspectionManifest):
        feature = FeatureHeuristic(
            label="heuristic_in_center",
            result_column="in_center",
            coordinate_labels=["nose"],
            description="In center heuristic",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)

    def test_heuristic_cloud_renders_points_and_timeline(self, manifest: InspectionManifest):
        feature = FeatureHeuristicCloud(
            label="heuristic_freezing",
            result_column="freezing",
            coordinate_labels=["nose"],
            description="Freezing cloud",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert len(fig.axes) == 2
        assert _has_imshow(fig)
        ax = fig.axes[0]
        assert len(ax.lines) == 0
        assert not any(isinstance(c, LineCollection) for c in ax.collections)
        assert any(isinstance(c, PathCollection) for c in ax.collections)
        texts = [t.get_text() for t in ax.texts]
        assert any("20%" in t for t in texts)

    def test_heuristic_decomposition_renders_multi_panel(self, manifest: InspectionManifest):
        feature = FeatureHeuristicDecomposition(
            label="heuristic_decomposition_body_proximity",
            heuristic_name="body_proximity",
            result_column="in_center",
            coordinate_labels=["nose"],
            perimeter_label="center",
            components=[
                HeuristicDecompositionComponent(
                    label="component_a",
                    result_column="near_center",
                    coordinate_labels=["nose"],
                    perimeter_label="center",
                ),
                HeuristicDecompositionComponent(
                    label="raw_combined",
                    result_column="freezing",
                    coordinate_labels=["nose"],
                    perimeter_label="center",
                ),
            ],
            show_final_result=True,
            description="Body proximity decomposition",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        assert len(fig.axes) == 3
        assert all(len(ax.texts) >= 1 for ax in fig.axes[:3])

    def test_freezing_decomposition_uses_cloud_panels(self, manifest: InspectionManifest):
        feature = FeatureHeuristicDecomposition(
            label="heuristic_decomposition_freezing",
            heuristic_name="freezing",
            result_column="freezing",
            coordinate_labels=["nose"],
            components=[
                HeuristicDecompositionComponent(
                    label="thresholded_rolling_displacement",
                    result_column="freezing",
                    coordinate_labels=["nose"],
                ),
            ],
            show_final_result=False,
            description="Freezing decomposition",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.lines) == 0
        assert not any(isinstance(c, LineCollection) for c in ax.collections)
        assert any(isinstance(c, PathCollection) for c in ax.collections)

    def test_angle_renders_arrow_at_chosen_frame(self, manifest: InspectionManifest):
        from bekiper_inspect.plot.theme import get_active_theme

        feature = FeatureAngle(
            label="angle_nose_tail_base",
            from_coordinate_label="nose",
            to_coordinate_label="tail_base",
            description="Body axis",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        from matplotlib.quiver import Quiver

        assert any(isinstance(coll, Quiver) for coll in fig.axes[0].collections)
        label_text = next(text for text in fig.axes[0].texts if "°" in text.get_text())
        assert label_text.get_text().endswith("°")
        assert label_text.get_fontsize() == max(get_active_theme().annotation_fontsize - 6, 4.5)

    def test_angle_linewidth_scales_quiver_geometry(self, manifest: InspectionManifest):
        from matplotlib.quiver import Quiver

        default_feature = FeatureAngle(
            label="angle_nose_tail_base_default",
            from_coordinate_label="nose",
            to_coordinate_label="tail_base",
            description="Body axis default",
        )
        wide_feature = FeatureAngle(
            label="angle_nose_tail_base_wide",
            from_coordinate_label="nose",
            to_coordinate_label="tail_base",
            description="Body axis wide",
            linewidth=6.0,
        )
        default_fig = plot_feature_on_frame(manifest, default_feature)
        wide_fig = plot_feature_on_frame(manifest, wide_feature)
        assert default_fig is not None
        assert wide_fig is not None
        default_quiver = next(coll for coll in default_fig.axes[0].collections if isinstance(coll, Quiver))
        wide_quiver = next(coll for coll in wide_fig.axes[0].collections if isinstance(coll, Quiver))
        assert wide_quiver.width > default_quiver.width
        assert wide_quiver.get_linewidths()[0] > default_quiver.get_linewidths()[0]

    def test_sparse_angle_heuristic_uses_footer_summary(
        self,
    ):
        n = 30
        manifest = _make_angle_manifest(
            n=n,
            coords_x=np.linspace(0.05, 0.95, n),
            coords_y=np.full(n, 0.15),
            angles_deg=np.full(n, 35.0),
            resolution=(1200, 400),
        )
        feature = FeatureHeuristic(
            label="heuristic_olfaction_object_1",
            result_column="olfaction_object_1",
            coordinate_labels=["nose"],
            perimeter_label="object_1",
            description="Sparse olfaction",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        assert len(fig.axes) == 2
        main_ax = next(ax for ax in fig.axes if ax.images)
        footer_ax = next(ax for ax in fig.axes if ax is not main_ax and any("rep:" in t.get_text() for t in ax.texts))
        main_texts = [t.get_text() for t in main_ax.texts]
        footer_texts = [t.get_text() for t in footer_ax.texts]
        assert not any("rep:" in t or "median:" in t for t in main_texts)
        assert any("rep:" in t for t in footer_texts)
        assert any("median:" in t for t in footer_texts)
        assert any("%" in t for t in footer_texts)

    def test_angle_heuristic_uses_boundary_target_diagnostics_not_perimeter_center(
        self,
    ):
        from matplotlib.quiver import Quiver

        n = 30
        manifest = _make_angle_manifest(
            n=n,
            coords_x=np.full(n, 0.10),
            coords_y=np.full(n, 0.10),
            angles_deg=np.full(n, 90.0),
            resolution=(800, 400),
        )
        manifest.perimeters[0] = PerimeterSpec(
            label="object_1",
            shape="circle",
            params={"center_x": 0.40, "center_y": 0.20, "radius": 0.02},
        )
        manifest.df = manifest.df.with_columns(
            [
                pl.Series(
                    "__bk_diag__olfaction_object_1__ray_min_angle_deg",
                    np.full(n, 90.0),
                ),
                pl.Series(
                    "__bk_diag__olfaction_object_1__ray_angle_margin_deg",
                    np.full(n, -10.0),
                ),
                pl.Series(
                    "__bk_diag__olfaction_object_1__ray_target_x",
                    np.full(n, 0.40),
                ),
                pl.Series(
                    "__bk_diag__olfaction_object_1__ray_target_y",
                    np.full(n, 0.10),
                ),
            ]
        )
        feature = FeatureHeuristic(
            label="heuristic_olfaction_object_1",
            result_column="olfaction_object_1",
            coordinate_labels=["nose"],
            perimeter_label="object_1",
            description="Boundary-target olfaction",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        main_ax = next(ax for ax in fig.axes if ax.images)
        quivers = [coll for coll in main_ax.collections if isinstance(coll, Quiver)]
        assert len(quivers) == 2
        compare_quiver = quivers[1]
        compare_angle = float(
            np.degrees(
                np.arctan2(
                    np.asarray(compare_quiver.V, dtype=float)[0],
                    np.asarray(compare_quiver.U, dtype=float)[0],
                )
            )
        )
        center_angle = float(np.degrees(np.arctan2(0.20 - 0.10, 0.40 - 0.10)))
        assert abs(compare_angle) < 5.0
        assert abs(compare_angle - center_angle) > 10.0

    def test_angle_heuristic_fallback_uses_boundary_target_for_legacy_manifest(
        self,
    ):
        from matplotlib.quiver import Quiver

        n = 24
        manifest = _make_angle_manifest(
            n=n,
            coords_x=np.full(n, 0.10),
            coords_y=np.full(n, 0.10),
            angles_deg=np.full(n, 0.0),
            resolution=(800, 400),
        )
        manifest.perimeters[0] = PerimeterSpec(
            label="object_1",
            shape="rectangle",
            params={"center_x": 0.40, "center_y": 0.20, "width": 0.10, "height": 0.20},
        )
        feature = FeatureHeuristic(
            label="heuristic_olfaction_object_1",
            result_column="olfaction_object_1",
            coordinate_labels=["nose"],
            perimeter_label="object_1",
            description="Legacy boundary-target olfaction",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        main_ax = next(ax for ax in fig.axes if ax.images)
        quivers = [coll for coll in main_ax.collections if isinstance(coll, Quiver)]
        assert len(quivers) == 2
        compare_quiver = quivers[1]
        compare_angle = float(
            np.degrees(
                np.arctan2(
                    np.asarray(compare_quiver.V, dtype=float)[0],
                    np.asarray(compare_quiver.U, dtype=float)[0],
                )
            )
        )
        center_angle = float(np.degrees(np.arctan2(0.20 - 0.10, 0.40 - 0.10)))
        assert abs(compare_angle) < 5.0
        assert abs(compare_angle - center_angle) > 10.0

    def test_whisker_overlay_prefers_selected_ray_diagnostics(
        self,
    ):
        n = 12
        manifest = _make_whisker_manifest(n=n)
        manifest.df = manifest.df.with_columns(
            [
                pl.Series("__bk_diag__whisker_obj__selected_ray_min_angle_deg", np.full(n, 70.0)),
                pl.Series("__bk_diag__whisker_obj__selected_ray_angle_margin_deg", np.full(n, -50.0)),
                pl.Series("__bk_diag__whisker_obj__selected_ray_target_x", np.full(n, 0.10)),
                pl.Series("__bk_diag__whisker_obj__selected_ray_target_y", np.full(n, 0.40)),
                pl.Series("__bk_diag__whisker_obj__selected_ray_dx", np.full(n, 0.0)),
                pl.Series("__bk_diag__whisker_obj__selected_ray_dy", np.full(n, 1.0)),
                pl.Series("__bk_diag__whisker_obj__left_ray_min_angle_deg", np.full(n, 2.0)),
                pl.Series("__bk_diag__whisker_obj__left_ray_angle_margin_deg", np.full(n, 18.0)),
                pl.Series("__bk_diag__whisker_obj__left_ray_target_x", np.full(n, 0.40)),
                pl.Series("__bk_diag__whisker_obj__left_ray_target_y", np.full(n, 0.10)),
                pl.Series("__bk_diag__whisker_obj__right_ray_min_angle_deg", np.full(n, 15.0)),
                pl.Series("__bk_diag__whisker_obj__right_ray_angle_margin_deg", np.full(n, 5.0)),
                pl.Series("__bk_diag__whisker_obj__right_ray_target_x", np.full(n, 0.20)),
                pl.Series("__bk_diag__whisker_obj__right_ray_target_y", np.full(n, 0.30)),
            ]
        )
        feature = FeatureHeuristic(
            label="heuristic_whisker_obj",
            result_column="whisker_obj",
            coordinate_labels=["center_ear"],
            perimeter_label="object_1",
            description="Whisker selected ray",
        )
        overlay = build_heuristic_angle_overlay(
            manifest,
            feature,
            frame_idx=n // 2,
            frame_shape=(480, 640, 3),
            theme=PlotTheme(),
            heuristic_name="whisker_obj",
        )
        assert overlay is not None
        sample = overlay.representative_sample
        assert sample.comparison_vector is not None
        compare_angle = float(np.degrees(np.arctan2(sample.comparison_vector[1], sample.comparison_vector[0])))
        assert abs(compare_angle - 90.0) < 5.0

    def test_rejected_angle_panel_draws_vectors_and_miss_summary(
        self,
    ):
        from matplotlib.quiver import Quiver

        n = 40
        true_mask = np.zeros(n, dtype=bool)
        manifest = _make_angle_manifest(
            n=n,
            coords_x=np.full(n, 0.10),
            coords_y=np.full(n, 0.10),
            angles_deg=np.full(n, 0.0),
            true_mask=true_mask,
            resolution=(640, 480),
        )
        miss_angle = float(np.degrees(np.arctan2(0.10, 0.30)))
        manifest.df = manifest.df.with_columns(
            [
                pl.Series(
                    "__bk_diag__olfaction_object_1__ray_min_angle_deg",
                    np.full(n, miss_angle),
                ),
                pl.Series(
                    "__bk_diag__olfaction_object_1__ray_angle_margin_deg",
                    np.full(n, 5.0 - miss_angle),
                ),
                pl.Series(
                    "__bk_diag__olfaction_object_1__ray_target_x",
                    np.full(n, 0.40),
                ),
                pl.Series(
                    "__bk_diag__olfaction_object_1__ray_target_y",
                    np.full(n, 0.20),
                ),
            ]
        )
        feature = FeatureHeuristic(
            label="heuristic_olfaction_object_1",
            result_column="olfaction_object_1",
            coordinate_labels=["nose"],
            perimeter_label="object_1",
            description="Rejected olfaction",
        )
        _fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(np.full((480, 640, 3), 40, dtype=np.uint8), aspect="equal")
        _plot_heuristic_feature(
            ax,
            manifest,
            feature,
            frame_shape=(480, 640, 3),
            frame_idx=n // 2,
            valid_only=False,
        )
        assert any(isinstance(coll, Quiver) for coll in ax.collections)
        texts = [text.get_text() for text in ax.texts]
        assert any("median miss:" in text for text in texts)
        assert any("%" in text for text in texts)
