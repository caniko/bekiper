from ._make_angle_manifest import _make_angle_manifest
from .test_fixture_video import *


class TestPlotFeatureOnFrameMore:
    def test_dense_angle_heuristic_adds_sidecar_and_raw_window_summary(
        self,
    ):
        from matplotlib.quiver import Quiver

        n = 240
        angles = np.full(n, 60.0)
        angles[119] = 3.0
        coords_x = 0.10 + (np.arange(n) % 25) * 0.03
        coords_y = 0.10 + ((np.arange(n) // 5) % 5) * 0.03
        manifest = _make_angle_manifest(
            n=n,
            coords_x=coords_x,
            coords_y=coords_y,
            angles_deg=angles,
            resolution=(640, 480),
        )
        feature = FeatureHeuristic(
            label="heuristic_olfaction_object_1",
            result_column="olfaction_object_1",
            coordinate_labels=["nose"],
            perimeter_label="object_1",
            description="Dense olfaction",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        assert len(fig.axes) == 3
        main_ax = next(ax for ax in fig.axes if ax.get_title() == feature.label)
        sidecar_ax = next(ax for ax in fig.axes if ax.get_title() == "Median angle (°)")
        footer_ax = next(
            ax
            for ax in fig.axes
            if ax not in (main_ax, sidecar_ax) and any("rep:" in text.get_text() for text in ax.texts)
        )
        quivers = [coll for coll in main_ax.collections if isinstance(coll, Quiver)]
        assert quivers
        assert len(quivers) == 2
        assert all(getattr(coll, "units", None) != "xy" for coll in quivers)
        unique_positions = _unique_position_count(_quiver_positions(quivers[0]))
        assert 10 < unique_positions <= 36
        assert np.all((_quiver_lengths(quivers[0]) > 20.0) & (_quiver_lengths(quivers[0]) < 30.0))
        assert np.all((_quiver_lengths(quivers[1]) > 20.0) & (_quiver_lengths(quivers[1]) < 30.0))
        assert not np.allclose(_quiver_rgba(quivers[0]), _quiver_rgba(quivers[1]))
        assert all(
            not isinstance(coll, PathCollection) or np.asarray(coll.get_offsets(), dtype=float).shape[0] <= 1
            for coll in main_ax.collections
        )
        pair_texts = _pair_angle_texts(main_ax)
        assert len(pair_texts) == unique_positions
        _assert_no_angle_label_overlap(fig, main_ax)
        assert not _long_axis_lines(main_ax)
        assert not _has_representative_angle_marker(main_ax)
        circle_lws = _circle_linewidths(main_ax)
        assert circle_lws
        assert all(lw < 2.0 for lw in circle_lws)
        main_texts = [t.get_text() for t in main_ax.texts]
        footer_texts = [t.get_text() for t in footer_ax.texts]
        assert not any("rep:" in t or "median:" in t or "%" in t for t in main_texts)
        assert any("rep:" in t and "°" in t for t in footer_texts)
        assert any("median:" in t and "°" in t for t in footer_texts)
        assert any("%" in t for t in footer_texts)
        assert len(sidecar_ax.images) >= 1
        sidecar_quivers = [coll for coll in sidecar_ax.collections if isinstance(coll, Quiver)]
        assert len(sidecar_quivers) == 2
        sidecar_unique_positions = _unique_position_count(_quiver_positions(sidecar_quivers[0]))
        assert len(_pair_angle_texts(sidecar_ax)) == sidecar_unique_positions
        assert all(
            not isinstance(coll, PathCollection) or np.asarray(coll.get_offsets(), dtype=float).shape[0] <= 1
            for coll in sidecar_ax.collections
        )
        _assert_no_angle_label_overlap(fig, sidecar_ax)
        assert not _long_axis_lines(sidecar_ax)
        assert not _has_representative_angle_marker(sidecar_ax)
        sidecar_circle_lws = _circle_linewidths(sidecar_ax)
        assert sidecar_circle_lws
        assert all(lw < 2.0 for lw in sidecar_circle_lws)

    def test_angle_pair_label_fallback_uses_short_neutral_tick(self):
        _fig, ax = plt.subplots(figsize=(1.2, 1.2))
        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(100.0, 0.0)
        ax.set_xticks([])
        ax.set_yticks([])

        samples = tuple(
            AngleOverlaySample(
                frame_index=index,
                origin_xy=(50.0, 50.0),
                primary_vector=(24.0, 0.0),
                comparison_vector=(0.0, -24.0),
                rendered_angle_deg=90.0 + index * 0.1,
                display_color="#00ff00",
                display_origin_xy=(50.0, 50.0),
            )
            for index in range(18)
        )

        laid_out = draw_angle_pair_labels(
            ax,
            samples,
            theme=PlotTheme(),
        )

        assert len(_pair_angle_texts(ax)) == len(samples)
        assert any(sample.show_association_tick for sample in laid_out)
        line_lengths = _axis_line_display_lengths(ax)
        assert line_lengths
        assert max(line_lengths) <= 12.0
        assert all(_is_neutral_gray(_line_rgba(line)) for line in ax.lines)
        assert not _long_axis_lines(ax)

    def test_heatmap_with_clustered_samples_keeps_labels_inside_axes(self):
        """Regression: a tight cluster of samples used to produce a heatmap
        whose extent was only a few data units wide. Vectors are
        ANGLE_HEURISTIC_VECTOR_PX (24) data units long, so quiver tips and
        the angle pair labels drawn around them landed far outside the axes.
        Combined with bbox_inches="tight" at save time, this ballooned the
        saved figure into vast whitespace (see the
        story_object_decomposition_whisker_interaction_familiar artifact).
        """
        from bekiper_inspect.plot.feature._angle_parts._best_angle_diag_for_frame import (
            _build_angle_heatmap,
        )
        from bekiper_inspect.plot.feature._angle_parts._render_angle_heatmap import (
            _render_angle_heatmap,
        )

        rng = np.random.default_rng(0)
        samples = tuple(
            AngleOverlaySample(
                frame_index=i,
                origin_xy=(
                    100.0 + float(rng.uniform(-3.0, 3.0)),
                    100.0 + float(rng.uniform(-3.0, 3.0)),
                ),
                primary_vector=(20.0, 0.0),
                comparison_vector=(15.0, 12.0),
                rendered_angle_deg=30.0 + i * 2.0,
                display_color="#00ff00",
            )
            for i in range(8)
        )
        heatmap = _build_angle_heatmap(samples, pad_px=4.0)
        assert heatmap is not None

        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        try:
            _render_angle_heatmap(
                ax,
                heatmap,
                theme=PlotTheme(),
                title="Median angle (°)",
                overlay_samples=samples,
            )
            fig.canvas.draw()

            renderer = fig.canvas.get_renderer()
            axes_bbox = ax.get_window_extent(renderer)
            for text in _pair_angle_texts(ax):
                bb = text.get_window_extent(renderer=renderer)
                bbox_area = max(bb.width, 1.0) * max(bb.height, 1.0)
                inside = type(bb).intersection(bb, axes_bbox)
                inside_area = max(inside.width, 0.0) * max(inside.height, 0.0) if inside is not None else 0.0
                outside_area = max(bbox_area - inside_area, 0.0)
                assert outside_area <= bbox_area * 0.5, (
                    f"angle label {text.get_text()!r} sits mostly outside the heatmap axes "
                    f"(outside={outside_area:.1f}px² of {bbox_area:.1f}px²)"
                )

            # The saved figure with bbox_inches='tight' must not balloon
            # because of escaping artists. Allow a small margin for the
            # title and axes spine; reject anything > 1.5x nominal.
            import io

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            from PIL import Image

            with Image.open(buf) as im:
                width_px, height_px = im.size
            assert width_px <= 300, f"saved width {width_px}px exceeds nominal 200px"
            assert height_px <= 300, f"saved height {height_px}px exceeds nominal 200px"
        finally:
            plt.close(fig)

    def test_is_angle_heuristic_pins_known_classifications(self):
        """The classifier currently uses substring matching on three tags
        (olfaction / whisker / exploring). Adding a new angle-kind on the
        Rust side requires extending that tuple. This test pins the
        current behavior so a regression caused by retiring a tag (or
        renaming an existing heuristic) is caught immediately.
        """
        from bekiper_inspect.plot.feature._renderers_parts._plot_ymaze_arm_sections_feature import (
            _ANGLE_HEURISTIC_NAME_TAGS,
            _is_angle_heuristic,
        )

        # Sanity: the tag set hasn't been silently expanded.
        assert _ANGLE_HEURISTIC_NAME_TAGS == ("olfaction", "whisker", "exploring")

        def _h(result_column: str) -> FeatureHeuristic:
            return FeatureHeuristic(
                label=f"heuristic_{result_column}",
                result_column=result_column,
                coordinate_labels=["nose"],
                description="x",
            )

        # Positive cases — production heuristic names that must stay
        # classified as angle-kind.
        for name in (
            "olfaction_object_1",
            "olfaction_familiar",
            "whisker_interaction_familiar",
            "exploring_arena_raw",
            "exploring_object_1",
        ):
            assert _is_angle_heuristic(_h(name)), f"{name!r} should be angle-kind"

        # Negative cases — must NOT be misclassified as angle-kind.
        for name in (
            "in_center",
            "in_periphery",
            "freezing",
            "body_proximity_object_1",
            "body_proximity_familiar",
        ):
            assert not _is_angle_heuristic(_h(name)), f"{name!r} should not be angle-kind"

        # Diagnostic component column for an angle-kind decomposition
        # parent: classified via `heuristic_name=` even when the column
        # itself doesn't contain a tag substring.
        proximity_component = FeatureHeuristic(
            label="heuristic_whisker_interaction_familiar_left_ear_near",
            result_column="__bk_diag__whisker_interaction_familiar__left_ear_near",
            coordinate_labels=["left_ear"],
            description="x",
        )
        assert _is_angle_heuristic(
            proximity_component,
            heuristic_name="whisker_interaction_familiar",
        )

    def test_angle_pair_label_raises_when_axes_too_small_for_overlay(self):
        """Regression: the layout fallback used to silently return a label
        position even when every candidate sat outside the axes. Now the
        renderer raises so the bug is caught immediately rather than at
        save time, when bbox_inches='tight' would balloon the figure.
        """
        # Set xlim/ylim *tighter* than the vectors extending from origin —
        # there is no way for the labels to fit.
        fig, ax = plt.subplots(figsize=(1.0, 1.0))
        ax.set_xlim(99.0, 101.0)
        ax.set_ylim(101.0, 99.0)
        ax.set_xticks([])
        ax.set_yticks([])

        sample = AngleOverlaySample(
            frame_index=0,
            origin_xy=(100.0, 100.0),
            primary_vector=(50.0, 0.0),
            comparison_vector=(0.0, 50.0),
            rendered_angle_deg=45.0,
            display_color="#00ff00",
        )

        try:
            with pytest.raises(RuntimeError, match="angle pair label could not be placed"):
                draw_angle_pair_labels(ax, (sample,), theme=PlotTheme())
        finally:
            plt.close(fig)

    def test_decomposition_panels_have_uniform_width_when_only_some_have_heatmap(
        self,
    ):
        """Regression: panels in a decomposition figure that didn't produce
        angle data used to take the full slot width while siblings with
        heatmap data split 3:2, leaving sibling panels in the same row
        wildly different sizes (visible in the original whisker bug). The
        renderer now pre-scans panels and uses the same sub-slot geometry
        for all of them.
        """
        n = 240
        coords_x = 0.10 + (np.arange(n) % 25) * 0.03
        coords_y = 0.10 + ((np.arange(n) // 5) % 5) * 0.03
        manifest = _make_angle_manifest(
            n=n,
            coords_x=coords_x,
            coords_y=coords_y,
            angles_deg=np.full(n, 60.0),
            resolution=(640, 480),
        )
        # First component matches the boolean column we populated, so it
        # produces angle samples and a heatmap. Second component points at
        # a column that's not in the dataframe — angle_data resolves to
        # None and the per-panel branch used to skip the sidecar split for
        # this panel only.
        feature = FeatureHeuristicDecomposition(
            label="heuristic_decomposition_olfaction_object_1",
            heuristic_name="olfaction_object_1",
            result_column="olfaction_object_1",
            coordinate_labels=["nose"],
            perimeter_label="object_1",
            components=[
                HeuristicDecompositionComponent(
                    label="ray_hits_object",
                    result_column="olfaction_object_1",
                    coordinate_labels=["nose"],
                    perimeter_label="object_1",
                ),
                HeuristicDecompositionComponent(
                    label="standalone_diag",
                    result_column="olfaction_object_1__missing_diag",
                    coordinate_labels=["nose"],
                    perimeter_label="object_1",
                ),
            ],
            show_final_result=True,
            description="mixed-heatmap decomposition",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        try:
            titles = {ax.get_title(): ax for ax in fig.axes if ax.get_title()}
            main_titles = ["ray_hits_object", "standalone_diag", "final"]
            mains = [titles[t] for t in main_titles if t in titles]
            assert len(mains) == 3, f"expected 3 main panels, got {sorted(titles)}"
            # Compare the gridspec sub-slot bounds (what the renderer
            # controls). Post-constrained-layout `ax.get_position()`
            # widths drift by ~1% across siblings because aspect="equal"
            # interacts differently with per-axes overlay content. The
            # bug being regressed against is much coarser: a panel
            # without heatmap used to take the full slot (~5/3× the
            # width of siblings with a sidecar split).
            slot_bboxes = [ax.get_subplotspec().get_position(fig) for ax in mains]
            slot_widths = [bb.width for bb in slot_bboxes]
            assert max(slot_widths) - min(slot_widths) < 1e-6, (
                f"main panel sub-slots differ across siblings: {dict(zip(main_titles, slot_widths))}"
            )
        finally:
            plt.close(fig)

    def test_angle_decomposition_falls_back_from_expanded_perimeter_label(
        self,
    ):
        from matplotlib.quiver import Quiver

        manifest = _make_angle_manifest(
            n=120,
            coords_x=np.linspace(0.05, 0.35, 120),
            coords_y=np.full(120, 0.15),
            angles_deg=np.linspace(15.0, 75.0, 120),
        )
        feature = FeatureHeuristicDecomposition(
            label="heuristic_decomposition_olfaction_object_1",
            heuristic_name="olfaction_object_1",
            result_column="olfaction_object_1",
            coordinate_labels=["nose"],
            perimeter_label="object_1_expanded_-0.01",
            components=[
                HeuristicDecompositionComponent(
                    label="nose_near",
                    result_column="olfaction_object_1",
                    coordinate_labels=["nose"],
                    perimeter_label="object_1_expanded_-0.01",
                )
            ],
            show_final_result=True,
            description="Olfaction decomposition",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        ax = next(ax for ax in fig.axes if ax.get_title() == "nose_near")
        assert any(isinstance(coll, Quiver) for coll in ax.collections)
        texts = [text.get_text() for text in ax.texts]
        assert any("°" in text for text in texts)
        assert not any("median:" in text for text in texts)
        assert any(
            "median:" in text.get_text() for other_ax in fig.axes if other_ax is not ax for text in other_ax.texts
        )

    def test_angle_decomposition_falls_back_to_generic_when_target_missing(
        self,
    ):
        from matplotlib.quiver import Quiver

        manifest = _make_angle_manifest(
            n=60,
            coords_x=np.linspace(0.05, 0.35, 60),
            coords_y=np.full(60, 0.15),
            angles_deg=np.linspace(10.0, 45.0, 60),
        )
        feature = FeatureHeuristicDecomposition(
            label="heuristic_decomposition_olfaction_missing",
            heuristic_name="olfaction_missing",
            result_column="olfaction_object_1",
            coordinate_labels=["nose"],
            perimeter_label="missing_expanded_-0.01",
            components=[
                HeuristicDecompositionComponent(
                    label="ray_hits_object",
                    result_column="olfaction_object_1",
                    coordinate_labels=["nose"],
                    perimeter_label="missing_expanded_-0.01",
                )
            ],
            show_final_result=False,
            description="Missing target decomposition",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        ax = next(ax for ax in fig.axes if ax.get_title() == "ray_hits_object")
        assert not any(isinstance(coll, Quiver) for coll in ax.collections)
        assert any(isinstance(coll, LineCollection) for coll in ax.collections)
        texts = [text.get_text() for text in ax.texts]
        assert any("%" in text for text in texts)

    def test_example_manifest_omits_context_target_decompositions(self):
        example_dir = Path(__file__).resolve().parents[3] / "examples/neo/dejana/output/1035/seq0_oultraining"
        if not example_dir.is_dir():
            pytest.skip("example inspection output is not present")

        manifest = InspectionManifest.from_directory(example_dir)
        heuristic_names = {heuristic.name for heuristic in manifest.heuristics}
        feature_labels = {getattr(feature, "label", None) for feature in manifest.feature_inspections}

        assert "exploring_arena" not in heuristic_names
        assert "exploring_center_zone" not in heuristic_names
        assert "body_proximity_arena" not in heuristic_names
        assert "olfaction_center_zone" not in heuristic_names
        assert "whisker_interaction_arena" not in heuristic_names

        assert "heuristic_decomposition_exploring_arena" not in feature_labels
        assert "heuristic_decomposition_exploring_center_zone" not in feature_labels
        assert "heuristic_decomposition_body_proximity_arena" not in feature_labels
        assert "heuristic_decomposition_olfaction_center_zone" not in feature_labels
        assert "heuristic_decomposition_whisker_interaction_arena" not in feature_labels

    def test_example_olfaction_object_decomposition_keeps_object_context(self):
        from matplotlib.patches import Circle
        from matplotlib.quiver import Quiver

        example_dir = Path(__file__).resolve().parents[3] / "examples/neo/dejana/output/1035/seq0_oultraining"
        if not example_dir.is_dir():
            pytest.skip("example inspection output is not present")

        manifest = InspectionManifest.from_directory(example_dir)
        feature = next(
            f
            for f in manifest.feature_inspections
            if getattr(f, "label", None) == "heuristic_decomposition_olfaction_object_1"
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        nose_near_ax = next(ax for ax in fig.axes if ax.get_title() == "nose_near")
        quivers = [coll for coll in nose_near_ax.collections if isinstance(coll, Quiver)]
        assert quivers
        assert all(getattr(coll, "units", None) != "xy" for coll in quivers)
        assert any(isinstance(patch, Circle) for patch in nose_near_ax.patches)
        circle_lws = _circle_linewidths(nose_near_ax)
        assert circle_lws
        assert all(lw < 2.0 for lw in circle_lws)
        assert not _has_representative_angle_marker(nose_near_ax)

    def test_trajectory_renders_line(self, manifest: InspectionManifest):
        feature = FeatureTrajectory(
            label="trajectory_nose",
            coordinate_labels=["nose"],
            description="Nose trajectory",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        # plot_coordinates adds a Line2D
        assert len(fig.axes[0].lines) >= 1

    def test_skeleton_trajectory_renders_zoomed_pose(self, manifest: InspectionManifest):
        n = len(manifest.df)
        df = manifest.df.with_columns(
            [
                pl.Series("left_ear_x", np.full(n, 0.135)),
                pl.Series("left_ear_y", np.full(n, 0.140)),
                pl.Series("right_ear_x", np.full(n, 0.165)),
                pl.Series("right_ear_y", np.full(n, 0.140)),
                pl.Series("center_ear_x", np.full(n, 0.150)),
                pl.Series("center_ear_y", np.full(n, 0.145)),
            ]
        )
        manifest.df = df
        manifest.labels = ["nose", "left_ear", "right_ear", "center_ear", "tail_base"]
        manifest.coordinate_columns.update(
            {
                "left_ear": CoordinateColumnSpec(x="left_ear_x", y="left_ear_y"),
                "right_ear": CoordinateColumnSpec(x="right_ear_x", y="right_ear_y"),
                "center_ear": CoordinateColumnSpec(x="center_ear_x", y="center_ear_y"),
            }
        )
        feature = FeatureTrajectory(
            label="skeleton_pose",
            coordinate_labels=["nose", "left_ear", "center_ear", "right_ear", "tail_base"],
            description="Skeleton pose",
            representative_frame_index=12,
        )

        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        assert len(fig.axes[0].lines) >= 3
        assert fig.axes[0].collections

    def test_raycast_renders_wedge_on_frame(self, manifest: InspectionManifest):
        feature = FeatureRayCast(
            label="raycast_nose",
            origin_label="nose",
            angle_column="heading_rad",
            length_meters=0.05,
            cone_width_deg=60.0,
            description="Olfactory cone",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        # A Wedge patch must be present on the axes.
        from matplotlib.patches import Wedge

        assert any(isinstance(p, Wedge) for p in fig.axes[0].patches)

    def test_detection_fields_draw_cone_and_both_fans_and_pick_whisker_frame(self, manifest: InspectionManifest):
        import polars as pl
        from matplotlib.patches import Polygon

        from bekiper_inspect.manifest import FeatureDetectionFields
        from bekiper_inspect.plot.feature._helpers import _pick_frame_for_feature

        n = len(manifest.df)
        nose_hit = np.zeros(n, dtype=bool)
        whisker_hit = np.zeros(n, dtype=bool)
        nose_hit[10:15] = True  # nose-only bout
        whisker_hit[20:23] = True  # whisker-only bout
        manifest.df = manifest.df.with_columns(
            pl.Series("near", nose_hit),
            pl.Series("gate", np.ones(n, dtype=bool)),
            pl.Series("fan", whisker_hit),
        )
        feature = FeatureDetectionFields(
            label="detection_fields",
            nose_label="nose",
            head_label="tail_base",
            cone_half_angle_deg=45.0,
            nose_reach_m=0.03,
            contact_distance_m=0.005,
            description="Detection fields",
            whisker_apex_fraction=0.8,
            whisker_half_angle_deg=30.0,
            whisker_length_m=0.025,
            object_labels=["arena"],
            nose_near_columns=["near"],
            nose_gate_columns=["gate"],
            whisker_hit_columns=["fan"],
        )
        assert _pick_frame_for_feature(manifest, feature) == 21
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        labels = [p.get_label() for p in fig.axes[0].patches if isinstance(p, Polygon)]
        assert {"nose cone", "contact", "left whisker fan", "right whisker fan"} <= set(labels)
        # Nose-only run (no fan parameters) still draws the cone.
        bare = FeatureDetectionFields(
            label="detection_fields",
            nose_label="nose",
            head_label="tail_base",
            cone_half_angle_deg=45.0,
            nose_reach_m=0.03,
            contact_distance_m=0.005,
            description="Detection fields",
        )
        fig = plot_feature_on_frame(manifest, bare)
        assert "nose cone" in [p.get_label() for p in fig.axes[0].patches]

    def test_scale_derivation_renders_with_frame(self, manifest: InspectionManifest):
        manifest.scale_derivation = ScaleDerivation(
            method="arena_side",
            physical_meters=0.4,
            pixel_measurement=400.0,
            arm_pixel_spans=[],
            reference_label="center",
            meters_per_pixel=0.001,
        )
        feature = FeatureScaleDerivation(
            label="scale_derivation",
            description="How meters-per-pixel was derived",
            representative_frame_index=0,
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)

    def test_timeseries_renders_dual_panel(self, manifest: InspectionManifest):
        # Add a speed column to the manifest's DataFrame.
        n = len(manifest.df)
        manifest.df = manifest.df.with_columns(pl.Series("speed_mps", np.random.uniform(0, 0.5, n)))
        feature = FeatureTimeSeries(
            label="timeseries_speed_mps",
            value_column="speed_mps",
            unit="m/s",
            description="Speed over time",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        # TimeSeries creates a dual-panel figure (two axes).
        assert len(fig.axes) == 2

    def test_proximity_renders_perimeter_and_buffer_ring(self, manifest: InspectionManifest):
        feature = FeatureProximity(
            label="proximity_center",
            perimeter_label="center",
            buffer_meters=0.02,
            result_column="near_center",
            coordinate_labels=["nose"],
            description="Proximity to center",
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        # The renderer adds at least two patches (perimeter + buffer ring).
        assert len(fig.axes[0].patches) >= 2

    def test_nort_discrimination_renders_objects(self, manifest: InspectionManifest):
        # Add a second perimeter for a "novel" object.
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
            description="NORT discrimination",
            discrimination_index=0.25,
            recognition_index=0.625,
        )
        fig = plot_feature_on_frame(manifest, feature)
        assert fig is not None
        assert _has_imshow(fig)
        # Should have text annotations for DI/RI.
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("DI" in t for t in texts)
