from ._quiver_positions import *
from bekiper_inspect.manifest import (
    FeatureAngle,
    FeatureHeuristic,
    FeatureHeuristicCloud,
    FeatureHeuristicDecomposition,
    FeatureNortDiscrimination,
    FeaturePerimeter,
    HeuristicDecompositionComponent,
)
from bekiper_inspect.plot.feature._grouping import StoryGroup
from bekiper_inspect.plot.feature._story import plot_story


class TestPlotStory:
    def test_geometry_story(self, manifest: InspectionManifest):
        group = StoryGroup(
            key="geometry",
            title="Arena Geometry",
            features=[
                FeaturePerimeter(label="perimeter_object_1", perimeter_label="object_1", description="..."),
                FeaturePerimeter(label="perimeter_arena", perimeter_label="arena", description="..."),
                FeatureAngle(
                    label="angle_nose_tail",
                    from_coordinate_label="nose",
                    to_coordinate_label="tail_base",
                    description="...",
                ),
            ],
        )
        fig = plot_story(manifest, group)
        assert isinstance(fig, Figure)
        # Geometry story is single-panel.
        assert len(fig.axes) == 1

    def test_heuristic_story_multi_panel(self, manifest: InspectionManifest):
        group = StoryGroup(
            key="object_1",
            title="Object 1",
            features=[
                FeatureHeuristic(
                    label="heuristic_body_proximity_object_1",
                    result_column="body_proximity_object_1",
                    coordinate_labels=["nose"],
                    perimeter_label="object_1",
                    description="...",
                ),
                FeatureHeuristic(
                    label="heuristic_olfaction_object_1",
                    result_column="olfaction_object_1",
                    coordinate_labels=["nose"],
                    perimeter_label="object_1",
                    description="...",
                ),
            ],
            heuristic_names=["body_proximity_object_1", "olfaction_object_1"],
            perimeter_label="object_1",
        )
        fig = plot_story(manifest, group)
        assert isinstance(fig, Figure)
        # This fixture lacks snout_dx/snout_dy, so the olfaction panel falls
        # back to the generic heuristic renderer with no angle footer.
        assert len(fig.axes) == 5

    def test_standalone_heuristic_story(self, manifest: InspectionManifest):
        group = StoryGroup(
            key="freezing",
            title="Freezing",
            features=[
                FeatureHeuristicCloud(
                    label="heuristic_freezing",
                    result_column="freezing",
                    coordinate_labels=["nose"],
                    description="...",
                ),
            ],
            heuristic_names=["freezing"],
        )
        fig = plot_story(manifest, group)
        assert isinstance(fig, Figure)
        # 1 cloud panel + 1 timeline = 2 axes.
        assert len(fig.axes) == 2

    def test_dense_angle_story_adds_valid_sidecar(self):
        from matplotlib.collections import PathCollection
        from matplotlib.quiver import Quiver

        manifest = _make_dense_angle_manifest()
        group = StoryGroup(
            key="object_1",
            title="Object 1",
            features=[
                FeatureHeuristic(
                    label="heuristic_olfaction_object_1",
                    result_column="olfaction_object_1",
                    coordinate_labels=["nose"],
                    perimeter_label="object_1",
                    description="...",
                ),
            ],
            heuristic_names=["olfaction_object_1"],
            perimeter_label="object_1",
        )
        fig = plot_story(manifest, group)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 5
        assert any(ax.get_title() == "Median angle (°)" for ax in fig.axes)
        assert any(ax.get_title() == "rejected" for ax in fig.axes)
        valid_ax = next(ax for ax in fig.axes if ax.get_title() == "Olfaction Object 1")
        footer_ax = next(ax for ax in fig.axes if any("rep:" in text.get_text() for text in ax.texts))
        quivers = [coll for coll in valid_ax.collections if isinstance(coll, Quiver)]
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
            for coll in valid_ax.collections
        )
        assert len(_pair_angle_texts(valid_ax)) == unique_positions
        _assert_no_angle_label_overlap(fig, valid_ax)
        assert not _long_axis_lines(valid_ax)
        assert not _has_representative_angle_marker(valid_ax)
        circle_lws = _circle_linewidths(valid_ax)
        assert circle_lws
        assert all(lw < 2.0 for lw in circle_lws)
        assert not any("median:" in text.get_text() or "%" in text.get_text() for text in valid_ax.texts)
        assert any("median:" in text.get_text() and "°" in text.get_text() for text in footer_ax.texts)
        assert any("%" in text.get_text() for text in footer_ax.texts)
        sidecar_ax = next(ax for ax in fig.axes if ax.get_title() == "Median angle (°)")
        assert len(sidecar_ax.images) == 1
        sidecar_quivers = [coll for coll in sidecar_ax.collections if isinstance(coll, Quiver)]
        assert len(sidecar_quivers) == 2
        assert len(_pair_angle_texts(sidecar_ax)) == _unique_position_count(_quiver_positions(sidecar_quivers[0]))
        assert all(
            not isinstance(coll, PathCollection) or np.asarray(coll.get_offsets(), dtype=float).shape[0] <= 1
            for coll in sidecar_ax.collections
        )
        _assert_no_angle_label_overlap(fig, sidecar_ax)
        assert not _long_axis_lines(sidecar_ax)
        assert not _has_representative_angle_marker(sidecar_ax)

    def test_dense_object_target_stories_render_separately(self):
        from matplotlib.quiver import Quiver

        manifest = _make_dense_angle_manifest(two_objects=True)
        for obj_label in ("object_1", "object_2"):
            group = StoryGroup(
                key=obj_label,
                title=obj_label.replace("_", " ").title(),
                features=[
                    FeatureHeuristic(
                        label=f"heuristic_olfaction_{obj_label}",
                        result_column=f"olfaction_{obj_label}",
                        coordinate_labels=["nose"],
                        perimeter_label=obj_label,
                        description="...",
                    ),
                ],
                heuristic_names=[f"olfaction_{obj_label}"],
                perimeter_label=obj_label,
            )
            fig = plot_story(manifest, group)
            assert isinstance(fig, Figure)
            assert len(fig.axes) == 5
            assert any(ax.get_title() == "Median angle (°)" for ax in fig.axes)
            assert any(ax.get_title() == "rejected" for ax in fig.axes)
            valid_ax = next(
                ax for ax in fig.axes if ax.get_title() == f"Olfaction {obj_label.replace('_', ' ').title()}"
            )
            footer_ax = next(ax for ax in fig.axes if any("rep:" in text.get_text() for text in ax.texts))
            quivers = [coll for coll in valid_ax.collections if isinstance(coll, Quiver)]
            assert len(quivers) == 2
            assert all(getattr(coll, "units", None) != "xy" for coll in quivers)
            unique_positions = _unique_position_count(_quiver_positions(quivers[0]))
            assert 10 < unique_positions <= 36
            assert np.all((_quiver_lengths(quivers[0]) > 20.0) & (_quiver_lengths(quivers[0]) < 30.0))
            assert np.all((_quiver_lengths(quivers[1]) > 20.0) & (_quiver_lengths(quivers[1]) < 30.0))
            assert len(_pair_angle_texts(valid_ax)) == unique_positions
            _assert_no_angle_label_overlap(fig, valid_ax)
            assert not _long_axis_lines(valid_ax)
            assert not _has_representative_angle_marker(valid_ax)
            circle_lws = _circle_linewidths(valid_ax)
            assert circle_lws
            assert all(lw < 2.0 for lw in circle_lws)
            assert not any(f"{obj_label}: rep" in text.get_text() for text in valid_ax.texts)
            assert any("rep:" in text.get_text() and "°" in text.get_text() for text in footer_ax.texts)
            sidecar_ax = next(ax for ax in fig.axes if ax.get_title() == "Median angle (°)")
            assert len(sidecar_ax.images) == 1
            sidecar_quivers = [coll for coll in sidecar_ax.collections if isinstance(coll, Quiver)]
            assert len(sidecar_quivers) == 2
            assert len(_pair_angle_texts(sidecar_ax)) == _unique_position_count(_quiver_positions(sidecar_quivers[0]))
            _assert_no_angle_label_overlap(fig, sidecar_ax)
            assert not _long_axis_lines(sidecar_ax)
            assert not _has_representative_angle_marker(sidecar_ax)

    def test_object_decomposition_story_renders_grouped_columns(self, manifest: InspectionManifest):
        body_1 = manifest.df["body_proximity_object_1"].to_numpy()
        body_2 = np.roll(body_1, 4)
        manifest.perimeters.append(
            PerimeterSpec(
                label="object_2",
                shape="circle",
                params={"center_x": 0.24, "center_y": 0.15, "radius": 0.02},
            )
        )
        manifest.df = manifest.df.with_columns(
            [
                pl.Series("body_proximity_object_2", body_2),
                pl.Series("__bk_diag__body_proximity_object_1__nose_near", body_1),
                pl.Series("__bk_diag__body_proximity_object_1__tail_base_near", np.roll(body_1, 2)),
                pl.Series("__bk_diag__body_proximity_object_2__nose_near", body_2),
            ]
        )

        group = StoryGroup(
            key="object_decomposition_body_proximity",
            title="Body Proximity",
            features=[
                FeatureHeuristicDecomposition(
                    label="heuristic_decomposition_body_proximity_object_1",
                    heuristic_name="body_proximity_object_1",
                    result_column="body_proximity_object_1",
                    coordinate_labels=["nose"],
                    perimeter_label="object_1",
                    components=[
                        HeuristicDecompositionComponent(
                            label="nose_near",
                            result_column="__bk_diag__body_proximity_object_1__nose_near",
                            coordinate_labels=["nose"],
                            perimeter_label="object_1",
                        ),
                        HeuristicDecompositionComponent(
                            label="tail_base_near",
                            result_column="__bk_diag__body_proximity_object_1__tail_base_near",
                            coordinate_labels=["tail_base"],
                            perimeter_label="object_1",
                        ),
                    ],
                    show_final_result=True,
                    description="...",
                ),
                FeatureHeuristicDecomposition(
                    label="heuristic_decomposition_body_proximity_object_2",
                    heuristic_name="body_proximity_object_2",
                    result_column="body_proximity_object_2",
                    coordinate_labels=["nose"],
                    perimeter_label="object_2",
                    components=[
                        HeuristicDecompositionComponent(
                            label="nose_near",
                            result_column="__bk_diag__body_proximity_object_2__nose_near",
                            coordinate_labels=["nose"],
                            perimeter_label="object_2",
                        ),
                    ],
                    show_final_result=True,
                    description="...",
                ),
            ],
        )
        fig = plot_story(manifest, group)
        assert isinstance(fig, Figure)
        titles = {ax.get_title() for ax in fig.axes}
        assert "nose_near" in titles
        assert "tail_base_near" in titles
        assert "final" in titles
        texts = [text.get_text() for ax in fig.axes for text in ax.texts]
        assert "Object 1" in texts
        assert "Object 2" in texts

    def test_object_decomposition_story_angle_panels_split_per_target(self):
        manifest = _make_dense_angle_manifest(two_objects=True)
        nose_1 = manifest.df["olfaction_object_1"].to_numpy()
        nose_2 = manifest.df["olfaction_object_2"].to_numpy()
        manifest.df = manifest.df.with_columns(
            [
                pl.Series("__bk_diag__olfaction_object_1__nose_near", nose_1),
                pl.Series("__bk_diag__olfaction_object_2__nose_near", nose_2),
            ]
        )

        for obj_label in ("object_1", "object_2"):
            group = StoryGroup(
                key=f"object_decomposition_olfaction_{obj_label}",
                title="Olfaction",
                features=[
                    FeatureHeuristicDecomposition(
                        label=f"heuristic_decomposition_olfaction_{obj_label}",
                        heuristic_name=f"olfaction_{obj_label}",
                        result_column=f"olfaction_{obj_label}",
                        coordinate_labels=["nose"],
                        perimeter_label=obj_label,
                        components=[
                            HeuristicDecompositionComponent(
                                label="nose_near",
                                result_column=f"__bk_diag__olfaction_{obj_label}__nose_near",
                                coordinate_labels=["nose"],
                                perimeter_label=obj_label,
                            ),
                        ],
                        show_final_result=True,
                        description="...",
                    ),
                ],
            )
            fig = plot_story(manifest, group)
            assert isinstance(fig, Figure)
            texts = [text.get_text() for ax in fig.axes for text in ax.texts]
            assert obj_label.replace("_", " ").title() in texts
            sidecar_axes = [ax for ax in fig.axes if ax.get_title() == "Median angle (°)"]
            assert len(sidecar_axes) == 2
            assert all(len(ax.images) == 1 for ax in sidecar_axes)
            assert all(not _long_axis_lines(ax) for ax in sidecar_axes)
            angle_axes = [ax for ax in fig.axes if ax.get_title() in {"nose_near", "final"}]
            assert all(not _has_representative_angle_marker(ax) for ax in angle_axes)
            for ax in angle_axes:
                circle_lws = _circle_linewidths(ax)
                assert circle_lws
                assert all(lw < 2.0 for lw in circle_lws)

    def test_discrimination_story(self, manifest: InspectionManifest):
        group = StoryGroup(
            key="discrimination",
            title="Object Discrimination",
            features=[
                FeatureNortDiscrimination(
                    label="object_discrimination",
                    object_labels=["object_1", "arena"],
                    exploration_times=[0.0, 0.0],
                    coordinate_labels=["nose"],
                    description="...",
                ),
            ],
        )
        fig = plot_story(manifest, group)
        assert isinstance(fig, Figure)

    def test_returns_none_without_frame(self):
        """Story with no video/frame returns None."""
        manifest = InspectionManifest(
            video=VideoMeta(fps=30.0, total_frames=0, resolution=(0, 0), meters_per_pixel=0.001),
            labels=[],
            label_colors={},
            coordinate_columns={},
            perimeters=[],
            heuristics=[],
            settings=InspectionSettings(
                minimum_seconds_tolerance=0.5, maximum_seconds_distraction=1 / 3, meters_per_pixel=0.001
            ),
            scale_derivation=None,
            df=pl.DataFrame(),
            quadrants=[],
            feature_inspections=[],
        )
        group = StoryGroup(
            key="geometry",
            title="Geometry",
            features=[
                FeaturePerimeter(label="p", perimeter_label="arena", description="..."),
            ],
        )
        fig = plot_story(manifest, group)
        assert fig is None
