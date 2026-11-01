"""Tests for story grouping logic.

Covers:
- Each heuristic gets its own story figure (keyed by heuristic name).
- Geometry grouping (Perimeter + Angle).
- Task-specific features staying as self-contained stories.
- Standalone heuristics (freezing, in_center) get their own story.
- Ungrouped features (ScaleDerivation, TimeSeries, Trajectory, Quadrant).
- Real NORT-like experiment with multiple objects.
"""

from __future__ import annotations

from bekiper_inspect.manifest import (
    FeatureAngle,
    FeatureArmSequence,
    FeatureCenterPeriphery,
    FeatureHeuristic,
    FeatureHeuristicCloud,
    FeatureHeuristicDecomposition,
    FeatureNortDiscrimination,
    FeaturePerimeter,
    FeatureProximity,
    FeatureQuadrant,
    FeatureScaleDerivation,
    FeatureTimeSeries,
    FeatureTrajectory,
    FeatureZone,
    HeuristicDecompositionComponent,
    HeuristicMeta,
    PerimeterSpec,
)
from bekiper_inspect.plot.feature._grouping import group_into_stories


def _h(name: str) -> HeuristicMeta:
    return HeuristicMeta(name=name, result_column=name, true_frames=10, total_frames=100, seconds=1.0)


def _perim(label: str) -> PerimeterSpec:
    return PerimeterSpec(label=label, shape="circle", params={"center_x": 0.1, "center_y": 0.1, "radius": 0.05})


def _decomp(name: str, perimeter_label: str) -> FeatureHeuristicDecomposition:
    return FeatureHeuristicDecomposition(
        label=f"heuristic_decomposition_{name}",
        heuristic_name=name,
        result_column=name,
        coordinate_labels=["nose"],
        perimeter_label=perimeter_label,
        components=[
            HeuristicDecompositionComponent(
                label="nose_near",
                result_column=f"{name}_nose_near",
                coordinate_labels=["nose"],
                perimeter_label=perimeter_label,
            )
        ],
        show_final_result=True,
        description="...",
    )


class TestGroupIntoStories:
    def test_empty_inputs(self):
        stories, ungrouped = group_into_stories([], [], [])
        assert stories == []
        assert ungrouped == []

    def test_geometry_group(self):
        features = [
            FeaturePerimeter(label="perimeter_arena", perimeter_label="arena", description="..."),
            FeaturePerimeter(label="perimeter_center", perimeter_label="center", description="..."),
            FeatureAngle(
                label="angle_nose_tail",
                from_coordinate_label="nose",
                to_coordinate_label="tail_base",
                description="...",
            ),
        ]
        stories, ungrouped = group_into_stories(features, [], [_perim("arena"), _perim("center")])

        assert len(stories) == 1
        assert stories[0].key == "geometry"
        assert stories[0].title == "Arena Geometry"
        assert len(stories[0].features) == 3
        assert ungrouped == []

    def test_perimeter_anchored_heuristics_group_per_target(self):
        """Physical object targets split into their own story."""
        perimeters = [_perim("object_1"), _perim("arena")]
        heuristics = [_h("body_proximity_object_1"), _h("olfaction_object_1")]
        features = [
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
        ]

        stories, ungrouped = group_into_stories(features, heuristics, perimeters)

        assert len(stories) == 1
        assert stories[0].key == "object_1"
        assert stories[0].title == "Object 1"
        assert len(stories[0].features) == 2
        assert set(stories[0].heuristic_names) == {"body_proximity_object_1", "olfaction_object_1"}
        assert ungrouped == []

    def test_standalone_heuristic(self):
        """Freezing (no perimeter suffix) gets its own story."""
        heuristics = [_h("freezing")]
        features = [
            FeatureHeuristicCloud(
                label="heuristic_freezing",
                result_column="freezing",
                coordinate_labels=["nose"],
                description="...",
            ),
        ]

        stories, ungrouped = group_into_stories(features, heuristics, [])

        assert len(stories) == 1
        assert stories[0].key == "freezing"
        assert stories[0].heuristic_names == ["freezing"]
        assert ungrouped == []

    def test_task_specific_discrimination(self):
        features = [
            FeatureNortDiscrimination(
                label="object_discrimination",
                object_labels=["novel", "familiar"],
                exploration_times=[12.5, 7.5],
                coordinate_labels=["nose"],
                description="...",
            ),
        ]

        stories, ungrouped = group_into_stories(features, [], [])
        assert len(stories) == 1
        assert stories[0].key == "discrimination"
        assert stories[0].title == "Object Discrimination"
        assert ungrouped == []

    def test_task_specific_arm_sequence(self):
        features = [
            FeatureArmSequence(
                label="arm_sequence",
                arm_sequence=["A", "B", "C"],
                coordinate_labels=["nose"],
                description="...",
            ),
        ]
        stories, ungrouped = group_into_stories(features, [], [])
        assert len(stories) == 1
        assert stories[0].key == "arm_sequence"
        assert stories[0].title == "Arm Entry Sequence"
        assert ungrouped == []

    def test_task_specific_center_periphery(self):
        features = [
            FeatureCenterPeriphery(
                label="center_periphery",
                center_perimeter_label="center",
                arena_perimeter_label="arena",
                coordinate_labels=["nose"],
                center_time_pct=35.0,
                periphery_time_pct=65.0,
                description="...",
            ),
        ]
        stories, ungrouped = group_into_stories(features, [], [])
        assert len(stories) == 1
        assert stories[0].key == "center_periphery"
        assert stories[0].title == "Center vs. Periphery"
        assert ungrouped == []

    def test_ungrouped_features(self):
        """ScaleDerivation, TimeSeries, Trajectory, Quadrant stay ungrouped."""
        features = [
            FeatureScaleDerivation(label="scale_derivation", description="..."),
            FeatureTimeSeries(label="ts_speed", value_column="speed", unit="m/s", description="..."),
            FeatureTrajectory(label="traj_nose", coordinate_labels=["nose"], description="..."),
            FeatureQuadrant(
                label="quadrant_grid", quadrant_label="grid", coordinate_labels=["nose"], description="..."
            ),
        ]

        stories, ungrouped = group_into_stories(features, [], [])
        assert stories == []
        assert len(ungrouped) == 4

    def test_nort_experiment_like(self):
        """Simulate a real NORT experiment: 3 perimeters, 14 heuristics."""
        perimeters = [_perim("object_1"), _perim("object_2"), _perim("arena")]
        heuristics = [
            _h("exploring_object_1"),
            _h("exploring_object_2"),
            _h("exploring_arena"),
            _h("body_proximity_object_1"),
            _h("body_proximity_object_2"),
            _h("body_proximity_arena"),
            _h("olfaction_object_1"),
            _h("olfaction_object_2"),
            _h("olfaction_arena"),
            _h("whisker_interaction_object_1"),
            _h("whisker_interaction_object_2"),
            _h("whisker_interaction_arena"),
            _h("in_center"),
            _h("freezing"),
        ]
        features = [
            FeaturePerimeter(label="perimeter_object_1", perimeter_label="object_1", description="..."),
            FeaturePerimeter(label="perimeter_object_2", perimeter_label="object_2", description="..."),
            FeaturePerimeter(label="perimeter_arena", perimeter_label="arena", description="..."),
        ]
        # Add heuristic features for each.
        for h in heuristics:
            plabel = None
            for p in perimeters:
                if h.name.endswith(f"_{p.label}"):
                    plabel = p.label
                    break
            feature_cls = FeatureHeuristicCloud if h.name == "freezing" else FeatureHeuristic
            features.append(
                feature_cls(
                    label=f"heuristic_{h.name}",
                    result_column=h.result_column,
                    coordinate_labels=["nose"],
                    perimeter_label=plabel,
                    description="...",
                )
            )
        # Add angle + NORT
        features.append(
            FeatureAngle(
                label="angle_nose_tail_base",
                from_coordinate_label="nose",
                to_coordinate_label="tail_base",
                description="...",
            )
        )
        features.append(
            FeatureNortDiscrimination(
                label="object_discrimination",
                object_labels=["object_1", "object_2"],
                exploration_times=[0.0, 0.0],
                coordinate_labels=["nose"],
                description="...",
            )
        )

        stories, ungrouped = group_into_stories(features, heuristics, perimeters)

        story_keys = {s.key for s in stories}
        assert "object_1" in story_keys
        assert "object_2" in story_keys
        assert "objects" in story_keys
        assert "in_center" in story_keys
        assert "freezing" in story_keys
        assert "geometry" in story_keys
        assert "discrimination" in story_keys
        assert ungrouped == []

        object_1 = next(s for s in stories if s.key == "object_1")
        object_2 = next(s for s in stories if s.key == "object_2")
        objects = next(s for s in stories if s.key == "objects")
        assert len(object_1.features) == 4
        assert len(object_1.heuristic_names) == 4
        assert len(object_2.features) == 4
        assert len(object_2.heuristic_names) == 4
        assert len(objects.features) == 8
        assert len(objects.heuristic_names) == 8
        context = next(s for s in stories if s.key == "arena_context")
        assert len(context.features) == 4
        assert len(context.heuristic_names) == 4
        assert all(f.perimeter_label == "arena" for f in context.features)

        # geometry should have 3 perimeters + 1 angle = 4 features.
        geom = next(s for s in stories if s.key == "geometry")
        assert len(geom.features) == 4

    def test_zone_feature_groups_by_perimeter(self):
        """Zone features (e.g. in_center → center perimeter) group correctly."""
        perimeters = [_perim("center")]
        heuristics = [_h("in_center")]
        features = [
            FeatureZone(
                label="zone_in_center",
                perimeter_label="center",
                result_column="in_center",
                coordinate_labels=["nose"],
                description="...",
            ),
        ]

        stories, ungrouped = group_into_stories(features, heuristics, perimeters)
        assert len(stories) == 1
        # in_center targets perimeter "center" → merged into "objects".
        assert stories[0].key == "objects"
        assert stories[0].title == "Object Exploration"
        assert ungrouped == []

    def test_mixed_heuristic_features_keep_first_seen_timeline_order(self):
        features = [
            FeatureProximity(
                label="near_object",
                perimeter_label="object_1",
                buffer_meters=0.02,
                result_column="body_proximity_object_1",
                coordinate_labels=["nose"],
                description="...",
            ),
            FeatureZone(
                label="zone_object",
                perimeter_label="object_1",
                result_column="olfaction_object_1",
                coordinate_labels=["nose"],
                description="...",
            ),
            FeatureHeuristic(
                label="duplicate_timeline",
                perimeter_label="object_1",
                result_column="body_proximity_object_1",
                coordinate_labels=["nose"],
                description="...",
            ),
        ]
        heuristics = [_h("olfaction_object_1"), _h("exploring_object_1"), _h("body_proximity_object_1")]

        stories, ungrouped = group_into_stories(features, heuristics, [_perim("object_1")])

        assert ungrouped == []
        assert len(stories) == 1
        assert stories[0].features == features
        assert all(actual is original for actual, original in zip(stories[0].features, features, strict=True))
        assert stories[0].heuristic_names == [
            "body_proximity_object_1",
            "olfaction_object_1",
            "exploring_object_1",
        ]

    def test_object_decomposition_groups_by_family(self):
        perimeters = [
            _perim("object_1"),
            _perim("object_2"),
            _perim("novel"),
            _perim("arena"),
            _perim("center_zone"),
        ]
        features = [
            _decomp("body_proximity_object_1", "object_1"),
            _decomp("body_proximity_object_2", "object_2"),
            _decomp("body_proximity_novel", "novel"),
            _decomp("body_proximity_arena", "arena"),
            _decomp("body_proximity_center_zone", "center_zone"),
        ]

        stories, ungrouped = group_into_stories(features, [], perimeters)

        assert len(stories) == 1
        assert stories[0].key == "object_decomposition_body_proximity"
        assert stories[0].title == "Body Proximity"
        assert len(stories[0].features) == 3
        assert {feature.heuristic_name for feature in stories[0].features} == {
            "body_proximity_object_1",
            "body_proximity_object_2",
            "body_proximity_novel",
        }
        assert {feature.heuristic_name for feature in ungrouped} == {
            "body_proximity_arena",
            "body_proximity_center_zone",
        }

    def test_angle_object_decompositions_split_per_target(self):
        perimeters = [
            _perim("object_1"),
            _perim("object_2"),
            _perim("novel"),
            _perim("arena"),
        ]
        features = [
            _decomp("olfaction_object_1", "object_1"),
            _decomp("olfaction_object_2", "object_2"),
            _decomp("olfaction_novel", "novel"),
            _decomp("olfaction_arena", "arena"),
        ]

        stories, ungrouped = group_into_stories(features, [], perimeters)

        story_keys = {story.key for story in stories}
        assert story_keys == {
            "object_decomposition_olfaction",
            "object_decomposition_olfaction_object_1",
            "object_decomposition_olfaction_object_2",
            "object_decomposition_olfaction_novel",
        }
        assert all(story.title == "Olfaction" for story in stories if story.key != "object_decomposition_olfaction")
        merged = next(s for s in stories if s.key == "object_decomposition_olfaction")
        assert merged.title == "Olfaction — All Objects"
        assert len(merged.features) == 3
        assert {feature.heuristic_name for feature in ungrouped} == {"olfaction_arena"}

    def test_heuristics_without_features_appear_in_timelines(self):
        """Heuristics with no feature entry still get timeline panels."""
        perimeters = [_perim("object_1")]
        heuristics = [_h("body_proximity_object_1"), _h("olfaction_object_1")]
        # Only body_proximity has a feature entry.
        features = [
            FeatureHeuristic(
                label="heuristic_body_proximity_object_1",
                result_column="body_proximity_object_1",
                coordinate_labels=["nose"],
                perimeter_label="object_1",
                description="...",
            ),
        ]

        stories, _ = group_into_stories(features, heuristics, perimeters)
        assert len(stories) == 1
        assert stories[0].key == "object_1"
        # Both heuristics should appear in timeline names.
        assert "body_proximity_object_1" in stories[0].heuristic_names
        assert "olfaction_object_1" in stories[0].heuristic_names

    def test_legacy_coarse_story_keys_are_reinterpreted(self):
        perimeters = [_perim("object_1"), _perim("object_2"), _perim("arena")]
        heuristics = [_h("olfaction_object_1"), _h("olfaction_object_2")]
        features = [
            FeatureHeuristic(
                label="heuristic_olfaction_object_1",
                result_column="olfaction_object_1",
                coordinate_labels=["nose"],
                perimeter_label="object_1",
                description="...",
                story_key="objects",
            ),
            FeatureHeuristicDecomposition(
                label="heuristic_decomposition_olfaction_object_2",
                heuristic_name="olfaction_object_2",
                result_column="olfaction_object_2",
                coordinate_labels=["nose"],
                perimeter_label="object_2",
                components=[],
                show_final_result=True,
                description="...",
                story_key="object_decomposition_olfaction",
            ),
        ]

        stories, ungrouped = group_into_stories(features, heuristics, perimeters)

        assert ungrouped == []
        assert {story.key for story in stories} == {
            "object_1",
            "object_decomposition_olfaction_object_2",
        }
