"""Restored composites share evaluated inputs with the per-object stories."""

import pytest

from bekiper_inspect.manifest import FeatureHeuristic, HeuristicMeta, PerimeterSpec
from bekiper_inspect.plot.feature import group_into_stories


@pytest.mark.parametrize("targets", [("object_1", "object_2"), ("A", "B")])
def test_combined_story_is_emitted_alongside_object_stories(targets):
    features = [
        FeatureHeuristic(
            label=f"heuristic_olfaction_{target}",
            result_column=f"olfaction_{target}",
            coordinate_labels=["nose"],
            perimeter_label=target,
            description="Olfaction",
            story_key="objects",
        )
        for target in targets
    ]
    perimeters = [
        PerimeterSpec(label=target, shape="circle", params={"center_x": x, "center_y": 0.0, "radius": 0.05})
        for target, x in zip(targets, [0.1, 0.3], strict=True)
    ]
    heuristics = [
        HeuristicMeta(name=f.result_column, result_column=f.result_column, true_frames=1, total_frames=2, seconds=0.1)
        for f in features
    ]
    groups, ungrouped = group_into_stories(features, heuristics, perimeters)
    by_key = {group.key: group for group in groups}
    assert {*targets, "objects"} <= by_key.keys()
    assert [f.result_column for f in by_key["objects"].features] == [f.result_column for f in features]
    assert not ungrouped


def test_shared_objects_preserve_arena_context_without_mixing_inputs():
    targets = ["object_1", "object_2", "arena"]
    features = [
        FeatureHeuristic(
            label=f"olfaction_{target}",
            result_column=f"olfaction_{target}",
            coordinate_labels=["nose"],
            perimeter_label=target,
            description=target,
        )
        for target in targets
    ]
    heuristics = [
        HeuristicMeta(name=f.result_column, result_column=f.result_column, true_frames=1, total_frames=2, seconds=0.1)
        for f in features
    ]
    perimeters = [
        PerimeterSpec(label=target, shape="circle", params={"center_x": 0.1, "center_y": 0.1, "radius": 0.05})
        for target in targets
    ]
    groups, _ = group_into_stories(features, heuristics, perimeters)
    by_key = {group.key: group for group in groups}
    assert by_key["objects"].perimeter_label is None
    assert by_key["objects"].features == features[:2]
    assert by_key["objects"].heuristic_names == [f.result_column for f in features[:2]]
    assert by_key["arena_context"].features == features[2:]
    assert by_key["arena_context"].perimeter_label == "arena"
