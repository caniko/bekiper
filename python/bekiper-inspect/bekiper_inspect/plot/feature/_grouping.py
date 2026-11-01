"""Group flat FeatureInspection entries into story-telling multi-panel figures.

The Rust emitter produces one FeatureInspection per behavioural element —
correct granularity for the data model. This module groups related entries
by deriving a *story key* from naming conventions (e.g. heuristics ending
in ``_object_1`` share a story about object_1).

Grouping is purely a presentation concern and lives entirely in Python.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from bekiper_inspect.manifest._features import (
    FeatureHeuristic,
    FeatureHeuristicCloud,
    FeatureHeuristicDecomposition,
    FeatureInspection,
    FeatureProximity,
    FeatureZone,
)
from bekiper_inspect.manifest._types import HeuristicMeta, PerimeterSpec

_CONTEXT_TARGETS = frozenset({"arena", "center", "center_zone"})
#: Self-contained task result panels: feature ``kind`` -> story key.
TASK_STORY_KEYS = {
    "object_discrimination": "discrimination",
    "arm_sequence": "arm_sequence",
    "ymaze_arm_sections": "arm_sequence",
    "center_periphery": "center_periphery",
}
_GEOMETRY_KINDS = frozenset({"perimeter", "angle"})
_GROUP_TITLES = {
    "geometry": "Arena Geometry",
    "discrimination": "Object Discrimination",
    "arm_sequence": "Arm Entry Sequence",
    "center_periphery": "Center vs. Periphery",
    "objects": "Object Exploration",
}


@dataclass
class StoryGroup:
    """A cluster of related features rendered as one multi-panel figure."""

    key: str
    title: str
    features: list[FeatureInspection] = field(default_factory=list)
    heuristic_names: list[str] = field(default_factory=list)
    perimeter_label: str | None = None


def _perimeter_labels(perimeters: list[PerimeterSpec]) -> set[str]:
    return {p.label for p in perimeters}


def _heuristic_target_suffix(name: str, perim_labels: set[str]) -> str | None:
    """Extract the target suffix from a heuristic name.

    For ``olfaction_object_1`` returns ``"object_1"``, for ``freezing``
    returns ``None``.  Checks against known perimeter labels to avoid
    false positives.
    """
    for plabel in sorted(perim_labels, key=len, reverse=True):
        if name.endswith(f"_{plabel}"):
            return plabel
        # Also handle "in_<label>" pattern (e.g. "in_center").
        if name == f"in_{plabel}":
            return plabel
    return None


def _feature_perimeter_label(feature: FeatureInspection) -> str | None:
    """Return the perimeter_label from a feature, if it has one."""
    return getattr(feature, "perimeter_label", None)


def _is_physical_object_target(label: str) -> bool:
    return (label.startswith("object_") and not label.startswith("object_decomposition_")) or label in {
        "novel",
        "familiar",
    }


def _is_angle_family(name: str) -> bool:
    return "olfaction" in name or "whisker_interaction" in name


def _decomposition_story_key(
    feature: FeatureHeuristicDecomposition,
    perim_labels: set[str],
    object_targets: set[str],
) -> str | None:
    """Return the merge key for real object-targeted decompositions."""
    target = _heuristic_target_suffix(feature.heuristic_name, perim_labels)
    if target is None or target in _CONTEXT_TARGETS:
        return None

    suffix = f"_{target}"
    if not feature.heuristic_name.endswith(suffix):
        return None

    family = feature.heuristic_name[: -len(suffix)]
    if not family:
        return None

    if target in object_targets and _is_angle_family(family):
        return f"object_decomposition_{family}_{target}"

    return f"object_decomposition_{family}"


def _resolved_story_key(
    *,
    explicit_key: str | None,
    derived_key: str | None,
    object_targets: set[str],
) -> str | None:
    """Prefer finer derived story keys over legacy coarse manifest keys."""
    if explicit_key is None:
        return derived_key
    if derived_key is None or explicit_key == derived_key:
        return explicit_key
    if explicit_key == "objects" and derived_key in object_targets:
        return derived_key
    if explicit_key.startswith("object_decomposition_") and derived_key.startswith(f"{explicit_key}_"):
        return derived_key
    return explicit_key


def group_into_stories(
    features: list[FeatureInspection],
    heuristics: list[HeuristicMeta],
    perimeters: list[PerimeterSpec],
) -> tuple[list[StoryGroup], list[FeatureInspection]]:
    """Group features into object stories, shared composites and singletons.

    Returns ``(stories, ungrouped)`` where *stories* is a list of
    :class:`StoryGroup` and *ungrouped* is features that remain
    individual figures.
    """
    perim_labels = _perimeter_labels(perimeters)
    object_targets = {label for label in perim_labels if _is_physical_object_target(label)}
    for target in perim_labels - _CONTEXT_TARGETS:
        if any(
            h.name.startswith(("olfaction_", "whisker_interaction_", "body_proximity_", "exploring_"))
            and h.name.endswith(f"_{target}")
            for h in heuristics
        ):
            object_targets.add(target)

    # Build a lookup: heuristic name -> target suffix.
    heuristic_targets: dict[str, str | None] = {}
    for h in heuristics:
        heuristic_targets[h.name] = _heuristic_target_suffix(h.name, perim_labels)

    # Build a lookup: heuristic result_column -> heuristic name.
    result_col_to_name: dict[str, str] = {h.result_column: h.name for h in heuristics}

    # ── Pass 1: assign each feature a story key ──────────────────────
    keyed: dict[str, list[FeatureInspection]] = defaultdict(list)
    ungrouped: list[FeatureInspection] = []

    for feature in features:
        explicit_key = getattr(feature, "story_key", None)
        derived_key = _derive_story_key(feature, perim_labels, heuristic_targets, result_col_to_name, object_targets)
        key = _resolved_story_key(
            explicit_key=explicit_key,
            derived_key=derived_key,
            object_targets=object_targets,
        )
        if key is not None:
            keyed[key].append(feature)
        else:
            ungrouped.append(feature)

    # ── Pass 2: build StoryGroup objects ─────────────────────────────
    stories: list[StoryGroup] = []
    for key, group_features in keyed.items():
        # Collect heuristic names that belong to this story.
        group_heuristic_names: list[str] = []
        group_perimeter: str | None = None

        for feat in group_features:
            if isinstance(feat, (FeatureHeuristic, FeatureHeuristicCloud, FeatureZone, FeatureProximity)):
                rc = feat.result_column
                hname = result_col_to_name.get(rc)
                if hname and hname not in group_heuristic_names:
                    group_heuristic_names.append(hname)
            plabel = _feature_perimeter_label(feat)
            if plabel and group_perimeter is None:
                group_perimeter = plabel

        # Also include heuristics that target this story key but may not
        # have a feature entry (they still need a timeline panel).
        for h in heuristics:
            target = heuristic_targets.get(h.name)
            if key == "objects":
                # Only context-target heuristics stay in the merged objects story.
                if target is not None and target not in object_targets and h.name not in group_heuristic_names:
                    group_heuristic_names.append(h.name)
            elif target == key and h.name not in group_heuristic_names:
                group_heuristic_names.append(h.name)

        title = _title_for_group(key, group_features, perim_labels)
        stories.append(
            StoryGroup(
                key=key,
                title=title,
                features=group_features,
                heuristic_names=group_heuristic_names,
                perimeter_label=group_perimeter,
            )
        )

    # Keep the readable per-object views and the historical all-object view.
    # Composites reference the same feature entries; they do not rescore data.
    object_stories = [story for story in stories if story.key in object_targets]
    if len(object_stories) > 1:
        context = next((story for story in stories if story.key == "objects"), None)
        if context is not None:
            context.key = "arena_context"
            context.title = "Arena and Zone Context"
        merged = StoryGroup(key="objects", title="Object Exploration — All Objects")
        stories.append(merged)
        for story in object_stories:
            merged.features.extend(story.features)
            merged.heuristic_names.extend(name for name in story.heuristic_names if name not in merged.heuristic_names)

    decomposition_families: dict[str, list[FeatureHeuristicDecomposition]] = defaultdict(list)
    for feature in features:
        if isinstance(feature, FeatureHeuristicDecomposition) and not feature.label.startswith("geometry_vectors_"):
            target = _heuristic_target_suffix(feature.heuristic_name, perim_labels)
            if target and target in object_targets:
                family = feature.heuristic_name[: -(len(target) + 1)]
                decomposition_families[family].append(feature)
    existing_keys = {story.key for story in stories}
    for family, members in decomposition_families.items():
        key = f"object_decomposition_{family}"
        if len(members) > 1 and key not in existing_keys:
            stories.append(
                StoryGroup(key=key, title=f"{family.replace('_', ' ').title()} — All Objects", features=members)
            )

    # Sort stories: perimeter-anchored first (by label), then standalone.
    stories.sort(key=lambda s: (s.perimeter_label is None, s.key))

    return stories, ungrouped


def _derive_story_key(
    feature: FeatureInspection,
    perim_labels: set[str],
    heuristic_targets: dict[str, str | None],
    result_col_to_name: dict[str, str],
    object_targets: set[str],
) -> str | None:
    """Return a story key for *feature*, or None if it should stay ungrouped."""

    # Task-specific composite features: already self-contained stories.
    if feature.kind in TASK_STORY_KEYS:
        return TASK_STORY_KEYS[feature.kind]

    # Perimeter and angle features → geometry story.
    if feature.kind in _GEOMETRY_KINDS:
        return "geometry"

    # Heuristic / HeuristicCloud / Zone / Proximity: perimeter-anchored heuristics all
    # merge into one "objects" story; standalone ones keep their own key.
    if isinstance(feature, (FeatureHeuristic, FeatureHeuristicCloud, FeatureZone, FeatureProximity)):
        rc = getattr(feature, "result_column", None)
        hname = result_col_to_name.get(rc) if rc else None
        if hname:
            target = heuristic_targets.get(hname)
            if target:
                return target if target in object_targets else "objects"
            # Standalone heuristic (e.g. "freezing", "in_center"):
            # use the heuristic name itself as the key.
            return hname
        # No matching heuristic meta — try perimeter_label.
        plabel = _feature_perimeter_label(feature)
        if plabel and plabel in perim_labels:
            return plabel if plabel in object_targets else "objects"

    if isinstance(feature, FeatureHeuristicDecomposition):
        return _decomposition_story_key(feature, perim_labels, object_targets)

    # RayCast, ScaleDerivation, TimeSeries, Trajectory, Quadrant: ungrouped.
    return None


def _title_for_group(
    key: str,
    group_features: list[FeatureInspection],
    perim_labels: set[str],
) -> str:
    """Human-readable title from a story key and its grouped features."""
    readable = key.replace("_", " ").title()
    if key in _GROUP_TITLES:
        return _GROUP_TITLES[key]
    if _is_physical_object_target(key):
        return readable
    if key.startswith("object_decomposition_"):
        feature = next(
            (candidate for candidate in group_features if isinstance(candidate, FeatureHeuristicDecomposition)),
            None,
        )
        if feature is not None:
            target = _heuristic_target_suffix(feature.heuristic_name, perim_labels)
            if target is not None and feature.heuristic_name.endswith(f"_{target}"):
                family = str(feature.heuristic_name[: -(len(target) + 1)])
                return family.replace("_", " ").title()
        family = key.removeprefix("object_decomposition_")
        return family.replace("_", " ").title()
    return readable
