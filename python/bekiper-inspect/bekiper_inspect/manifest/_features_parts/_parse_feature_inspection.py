from typing import Any, cast

import numpy as np

from .featureperimeter import *


def _parse_feature_inspection(raw: dict[str, Any]) -> FeatureInspection | None:
    """Build the right FeatureInspection variant from a TOML table.

    Returns None for unknown `kind` values so that future-Rust schemas
    do not break older Python clients — the unknown feature is silently
    skipped by the plot dispatcher.
    """
    kind = raw.get("kind")
    rep_frame = raw.get("representative_frame_index")
    # Backwards-compatible coordinate_labels resolution (P4.1).
    coord_labels = raw.get("coordinate_labels")
    if coord_labels is None:
        single = raw.get("coordinate_label")
        coord_labels = [single] if single else []
    # Per-feature styling overrides (P4.2).
    styling = {
        "color": raw.get("color"),
        "linewidth": raw.get("linewidth"),
        "title_override": raw.get("title_override"),
        "story_key": raw.get("story_key"),
    }
    styling_kwargs = cast(Any, styling)
    if kind == "state_distribution":
        return FeatureStateDistribution(
            **{
                key: raw[key]
                for key in (
                    "label",
                    "category_labels",
                    "frame_counts",
                    "seconds",
                    "percentages",
                    "denominator_frames",
                    "denominator_seconds",
                    "source_columns",
                    "basis",
                    "style",
                    "description",
                )
            },
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "perimeter":
        return FeaturePerimeter(
            label=raw["label"],
            perimeter_label=raw["perimeter_label"],
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "zone":
        return FeatureZone(
            label=raw["label"],
            perimeter_label=raw["perimeter_label"],
            result_column=raw["result_column"],
            coordinate_labels=coord_labels,
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "quadrant":
        return FeatureQuadrant(
            label=raw["label"],
            quadrant_label=raw["quadrant_label"],
            coordinate_labels=coord_labels,
            description=raw["description"],
            occupancy_grid=raw.get("occupancy_grid"),
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "heuristic":
        return FeatureHeuristic(
            label=raw["label"],
            result_column=raw["result_column"],
            coordinate_labels=coord_labels,
            perimeter_label=raw.get("perimeter_label"),
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "heuristic_decomposition":
        return FeatureHeuristicDecomposition(
            label=raw["label"],
            heuristic_name=raw["heuristic_name"],
            result_column=raw["result_column"],
            coordinate_labels=coord_labels,
            components=[
                HeuristicDecompositionComponent(
                    label=component["label"],
                    result_column=component["result_column"],
                    coordinate_labels=component.get("coordinate_labels", []),
                    perimeter_label=component.get("perimeter_label"),
                )
                for component in raw.get("components", [])
            ],
            perimeter_label=raw.get("perimeter_label"),
            show_final_result=bool(raw.get("show_final_result", False)),
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "heuristic_cloud":
        return FeatureHeuristicCloud(
            label=raw["label"],
            result_column=raw["result_column"],
            coordinate_labels=coord_labels,
            perimeter_label=raw.get("perimeter_label"),
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "angle":
        return FeatureAngle(
            label=raw["label"],
            from_coordinate_label=raw["from_coordinate_label"],
            to_coordinate_label=raw["to_coordinate_label"],
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "trajectory":
        return FeatureTrajectory(
            label=raw["label"],
            coordinate_labels=coord_labels,
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "scale_derivation":
        return FeatureScaleDerivation(
            label=raw["label"],
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "ray_cast":
        return FeatureRayCast(
            label=raw["label"],
            origin_label=raw["origin_label"],
            angle_column=raw["angle_column"],
            length_meters=raw["length_meters"],
            cone_width_deg=raw["cone_width_deg"],
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "detection_fields":
        return FeatureDetectionFields(
            **{
                key: raw[key]
                for key in (
                    "label",
                    "nose_label",
                    "head_label",
                    "cone_half_angle_deg",
                    "nose_reach_m",
                    "contact_distance_m",
                    "description",
                )
            },
            **{
                key: raw.get(key, default)
                for key, default in (
                    ("whisker_apex_fraction", None),
                    ("whisker_half_angle_deg", None),
                    ("whisker_length_m", None),
                    ("object_labels", []),
                    ("nose_near_columns", []),
                    ("nose_gate_columns", []),
                    ("whisker_hit_columns", []),
                )
            },
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "time_series":
        return FeatureTimeSeries(
            label=raw["label"],
            value_column=raw["value_column"],
            unit=raw["unit"],
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "proximity":
        return FeatureProximity(
            label=raw["label"],
            perimeter_label=raw["perimeter_label"],
            buffer_meters=raw["buffer_meters"],
            result_column=raw["result_column"],
            coordinate_labels=coord_labels,
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "object_discrimination":
        raw_qc = raw.get("qualia_channels")
        qualia_channels = None
        if raw_qc:
            qualia_channels = [
                QualiaChannelSpec(
                    heuristic_type=ch["heuristic_type"],
                    per_object_seconds=ch["per_object_seconds"],
                    per_object_bouts=[int(b) for b in ch["per_object_bouts"]],
                    relative_bias=ch["relative_bias"],
                )
                for ch in raw_qc
            ]
        return FeatureNortDiscrimination(
            label=raw["label"],
            protocol=raw.get("protocol"),
            object_labels=raw["object_labels"],
            exploration_times=raw["exploration_times"],
            coordinate_labels=coord_labels,
            description=raw["description"],
            discrimination_index=raw.get("discrimination_index"),
            recognition_index=raw.get("recognition_index"),
            qualia_channels=qualia_channels,
            alternation_count=raw.get("alternation_count"),
            merged_channels=raw.get("merged_channels"),
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "arm_sequence":
        return FeatureArmSequence(
            label=raw["label"],
            arm_sequence=raw["arm_sequence"],
            coordinate_labels=coord_labels,
            description=raw["description"],
            alternation_pct=raw.get("alternation_pct"),
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "ymaze_arm_sections":
        return FeatureYMazeArmSections(
            label=raw["label"],
            perimeter_label=raw["perimeter_label"],
            proximal_threshold=raw["proximal_threshold"],
            threshold_lines=[(t["arm"], tuple(t["p1"]), tuple(t["p2"])) for t in raw.get("threshold_lines", [])],
            coordinate_labels=coord_labels,
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    if kind == "center_periphery":
        return FeatureCenterPeriphery(
            label=raw["label"],
            center_perimeter_label=raw["center_perimeter_label"],
            arena_perimeter_label=raw["arena_perimeter_label"],
            coordinate_labels=coord_labels,
            center_time_pct=raw["center_time_pct"],
            periphery_time_pct=raw["periphery_time_pct"],
            description=raw["description"],
            representative_frame_index=rep_frame,
            **styling_kwargs,
        )
    return None


__all__ = ["_parse_feature_inspection", "find_representative_frame"]


def find_representative_frame(boolean_index: np.ndarray) -> int | None:
    """Pick the middle frame of the longest True bout in a boolean array.

    Used to choose a "meaningful" video frame for a heuristic-anchored
    inspection figure — the frame where the behaviour was clearly active
    rather than an arbitrary frame 0. Returns ``None`` when the boolean
    array contains no True values.
    """
    if boolean_index.size == 0:
        return None
    arr = np.asarray(boolean_index, dtype=bool)
    if not arr.any():
        return None

    # Find contiguous True spans by diffing the boolean array.
    padded = np.concatenate(([False], arr, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0:
        return None

    lengths = ends - starts
    longest = int(np.argmax(lengths))
    start = int(starts[longest])
    end = int(ends[longest])
    return (start + end - 1) // 2
