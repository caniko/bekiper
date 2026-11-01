import json
from pathlib import Path

import pytest

from reference_post_analysis.events import nort_bouts, oft_zone_intervals, ymaze_coordinate_entries, ymaze_provenance

GEOMETRY = {
    "version": "imen_radial_vertex_span_v1",
    "canonical_vertices": True,
    "center": [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
    "arms": {
        "A": [[0.1, 0], [0.6, 0.05], [1.1, 0], [0.6, -0.05]],
        "B": [[0, 0.1], [0.05, 0.6], [0, 1.1], [-0.05, 0.6]],
        "C": [[-0.1, 0], [-0.6, 0.05], [-1.1, 0], [-0.6, -0.05]],
    },
}


@pytest.mark.parametrize("threshold", [25, 45, 65])
@pytest.mark.parametrize(
    "head,tail,frames",
    [
        ([0.2, 0.9, 0.2], [0.2, 0.2, 0.2], []),
        ([0.2, 0.2, 0.2], [0.2, 0.9, 0.2], []),
        ([0.2, 0.9, 0.9, 0.2], [0.2, 0.2, 0.9, 0.2], [2]),
        ([0.2, 0.2, 0.9, 0.2], [0.2, 0.9, 0.9, 0.2], [2]),
        ([0.2, 0.9, 0.2], [0.2, 0.9, 0.2], [1]),
    ],
)
def test_dual_landmark_no_dwell(threshold, head, tail, frames):
    samples = [
        {"timestamp_s": i / 30, "head_center_x": h, "head_center_y": 0, "tail_base_x": t, "tail_base_y": 0}
        for i, (h, t) in enumerate(zip(head, tail, strict=True))
    ]
    result = ymaze_coordinate_entries(samples, GEOMETRY, threshold)
    assert [r["sample_index"] for r in result["event_debug"] if r["entry"]] == frames
    assert len(result["valid_arm_entry_sequence"]) == len(frames)


def segment(identity, start, end, object_id="novel", angles=None, speed=0.01, **extra):
    return {
        "segment_id": identity,
        "start_s": start,
        "end_s": end,
        "object_id": object_id,
        "candidate_eligible": True,
        "angles_degrees": angles or [0, 0, 90],
        "nose_speeds_m_per_s": [speed] * 3,
        **extra,
    }


def score(segments, duration="exclude_gaps"):
    return nort_bouts(segments, gap_duration=duration, gap_medians="observed_samples_only")


def test_merge_before_duration_with_explicit_gap_policy_and_lineage():
    parts = [segment("a", 0, 0.6), segment("b", 1, 1.6)]
    result = score(parts)
    assert result["object_exploration_s"]["novel"] == pytest.approx(1.2)
    merged = result["bout_debug"][-1]
    assert merged["component_ids"] == ["a", "b"]
    assert merged["median_angle_degrees"] == 0  # brief frame-angle failure survives
    assert score(parts, "include_gaps")["object_exploration_s"]["novel"] == 1.6


def test_gap_point_five_is_not_merged_and_different_objects_never_merge():
    assert score([segment("a", 0, 0.6), segment("b", 1.1, 1.7)])["valid_bouts"] == 0
    assert score([segment("a", 0, 0.6), segment("b", 0.7, 1.3, "familiar")])["valid_bouts"] == 0


@pytest.mark.parametrize(
    "duration,speed,angle,valid",
    [(1, 0.049999, 45, True), (0.999999, 0.01, 0, False), (1, 0.05, 0, False), (1, 0.01, 45.000001, False)],
)
def test_all_strict_and_inclusive_bout_boundaries(duration, speed, angle, valid):
    result = score([segment("a", 0, duration, angles=[angle], speed=speed)])
    assert result["bout_debug"][-1]["valid"] is valid


def test_behaviour_exclusion_object_directed_exception():
    assert score([segment("a", 0, 1, behaviour="grooming")])["valid_bouts"] == 0
    assert score([segment("a", 0, 1, behaviour="grooming", object_directed=True)])["valid_bouts"] == 1


@pytest.mark.parametrize("duration", ["include_gaps", "exclude_gaps"])
def test_rejected_behaviour_on_another_object_is_a_session_wide_merge_barrier(duration):
    result = score(
        [
            segment("a1", 0, 0.6, "A"),
            segment("grooming", 0.65, 0.95, "B", behaviour="grooming"),
            segment("a2", 1, 1.6, "A"),
        ],
        duration,
    )
    assert result["valid_bouts"] == 0
    assert result["object_exploration_s"]["A"] == 0
    merged = [b for b in result["bout_debug"] if b["stage"] == "merged" and b["object_id"] == "A"]
    assert [b["component_ids"] for b in merged] == [["a1"], ["a2"]]
    assert merged[1]["merge_barrier_ids"] == ["grooming"]


def test_oft_dwell_boundary_occupancy_and_no_entry_policy():
    intervals = [
        {"zone": "center", "start_s": 4, "end_s": 4.999},
        {"zone": "periphery", "start_s": 4.999, "end_s": 6},
        {"zone": "center", "start_s": 6, "end_s": 7},
    ]
    result = oft_zone_intervals(intervals, first_detection_s=4, occupancy="qualified_only", no_entry="missing")
    assert result["center_entries"] == 1
    assert result["latency_to_first_center_entry_s"] == 2
    assert result["center_time_s"] == 1
    empty = oft_zone_intervals([], first_detection_s=4, occupancy="all_observed", no_entry="missing")
    assert empty["latency_to_first_center_entry_s"] is None


def test_historical_no_dwell_is_not_incompatible_but_missing_geometry_remains_unresolved():
    result = ymaze_provenance({"scoring_provenance": {"apply_dwell_filter": False, "minimum_dwell_seconds": 0}}, {})
    assert result["status"] == "unresolved"
    assert "no_dwell" not in result["reasons"]
    assert ymaze_provenance({"scoring_provenance": {"apply_dwell_filter": True}}, {})["status"].startswith(
        "requires_regeneration"
    )


def test_historical_mitigation_requires_event_audit_not_automatic_regeneration():
    result = ymaze_provenance(
        {"scoring_provenance": {"apply_dwell_filter": False, "minimum_dwell_seconds": 0, "zone_max_gap_seconds": 1}}, {}
    )
    assert result["status"] == "requires_event_set_audit"


def test_shared_rust_python_principal_entry_fixture():
    path = (
        Path(__file__).resolve().parents[3]
        / "rust/bekiper-behaviour/src/radial_arm/tests/golden/principal_entry_parity.json"
    )
    fixture = json.loads(path.read_text())
    geometry = {
        "version": fixture["geometry_version"],
        "canonical_vertices": True,
        "center": fixture["center"],
        "arms": fixture["arms"],
    }
    for case in fixture["cases"]:
        samples = [dict(zip(fixture["sample_columns"], values, strict=True)) for values in case["samples"]]
        for threshold in case["thresholds"]:
            result = ymaze_coordinate_entries(samples, geometry, threshold)
            expected = case["expected"][str(threshold)]
            assert result["valid_arm_entry_sequence"] == expected["sequence"], case["name"]
            for arm in "ABC":
                assert [r["sample_index"] for r in result["event_debug"] if r["arm"] == arm and r["entry"]] == expected[
                    "entry_frames"
                ][arm]
            if "arm_A_predicate" in expected:
                assert [r["predicate"] for r in result["event_debug"] if r["arm"] == "A"] == expected["arm_A_predicate"]


@pytest.mark.parametrize("object_id", ["A", "B"])
def test_overlapping_behaviour_exclusion_cannot_be_counted_as_exploration(object_id):
    with pytest.raises(ValueError, match="Overlapping behavioural exclusion"):
        score([segment("accepted", 0, 2, "A"), segment("grooming", 0.5, 1, object_id, behaviour="grooming")])


@pytest.mark.parametrize("split", [0.6, 1])
def test_oft_continuous_center_residence_is_invariant_to_fragmentation(split):
    result = oft_zone_intervals(
        [
            {"zone": "center", "start_s": 0, "end_s": split},
            {"zone": "center", "start_s": split, "end_s": 2},
        ],
        first_detection_s=0,
        occupancy="qualified_only",
        no_entry="missing",
    )
    assert result["center_entries"] == 1
    assert result["center_time_s"] == 2
    assert result["latency_to_first_center_entry_s"] == 0
    assert result["event_debug"][0]["component_indices"] == [0, 1]


def test_oft_tracking_gap_breaks_continuous_residence():
    result = oft_zone_intervals(
        [
            {"zone": "center", "start_s": 0, "end_s": 0.6},
            {"zone": "center", "start_s": 0.7, "end_s": 1.3},
        ],
        first_detection_s=0,
        occupancy="qualified_only",
        no_entry="missing",
    )
    assert result["center_entries"] == 0
