"""Timestamped event adapters; no videos, pose training or implicit interpolation."""

from __future__ import annotations

import math
from statistics import median

from .metrics import Record, nonnegative

GEOMETRY = "imen_radial_vertex_span_v1"
ENTRY_RULE = "dual_landmark_false_to_true_v1"


def reconstruction_window(row: Record, policy: Record | str | None) -> Record | None:
    """Only pre-windowed input is supported; never infer an origin or clip samples."""
    origin_field = "first_detection_s" if row["assay"] == "ymaze" else "analysis_start_s"
    expected = {
        "interval": "half_open",
        "origin": origin_field,
        "duration": "analysis_duration_s",
        "out_of_window": "reject",
    }
    if row["assay"] == "ymaze":
        expected["missing_pose"] = "observed_samples_break_on_missing_no_interpolation"
    else:
        expected["protocol_departures"] = "flag_only"
    if not isinstance(policy, dict) or policy != expected:
        return None
    if row.get(origin_field) is None or row.get("analysis_duration_s") is None:
        return None
    origin = nonnegative(row[origin_field], origin_field)
    duration = nonnegative(row["analysis_duration_s"], "analysis_duration_s")
    if duration <= 0 or not math.isfinite(origin + duration):
        raise ValueError("Reconstruction requires a finite positive session window")
    return {"start_s": origin, "end_s": origin + duration, "policy": policy}


def validate_window(items: list[Record], window: Record, *, intervals: bool) -> None:
    for item in items:
        start = nonnegative(item["start_s" if intervals else "timestamp_s"], "window timestamp")
        end = nonnegative(item["end_s"], "window end") if intervals else start
        if start < window["start_s"] or start >= window["end_s"] or end > window["end_s"]:
            raise ValueError("Timestamp/interval outside declared half-open session window")


def nort_object_roles(row: Record, mapping: Record) -> Record | None:
    from .inputs import KEY, session_key

    if not isinstance(mapping, dict):
        raise TypeError("NORT object roles must be a mapping")
    sessions = mapping.get("sessions", [])
    if not isinstance(sessions, list):
        raise TypeError("NORT object roles sessions must be a list")
    indexed = {}
    for assignment in sessions:
        key = session_key(assignment)
        if key in indexed:
            raise ValueError("Duplicate NORT object roles session assignment")
        indexed[key] = assignment
    assignment = indexed.get(tuple(str(row[k]) for k in KEY))
    roles = assignment.get("roles") if assignment is not None else mapping.get(row["phase"])
    if roles is None:
        return None
    fields = {"TA", "TB"} if row["phase"] == "training" else {"TN", "TF"}
    if (
        not isinstance(roles, dict)
        or set(roles) != fields
        or any(not isinstance(v, str) or not v.strip() for v in roles.values())
        or len(set(roles.values())) != 2
    ):
        raise ValueError("NORT object roles require exact, distinct nonempty phase assignments")
    configuration = row.get("object_configuration")
    if not configuration:
        return None
    if assignment and "object_configuration" in assignment and assignment["object_configuration"] != configuration:
        raise ValueError("NORT object roles conflict with session object configuration")
    if isinstance(configuration, dict):
        objects = configuration.get("objects", {})
        if not isinstance(objects, dict) or not set(roles.values()) <= set(objects):
            raise ValueError("NORT object roles reference absent configuration objects")
    return dict(roles)


def polygon_contains(x: float, y: float, vertices: list[list[float]]) -> bool:
    """Same ray-casting inequalities as bekiper-math::confinement."""
    inside = False
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def validate_polygon(vertices: list[list[float]]) -> None:
    if len(vertices) < 3 or len({tuple(v) for v in vertices}) != len(vertices):
        raise ValueError("Geometry requires unique canonical polygon vertices")
    if any(len(v) != 2 or any(not math.isfinite(n) for n in v) for v in vertices):
        raise ValueError("Geometry has nonfinite or invalid coordinates")
    area = sum(
        vertices[i][0] * vertices[(i + 1) % len(vertices)][1] - vertices[(i + 1) % len(vertices)][0] * vertices[i][1]
        for i in range(len(vertices))
    )
    if area == 0:
        raise ValueError("Degenerate polygon")


def ymaze_coordinate_entries(samples: list[Record], geometry: Record, threshold_pct: int) -> Record:
    """Authoritative principal geometry + instantaneous second-landmark crossing."""
    if geometry.get("version") != GEOMETRY or geometry.get("canonical_vertices") is not True:
        raise ValueError("Y-maze geometry needs explicit canonical imen_radial_vertex_span_v1 vertices")
    if threshold_pct not in [25, 45, 65] or set(geometry["arms"]) != set("ABC"):
        raise ValueError("Y-maze requires 25/45/65 and three explicitly labelled arms")
    validate_polygon(geometry["center"])
    for vertices in geometry["arms"].values():
        validate_polygon(vertices)
    cx = sum(v[0] for v in geometry["center"]) / len(geometry["center"])
    cy = sum(v[1] for v in geometry["center"]) / len(geometry["center"])
    radii = {}
    for arm, vertices in geometry["arms"].items():
        distances = [math.hypot(v[0] - cx, v[1] - cy) for v in vertices]
        radii[arm] = min(distances) + threshold_pct / 100 * (max(distances) - min(distances))
    debug = []
    sequence = []
    previous = dict.fromkeys("ABC", False)
    last_time = -math.inf
    for i, sample in enumerate(samples):
        time = nonnegative(sample["timestamp_s"], "timestamp_s")
        if time <= last_time:
            raise ValueError("Nonmonotonic timestamps")
        last_time = time
        coordinates = [sample.get(key) for key in ["head_center_x", "head_center_y", "tail_base_x", "tail_base_y"]]
        present = all(v is not None and isinstance(v, (int, float)) and math.isfinite(v) for v in coordinates)
        states = {}
        for arm, vertices in geometry["arms"].items():
            head = tail = False
            if present:
                hx, hy, tx, ty = [float(v) for v in coordinates if v is not None]
                head = polygon_contains(hx, hy, vertices) and (hx - cx) ** 2 + (hy - cy) ** 2 >= radii[arm] ** 2
                tail = polygon_contains(tx, ty, vertices) and (tx - cx) ** 2 + (ty - cy) ** 2 >= radii[arm] ** 2
            state = head and tail
            states[arm] = state
            debug.append(
                {
                    "sample_index": i,
                    "timestamp_s": time,
                    "arm": arm,
                    "head_past": head,
                    "tail_past": tail,
                    "predicate": state,
                    "entry": state and not previous[arm],
                    "reason": None if present else "tracking_missing",
                    "threshold_pct": threshold_pct,
                    "geometry_version": GEOMETRY,
                    "entry_rule": ENTRY_RULE,
                }
            )
        if sum(states.values()) > 1:
            raise ValueError("Overlapping arm geometry: simultaneous arm membership")
        for arm, state in states.items():
            if state and not previous[arm]:
                sequence.append(arm)
        previous = states
    return {
        "valid_arm_entry_sequence": sequence,
        "event_debug": debug,
        "threshold_radii": radii,
        "geometry_version": GEOMETRY,
        "entry_rule": ENTRY_RULE,
    }


def ymaze_provenance(row: Record, landmark_roles: Record) -> Record:
    """Classify evidence, never bless historical exports solely from no dwell."""
    evidence = row.get("scoring_provenance") or {}
    if not isinstance(evidence, dict):
        raise TypeError("scoring_provenance must be an object")
    if not evidence:
        return {"status": "unresolved", "reasons": ["missing_scoring_provenance"]}
    incompatible = []
    if evidence.get("geometry_version") not in [None, GEOMETRY]:
        incompatible.append("different_geometry")
    if evidence.get("apply_dwell_filter") is True or evidence.get("minimum_dwell_seconds", 0) > 0:
        incompatible.append("incompatible_dwell_filter")
    if evidence.get("requires_both_landmarks_same_arm") is False:
        incompatible.append("single_landmark_or_proxy")
    mitigated = (
        evidence.get("arm_membership_hysteresis") is True
        or evidence.get("zone_max_gap_seconds", 0) > 0
        or evidence.get("pose_gap_fill_seconds", 0) > 0
    )
    fraction = evidence.get("proximal_threshold")
    if fraction is not None and fraction * 100 != row.get("threshold_pct"):
        incompatible.append("threshold_identity_mismatch")
    if incompatible:
        return {"status": "requires_regeneration_or_registered_amendment", "reasons": incompatible}
    if mitigated:
        return {"status": "requires_event_set_audit", "reasons": ["historical_temporal_mitigation"]}
    checks = {
        "geometry": evidence.get("geometry_version") == GEOMETRY,
        "dual_landmarks": evidence.get("requires_both_landmarks_same_arm") is True,
        "no_dwell": evidence.get("apply_dwell_filter") is False and evidence.get("minimum_dwell_seconds") == 0,
        "landmark_roles": complete_landmark_roles(landmark_roles)
        and evidence.get("head_columns") == landmark_roles.get("head_center")
        and evidence.get("tail_columns") == landmark_roles.get("tail_base"),
        "version": bool(
            row.get("upstream_revision")
            or evidence.get("code_revision")
            or evidence.get("entry_implementation_version")
        ),
        "no_temporal_mitigation": (
            evidence.get("arm_membership_hysteresis") is False
            and evidence.get("zone_max_gap_seconds") == 0
            and evidence.get("pose_gap_fill_seconds") == 0
        ),
        "transition_rule": evidence.get("entry_rule") == ENTRY_RULE,
    }
    return {
        "status": "verified_clarified_entry_definition" if all(checks.values()) else "unresolved",
        "reasons": [key for key, passed in checks.items() if not passed],
        "evidence": evidence,
    }


def complete_landmark_roles(roles: Record) -> bool:
    if not isinstance(roles, dict):
        return False
    columns: list[str] = []
    for name in ["head_center", "tail_base"]:
        mapping = roles.get(name)
        if not isinstance(mapping, dict) or set(mapping) != {"x", "y"}:
            return False
        if any(not isinstance(v, str) or not v.strip() for v in mapping.values()):
            return False
        columns.extend(mapping.values())
    return len(set(columns)) == 4


def assay_provenance(row: Record, config: Record) -> Record:
    """Compare exported definitions to a documented audit and byte-pinned sources.

    This is evidence of rule definitions, not an independent event-set reanalysis.
    Candidate geometry and behavioural annotations still originate upstream.
    """
    if row["assay"] == "oft":
        expected: Record = {
            "arena_cm": [40, 40, 34],
            "center_cm": [24, 24],
            "border_cm": 8,
            "entry_landmark": "body_centroid",
            "minimum_dwell_seconds": 1,
            "time_origin": "first_valid_detection",
            "analysis_duration_s": 300,
        }
    else:
        expected = {
            "zone_extension_m": 0.03,
            "body_centroid_outside_object_mask": True,
            "candidate_orientation_or_contact": True,
            "angle_aggregation": "bout_median",
            "angle_max_degrees": 45,
            "merge_gap_seconds": 0.5,
            "merge_gap_comparison": "<",
            "merge_before_duration": True,
            "minimum_dwell_seconds": 1,
            "median_nose_speed_max_m_per_s": 0.05,
            "speed_comparison": "<",
            "behaviour_exclusions": ["climbing", "sitting_on_object", "grooming", "rearing"],
            "object_directed_exception": True,
        }
    evidence = row.get("scoring_provenance") or {}
    audit = row.get("_rule_audit") or {}
    if not isinstance(evidence, dict) or not isinstance(audit, dict):
        raise TypeError("Rule provenance and audit must be objects")
    approved_origin = audit.get("origin") in ["existing_analysis_plan", "researcher_decision"] or (
        audit.get("origin") == "development_fixture"
        and config.get("run_mode") == "synthetic"
        and config.get("synthetic_fixture") is True
    )
    audit_valid = (
        approved_origin
        and bool(audit.get("approval_or_plan_reference"))
        and bool(audit.get("rationale"))
        and isinstance(audit.get("rules"), dict)
        and bool(audit.get("code_revision"))
    )
    rules = audit.get("rules")
    audited_rules = rules if isinstance(rules, dict) else {}
    checks = {
        "rule_audit": bool(audit_valid),
        "version": bool(evidence.get("code_revision")) and evidence.get("code_revision") == audit.get("code_revision"),
    }
    for name, expected_value in expected.items():
        checks[name] = evidence.get(name) == expected_value and audited_rules.get(name) == expected_value
        if isinstance(expected_value, bool):
            checks[name] = evidence.get(name) is expected_value and audited_rules.get(name) is expected_value
    files = row.get("_rule_evidence_files", [])
    available_roles = {f["role"] for f in files}
    required_roles = {"rule_audit", "implementation", "geometry", "landmarks", "window"}
    if row["assay"] == "nort":
        required_roles.add("annotations")
    checks["supporting_sources"] = required_roles <= available_roles and all(
        isinstance(f.get("sha256"), str)
        and len(f["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in f["sha256"])
        and isinstance(f.get("bytes"), int)
        and f["bytes"] > 0
        and bool(f.get("name"))
        for f in files
    )
    return {
        "status": "verified_source_rule_definition" if all(checks.values()) else "upstream_unverified",
        "reasons": [name for name, passed in checks.items() if not passed],
        "level": "documented_rule_audit_not_event_set_audit",
        "checks": checks,
        "evidence_files": files,
        "audit_reference": audit.get("approval_or_plan_reference"),
        "evidence": evidence,
    }


def nort_bouts(segments: list[Record], *, gap_duration: str, gap_medians: str) -> Record:
    """Candidates already satisfy geometry/contact rules; median filters follow merging.

    Segments use explicit [start_s,end_s) intervals and observed-sample arrays.
    This adapter does not assert that upstream candidate geometry was verified.
    """
    if gap_duration not in ["include_gaps", "exclude_gaps"] or gap_medians != "observed_samples_only":
        raise ValueError("D07 requires supported explicit duration and median gap accounting")
    raw = []
    groups: dict[str, list[Record]] = {}
    identifiers = set()
    for segment in segments:
        start = nonnegative(segment["start_s"], "start_s")
        end = nonnegative(segment["end_s"], "end_s")
        if end <= start or segment["segment_id"] in identifiers:
            raise ValueError("Invalid/duplicate candidate interval")
        identifiers.add(segment["segment_id"])
        if not isinstance(segment.get("candidate_eligible"), bool):
            raise TypeError("Candidate eligibility must be explicitly supplied as a boolean")
        reasons = []
        if not segment["candidate_eligible"]:
            reasons.append("upstream_candidate_rejection")
        if (
            segment.get("behaviour") in ["climbing", "sitting_on_object", "grooming", "rearing"]
            and segment.get("object_directed") is not True
        ):
            reasons.append("behaviour_not_object_directed")
        for name, maximum in [("angles_degrees", 180), ("nose_speeds_m_per_s", None)]:
            if not segment.get(name):
                raise ValueError("Bout medians require observed samples, not medians-of-medians")
            for value in segment[name]:
                nonnegative(value, name, maximum)
        raw.append({**segment, "stage": "candidate", "rejection_reasons": reasons, "valid": not reasons})
        # Rejected behavioural intervals remain barriers; never bridge them as distractions.
        groups.setdefault(segment["object_id"], []).append({**segment, "rejected": bool(reasons)})
    merged_debug = []
    behavioural_barriers = [s for s in raw if "behaviour_not_object_directed" in s["rejection_reasons"]]
    accepted = sorted([s for s in raw if s["valid"]], key=lambda s: s["start_s"])
    for candidate in accepted:
        if any(b["start_s"] < candidate["end_s"] and b["end_s"] > candidate["start_s"] for b in behavioural_barriers):
            raise ValueError(
                "Overlapping behavioural exclusion and accepted candidate require upstream resolution (D07)"
            )
    for i, first in enumerate(accepted):
        for second in accepted[i + 1 :]:
            if second["start_s"] >= first["end_s"]:
                break
            if second["object_id"] != first["object_id"]:
                raise ValueError("Simultaneous different-object candidates need upstream registered arbitration (D07)")
    times: dict[str, float] = {}
    for object_id, candidates in groups.items():
        candidates.sort(key=lambda s: s["start_s"])
        batches: list[Record] = []
        prior_end = -math.inf
        pending_barriers = []
        for segment in candidates:
            if segment["start_s"] < prior_end:
                raise ValueError("Overlapping same-object candidate intervals")
            prior_end = segment["end_s"]
            if segment["rejected"]:
                pending_barriers.append(segment["segment_id"])
                continue
            previous_end = batches[-1]["components"][-1]["end_s"] if batches else -math.inf
            barrier_ids = sorted(
                set(
                    pending_barriers
                    + [
                        b["segment_id"]
                        for b in behavioural_barriers
                        if b["start_s"] < segment["start_s"] and b["end_s"] > previous_end
                    ]
                )
            )
            if batches and not barrier_ids and segment["start_s"] - previous_end < 0.5:
                batches[-1]["components"].append(segment)
            else:
                batches.append({"components": [segment], "merge_barrier_ids": barrier_ids})
            pending_barriers = []
        times[object_id] = 0.0  # object is observed, unlike an absent object
        for merged in batches:
            batch = merged["components"]
            observed = sum(s["end_s"] - s["start_s"] for s in batch)
            span = batch[-1]["end_s"] - batch[0]["start_s"]
            duration = span if gap_duration == "include_gaps" else observed
            angle = median([a for s in batch for a in s["angles_degrees"]])
            speed = median([v for s in batch for v in s["nose_speeds_m_per_s"]])
            reasons = [
                name
                for name, fails in [("duration", duration < 1.0), ("angle", angle > 45), ("speed", speed >= 0.05)]
                if fails
            ]
            merged_debug.append(
                {
                    "stage": "merged",
                    "object_id": object_id,
                    "component_ids": [s["segment_id"] for s in batch],
                    "merge_barrier_ids": merged["merge_barrier_ids"],
                    "start_s": batch[0]["start_s"],
                    "end_s": batch[-1]["end_s"],
                    "duration_s": duration,
                    "observed_duration_s": observed,
                    "gap_s": span - observed,
                    "median_angle_degrees": angle,
                    "median_nose_speed_m_per_s": speed,
                    "gap_duration_policy": gap_duration,
                    "gap_median_policy": gap_medians,
                    "rejection_reasons": reasons,
                    "valid": not reasons,
                }
            )
            if not reasons:
                times[object_id] += duration
    return {
        "object_exploration_s": times,
        "bout_debug": raw + merged_debug,
        "valid_bouts": sum(b["valid"] for b in merged_debug),
        "removed_duration_bouts": sum("duration" in b["rejection_reasons"] for b in merged_debug),
        "removed_speed_bouts": sum("speed" in b["rejection_reasons"] for b in merged_debug),
        "removed_angle_bouts": sum("angle" in b["rejection_reasons"] for b in merged_debug),
    }


def oft_zone_intervals(intervals: list[Record], *, first_detection_s: float, occupancy: str, no_entry: str) -> Record:
    """Already classified centroid intervals with explicit [start,end) semantics."""
    if occupancy not in ["all_observed", "qualified_only"] or no_entry not in ["missing", "right_censored_300"]:
        raise ValueError("D05 requires explicit occupancy and no-entry policy")
    origin = nonnegative(first_detection_s, "first_detection_s")
    prior_end = origin
    runs: list[Record] = []
    for index, interval in enumerate(intervals):
        start = nonnegative(interval["start_s"], "start_s")
        end = nonnegative(interval["end_s"], "end_s")
        if end <= start or start < prior_end or start < origin or end > origin + 300:
            raise ValueError("Invalid/overlapping OFT interval or interval outside first-detection 300s window")
        if interval["zone"] not in ["center", "periphery"]:
            raise ValueError("Unknown OFT zone")
        prior_end = end
        if runs and runs[-1]["zone"] == interval["zone"] and runs[-1]["end_s"] == start:
            runs[-1]["end_s"] = end
            runs[-1]["component_indices"].append(index)
            runs[-1]["components"].append(dict(interval))
        else:
            runs.append({**interval, "component_indices": [index], "components": [dict(interval)]})
    debug = [
        {
            **run,
            "duration_s": run["end_s"] - run["start_s"],
            "valid_entry": run["end_s"] - run["start_s"] >= 1,
            "rejection_reason": None if run["end_s"] - run["start_s"] >= 1 else "short_dwell",
        }
        for run in runs
    ]
    centers = [i for i in debug if i["zone"] == "center"]
    entries = [i for i in centers if i["valid_entry"]]
    result = {
        "center_time_s": sum(i["duration_s"] for i in centers if occupancy == "all_observed" or i["valid_entry"]),
        "center_entries": len(entries),
        "latency_to_first_center_entry_s": entries[0]["start_s"] - origin if entries else None,
        "event_debug": debug,
        "observed_zone_time_s": sum(i["duration_s"] for i in debug),
    }
    if not entries:
        result["missing_reasons"] = {"latency_to_first_center_entry_s": "not_observed:no_center_entry"}
        if no_entry == "right_censored_300":
            result["latency_to_first_center_entry_s"] = 300
            result["latency_right_censored"] = True
            result["missing_reasons"] = {}
    return result
