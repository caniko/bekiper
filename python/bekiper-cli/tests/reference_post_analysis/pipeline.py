"""Input -> metrics -> inclusion -> registered inference -> auditable artifacts."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .events import (
    assay_provenance,
    nort_bouts,
    nort_object_roles,
    oft_zone_intervals,
    reconstruction_window,
    validate_window,
    ymaze_coordinate_entries,
    ymaze_provenance,
)
from .inputs import (
    KEY,
    choice,
    fingerprint,
    load_inputs,
    requested_output_ids,
    resolved_config,
    session_key,
    subgroup_plan,
)
from .metrics import (
    COUNT_FIELDS,
    OFT_PRIMARY,
    OFT_SECONDARY,
    YM_SCORES,
    Record,
    missing,
    nonnegative,
    nort_scores,
    oft_scores,
    ymaze_scores,
)
from .statistics import adjust_family, fit_factorial, pairing, spearman_pairs, within_group


def record_computed(row: Record, values: Record, source: Record, *, kind: str = "derived") -> None:
    """Keep imported evidence even when reconstruction differs or is unavailable."""
    reasons = values.get("missing_reasons", {})
    for field, value in list(values.items()):
        if field in ["missing_reasons", "disagreements"]:
            continue
        supplied = source.get(field)
        reason = reasons.get(field)
        if (
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and isinstance(supplied, (float, int))
            and not isinstance(supplied, bool)
        ):
            equal = math.isclose(value, supplied, abs_tol=1e-8)
        else:
            equal = value == supplied
        if supplied is not None and not equal:
            row[f"upstream_{field}"] = supplied
            row[f"recomputed_{field}"] = value
            if field not in row.setdefault("disagreements", []):
                row["disagreements"].append(field)
        selected_kind = kind
        # An unavailable recomputation is not evidence that a supplied value is
        # missing. Undefined results and unapproved decisions still withhold it.
        if value is None and supplied is not None and reason and reason.startswith("not_measured"):
            value = supplied
            selected_kind = "upstream"
            row.setdefault("recomputation_missing_reasons", {})[field] = reason
            reason = None
        row[field] = value
        row.setdefault("metric_sources", {})[field] = selected_kind
        if reason:
            row.setdefault("missing_reasons", {})[field] = reason
        else:
            row.setdefault("missing_reasons", {}).pop(field, None)


def score_records(rows: list[Record], config: Record) -> tuple[list[Record], list[Record]]:
    scored, debug = [], []
    for source in rows:
        session_key(source)
        row = dict(source)
        for field in ["missing_reasons", "metric_sources", "recomputation_missing_reasons"]:
            row[field] = dict(source.get(field, {}))
        row["disagreements"] = list(source.get("disagreements", []))
        if "blocked_outputs" in source:
            row["blocked_outputs"] = list(source["blocked_outputs"])
        assay = row["assay"]
        for field, value in source.items():
            if value is not None and (field.endswith(("_s", "_m", "_m_per_s")) or field in COUNT_FIELDS):
                nonnegative(value, field)
                if field in COUNT_FIELDS and int(value) != value:
                    raise ValueError(f"{field}: requires an integer count")
        for field in ["DI", "RI", "percent_novel", "object_bias", "side_bias"] + YM_SCORES:
            value = source.get(field)
            if value is not None:
                lower, upper = (
                    (-1, 1)
                    if field in ["DI", "object_bias", "side_bias"]
                    else (0, 100 if field == "percent_novel" or field in YM_SCORES else 1)
                )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not lower <= value <= upper
                ):
                    raise ValueError(f"Impossible supplied {field}")
        units = config["inputs"].get("units_and_calibration", {}).get(assay, {})
        if source["measurement_available"] and any(
            units.get(k) != v for k, v in {"time": "s", "distance": "m", "speed": "m/s"}.items()
        ):
            raise ValueError(f"{assay}: explicit SI time/distance/speed units required; conversion is never inferred")
        flags: Record = {}
        row_debug = []
        verification: Record = assay_provenance(row, config) if assay != "ymaze" else {}
        if assay == "oft":
            if "center_intervals" in row:
                occupancy = choice(config, "assays.oft.center_occupancy_policy", "D05")
                latency = choice(config, "assays.oft.no_center_entry_latency_policy", "D05")
                window = choice(config, "assays.oft.window_and_tracking_policy", "D04")
                start_policy = choice(config, "assays.oft.starting_in_center_policy", "D05")
                timestamp_policy = choice(config, "assays.oft.entry_timestamp_policy", "D04")
                if (
                    occupancy
                    and latency
                    and window == "half_open_300s_from_first_detection"
                    and start_policy == "count_as_entry_at_origin"
                    and timestamp_policy == "initial_crossing"
                ):
                    events = oft_zone_intervals(
                        row["center_intervals"],
                        first_detection_s=row["first_detection_s"],
                        occupancy=occupancy,
                        no_entry=latency,
                    )
                    row_debug = events.pop("event_debug")
                    record_computed(row, events, source, kind="reconstructed")
                else:
                    for field in ["center_time_s", "center_entries", "latency_to_first_center_entry_s"]:
                        missing(row, field, "blocked:D04/D05:interval_policy")
            metrics = oft_scores(row)
            reconstructed_sources = dict(row["metric_sources"])
            record_computed(row, metrics, source, kind="upstream")
            row["metric_sources"].update({k: v for k, v in reconstructed_sources.items() if k in metrics})
            if row.get("center_time_s") is not None:
                row["metric_sources"]["center_time_pct"] = "derived"
            duration = row.get("analysis_duration_s")
            flags["incomplete_recording"] = duration < 300 if duration is not None else None
            flags["tracking_incomplete"] = row["observed_zone_time_s"] < 300 if "observed_zone_time_s" in row else None
        elif assay == "ymaze":
            reconstructed = False
            for field, expected in [
                ("threshold_geometry", "imen_radial_vertex_span_v1"),
                ("principal_entry_rule", "dual_landmark_false_to_true_v1"),
                ("principal_minimum_dwell_s", 0),
            ]:
                if config["assays"]["ymaze"].get(field, expected) != expected:
                    raise ValueError("Principal Y-maze config conflicts with registered researcher amendment")
            if "coordinate_samples" in row:
                window_policy = choice(config, "assays.ymaze.window_and_tracking_policy", "D04")
                window = reconstruction_window(row, window_policy)
                if window:
                    validate_window(row["coordinate_samples"], window, intervals=False)
                    events = ymaze_coordinate_entries(row["coordinate_samples"], row["geometry"], row["threshold_pct"])
                    row_debug = events.pop("event_debug")
                    events.update(
                        raw_arm_entry_sequence=list(events["valid_arm_entry_sequence"]), removed_short_entries=0
                    )
                    record_computed(row, events, source, kind="reconstructed")
                    reconstructed = True
                    row["reconstruction_window"] = window
                    verification = {
                        "status": "verified_clarified_entry_definition",
                        "reasons": [],
                        "source_level": "explicit_head_center_tail_base_samples",
                    }
                else:
                    row["blocked_outputs"] = [*row.get("blocked_outputs", []), "ymaze_coordinate_reconstruction:D04"]
            if not reconstructed:
                roles = config["inputs"].get("upstream_rule_provenance", {}).get("ymaze", {}).get("landmark_roles", {})
                verification = ymaze_provenance(row, roles)
            policy = choice(config, "assays.ymaze.triad_policy", "D01")
            short = choice(config, "assays.ymaze.fewer_than_three_entries_policy", "D02")
            if row.get("valid_arm_entry_sequence") is not None:
                metrics = ymaze_scores(
                    row["valid_arm_entry_sequence"],
                    policy=policy,
                    short_policy=short,
                    mapping=choice(config, "assays.ymaze.custom_triad_mapping", "D01"),
                )
            else:
                metrics = {"missing_reasons": {}}
                for field in YM_SCORES + ["valid_total_arm_entries", "usable_triads", "low_quality"]:
                    missing(
                        metrics,
                        field,
                        "blocked:D01"
                        if policy is None and field != "valid_total_arm_entries"
                        else "not_measured:valid_sequence",
                    )
            primary = choice(config, "assays.ymaze.primary_threshold_arm_length_pct", "D03")
            if primary is not None and primary not in [25, 45, 65]:
                raise ValueError("Registered primary Y-maze threshold must be 25, 45 or 65")
            metrics["primary_threshold"] = row.get("threshold_pct") == primary if primary is not None else None
            if primary is None:
                metrics["missing_reasons"]["primary_threshold"] = "blocked:D03"
            for field in [
                "raw_total_arm_entries",
                "removed_short_entries",
                "arm_preference",
                "total_distance_m",
                "mean_speed_m_per_s",
                "freezing_s",
            ]:
                if row.get(field) is None:
                    missing(metrics, field, "not_measured")
            if row.get("raw_arm_entry_sequence") is not None:
                metrics["raw_total_arm_entries"] = len(row["raw_arm_entry_sequence"])
                metrics["missing_reasons"].pop("raw_total_arm_entries", None)
            # A raw-valid difference is not proof all removals were short dwells.
            if (
                row.get("removed_short_entries") is not None
                and verification.get("status") != "verified_clarified_entry_definition"
            ):
                flags["removal_reason_unverified"] = True
            record_computed(row, metrics, source)
            if (
                policy is not None
                and row.get("valid_total_arm_entries") is not None
                and row.get("usable_triads") is not None
            ):
                row["low_quality"] = row["valid_total_arm_entries"] < 8 or row["usable_triads"] < 5
                row["missing_reasons"].pop("low_quality", None)
                row["metric_sources"]["low_quality"] = "derived"
            flags["low_quality"] = row.get("low_quality")
        else:
            if "candidate_segments" in row:
                gap = choice(config, "assays.nort.merged_gap_duration_policy", "D07")
                median_policy = choice(config, "assays.nort.merged_gap_median_policy", "D07")
                role_mapping = choice(config, "assays.nort.object_role_and_location_mapping", "D10")
                roles = nort_object_roles(row, role_mapping) if role_mapping else None
                window_policy = choice(config, "assays.nort.window_and_tracking_policy", "D14")
                window = reconstruction_window(row, window_policy)
                if not window:
                    row.setdefault("blocked_outputs", []).append("nort_bout_reconstruction:D14:window_policy")
                if gap and median_policy and roles and window:
                    validate_window(row["candidate_segments"], window, intervals=True)
                    if any(s["object_id"] not in roles.values() for s in row["candidate_segments"]):
                        raise ValueError("NORT object roles do not cover candidate object identities")
                    events = nort_bouts(row["candidate_segments"], gap_duration=gap, gap_medians=median_policy)
                    row["resolved_object_roles"] = roles
                    row["reconstruction_window"] = window
                    row_debug = events.pop("bout_debug")
                    for field, object_id in roles.items():
                        value = events["object_exploration_s"].get(object_id)
                        record_computed(
                            row,
                            {
                                field: value,
                                "missing_reasons": {field: "not_measured:object_candidates"} if value is None else {},
                            },
                            source,
                            kind="reconstructed",
                        )
                    record_computed(
                        row,
                        {k: v for k, v in events.items() if k != "object_exploration_s"},
                        source,
                        kind="reconstructed",
                    )
                elif not (gap and median_policy and roles):
                    row.setdefault("blocked_outputs", []).append("nort_bout_reconstruction:D07/D10")
            if row["phase"] == "training":
                row.setdefault("TA", row.get("object_A_exploration_s"))
                row.setdefault("TB", row.get("object_B_exploration_s"))
            metrics = nort_scores(**{k: row.get(k) for k in ["TA", "TB", "TN", "TF"]})
            pseudocount = config["assays"]["nort"].get("lr_pseudocount", {})
            if pseudocount.get("enabled") and row["phase"] == "test":
                approved = choice(config, "assays.nort.lr_pseudocount", "D13")
                if approved and approved.get("units") == "s" and approved.get("formula") == "ln((TN+p)/(TF+p))":
                    p = nonnegative(approved["value"], "lr_pseudocount")
                    if p <= 0:
                        raise ValueError("Registered LR pseudocount must be positive")
                    if row.get("TN") is not None and row.get("TF") is not None:
                        metrics["LR"] = math.log((row["TN"] + p) / (row["TF"] + p))
                        metrics["missing_reasons"].pop("LR", None)
                else:
                    missing(metrics, "LR", "blocked:D13:pseudocount_definition")
            phase_fields = (
                ["TA", "TB", "total_training_exploration", "object_bias", "side_bias"]
                if row["phase"] == "training"
                else ["TN", "TF", "total_test_exploration", "DI", "RI", "percent_novel", "LR"]
            )
            for field in list(metrics):
                if field not in phase_fields and field != "missing_reasons":
                    metrics.pop(field)
                    metrics["missing_reasons"].pop(field, None)
            components = ["TA", "TB"] if row["phase"] == "training" else ["TN", "TF"]
            component_sources = {field: row["metric_sources"].get(field, "upstream") for field in components}
            record_computed(row, metrics, source)
            row["metric_sources"].update(component_sources)
            definition = choice(config, "assays.nort.exploration_quality_flag", "D06")
            flag_name = "passes_15s_exploration_threshold"
            flags[flag_name] = None
            if definition and definition.get("phase") == row["phase"]:
                quantity = row.get(definition.get("quantity"))
                operator = definition.get("operator")
                if definition.get("threshold_s") != 15 or operator not in [">=", ">"]:
                    raise ValueError("NORT quality flag requires registered phase, quantity and >= or > 15s")
                if quantity is not None:
                    flags[flag_name] = quantity >= 15 if operator == ">=" else quantity > 15
            if flags[flag_name] is None:
                row["missing_reasons"][flag_name] = (
                    "blocked:D06"
                    if not definition
                    else "not_applicable:quality_flag_phase"
                    if definition.get("phase") != row["phase"]
                    else "not_measured:quality_quantity"
                )
            if row.get(flag_name) is not None:
                row["upstream_" + flag_name] = row[flag_name]
            if not row.get("object_configuration"):
                flags["object_configuration_unavailable"] = True
            protocol = row.get("protocol_metadata") or {}
            expected_protocol = {"familiarization_s": 360, "delay_h": 5, "test_s": 360}
            if not isinstance(protocol, dict):
                raise ValueError("NORT protocol_metadata must be an object")
            flags["protocol_timing_unverified"] = any(protocol.get(field) is None for field in expected_protocol)
            departures = []
            for field, expected in expected_protocol.items():
                if protocol.get(field) is not None and nonnegative(protocol[field], field) != expected:
                    departures.append(field)
            row["protocol_timing_departures"] = departures
            flags["protocol_timing_departure"] = bool(departures)
        compact = {
            k: v
            for k, v in row.items()
            if k
            not in [
                "coordinate_samples",
                "candidate_segments",
                "center_intervals",
                "comments",
                "All comments",
                "source_path",
                "video_path",
                "_rule_audit",
                "_rule_evidence_files",
            ]
        }
        if not compact["disagreements"]:
            compact.pop("disagreements")
        compact["qc_flags"] = flags
        compact["input_verification"] = verification
        scored.append(compact)
        debug.extend([{**{k: row[k] for k in KEY}, **event} for event in row_debug])
    return scored, debug


def failed_choices(config: Record, prerequisites: list[tuple[str, str, Any]]) -> list[str]:
    failed = []
    for path, decision, supported in prerequisites:
        value = choice(config, path, decision)
        if value is None or (supported is not None and value != supported):
            failed.append(f"{decision}:{path}")
    return failed


def run_models(rows: list[Record], config: Record) -> tuple[list[Record], list[Record], list[Record], list[Record]]:
    models: list[Record] = []
    pair_ledger: list[Record] = []
    correlations: list[Record] = []
    checks: list[Record] = []
    requested = set(requested_output_ids(config, rows))
    if not requested.intersection({"models", "correlations", "within_group_tests"}):
        return models, pair_ledger, correlations, checks
    inclusion_prerequisites = [
        ("inclusion.existing_metadata_exclusions_policy", "D12", "honor_metadata"),
        ("inclusion.qc_flag_exclusion_policy", "D12", "flags_only"),
        ("inclusion.missing_metric_policy", "D12", "endpoint_complete_cases"),
        ("inclusion.upstream_verification_requirement", "D12", None),
    ]
    requirement = choice(config, "inclusion.upstream_verification_requirement", "D12")
    if requirement not in [None, "allow_unverified", "require_verified"]:
        raise ValueError("Unknown upstream verification requirement")
    for assay, endpoints in [("oft", OFT_PRIMARY), ("ymaze", [YM_SCORES[0]]), ("nort", ["DI"])]:
        subset = [r for r in rows if r["assay"] == assay and (assay != "nort" or r["phase"] == "test")]
        if not subset:
            continue
        if assay == "ymaze":
            subset = [r for r in subset if r.get("primary_threshold") is True]
        has_pre = any(r["timepoint"] == "pre" for r in subset)
        for endpoint in endpoints:
            if "models" not in requested:
                continue
            model_id = f"{assay}.{endpoint}"
            baseline = has_pre and assay != "nort"
            if baseline:
                post, ledger = pairing(subset, endpoint, verification_requirement=requirement)
                pair_ledger.extend(ledger)
            else:
                post = [r for r in subset if r["timepoint"] == "post"]
            prerequisites = inclusion_prerequisites + [
                ("inputs.group_mapping", "D10", None),
                ("inputs.factor_levels", "D10", None),
                ("statistics.model_failure_policy", "D09", "report_without_simplification"),
            ]
            if baseline:
                prerequisites.append(("inclusion.unmatched_prepost_policy", "D12", "report_and_withhold_pair"))
            if baseline and assay == "oft":
                prerequisites.append(("statistics.oft_baseline_strategy", "D09", "baseline_adjusted"))
            if not baseline and assay == "ymaze":
                prerequisites.append(
                    (
                        "statistics.ymaze_post_only_formula",
                        "D09",
                        "post_SA ~ " + "5XFAD_status * HCAR1_status * treatment",
                    )
                )
            failed = failed_choices(config, prerequisites)
            if assay == "ymaze" and not subset:
                failed.append("D03:primary_threshold")
            fields = [
                "factor_coding",
                "covariance_estimator",
                "contrast_estimand_and_weighting",
                "test_tails",
                "confidence_interval_method",
                "confidence_level",
                "contrast_roles",
            ]
            settings = {field: choice(config, f"statistics.{field}", "D09") for field in fields}
            families = choice(config, "statistics.model_family_by_endpoint", "D09") or {}
            settings["model_family"] = families.get(model_id)
            settings["additional_covariates"] = []
            if assay == "nort":
                covariates = choice(config, "statistics.nort_additional_covariates", "D09")
                if covariates is None:
                    failed.append("D09:nort_predefined_covariates")
                else:
                    settings["additional_covariates"] = covariates
            if any(value is None for value in settings.values()):
                failed.append("D09:inference_settings")
            if failed:
                models.append(
                    {
                        "model_id": model_id,
                        "endpoint": endpoint,
                        "status": "blocked",
                        "reason": failed,
                        "n_animals": 0,
                        "contrast": None,
                        "subset": [],
                    }
                )
            else:
                result = fit_factorial(
                    post,
                    endpoint,
                    levels=config["inputs"]["factor_levels"],
                    groups=config["inputs"]["group_mapping"],
                    settings=settings,
                    baseline=baseline,
                    verification_requirement=requirement,
                )
                models.append({"model_id": model_id, **result})
        if assay == "ymaze" and "correlations" in requested:
            failed = failed_choices(
                config,
                inclusion_prerequisites
                + [
                    (
                        "statistics.spearman_missing_and_constant_input_policy",
                        "D09",
                        "pairwise_complete_constant_missing",
                    ),
                    ("statistics.test_tails", "D09", "two-sided"),
                ],
            )
            post = [r for r in subset if r["timepoint"] == "post"]
            if not subset:
                failed.append("D03:primary_threshold_or_no_ymaze_measurements")
            if not failed:
                correlations.extend(spearman_pairs(post, verification_requirement=requirement))
            else:
                correlations.append({"status": "blocked", "reason": failed, "n_animals": 0, "p_raw": None})
        if assay == "nort" and "within_group_tests" in requested:
            failed = failed_choices(
                config,
                inclusion_prerequisites
                + [
                    ("inputs.group_mapping", "D10", None),
                    ("statistics.nort_within_group_checks", "D09", None),
                    ("statistics.nort_within_group_test_method", "D09", "one_sample_t"),
                    ("statistics.test_tails", "D09", "two-sided"),
                    ("statistics.confidence_interval_method", "D09", "student_t"),
                    ("statistics.confidence_level", "D09", None),
                    ("statistics.multiplicity", "D09", None),
                ],
            )
            registered = choice(config, "statistics.nort_within_group_checks", "D09")
            multiplicity = choice(config, "statistics.multiplicity", "D09")
            confidence = choice(config, "statistics.confidence_level", "D09")
            family = multiplicity.get("families", {}).get("nort_within_group") if multiplicity else None
            if not isinstance(family, str) or not family or multiplicity.get("method") != "holm":
                failed.append("D09:NORT_multiple_comparison_correction")
            if not failed:
                assert isinstance(family, str)
                post = [r for r in subset if r["timepoint"] == "post"]
                checks = within_group(post, registered, confidence, verification_requirement=requirement)
                adjust_family(checks, multiplicity["method"], family)
            else:
                checks = [{"status": "blocked", "reason": failed, "n_animals": 0, "p_raw": None}]
    multiplicity = choice(config, "statistics.multiplicity", "D09")
    contrasts = [model.get("contrast") or {"p_raw": None, "model_id": model["model_id"]} for model in models]
    if models:
        if multiplicity and multiplicity.get("families", {}).get("planned_contrasts"):
            adjust_family(contrasts, multiplicity["method"], multiplicity["families"]["planned_contrasts"])
        else:
            for contrast in contrasts:
                contrast.update(p_adjusted=None, adjustment_reason="blocked:D09:multiplicity_family")
    return models, pair_ledger, correlations, checks


def inclusion_ledger(rows: list[Record]) -> list[Record]:
    ledger = []
    for row in rows:
        fields = (
            OFT_PRIMARY + OFT_SECONDARY
            if row["assay"] == "oft"
            else YM_SCORES
            + [
                "valid_total_arm_entries",
                "usable_triads",
                "raw_total_arm_entries",
                "removed_short_entries",
                "arm_preference",
                "total_distance_m",
                "mean_speed_m_per_s",
                "freezing_s",
                "low_quality",
            ]
            if row["assay"] == "ymaze"
            else [
                "TA",
                "TB",
                "total_training_exploration",
                "object_bias",
                "side_bias",
                "passes_15s_exploration_threshold",
            ]
            if row["phase"] == "training"
            else ["TN", "TF", "DI", "RI", "percent_novel", "LR", "total_test_exploration"]
        )
        if row["assay"] == "nort" and row["phase"] == "test":
            fields.append("passes_15s_exploration_threshold")
        for endpoint in fields:
            reason = row.get("missing_reasons", {}).get(endpoint)
            value = row["qc_flags"].get(endpoint, row.get(endpoint))
            status = "available" if value is not None else "unavailable"
            if reason:
                status = (
                    "blocked"
                    if reason.startswith("blocked")
                    else "undefined"
                    if reason.startswith("undefined")
                    else "not_applicable"
                    if reason.startswith("not_applicable")
                    else "unavailable"
                )
            ledger.append(
                {
                    **{k: row[k] for k in KEY},
                    "threshold_pct": row.get("threshold_pct"),
                    "endpoint": endpoint,
                    "metric_status": status,
                    "missing_reason": reason,
                    "metadata_excluded": row["excluded"],
                    "exclusion_reason": row["exclusion_reason"],
                    "qc_flags": row["qc_flags"],
                    "metric_source": row.get("metric_sources", {}).get(
                        endpoint, "upstream" if value is not None else None
                    ),
                    "eligible_before_model_policy": status == "available" and not row["excluded"],
                }
            )
    return ledger


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def revision() -> Record:
    root = Path(__file__).resolve().parents[4]

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip()

    return {"revision": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def result_status(results: list[Record], success: str) -> Record:
    if not results:
        return {"status": "unavailable", "reason": "no_applicable_observations_or_outputs"}
    statuses = [item.get("status") for item in results]
    complete = all(status == success for status in statuses)
    if success == "fitted":
        complete = complete and all((item.get("contrast") or {}).get("p_adjusted") is not None for item in results)
    return {
        "status": "complete" if complete else "blocked" if all(s == "blocked" for s in statuses) else "partial",
        "attempts": len(results),
        "successful_attempts": sum(s == success for s in statuses),
        "reasons": sorted(
            {
                str(reason)
                for item in results
                for reason in [
                    item.get("reason"),
                    (item.get("contrast") or {}).get("reason"),
                    (item.get("contrast") or {}).get("adjustment_reason"),
                ]
                if reason
            }
        ),
    }


def run_subgroups(rows: list[Record], config: Record) -> tuple[list[Record], Record]:
    plan = subgroup_plan(config)
    if plan is None:
        return [], {
            "output": "subgroup_inference",
            "status": "blocked",
            "decision_ids": ["D09", "D12"],
            "reason": "Register statistics.subgroup_analysis_plan under both D09 and D12",
        }
    scoped_config = copy.deepcopy(config)
    scoped_config["outputs"]["requested_outputs"] = plan["outputs"]
    results = []
    for stratum in plan["strata"]:
        for level in stratum["levels"]:
            candidates = [r for r in rows if r["assay"] in stratum["assays"]]
            member_rows = [r for r in candidates if r.get(stratum["field"]) == level]
            models, pairs, correlations, checks = run_models(member_rows, scoped_config)
            membership = [
                {
                    **{k: r[k] for k in KEY},
                    "threshold_pct": r.get("threshold_pct"),
                    "reason": None
                    if r.get(stratum["field"]) == level
                    else "missing_stratum_metadata"
                    if r.get(stratum["field"]) is None
                    else "outside_registered_stratum",
                }
                for r in candidates
            ]
            results.append(
                {
                    "field": stratum["field"],
                    "level": level,
                    "assays": stratum["assays"],
                    "models": models,
                    "pairing_ledger": pairs,
                    "correlations": correlations,
                    "within_group_tests": checks,
                    "membership_ledger": membership,
                }
            )
    tests = [model.get("contrast") or {"p_raw": None} for item in results for model in item["models"]]
    tests += [test for item in results for key in ["correlations", "within_group_tests"] for test in item[key]]
    adjust_family(tests, plan["multiplicity"]["method"], plan["multiplicity"]["family"])
    for item in results:
        statuses = {
            key: result_status(item[key], "fitted" if key == "models" else "computed") for key in plan["outputs"]
        }
        complete = all(s["status"] == "complete" for s in statuses.values())
        item.update(
            output_statuses=statuses,
            status="complete"
            if complete
            else "partial"
            if any(s["successful_attempts"] for s in statuses.values() if "successful_attempts" in s)
            else "blocked",
            reason=None if complete else {key: s for key, s in statuses.items() if s["status"] != "complete"},
        )
    return results, {"output": "subgroup_inference", **result_status(results, "complete"), "plan": plan}


def run(
    config_path: Path,
    *,
    output: Path | None = None,
    requested_outputs: list[str] | None = None,
    sensitivity_thresholds: list[int] | None = None,
    env: dict[str, str] | None = None,
) -> Record:
    from .plots import make_figures

    config = resolved_config(
        config_path,
        output=output,
        requested_outputs=requested_outputs,
        sensitivity_thresholds=sensitivity_thresholds,
        env=env,
    )
    directory = Path(config["outputs"]["directory"])
    if not directory.is_absolute():
        directory = config_path.parent / directory
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a fresh output directory to preserve reproducibility/provenance")
    rows, fingerprints = load_inputs(config, config_path.parent)
    if config["run_mode"] == "production" and any(r.get("synthetic_observation") for r in rows):
        raise ValueError("Synthetic observations cannot be labelled as a production execution")
    requested = requested_output_ids(config, rows)
    config["outputs"]["requested_outputs"] = requested
    root = Path(__file__).resolve().parents[4]
    reference = config.get("source", {}).get("reference", "reference/BEHAVIORAL_RECIPE.md")
    recipe = config_path.parent / reference
    if not recipe.is_file() and reference == "reference/BEHAVIORAL_RECIPE.md":
        recipe = root / "behavioral_post_analysis_harness/reference/BEHAVIORAL_RECIPE.md"
    if not recipe.is_file():
        raise ValueError(
            "source.reference must locate BEHAVIORAL_RECIPE.md; installed runs need an explicit reference path"
        )
    amendment = recipe.parent / "YMAZE_AMENDMENT.json"
    source_fingerprint, amendment_fingerprint = fingerprint(recipe), fingerprint(amendment)
    scored, debug = score_records(rows, config)
    models, pairs, correlations, checks = run_models(scored, config)
    subgroups, subgroup_status = run_subgroups(scored, config) if "subgroup_inference" in requested else ([], {})
    ledger = inclusion_ledger(scored)
    directory.mkdir(parents=True, exist_ok=True)
    figures, plotting_data, summaries = (
        make_figures(scored, config, directory) if "figures" in requested else ([], [], [])
    )
    limitations = sorted({str(item.get("reason")) for item in models + checks + correlations if item.get("reason")})
    missing_count = sum(row["metric_status"] not in ["available", "not_applicable"] for row in ledger)
    decision_report = [
        {
            "decision_id": f"D{i:02}",
            "resolution": config.get("decision_resolutions", {}).get(f"D{i:02}"),
            "status": "recorded_dependency_scoped"
            if f"D{i:02}" in config.get("decision_resolutions", {})
            else "unresolved",
        }
        for i in range(1, 15)
    ]
    extensions: list[Record] = [
        subgroup_status,
        {
            "output": "spatial_location_bias",
            "status": "blocked",
            "decision_ids": ["D08"],
            "reason": "Source side_bias is retained; spatial extension requires object-to-location definition",
        },
    ]
    supplementary = [item for item in extensions if item.get("output") in requested]
    output_statuses: Record = {}
    if "metrics" in requested:
        output_statuses["metrics"] = {
            "status": "complete"
            if scored and not missing_count and not any(r.get("blocked_outputs") for r in scored)
            else "partial"
            if scored
            else "unavailable",
            "missing_metric_count": missing_count,
        }
    for name, results, success in [
        ("models", models, "fitted"),
        ("correlations", correlations, "computed"),
        ("within_group_tests", checks, "computed"),
        ("figures", figures, "rendered"),
    ]:
        if name in requested:
            output_statuses[name] = result_status(results, success)
    output_statuses.update({item["output"]: item for item in supplementary})
    completed = sum(value["status"] == "complete" for value in output_statuses.values())
    overall_status = (
        "complete"
        if completed == len(requested)
        else "partial"
        if completed or any(value["status"] == "partial" for value in output_statuses.values())
        else "blocked"
    )
    limitations = sorted(set(limitations + [str(item["reason"]) for item in supplementary if item.get("reason")]))
    limitations = sorted(
        set(limitations + [reason for status in output_statuses.values() for reason in status.get("reasons", [])])
    )
    dependencies = {}
    for package in ["numpy", "scipy", "polars", "matplotlib", "bekiper-cli"]:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "source_checkout"
    manifest = {
        "schema_version": "1.1",
        "status": overall_status,
        "requested_outputs": requested,
        "output_statuses": output_statuses,
        "run_mode": config["run_mode"],
        "synthetic_fixture": config.get("synthetic_fixture", False),
        "source_recipe": source_fingerprint,
        "ymaze_amendment": amendment_fingerprint,
        "config_fingerprint": fingerprint(config_path),
        "inputs": fingerprints,
        "resolved_config": config,
        "decision_ids": sorted(config.get("decision_resolutions", {})),
        "code": revision(),
        "dependencies": dependencies,
        "random_seed": config["outputs"].get("random_seed"),
        "implementation_sources": [fingerprint(path) for path in sorted(Path(__file__).parent.glob("*.py"))],
        "input_verification": [
            {**{k: r[k] for k in KEY}, "threshold_pct": r.get("threshold_pct"), **r["input_verification"]}
            for r in scored
        ],
        "missing_metric_count": missing_count,
        "limitations": limitations,
        "figures": figures,
        "engineering_extensions": ["D02 nonnegative window safeguard", "D13 undefined zero training biases"],
        "real_data_execution": config["run_mode"] != "synthetic"
        and not any(r.get("synthetic_observation") for r in scored),
        "scientific_readiness": "fixture_only"
        if config["run_mode"] == "synthetic"
        else "see_dependency_scoped_decision_report",
        "supplementary_output_statuses": supplementary,
    }
    # Operational/private paths are replaced by basenames in the public manifest.
    for field in list(manifest["resolved_config"].get("inputs", {})):
        if field.endswith("_path"):
            manifest["resolved_config"]["inputs"][field] = fingerprints_for_role(
                fingerprints, field.removesuffix("_path")
            )
    for evidence in manifest["resolved_config"]["inputs"].get("upstream_rule_provenance", {}).values():
        for file in evidence.get("evidence_files", []):
            file["path"] = Path(file["path"]).name
    manifest["resolved_config"]["outputs"]["directory"] = directory.name
    manifest["resolved_config"].setdefault("source", {})["reference"] = recipe.name
    for name, value in {
        "metrics": scored,
        "inclusion_ledger": ledger,
        "event_bout_debug": debug,
        "models": models,
        "pairing_ledger": pairs,
        "correlations": correlations,
        "within_group_tests": checks,
        "plotting_data": plotting_data,
        "summaries": summaries,
        "decision_report": decision_report,
        "subgroup_analyses": subgroups,
        "supplementary_statuses": supplementary,
        "run_manifest": manifest,
    }.items():
        write_json(directory / f"{name}.json", value)
    (directory / "report.md").write_text(report(manifest, scored, models, ledger))
    return manifest


def fingerprints_for_role(fingerprints: list[Record], role: str) -> list[Record]:
    return [f for f in fingerprints if f["role"] == role]


def report(manifest: Record, rows: list[Record], models: list[Record], ledger: list[Record]) -> str:
    text = [
        "# Behavioural post-analysis run",
        "",
        f"Status: **{manifest['status']}**; mode: **{manifest['run_mode']}**.",
        f"Requested outputs: {', '.join(manifest['requested_outputs'])}.",
        "Completion describes these outputs; verification and scientific decisions are reported separately.",
        "",
        "Synthetic mode contains test observations only; no biological conclusion is implied.",
        f"Sessions/threshold records: {len(rows)}. Animals: {len({r['animal_id'] for r in rows})}.",
        "",
        "## Registered models and planned contrast",
        "",
        "| Model | Status | Animal n | Contrast estimate | Raw P | Reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    for model in models:
        contrast = model.get("contrast") or {}
        text.append(
            f"| {model['model_id']} | {model['status']} | {model['n_animals']} | {contrast.get('estimate')} | "
            f"{contrast.get('p_raw')} | {model.get('reason')} |"
        )
    text += [
        "",
        "The planned sign is (G5−G6)−(G1−G2). Exact specifications, coefficient uncertainty,",
        "subset accounting, residual diagnostics and adjustment families are in models.json.",
        "QC flags are not exclusions; metadata exclusions and endpoint/model missingness are separate.",
        "Source side_bias is algebraic (−object_bias), not proven spatial bias.",
        "OFT center percentages always divide by 300 s; locomotion accompanies avoidance readouts.",
        "Principal Y-maze 25/45/65% entries follow amendment YM-2026-10-04 with no dwell.",
        "Imported measurements do not establish upstream geometry, orientation or behavioural verification.",
        "",
        "## Limitations / blocked and unavailable outputs",
        "",
    ]
    text += [f"- {reason}" for reason in manifest["limitations"]]
    text += [f"- {reason}" for reason in sorted({r["missing_reason"] for r in ledger if r["missing_reason"]})]
    text += [
        "",
        "See inclusion_ledger.json for every session/endpoint; pairing_ledger.json for unmatched animals;",
        "plotting_data.json and summaries.json for actual plotted animals and n/uncertainty.",
    ]
    text += [
        "",
        "Registered subgroup attempts, membership accounting and correction families are in subgroup_analyses.json.",
    ]
    return "\n".join(text) + "\n"
