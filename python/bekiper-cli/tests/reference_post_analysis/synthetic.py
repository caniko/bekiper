"""Deterministic DEVELOPMENT fixtures, never experimental observations/defaults."""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

from .inputs import FACTORS, load_json
from .metrics import OFT_PRIMARY
from .pipeline import write_json


def create_fixture(directory: Path) -> Path:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Fixture directory must be fresh")
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[4]
    config = copy.deepcopy(load_json(root / "behavioral_post_analysis_harness/config/analysis_config.template.json"))
    # This migration fixture retains the October 4 event sets explicitly; the
    # current native Y-maze default is >= 1 s (instruction 2026-10-07).
    config["assays"]["ymaze"]["principal_minimum_dwell_s"] = 0
    config["assays"]["ymaze"]["temporal_policy"] = {"min_duration_seconds": 0.0, "duration_comparison": "at_least"}
    levels = {"five_xfad": ["WT", "AD"], "hcar1": ["KO", "WT"], "treatment": ["vehicle", "lactate"]}
    known: dict[tuple[str, ...], str] = {
        ("AD", "WT", "lactate"): "G5",
        ("AD", "WT", "vehicle"): "G6",
        ("AD", "KO", "lactate"): "G1",
        ("AD", "KO", "vehicle"): "G2",
    }
    groups = {}
    metadata, oft, ymaze, nort = [], [], [], []
    sequences = ["ABCABBAC", "ABCABCABA", "ABABACBA", "ABCABCABCA", "ABBBACCA", "ABCAABCB", "ABBABCACAB", "ABCABACB"]
    for cell, factors in enumerate(itertools.product(*levels.values())):
        group = known.get(factors, f"CONTROL_{cell}")  # nonconsecutive control identifiers
        groups[group] = dict(zip(FACTORS, factors, strict=True))
        for rep in range(4):
            animal = f"SYNTHETIC_{cell}_{rep}"
            noise = [-0.03, -0.01, 0.01, 0.03][rep]
            effect = 0.1 * (factors[2] == "lactate") + 0.15 * (
                factors[0] == "AD" and factors[1] == "WT" and factors[2] == "lactate"
            )
            for assay, phases, timepoints in [
                ("oft", ["session"], ["pre", "post"]),
                ("ymaze", ["session"], ["pre", "post"]),
                ("nort", ["training", "test"], ["post"]),
            ]:
                for timepoint, phase in itertools.product(timepoints, phases):
                    identity = {
                        "animal_id": animal,
                        "assay": assay,
                        "phase": phase,
                        "timepoint": timepoint,
                        "session_id": f"{animal}_{assay}_{timepoint}_{phase}",
                    }
                    excluded = cell == 0 and rep == 0  # explicit metadata, not a quality filter
                    metadata.append(
                        {
                            **identity,
                            **groups[group],
                            "group": group,
                            "sex": "female" if rep % 2 else "male",
                            "age_treatment_window": "synthetic_fixed_window",
                            "excluded": excluded,
                            "exclusion_reason": "synthetic_metadata_exclusion" if excluded else None,
                        }
                    )
                    if assay == "oft":
                        post_noise = [0.01, -0.02, 0.03, -0.01][rep] if timepoint == "post" else 0
                        center = (
                            30
                            + rep * 4
                            + (cell % 3) * 3
                            + (effect * 60 if timepoint == "post" else 0)
                            + post_noise * 10
                        )
                        oft.append(
                            {
                                **identity,
                                "center_time_s": center,
                                "center_entries": 4 + rep + ([0, 1, 0, -1][rep] if timepoint == "post" else 0),
                                "latency_to_first_center_entry_s": 10 + cell,  # intentionally singular baseline fixture
                                "total_distance_m": 10
                                + cell
                                + rep
                                + (effect if timepoint == "post" else 0)
                                + post_noise,
                                "mean_speed_m_per_s": (10 + cell + rep + post_noise) / 300,
                                "periphery_time_pct": (300 - center) / 300 * 100,
                                "analysis_duration_s": 300,
                                "freezing_s": 5 + rep,
                                "grooming_s": 2 + rep,
                                "rearing_s": 3 + rep,
                                "immobility_s": 6 + rep,
                                "fecal_boli_count": rep,
                                "synthetic_observation": True,
                            }
                        )
                    elif assay == "ymaze":
                        for threshold in [25, 45, 65]:
                            index = (cell + rep + (1 if timepoint == "post" else 0) + threshold // 20) % len(sequences)
                            ymaze.append(
                                {
                                    **identity,
                                    "threshold_pct": threshold,
                                    "valid_arm_entry_sequence": list(sequences[index]),
                                    "raw_arm_entry_sequence": list(sequences[index]),
                                    "removed_short_entries": 0,
                                    "total_distance_m": 7 + cell + rep / 2,
                                    "mean_speed_m_per_s": (7 + cell + rep / 2) / 360,
                                    "freezing_s": 4 + rep,
                                    "arm_preference": rep / 3,
                                    "analysis_duration_s": 360,
                                    "synthetic_observation": True,
                                    "upstream_revision": "synthetic_fixture_not_real_code_revision",
                                    "scoring_provenance": {
                                        "geometry_version": "imen_radial_vertex_span_v1",
                                        "entry_rule": "dual_landmark_false_to_true_v1",
                                        "head_columns": {"x": "head_center_x", "y": "head_center_y"},
                                        "tail_columns": {"x": "tail_base_x", "y": "tail_base_y"},
                                        "requires_both_landmarks_same_arm": True,
                                        "apply_dwell_filter": False,
                                        "arm_membership_hysteresis": False,
                                        "zone_max_gap_seconds": 0,
                                        "pose_gap_fill_seconds": 0,
                                        "proximal_threshold": threshold / 100,
                                        "minimum_dwell_seconds": 0,
                                    },
                                }
                            )
                    else:
                        values = (
                            {"TA": 9 + rep, "TB": 6 + cell % 3}
                            if phase == "training"
                            else {"TN": 10 * (1 + effect + noise), "TF": 10 * (1 - effect - noise)}
                        )
                        candidate_segments = []
                        for field, seconds in values.items():
                            candidate_segments.append(
                                {
                                    "segment_id": field,
                                    "object_id": field,
                                    "start_s": 0 if field in ["TA", "TN"] else 40,
                                    "end_s": seconds if field in ["TA", "TN"] else 40 + seconds,
                                    "candidate_eligible": True,
                                    "angles_degrees": [0, 0, 90],
                                    "nose_speeds_m_per_s": [0.01, 0.01, 0.01],
                                }
                            )
                        nort.append(
                            {
                                **identity,
                                **values,
                                "candidate_segments": candidate_segments,
                                "object_configuration": "synthetic_counterbalanced",
                                "analysis_duration_s": 360,
                                "analysis_start_s": 0,
                                "protocol_metadata": {"familiarization_s": 360, "delay_h": 5, "test_s": 360},
                                "synthetic_observation": True,
                            }
                        )
    config.update(run_mode="synthetic", synthetic_fixture=True)
    config["inputs"].update(
        metadata_path="metadata.json",
        oft_path="oft.json",
        ymaze_path="ymaze.json",
        nort_path="nort.json",
        factor_levels=levels,
        group_mapping=groups,
        units_and_calibration={a: {"time": "s", "distance": "m", "speed": "m/s"} for a in ["oft", "ymaze", "nort"]},
        upstream_rule_provenance={
            "ymaze": {
                "landmark_roles": {
                    "head_center": {"x": "head_center_x", "y": "head_center_y"},
                    "tail_base": {"x": "tail_base_x", "y": "tail_base_y"},
                }
            }
        },
    )
    config["assays"]["ymaze"].update(
        triad_policy="source_table", fewer_than_three_entries_policy="zero_windows", primary_threshold_arm_length_pct=45
    )
    config["assays"]["nort"].update(
        merged_gap_duration_policy="exclude_gaps",
        merged_gap_median_policy="observed_samples_only",
        window_and_tracking_policy={
            "interval": "half_open",
            "origin": "analysis_start_s",
            "duration": "analysis_duration_s",
            "out_of_window": "reject",
            "protocol_departures": "flag_only",
        },
        object_role_and_location_mapping={"training": {"TA": "TA", "TB": "TB"}, "test": {"TN": "TN", "TF": "TF"}},
        exploration_quality_flag={
            "name": "passes_15s_exploration_threshold",
            "phase": "test",
            "quantity": "total_test_exploration",
            "operator": ">=",
            "threshold_s": 15,
        },
    )
    config["inclusion"].update(
        existing_metadata_exclusions_policy="honor_metadata",
        qc_flag_exclusion_policy="flags_only",
        missing_metric_policy="endpoint_complete_cases",
        unmatched_prepost_policy="report_and_withhold_pair",
        upstream_verification_requirement="allow_unverified",
    )
    config["statistics"].update(
        oft_baseline_strategy="baseline_adjusted",
        model_family_by_endpoint={
            **{f"oft.{f}": "gaussian_ols" for f in OFT_PRIMARY},
            "ymaze.spontaneous_alternation_pct": "gaussian_ols",
            "nort.DI": "gaussian_ols",
        },
        factor_coding="treatment",
        covariance_estimator="classical",
        test_tails="two-sided",
        confidence_interval_method="student_t",
        confidence_level=0.95,
        summary_uncertainty="SEM",
        contrast_estimand_and_weighting={"weighting": "fixed_reference", "reference_values": {"pre_outcome": "mean"}},
        contrast_roles={
            "ad_level": "AD",
            "hcar1_wt": "WT",
            "hcar1_ko": "KO",
            "lactate": "lactate",
            "vehicle": "vehicle",
        },
        nort_additional_covariates=[],
        nort_within_group_checks=["DI"],
        nort_within_group_test_method="one_sample_t",
        spearman_missing_and_constant_input_policy="pairwise_complete_constant_missing",
        model_failure_policy="report_without_simplification",
        multiplicity={
            "method": "holm",
            "families": {"nort_within_group": "synthetic_DI_groups", "planned_contrasts": "synthetic_primary_models"},
        },
    )
    config["outputs"].update(directory="results", figure_formats=["svg"], random_seed=0)
    resolutions: dict[str, object] = {}
    for decision, paths in {
        "D01": ["assays.ymaze.triad_policy"],
        "D02": ["assays.ymaze.fewer_than_three_entries_policy"],
        "D03": ["assays.ymaze.primary_threshold_arm_length_pct"],
        "D06": ["assays.nort.exploration_quality_flag"],
        "D07": ["assays.nort.merged_gap_duration_policy", "assays.nort.merged_gap_median_policy"],
        "D09": [f"statistics.{field}" for field, value in config["statistics"].items() if value is not None],
        "D10": ["inputs.group_mapping", "inputs.factor_levels", "assays.nort.object_role_and_location_mapping"],
        "D12": [f"inclusion.{field}" for field in config["inclusion"]],
        "D14": ["assays.nort.window_and_tracking_policy"],
    }.items():
        decision_values: dict[str, object] = {}
        for path in paths:
            value = config
            for part in path.split("."):
                value = value[part]
            decision_values[path] = value
        resolutions[decision] = {
            "value": decision_values,
            "scope": ["assays", "statistics", "inputs", "inclusion"],
            "origin": "development_fixture",
            "rationale": "Deterministic test-only coverage, not scientific approval",
            "approval_or_plan_reference": "bekiper_cli.post_analysis.synthetic fixture v1",
        }
    config["decision_resolutions"] = resolutions
    for name, rows in [("metadata", metadata), ("oft", oft), ("ymaze", ymaze), ("nort", nort), ("config", config)]:
        write_json(directory / f"{name}.json", rows)
    return directory / "config.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    print(create_fixture(args.directory))
