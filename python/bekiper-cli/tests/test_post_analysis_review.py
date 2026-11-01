"""Window boundaries, context mappings and registered supplementary outputs."""

import copy
import json

import pytest

from reference_post_analysis.inputs import load_inputs
from reference_post_analysis.pipeline import run, score_records
from reference_post_analysis.synthetic import create_fixture


@pytest.fixture
def inputs(tmp_path):
    path = create_fixture(tmp_path / "fixture")
    config = json.loads(path.read_text())
    rows, _ = load_inputs(config, path.parent)
    return path, config, rows


def register(config, decision, path, value):
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = copy.deepcopy(value)
    record = config["decision_resolutions"].setdefault(
        decision,
        {
            "value": {},
            "scope": ["assays", "statistics", "inclusion"],
            "origin": "development_fixture",
            "rationale": "Review fixture",
            "approval_or_plan_reference": "post_analysis_review",
        },
    )
    record["value"][path] = copy.deepcopy(value)


@pytest.mark.parametrize("time", [3, 364, 400])
def test_ymaze_coordinates_outside_first_detection_window_are_rejected(inputs, time):
    _, config, rows = inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "ymaze"))
    row["first_detection_s"] = 4
    row["geometry"] = {
        "version": "imen_radial_vertex_span_v1",
        "canonical_vertices": True,
        "center": [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
        "arms": {
            "A": [[0.1, 0], [0.6, 0.05], [1.1, 0], [0.6, -0.05]],
            "B": [[0, 0.1], [0.05, 0.6], [0, 1.1], [-0.05, 0.6]],
            "C": [[-0.1, 0], [-0.6, 0.05], [-1.1, 0], [-0.6, -0.05]],
        },
    }
    row["coordinate_samples"] = [
        {"timestamp_s": time, "head_center_x": 0.9, "head_center_y": 0, "tail_base_x": 0.9, "tail_base_y": 0}
    ]
    register(
        config,
        "D04",
        "assays.ymaze.window_and_tracking_policy",
        {
            "interval": "half_open",
            "origin": "first_detection_s",
            "duration": "analysis_duration_s",
            "out_of_window": "reject",
            "missing_pose": "observed_samples_break_on_missing_no_interpolation",
        },
    )
    with pytest.raises(ValueError, match="outside.*window"):
        score_records([row], config)


def test_nort_interval_can_end_exactly_at_window_boundary(inputs):
    _, config, rows = inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test"))
    row["candidate_segments"][0].update(start_s=359, end_s=360)
    scored, _ = score_records([row], config)
    assert scored[0]["TN"] == 1
    assert scored[0]["reconstruction_window"]["end_s"] == 360


def test_source_session_context_mappings_apply_to_metadata_and_measurements(inputs):
    path, config, _ = inputs
    for filename in ["metadata.json", "oft.json", "ymaze.json", "nort.json"]:
        file = path.parent / filename
        rows = json.loads(file.read_text())
        for row in rows:
            row["timepoint"] = {"pre": "before", "post": "after"}[row["timepoint"]]
            if row["phase"] == "training":
                row["phase"] = "familiarization"
        file.write_text(json.dumps(rows))
    config["inputs"]["session_key_mapping"] = {
        "timepoint": {"before": "pre", "after": "post"},
        "phase": {"familiarization": "training"},
    }
    rows, _ = load_inputs(config, path.parent)
    assert {r["timepoint"] for r in rows} == {"pre", "post"}
    assert {r["phase"] for r in rows if r["assay"] == "nort"} == {"training", "test"}


def test_registered_subgroups_fit_prespecified_factorial_models_and_export_population(inputs, tmp_path):
    path, config, _ = inputs
    plan = {
        "strata": [{"field": "sex", "levels": ["male", "female"], "assays": ["nort"]}],
        "outputs": ["models"],
        "multiplicity": {"method": "holm", "family": "synthetic_sex_models"},
    }
    register(config, "D09", "statistics.subgroup_analysis_plan", plan)
    register(config, "D12", "statistics.subgroup_analysis_plan", plan)
    config["outputs"]["requested_outputs"] = ["subgroup_inference", "figures"]
    path.write_text(json.dumps(config))
    result = run(path, output=tmp_path / "results", env={})
    assert result["output_statuses"]["subgroup_inference"]["status"] == "complete"
    subgroups = json.loads((tmp_path / "results/subgroup_analyses.json").read_text())
    assert [s["level"] for s in subgroups] == ["male", "female"]
    for item in subgroups:
        model = item["models"][0]
        assert model["status"] == "fitted"
        assert model["n_animals"] == (15 if item["level"] == "male" else 16)
        assert model["contrast"]["correction_family"] == "synthetic_sex_models"
        assert model["contrast"]["family_size"] == 2
        assert model["formula"] == "DI ~ 5XFAD_status * HCAR1_status * treatment"
    points = json.loads((tmp_path / "results/plotting_data.json").read_text())
    strata_points = [r for r in points if r["figure"].startswith("subgroup_")]
    assert len(strata_points) == 31
    assert all(r["label"].endswith(r["sex"]) for r in strata_points)


def test_subgroup_plan_requires_both_inference_and_inclusion_registration(inputs, tmp_path):
    path, config, _ = inputs
    plan = {
        "strata": [{"field": "sex", "levels": ["male"], "assays": ["nort"]}],
        "outputs": ["models"],
        "multiplicity": {"method": "holm", "family": "sex_models"},
    }
    register(config, "D09", "statistics.subgroup_analysis_plan", plan)
    path.write_text(json.dumps(config))
    manifest = run(path, output=tmp_path / "results", requested_outputs=["subgroup_inference"], env={})
    assert manifest["status"] == "blocked"
    assert manifest["output_statuses"]["subgroup_inference"]["status"] == "blocked"


def test_oft_audited_rule_document_verifies_definition_with_pinned_supporting_sources(inputs):
    from reference_post_analysis.inputs import fingerprint

    path, config, rows = inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "oft"))
    row.pop("_rule_audit", None)
    row.pop("_rule_evidence_files", None)
    rules = {
        "arena_cm": [40, 40, 34],
        "center_cm": [24, 24],
        "border_cm": 8,
        "entry_landmark": "body_centroid",
        "minimum_dwell_seconds": 1,
        "time_origin": "first_valid_detection",
        "analysis_duration_s": 300,
    }
    row["scoring_provenance"] = {**rules, "code_revision": "synthetic_audited_revision"}
    audited = {
        "origin": "development_fixture",
        "approval_or_plan_reference": "test_source_rule_audit",
        "rationale": "Synthetic source audit",
        "code_revision": "synthetic_audited_revision",
        "rules": rules,
    }
    (path.parent / "audit.json").write_text(json.dumps(audited))
    evidence = [
        {"path": "audit.json", "role": "rule_audit", "sha256": fingerprint(path.parent / "audit.json")["sha256"]}
    ]
    for role in ["implementation", "geometry", "landmarks", "window"]:
        source = path.parent / f"{role}.json"
        source.write_text(json.dumps({"synthetic_evidence": role}))
        evidence.append({"path": source.name, "role": role, "sha256": fingerprint(source)["sha256"]})
    config["inputs"]["upstream_rule_provenance"]["oft"] = {"evidence_files": evidence}
    measurements = json.loads((path.parent / "oft.json").read_text())
    measurements[0] = row
    (path.parent / "oft.json").write_text(json.dumps(measurements))
    loaded, fingerprints = load_inputs(config, path.parent)
    scored, _ = score_records([r for r in loaded if r["assay"] == "oft"], config)
    assert scored[0]["input_verification"]["status"] == "verified_source_rule_definition"
    assert scored[0]["input_verification"]["level"] == "documented_rule_audit_not_event_set_audit"
    assert len([f for f in fingerprints if f["role"].startswith("oft_evidence:")]) == 5
    assert all(r["input_verification"]["status"] == "upstream_unverified" for r in scored[1:])
    production = copy.deepcopy(config)
    production.update(run_mode="production", synthetic_fixture=False)
    denied, _ = score_records([r for r in loaded if r["assay"] == "oft"], production)
    assert denied[0]["input_verification"]["status"] == "upstream_unverified"
    (path.parent / "geometry.json").write_text("changed bytes")
    with pytest.raises(ValueError, match="evidence fingerprint"):
        load_inputs(config, path.parent)


def test_nort_declaration_alone_does_not_verify_candidate_geometry(inputs):
    _, config, rows = inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test"))
    row["scoring_provenance"] = {
        "code_revision": "unsubstantiated",
        "zone_extension_m": 0.03,
        "body_centroid_outside_object_mask": True,
    }
    scored, _ = score_records([row], config)
    assert scored[0]["input_verification"]["status"] == "upstream_unverified"
    assert "rule_audit" in scored[0]["input_verification"]["reasons"]


def test_verified_nort_source_rules_require_geometry_annotations_and_versioned_audit(inputs):
    from reference_post_analysis.events import assay_provenance
    from reference_post_analysis.inputs import fingerprint
    from reference_post_analysis.statistics import verification_reason

    path, config, rows = inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test"))
    rules = {
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
    row["scoring_provenance"] = {**rules, "code_revision": "synthetic_nort_revision"}
    row["_rule_audit"] = {
        "origin": "development_fixture",
        "approval_or_plan_reference": "synthetic_rule_audit",
        "rationale": "Synthetic audit coverage",
        "code_revision": "synthetic_nort_revision",
        "rules": rules,
    }
    row["_rule_evidence_files"] = []
    for role in ["rule_audit", "implementation", "geometry", "landmarks", "window", "annotations"]:
        file = path.parent / f"{role}.json"
        file.write_text(json.dumps({"synthetic_evidence": role}))
        row["_rule_evidence_files"].append({**fingerprint(file), "role": role})
    verified = assay_provenance(row, config)
    assert verified["status"] == "verified_source_rule_definition"
    assert verification_reason({"input_verification": verified}, "require_verified") is None
    row["scoring_provenance"]["angle_aggregation"] = "framewise_rejection"
    assert "angle_aggregation" in assay_provenance(row, config)["reasons"]
