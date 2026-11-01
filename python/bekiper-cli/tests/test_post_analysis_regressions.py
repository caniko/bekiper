"""Adversarial input/provenance cases discovered during implementation review."""

import copy
import json

import pytest

from reference_post_analysis.inputs import load_inputs
from reference_post_analysis.pipeline import run, run_models, score_records
from reference_post_analysis.synthetic import create_fixture


def register_choice(config, decision, path, value):
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    record = config["decision_resolutions"].setdefault(
        decision,
        {
            "value": {},
            "scope": ["assays", "statistics", "inputs", "inclusion"],
            "origin": "development_fixture",
            "rationale": "Review regression fixture",
            "approval_or_plan_reference": "post_analysis_review_regressions",
        },
    )
    record["value"][path] = copy.deepcopy(value)


@pytest.fixture
def fixture_inputs(tmp_path):
    path = create_fixture(tmp_path / "fixture")
    config = json.loads(path.read_text())
    rows, _ = load_inputs(config, path.parent)
    return path, config, rows


@pytest.mark.parametrize(
    "field,value",
    [
        ("sex", "conflicting_sex"),
        ("age_treatment_window", "conflicting_age"),
        ("pair_id", "wrong_pair"),
    ],
)
def test_measurements_cannot_replace_authoritative_metadata(fixture_inputs, field, value):
    path, config, _ = fixture_inputs
    meta_path = path.parent / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    if field == "pair_id":
        for row in metadata:
            row[field] = "registered_pair"
        meta_path.write_text(json.dumps(metadata))
    measurements_path = path.parent / "ymaze.json"
    measurements = json.loads(measurements_path.read_text())
    measurements[0][field] = value
    measurements_path.write_text(json.dumps(measurements))
    with pytest.raises(ValueError, match=f"metadata mismatch for {field}"):
        load_inputs(config, path.parent)


def test_score_only_import_retains_measured_ymaze_endpoint_and_provenance(fixture_inputs):
    _, config, rows = fixture_inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "ymaze"))
    row.pop("valid_arm_entry_sequence")
    row.pop("raw_arm_entry_sequence")
    row.update(
        spontaneous_alternation_pct=50,
        spontaneous_alternation_pct_standard=40,
        valid_total_arm_entries=7,
        usable_triads=4,
    )
    result, _ = score_records([row], config)
    assert result[0]["spontaneous_alternation_pct"] == 50
    assert result[0]["spontaneous_alternation_pct_standard"] == 40
    assert result[0]["metric_sources"]["spontaneous_alternation_pct"] == "upstream"
    assert result[0]["input_verification"]["status"] == "verified_clarified_entry_definition"
    assert result[0]["qc_flags"]["low_quality"] is True
    assert "spontaneous_alternation_pct" not in result[0]["missing_reasons"]


def test_imported_score_does_not_bypass_unresolved_triad_policy(fixture_inputs):
    _, config, rows = fixture_inputs
    config["decision_resolutions"].pop("D01")
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "ymaze"))
    row.pop("valid_arm_entry_sequence")
    row["spontaneous_alternation_pct"] = 50
    result, _ = score_records([row], config)
    assert result[0]["spontaneous_alternation_pct"] is None
    assert result[0]["upstream_spontaneous_alternation_pct"] == 50
    assert result[0]["missing_reasons"]["spontaneous_alternation_pct"] == "blocked:D01"


def test_coordinate_sequence_disagreement_preserves_both_event_sets(fixture_inputs):
    _, config, rows = fixture_inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "ymaze"))
    policy = {
        "interval": "half_open",
        "origin": "first_detection_s",
        "duration": "analysis_duration_s",
        "out_of_window": "reject",
        "missing_pose": "observed_samples_break_on_missing_no_interpolation",
    }
    row["first_detection_s"] = 0
    config["assays"]["ymaze"]["window_and_tracking_policy"] = policy
    config["decision_resolutions"]["D04"] = {
        "value": {"assays.ymaze.window_and_tracking_policy": policy},
        "scope": ["assays.ymaze"],
        "origin": "development_fixture",
        "rationale": "Review regression fixture",
        "approval_or_plan_reference": "test_coordinate_sequence_disagreement",
    }
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
    points = [(0, 0), (0.9, 0), (0, 0), (-0.9, 0), (0, 0), (0, 0.9), (0, 0)]
    row["coordinate_samples"] = [
        {"timestamp_s": i / 30, "head_center_x": x, "head_center_y": y, "tail_base_x": x, "tail_base_y": y}
        for i, (x, y) in enumerate(points)
    ]
    row["valid_arm_entry_sequence"] = list("ABC")
    row["raw_arm_entry_sequence"] = list("ABC")
    result, _ = score_records([row], config)
    assert result[0]["valid_arm_entry_sequence"] == list("ACB")
    assert result[0]["upstream_valid_arm_entry_sequence"] == list("ABC")
    assert result[0]["upstream_raw_arm_entry_sequence"] == list("ABC")
    assert set(result[0]["disagreements"]) >= {"valid_arm_entry_sequence", "raw_arm_entry_sequence"}
    assert "disagreements" not in row  # scoring must not mutate its caller's evidence


def test_nort_score_only_import_and_unavailable_reconstruction_retain_measurements(fixture_inputs):
    _, config, rows = fixture_inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test"))
    row.pop("candidate_segments")
    row.pop("TN")
    row.pop("TF")
    row["DI"] = 0.25
    result, _ = score_records([row], config)
    assert result[0]["DI"] == 0.25
    assert result[0]["metric_sources"]["DI"] == "upstream"
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test"))
    row["candidate_segments"] = []
    result, _ = score_records([row], config)
    assert result[0]["TN"] == row["TN"]
    assert result[0]["TF"] == row["TF"]
    assert result[0]["recomputed_TN"] is None
    assert result[0]["metric_sources"]["TN"] == "upstream"


def register_requirement(config, value):
    config["inclusion"]["upstream_verification_requirement"] = value
    config["decision_resolutions"]["D12"]["value"]["inclusion.upstream_verification_requirement"] = value


def test_strict_verification_withholds_pairs_with_unverified_baselines(fixture_inputs):
    _, config, rows = fixture_inputs
    register_requirement(config, "require_verified")
    scored, _ = score_records([r for r in rows if r["assay"] == "ymaze"], config)
    for row in scored:
        if row["timepoint"] == "pre":
            row["input_verification"] = {"status": "unresolved", "reasons": ["missing_pre_evidence"]}
    models, ledger, correlations, _ = run_models(scored, config)
    assert models[0]["n_animals"] == 0
    assert models[0]["status"] == "unavailable"
    assert sum(item["reason"].startswith("upstream_unverified:pre") for item in ledger) == 31
    assert sum(item["reason"] == "excluded:paired_metadata" for item in ledger) == 1
    assert correlations[0]["n_animals"] == 31  # verified post-only QC does not require a baseline


def test_strict_verification_applies_to_spearman_with_an_auditable_subset(fixture_inputs):
    _, config, rows = fixture_inputs
    register_requirement(config, "require_verified")
    scored, _ = score_records([r for r in rows if r["assay"] == "ymaze"], config)
    for row in scored:
        row["input_verification"] = {"status": "unresolved", "reasons": ["missing_evidence"]}
    _, _, correlations, _ = run_models(scored, config)
    assert all(item["n_animals"] == 0 and item["p_raw"] is None for item in correlations)
    assert all(len(item["subset"]) == 32 for item in correlations)
    assert any(entry["reason"].startswith("upstream_unverified") for entry in correlations[0]["subset"])


def test_unregistered_inclusion_policy_blocks_correlations(fixture_inputs):
    _, config, rows = fixture_inputs
    config["decision_resolutions"].pop("D12")
    scored, _ = score_records([r for r in rows if r["assay"] == "ymaze"], config)
    _, _, correlations, _ = run_models(scored, config)
    assert correlations and all(item["status"] == "blocked" for item in correlations)


def test_nort_one_sample_checks_do_not_depend_on_factorial_covariance(fixture_inputs):
    _, config, rows = fixture_inputs
    config["statistics"]["covariance_estimator"] = None
    config["decision_resolutions"]["D09"]["value"].pop("statistics.covariance_estimator")
    scored, _ = score_records([r for r in rows if r["assay"] == "nort"], config)
    models, _, _, checks = run_models(scored, config)
    assert models[0]["status"] == "blocked"
    assert len(checks) == 8
    assert all(item["p_adjusted"] is not None for item in checks)
    assert sum(item["n_animals"] for item in checks) == 31
    assert all("subset" in item for item in checks)


def test_metrics_only_run_completes_without_inapplicable_supplementary_blocks(fixture_inputs, tmp_path):
    path, config, _ = fixture_inputs
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_path.write_text(json.dumps([r for r in metadata if r["assay"] == "oft"]))
    config["inputs"]["ymaze_path"] = None
    config["inputs"]["nort_path"] = None
    config["outputs"]["requested_outputs"] = ["metrics"]
    path.write_text(json.dumps(config))
    manifest = run(path, output=tmp_path / "results", env={})
    assert manifest["status"] == "complete"
    assert manifest["requested_outputs"] == ["metrics"]
    assert manifest["supplementary_output_statuses"] == []
    assert manifest["figures"] == []
    assert json.loads((tmp_path / "results/models.json").read_text()) == []


def test_explicitly_requested_unsupported_extension_is_reported_partial(fixture_inputs, tmp_path):
    path, _, _ = fixture_inputs
    manifest = run(path, output=tmp_path / "results", env={}, requested_outputs=["metrics", "subgroup_inference"])
    assert manifest["status"] == "partial"
    assert manifest["supplementary_output_statuses"][0]["status"] == "blocked"
    assert manifest["output_statuses"]["subgroup_inference"]["status"] == "blocked"


def test_strict_verification_is_visible_in_plot_populations(fixture_inputs, tmp_path):
    from reference_post_analysis.plots import make_figures

    _, config, rows = fixture_inputs
    register_requirement(config, "require_verified")
    scored, _ = score_records([r for r in rows if r["assay"] == "ymaze"], config)
    for row in scored:
        if row["animal_id"] == "SYNTHETIC_0_1":
            row["input_verification"] = {"status": "unresolved", "reasons": ["missing_evidence"]}
    figures, plotting_data, _ = make_figures(scored, config, tmp_path)
    assert figures and plotting_data
    assert all(r["animal_id"] != "SYNTHETIC_0_1" for r in plotting_data)
    assert all(r["population_policy"] == "require_verified" for r in plotting_data)
    assert all(r["input_verification_status"] == "verified_clarified_entry_definition" for r in plotting_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("spontaneous_alternation_pct", 101),
        ("spontaneous_alternation_pct", True),
        ("usable_triads", -1),
        ("usable_triads", 4.5),
    ],
)
def test_score_only_import_rejects_impossible_values(fixture_inputs, field, value):
    _, config, rows = fixture_inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "ymaze"))
    row.pop("valid_arm_entry_sequence")
    row[field] = value
    with pytest.raises(ValueError):
        score_records([row], config)


def test_supplied_score_cannot_override_a_known_zero_exploration_total(fixture_inputs):
    _, config, rows = fixture_inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test"))
    row.pop("candidate_segments")
    row.update(TN=0, TF=0, DI=0.5)
    scored, _ = score_records([row], config)
    assert scored[0]["DI"] is None
    assert scored[0]["upstream_DI"] == 0.5
    assert scored[0]["missing_reasons"]["DI"] == "undefined:zero_exploration_total"


def test_missing_reference_fails_before_creating_output_artifacts(fixture_inputs, tmp_path):
    path, config, _ = fixture_inputs
    config["source"]["reference"] = "missing_recipe.md"
    path.write_text(json.dumps(config))
    directory = tmp_path / "results"
    with pytest.raises(ValueError, match="source.reference"):
        run(path, output=directory, env={})
    assert not directory.exists()


@pytest.mark.parametrize(
    "role,columns",
    [
        ("head_center", "head_columns"),
        ("tail_base", "tail_columns"),
    ],
)
@pytest.mark.parametrize("incomplete", [None, {}, {"x": "only_x"}, {"x": "", "y": ""}])
def test_missing_landmark_evidence_never_enters_strict_inference(fixture_inputs, role, columns, incomplete):
    _, config, rows = fixture_inputs
    register_requirement(config, "require_verified")
    config["inputs"]["upstream_rule_provenance"]["ymaze"]["landmark_roles"][role] = incomplete
    rows = copy.deepcopy([r for r in rows if r["assay"] == "ymaze"])
    for row in rows:
        row["scoring_provenance"][columns] = incomplete
    scored, _ = score_records(rows, config)
    assert all(r["input_verification"]["status"] == "unresolved" for r in scored)
    models, _, correlations, _ = run_models(scored, config)
    assert models[0]["n_animals"] == 0
    assert all(c["n_animals"] == 0 for c in correlations)


@pytest.mark.parametrize("start,end", [(0, 500), (400, 402), (359, 361)])
def test_nort_reconstruction_rejects_candidates_outside_declared_window(fixture_inputs, start, end):
    _, config, rows = fixture_inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test"))
    row["candidate_segments"][0].update(start_s=start, end_s=end)
    with pytest.raises(ValueError, match="outside.*window"):
        score_records([row], config)


def test_nort_reconstruction_requires_registered_window_policy(fixture_inputs):
    _, config, rows = fixture_inputs
    config["decision_resolutions"].pop("D14", None)
    row = next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test")
    scored, _ = score_records([row], config)
    assert "nort_bout_reconstruction:D14:window_policy" in scored[0]["blocked_outputs"]
    assert scored[0]["metric_sources"]["TN"] == "upstream"


def test_nort_actual_protocol_departure_is_flagged(fixture_inputs):
    _, config, rows = fixture_inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test"))
    row["protocol_metadata"].update(familiarization_s=1, delay_h=0)
    scored, _ = score_records([row], config)
    assert scored[0]["qc_flags"]["protocol_timing_departure"] is True
    assert scored[0]["protocol_timing_departures"] == ["familiarization_s", "delay_h"]
    assert scored[0]["protocol_metadata"]["delay_h"] == 0


@pytest.mark.parametrize("roles", [{"TN": "TN", "TF": "TN"}, {"TN": "TN"}, {"TN": "TN", "TF": "TF", "DI": "TN"}])
def test_nort_role_mapping_requires_exact_distinct_phase_roles(fixture_inputs, roles):
    _, config, rows = fixture_inputs
    mapping = copy.deepcopy(config["assays"]["nort"]["object_role_and_location_mapping"])
    mapping["test"] = roles
    register_choice(config, "D10", "assays.nort.object_role_and_location_mapping", mapping)
    with pytest.raises(ValueError, match="object.*roles"):
        score_records([next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test")], config)


def test_nort_counterbalanced_roles_are_resolved_per_session(fixture_inputs):
    _, config, rows = fixture_inputs
    row = copy.deepcopy(next(r for r in rows if r["assay"] == "nort" and r["phase"] == "test"))
    row.update(TN=6, TF=4)
    row["candidate_segments"][0].update(end_s=6)
    row["candidate_segments"][1].update(end_s=44)
    mapping = {
        "sessions": [
            {
                **{k: row[k] for k in ("animal_id", "assay", "session_id", "phase", "timepoint")},
                "roles": {"TN": "TF", "TF": "TN"},
            }
        ]
    }
    register_choice(config, "D10", "assays.nort.object_role_and_location_mapping", mapping)
    scored, _ = score_records([row], config)
    assert scored[0]["TN"] == 4
    assert scored[0]["TF"] == 6
    assert scored[0]["DI"] == pytest.approx(-0.2)
    assert scored[0]["upstream_TN"] == 6
    assert scored[0]["resolved_object_roles"] == {"TN": "TF", "TF": "TN"}


@pytest.mark.parametrize("field,value", [("phase", "unregistered"), ("timepoint", "unregistered")])
def test_unknown_canonical_session_context_is_rejected(fixture_inputs, field, value):
    from reference_post_analysis.inputs import validate_metadata

    _, config, rows = fixture_inputs
    row = copy.deepcopy(rows[0])
    row[field] = value
    with pytest.raises(ValueError, match=f"Canonical {field}"):
        validate_metadata([row], config)


def test_censoring_prerequisite_ignores_metadata_excluded_animals(fixture_inputs):
    _, config, rows = fixture_inputs
    config["outputs"]["requested_outputs"] = ["models"]
    scored, _ = score_records([r for r in rows if r["assay"] == "oft" and r["timepoint"] == "post"], config)
    endpoint = "latency_to_first_center_entry_s"
    for i, row in enumerate(scored):
        row[endpoint] += [0.2, -0.1, 0.3, -0.2][i % 4]
        if row["excluded"]:
            row["latency_right_censored"] = True
    models, _, _, _ = run_models(scored, config)
    model = next(m for m in models if m["endpoint"] == endpoint)
    assert model["status"] == "fitted"
    assert model["n_animals"] == 31
    assert len(model["subset"]) == 32


def test_paired_panel_uses_only_accepted_session_pairs(fixture_inputs, tmp_path):
    from reference_post_analysis.plots import make_figures

    _, config, rows = fixture_inputs
    scored, _ = score_records([r for r in rows if r["assay"] == "ymaze"], config)
    for row in scored:
        if row["animal_id"] == "SYNTHETIC_0_1":
            row["pair_id"] = "different_" + row["timepoint"]
    _, data, _ = make_figures(scored, config, tmp_path)
    pairs = [r for r in data if r.get("label") == "Paired pre/post SA (%)"]
    assert len(pairs) == 60
    assert all(r["animal_id"] != "SYNTHETIC_0_1" for r in pairs)
    assert all(r["plot_pair_id"] and "pair_id" in r for r in pairs)
    assert any(r["animal_id"] == "SYNTHETIC_0_1" and r.get("label") == "Post-treatment SA (%) — primary" for r in data)


def test_sensitivity_panels_respect_requested_thresholds(fixture_inputs, tmp_path):
    from reference_post_analysis.plots import make_figures

    _, config, rows = fixture_inputs
    config["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"] = [65]
    scored, _ = score_records([r for r in rows if r["assay"] == "ymaze"], config)
    figures, data, _ = make_figures(scored, config, tmp_path)
    supplement = next(f for f in figures if f["name"] == "supplement_ymaze")
    assert len(supplement["panels"]) == 2
    assert {r["threshold_pct"] for r in data if r["figure"] == "supplement_ymaze"} == {65}
