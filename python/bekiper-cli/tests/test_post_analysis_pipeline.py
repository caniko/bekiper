import copy
import json

import pytest

from reference_post_analysis.inputs import choice, load_inputs
from reference_post_analysis.pipeline import run, score_records
from reference_post_analysis.synthetic import create_fixture


def test_synthetic_multi_animal_pipeline_and_semantic_reproducibility(tmp_path):
    config = create_fixture(tmp_path / "fixture")
    first = run(config, output=tmp_path / "first", env={})
    second = run(config, output=tmp_path / "second", env={})
    assert first["synthetic_fixture"] is True
    assert len(first["inputs"]) == 4
    assert first["source_recipe"]["sha256"]
    assert first["ymaze_amendment"]["sha256"]
    for name in [
        "metrics",
        "models",
        "inclusion_ledger",
        "pairing_ledger",
        "correlations",
        "within_group_tests",
        "plotting_data",
        "summaries",
    ]:
        assert json.loads((tmp_path / "first" / f"{name}.json").read_text()) == json.loads(
            (tmp_path / "second" / f"{name}.json").read_text()
        )
    metrics = json.loads((tmp_path / "first/metrics.json").read_text())
    assert len(metrics) == 320
    ymaze = [r for r in metrics if r["assay"] == "ymaze"]
    assert {r["threshold_pct"] for r in ymaze} == {25, 45, 65}
    assert all(r["input_verification"]["status"] == "verified_clarified_entry_definition" for r in ymaze)
    assert any(r["qc_flags"].get("low_quality") and not r["excluded"] for r in ymaze)
    models = json.loads((tmp_path / "first/models.json").read_text())
    fitted = [m for m in models if m["status"] == "fitted"]
    assert {m["model_id"] for m in fitted} >= {"nort.DI", "ymaze.spontaneous_alternation_pct", "oft.total_distance_m"}
    assert all(m["n_animals"] <= 32 for m in models)
    assert any(m["reason"] for m in models)  # singular baselines reported, never simplified
    assert first["status"] == second["status"] == "partial"
    plot_data = json.loads((tmp_path / "first/plotting_data.json").read_text())
    assert {r.get("reference_line") for r in plot_data if r["figure"] == "figure_nort"} >= {0, 50}
    assert (tmp_path / "first/figure_oft.svg").is_file()
    assert "source_table" in (tmp_path / "first/run_manifest.json").read_text()
    assert "/data/" not in (tmp_path / "first/run_manifest.json").read_text()


def test_unresolved_decisions_block_only_dependants_and_cannot_leak_fixture_defaults(tmp_path):
    path = create_fixture(tmp_path / "fixture")
    config = json.loads(path.read_text())
    rows, _ = load_inputs(config, path.parent)
    config["run_mode"] = "production"
    assert choice(config, "assays.ymaze.triad_policy", "D01") is None
    scored, _ = score_records(rows, config)
    ymaze = next(r for r in scored if r["assay"] == "ymaze")
    nort = next(r for r in scored if r["assay"] == "nort" and r["phase"] == "test")
    assert ymaze["spontaneous_alternation_pct"] is None
    assert ymaze["missing_reasons"]["spontaneous_alternation_pct"] == "blocked:D01"
    assert nort["DI"] is not None  # arithmetic independent of approvals
    assert nort["qc_flags"]["passes_15s_exploration_threshold"] is None


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_metadata", "Duplicate"),
        ("duplicate_measurement", "Duplicate"),
        ("unit", "units"),
        ("inconsistent_animal", "Inconsistent"),
    ],
)
def test_ingestion_failures(tmp_path, mutation, match):
    path = create_fixture(tmp_path / "fixture")
    config = json.loads(path.read_text())
    meta_path = path.parent / "metadata.json"
    meta = json.loads(meta_path.read_text())
    if mutation == "duplicate_metadata":
        meta.append(copy.deepcopy(meta[0]))
        meta_path.write_text(json.dumps(meta))
    elif mutation == "duplicate_measurement":
        input_path = path.parent / "ymaze.json"
        data = json.loads(input_path.read_text())
        data.append(data[0])
        input_path.write_text(json.dumps(data))
    elif mutation == "unit":
        config["inputs"]["units_and_calibration"]["oft"]["time"] = "frames"
    else:
        meta[1]["sex"] = "different"
        meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=match):
        rows, _ = load_inputs(config, path.parent)
        score_records(rows, config)


def test_unresolved_primary_threshold_keeps_per_threshold_arithmetic(tmp_path):
    path = create_fixture(tmp_path / "fixture")
    config = json.loads(path.read_text())
    config["decision_resolutions"].pop("D03")
    rows, _ = load_inputs(config, path.parent)
    scored, _ = score_records(rows, config)
    ymaze = next(r for r in scored if r["assay"] == "ymaze")
    assert ymaze["spontaneous_alternation_pct"] is not None
    assert ymaze["primary_threshold"] is None


def test_absent_optional_inputs_preserve_metadata_and_missingness(tmp_path):
    path = create_fixture(tmp_path / "fixture")
    config = json.loads(path.read_text())
    config["inputs"]["oft_path"] = None
    rows, _ = load_inputs(config, path.parent)
    scored, _ = score_records(rows, config)
    oft = next(r for r in scored if r["assay"] == "oft")
    assert oft["freezing_s"] is None
    assert oft["missing_reasons"]["freezing_s"] == "not_measured"
