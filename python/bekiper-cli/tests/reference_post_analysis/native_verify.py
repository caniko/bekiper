"""Independent artifact/numerical checks for the fresh native cohort example."""

import argparse
import copy
import hashlib
import itertools
import json
import math
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats


def read(directory, name):
    return json.loads(artifact_path(directory, name).read_text())


def artifact_path(directory, name):
    manifest_path = directory / "run_manifest.json"
    if name != "run_manifest.json" and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        name = manifest.get("package_layout", {}).get("paths", {}).get(name, name)
    return directory / name


def verify_package(directory):
    files = read(directory, "artifact_manifest.json")["files"]
    for item in files:
        path = directory / item["path"]
        assert path.is_file() and not path.is_symlink()
        content = path.read_bytes()
        assert len(content) == item["bytes"]
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
    assert {item["path"] for item in files} == {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "artifact_manifest.json"
    }
    return len(files)


def restore_toml(value, nulls, integers):
    """Recover original JSON types using the declared RFC 6901 locations."""
    value = copy.deepcopy(value)
    for paths, restore in [(nulls, lambda _: None), (integers, int)]:
        for pointer in paths:
            keys = [key.replace("~1", "/").replace("~0", "~") for key in pointer.split("/")[1:]]
            node = value
            for key in keys[:-1]:
                node = node[int(key)] if isinstance(node, list) else node[key]
            key = int(keys[-1]) if isinstance(node, list) else keys[-1]
            node[key] = restore(node[key])
    return value


def verify_human_exports(directory):
    manifest = read(directory, "run_manifest.json")
    rows = read(directory, "metrics.json")
    human = tomllib.loads((directory / "metrics.toml").read_text())
    parameters = tomllib.loads((directory / "parameters.toml").read_text())
    seen = set()
    identity = {"animal_id", "assay", "session_id", "timepoint", "phase"}
    for assay in ("oft", "ymaze", "nort"):
        for session in human[assay]["sessions"]:
            index = session["source_record_index"]
            assert index not in seen and rows[index]["assay"] == assay
            seen.add(index)
            session = restore_toml(session, session["null_fields"], session["large_integer_fields"])
            reconstructed = {key: session[key] for key in identity if key in session}
            for key, fields in session.items():
                if key in identity | {"source_record_index", "null_fields", "large_integer_fields"}:
                    continue
                assert not reconstructed.keys() & fields.keys(), "duplicate canonical field"
                reconstructed.update(fields)
            assert reconstructed == rows[index], f"human metric parity: {index}"
    assert seen == set(range(len(rows)))
    encoding = parameters["representation"]
    parameters = restore_toml(parameters, encoding["null_fields"], encoding["large_integer_fields"])
    reserved = parameters.get("reserved_configuration_fields", {})
    for key, value in manifest["resolved_config"].items():
        assert (reserved[key] if key in reserved else parameters[key]) == value, f"parameter parity: {key}"
    assert parameters["code"] == manifest["code"]
    index = (directory / "README.md").read_text()
    for suffix in index.split("](")[1:]:
        assert (directory / suffix.split(")", 1)[0]).is_file(), "broken package-index link"
    return len(seen)


def independent_model(model, rows):
    """Fit cell means with an optional baseline; do not reuse Rust treatment coding."""
    keys = ("animal_id", "assay", "session_id", "timepoint", "phase", "threshold_pct")
    index = {tuple(row.get(key) for key in keys): row for row in rows}
    included = [index[tuple(item.get(key) for key in keys)] for item in model["subset"] if item["reason"] is None]
    factors = ("five_xfad", "hcar1", "treatment")
    cells = list(itertools.product(*(model["factor_levels"][factor] for factor in factors)))
    matrix = np.zeros((len(included), len(cells)))
    for i, row in enumerate(included):
        matrix[i, cells.index(tuple(row[factor] for factor in factors))] = 1
    baseline = "pre_outcome" in model["coefficient_names"]
    if baseline:
        pre = {
            (row["animal_id"], row["assay"], row["phase"], row.get("threshold_pct")): row
            for row in rows
            if row["timepoint"] == "pre"
        }
        matrix = np.column_stack(
            [
                matrix,
                [
                    pre[(row["animal_id"], row["assay"], row["phase"], row.get("threshold_pct"))][model["endpoint"]]
                    for row in included
                ],
            ]
        )
    response = np.array([row[model["endpoint"]] for row in included])
    q, r = np.linalg.qr(matrix, mode="reduced")
    beta = np.linalg.solve(r, q.T @ response)
    residual = response - matrix @ beta
    degrees = len(response) - matrix.shape[1]
    variance = residual @ residual / degrees
    inverse_r = np.linalg.solve(r, np.eye(len(r)))
    covariance = variance * (inverse_r @ inverse_r.T)
    contrast = np.zeros(matrix.shape[1])
    groups = {row["group"]: tuple(row[factor] for factor in factors) for row in included}
    for group, weight in {"G5": 1, "G6": -1, "G1": -1, "G2": 1}.items():
        contrast[cells.index(groups[group])] += weight
    estimate = contrast @ beta
    standard_error = math.sqrt(contrast @ covariance @ contrast)
    probability = float(2 * stats.t.sf(abs(estimate / standard_error), degrees))
    assert model["n_animals"] == len(included)
    assert model["residual_df"] == degrees
    assert math.isclose(model["contrast"]["estimate"], estimate, rel_tol=1e-9, abs_tol=1e-10)
    assert math.isclose(model["contrast"]["standard_error"], standard_error, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(model["contrast"]["p_raw"], probability, rel_tol=1e-8, abs_tol=1e-10)
    assert model["contrast"]["weights"] == {"G5": 1, "G6": -1, "G1": -1, "G2": 1}
    return model["model_id"]


def verify_cohort(root):
    directory = root / "data/_post_analysis/native-three-assay"
    manifest = read(directory, "run_manifest.json")
    rows = read(directory, "metrics.json")
    native = read(root / "data", "native_trials.json")
    assert len(native) == 317 and len(rows) == 318
    assert manifest["run_mode"] == "synthetic" and manifest["real_data_execution"] is False
    assert manifest["status"] == "partial"
    if "package_layout" in manifest:
        assert verify_human_exports(directory) == len(rows)
    assert not (root / "results.ods").exists()
    assert {row["assay"] for row in rows} == {"oft", "ymaze", "nort"}
    assert any(row["excluded"] for row in rows)
    assert any(row.get("upstream_qc_join_conflicts") for row in rows)
    assert any(not row["measurement_available"] for row in rows)
    assert any(row.get("disagreements") for row in rows)
    ymaze = [row for row in rows if row["assay"] == "ymaze" and row["measurement_available"]]
    assert {row["threshold_pct"] for row in ymaze} == {25, 45, 65}
    assert all(row["removed_short_entries"] == 0 for row in ymaze)
    assert all(row["input_verification"]["status"] == "verified_clarified_entry_definition" for row in ymaze)
    assert read(directory, "event_bout_debug.json")
    pairing = read(directory, "pairing_ledger.json")
    assert any(item["animal_id"] == "SYNTHETIC_1_1" and item["assay"] == "ymaze" and item["reason"] for item in pairing)
    exact = next(
        row for row in rows if row["animal_id"] == "SYNTHETIC_1_0" and row["assay"] == "nort" and row["phase"] == "test"
    )
    assert exact["resolved_object_roles"] == {"TN": "right", "TF": "left"}
    assert exact["TN"] == exact["right_exploration_s"] and exact["TF"] == exact["left_exploration_s"]
    nort = [row for row in rows if row["assay"] == "nort"]
    assert all(row["analysis_duration_s"] == 24 and row["analysis_window_s"] == 360 for row in nort)
    assert any(row.get("upstream_DI") != row.get("recomputed_DI") for row in nort if row["phase"] == "test")
    for row in nort:
        if row["phase"] == "test":
            assert math.isclose(row["DI"], (row["TN"] - row["TF"]) / (row["TN"] + row["TF"]), abs_tol=1e-12)
    for row in rows:
        if row["assay"] == "oft":
            assert math.isclose(row["center_time_pct"], row["center_time_s"] / 300 * 100, abs_tol=1e-12)
    selected = {"figure_oft", "figure_ymaze", "figure_nort", "supplement_ymaze"}
    assert {figure["name"] for figure in manifest["figures"]} == selected
    assert len(manifest["unrendered_figures"]) == 3
    for figure in manifest["figures"]:
        assert figure["dpi"] == 180
        with Image.open(artifact_path(directory, f"{figure['name']}.png")) as image:
            assert abs(image.info["dpi"][0] - 180) < 0.02
    assert (directory / "report.pdf").read_bytes().startswith(b"%PDF-")
    models = read(directory, "models.json")
    verified = [independent_model(model, rows) for model in models if model["status"] == "fitted"]
    assert len(verified) == 4
    return {
        "native_trials": len(native),
        "session_rows": len(rows),
        "artifact_members": verify_package(directory),
        "independent_qr_models": verified,
        "report_bytes": (directory / "report.pdf").stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    print(json.dumps(verify_cohort(parser.parse_args().run_root), indent=2))
