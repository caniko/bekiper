"""Explicit adapters for Bekiper aggregate tables and canonical session records."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .metrics import COUNT_FIELDS, Record

KEY = ("animal_id", "assay", "session_id", "timepoint", "phase")
FACTORS = ("five_xfad", "hcar1", "treatment")
ALIASES = {
    "animal": "animal_id",
    "Animal ID": "animal_id",
    "5XFAD_status": "five_xfad",
    "HCAR1_status": "hcar1",
    "5XFAD": "five_xfad",
    "HCAR1": "hcar1",
    "total_mean_speed": "mean_speed_m_per_s",
    "center_duration_s": "center_time_s",
    "latency_to_center_s": "latency_to_first_center_entry_s",
    "novel_s": "TN",
    "familiar_s": "TF",
    "pct_novel": "percent_novel",
    "log_ratio": "LR",
    "total_duration_s": "analysis_duration_s",
}
NUMERIC = {
    "TA",
    "TB",
    "TN",
    "TF",
    "DI",
    "RI",
    "LR",
    "object_bias",
    "side_bias",
    "percent_novel",
    "threshold_pct",
    "proximal_threshold",
} | COUNT_FIELDS
METADATA_FIELDS = (*FACTORS, "group", "sex", "age_treatment_window", "excluded", "exclusion_reason", "pair_id")
OUTPUT_IDS = {
    "metrics",
    "models",
    "correlations",
    "within_group_tests",
    "figures",
    "subgroup_inference",
    "spatial_location_bias",
}


def reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(), parse_constant=reject_constant)


def fingerprint(path: Path) -> Record:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def choice(config: Record, path: str, decision: str) -> Any:
    """An unset or unapproved setting cannot become a production default."""
    value: Any = config
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None:
        return None
    record = config.get("decision_resolutions", {}).get(decision, {})
    required = ["value", "scope", "origin", "rationale", "approval_or_plan_reference"]
    if any(not record.get(key) for key in required):
        return None
    if record["origin"] == "development_fixture":
        if config.get("run_mode") != "synthetic" or config.get("synthetic_fixture") is not True:
            return None
    elif record["origin"] not in ["existing_analysis_plan", "researcher_decision"]:
        return None
    if record["value"].get(path) != value:
        return None
    scopes = record["scope"]
    if not isinstance(scopes, list) or not any(path == scope or path.startswith(scope + ".") for scope in scopes):
        return None
    return value


def resolved_config(
    path: Path,
    *,
    output: Path | None = None,
    requested_outputs: list[str] | None = None,
    sensitivity_thresholds: list[int] | None = None,
    env: Mapping[str, str] | None = None,
) -> Record:
    config = load_json(path)
    if not isinstance(config, dict) or config.get("run_mode") not in ["development", "production", "synthetic"]:
        raise ValueError("Config run_mode must be development, production or synthetic")
    lookup = os.environ if env is None else env
    config.setdefault("outputs", {})["directory"] = (
        str(output)
        if output
        else lookup.get(
            "BEKIPER_POST_ANALYSIS_OUTPUT", config.get("outputs", {}).get("directory") or "post_analysis_output"
        )
    )
    requested = requested_outputs
    if requested is None:
        raw = lookup.get("BEKIPER_POST_ANALYSIS_REQUESTED_OUTPUTS")
        requested = (
            [part.strip() for part in raw.split(",")] if raw is not None else config["outputs"].get("requested_outputs")
        )
    if requested is not None and (
        not isinstance(requested, list)
        or not requested
        or any(not isinstance(item, str) or item not in OUTPUT_IDS for item in requested)
        or len(requested) != len(set(requested))
    ):
        raise ValueError(f"requested_outputs must be a nonempty unique list from {sorted(OUTPUT_IDS)}")
    config["outputs"]["requested_outputs"] = requested
    ymaze = config.setdefault("assays", {}).setdefault("ymaze", {})
    thresholds = sensitivity_thresholds
    if thresholds is None:
        raw = lookup.get("BEKIPER_POST_ANALYSIS_SENSITIVITY_THRESHOLDS")
        thresholds = (
            ([int(part.strip()) for part in raw.split(",")] if raw else [])
            if raw is not None
            else ymaze.get("requested_sensitivity_thresholds_arm_length_pct", [25, 45, 65])
        )
    validate_sensitivity_thresholds(thresholds)
    ymaze["requested_sensitivity_thresholds_arm_length_pct"] = thresholds
    return dict(config)


def validate_sensitivity_thresholds(thresholds: Any) -> None:
    if (
        not isinstance(thresholds, list)
        or any(type(t) is not int or t not in [25, 45, 65] for t in thresholds)
        or len(set(thresholds)) != len(thresholds)
    ):
        raise ValueError("Requested sensitivity thresholds must be a unique list of 25/45/65 (empty disables panels)")


def subgroup_plan(config: Record) -> Record | None:
    path = "statistics.subgroup_analysis_plan"
    plan = choice(config, path, "D09")
    if plan is None or choice(config, path, "D12") is None:
        return None
    if not isinstance(plan, dict) or not isinstance(plan.get("strata"), list) or not plan["strata"]:
        raise ValueError("Subgroup plan requires explicit strata")
    seen = set()
    for stratum in plan["strata"]:
        if not isinstance(stratum, dict) or stratum.get("field") not in ["sex", "age_treatment_window", "treatment"]:
            raise ValueError("Subgroup strata require sex, age_treatment_window or treatment")
        levels, assays = stratum.get("levels"), stratum.get("assays")
        if (
            not isinstance(levels, list)
            or not levels
            or any(not isinstance(v, str) or not v for v in levels)
            or len(set(levels)) != len(levels)
            or not isinstance(assays, list)
            or not assays
            or any(a not in ["oft", "ymaze", "nort"] for a in assays)
            or len(set(assays)) != len(assays)
        ):
            raise ValueError("Subgroup strata require unique explicit levels and assays")
        for level in levels:
            key = (stratum["field"], level)
            if key in seen:
                raise ValueError("Duplicate subgroup stratum")
            seen.add(key)
    outputs, correction = plan.get("outputs"), plan.get("multiplicity", {})
    if (
        not isinstance(outputs, list)
        or not outputs
        or len(set(outputs)) != len(outputs)
        or any(o not in ["models", "correlations", "within_group_tests"] for o in outputs)
        or not isinstance(correction, dict)
        or correction.get("method") != "holm"
        or not isinstance(correction.get("family"), str)
        or not correction["family"]
    ):
        raise ValueError("Subgroup inference requires explicit outputs and a Holm family")
    return plan


def requested_output_ids(config: Record, rows: list[Record]) -> list[str]:
    requested = config.get("outputs", {}).get("requested_outputs")
    if requested is not None:
        if not isinstance(requested, list) or not requested or any(item not in OUTPUT_IDS for item in requested):
            raise ValueError("Invalid requested_outputs")
        return list(requested)
    assays = {row["assay"] for row in rows}
    requested = ["metrics", "models", "figures"]
    if "ymaze" in assays:
        requested.append("correlations")
    if "nort" in assays:
        requested.append("within_group_tests")
    if config.get("statistics", {}).get("subgroup_analysis_plan") is not None:
        requested.append("subgroup_inference")
    return requested


def read_table(path: Path, sheet: str | None = None) -> list[Record]:
    if path.suffix == ".json":
        data = load_json(path)
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError("JSON input must be an array of session objects")
        return data
    if path.suffix == ".csv":
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("Missing or duplicate CSV headers")
            return list(reader)
    if path.suffix == ".parquet":
        import polars as pl

        return pl.read_parquet(path).to_dicts()
    if path.suffix == ".ods":
        return read_ods(path, sheet)
    raise ValueError(f"Unsupported table extension {path.suffix}; export normalized metadata to CSV/JSON")


def read_ods(path: Path, sheet: str | None) -> list[Record]:
    if not sheet:
        raise ValueError("ODS requires an explicit sheet")
    ns = {
        "t": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
        "o": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    }
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("content.xml"))
    table = next((t for t in root.findall(".//t:table", ns) if t.get(f"{{{ns['t']}}}name") == sheet), None)
    if table is None:
        raise ValueError(f"ODS sheet {sheet!r} not present")
    rows = []
    for row in table.findall("t:table-row", ns):
        cells = []
        for cell in row:
            repeat = int(cell.get(f"{{{ns['t']}}}number-columns-repeated", "1"))
            if repeat > 16384:
                raise ValueError("Oversized ODS cell repeat")
            kind = cell.get(f"{{{ns['o']}}}value-type")
            value: Any = "".join(cell.itertext()).strip() or None
            if kind == "float":
                value = float(cell.get(f"{{{ns['o']}}}value", "nan"))
            elif kind == "boolean":
                value = cell.get(f"{{{ns['o']}}}boolean-value") == "true"
            cells.extend([value] * repeat)
        if any(c is not None for c in cells):
            repeat = int(row.get(f"{{{ns['t']}}}number-rows-repeated", "1"))
            if repeat > 100000:
                raise ValueError("Oversized ODS row repeat")
            rows.extend([cells] * repeat)
    if not rows or len(rows[0]) != len(set(rows[0])):
        raise ValueError("Empty ODS sheet or duplicate headers")
    return [dict(zip(rows[0], row, strict=True)) for row in rows[1:]]


def normalize(row: Record, mapping: Record) -> Record:
    result: Record = {}
    for source, value in row.items():
        target = mapping.get(source, ALIASES.get(source, source))
        if target in result:
            raise ValueError(f"Column alias collision for {target}")
        if value == "":
            value = None
        if (target in NUMERIC or target.endswith(("_s", "_m", "_pct", "_m_per_s"))) and isinstance(value, str):
            value = float(value)
        if target in ["excluded", "passes_15s_exploration_threshold"] and isinstance(value, str):
            if value.lower() not in ["true", "false"]:
                raise ValueError(f"{target}: must be boolean")
            value = value.lower() == "true"
        if target == "scoring_provenance" and isinstance(value, str):
            value = json.loads(value, parse_constant=reject_constant)
        if target in ["valid_arm_entry_sequence", "raw_arm_entry_sequence"] and isinstance(value, str):
            value = value.split() if " " in value else list(value)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{target}: nonfinite supplied value")
        result[target] = value
    for name in ["animal_id", "session_id", "pair_id", "group"] + list(FACTORS):
        if result.get(name) is not None:
            value = result[name]
            result[name] = (
                str(int(value))
                if isinstance(value, (float, int)) and not isinstance(value, bool) and int(value) == value
                else str(value)
            )
    return result


def session_key(row: Record) -> tuple[str, ...]:
    if any(row.get(key) is None or row[key] == "" for key in KEY):
        raise ValueError(f"Session key requires {KEY}; map assay/timepoint/phase explicitly")
    if row["assay"] not in ["oft", "ymaze", "nort"]:
        raise ValueError("Canonical assay must be oft, ymaze or nort")
    if row["timepoint"] not in ["pre", "post"]:
        raise ValueError("Canonical timepoint must be pre or post; map source context explicitly")
    phases = ["training", "test"] if row["assay"] == "nort" else ["session"]
    if row["phase"] not in phases:
        raise ValueError(f"Canonical phase for {row['assay']} must be one of {phases}")
    return tuple(str(row[key]) for key in KEY)


def validate_metadata(rows: list[Record], config: Record) -> dict[tuple[str, ...], Record]:
    indexed: dict[tuple[str, ...], Record] = {}
    biology: dict[str, tuple[Any, ...]] = {}
    levels = config["inputs"].get("factor_levels", {})
    groups = config["inputs"].get("group_mapping", {})
    for row in rows:
        key = session_key(row)
        if key in indexed:
            raise ValueError("Duplicate animal/assay/session/timepoint/phase metadata key")
        required = list(FACTORS) + ["group", "sex", "age_treatment_window", "excluded", "exclusion_reason"]
        if any(name not in row for name in required):
            raise ValueError(f"Metadata requires {required}")
        if not isinstance(row["excluded"], bool) or (row["excluded"] and not row["exclusion_reason"]):
            raise ValueError("Exclusion needs a boolean and an explicit reason")
        if not row["group"]:
            raise ValueError("Explicit group identity is required")
        for name in FACTORS:
            if not row[name] or (levels.get(name) and row[name] not in levels[name]):
                raise ValueError(f"Unmapped factor level for {name}")
        if groups and (row["group"] not in groups or any(groups[row["group"]].get(f) != row[f] for f in FACTORS)):
            raise ValueError("Group/factor mapping conflicts with metadata")
        # Biological/treatment identity must round-trip across paired sessions.
        identity = tuple(row[f] for f in FACTORS) + (row["sex"], row["group"])
        if row["animal_id"] in biology and biology[row["animal_id"]] != identity:
            raise ValueError("Inconsistent animal factors across sessions")
        biology[row["animal_id"]] = identity
        indexed[key] = row
    return indexed


def load_inputs(config: Record, base: Path) -> tuple[list[Record], list[Record]]:
    inputs = config["inputs"]
    if not inputs.get("metadata_path"):
        raise ValueError("metadata_path is required; identifiers are never inferred from row order")
    descriptors: list[Record] = []
    evidence_by_assay = {}
    for assay in ["oft", "nort"]:
        evidence = inputs.get("upstream_rule_provenance", {}).get(assay, {}).get("evidence_files", [])
        files, audit = [], None
        roles = set()
        for spec in evidence:
            if spec.get("role") in roles or not spec.get("role") or not spec.get("sha256"):
                raise ValueError("Rule evidence requires unique roles and pinned SHA-256 fingerprints")
            roles.add(spec["role"])
            path = base / spec["path"]
            actual = fingerprint(path)
            if actual["sha256"] != spec["sha256"]:
                raise ValueError(f"{assay}: rule evidence fingerprint mismatch for {path.name}")
            files.append({**actual, "role": spec["role"]})
            descriptors.append({**actual, "role": f"{assay}_evidence:{spec['role']}"})
            if spec["role"] == "rule_audit":
                audit = load_json(path)
        evidence_by_assay[assay] = {"_rule_evidence_files": files, "_rule_audit": audit}

    def table(spec: Any, name: str) -> list[Record]:
        descriptor: Record = {"path": spec} if isinstance(spec, str) else spec
        path = base / descriptor["path"]
        data = read_table(path, descriptor.get("sheet"))
        descriptors.append({**fingerprint(path), "role": name, "sheet": descriptor.get("sheet")})
        constants: Record = descriptor.get("constants", {})
        normalized = [normalize({**constants, **r}, inputs.get("column_mapping", {}).get(name, {})) for r in data]
        for row in normalized:
            if any(field in row for field in ["_rule_audit", "_rule_evidence_files"]):
                raise ValueError("Rule evidence is adapter-owned and cannot be supplied by a measurement row")
            for field, mapping in inputs.get("session_key_mapping", {}).items():
                if field not in ["assay", "phase", "timepoint"] or not isinstance(mapping, dict):
                    raise ValueError("session_key_mapping requires assay/phase/timepoint value dictionaries")
                if row.get(field) in mapping:
                    row[field] = mapping[row[field]]
        return normalized

    meta = validate_metadata(table(inputs["metadata_path"], "metadata"), config)
    measurements: dict[tuple[str, ...], list[Record]] = {}
    seen = set()
    for assay in ["oft", "ymaze", "nort"]:
        spec = inputs.get(f"{assay}_path")
        if not spec:
            continue
        specs = spec if isinstance(spec, list) else [spec]
        for descriptor in specs:
            for row in table(descriptor, assay):
                key = session_key(row)
                if row["assay"] != assay or key not in meta:
                    raise ValueError("Measurement has no unique matching metadata session")
                threshold = row.get("threshold_pct")
                if assay == "ymaze":
                    if threshold is None and row.get("proximal_threshold") is not None:
                        threshold = row["proximal_threshold"] * 100
                        row["threshold_pct"] = threshold
                    if threshold not in [25, 45, 65]:
                        raise ValueError("Principal Y-maze inputs need explicit threshold_pct 25/45/65")
                full_key = key + (threshold,)
                if full_key in seen:
                    raise ValueError("Duplicate measurement session/threshold; many-to-many join forbidden")
                seen.add(full_key)
                for field in METADATA_FIELDS:
                    if field in row and (field not in meta[key] or row[field] != meta[key][field]):
                        raise ValueError(f"Measurement metadata mismatch for {field}")
                measurements.setdefault(key, []).append(row)
    records = []
    for key, metadata in meta.items():
        for measurement in measurements.get(key, [{}]):
            records.append(
                {
                    **metadata,
                    **measurement,
                    **evidence_by_assay.get(metadata["assay"], {}),
                    "measurement_available": bool(measurement),
                }
            )
    return records, descriptors
