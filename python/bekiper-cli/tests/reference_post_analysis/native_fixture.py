"""Generate explicitly synthetic registrations for the native cohort example.

Run with PYTHONPATH=python/bekiper-cli/tests python -m
reference_post_analysis.native_fixture --directory <fresh-input-directory>.
No numerical analysis or report generation runs in this fixture producer.
"""

import argparse
import json
from pathlib import Path

from .synthetic import create_fixture


def create_native_fixture(directory: Path) -> Path:
    plan_path = create_fixture(directory)
    plan = json.loads(plan_path.read_text())
    for assay in ("oft", "ymaze", "nort"):
        plan["inputs"][f"{assay}_path"] = "must-not-read-configured-measurements.json"
    plan["inclusion"]["upstream_verification_requirement"] = "allow_unverified"
    roles = {
        "training": {"TA": "left", "TB": "right"},
        "test": {"TN": "left", "TF": "right"},
        "sessions": [
            {
                "animal_id": "SYNTHETIC_1_0",
                "assay": "nort",
                "session_id": "SYNTHETIC_1_0_nort_post_test",
                "timepoint": "post",
                "phase": "test",
                "roles": {"TN": "right", "TF": "left"},
            }
        ],
    }
    plan["assays"]["nort"]["object_role_and_location_mapping"] = roles
    plan["assays"]["ymaze"]["window_and_tracking_policy"] = {
        "interval": "half_open",
        "origin": "first_detection_s",
        "duration": "analysis_duration_s",
        "out_of_window": "reject",
        "missing_pose": "observed_samples_break_on_missing_no_interpolation",
    }
    plan["assays"]["nort"]["window_and_tracking_policy"] = {
        "interval": "half_open",
        "origin": "analysis_start_s",
        "duration": "analysis_duration_s",
        "out_of_window": "reject",
        "protocol_departures": "flag_only",
    }
    for decision, paths in {
        "D10": ["assays.nort.object_role_and_location_mapping"],
        "D12": ["inclusion.upstream_verification_requirement"],
        "D04": ["assays.ymaze.window_and_tracking_policy"],
        "D14": ["assays.nort.window_and_tracking_policy"],
    }.items():
        for path in paths:
            value = plan
            for part in path.split("."):
                value = value[part]
            registration = plan["decision_resolutions"].setdefault(
                decision,
                {
                    "origin": "development_fixture",
                    "rationale": "Fresh native synthetic cohort only",
                    "approval_or_plan_reference": "reference_post_analysis.native_fixture",
                    "scope": ["assays"],
                    "value": {},
                },
            )
            registration["value"][path] = value
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for row in metadata:
        row["qc_flags"] = {"metadata_warning": True, "source_warning": False}
        if row["assay"] == "nort":
            row["object_configuration"] = {"objects": {"left": {}, "right": {}}}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    project_path = directory / "bekiper_config.toml"
    project_path.write_text("""[project]
project_name = "Native synthetic three-assay cohort"
data_directory = "."
[aggregate]
enabled = false
[post_analysis]
enabled = true
plan_path = "config.json"
run_id = "native-three-assay"
[post_analysis.figures]
enabled = true
selection = ["figure_oft", "figure_ymaze", "figure_nort", "supplement_ymaze"]
formats = ["svg", "png", "pdf"]
dpi = 180
[post_analysis.report]
enabled = true
title = "Native synthetic OFT / Y-maze / NORT"
compiler = "lualatex"
""")
    return project_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    print(create_native_fixture(parser.parse_args().directory))
