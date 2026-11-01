"""Compatibility transport to the native engine; no analysis is performed here."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def run(
    config_path: Path,
    *,
    output: Path | None = None,
    requested_outputs: list[str] | None = None,
    sensitivity_thresholds: list[int] | None = None,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    child_env = dict(os.environ)
    if env is not None:
        child_env = {key: value for key, value in child_env.items() if not key.startswith("BEKIPER_POST_ANALYSIS_")}
        child_env.update(env)
    program = executable or child_env.get("BEKIPER_POST_ANALYSIS_BIN") or shutil.which("bekiper-post-analysis")
    if not program:
        raise OSError(
            "Native bekiper-post-analysis is required; install the Bekiper Nix package or build the Rust crate"
        )
    # Old editable installations may still have the retired Python console
    # script on PATH. Refuse recursive invocation with an actionable error.
    with Path(program).open("rb") as stream:
        prefix = stream.read(4096)
    if b"bekiper_cli.post_analysis.__main__" in prefix:
        raise OSError(
            "Retired Python bekiper-post-analysis script found; refresh the installation and use the native binary"
        )
    command = [program, "--config", str(config_path), "--json"]
    if output is not None:
        command.extend(["--output", str(output)])
    if requested_outputs is not None:
        if not requested_outputs:
            raise ValueError("requested_outputs must be nonempty")
        for request in requested_outputs:
            command.extend(["--request", request])
    if sensitivity_thresholds == []:
        command.append("--no-sensitivity")
    elif sensitivity_thresholds is not None:
        for threshold in sensitivity_thresholds:
            command.extend(["--sensitivity-threshold", str(threshold)])
    result = subprocess.run(command, env=child_env, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"Native post-analysis exited with status {result.returncode}")
    manifest = json.loads(result.stdout)
    if not isinstance(manifest, dict) or "status" not in manifest:
        raise ValueError("Native engine returned an invalid manifest")
    return manifest
