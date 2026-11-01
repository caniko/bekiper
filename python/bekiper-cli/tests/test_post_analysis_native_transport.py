import sys
from pathlib import Path

import pytest

from bekiper_cli.post_analysis.native import run


def fake_engine(tmp_path: Path, body: str) -> str:
    program = tmp_path / "native engine"
    program.write_text(f"#!{sys.executable}\n{body}\n")
    program.chmod(0o755)
    return str(program)


def test_transport_preserves_argument_boundaries_and_empty_sensitivity(tmp_path):
    program = fake_engine(tmp_path, "import json, sys; print(json.dumps({'status':'complete','args':sys.argv[1:]}))")
    result = run(
        Path("analysis config;literal.json"),
        output=Path("fresh results"),
        requested_outputs=["metrics", "models"],
        sensitivity_thresholds=[],
        executable=program,
    )
    assert result["args"] == [
        "--config",
        "analysis config;literal.json",
        "--json",
        "--output",
        "fresh results",
        "--request",
        "metrics",
        "--request",
        "models",
        "--no-sensitivity",
    ]


def test_transport_reports_engine_failure(tmp_path):
    program = fake_engine(tmp_path, "import sys; print('invalid supplied duration',file=sys.stderr); sys.exit(2)")
    with pytest.raises(ValueError, match="invalid supplied duration"):
        run(Path("config.json"), executable=program)


def test_transport_does_not_mutate_process_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BEKIPER_POST_ANALYSIS_REQUESTED_OUTPUTS", "figures")
    program = fake_engine(
        tmp_path,
        "import json, os; print(json.dumps({'status':'complete','outputs':os.getenv('BEKIPER_POST_ANALYSIS_REQUESTED_OUTPUTS')}))",
    )
    result = run(Path("config.json"), executable=program, env={})
    assert result["outputs"] is None
    import os

    assert os.environ["BEKIPER_POST_ANALYSIS_REQUESTED_OUTPUTS"] == "figures"


def test_retired_console_script_cannot_recurse(tmp_path):
    program = fake_engine(tmp_path, "from bekiper_cli.post_analysis.__main__ import main; main()")
    with pytest.raises(OSError, match="Retired Python"):
        run(Path("config.json"), executable=program)
