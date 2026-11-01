"""Design-system inputs (tokens.json, archetypes.json) must match their sources."""

import subprocess
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[3] / "design/review-package/build"


@pytest.mark.skipif(not BUILD.is_dir(), reason="design/ is not part of a packaged install")
@pytest.mark.parametrize("flag", ["--self-test", "--check"])
def test_gen_py(flag):
    run = subprocess.run([sys.executable, str(BUILD / "gen.py"), flag], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr or run.stdout
