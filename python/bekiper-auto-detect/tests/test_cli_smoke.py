from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bekiper_auto_detect.__main__ import main


def test_smoke_accepts_valid_recipe_backend_without_video_or_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "perimeters.toml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bekiper_auto_detect",
            "--smoke",
            "--recipe",
            "nort_training",
            "--backend",
            "sam2_auto",
            "--output",
            str(output),
        ],
    )

    main()

    assert not output.exists()


def test_smoke_rejects_invalid_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bekiper_auto_detect",
            "--smoke",
            "--recipe",
            "nort_training",
            "--backend",
            "not_a_backend",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
