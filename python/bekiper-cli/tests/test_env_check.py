"""Tests for the bekiper-cli startup environment diagnostic.

Uses injected stub environment and filesystem predicates so tests are
thread-safe and never touch the real process state.
"""

from __future__ import annotations

from pathlib import Path

from bekiper_cli.env_check import EnvCheckReport, check_env


def _fake_exists(existing: set[str]) -> callable:
    """Build a predicate that reports ``True`` only for paths in ``existing``."""

    def predicate(p: Path) -> bool:
        return str(p) in existing

    return predicate


class TestReport:
    def test_empty_report_is_clean(self):
        r = EnvCheckReport()
        assert r.is_clean()

    def test_hard_issue_not_clean(self):
        r = EnvCheckReport(hard_issues=["x"])
        assert not r.is_clean()

    def test_soft_missing_not_clean(self):
        r = EnvCheckReport(soft_missing=["x"])
        assert not r.is_clean()


class TestCheckEnv:
    def test_clean_env_has_no_issues(self):
        env = {
            "ORT_DYLIB_PATH": "/nix/store/xxx/lib/libonnxruntime.so",
            "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
            "PYTORCH_ROCM_ARCH": "gfx1100",
        }
        report = check_env(
            env=env,
            path_exists=_fake_exists({"/nix/store/xxx/lib/libonnxruntime.so"}),
        )
        assert report.is_clean(), f"expected clean, got {report!r}"

    def test_missing_ort_dylib_is_hard_issue(self):
        report = check_env(env={}, path_exists=_fake_exists(set()))
        assert any("unset" in s for s in report.hard_issues)

    def test_empty_ort_dylib_is_hard_issue(self):
        report = check_env(
            env={"ORT_DYLIB_PATH": ""},
            path_exists=_fake_exists(set()),
        )
        assert any("unset" in s for s in report.hard_issues)

    def test_nonexistent_ort_dylib_is_hard_issue(self):
        report = check_env(
            env={"ORT_DYLIB_PATH": "/does/not/exist.so"},
            path_exists=_fake_exists(set()),
        )
        assert any("does not exist" in s for s in report.hard_issues)

    def test_existing_ort_dylib_yields_no_hard_issue(self):
        report = check_env(
            env={"ORT_DYLIB_PATH": "/real/libonnxruntime.so"},
            path_exists=_fake_exists({"/real/libonnxruntime.so"}),
        )
        assert not report.hard_issues

    def test_missing_rocm_vars_are_soft_hints(self):
        report = check_env(
            env={"ORT_DYLIB_PATH": "/ok.so"},
            path_exists=_fake_exists({"/ok.so"}),
        )
        assert not report.hard_issues
        assert "HSA_OVERRIDE_GFX_VERSION" in report.soft_missing
        assert "PYTORCH_ROCM_ARCH" in report.soft_missing

    def test_partially_set_rocm_vars(self):
        report = check_env(
            env={
                "ORT_DYLIB_PATH": "/ok.so",
                "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
            },
            path_exists=_fake_exists({"/ok.so"}),
        )
        assert not report.hard_issues
        assert report.soft_missing == ["PYTORCH_ROCM_ARCH"]

    def test_empty_string_rocm_var_counts_as_missing(self):
        report = check_env(
            env={
                "ORT_DYLIB_PATH": "/ok.so",
                "HSA_OVERRIDE_GFX_VERSION": "",
                "PYTORCH_ROCM_ARCH": "gfx1100",
            },
            path_exists=_fake_exists({"/ok.so"}),
        )
        assert report.soft_missing == ["HSA_OVERRIDE_GFX_VERSION"]

    def test_defaults_to_os_environ_when_no_arg(self, monkeypatch, tmp_path):
        """Ensure the default path through ``check_env()`` reads from os.environ."""
        monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
        report = check_env(path_exists=_fake_exists(set()))
        assert any("unset" in s for s in report.hard_issues)
        assert "HSA_OVERRIDE_GFX_VERSION" in report.soft_missing

    def test_defaults_to_real_path_exists_when_no_arg(self, tmp_path):
        """The default path_exists uses the real filesystem."""
        existing = tmp_path / "libonnxruntime.so"
        existing.write_text("stub")
        report = check_env(env={"ORT_DYLIB_PATH": str(existing)})
        assert not any("does not exist" in s for s in report.hard_issues)

    def test_hard_and_soft_can_coexist(self):
        report = check_env(
            env={"ORT_DYLIB_PATH": "/bogus.so"},
            path_exists=_fake_exists(set()),
        )
        assert report.hard_issues  # ORT path bogus
        assert "HSA_OVERRIDE_GFX_VERSION" in report.soft_missing
