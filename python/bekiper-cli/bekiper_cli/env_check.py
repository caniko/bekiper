"""Startup diagnostic for ROCm / PyTorch / ONNX Runtime environment variables.

The pure-logic ``check_env`` and ``EnvCheckReport`` are re-exported from
``bekiper_rs`` (built from `rust/bekiper-inspect-py` against the
single-source-of-truth ``bekiper_core::env_check``). Only the
:func:`warn_if_env_incomplete` glue, which converts a report into Python
``logging`` output, lives in this module.
"""

from __future__ import annotations

import logging

from bekiper_rs import EnvCheckReport, check_env

logger = logging.getLogger(__name__)


def warn_if_env_incomplete() -> None:
    """Check the runtime environment and log a warning if anything
    critical is missing. Call this exactly once at CLI startup.
    """
    report = check_env()

    if report.hard_issues:
        logger.warning(
            "bekiper runtime environment is incomplete: %s. "
            "This usually means you are running outside the nix devshell. "
            "Run `nix develop` in the project directory, then re-invoke `bekiper`.",
            "; ".join(report.hard_issues),
        )

    if report.soft_missing:
        logger.info(
            "ROCm env hint: %s not set. If you plan to run GPU pose inference or detection, source the nix devshell.",
            ", ".join(report.soft_missing),
        )


__all__ = ["EnvCheckReport", "check_env", "warn_if_env_incomplete"]
