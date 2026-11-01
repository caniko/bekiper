"""Import-light entry point: python -m bekiper_cli.post_analysis --config ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .native import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Recipe-based OFT/Y-maze/NORT post-analysis")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Fresh output directory (overrides env/config)")
    parser.add_argument("--request", action="append", help="Output group to request (repeatable; overrides env/config)")
    parser.add_argument(
        "--sensitivity-threshold",
        action="append",
        type=int,
        choices=[25, 45, 65],
        help="Y-maze sensitivity threshold (repeatable; overrides env/config)",
    )
    parser.add_argument("--no-sensitivity", action="store_true", help="Disable supplementary sensitivity panels")
    args = parser.parse_args()
    if args.no_sensitivity and args.sensitivity_threshold:
        parser.error("--no-sensitivity conflicts with --sensitivity-threshold")
    try:
        manifest = run(
            args.config,
            output=args.output,
            requested_outputs=args.request,
            sensitivity_thresholds=[] if args.no_sensitivity else args.sensitivity_threshold,
        )
    except (ValueError, TypeError, OSError, KeyError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": manifest["status"], "run_mode": manifest["run_mode"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
