"""bekiper — unified CLI for bekiper Python tools."""

from __future__ import annotations

import logging

import typer

from bekiper_cli.env_check import warn_if_env_incomplete
from bekiper_cli.inspect import app as inspect_app
from bekiper_cli.post_analysis_cli import app as post_analysis_app

app = typer.Typer(
    name="bekiper",
    help="bekiper: Kinematic analysis tools for behavioral neuroscience.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Initialize logging and diagnose missing ROCm / ONNX env vars.

    Runs before any subcommand so the warning is emitted exactly once
    per invocation, even for lazily-imported subcommands like `detect`.
    """
    # Only configure logging if the user hasn't already (e.g. via a
    # subcommand's own `logging.basicConfig` call below).
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warn_if_env_incomplete()


# Detect subcommand stays import-light by deferring auto-detect imports
# inside its command handlers, so the command remains available even in
# environments that intentionally omit the detector stack.
from bekiper_cli.detect import app as detect_app

app.add_typer(detect_app, name="detect")

app.add_typer(inspect_app, name="inspect")
app.add_typer(post_analysis_app, name="post-analysis")

if __name__ == "__main__":
    app()
