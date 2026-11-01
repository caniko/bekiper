"""Unified Bekiper CLI adapter; numerical imports are deferred."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Reproducible behavioural post-analysis from exported measurements.")


@app.command("run")
def run_command(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    request: Annotated[list[str] | None, typer.Option("--request", help="Requested output group; repeatable.")] = None,
    sensitivity_threshold: Annotated[
        list[int] | None,
        typer.Option("--sensitivity-threshold", help="Y-maze sensitivity threshold (25/45/65); repeatable."),
    ] = None,
    no_sensitivity: Annotated[
        bool, typer.Option("--no-sensitivity", help="Disable supplementary sensitivity panels.")
    ] = False,
) -> None:
    from bekiper_cli.post_analysis.native import run

    if no_sensitivity and sensitivity_threshold:
        raise typer.BadParameter("--no-sensitivity conflicts with --sensitivity-threshold")
    try:
        manifest = run(
            config,
            output=output,
            requested_outputs=request,
            sensitivity_thresholds=[] if no_sensitivity else sensitivity_threshold,
        )
    except (ValueError, TypeError, OSError, KeyError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    typer.echo(f"Post-analysis status: {manifest['status']} ({manifest['run_mode']})")
