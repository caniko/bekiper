"""bekiper inspect — visualization and inspection commands."""

from __future__ import annotations

import os
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import typer

app = typer.Typer(help="Visualization tools for bekiper analysis output.")

FigFormat = Literal[".svgz", ".jpg", ".png", ".pdf", ".webp"]
RenderMode = Literal["line", "density", "both"]
DensityScale = Literal["linear", "log"]
DEFAULT_CONFIG_FILENAME = "bekiper_config.toml"
ENV_PREFIX = "BEKIPER"
_VIDEO_SHIM_ENV = f"{ENV_PREFIX}_PYTHON_VIDEO_SHIM"


def _should_emit(stem: str, fmt: str, only: str | None, skip: str | None) -> bool:
    """Return True if a figure with *stem* passes the --only/--skip filters.

    The glob patterns are matched against the full filename including
    extension (e.g. ``scale_derivation.png``).
    """
    filename = f"{stem}{fmt}"
    if only is not None and not fnmatch(filename, only):
        return False
    if skip is not None and fnmatch(filename, skip):
        return False
    return True


def _feature_stem(feature: object) -> str:
    return f"feature_{feature.kind}_{feature.label}"  # type: ignore[attr-defined]


def _story_stem(group: object) -> str:
    return f"story_{group.key}"  # type: ignore[attr-defined]


def _should_emit_story(group: object, fmt: str, only: str | None, skip: str | None) -> bool:
    """Apply --only/--skip to a story or any of its member feature stems."""
    stems = [_story_stem(group)]
    stems.extend(_feature_stem(feature) for feature in group.features)  # type: ignore[attr-defined]
    filenames = [f"{stem}{fmt}" for stem in stems]

    if only is not None and not any(fnmatch(filename, only) for filename in filenames):
        return False
    if skip is not None and any(fnmatch(filename, skip) for filename in filenames):
        return False
    return True


@app.command()
def plot(
    inspection_dir: Path = typer.Argument(..., help="Path to inspection output directory."),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output directory for figures."),
    fmt: str | None = typer.Option(None, "-f", "--format", help="Figure format (.webp, .jpg, .png, .svgz, .pdf)."),
    config: Path | None = typer.Option(
        None, "--config", help="Path to bekiper_config.toml for the [inspect.plot] theme."
    ),
    dpi: int | None = typer.Option(None, "--dpi", help="Minimum DPI for saved figures."),
    only: str | None = typer.Option(None, "--only", help="Only emit figures matching this glob pattern."),
    skip: str | None = typer.Option(None, "--skip", help="Skip figures matching this glob pattern."),
    features_only: bool = typer.Option(False, "--features-only", help="Only emit feature inspection figures."),
    no_features: bool = typer.Option(False, "--no-features", help="Skip feature inspection figures."),
    no_stories: bool = typer.Option(
        False,
        "--no-stories",
        help="Emit individual figures instead of consolidated story panels.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help=(
            "Apply a plot profile globally (overlays its overrides on the base theme). "
            "Built-ins: line, density, dense_trajectory, temporal, print_friendly, "
            "active_only, density_log, velocity_field, state_colored, "
            "small_multiples_3, small_multiples_4. Custom profiles defined under "
            "[inspect.plot.profiles] in --config are also accepted."
        ),
    ),
    render_mode: RenderMode | None = typer.Option(
        None,
        "--render-mode",
        help="Trajectory render mode for panels that support it.",
        case_sensitive=False,
    ),
    density_scale: DensityScale | None = typer.Option(
        None,
        "--density-scale",
        help="Linear or log occupancy histogram (log compresses dynamic range).",
        case_sensitive=False,
    ),
    min_speed: float | None = typer.Option(
        None,
        "--min-speed",
        help="Drop frames slower than this many m/s before rendering trajectories.",
    ),
    velocity_field: bool | None = typer.Option(
        None,
        "--velocity-field/--no-velocity-field",
        help="Overlay a quiver of mean velocity per spatial bin.",
    ),
    small_multiples: int | None = typer.Option(
        None,
        "--small-multiples",
        "-N",
        help=(
            "Split long sessions into N side-by-side temporal panels. Currently "
            "honoured by Y-maze arm-sections panels. Pass 1 to disable."
        ),
        min=1,
    ),
    quiver_stride: int | None = typer.Option(
        None, "--quiver-stride", min=1, help="One heading-quiver arrow per N selected frames (default 15)."
    ),
    quiver_single_cmap: str | None = typer.Option(
        None, "--quiver-single-cmap", help="Time colormap of single-case quiver figures (default RdPu)."
    ),
    quiver_binary_cmaps: str | None = typer.Option(
        None, "--quiver-binary-cmaps", help="Comma-separated time colormaps of the two classes (default Reds,Blues)."
    ),
    theme_mode: str | None = typer.Option(
        None, "--theme", help="Colour mode: dark (night mode, the default) or light."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show which figures would be generated without rendering.",
    ),
) -> None:
    """Generate all inspection plots from an analysis output directory.

    The active visual theme is built from (highest priority first):
    CLI flag overrides, then --profile overlay, then the inspect.plot
    table in --config, then built-in defaults. Per-figure routing rules
    in inspect.plot.figure_profiles still apply on top of that baseline
    so a project can mix global flags with finer-grained per-feature
    profiles.
    """
    if features_only and no_features:
        raise typer.BadParameter("--features-only and --no-features are mutually exclusive.")
    from bekiper_inspect.manifest import InspectionManifest
    from bekiper_cli.inspection_export import execute_plan, render_plan
    from bekiper_inspect.plot.theme import PlotTheme, set_active_theme

    overrides: dict[str, object] = {}
    if fmt is not None:
        overrides["fig_format"] = fmt
    if dpi is not None:
        overrides["min_dpi"] = dpi
    if render_mode is not None:
        overrides["trajectory_render_mode"] = render_mode
    if density_scale is not None:
        overrides["density_scale"] = density_scale
    if min_speed is not None:
        overrides["min_speed_filter"] = min_speed
    if velocity_field is not None:
        overrides["show_velocity_field"] = velocity_field
    if small_multiples is not None:
        overrides["small_multiples_n"] = small_multiples
    if quiver_stride is not None:
        overrides["quiver_stride"] = quiver_stride
    if quiver_single_cmap is not None:
        overrides["quiver_single_cmap"] = quiver_single_cmap
    if quiver_binary_cmaps is not None:
        overrides["quiver_binary_cmaps"] = tuple(n.strip() for n in quiver_binary_cmaps.split(","))
    if theme_mode is not None:
        overrides["color_mode"] = theme_mode
    retained_config = inspection_dir / "inspection_config.toml"
    if config is None and retained_config.exists():
        config = retained_config
    theme = PlotTheme.load(
        config_path=config,
        env={} if config is not None and config.name == "inspection_config.toml" else None,
        **overrides,
    )
    if profile is not None:
        theme = theme.with_profile(profile)
        # CLI flag overrides should still win over the chosen profile, so
        # re-apply them after the profile overlay.
        if overrides:
            theme = theme.with_overrides(**overrides)
    set_active_theme(theme)

    inspection_dir = inspection_dir.resolve()
    manifest = InspectionManifest.from_directory(inspection_dir)
    output_dir = output.resolve() if output else inspection_dir / "figures"

    fig_fmt: FigFormat = theme.fig_format

    emit_heuristics = not features_only
    emit_features = not no_features

    jobs = render_plan(manifest, theme, emit_features, emit_heuristics, no_stories, fig_fmt, only, skip)
    if dry_run:
        for job in jobs:
            if job.selected:
                typer.echo(f"  {job.stem}")
        return
    selection = {
        "features_only": features_only,
        "no_features": no_features,
        "no_stories": no_stories,
        "only": only,
        "skip": skip,
    }
    for path in execute_plan(jobs, theme, inspection_dir, output_dir, selection):
        typer.echo(f"Saved: {path}")

    typer.echo(f"All figures written to {output_dir}")


@app.command()
def video(
    inspection_dir: Path = typer.Argument(..., help="Path to inspection output directory."),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output video path."),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Batch-mode output directory."),
    codec: str = typer.Option("h264", help="Video codec (h264, mpeg4, hevc_nvenc, av1_qsv)."),
    heuristic: str | None = typer.Option(None, help="Heuristic selector for compatibility mode."),
    feature: str | None = typer.Option(None, help="Render the feature with this exact manifest label."),
    all_features: bool = typer.Option(False, "--all-features", help="Render overview plus every renderable feature."),
    config: Path | None = typer.Option(
        None, "--config", help="Path to bekiper_config.toml for the [inspect.plot] theme."
    ),
    ffmpeg_preset: str | None = typer.Option(
        None,
        "--ffmpeg-preset",
        help="ffmpeg encoder preset (e.g. slower, medium, veryfast).",
    ),
    source_video: Path | None = typer.Option(
        None, "--source-video", help="Decode this video instead of the manifest's recorded video_path."
    ),
    crf: int | None = typer.Option(None, "--crf", help="Constant rate factor for CRF encoders (default 23)."),
    scored: bool = typer.Option(False, "--scored", help="Overview with scored-heuristic colours and a mask timeline."),
) -> None:
    """Compatibility wrapper around the Rust-native `bekiper inspect video` command."""
    if os.environ.get(_VIDEO_SHIM_ENV) == "1":
        raise RuntimeError(
            "Python inspect-video wrapper re-entered itself. Ensure the Rust `bekiper` binary is ahead of the Python wrapper on PATH."
        )

    args = ["bekiper", "inspect", "video", str(inspection_dir.resolve()), "--codec", codec]
    if output is not None:
        args.extend(["-o", str(output)])
    if output_dir is not None:
        args.extend(["--output-dir", str(output_dir)])
    if heuristic is not None:
        args.extend(["--heuristic", heuristic])
    if feature is not None:
        args.extend(["--feature", feature])
    if all_features:
        args.append("--all-features")
    if config is not None:
        args.extend(["--config", str(config)])
    if ffmpeg_preset is not None:
        args.extend(["--ffmpeg-preset", ffmpeg_preset])
    if source_video is not None:
        args.extend(["--source-video", str(source_video)])
    if crf is not None:
        args.extend(["--crf", str(crf)])
    if scored:
        args.append("--scored")

    env = os.environ.copy()
    env[_VIDEO_SHIM_ENV] = "1"

    try:
        result = subprocess.run(args, env=env, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("Rust `bekiper` binary not found on PATH.") from exc

    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.command()
def info(
    inspection_dir: Path = typer.Argument(..., help="Path to inspection output directory."),
) -> None:
    """Display summary information about an inspection output."""
    from bekiper_inspect.manifest import InspectionManifest

    inspection_dir = inspection_dir.resolve()
    manifest = InspectionManifest.from_directory(inspection_dir)

    typer.echo(f"Video: {manifest.video.path or 'N/A'}")
    typer.echo(f"  FPS: {manifest.video.fps}, Frames: {manifest.video.total_frames}")
    typer.echo(f"  Resolution: {manifest.video.resolution}")
    typer.echo(f"  Duration: {manifest.video.duration_seconds:.1f}s")
    typer.echo(f"  Meters/pixel: {manifest.settings.meters_per_pixel}")
    typer.echo(f"Labels ({len(manifest.labels)}): {', '.join(manifest.labels) or 'N/A'}")
    typer.echo(f"Perimeters ({len(manifest.perimeters)}):")
    for p in manifest.perimeters:
        typer.echo(f"  - {p.label} ({p.shape})")
    typer.echo(f"Heuristics ({len(manifest.heuristics)}):")
    for h in manifest.heuristics:
        typer.echo(f"  - {h.name}: {h.seconds:.1f}s ({h.percentage:.1f}%)")
    typer.echo(f"DataFrame: {manifest.df.shape[0]} rows x {manifest.df.shape[1]} columns")


@app.command()
def composite(
    jobs: Path = typer.Option(..., "--jobs", help="JSON list of composite jobs (one per recording)."),
    theme_mode: str | None = typer.Option(
        None, "--theme", help="Colour mode (dark or light) for jobs that do not carry one."
    ),
) -> None:
    """Render one mega-composite WebP per recording with a coverage-receipt sidecar.

    Each job is ``{"key", "out", "header", "variants": [{"name", "dir"}], "theme"?}``. A job
    writes ``<out>.webp`` atomically, then ``<out>.json``; a failure writes
    ``<out>.failed.json`` and the next job continues.
    """
    import hashlib
    import json
    import time
    import traceback

    from matplotlib import pyplot as plt
    from PIL import Image

    from bekiper_inspect.manifest import InspectionManifest
    from bekiper_inspect.plot.composite import COMPOSITE_VERSION, WEBP_OPTIONS, Variant, render_composite
    from bekiper_inspect.plot.theme import PlotTheme, set_active_theme

    failures = 0
    for job in json.loads(jobs.read_text()):
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            dirs = [Path(v["dir"]) for v in job["variants"]]
            config = dirs[0] / "inspection_config.toml"
            theme = PlotTheme.load(
                config_path=config if config.exists() else None, env={}, color_mode=job.get("theme") or theme_mode
            )
            set_active_theme(theme.with_overrides(min_dpi=theme.composite_dpi, fig_format=".webp"))
            variants = [
                Variant(v["name"], InspectionManifest.from_directory(d))
                for v, d in zip(job["variants"], dirs, strict=True)
            ]
            fig, receipt = render_composite(variants, job.get("header"))
            tmp = out.with_suffix(".tmp.webp")
            try:
                fig.savefig(tmp, format="webp", dpi=theme.composite_dpi, pil_kwargs=dict(WEBP_OPTIONS))
            finally:
                plt.close(fig)
            webp = out.with_suffix(".webp")
            tmp.replace(webp)
            with Image.open(webp) as image:
                width, height = image.size
            sidecar = {
                "key": job["key"],
                "composite_version": COMPOSITE_VERSION,
                "color_mode": theme.color_mode,
                "width": width,
                "height": height,
                "webp_sha256": hashlib.sha256(webp.read_bytes()).hexdigest(),
                "render_seconds": round(time.monotonic() - started, 2),
                "coverage": receipt.as_dict(),
            }
            out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1))
            out.with_suffix(".failed.json").unlink(missing_ok=True)
            typer.echo(f"Saved: {webp} ({sidecar['render_seconds']} s)")
        except Exception as exc:  # one failed recording must not stop the shard
            failures += 1
            out.with_suffix(".failed.json").write_text(
                json.dumps({"key": job["key"], "error": repr(exc), "traceback": traceback.format_exc()}, indent=1)
            )
            typer.echo(f"FAILED: {out}: {exc!r}", err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command("animal-summary")
def animal_summary(
    jobs: Path = typer.Option(..., "--jobs", help="JSON list of per-animal summary jobs."),
) -> None:
    """Render one summary WebP per animal (recorded endpoints vs its group) with a sidecar.

    Each job is ``{"key", "out", "title", "subtitle", "panels", "theme"}``; a job writes
    ``<out>.webp`` atomically, then ``<out>.json``; a failure writes ``<out>.failed.json``.
    """
    import hashlib
    import json
    import traceback

    from matplotlib import pyplot as plt
    from PIL import Image

    from bekiper_inspect.plot.animal import ANIMAL_SUMMARY_VERSION, render_animal_summary
    from bekiper_inspect.plot.composite import WEBP_OPTIONS
    from bekiper_inspect.plot.theme import PlotTheme, set_active_theme

    failures = 0
    for job in json.loads(jobs.read_text()):
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            theme = PlotTheme.load(env={}, color_mode=job.get("theme"))
            set_active_theme(theme)
            fig = render_animal_summary(job)
            tmp = out.with_suffix(".tmp.webp")
            try:
                fig.savefig(tmp, format="webp", dpi=theme.composite_dpi, pil_kwargs=dict(WEBP_OPTIONS))
            finally:
                plt.close(fig)
            webp = out.with_suffix(".webp")
            tmp.replace(webp)
            with Image.open(webp) as image:
                width, height = image.size
            sidecar = {
                "key": job["key"],
                "animal_summary_version": ANIMAL_SUMMARY_VERSION,
                "color_mode": theme.color_mode,
                "width": width,
                "height": height,
                "panels": len(job.get("panels", [])),
                "webp_sha256": hashlib.sha256(webp.read_bytes()).hexdigest(),
            }
            out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1))
            out.with_suffix(".failed.json").unlink(missing_ok=True)
            typer.echo(f"Saved: {webp}")
        except Exception as exc:  # one failed animal must not stop the shard
            failures += 1
            out.with_suffix(".failed.json").write_text(
                json.dumps({"key": job["key"], "error": repr(exc), "traceback": traceback.format_exc()}, indent=1)
            )
            typer.echo(f"FAILED: {out}: {exc!r}", err=True)
    if failures:
        raise typer.Exit(code=1)


@app.command("review-figures")
def review_figures(
    context: Path = typer.Option(..., "--context", help="Post-analysis context directory (e.g. primary45)."),
    output: Path = typer.Option(..., "--output", help="Output directory for reviewer cohort figures."),
    groups: Path | None = typer.Option(None, "--groups", help="JSON {group code: display name}."),
    primary: Path | None = typer.Option(None, "--primary", help="JSON {assay: [criteria endpoints]}."),
    assay: list[str] | None = typer.Option(
        None, "--assay", help="Only these assays (repeatable); default: the context's assays."
    ),
    theme_mode: str | None = typer.Option(None, "--theme", help="Colour mode: dark (default) or light."),
) -> None:
    """Write readable reviewer cohort figures and cohort_index.json for one context."""
    from bekiper_cli.post_analysis.review_figures import main

    args = ["--context", str(context), "--output", str(output)]
    if theme_mode:
        args += ["--theme", theme_mode]
    for name in assay or []:
        args += ["--assay", name]
    for flag, value in (("--groups", groups), ("--primary", primary)):
        if value is not None:
            args += [flag, str(value)]
    raise typer.Exit(code=main(args))


@app.command("extension-figures")
def extension_figures(
    run: Path = typer.Option(..., "--run", help="Run output directory (contains data/<trial>/)."),
    output: Path = typer.Option(..., "--output", help="Output directory for the exploratory figures and index.json."),
    groups: Path | None = typer.Option(None, "--groups", help="JSON {group code: display name}."),
    theme_mode: str | None = typer.Option(None, "--theme", help="Colour mode: dark (default) or light."),
) -> None:
    """Write the exploratory research-extension figures from existing run outputs (nothing is rescored)."""
    import json

    from bekiper_inspect.plot.extensions import write_extension_figures

    index = write_extension_figures(run, output, json.loads(groups.read_text()) if groups else {}, theme_mode)
    for figure in index["figures"]:
        typer.echo(f"{figure['id']}: {figure['status']}" + (f" ({figure['reason']})" if figure["reason"] else ""))
