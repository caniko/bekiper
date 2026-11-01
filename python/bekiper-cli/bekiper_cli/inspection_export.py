"""One render plan for execution, dry-run accounting and verified receipts."""

from __future__ import annotations

import hashlib
import gc
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from matplotlib.figure import Figure


@dataclass
class RenderJob:
    stem: str
    draw: Callable[[], Figure | None]
    selected: bool
    reason: str | None = None
    features: tuple[str, ...] = ()


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def render_plan(manifest, theme, emit_features, emit_heuristics, no_stories, fmt, only, skip):
    from bekiper_cli.inspect import _feature_stem, _should_emit, _should_emit_story, _story_stem
    from bekiper_inspect.plot.feature import group_into_stories, plot_feature_on_frame, plot_story
    from bekiper_inspect.plot.quiver import plot_quiver, quiver_specs
    from bekiper_inspect.plot.heuristic import (
        plot_heuristic_on_coordinates,
        plot_heuristic_summary,
        plot_heuristic_timeline,
    )

    jobs = []

    def add(stem, draw, selected=None, features=()):
        selected = _should_emit(stem, fmt, only, skip) if selected is None else selected
        jobs.append(RenderJob(stem, draw, selected, None if selected else "output_selector", tuple(features)))

    if emit_heuristics:
        add("summary", partial(plot_heuristic_summary, manifest))
    features = []
    if emit_features:
        for feature in manifest.feature_inspections:
            if theme.should_skip_feature(feature.kind, feature.label):
                jobs.append(
                    RenderJob(
                        _feature_stem(feature),
                        partial(plot_feature_on_frame, manifest, feature),
                        False,
                        "configured_feature_skip",
                        (feature.label,),
                    )
                )
            else:
                features.append(feature)
    if not no_stories and emit_features:
        groups, features = group_into_stories(features, manifest.heuristics, manifest.perimeters)
        for group in groups:
            add(
                _story_stem(group),
                partial(plot_story, manifest, group),
                _should_emit_story(group, fmt, only, skip),
                [f.label for f in group.features],
            )
    elif emit_heuristics:
        for heuristic in manifest.heuristics:
            add(f"timeline_{heuristic.name}", partial(plot_heuristic_timeline, manifest, heuristic.name))
            for label in manifest.labels:
                if label in manifest.coordinate_columns:
                    add(
                        f"coords_{heuristic.name}_{label}",
                        partial(plot_heuristic_on_coordinates, manifest, heuristic.name, label),
                    )
    for feature in features:
        add(_feature_stem(feature), partial(plot_feature_on_frame, manifest, feature), features=(feature.label,))
    if emit_features:
        for spec in quiver_specs(manifest):
            add(spec.stem, partial(plot_quiver, manifest, spec))
    stems = [job.stem for job in jobs if job.selected]
    if len(stems) != len(set(stems)):
        raise ValueError("Inspection plan has duplicate artifact names")
    if only is not None and not stems:
        raise ValueError(f"Requested inspection selector matched no outputs: {only}")
    return jobs


def execute_plan(jobs, theme, inspection_dir: Path, output_dir: Path, selection: dict):
    from matplotlib import pyplot as plt
    from bekiper_inspect.plot.generic import save_figure

    inputs = sorted(inspection_dir.glob("*_inspection.toml")) + sorted(inspection_dir.glob("*_evaluated.parquet"))
    retained = inspection_dir / "inspection_config.toml"
    if retained.exists():
        inputs.append(retained)
    receipt = {
        "schema_version": 1,
        "status": "running",
        "inputs": [fingerprint(p) for p in inputs],
        "effective_theme": asdict(theme),
        "selection": selection,
        "artifacts": [],
        "inventory": [
            {"stem": j.stem, "selected": j.selected, "reason": j.reason, "features": j.features} for j in jobs
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "render_receipt.json"

    def write():
        temporary = receipt_path.with_suffix(".json.pending")
        temporary.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        temporary.replace(receipt_path)

    write()
    try:
        for job in jobs:
            if not job.selected:
                continue
            if Path(job.stem).name != job.stem or job.stem in {".", ".."}:
                raise ValueError(f"Unsafe inspection artifact stem: {job.stem}")
            figure = job.draw()
            if figure is None:
                raise RuntimeError(f"Declared inspection renderer produced no figure: {job.stem}")
            effective = getattr(figure, "_bekiper_plot_theme", theme)
            try:
                path = save_figure(figure, output_dir / f"{job.stem}{theme.fig_format}", fmt=theme.fig_format)
            finally:
                # Matplotlib figures contain artist/canvas cycles. Release the
                # completed figure before the next multi-panel raster is built.
                plt.close(figure)
                del figure
                gc.collect()
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Inspection artifact missing or empty: {path}")
            artifact = fingerprint(path)
            artifact["effective_theme"] = asdict(effective)
            artifact["format"] = path.suffix
            receipt["artifacts"].append(artifact)
            yield path
        if receipt["inputs"] != [fingerprint(p) for p in inputs]:
            raise RuntimeError("Inspection inputs changed while rendering")
        receipt["status"] = "complete"
        write()
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        write()
        raise
