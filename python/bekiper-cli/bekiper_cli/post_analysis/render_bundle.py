"""Draw a native plotting bundle. Eligibility and statistics belong to Rust.

Colours follow the shared themekit palette (``bekiper_rs.theme_*``): dark by
default, ``--theme light`` or ``BEKIPER_PLOT_COLOR_MODE=light`` for light.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bekiper_rs import theme_palette, theme_rc, theme_resolve

Record = dict[str, Any]
FORMATS = {"svg", "pdf", "png"}
SCHEMA_VERSION = "1.0"

# ponytail: one process-wide palette (a renderer draws one mode at a time);
# thread a palette argument through draw_* if figures must mix modes.
_palette: Record = dict(theme_palette(theme_resolve(None, os.environ.get("BEKIPER_PLOT_COLOR_MODE"))))


def use_color_mode(mode: str | None = None) -> Record:
    """Select the palette for later draws; returns its matplotlib rcParams.

    ``mode`` is the CLI layer; ``BEKIPER_PLOT_COLOR_MODE`` and the dark default follow.
    """
    global _palette
    resolved = theme_resolve(mode, os.environ.get("BEKIPER_PLOT_COLOR_MODE"))
    _palette = dict(theme_palette(resolved))
    return dict(theme_rc(resolved))


def swarm_coordinates(ax: Any, points: list[Record], center: float) -> tuple[list[float], float]:
    """Deterministic display-space packing; measured values stay untouched.

    Reduce marker area only when a tied population cannot fit its category.
    Identity breaks ties, so reordering input records does not change placement.
    """
    if not points:
        return [], 20.0
    xy = ax.transData.transform([[center, p["value"]] for p in points])
    center_px = xy[0, 0]
    half_width = abs(ax.transData.transform([[center + 0.38, points[0]["value"]]])[0, 0] - center_px)
    order = sorted(range(len(points)), key=lambda i: (xy[i, 1], points[i]["animal_id"], points[i]["session_id"]))
    size = 20.0
    while True:
        diameter = math.sqrt(size) * ax.figure.dpi / 72 * 1.15
        placed: list[tuple[float, float]] = []
        offsets = np.zeros(len(points))
        for i in order:
            y = xy[i, 1]
            neighbors = [(x, py) for x, py in placed if abs(y - py) < diameter]
            candidates = [0.0]
            for x, py in neighbors:
                separation = math.sqrt(max(0, diameter**2 - (y - py) ** 2))
                candidates.extend([x - separation, x + separation])
            choice = next(
                (
                    x
                    for x in sorted(candidates, key=lambda x: (abs(x), x))
                    if abs(x) <= half_width
                    and all(math.hypot(x - px, y - py) >= diameter - 1e-7 for px, py in neighbors)
                ),
                None,
            )
            if choice is None:
                break
            offsets[i] = choice
            placed.append((choice, y))
        else:
            coords = ax.transData.inverted().transform(np.column_stack([center_px + offsets, xy[:, 1]]))
            return coords[:, 0].tolist(), size
        size *= 0.7


def _layout_swarms(ax: Any) -> None:
    locations = {}
    for center, points, artists in getattr(ax, "_bekiper_swarm_groups", []):
        coords, size = swarm_coordinates(ax, points, center)
        for point, coordinate in zip(points, coords, strict=True):
            locations[(point["category"], point["animal_id"])] = coordinate
        for artist, indices in artists:
            artist.set_offsets([[coords[i], points[i]["value"]] for i in indices])
            artist.set_sizes([size])
    for artist, members in getattr(ax, "_bekiper_pair_lines", []):
        artist.set_xdata([locations[(p["category"], p["animal_id"])] for p in members])


def _number(value: Any, field: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field}: requires a finite number")


def _count(value: Any, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field}: requires a nonnegative integer")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: requires a nonempty string")
    return value


def _validate_panel_points(panel: Record) -> tuple[dict[str, set[str]], dict[str, int]]:
    descriptive = panel.get("population_policy") == "recorded_descriptive"
    identities: set[tuple[str, ...]] = set()
    animals: set[str] = set()
    categories: dict[str, set[str]] = {}
    observation_counts: dict[str, int] = {}
    pairs: dict[str, list[Record]] = {}
    for point in panel["points"]:
        if not isinstance(point, dict):
            raise TypeError("Plot point must be an object")
        for field in ["animal_id", "session_id"]:
            _text(point.get(field), field)
        _number(point.get("value"), "point value")
        animals.add(point["animal_id"])
        category = "scatter"
        if panel["kind"] == "categorical":
            category = _text(point.get("category"), "category")
            categories.setdefault(category, set()).add(point["animal_id"])
            observation_counts[category] = observation_counts.get(category, 0) + 1
        else:
            _number(point.get("x"), "point x")
        identity = (category, point["animal_id"])
        if descriptive:
            identity += (point["session_id"], _text(point.get("analysis_id"), "analysis_id"))
        if identity in identities:
            raise ValueError("Plot panel/category contains repeated animal replicates")
        identities.add(identity)
        paired = point.get("pair_series", False)
        if not isinstance(paired, bool):
            raise TypeError("pair_series must be boolean")
        if not isinstance(point.get("censored", False), bool):
            raise TypeError("censored must be boolean")
        if paired:
            _text(point.get("plot_pair_id"), "plot_pair_id")
            _count(point.get("pair_order"), "pair_order")
            pairs.setdefault(point["plot_pair_id"], []).append(point)
    if panel["n_animals"] != len(animals):
        raise ValueError("Panel animal count differs from native points")
    for members in pairs.values():
        if (
            len(members) > 2
            or len({p["animal_id"] for p in members}) != 1
            or len({p["pair_order"] for p in members}) != len(members)
        ):
            raise ValueError("Pairing must identify at most two distinct observations of the same animal")
    return categories, observation_counts


def _validate_panel_summaries(
    panel: Record, categories: dict[str, set[str]], observation_counts: dict[str, int]
) -> None:
    descriptive = panel.get("population_policy") == "recorded_descriptive"
    seen_categories: set[str] = set()
    for summary in panel["summaries"]:
        if not isinstance(summary, dict):
            raise TypeError("Summary must be an object")
        category = _text(summary.get("category"), "summary category")
        if category in seen_categories:
            raise ValueError("Duplicate summary category")
        seen_categories.add(category)
        _count(summary.get("n_animals"), "summary n_animals")
        if summary["n_animals"] != len(categories.get(category, set())):
            raise ValueError("Summary animal count differs from native points")
        population_count = summary["n_animals"]
        if descriptive:
            _count(summary.get("n_observations"), "summary n_observations")
            population_count = summary["n_observations"]
            if population_count != observation_counts.get(category, 0):
                raise ValueError("Summary observation count differs from native points")
        status = _text(summary.get("status"), "summary status")
        _number(
            summary.get("mean"), "summary mean", nullable=summary["n_animals"] == 0 or status.startswith("blocked:")
        )
        _number(summary.get("error_half_width"), "error_half_width", nullable=True)
        if summary.get("error_half_width") is not None and summary["error_half_width"] < 0:
            raise ValueError("Uncertainty half width must be nonnegative")
        if summary.get("uncertainty") not in {None, "SEM", "95% CI"}:
            raise ValueError("Unknown native summary uncertainty")
        if summary.get("uncertainty") is None and summary.get("error_half_width") is not None:
            raise ValueError("Unregistered uncertainty cannot have an error bar")
        if summary["n_animals"] < 2 and summary.get("error_half_width") is not None:
            raise ValueError("Uncertainty requires at least two animals")
        if summary.get("mean") is None and summary.get("error_half_width") is not None:
            raise ValueError("A withheld mean cannot have an error bar")
        if panel.get("style"):
            _validate_distribution(summary.get("distribution", {}), population_count)
    if not set(categories) <= seen_categories:
        raise ValueError("Native category summary is missing")


def _validate_distribution(distribution: Record, population_count: int) -> None:
    if distribution.get("status") not in {"available", "unavailable:censored"}:
        raise ValueError("Distribution panels require native distribution payloads")
    if distribution.get("status") != "available":
        return
    box = distribution["box"]
    values = [box[key] for key in ["whislo", "q1", "med", "q3", "whishi"]]
    for value in values + box["fliers"]:
        _number(value, "box value")
    if values != sorted(values):
        raise ValueError("Native box quantiles/whiskers must be ordered")
    hist = distribution["histogram"]
    for value in hist["edges"]:
        _number(value, "histogram edge")
    for count in hist["counts"]:
        _count(count, "histogram count")
    if (
        len(hist["edges"]) != len(hist["counts"]) + 1
        or any(a >= b for a, b in zip(hist["edges"], hist["edges"][1:]))
        or sum(hist["counts"]) != population_count
    ):
        raise ValueError("Native histogram shape/population mismatch")
    density = distribution["density"]
    if len(density["x"]) != len(density["y"]):
        raise ValueError("Native density array mismatch")
    for value in density["x"] + density["y"]:
        _number(value, "density value")
    if any(v < 0 for v in density["y"]) or any(a >= b for a, b in zip(density["x"], density["x"][1:])):
        raise ValueError("Native density must be nonnegative on an ordered grid")


def _validate_panel(panel: Record) -> None:
    if not isinstance(panel, dict) or panel.get("kind") not in {"categorical", "scatter"}:
        raise ValueError("Panel kind must be categorical or scatter")
    for field in ["endpoint", "label"]:
        _text(panel.get(field), field)
    if panel.get("style") not in {None, "violin", "boxplot", "histogram"}:
        raise ValueError("Unknown native distribution style")
    for field in ["x_label", "y_label"]:
        if field in panel:
            _text(panel[field], field)
    _number(panel.get("reference_line"), "reference_line", nullable=True)
    _count(panel.get("n_animals"), "panel n_animals")
    points, summaries = panel.get("points"), panel.get("summaries")
    if not isinstance(points, list) or not isinstance(summaries, list):
        raise TypeError("Panel requires point and summary arrays")
    categories, observation_counts = _validate_panel_points(panel)
    if panel["kind"] == "scatter":
        if summaries:
            raise ValueError("Scatter panels do not use category summaries")
        return
    _validate_panel_summaries(panel, categories, observation_counts)


def validate_bundle(bundle: Record) -> None:
    """Check the boundary before writing; do not calculate statistical quantities."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Plotting bundle schema_version must be {SCHEMA_VERSION}")
    provenance = bundle.get("scoring_provenance_by_analysis")
    if provenance is not None and (
        not isinstance(provenance, dict)
        or any(
            not isinstance(key, str) or not key or not (value is None or isinstance(value, dict))
            for key, value in provenance.items()
        )
    ):
        raise ValueError("Native scoring provenance requires an analysis-identity table")

    def validate_references(records: list[Record]) -> None:
        for record in records:
            if provenance is None and "scoring_provenance_ref" not in record:
                continue
            reference = record.get("scoring_provenance_ref")
            if (
                not isinstance(reference, str)
                or reference != record.get("analysis_id")
                or provenance is None
                or reference not in provenance
                or "scoring_provenance" in record
            ):
                raise ValueError("Native scoring provenance reference must resolve its exact analysis identity")

    for field in ["plotting_data", "eligibility_ledger"]:
        validate_references(bundle.get(field, []))
    figures = bundle.get("figures")
    if not isinstance(figures, list):
        raise TypeError("Bundle figures must be an array")
    seen: set[str] = set()
    for figure in figures:
        if not isinstance(figure, dict):
            raise TypeError("Figure must be an object")
        name = figure.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) or name in seen:
            raise ValueError("Figure names must be unique portable slugs")
        seen.add(name)
        dpi = figure.get("dpi", 150)
        if type(dpi) is not int or not 0 < dpi <= 2**32 - 1:
            raise ValueError("Figure dpi must be a positive u32")
        formats = figure.get("formats")
        if (
            not isinstance(formats, list)
            or not formats
            or any(not isinstance(f, str) or f not in FORMATS for f in formats)
            or len(formats) != len(set(formats))
        ):
            raise ValueError("Figure formats must be a unique nonempty list of svg/pdf/png")
        panels = figure.get("panels")
        if not isinstance(panels, list) or not panels:
            raise ValueError("Figure requires a nonempty panel list")
        for panel in panels:
            _validate_panel(panel)
            for field in ["points", "eligibility_ledger"]:
                validate_references(panel.get(field, []))
        if "summary_label" in figure:
            _text(figure["summary_label"], "summary_label")


def _draw_histogram_panel(ax: Any, panel: Record) -> None:
    for summary in panel["summaries"]:
        distribution = summary.get("distribution", {})
        if distribution.get("status") == "available":
            hist = distribution["histogram"]
            ax.stairs(hist["counts"], hist["edges"], label=f"{summary['category']} (n={summary['n_animals']})")
    ax.set_xlabel(panel.get("y_label", panel["label"]))
    ax.set_ylabel("Observations" if panel.get("population_policy") == "recorded_descriptive" else "Animals")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    if not panel["points"]:
        ax.text(0.5, 0.5, "Unavailable / blocked", transform=ax.transAxes, ha="center")
    ax.spines[["top", "right"]].set_visible(False)


def _set_categorical_limits(ax: Any, panel: Record) -> None:
    # Freeze common limits before display-space placement. Include the native
    # KDE grid, while labelling its tails as smoothing in the figure caption.
    bounds = [p["value"] for p in panel["points"]]
    for summary in panel["summaries"]:
        distribution = summary.get("distribution", {})
        if panel.get("style") == "violin" and distribution.get("status") == "available":
            bounds.extend(distribution["density"]["x"])
        if summary["mean"] is not None:
            error = summary.get("error_half_width") or 0
            bounds.extend([summary["mean"] - error, summary["mean"] + error])
    reference = panel.get("reference_line")
    if reference is not None:
        bounds.append(reference)
    if bounds:
        low, high = min(bounds), max(bounds)
        pad = 0.08 * (high - low) if high > low else max(abs(low) * 0.08, 0.1)
        ax.set_ylim(low - pad, high + pad)
    ax.set_xlim(-0.6, len(panel["summaries"]) - 0.4)


def _draw_distribution(ax: Any, summary: Record, style: str | None, x: int) -> None:
    distribution = summary.get("distribution", {})
    if distribution.get("status") != "available":
        return
    if style == "boxplot":
        ax.bxp(
            [distribution["box"]],
            positions=[x],
            widths=0.55,
            showfliers=False,
            manage_ticks=False,
            patch_artist=True,
            boxprops={"facecolor": _palette["box_fill"], "alpha": 0.75},
        )
    elif style == "violin":
        density = distribution["density"]
        peak = max(density["y"], default=0)
        if peak > 0:
            widths = [0.38 * value / peak for value in density["y"]]
            ax.fill_betweenx(
                density["x"], [x - w for w in widths], [x + w for w in widths], color=_palette["box_fill"], alpha=0.75
            )


def _draw_category_points(ax: Any, group: list[Record], x: int, locations: dict[tuple[str, str], float]) -> None:
    coordinates, size = swarm_coordinates(ax, group, x)
    artists = []
    for point, coordinate in zip(group, coordinates, strict=True):
        locations[(point["category"], point["animal_id"])] = coordinate
    for censored, marker in [(False, "o"), (True, "^")]:
        displayed = [
            (point, coordinate)
            for point, coordinate in zip(group, coordinates, strict=True)
            if point.get("censored", False) == censored
        ]
        if displayed:
            artist = ax.scatter(
                [coordinate for _, coordinate in displayed],
                [point["value"] for point, _ in displayed],
                s=size,
                linewidths=0,
                marker=marker,
                color=_palette["point"],
                alpha=0.8,
                zorder=3,
            )
            artists.append((artist, [i for i, p in enumerate(group) if p.get("censored", False) == censored]))
    ax._bekiper_swarm_groups.append((x, group, artists))


def _draw_pair_lines(ax: Any, points: list[Record], locations: dict[tuple[str, str], float]) -> None:
    pairs: dict[str, list[Record]] = {}
    for point in points:
        if point.get("pair_series"):
            pairs.setdefault(point["plot_pair_id"], []).append(point)
    for members in pairs.values():
        if len(members) == 2:
            members = sorted(members, key=lambda point: point["pair_order"])
            (artist,) = ax.plot(
                [locations[(point["category"], point["animal_id"])] for point in members],
                [point["value"] for point in members],
                color=_palette["reference"],
                alpha=0.45,
                linewidth=0.7,
                zorder=1,
            )
            ax._bekiper_pair_lines.append((artist, members))


def _draw_categorical_panel(ax: Any, panel: Record) -> None:
    _set_categorical_limits(ax, panel)
    locations: dict[tuple[str, str], float] = {}
    ax._bekiper_swarm_groups = []
    ax._bekiper_pair_lines = []
    for x, summary in enumerate(panel["summaries"]):
        _draw_distribution(ax, summary, panel.get("style"), x)
        group = [point for point in panel["points"] if point["category"] == summary["category"]]
        _draw_category_points(ax, group, x, locations)
        if summary["mean"] is not None:
            error = summary.get("error_half_width")
            if error is None:
                ax.plot(x, summary["mean"], "_", color=_palette["ink"], markersize=9, zorder=4)
            else:
                ax.errorbar(
                    x, summary["mean"], yerr=error, fmt="_", color=_palette["ink"], capsize=3, markersize=9, zorder=4
                )
    _draw_pair_lines(ax, panel["points"], locations)
    # Categories that arrive pre-wrapped (one factor per line) are drawn upright.
    stacked = any("\n" in summary["category"] for summary in panel["summaries"])
    dense = len(panel["summaries"]) > 10 or any(len(summary["category"]) > 6 for summary in panel["summaries"])
    # draw_figure grows the figure to the measured label height, so long
    # names never squash the plot area.
    labels = [
        f"{summary['category']} (n={summary['n_animals']})"
        if dense and not stacked
        else f"{summary['category']}\nn={summary['n_animals']}"
        for summary in panel["summaries"]
    ]
    ax.set_xticks(
        range(len(labels)),
        labels,
        rotation=0 if stacked else 90 if dense else 35,
        ha="center" if stacked or dense else "right",
        fontsize=8 if dense else 9,
    )


def draw_panel(ax: Any, panel: Record) -> None:
    """Map native values to artists; only decorative coordinates are calculated."""
    ax.set_title(f"{textwrap.fill(panel['label'], width=32)}\nn={panel['n_animals']} animals")
    ax.set_ylabel(textwrap.fill(_text(panel.get("y_label", panel["label"]), "y_label"), width=40))
    reference = panel.get("reference_line")
    if reference is not None:
        reference_artist = ax.axvline if panel.get("style") == "histogram" else ax.axhline
        reference_artist(reference, linestyle="--", color=_palette["reference"], linewidth=0.9)
    points = panel["points"]
    if panel["kind"] == "scatter":
        ax.scatter(
            [point["x"] for point in points],
            [point["value"] for point in points],
            s=20,
            color=_palette["point"],
            alpha=0.8,
        )
        ax.set_xlabel(textwrap.fill(panel.get("x_label", ""), width=40))
    elif panel.get("style") == "histogram":
        _draw_histogram_panel(ax, panel)
        return
    else:
        _draw_categorical_panel(ax, panel)
    if not points:
        ax.text(0.5, 0.5, "Unavailable / blocked\nsee inclusion ledger", transform=ax.transAxes, ha="center")
    ax.spines[["top", "right"]].set_visible(False)
    _layout_swarms(ax)


def draw_figure(figure: Record) -> Any:
    """Lay out one figure record's panels with its caption footer."""
    count = len(figure["panels"])
    columns = min(count, figure.get("columns", 3))
    rows = math.ceil(count / columns)
    width = figure.get("panel_width", 4)
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, 3.8 * rows), squeeze=False)
    for ax, panel in zip(axes.flat, figure["panels"], strict=False):
        draw_panel(ax, panel)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    # Grow each row by its tallest tick labels instead of shrinking the axes.
    renderer = fig.canvas.get_renderer()
    label_inches = max(
        (label.get_window_extent(renderer).height / fig.dpi for ax in axes.flat for label in ax.get_xticklabels()),
        default=0.0,
    )
    fig.set_figheight(fig.get_figheight() + rows * max(0.0, label_inches - 0.6))
    caption = figure.get("summary_label", "Individual animals; summaries supplied by the native engine.")
    caption = textwrap.fill(caption, width=max(25, int(fig.get_figwidth() * 12)))
    fig.text(0.5, 0.008, caption, ha="center", va="bottom", fontsize=8)
    footer_height = (0.15 + len(caption.splitlines()) * 10 / 72) / fig.get_figheight()
    fig.tight_layout(rect=(0, footer_height, 1, 1))
    for ax in axes.flat:
        _layout_swarms(ax)
    return fig


def render_bundle(
    bundle: Record, directory: Path, color_mode: str | None = None, workers: int | None = None
) -> list[Path]:
    validate_bundle(bundle)
    paths = [
        directory / f"{figure['name']}.{extension}" for figure in bundle["figures"] for extension in figure["formats"]
    ]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Figure output already exists: {path.name}")
    if directory.is_symlink():
        raise ValueError("Figure output directory cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    rc = {
        **use_color_mode(color_mode),
        "svg.hashsalt": "bekiper-post-analysis-v1",
        "pdf.fonttype": 42,
        "text.usetex": False,
        "text.parse_math": False,
        "font.family": "DejaVu Sans",
        "font.size": 9,
    }
    figures = bundle["figures"]
    workers = min(resolve_workers(workers), len(figures))
    if workers <= 1:
        with plt.rc_context(rc):
            for figure in figures:
                _save_figure(figure, directory)
    else:
        # Figures are independent. Each worker selects the same palette and
        # rcParams, so output bytes equal the serial path. forkserver avoids
        # forking this (threaded) process.
        context = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(
            workers, mp_context=context, initializer=_init_worker, initargs=(color_mode, rc)
        ) as pool:
            for _ in pool.map(_save_figure, figures, [directory] * len(figures), chunksize=4):
                pass
    return paths


def resolve_workers(workers: int | None = None) -> int:
    """Renderer processes: ``--workers``, ``BEKIPER_POST_ANALYSIS_RENDER_WORKERS``, then min(cores, 8)."""
    if workers is None:
        raw = os.environ.get("BEKIPER_POST_ANALYSIS_RENDER_WORKERS")
        workers = int(raw) if raw else min(os.cpu_count() or 1, 8)
    if workers < 1:
        raise ValueError("render workers must be at least 1")
    return workers


def _init_worker(color_mode: str | None, rc: Record) -> None:
    use_color_mode(color_mode)
    plt.rcParams.update(rc)


def _save_figure(figure: Record, directory: Path) -> None:
    fig = draw_figure(figure)
    try:
        for extension in figure["formats"]:
            metadata: Record = {"Creator": "Bekiper native plotting bundle"}
            if extension == "pdf":
                metadata.update(CreationDate=None, ModDate=None)
            elif extension == "svg":
                metadata["Date"] = None
            # Exclusive file creation also refuses symlink targets that
            # appear between preflight and save. Publication is owned by
            # the Rust pipeline's staging-directory transaction.
            with (directory / f"{figure['name']}.{extension}").open("xb") as stream:
                fig.savefig(stream, format=extension, dpi=figure.get("dpi", 150), metadata=metadata)
    finally:
        plt.close(fig)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--theme", choices=["dark", "light"], help="Colour mode (default: env, then dark).")
    parser.add_argument("--workers", type=int, help="Renderer processes (default: env, then min(cores, 8)).")
    args = parser.parse_args(argv)
    try:
        bundle = json.loads(args.bundle.read_text(), parse_constant=_reject_constant)
        paths = render_bundle(bundle, args.output, args.theme, args.workers)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "rendered",
                "files": [path.name for path in paths],
                "schema_version": SCHEMA_VERSION,
                "runtime_versions": {
                    "python": ".".join(map(str, sys.version_info[:3])),
                    "matplotlib": matplotlib.__version__,
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
