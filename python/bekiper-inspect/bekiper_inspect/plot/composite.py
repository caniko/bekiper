"""One mega-composite inspection figure per trial (reviewer package).

The figure reads top to bottom as *who -> what was measured -> where -> how
each score was decided -> when*:

0. header: identity, criteria endpoint values, QC flags;
1. overview: geometry + full trajectory, the assay result panels, pose quality;
2. decision matrix: every scored heuristic (final mask first, then each
   component), diagnostic channels and other decompositions as final masks;
3. aligned recording-time timelines and key time series;
4. state distributions, legend and parameters.

Only recorded masks and results are drawn; nothing is rescored. Every feature
in the manifest is reported in the coverage receipt as drawn, folded or omitted.
Scoring variants of one recording (threshold contexts) share one figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure, SubFigure

from bekiper_inspect.experiments import all_features, descriptors, is_object, section
from bekiper_inspect.manifest import (
    FeatureCenterPeriphery,
    FeatureHeuristicDecomposition,
    FeatureInspection,
    FeaturePerimeter,
    FeatureStateDistribution,
    FeatureTimeSeries,
    InspectionManifest,
)
from bekiper_inspect.plot.feature._dispatch import draw_feature_on_ax
from bekiper_inspect.plot.feature._helpers import _calibration_resolution, _metric_to_frame_pixels
from bekiper_inspect.plot.feature._renderers import _draw_perimeter_overlay
from bekiper_inspect.plot.frame import read_frame_at
from bekiper_inspect.plot.theme import get_active_theme

# ponytail: bump when the layout changes so cached composites are re-rendered.
COMPOSITE_VERSION = 3
#: WebP encoder settings for every reviewer figure (lossy, visually lossless at 100 %).
#: Measured 2026-10-24 on NORT/OFT/Y-maze composites: method 6 saves 2-3 % for up to
#: 2.6x the encode time, q80 saves ~15 % at -0.2 dB, lossless is 2-4x larger.
WEBP_OPTIONS = {"quality": 85, "method": 4}

# Frame-relative (drawn on video), so the same in both colour modes.
PERIMETER_COLOR = "#00a676"
WIDTH_IN = 22.0
CELL_IN = 2.6


@dataclass
class Receipt:
    """Where every manifest feature ended up."""

    drawn: dict[str, str] = field(default_factory=dict)
    folded: dict[str, str] = field(default_factory=dict)
    omitted: dict[str, str] = field(default_factory=dict)

    def mark(self, kind: str, feature: FeatureInspection, where: str) -> None:
        getattr(self, kind)[feature.label] = where

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {"drawn": self.drawn, "folded": self.folded, "omitted": self.omitted}


@dataclass
class Variant:
    """One scoring of the recording (a single trial for OFT/NORT)."""

    name: str
    manifest: InspectionManifest


def _pal() -> dict[str, Any]:
    """Page-relative colours of the active theme's mode (themekit tokens)."""
    return get_active_theme().colors()


def _time_cmap() -> ListedColormap:
    pal = _pal()
    start, end = pal["time_range"]
    return ListedColormap(colormaps[pal["time_cmap"]](np.linspace(start, end, 256)))


def _times(manifest: InspectionManifest) -> np.ndarray:
    for column in ("elapsed_s", "elapsed"):
        if column in manifest.df.columns:
            return manifest.df[column].to_numpy().astype(float)
    fps = manifest.video.fps or 1.0
    return np.arange(manifest.df.height, dtype=float) / fps


def _frame(manifest: InspectionManifest) -> np.ndarray:
    """A cached frame (never a video decode when frames were extracted)."""
    frames = sorted(Path(manifest.frames_dir).glob("frame_*.png")) if manifest.frames_dir else []
    if frames:
        index = int(frames[0].stem.split("_")[-1])
        frame = read_frame_at(None, index, frames_dir=manifest.frames_dir)
        if frame is not None:
            return frame
    w, h = manifest.video.resolution
    return np.full((max(h, 1), max(w, 1), 3), 40, dtype=np.uint8)


def _mask(manifest: InspectionManifest, column: str) -> np.ndarray | None:
    """Boolean mask as float (1 true, 0 false, nan unavailable)."""
    if column not in manifest.df.columns:
        return None
    values = manifest.df[column]
    out = values.cast(float).to_numpy().astype(float)
    return out


def _runs(mask: np.ndarray, value: float) -> list[tuple[int, int]]:
    hit = mask == value if not math.isnan(value) else np.isnan(mask)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], hit.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2] - edges[::2], strict=True))


def _seconds_true(manifest: InspectionManifest, mask: np.ndarray) -> float:
    times = _times(manifest)
    if len(times) < 2:
        return 0.0
    dt = np.diff(times, append=times[-1] + np.median(np.diff(times)))
    return float(np.nansum(np.where(mask == 1.0, dt, 0.0)))


def _trajectory_label(manifest: InspectionManifest) -> str:
    for feature in manifest.feature_inspections:
        if isinstance(feature, FeatureCenterPeriphery) and feature.coordinate_labels:
            return feature.coordinate_labels[0]
    for name in ("body_center", "mouse_center", "center_ear"):
        if name in manifest.coordinate_columns:
            return name
    return next(iter(manifest.coordinate_columns))


def _draw_trajectory(ax: Any, manifest: InspectionManifest, frame_shape: tuple[int, ...]) -> LineCollection:
    """Full trajectory coloured by recording time; gaps are not joined."""
    coords = manifest.get_coordinates(_trajectory_label(manifest))
    px = _metric_to_frame_pixels(
        coords, manifest.settings.meters_per_pixel, _calibration_resolution(manifest), frame_shape
    )
    segments = np.stack([px[:-1], px[1:]], axis=1)
    keep = np.isfinite(segments).all(axis=(1, 2))
    times = _times(manifest)[1:]
    collection = LineCollection(segments[keep], cmap=_time_cmap(), linewidths=0.6, alpha=0.85)
    collection.set_array(times[keep])
    collection.set_rasterized(True)
    ax.add_collection(collection)
    return collection


def _decomposition_by_heuristic(manifest: InspectionManifest) -> dict[str, FeatureHeuristicDecomposition]:
    primary: dict[str, FeatureHeuristicDecomposition] = {}
    for feature in manifest.feature_inspections:
        if (
            isinstance(feature, FeatureHeuristicDecomposition)
            and feature.label == f"heuristic_decomposition_{feature.heuristic_name}"
        ):
            primary[feature.heuristic_name] = feature
    return primary


def _scored(manifest: InspectionManifest) -> tuple[list[str], list[str]]:
    """(scored heuristics, diagnostic channels). A primary decomposition without a
    heuristic row (posture proxies: they gate NORT contact rescue) counts as scored;
    descriptor ``features.diagnostic_prefixes`` name the channels that never score."""
    names = [h.name for h in manifest.heuristics]
    names += sorted(n for n in _decomposition_by_heuristic(manifest) if n not in names)
    diagnostic = [n for n in names if n.startswith(tuple(all_features("diagnostic_prefixes")))]
    return [n for n in names if n not in diagnostic], diagnostic


def _endpoint_lines(header: dict[str, Any]) -> list[str]:
    """Header endpoint text: one line, or one line per variant for Y-maze tables."""
    endpoints = header.get("endpoints", {})
    if endpoints and all(isinstance(v, dict) for v in endpoints.values()):
        return [
            f"{variant}: " + "   ".join(f"{k} = {_fmt(v)}" for k, v in values.items())
            for variant, values in endpoints.items()
        ]
    return ["   ".join(f"{k} = {_fmt(v)}" for k, v in endpoints.items())] if endpoints else []


def _header_height(header: dict[str, Any]) -> float:
    return 1.2 + 0.24 * max(len(_endpoint_lines(header)), 1)


def _header(fig: SubFigure, header: dict[str, Any], variants: list[Variant]) -> None:
    manifest = variants[0].manifest
    height = _header_height(header)
    y = lambda inches: 1.0 - inches / height  # noqa: E731 - rows are placed in inches from the top
    title = header.get("title") or Path(manifest.video.path or "trial").name
    fig.text(0.005, y(0.1), title, fontsize=17, weight="bold", va="top")
    identity = " · ".join(str(v) for v in header.get("identity", []) if v not in (None, ""))
    fps = manifest.video.fps
    times = _times(manifest)
    span = f"{times[-1] - times[0]:.1f} s analysed at {fps:.2f} fps" if len(times) else ""
    method = getattr(manifest.scale_derivation, "method", "")
    fig.text(
        0.005,
        y(0.42),
        " · ".join(x for x in (identity, span, f"scale {method}" if method else "") if x),
        fontsize=11,
        va="top",
    )
    lines = _endpoint_lines(header)
    for n, line in enumerate(lines):
        fig.text(0.005, y(0.7 + 0.24 * n), line, fontsize=11, family="monospace", va="top")
    x = 0.005
    pal = _pal()
    flags_y = y(0.78 + 0.24 * max(len(lines), 1))
    for name, good in header.get("flags", {}).items():
        fig.text(
            x,
            flags_y,
            f" {name} ",
            fontsize=10,
            color=pal["on_status"],
            va="top",
            bbox={"facecolor": pal["positive"] if good else pal["negative"], "edgecolor": "none", "pad": 2},
        )
        x += 0.012 + 0.0062 * len(name)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        # `+ 0.0` turns a recorded -0.0 into 0.
        return "NA" if math.isnan(value) else f"{value + 0.0:.3g}"
    return "NA" if value is None else str(value)


def _overview(fig: SubFigure, variants: list[Variant], frame: np.ndarray, receipt: Receipt) -> None:
    manifest = variants[0].manifest
    kinds, excluded = _panel_kinds()
    panels: list[tuple[Variant, FeatureInspection]] = []
    for variant in variants:
        for feature in variant.manifest.feature_inspections:
            if feature.label in excluded:
                receipt.mark("folded", feature, f"1 same data as the {excluded[feature.label]} panel")
            elif feature.kind in kinds:
                panels.append((variant, feature))
    per_row = 4 if len(panels) > 3 else max(len(panels), 1)
    n_rows = max(1, math.ceil(len(panels) / per_row))
    # Each panel row is followed by a caption row for the renderers' summary boxes.
    image_in = WIDTH_IN / (per_row + 1.6 * min(n_rows, 2) + 0.5) * frame.shape[0] / max(frame.shape[1], 1)
    grid = fig.add_gridspec(
        2 * n_rows,
        per_row + 2,
        width_ratios=[1.6 * min(n_rows, 2)] + [1.0] * per_row + [0.5],
        height_ratios=[image_in, CAPTION_IN] * n_rows,
        left=0.01,
        right=0.99,
        top=0.93,
        bottom=0.01,
        wspace=0.12,
        hspace=0.08,
    )
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(frame, aspect="equal")
    for feature in manifest.feature_inspections:
        if isinstance(feature, FeaturePerimeter):
            draw_feature_on_ax(ax, manifest, feature, frame.shape, 0)
            receipt.mark("drawn", feature, "1 overview")
    trajectory = _draw_trajectory(ax, manifest, frame.shape)
    ax.set_title("Geometry + full trajectory (colour = recording time)", fontsize=10)
    ax.set_axis_off()
    bar = fig.colorbar(trajectory, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01, aspect=40)
    bar.set_label("recording time (s)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    for i, (variant, feature) in enumerate(panels):
        ax = fig.add_subplot(grid[2 * (i // per_row), 1 + i % per_row])
        ax.imshow(frame, aspect="equal")
        draw_feature_on_ax(ax, variant.manifest, feature, frame.shape, 0)
        # Heatmaps autoscale to the trajectory; keep the whole frame in view.
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)
        _caption_boxes(ax)
        suffix = f" · {variant.name}" if len(variants) > 1 else ""
        ax.set_title(f"{(feature.title_override or feature.label).replace('_', ' ')}{suffix}", fontsize=9)
        ax.set_axis_off()
        receipt.mark("drawn", feature, "1 overview")
    _pose_quality(fig.add_subplot(grid[:, -1]), manifest)


def _caption_boxes(ax: Any) -> None:
    """Move axes-anchored summary boxes below the panel so they never overlap."""
    boxed = [t for t in ax.texts if t.get_transform() == ax.transAxes]
    if not boxed:
        return
    text = "\n".join(t.get_text() for t in sorted(boxed, key=lambda t: t.get_position()[0]))
    for t in boxed:
        t.remove()
    ax.text(
        0.0,
        -0.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        family="monospace",
        color=_pal()["ink"],
        clip_on=False,
    )


def _panel_kinds() -> tuple[set[str], dict[str, str]]:
    """Overview panel kinds (descriptor ``features.panels``) and {excluded label: panel showing its data}."""
    kinds: set[str] = set()
    excluded: dict[str, str] = {}
    for d in descriptors():
        features = section(d, "features")
        panels = features.get("panels") or []
        kinds.update(panels)
        for label in features.get("panel_exclude_labels") or []:
            excluded[label] = panels[0].replace("_", " ") if panels else "overview"
    return kinds, excluded


def _overview_panel_count(variants: list[Variant]) -> int:
    kinds, excluded = _panel_kinds()
    return sum(
        1 for v in variants for f in v.manifest.feature_inspections if f.kind in kinds and f.label not in excluded
    )


def _overview_height(variants: list[Variant], aspect: float) -> float:
    panels = _overview_panel_count(variants)
    per_row = 4 if panels > 3 else max(panels, 1)
    rows = max(1, math.ceil(panels / per_row))
    return (WIDTH_IN / (per_row + 1.6 * min(rows, 2) + 0.5) * aspect + CAPTION_IN) * rows + 0.6


def _pose_quality(ax: Any, manifest: InspectionManifest) -> None:
    labels = [label for label in manifest.labels if label in manifest.coordinate_columns]
    tracked = []
    for label in labels:
        coords = manifest.get_coordinates(label)
        tracked.append(100.0 * np.isfinite(coords).all(axis=1).mean() if len(coords) else 0.0)
    pal = _pal()
    ax.barh(range(len(labels)), tracked, color=[pal["true_mask"] if t >= 95 else pal["false_mask"] for t in tracked])
    ax.set_xlim(0, 100)
    ax.set_title("Pose: tracked frames (%)", fontsize=10)
    ax.tick_params(labelsize=8)
    # Names inside the bars: the narrow column has no room for tick labels.
    ax.set_yticks([])
    for i, (label, t) in enumerate(zip(labels, tracked, strict=True)):
        ax.text(
            2, i, f"{label}  {t:.0f}%", va="center", ha="left", fontsize=7.5, color=pal["on_status"], fontweight="bold"
        )


Cell = tuple[Variant, str, str, str | None, str | None]


def _matrix_groups(variants: list[Variant], receipt: Receipt) -> list[tuple[str, bool, list[Cell]]]:
    """(title, scored, cells) per heuristic across all variants.

    Each group starts with the final mask of every variant; a component cell is
    repeated for a variant only where its mask differs from the one already shown.
    """
    multi = len(variants) > 1
    groups: list[tuple[str, bool, list[Cell]]] = []
    first = variants[0].manifest
    scored, diagnostic = _scored(first)
    for name in scored:
        finals: list[Cell] = []
        components: list[Cell] = []
        shown: dict[str, np.ndarray | None] = {}
        for variant in variants:
            feature = _decomposition_by_heuristic(variant.manifest).get(name)
            if feature is None:
                continue
            receipt.mark("drawn", feature, "2 decision matrix")
            tag = f" · {variant.name}" if multi else ""
            finals.append(
                (
                    variant,
                    f"final{tag}",
                    feature.result_column,
                    _first(feature.coordinate_labels),
                    feature.perimeter_label,
                )
            )
            for component in feature.components:
                mask = _mask(variant.manifest, component.result_column)
                previous = shown.get(component.label)
                if component.label in shown and (
                    mask is not None and previous is not None and np.array_equal(mask, previous, equal_nan=True)
                ):
                    continue
                shown[component.label] = mask
                components.append(
                    (
                        variant,
                        f"{component.label}{tag}",
                        component.result_column,
                        _first(component.coordinate_labels or feature.coordinate_labels),
                        component.perimeter_label or feature.perimeter_label,
                    )
                )
        if finals:
            if multi:
                # Components identical in every variant are labelled once, without a variant tag.
                counts: dict[str, int] = {}
                for cell in components:
                    base = cell[1].rsplit(" · ", 1)[0]
                    counts[base] = counts.get(base, 0) + 1
                components = [
                    (c[0], c[1].rsplit(" · ", 1)[0] + " (all variants)", *c[2:])
                    if counts[c[1].rsplit(" · ", 1)[0]] == 1
                    else c
                    for c in components
                ]
            groups.append((name, True, finals + components))
    primary = _decomposition_by_heuristic(first)
    diagnostic_cells: list[Cell] = []
    for name in diagnostic:
        feature = primary.get(name)
        if feature is not None:
            receipt.mark("folded", feature, "2 diagnostic channel (final mask)")
            diagnostic_cells.append(
                (variants[0], name, feature.result_column, _first(feature.coordinate_labels), feature.perimeter_label)
            )
    if diagnostic_cells:
        groups.append(("diagnostic channels (not scoring inputs)", False, diagnostic_cells))
    return groups


def _cell(
    ax: Any,
    manifest: InspectionManifest,
    frame: np.ndarray,
    title: str,
    column: str,
    coordinate: str | None,
    perimeter: str | None,
    bold: bool = False,
) -> None:
    """Frame + perimeter + trajectory points coloured by one mask."""
    pal = _pal()
    # A dimmed frame keeps the points readable; dark pages need less dimming.
    ax.imshow(frame, aspect="equal", alpha=0.75 if pal["mode"] == "dark" else 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    mask = _mask(manifest, column)
    label = coordinate if coordinate in manifest.coordinate_columns else _trajectory_label(manifest)
    if perimeter:
        try:
            _draw_perimeter_overlay(
                ax, manifest, perimeter, color=PERIMETER_COLOR, frame_shape=frame.shape, linewidth=1.2
            )
        except StopIteration:
            pass
    if mask is None:
        _cell_label(ax, f"{title}\nunavailable", bold)
        return
    px = _metric_to_frame_pixels(
        manifest.get_coordinates(label),
        manifest.settings.meters_per_pixel,
        _calibration_resolution(manifest),
        frame.shape,
    )
    marks = (
        (0.0, pal["false_mask"], 0.6, 0.12),
        (math.nan, pal["unavailable"], 0.6, 0.3),
        (1.0, pal["true_mask"], 1.6, 0.9),
    )
    for value, color, size, alpha in marks:
        hit = np.isnan(mask) if math.isnan(value) else mask == value
        ax.scatter(px[hit, 0], px[hit, 1], s=size, c=color, alpha=alpha, linewidths=0, rasterized=True)
    x0, x1, y0, y1 = _view(px, frame.shape)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    # Overlay labels from shared renderers may sit outside the zoom; they must
    # neither shrink the grid nor draw outside the cell.
    for artist in ax.get_children():
        if artist is not ax.patch and artist not in ax.spines.values():
            artist.set_in_layout(False)
            artist.set_clip_on(True)
    seconds = _seconds_true(manifest, mask)
    share = 100.0 * np.nanmean(mask == 1.0) if len(mask) else 0.0
    _cell_label(ax, f"{title}\n{seconds:.1f} s · {share:.0f}%", bold)


def _view(px: np.ndarray, shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    """Zoom to where the animal was (1-99th percentile, 12 % margin), as a square-ish window."""
    finite = px[np.isfinite(px).all(axis=1)]
    if len(finite) < 10:
        return 0, shape[1], 0, shape[0]
    (x0, y0), (x1, y1) = np.percentile(finite, [1, 99], axis=0)
    pad = 0.12 * max(x1 - x0, y1 - y0, 1.0)
    return max(0, x0 - pad), min(shape[1], x1 + pad), max(0, y0 - pad), min(shape[0], y1 + pad)


def _cell_label(ax: Any, text: str, bold: bool) -> None:
    """Label drawn inside the cell so long names never shrink the grid."""
    label = ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        weight="bold" if bold else "normal",
        clip_on=True,
        bbox={"facecolor": _pal()["surface"], "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
    )
    label.set_in_layout(False)


MAX_COLS = 9
CAPTION_IN = 1.3


def _pack(groups: list[Any]) -> list[list[tuple[int, Any]]]:
    """Greedy rows of MAX_COLS cells; a group never splits unless it is wider than a row."""
    rows: list[list[tuple[int, Any]]] = [[]]
    for g, (_, _, cells) in enumerate(groups):
        for start in range(0, len(cells), MAX_COLS):
            chunk = cells[start : start + MAX_COLS]
            if len(rows[-1]) + len(chunk) > MAX_COLS:
                rows.append([])
            rows[-1] += [(g, (cell, i == 0 and start == 0)) for i, cell in enumerate(chunk)]
    return [r for r in rows if r]


def _decision_matrix(fig: SubFigure, groups: list[Any], frame: np.ndarray) -> None:
    """Scored heuristics as cell groups (final masks first, outlined); short groups share a row."""
    packed = _pack(groups)
    if not packed:
        return
    grid = fig.add_gridspec(
        len(packed), MAX_COLS, left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04, hspace=0.06
    )
    for r, cells in enumerate(packed):
        for c, (g, (cell, first)) in enumerate(cells):
            variant, name, column, coordinate, perimeter = cell
            title, scored, _ = groups[g]
            ax = fig.add_subplot(grid[r, c])
            label = f"{title} · {name}" if scored and first else name
            final = scored and name.startswith("final")
            _cell(ax, variant.manifest, frame, label.replace("_", " "), column, coordinate, perimeter, bold=final)
            if final:
                for spine in ax.spines.values():
                    spine.set_edgecolor(_pal()["ink"])
                    spine.set_linewidth(2.2)
            if first and not scored:
                ax.text(0, -0.04, title, transform=ax.transAxes, fontsize=8, style="italic", va="top")


def _first(labels: Any) -> str | None:
    labels = list(labels or [])
    return labels[0] if labels else None


def _timelines(fig: SubFigure, variants: list[Variant], series: list[FeatureTimeSeries]) -> None:
    rows: list[tuple[str, InspectionManifest, str, bool]] = []
    for variant in variants:
        manifest = variant.manifest
        primary = _decomposition_by_heuristic(manifest)
        scored, _ = _scored(manifest)
        suffix = f" · {variant.name}" if len(variants) > 1 else ""
        for name in scored:
            feature = primary.get(name)
            meta = next((h for h in manifest.heuristics if h.name == name), None)
            column = meta.result_column if meta is not None else feature.result_column if feature else None
            if column is None:
                continue
            rows.append((f"{name}{suffix}", manifest, column, True))
            if feature is not None and variant is variants[0]:
                rows += [(f"  {c.label}", manifest, c.result_column, False) for c in feature.components]
    height_ratios = [max(len(rows), 1) * 0.22] + [1.4] * len(series)
    axes = np.atleast_1d(
        fig.subplots(
            1 + len(series),
            1,
            sharex=True,
            gridspec_kw={
                "height_ratios": height_ratios,
                "left": 0.11,
                "right": 0.99,
                "top": 0.95,
                "bottom": 0.07,
                "hspace": 0.32,
            },
        )
    )
    ax = axes[0]
    end = 0.0
    pal = _pal()
    for y, (label, manifest, column, bold) in enumerate(reversed(rows)):
        mask = _mask(manifest, column)
        if mask is None:
            continue
        times = _times(manifest)
        end = max(end, float(times[-1]) if len(times) else 0.0)
        dt = np.diff(times, append=times[-1] + (np.median(np.diff(times)) if len(times) > 1 else 0.0))
        for value, color in ((1.0, pal["true_mask"]), (0.0, pal["false_mask"]), (math.nan, pal["unavailable"])):
            spans = [(times[s], float(np.sum(dt[s : s + n]))) for s, n in _runs(mask, value)]
            ax.broken_barh(
                spans, (y + 0.1, 0.8), facecolors=color, alpha=0.95 if bold else (0.25 if value == 0.0 else 0.7)
            )
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=7)
    for tick, (_, _, _, bold) in zip(ax.get_yticklabels(), reversed(rows), strict=True):
        tick.set_fontweight("bold" if bold else "normal")
        tick.set_color(pal["ink"] if bold else pal["muted"])
    ax.set_ylim(0, max(len(rows), 1))
    ax.set_title(
        "When: scored masks (bold) and their components — blue true · red false · gray unavailable", fontsize=10
    )
    for ax, feature in zip(axes[1:], series, strict=True):
        manifest = variants[0].manifest
        values = manifest.df[feature.value_column].to_numpy().astype(float)
        ax.plot(_times(manifest), values, linewidth=0.6, color=pal["ink"])
        ax.set_ylabel(feature.unit, fontsize=8)
        ax.set_title(_series_title(feature, manifest), fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axes[-1].set_xlim(0, end or None)
    axes[-1].set_xlabel("Recording time (s) — common scale for all rows")


def _series_title(feature: FeatureTimeSeries, manifest: InspectionManifest) -> str:
    column = feature.value_column
    for name in (p.label for p in manifest.perimeters if is_object(p.label)):
        if f"exploring_{name}__within_angle_deg" in column:
            target = name.replace("_", " ")
            target = target if "object" in target else f"{target} object"
            return f"Head angle to the {target} (°, nearest perimeter point; bout median ≤ threshold)"
    if "speed" in feature.label:
        return f"{feature.title_override or 'Nose speed'} ({feature.unit}) — descriptive"
    return feature.title_override or feature.description or feature.label


def _series_for(manifest: InspectionManifest, receipt: Receipt) -> list[FeatureTimeSeries]:
    """Head angle to scored objects and speeds; other series are folded into the receipt."""
    scored, _ = _scored(manifest)
    chosen: list[FeatureTimeSeries] = []
    for feature in manifest.feature_inspections:
        if not isinstance(feature, FeatureTimeSeries):
            continue
        scored_angle = any(f"__{name}__" in feature.value_column for name in scored)
        if (
            (scored_angle or "speed" in feature.label)
            and len(chosen) < 3
            and feature.value_column in manifest.df.columns
        ):
            chosen.append(feature)
            receipt.mark("drawn", feature, "3 time series")
        else:
            receipt.mark("folded", feature, "3 diagnostic series (not a scoring input)")
    return chosen


def _distributions(fig: SubFigure, manifest: InspectionManifest, receipt: Receipt, footer: str) -> None:
    """Whole-session state shares: target composition and sensory agreement, one shared legend each."""
    states = [f for f in manifest.feature_inspections if isinstance(f, FeatureStateDistribution)]
    composition = [f for f in states if "Unassigned" in f.category_labels]
    agreement = [f for f in states if f not in composition]
    groups = [
        (g, title)
        for g, title in (
            (composition, "Session composition: target of each channel"),
            (agreement, "Sensory agreement per channel and object"),
        )
        if g
    ]
    pal = _pal()
    fig.text(0.005, 0.01, footer, fontsize=8, color=pal["muted"])
    if not groups:
        return
    axes = np.atleast_1d(
        fig.subplots(
            1, len(groups), gridspec_kw={"left": 0.11, "right": 0.99, "top": 0.86, "bottom": 0.36, "wspace": 0.35}
        )
    )
    for ax, (features, title) in zip(axes, groups, strict=True):
        skip = {"Unassigned", "Missing"} if features is composition else set()
        order = [c for c in dict.fromkeys(c for f in features for c in f.category_labels) if c not in skip]
        cycle = pal["categorical"]
        colors = {c: pal["unavailable"] if c == "Missing" else cycle[k % len(cycle)] for k, c in enumerate(order)}
        used: dict[str, Any] = {}
        names = []
        for row, feature in enumerate(features):
            left = 0.0
            for label, pct in zip(feature.category_labels, feature.percentages, strict=True):
                if label in skip:
                    continue
                if pct > 0:
                    used[label] = ax.barh(
                        row, pct, left=left, color=colors[label], edgecolor=pal["background"], height=0.8
                    )
                left += pct
            name = feature.label.replace("sensory_disagreement_", "").replace("session_composition_", "")
            name = name.replace("whisker_interaction", "whisker").replace("_", " ")
            if skip:
                unassigned = sum(
                    p for c, p in zip(feature.category_labels, feature.percentages, strict=True) if c in skip
                )
                name += f" (unassigned {unassigned:.0f}%)"
            names.append(name)
            receipt.mark("drawn", feature, "4 state distributions")
        ax.set_yticks(range(len(features)), names, fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        if not skip:
            ax.set_xlim(0, 100)
        ax.set_xlabel("Whole-session duration (%)", fontsize=8)
        detail = (features[0].description or "").splitlines()[-1] if features is agreement else ""
        ax.set_title(f"{title}{' — ' + detail if detail else ''}", fontsize=9, loc="left")
        if used:
            ax.legend(
                list(used.values()),
                list(used),
                fontsize=7,
                ncol=min(len(used), 5),
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.32),
            )


def render_composite(variants: list[Variant], header: dict[str, Any] | None = None) -> tuple[Figure, Receipt]:
    """Render one recording (all its scoring variants) as a single figure."""
    header = header or {}
    receipt = Receipt()
    manifest = variants[0].manifest
    frame = _frame(manifest)
    aspect = frame.shape[0] / max(frame.shape[1], 1)
    groups = _matrix_groups(variants, receipt)
    matrix_rows = len(_pack(groups))
    series = _series_for(manifest, receipt)
    timeline_rows = sum(len(_scored(v.manifest)[0]) for v in variants) + sum(
        len(d.components)
        for d in _decomposition_by_heuristic(manifest).values()
        if d.heuristic_name in _scored(manifest)[0]
    )
    heights = [
        _header_height(header),
        _overview_height(variants, aspect),
        matrix_rows * (WIDTH_IN / MAX_COLS * aspect + 0.62),
        timeline_rows * 0.22 + 1.6 * len(series) + 1.0,
        # State distributions need room; without them the band only holds the footer.
        2.6 if any(isinstance(f, FeatureStateDistribution) for f in manifest.feature_inspections) else 0.35,
    ]
    # Explicit grid margins per band: deterministic and much faster than a layout engine.
    fig = plt.figure(figsize=(WIDTH_IN, sum(heights)))
    bands = fig.subfigures(len(heights), 1, height_ratios=heights)
    _header(bands[0], header, variants)
    _overview(bands[1], variants, frame, receipt)
    _decision_matrix(bands[2], groups, frame)
    _timelines(bands[3], variants, series)
    footer = (
        f"Blue = true, red = false, gray = unavailable; trajectory colour = recording time (bar under the map). "
        f"Trial {header.get('trial_id', '')}. Composite v{COMPOSITE_VERSION}."
    )
    _distributions(bands[4], manifest, receipt, footer)
    for variant in variants:
        for feature in variant.manifest.feature_inspections:
            if not any(feature.label in d for d in (receipt.drawn, receipt.folded, receipt.omitted)):
                reason = _omission_reason(feature)
                receipt.mark("folded" if reason.startswith("folded") else "omitted", feature, reason)
    return fig, receipt


def _omission_reason(feature: FeatureInspection) -> str:
    # Experiment-specific emitter features: descriptor ``features.folded`` ({prefix, reason}).
    for rule in all_features("folded"):
        if feature.label.startswith(rule["prefix"]):
            return rule["reason"]
    kind = feature.kind
    if kind == "trajectory":
        return "folded: overview trajectory and pose panel"
    if kind in ("heuristic", "heuristic_cloud", "zone"):
        return "folded: final mask is the first cell of its row and on the timeline"
    if kind in ("quadrant", "scale_derivation", "angle"):
        return "folded: geometry in the overview; scale method in the header"
    return "omitted: not part of the reviewer composite"
