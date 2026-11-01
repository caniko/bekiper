"""Time-encoded snout-heading quivers drawn over the trial frame.

One unit arrow per sampled frame points along ``snout_dx/dy`` (nose minus
center-ear) from the center-ear position; its colour is the recording time.
Binary figures give each class its own time-graded sequential colormap
(``Reds`` / ``Blues`` by default) and colourbar; single-case figures use one
magenta/pink colormap. Only recorded columns are drawn; nothing is rescored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from matplotlib.figure import Figure

from bekiper_inspect.manifest import InspectionManifest
from bekiper_inspect.plot.composite import _frame, _times
from bekiper_inspect.plot.feature._helpers import (
    _calibration_resolution,
    _figsize_from_frame,
    _metric_to_frame_pixels,
)
from bekiper_inspect.plot.theme import get_active_theme

#: Both ends of every colormap that are cut off, so early arrows stay visible on night mode
#: and late ones do not turn black on dark frames.
CMAP_LOW = 0.3
CMAP_HIGH = 0.9
#: Arrow length as a fraction of the frame width.
ARROW_FRACTION = 0.04
_VECTOR_COLUMNS = ("snout_dx", "snout_dy", "center_ear_x", "center_ear_y")


@dataclass
class QuiverSpec:
    """One figure: ``classes`` is ``(name, frame mask)`` pairs; two means binary."""

    stem: str
    title: str
    classes: tuple[tuple[str, np.ndarray], ...]


def time_cmap(name: str, alpha: float = 1.0) -> ListedColormap:
    """``name`` truncated to ``[CMAP_LOW, CMAP_HIGH]``, every colour at ``alpha``."""
    colours = colormaps[name](np.linspace(CMAP_LOW, CMAP_HIGH, 256))
    colours[:, 3] = alpha
    return ListedColormap(colours, name=f"{name}_t")


def _mask(manifest: InspectionManifest, column: str | None) -> np.ndarray | None:
    if column is None or column not in manifest.df.columns:
        return None
    return manifest.df[column].fill_null(False).to_numpy().astype(bool)


def _heuristic(manifest: InspectionManifest, name: str) -> np.ndarray | None:
    return _mask(manifest, next((h.result_column for h in manifest.heuristics if h.name == name), None))


def _union(masks: list[np.ndarray | None]) -> np.ndarray | None:
    present = [m for m in masks if m is not None]
    return np.logical_or.reduce(present) if present else None


def quiver_specs(manifest: InspectionManifest) -> list[QuiverSpec]:
    """The quiver figures whose class columns exist in this trial's table."""
    if not all(c in manifest.df.columns for c in _VECTOR_COLUMNS):
        return []
    everything = np.ones(manifest.df.height, dtype=bool)
    specs = [QuiverSpec("quiver_heading", "Snout heading, whole trial", (("heading", everything),))]

    def binary(stem: str, title: str, a: tuple[str, np.ndarray | None], b: tuple[str, np.ndarray | None]) -> None:
        if a[1] is not None and b[1] is not None:
            specs.append(QuiverSpec(stem, title, ((a[0], a[1]), (b[0], b[1]))))

    binary(
        "quiver_novel_vs_familiar",
        "Snout heading while exploring",
        ("novel", _heuristic(manifest, "exploring_novel")),
        ("familiar", _heuristic(manifest, "exploring_familiar")),
    )
    whisker = _union([_mask(manifest, f"whisker_fan_{o}") for o in ("novel", "familiar")])
    nose = _union([_heuristic(manifest, f"olfaction_{o}") for o in ("novel", "familiar")])
    if whisker is not None and nose is not None:
        # Nose-only: olfaction frames the whisker fan did not also hit.
        binary(
            "quiver_whisker_vs_nose",
            "Snout heading, whisker-fan hit vs nose only",
            ("whisker-fan hit", whisker),
            ("nose only", nose & ~whisker),
        )
    binary(
        "quiver_center_vs_periphery",
        "Snout heading, center vs periphery",
        ("center", _heuristic(manifest, "in_center")),
        ("periphery", _heuristic(manifest, "in_periphery")),
    )
    arms = {
        h.name.removeprefix("in_arm_"): _heuristic(manifest, h.name)
        for h in manifest.heuristics
        if h.name.startswith("in_arm_")
    }
    arms = {k: v for k, v in arms.items() if v is not None}
    if arms:
        binary(
            "quiver_arm_vs_center",
            "Snout heading, in arm vs center",
            ("in arm", _union(list(arms.values()))),
            ("center", _heuristic(manifest, "in_center")),
        )
    for arm, mask in arms.items():
        specs.append(QuiverSpec(f"quiver_heading_arm_{arm}", f"Snout heading in arm {arm}", ((f"arm {arm}", mask),)))
    return specs


def _arrows(manifest: InspectionManifest, mask: np.ndarray, stride: int, frame_shape: tuple[int, ...]):
    """Pixel positions, unit headings and times of every ``stride``-th valid selected frame."""
    df = manifest.df
    cols = np.column_stack([df[c].cast(float).to_numpy() for c in _VECTOR_COLUMNS])
    times = _times(manifest)
    norm = np.hypot(cols[:, 0], cols[:, 1])
    keep = mask & np.isfinite(cols).all(axis=1) & np.isfinite(times) & (norm > 0)
    rows = np.flatnonzero(keep)[:: max(int(stride), 1)]  # every stride-th selected row, so sparse classes survive
    px = _metric_to_frame_pixels(
        cols[rows, 2:], manifest.settings.meters_per_pixel, _calibration_resolution(manifest), frame_shape
    )
    # ponytail: unit vectors in metric space; a non-square pixel aspect skews the angle slightly.
    return px, cols[rows, :2] / norm[rows, None], times[rows]


def plot_quiver(manifest: InspectionManifest, spec: QuiverSpec) -> Figure:
    """Draw ``spec`` over the cached frame; an empty selection still gives a valid figure."""
    theme = get_active_theme()
    frame = _frame(manifest)
    fig, ax = plt.subplots(figsize=_figsize_from_frame(frame), layout="constrained")
    ax.imshow(frame)
    ax.set_axis_off()
    fig.suptitle(spec.title, fontsize=theme.title_fontsize)
    names = theme.quiver_binary_cmaps if len(spec.classes) == 2 else (theme.quiver_single_cmap,)
    drawn = [_arrows(manifest, mask, theme.quiver_stride, frame.shape) for _, mask in spec.classes]
    # One time scale for the whole trial, so every quiver figure of it shades alike.
    trial = _times(manifest)
    norm = Normalize(0.0, float(np.nanmax(trial))) if np.isfinite(trial).any() else None
    halo = to_rgba(theme.colors()["ink"], 0.45)
    length = ARROW_FRACTION * frame.shape[1]
    counts = ", ".join(f"{name}: {len(t)} arrows" for (name, _), (_, _, t) in zip(spec.classes, drawn, strict=True))
    ax.set_title(f"{counts} (every {theme.quiver_stride} frames)", fontsize=theme.fontsize, color=theme.hint_color)
    empty = []
    for (name, _), cmap_name, (px, heading, t) in zip(spec.classes, names, drawn, strict=True):
        if not len(t):
            empty.append(name)
            continue
        quiver = ax.quiver(
            px[:, 0],
            px[:, 1],
            heading[:, 0] * length,
            heading[:, 1] * length,
            t,
            cmap=time_cmap(cmap_name, theme.quiver_alpha),
            norm=norm,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            edgecolors=halo,  # thin ink halo keeps dark arrows readable on dark frames
            linewidth=0.4,
            clip_on=True,
        )
        bar = fig.colorbar(quiver, ax=ax, shrink=0.6, pad=0.02)
        bar.set_label(f"{name}, time (s)", fontsize=theme.fontsize + 1)
        bar.ax.tick_params(labelsize=theme.fontsize)
    if empty:
        ax.text(
            0.5,
            0.5,
            "no frames: " + ", ".join(empty),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=theme.fontsize,
            color=theme.hint_color,
        )
    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)
    return fig
