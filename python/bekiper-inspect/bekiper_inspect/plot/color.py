from __future__ import annotations

from collections.abc import Generator, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import seaborn
from matplotlib import colormaps as _cm
from matplotlib.colors import Colormap
from numpy.typing import NDArray

from bekiper_inspect.plot.theme import get_active_theme


def identity_color(label: str, overrides: dict[str, str] | None = None) -> str:
    """Stable identity styling, independent of panel order or subset membership."""
    if overrides and label in overrides:
        return overrides[label]
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#332288"]
    landmarks = {
        "nose": 0,
        "left_ear": 1,
        "right_ear": 2,
        "center_ear": 3,
        "tail": 4,
        "base_tail": 4,
        "tail_base": 4,
        "body_center": 5,
    }
    states = {"Missing": "#777777", "Unassigned": "#dddddd", "Overlapping targets": "#332288"}
    if label in states:
        return states[label]
    if label in landmarks:
        return palette[landmarks[label]]
    parts = label.split()
    if (
        len(parts) == 3
        and parts[0] in {"P0", "P1"}
        and parts[1] in {"G0", "G1"}
        and parts[2] in {"F0", "F1", "O0", "O1"}
    ):
        return palette[int(parts[0][1]) + 2 * int(parts[1][1]) + 4 * int(parts[2][1])]
    suffix = label[7:]
    if label.startswith("object_") and suffix.isascii() and suffix.isdigit() and any(c != "0" for c in suffix):
        remainder = 0
        for digit in suffix:
            remainder = (10 * remainder + ord(digit) - ord("0")) % len(palette)
        return palette[(remainder - 1) % len(palette)]
    if label in {"familiar", "novel"}:
        return palette[0 if label == "familiar" else 1]
    digest = 0xCBF29CE484222325
    for byte in label.encode():
        digest = ((digest ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return palette[digest % len(palette)]


def _true_cmap() -> Colormap:
    return _cm[get_active_theme().true_colormap]


def _false_cmap() -> Colormap:
    return _cm[get_active_theme().false_colormap]


@lru_cache
def make_color_palette(n_colors: int) -> list[Any]:
    return seaborn.color_palette(get_active_theme().palette, n_colors)  # type: ignore[no-any-return]


def color_map_by_number(number: int, cmap: Colormap | None = None) -> NDArray[np.floating[Any]]:
    cmap = cmap or _true_cmap()
    return cmap(np.linspace(0, 1, number))


def boolean_index_colormap(
    boolean_index: Sequence[bool],
    cmap_true: Colormap | None = None,
    cmap_false: Colormap | None = None,
) -> Generator[Any, None, None]:
    cmap_true = cmap_true or _true_cmap()
    cmap_false = cmap_false or _false_cmap()
    length = len(boolean_index)
    for state, color_true, color_false in zip(
        boolean_index,
        color_map_by_number(length, cmap_true),
        color_map_by_number(length, cmap_false),
    ):
        yield color_true if state else color_false
