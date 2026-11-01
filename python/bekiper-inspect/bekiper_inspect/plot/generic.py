from __future__ import annotations

import colorsys
import logging
import math
import os
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from matplotlib import colormaps as _cm
from matplotlib import colors, patches, rcParams
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from bekiper_inspect.plot.color import color_map_by_number
from bekiper_inspect.plot.theme import get_active_theme

logger = logging.getLogger(__name__)

# Legacy module-level constants, retained for downstream code that imports
# them directly. Prefer `get_active_theme().fig_format` / `.min_dpi` for
# new call sites so env/TOML overrides propagate automatically.
INSPECT_FIG_FILE_FORMAT: Literal[".svgz", ".jpg", ".png", ".pdf", ".webp"] = ".webp"
MINIMUM_FIG_DPI = 300
WEBP_MAX_DIMENSION_PX = 16383
WEBP_FALLBACK_FORMAT: Literal[".png"] = ".png"

# Cap on how much `bbox_inches="tight"` is allowed to grow a saved
# figure beyond its nominal size. When a renderer accidentally lets
# overlay artists escape their axes (text without `clip_on=True`,
# quivers extending past `set_xlim` bounds, ...), tight-bbox engulfs
# them and the saved image balloons. The whisker-decomposition story
# regression saved at ~1.6× nominal width and 4× nominal height; a
# 1.6× ceiling rejects that class of bug while leaving room for
# legitimate overhang from suptitles and footers. Callers can opt out
# by passing `bbox_growth_limit=math.inf` to `save_figure`.
BBOX_TIGHT_GROWTH_LIMIT = 1.6


def _tight_bbox_size_inches(fig: Figure) -> tuple[float, float]:
    """Return the effective save size for ``bbox_inches='tight'`` in inches."""
    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    tight_bbox = fig.get_tightbbox(renderer)
    if tight_bbox is None or tight_bbox.width <= 0 or tight_bbox.height <= 0:
        fig_w, fig_h = fig.get_size_inches()
        return float(fig_w), float(fig_h)

    pad_inches = float(rcParams["savefig.pad_inches"])
    return (
        float(tight_bbox.width + (2.0 * pad_inches)),
        float(tight_bbox.height + (2.0 * pad_inches)),
    )


def _webp_safe_dpi(fig: Figure, requested_dpi: int) -> int:
    """Clamp DPI so the saved WebP stays within Pillow's 16383px limit."""
    fig_w, fig_h = _tight_bbox_size_inches(fig)
    max_inches = max(fig_w, fig_h)
    if max_inches <= 0:
        return max(1, requested_dpi)

    max_safe_dpi = int((WEBP_MAX_DIMENSION_PX - 1) / max_inches)
    return max(1, min(requested_dpi, max_safe_dpi))


def _is_webp_dimension_error(exc: Exception) -> bool:
    """Return True when Pillow rejects a WebP because it is too large."""
    message = str(exc).lower()
    return "webp limit" in message or ("encoding error 5" in message and "image size exceeds" in message)


def save_figure(
    fig: Figure,
    output_path: Path,
    fmt: Literal[".svgz", ".jpg", ".png", ".pdf", ".webp"] | None = None,
    close: bool = True,
    bbox_growth_limit: float = BBOX_TIGHT_GROWTH_LIMIT,
) -> Path:
    """Save a figure to disk, creating directories as needed.

    Format and DPI default to :attr:`PlotTheme.fig_format` /
    :attr:`PlotTheme.min_dpi` from the active theme; pass ``fmt``
    explicitly to override. Oversized WebP exports are downscaled to a
    safe DPI first and fall back to PNG if Pillow still rejects them.

    Raises ``RuntimeError`` if ``bbox_inches="tight"`` would grow the
    saved figure beyond ``bbox_growth_limit``× the nominal figure size
    in either dimension — a sign that an artist (text without
    ``clip_on=True``, quiver past ``set_xlim``, ...) escaped its axes.
    Pass ``bbox_growth_limit=math.inf`` to disable for figures that
    legitimately overhang.
    """
    theme = getattr(fig, "_bekiper_plot_theme", get_active_theme())
    effective_fmt = fmt or theme.fig_format
    os.makedirs(output_path.parent, exist_ok=True)
    save_path = output_path.with_suffix(effective_fmt)
    dpi = theme.min_dpi

    # Catch artists that escaped their axes BEFORE saving — silently
    # producing a 4×-larger image is worse than refusing to save.
    if math.isfinite(bbox_growth_limit):
        nominal_w, nominal_h = (float(v) for v in fig.get_size_inches())
        tight_w, tight_h = _tight_bbox_size_inches(fig)
        if (nominal_w > 0 and tight_w / nominal_w > bbox_growth_limit) or (
            nominal_h > 0 and tight_h / nominal_h > bbox_growth_limit
        ):
            suptitle = getattr(fig, "_suptitle", None)
            title = suptitle.get_text() if suptitle else "<no suptitle>"
            raise RuntimeError(
                f"save_figure: bbox_inches='tight' would grow {save_path.name} "
                f"({title!r}) from {nominal_w:.1f}×{nominal_h:.1f} in to "
                f"{tight_w:.1f}×{tight_h:.1f} in (limit {bbox_growth_limit:.1f}×). "
                "An artist likely escaped its axes — check for missing clip_on=True "
                "on text, or quivers extending past set_xlim/set_ylim bounds."
            )

    # WebP has a hard 16383-pixel limit per dimension. Clamp against the
    # tight bounding box, not just the nominal figure size, so annotations
    # added by `bbox_inches="tight"` do not trip the encoder.
    if effective_fmt == ".webp":
        dpi = _webp_safe_dpi(fig, dpi)

    savefig_kwargs: dict[str, Any] = {"bbox_inches": "tight", "dpi": dpi}
    if effective_fmt in (".webp", ".jpg"):
        savefig_kwargs["pil_kwargs"] = {"quality": 85}

    try:
        fig.savefig(save_path, **savefig_kwargs)
    except (OSError, ValueError) as exc:
        if effective_fmt != ".webp" or not _is_webp_dimension_error(exc):
            raise

        fallback_path = output_path.with_suffix(WEBP_FALLBACK_FORMAT)
        fallback_kwargs = dict(savefig_kwargs)
        fallback_kwargs.pop("pil_kwargs", None)
        logger.warning(
            "WebP export exceeded %spx for %s; retrying as %s",
            WEBP_MAX_DIMENSION_PX,
            save_path,
            fallback_path,
        )
        fig.savefig(fallback_path, **fallback_kwargs)
        save_path = fallback_path

    if close:
        plt.close(fig)
    return save_path


def trajectory_segments(
    coordinates: np.ndarray[Any, Any],
    *,
    max_jump: float | None = None,
    jump_mad_factor: float | None = None,
) -> list[slice]:
    """Slices of contiguous trajectory points safe to render as a single line.

    A segment ends when the next sample is non-finite (NaN/inf) or when the
    Euclidean step to the next sample exceeds an outlier threshold. The
    threshold is *max_jump* if given; otherwise it is derived from the
    per-step distance distribution as ``median + jump_mad_factor * MAD``.
    When *jump_mad_factor* is None, the value comes from the active theme.
    Pass ``jump_mad_factor=math.inf`` to disable spatial splitting and rely
    only on NaN gaps.
    """
    coords = np.asarray(coordinates)
    n = len(coords)
    if n == 0:
        return []
    finite = np.isfinite(coords).all(axis=1)
    if jump_mad_factor is None:
        jump_mad_factor = get_active_theme().trajectory_jump_mad_factor

    breaks = np.zeros(max(n - 1, 0), dtype=bool)
    if n >= 2:
        valid_pairs = finite[:-1] & finite[1:]
        if valid_pairs.any():
            diffs = np.diff(coords, axis=0)
            d = np.full(n - 1, np.nan, dtype=np.float64)
            d[valid_pairs] = np.linalg.norm(diffs[valid_pairs], axis=1)
            threshold = max_jump
            if threshold is None and np.isfinite(jump_mad_factor):
                d_finite = d[np.isfinite(d)]
                if len(d_finite) >= 8:
                    med = float(np.median(d_finite))
                    mad = float(np.median(np.abs(d_finite - med)))
                    # Floor MAD to avoid splitting on near-stationary segments
                    # where natural jitter dominates.
                    threshold = med + jump_mad_factor * max(mad, 1e-9)
            if threshold is not None:
                breaks[valid_pairs] = d[valid_pairs] > threshold

    segments: list[slice] = []
    start: int | None = None
    for i in range(n):
        if not finite[i]:
            if start is not None:
                segments.append(slice(start, i))
                start = None
            continue
        if start is None:
            start = i
        elif i > 0 and breaks[i - 1]:
            segments.append(slice(start, i))
            start = i
    if start is not None:
        segments.append(slice(start, n))
    return segments


def boolean_index_truth_sequences(boolean_index: np.ndarray[Any, Any]) -> list[tuple[int, int]]:
    """Find contiguous True sequences, returning (start, end) pairs."""
    if len(boolean_index) == 0:
        return []

    sequences: list[tuple[int, int]] = []
    in_sequence = False
    start = 0

    for i, val in enumerate(boolean_index):
        if val and not in_sequence:
            start = i
            in_sequence = True
        elif not val and in_sequence:
            sequences.append((start, i))
            in_sequence = False

    if in_sequence:
        sequences.append((start, len(boolean_index)))

    return sequences


def ax_hue_plot_coordinate_with_boolean_index(
    ax: Axes,
    boolean_index: np.ndarray[Any, Any],
    coordinates: np.ndarray[Any, Any],
    plot_non_confinement: bool = False,
    plot_line: bool = False,
) -> None:
    """Plot coordinates with color-coding based on a boolean index.

    When plot_line is True, draws colored line segments for True sequences.
    Otherwise, draws scatter points using the active theme's valid/invalid
    scatter colors.
    """
    theme = get_active_theme()
    import polars as pl

    state = pl.Series(np.asarray(boolean_index).tolist(), dtype=pl.Boolean)
    if plot_line:
        from bekiper_inspect.plot.feature._diagnostic import draw_state_trajectory

        draw_state_trajectory(ax, coordinates, state, valid_only=None if plot_non_confinement else True)
    else:
        known = ~state.is_null().to_numpy()
        approved = state.fill_null(False).to_numpy()
        ax.scatter(*coordinates[known & approved].T, label="Valid", color=theme.valid_scatter_color)
        if plot_non_confinement:
            ax.scatter(
                *coordinates[known & ~approved].T,
                label="Invalid",
                color=theme.invalid_scatter_color,
            )
        if (~known).any():
            ax.scatter(*coordinates[~known].T, color="0.6", marker="+", label="unknown")


def ax_hue_plot_coordinates(ax: Axes, coordinates: np.ndarray[Any, Any], marker: str = "x") -> None:
    """Plot coordinates with hue gradient."""
    for color, coord in zip(color_map_by_number(len(coordinates)), coordinates):
        ax.scatter(*coord.T, color=color, marker=marker)


def ax_hue_plot_coordinate_pair_as_lines(
    ax: Axes, coordinates_a: np.ndarray[Any, Any], coordinates_b: np.ndarray[Any, Any]
) -> None:
    """Connect paired coordinates with colored lines."""
    paired = np.dstack((coordinates_a, coordinates_b))
    for color, pair in zip(color_map_by_number(len(paired)), paired):
        ax.plot(*pair, color=color)


def plot_coordinates(
    ax: Axes,
    coordinates: np.ndarray[Any, Any],
    color: Any = None,
    **plot_kwargs: Any,
) -> None:
    """Plot a coordinate trajectory on an axis.

    Splits on NaN gaps and outlier per-step jumps via :func:`trajectory_segments`
    so tracking artifacts do not draw spurious bridges across the panel.
    """
    for seg in trajectory_segments(coordinates):
        chunk = coordinates[seg]
        if len(chunk) < 2:
            continue
        ax.plot(*chunk.T, color=color, **plot_kwargs)


def plot_gradient_line(
    ax: Axes,
    coordinates: np.ndarray[Any, Any],
    object_index: int = 0,
    **plot_kwargs: Any,
) -> None:
    """Plot a trajectory as a line with a color gradient from start to end.

    The start/end colors are picked from a categorical colormap (tab10)
    based on *object_index*, then lightened for the start and saturated
    for the end to create a temporal gradient. The gradient is parameterised
    by global frame position, so segments split out by
    :func:`trajectory_segments` (NaN gaps, tracking-jump outliers) keep their
    correct temporal colour.
    """
    theme = get_active_theme()
    cmap = _cm[theme.trajectory_gradient_colormap]
    base_color = np.array(cmap(object_index % cmap.N)[:3])

    # Derive start (lighter) and end (darker/saturated) from base.
    start = base_color * 0.4 + 0.6  # lighten
    end = base_color * 0.9  # darken slightly

    n = len(coordinates)
    if n < 2:
        return
    x, y = coordinates.T
    lw = plot_kwargs.pop("linewidth", theme.linewidth)
    alpha = plot_kwargs.pop("alpha", 1.0)
    denom = max(n - 1, 1)
    for seg in trajectory_segments(coordinates):
        for i in range(seg.start, seg.stop - 1):
            t = i / denom
            c = start * (1 - t) + end * t
            ax.plot(
                [x[i], x[i + 1]],
                [y[i], y[i + 1]],
                color=c,
                linewidth=lw,
                alpha=alpha,
                **plot_kwargs,
            )


def temporal_color(
    t: float,
    approved: bool,
    object_index: int = 0,
    n_objects: int = 1,
) -> tuple[float, float, float]:
    """Return an RGB colour encoding temporal position within a green/red family.

    Parameters
    ----------
    t:
        Normalised time in ``[0, 1]``.  0 → lightest shade, 1 → darkest.
    approved:
        ``True`` picks the green hue family, ``False`` the red family.
    object_index, n_objects:
        When multiple objects share a panel, each gets a distinct hue
        within the family so trajectories remain distinguishable even
        at the dark end.
    """
    if approved:
        h_center, h_range = 120.0, 90.0  # chartreuse(75) → green(120) → teal(165)
    else:
        h_center, h_range = 0.0, 50.0  # magenta-red(335) → red(0) → orange(25)

    if n_objects > 1:
        h = h_center - h_range / 2 + object_index * h_range / (n_objects - 1)
    else:
        h = h_center
    h = h % 360

    s = 0.85
    l = 0.70 - 0.40 * t  # light early → dark late
    return colorsys.hls_to_rgb(h / 360.0, l, s)


def plot_temporal_shaded_line(
    ax: Axes,
    coordinates: np.ndarray[Any, Any],
    frame_indices: np.ndarray[Any, Any],
    total_frames: int,
    *,
    approved: bool = True,
    object_index: int = 0,
    n_objects: int = 1,
    linewidth: float | None = None,
    alpha: float = 0.8,
) -> None:
    """Plot a trajectory with per-segment temporal shading via *LineCollection*.

    Each segment is coloured by its midpoint frame position: lighter
    shades for early frames, darker for later ones.  The hue is picked
    from a green (approved) or red (rejected) family, offset by
    *object_index* when multiple objects share a panel.
    """
    theme = get_active_theme()
    lw = linewidth if linewidth is not None else theme.linewidth
    n = len(coordinates)
    if n < 2:
        return

    x, y = coordinates.T
    # Build segment pairs and corresponding colours. trajectory_segments()
    # already skips NaN gaps and outlier tracking jumps, so we only iterate
    # over consecutive index pairs inside each contiguous segment.
    segments: list[list[list[float]]] = []
    seg_colors: list[tuple[float, float, float]] = []
    tf = max(total_frames, 1)
    for seg in trajectory_segments(coordinates):
        for i in range(seg.start, seg.stop - 1):
            # Filtered masks must not draw across excluded source frames.
            if frame_indices[i + 1] != frame_indices[i] + 1:
                continue
            segments.append([[x[i], y[i]], [x[i + 1], y[i + 1]]])
            mid_t = (frame_indices[i] + frame_indices[i + 1]) / (2.0 * tf)
            seg_colors.append(temporal_color(mid_t, approved, object_index, n_objects))

    if not segments:
        return
    lc = LineCollection(segments, colors=seg_colors, linewidths=lw, alpha=alpha)
    ax.add_collection(lc)


def plot_occupancy_heatmap(
    ax: Axes,
    coordinates: np.ndarray[Any, Any],
    *,
    bins: int | None = None,
    cmap: str | None = None,
    alpha: float | None = None,
    interpolation: str = "gaussian",
    scale: Literal["linear", "log"] | None = None,
) -> bool:
    """Overlay a 2D occupancy density (histogram) of *coordinates* on *ax*.

    Useful for trajectory panels where a long session draws so many overlapping
    line segments that the path becomes illegible (see e.g. tightly-cropped
    Y-maze arm sections). The density layer reveals where time was actually
    spent without losing the underlying axis extent.

    Non-finite samples are dropped. Returns ``True`` when a heatmap was drawn,
    ``False`` when there were not enough valid points to form a 2D extent.
    Defaults for *bins* / *cmap* / *alpha* come from the active theme.
    """
    theme = get_active_theme()
    pts = np.asarray(coordinates, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return False
    valid = np.isfinite(pts).all(axis=1)
    if not valid.any():
        return False
    pts = pts[valid]

    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    if not (x_max > x_min and y_max > y_min):
        return False

    n_bins = bins if bins is not None else theme.density_bins
    heatmap, xedges, yedges = np.histogram2d(
        pts[:, 0],
        pts[:, 1],
        bins=n_bins,
        range=[(x_min, x_max), (y_min, y_max)],
    )
    heatmap = heatmap.T
    effective_scale = scale if scale is not None else theme.density_scale
    if effective_scale == "log":
        # log1p compresses the dynamic range so a single-frame visit in a
        # rare bin still has nonzero brightness against long-dwell zones.
        heatmap = np.log1p(heatmap)
    max_val = float(heatmap.max())
    if max_val <= 0:
        return False
    heatmap /= max_val

    # Render with per-pixel alpha proportional to density so:
    #   • zero-count bins are fully transparent (the frame outside the
    #     occupancy footprint is preserved — important for non-rectangular
    #     arenas like Y-maze / radial maze);
    #   • low-occupancy bins are translucent rather than painting dark
    #     even when the colormap starts near black (e.g. "hot");
    #   • high-occupancy bins reach the configured peak alpha.
    cmap_obj = plt.get_cmap(cmap if cmap is not None else theme.density_colormap)
    rgba = cmap_obj(heatmap)  # shape (H, W, 4); alpha channel is currently 1.0
    peak_alpha = alpha if alpha is not None else theme.density_alpha
    # Boost the alpha curve with sqrt so mid-density bins are visible
    # without making low-density ones over-saturated.
    rgba[..., 3] = peak_alpha * np.sqrt(np.clip(heatmap, 0.0, 1.0))

    ax.imshow(
        rgba,
        extent=(float(xedges[0]), float(xedges[-1]), float(yedges[-1]), float(yedges[0])),
        aspect="auto",
        interpolation=interpolation,
    )
    return True


def filter_coordinates_by_speed(
    coordinates: np.ndarray[Any, Any],
    *,
    min_speed_units_per_frame: float,
) -> np.ndarray[Any, Any]:
    """Return *coordinates* with low-speed samples replaced by NaN.

    *min_speed_units_per_frame* is the per-step Euclidean distance below
    which a sample is considered "stationary" and masked. Callers convert
    a physical speed (m/s) to this scalar via
    ``min_speed_mps / (meters_per_pixel * fps)`` when working in pixel
    coordinates.

    Per-step distance is symmetric: a sample is masked when its step to
    the *previous* AND to the *next* sample fall below the threshold, so
    isolated detections at the edges of stationary periods are kept.
    Endpoints are kept iff their single available step is above threshold.
    Non-finite inputs propagate.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        return coords.copy()
    if not (np.isfinite(min_speed_units_per_frame) and min_speed_units_per_frame > 0):
        return coords.copy()

    diffs = np.diff(coords, axis=0)
    step = np.linalg.norm(diffs, axis=1)
    n = len(coords)
    prev_step = np.full(n, np.inf)
    next_step = np.full(n, np.inf)
    prev_step[1:] = step
    next_step[:-1] = step
    # A sample is "slow" only when both adjacent steps are below threshold;
    # this avoids masking the first/last frame of a stationary bout.
    slow = (prev_step < min_speed_units_per_frame) & (next_step < min_speed_units_per_frame)
    out = coords.copy()
    out[slow] = np.nan
    return out


def plot_velocity_field(
    ax: Axes,
    coordinates: np.ndarray[Any, Any],
    *,
    bins: int | None = None,
    color: str = "yellow",
    edge_color: str = "black",
    alpha: float = 0.95,
    min_samples_per_bin: int = 3,
) -> bool:
    """Overlay a quiver of mean per-bin velocity computed from *coordinates*.

    The plotted region is split into ``bins x bins`` cells; for each cell
    with at least *min_samples_per_bin* samples we draw an arrow from the
    cell centre in the direction of the mean step vector for samples
    falling in that cell. Magnitudes are normalised per-figure so a single
    fast frame does not dominate.

    Returns ``True`` when at least one arrow was drawn. Defaults for
    *bins* come from the active theme. Works in any coordinate system —
    the caller decides whether *coordinates* are pixel- or metric-space.
    """
    theme = get_active_theme()
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        return False

    diffs = np.diff(coords, axis=0)
    valid = np.isfinite(coords[:-1]).all(axis=1) & np.isfinite(diffs).all(axis=1)
    if not valid.any():
        return False
    pts = coords[:-1][valid]
    vec = diffs[valid]

    n_bins = bins if bins is not None else theme.velocity_field_bins
    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    if not (x_max > x_min and y_max > y_min):
        return False

    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)
    ix = np.clip(np.digitize(pts[:, 0], x_edges) - 1, 0, n_bins - 1)
    iy = np.clip(np.digitize(pts[:, 1], y_edges) - 1, 0, n_bins - 1)

    sum_u = np.zeros((n_bins, n_bins))
    sum_v = np.zeros((n_bins, n_bins))
    cnt = np.zeros((n_bins, n_bins), dtype=np.int64)
    np.add.at(sum_u, (iy, ix), vec[:, 0])
    np.add.at(sum_v, (iy, ix), vec[:, 1])
    np.add.at(cnt, (iy, ix), 1)

    mask = cnt >= min_samples_per_bin
    if not mask.any():
        return False
    mean_u = np.zeros_like(sum_u)
    mean_v = np.zeros_like(sum_v)
    mean_u[mask] = sum_u[mask] / cnt[mask]
    mean_v[mask] = sum_v[mask] / cnt[mask]

    cx = 0.5 * (x_edges[:-1] + x_edges[1:])
    cy = 0.5 * (y_edges[:-1] + y_edges[1:])
    XX, YY = np.meshgrid(cx, cy)

    # Normalise so the longest arrow scales to roughly one cell width.
    scale_u = (x_max - x_min) / max(n_bins, 1)
    norm = np.hypot(mean_u, mean_v)
    peak = float(norm.max()) if mask.any() else 0.0
    if peak <= 0:
        return False
    factor = scale_u / peak
    ax.quiver(
        XX[mask],
        YY[mask],
        mean_u[mask] * factor,
        mean_v[mask] * factor,
        color=color,
        alpha=alpha,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
        headwidth=4.5,
        headlength=5.0,
        # Black outline keeps the arrows readable on both light frames
        # (e.g. white-floored arenas) and dark frames (open field at night).
        edgecolors=edge_color,
        linewidth=0.6,
    )
    return True


def plot_state_colored_trajectory(
    ax: Axes,
    coordinates: np.ndarray[Any, Any],
    states: dict[str, np.ndarray[Any, Any]],
    colors: dict[str, Any] | None = None,
    *,
    default_color: Any = "0.5",
    linewidth: float | None = None,
    alpha: float = 0.85,
) -> None:
    """Draw *coordinates* with each segment coloured by behavioural state.

    *states* maps a label (e.g. ``"freezing"``, ``"exploring"``) to a
    boolean array aligned with *coordinates*. The first matching state
    for a given frame wins (so dict insertion order acts as priority).
    Frames matching no state use *default_color*.

    Splits on tracking-jump outliers and NaN gaps via
    :func:`trajectory_segments`, like the other line helpers.
    """
    theme = get_active_theme()
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        return
    n = len(coords)
    palette = dict(colors) if colors else {}

    # Resolve a per-frame colour table once.
    frame_colors: list[Any] = [default_color] * n
    for label, mask in states.items():
        m = np.asarray(mask, dtype=bool)
        if m.shape[0] != n:
            continue
        c = palette.get(label, default_color)
        for i in range(n):
            if m[i] and frame_colors[i] is default_color:
                frame_colors[i] = c

    lw = linewidth if linewidth is not None else theme.linewidth
    x, y = coords.T
    for seg in trajectory_segments(coords):
        for i in range(seg.start, seg.stop - 1):
            ax.plot(
                [x[i], x[i + 1]],
                [y[i], y[i + 1]],
                color=frame_colors[i],
                linewidth=lw,
                alpha=alpha,
            )


def plot_trajectory_small_multiples(
    fig: Figure,
    coordinates: np.ndarray[Any, Any],
    *,
    n_panels: int,
    frame: np.ndarray[Any, Any] | None = None,
    extent: tuple[float, float, float, float] | None = None,
    titles: list[str] | None = None,
) -> list[Axes]:
    """Split *coordinates* into *n_panels* equal temporal chunks and render each.

    Returns the created axes. *frame* is an optional background image
    (e.g. a video frame) shown under every panel; *extent* fixes the
    axis range so panels are spatially comparable. *titles* labels each
    panel (defaults to ``"chunk i/N"``).

    Useful for sessions long enough that overplotting hides any temporal
    structure: laying chunks side-by-side trades spatial detail for time
    legibility without inventing a colormap.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if n_panels < 1:
        raise ValueError("n_panels must be >= 1")
    n = len(coords)
    chunk = max(1, math.ceil(n / n_panels))

    axes: list[Axes] = []
    for i in range(n_panels):
        ax = fig.add_subplot(1, n_panels, i + 1)
        if frame is not None:
            ax.imshow(frame, aspect="equal")
        sub = coords[i * chunk : min(n, (i + 1) * chunk)]
        plot_coordinates(ax, sub, color=get_active_theme().video_point_color, alpha=0.7)
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[3], extent[2])  # image y is inverted
        title = titles[i] if titles and i < len(titles) else f"chunk {i + 1}/{n_panels}"
        ax.set_title(title, fontsize=get_active_theme().annotation_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
    return axes


def plot_ellipse(
    ax: Axes,
    center: tuple[float, float],
    radius: tuple[float, float] | float,
    color: Any = None,
    linewidth: float = 2.0,
) -> None:
    """Draw a circle or ellipse; a radius tuple contains horizontal/vertical semiaxes."""
    if isinstance(radius, tuple) and radius[0] == radius[1]:
        radius = radius[0]

    color_to_assign = color or "g"

    if isinstance(radius, float):
        circle = patches.Circle(
            center,
            radius,
            fill=False,
            color=colors.to_rgba(color_to_assign),
            linewidth=linewidth,
        )
        ax.add_artist(circle)
    elif isinstance(radius, tuple):
        ellipse = patches.Ellipse(
            center,
            width=2 * radius[0],
            height=2 * radius[1],
            edgecolor=color_to_assign,
            facecolor="none",
            linewidth=linewidth,
        )
        ax.add_patch(ellipse)

    ax.scatter(*center, marker=",")
