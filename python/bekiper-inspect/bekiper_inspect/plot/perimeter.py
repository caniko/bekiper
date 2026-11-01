from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bekiper_inspect.manifest import InspectionManifest, PerimeterSpec
from bekiper_inspect.plot.generic import plot_ellipse


def _draw_perimeter(
    ax: Axes,
    perimeter: PerimeterSpec,
    scale: float,
    sx: float,
    sy: float,
    color: Any,
    linewidth: float | None = None,
) -> None:
    """Core drawing logic shared by both pixel and legacy perimeter paths."""
    resolved_linewidth = 2.0 if linewidth is None else linewidth
    arm_linewidth = 1.5 if linewidth is None else max(resolved_linewidth * 0.75, 0.5)
    if perimeter.shape == "circle":
        center = (perimeter.params["center_x"] * scale * sx, perimeter.params["center_y"] * scale * sy)
        radius = (perimeter.params["radius"] * scale * sx, perimeter.params["radius"] * scale * sy)
        plot_ellipse(ax, center, radius, color=color, linewidth=resolved_linewidth)

    elif perimeter.shape == "rectangle":
        cx = perimeter.params["center_x"] * scale * sx
        cy = perimeter.params["center_y"] * scale * sy
        w = perimeter.params["width"] * scale * sx
        h = perimeter.params["height"] * scale * sy
        rect = patches.Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            fill=False,
            edgecolor=color,
            linewidth=resolved_linewidth,
        )
        ax.add_patch(rect)

    elif perimeter.shape in ("polygon", "triangle"):
        verts = np.array(perimeter.params["vertices"]) * scale
        verts[:, 0] *= sx
        verts[:, 1] *= sy
        closed = np.vstack([verts, verts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=resolved_linewidth)

    elif perimeter.shape == "radial_maze":
        center_verts = np.array(perimeter.params["center_vertices"]) * scale
        center_verts[:, 0] *= sx
        center_verts[:, 1] *= sy
        closed = np.vstack([center_verts, center_verts[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=resolved_linewidth)
        for arm_verts in perimeter.params["arms"]:
            arm = np.array(arm_verts) * scale
            arm[:, 0] *= sx
            arm[:, 1] *= sy
            closed_arm = np.vstack([arm, arm[0]])
            ax.plot(
                closed_arm[:, 0],
                closed_arm[:, 1],
                color=color,
                linewidth=arm_linewidth,
                linestyle="--",
            )


def _frame_scale(
    video_resolution: tuple[int, int],
    frame_shape: tuple[int, ...] | None,
) -> tuple[float, float]:
    """Return (sx, sy) to map video-pixel coords to displayed-frame pixels."""
    if frame_shape is not None and frame_shape[0] > 0 and frame_shape[1] > 0:
        vid_w, vid_h = video_resolution
        frame_h, frame_w = frame_shape[0], frame_shape[1]
        if vid_w > 0 and vid_h > 0:
            return frame_w / vid_w, frame_h / vid_h
    return 1.0, 1.0


def plot_pixel_perimeter_on_ax(
    ax: Axes,
    perimeter: PerimeterSpec,
    color: Any = "g",
    video_resolution: tuple[int, int] = (0, 0),
    frame_shape: tuple[int, ...] | None = None,
    linewidth: float | None = None,
) -> None:
    """Draw a perimeter that is already in video-pixel space.

    Only applies the frame-scale ratio (displayed frame vs video resolution).
    This is the preferred path when ``pixel_perimeters`` are available in
    the manifest — every element goes through the same scaling, enforced
    by the Rust ``ToVideoPixels`` trait.
    """
    sx, sy = _frame_scale(video_resolution, frame_shape)
    _draw_perimeter(ax, perimeter, 1.0, sx, sy, color, linewidth=linewidth)


def plot_perimeter_on_ax(
    ax: Axes,
    perimeter: PerimeterSpec,
    color: Any = "g",
    meters_per_pixel: float = 0.0,
    as_pixels: bool = False,
    video_resolution: tuple[int, int] = (0, 0),
    frame_shape: tuple[int, ...] | None = None,
    linewidth: float | None = None,
) -> None:
    """Draw a perimeter shape on the given axis.

    Supports circle, rectangle, polygon, triangle, and radial_maze.

    .. deprecated::
        Prefer :func:`plot_pixel_perimeter_on_ax` with pre-computed
        pixel-space perimeters from the manifest's ``pixel_perimeters``.
        This legacy function is kept for backwards compatibility with
        old manifests and non-feature plot paths.
    """
    scale = 1.0 / meters_per_pixel if (as_pixels and meters_per_pixel > 0) else 1.0
    sx, sy = _frame_scale(video_resolution, frame_shape)
    _draw_perimeter(ax, perimeter, scale, sx, sy, color, linewidth=linewidth)


def subplot_with_perimeter(
    manifest: InspectionManifest,
    perimeter_label: str | None = None,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **subplot_kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a figure with perimeter(s) pre-drawn on all axes.

    If video resolution is available, sizes the figure accordingly.
    """
    if figsize is None:
        w, h = manifest.video.resolution
        if w > 0 and h > 0:
            aspect = h / w
            figsize = (10, 10 * aspect * nrows / ncols)
        else:
            figsize = (10, 8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplot_kwargs)

    perimeters = [manifest.get_perimeter(perimeter_label)] if perimeter_label else manifest.perimeters

    for ax in np.array(axes).flatten():
        for p in perimeters:
            plot_perimeter_on_ax(ax, p, meters_per_pixel=manifest.settings.meters_per_pixel)

    return fig, axes
