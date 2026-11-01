from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bekiper_inspect.manifest import InspectionManifest, PerimeterSpec, ScaleDerivation
from bekiper_inspect.plot.frame import read_first_frame
from bekiper_inspect.plot.perimeter import plot_perimeter_on_ax
from bekiper_inspect.plot.theme import get_active_theme


def _find_perimeter(perimeters: list[PerimeterSpec], label: str) -> PerimeterSpec | None:
    return next((p for p in perimeters if p.label == label), None)


def _draw_dimension_line(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    color: str = "yellow",
    offset: float = 0.0,
) -> None:
    """Draw a double-headed arrow between two points with a label."""
    x0, y0 = start
    x1, y1 = end

    # Apply offset perpendicular to the line direction
    if offset != 0.0:
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy)
        if length > 0:
            nx, ny = -dy / length * offset, dx / length * offset
            x0, y0 = x0 + nx, y0 + ny
            x1, y1 = x1 + nx, y1 + ny

    theme = get_active_theme()
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "lw": 2.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(
        mid_x,
        mid_y,
        label,
        color=color,
        fontsize=theme.annotation_fontsize,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "black",
            "alpha": theme.bbox_alpha,
            "edgecolor": "none",
        },
    )


def _annotate_arena_side(
    ax: Axes,
    perimeter: PerimeterSpec,
    derivation: ScaleDerivation,
) -> None:
    """Draw dimension lines on the arena perimeter showing the measured side."""
    pixel_side = derivation.pixel_measurement
    if pixel_side is None:
        return

    if perimeter.shape == "rectangle":
        cx = perimeter.params["center_x"]
        cy = perimeter.params["center_y"]
        w = perimeter.params["width"]
        h = perimeter.params["height"]
        # Draw on the shorter side
        if w <= h:
            start = (cx - w / 2, cy - h / 2)
            end = (cx + w / 2, cy - h / 2)
        else:
            start = (cx - w / 2, cy - h / 2)
            end = (cx - w / 2, cy + h / 2)
        label = f"{pixel_side:.0f} px = {derivation.physical_meters:.2f} m"
        _draw_dimension_line(ax, start, end, label, offset=-15)

    elif perimeter.shape in ("polygon", "circle"):
        if perimeter.shape == "polygon":
            verts = np.array(perimeter.params["vertices"])
            xs, ys = verts[:, 0], verts[:, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            w, h = x_max - x_min, y_max - y_min
            if w <= h:
                start = (x_min, y_min)
                end = (x_max, y_min)
            else:
                start = (x_min, y_min)
                end = (x_min, y_max)
        else:
            cx = perimeter.params["center_x"]
            cy = perimeter.params["center_y"]
            r = perimeter.params["radius"]
            start = (cx - r, cy)
            end = (cx + r, cy)
        label = f"{pixel_side:.0f} px = {derivation.physical_meters:.2f} m"
        _draw_dimension_line(ax, start, end, label, offset=-15)


def _annotate_diagonal(
    ax: Axes,
    derivation: ScaleDerivation,
) -> None:
    """Draw the measured diagonal between the two annotation endpoints."""
    if derivation.endpoints is None:
        return
    (x0, y0), (x1, y1) = derivation.endpoints
    pixel_len = derivation.pixel_measurement
    phys = derivation.physical_meters
    if pixel_len is not None and phys is not None:
        label = f"{pixel_len:.0f} px = {phys:.2f} m"
    elif pixel_len is not None:
        label = f"{pixel_len:.0f} px"
    else:
        label = "diagonal"
    _draw_dimension_line(ax, (x0, y0), (x1, y1), label)


def _annotate_arm_length(
    ax: Axes,
    perimeter: PerimeterSpec,
    derivation: ScaleDerivation,
) -> None:
    """Draw dimension lines on each arm of a RadialMaze perimeter."""
    if perimeter.shape != "radial_maze":
        return

    arms = perimeter.params.get("arms", [])
    arm_spans = derivation.arm_pixel_spans

    for i, arm_verts in enumerate(arms):
        verts = np.array(arm_verts)
        # Find the two most distant vertices (max span)
        max_dist = 0.0
        best_pair = (0, 1)
        for j in range(len(verts)):
            for k in range(j + 1, len(verts)):
                d = np.linalg.norm(verts[j] - verts[k])
                if d > max_dist:
                    max_dist = float(d)
                    best_pair = (j, k)

        start = tuple(verts[best_pair[0]])
        end = tuple(verts[best_pair[1]])
        span_label = f"{arm_spans[i]:.0f} px" if i < len(arm_spans) else f"{max_dist:.0f} px"
        _draw_dimension_line(ax, start, end, span_label, color="cyan", offset=8)


def plot_scale_derivation(
    manifest: InspectionManifest,
    figsize: tuple[float, float] | None = None,
) -> Figure | None:
    """Create an inspection figure showing how meters-per-pixel was derived.

    Overlays dimension annotations on the video's first frame (if available)
    to show which perimeter measurements determined the scale factor.

    Returns None if the scale was manually specified (no derivation to show).
    """
    derivation = manifest.scale_derivation
    if derivation is None or derivation.method == "manual":
        return None

    # Try to get video frame as background
    frame = read_first_frame(manifest.video.path)

    if figsize is None:
        if frame is not None:
            h, w = frame.shape[:2]
            aspect = h / w
            figsize = (12, 12 * aspect)
        else:
            figsize = (12, 10)

    fig, ax = plt.subplots(figsize=figsize)

    if frame is not None:
        ax.imshow(frame, aspect="equal")

    # Find the reference perimeter (pixel-space: convert from meter-space)
    ref_label = derivation.reference_label
    ref_perimeter = _find_perimeter(manifest.perimeters, ref_label) if ref_label else None

    # Draw all perimeters in pixel space on the frame
    mpp = manifest.settings.meters_per_pixel
    for p in manifest.perimeters:
        color = "lime" if p.label == ref_label else "gray"
        plot_perimeter_on_ax(ax, p, color=color, meters_per_pixel=mpp, as_pixels=True)

    # Draw dimension annotations (convert meter-space perimeter back to pixel-space)
    if ref_perimeter is not None:
        # Create pixel-space copy for annotation geometry
        pixel_params = {}
        for key, val in ref_perimeter.params.items():
            if isinstance(val, (int, float)) and key not in ("label",):
                pixel_params[key] = val / mpp if mpp > 0 else val
            elif isinstance(val, list):
                pixel_params[key] = _scale_nested(val, 1.0 / mpp if mpp > 0 else 1.0)
            else:
                pixel_params[key] = val

        pixel_perim = PerimeterSpec(
            label=ref_perimeter.label,
            shape=ref_perimeter.shape,
            params=pixel_params,
        )

        if derivation.method == "arena_side":
            _annotate_arena_side(ax, pixel_perim, derivation)
        elif derivation.method == "arm_length":
            _annotate_arm_length(ax, pixel_perim, derivation)

    if derivation.method.startswith("annotation_"):
        _annotate_diagonal(ax, derivation)

    # Summary text box
    lines = [f"Method: {derivation.method}"]
    if derivation.physical_meters is not None:
        lines.append(f"Known dimension: {derivation.physical_meters:.4f} m ({derivation.physical_meters * 100:.1f} cm)")
    if derivation.pixel_measurement is not None:
        lines.append(f"Measured: {derivation.pixel_measurement:.1f} px")
    if derivation.arm_pixel_spans:
        spans_str = ", ".join(f"{s:.1f}" for s in derivation.arm_pixel_spans)
        lines.append(f"Arm spans: [{spans_str}] px")
    lines.append(f"Result: {derivation.meters_per_pixel:.6f} m/px")

    theme = get_active_theme()
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=theme.fontsize,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "black",
            "alpha": min(1.0, theme.bbox_alpha + 0.1),
            "edgecolor": "white",
        },
        color="white",
    )

    ax.set_title(f"Scale Derivation — {derivation.method}", fontsize=theme.title_fontsize)

    if frame is not None:
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
    else:
        ax.set_aspect("equal")

    fig.tight_layout()
    return fig


def _scale_nested(val: Any, factor: float) -> Any:
    """Recursively scale numeric values in nested lists (for vertex coordinates)."""
    if isinstance(val, list):
        return [_scale_nested(item, factor) for item in val]
    if isinstance(val, (int, float)):
        return val * factor
    return val
