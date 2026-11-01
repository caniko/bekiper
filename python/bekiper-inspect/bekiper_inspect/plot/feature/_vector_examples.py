"""Frame-labelled geometric examples drawn exclusively from native bindings."""

from __future__ import annotations
from bekiper_inspect.plot.feature._helpers import _calibration_resolution
import math
import numpy as np
from matplotlib import pyplot as plt
from bekiper_inspect.plot.perimeter import plot_perimeter_on_ax
from bekiper_inspect.plot.theme import get_active_theme
from ._helpers import _metric_to_frame_pixels
from ._diagnostic import metric_axes


def draw_vector_example(ax, manifest, feature, frame, index):
    c = feature.components[0]
    ax.imshow(frame, aspect="equal")
    columns = [
        c.ray_direction_x_column,
        c.ray_direction_y_column,
        c.ray_target_x_column,
        c.ray_target_y_column,
        c.value_column,
    ]
    if c.ray_origin_label not in manifest.coordinate_columns or any(col not in manifest.df.columns for col in columns):
        ax.text(0.5, 0.5, "Native geometric inputs unavailable", transform=ax.transAxes, ha="center")
        return
    values = np.array([manifest.df[col][index] for col in columns], dtype=float)
    origin = manifest.get_coordinates(c.ray_origin_label)[index]
    if not np.isfinite(values).all() or not np.isfinite(origin).all() or np.linalg.norm(values[:2]) <= 0:
        ax.text(0.5, 0.5, "Native geometric sample unavailable", transform=ax.transAxes, ha="center")
        return
    target = values[2:4]
    # Heading length is decorative. Both endpoints undergo the same affine
    # transform as the boundary and origin; the annotated angle stays native.
    length = max(float(np.linalg.norm(target - origin)), 0.01)
    heading = origin + values[:2] / np.linalg.norm(values[:2]) * length
    points = _metric_to_frame_pixels(
        np.array([origin, heading, target]),
        manifest.settings.meters_per_pixel,
        _calibration_resolution(manifest),
        frame.shape,
    )
    if c.evaluated_perimeter is not None:
        plot_perimeter_on_ax(
            ax,
            c.evaluated_perimeter,
            meters_per_pixel=manifest.settings.meters_per_pixel,
            as_pixels=True,
            video_resolution=_calibration_resolution(manifest),
            frame_shape=frame.shape,
            color="0.5",
        )
    for endpoint, color, label in [(points[1], "#0072B2", "heading"), (points[2], "#E69F00", "native boundary target")]:
        delta = endpoint - points[0]
        ax.quiver(*points[0], *delta, color=color, angles="xy", scale_units="xy", scale=1, label=label, width=0.006)
    ax.scatter(*points[0], color="#CC79A7", s=24)
    ax.scatter(*points[2], color="#E69F00", marker="x", s=28)
    state = manifest.df[c.result_column][index] if c.result_column in manifest.df.columns else None
    status = "missing" if state is None else "accepted" if state else "rejected"
    clock = next((col for col in ["elapsed_s", "elapsed"] if col in manifest.df.columns), None)
    time = manifest.df[clock][index] if clock else index / manifest.video.fps if manifest.video.fps > 0 else None
    time_label = f" · {time:.2f} s" if time is not None and np.isfinite(time) else ""
    ax.set_title(
        f"Frame {manifest.source_frame_index(index)}{time_label} · {status}\nNative angle: {values[4]:.2f}°",
        fontsize=get_active_theme().fontsize,
    )
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    pad = max(12.0, float(np.ptp(points, axis=0).max()) * 0.25)
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].max() + pad, points[:, 1].min() - pad)
    metric_axes(ax, manifest, frame.shape)


def plot_vector_examples(manifest, features):
    from ._story_parts.plot_story import _resolve_frame_for_feature

    columns = min(3, len(features))
    rows = math.ceil(len(features) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5.2 * rows), squeeze=False, layout="constrained")
    for ax, feature in zip(axes.flat, features, strict=False):
        frame, index = _resolve_frame_for_feature(manifest, feature)
        if frame is not None:
            draw_vector_example(ax, manifest, feature, frame, index)
    for ax in list(axes.flat)[len(features) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Native geometric examples — heading and selected boundary vector", fontsize=get_active_theme().title_fontsize
    )
    return fig
