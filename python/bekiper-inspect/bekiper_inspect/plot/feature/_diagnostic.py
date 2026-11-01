"""Full-trial diagnostic drawing from explicit native column bindings.

Sampling affects artists only. Masks, geometry decisions and scalar diagnostics
remain those written by the native analysis, including rejected samples.
"""

from __future__ import annotations
from bekiper_inspect.plot.feature._helpers import _calibration_resolution

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from bekiper_inspect.manifest import InspectionManifest
from bekiper_inspect.plot.generic import plot_gradient_line, plot_temporal_shaded_line
from bekiper_inspect.plot.perimeter import plot_perimeter_on_ax
from bekiper_inspect.plot.theme import get_active_theme

from ._helpers import _frame_scale, _metric_to_frame_pixels


def bound_ray_components(manifest: InspectionManifest, result_column: str) -> list[object]:
    """Use the canonical native decomposition for a parent feature's field."""
    names = {meta.name for meta in manifest.heuristics if meta.result_column == result_column}
    for feature in manifest.feature_inspections:
        if (
            feature.kind == "heuristic_decomposition"
            and feature.label.startswith("heuristic_decomposition_")
            and (feature.result_column == result_column or feature.heuristic_name in names)
        ):
            components = [c for c in feature.components if c.ray_origin_label is not None]
            if components:
                return components
    return []


def sample_indices(indices: np.ndarray, limit: int = 120) -> np.ndarray:
    """Deterministic full-span sampling that retains both temporal endpoints."""
    if len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices) - 1, limit, dtype=int)]


def metric_axes(ax: Axes, manifest: InspectionManifest, frame_shape: tuple[int, ...]) -> None:
    mpp = manifest.settings.meters_per_pixel
    sx, sy = _frame_scale(_calibration_resolution(manifest), frame_shape)
    if mpp > 0 and sx > 0 and sy > 0:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * mpp / sx:.2f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * mpp / sy:.2f}"))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    else:
        ax.set_xlabel("x (pixels; scale unavailable)")
        ax.set_ylabel("y (pixels; scale unavailable)")


def draw_state_trajectory(
    ax: Axes,
    coordinates: np.ndarray,
    state: pl.Series,
    *,
    valid_only: bool | None = None,
    object_index: int = 0,
    n_objects: int = 1,
) -> None:
    """Draw recorded states with gaps and markers, including isolated samples.

    In accepted/rejected pairs, gray unknown observations appear once on the
    accepted-context panel; they are never classified as accepted or rejected.
    """
    theme = get_active_theme()
    finite = np.isfinite(coordinates).all(axis=1)
    known = ~state.is_null().to_numpy()
    approved = state.fill_null(False).to_numpy().astype(bool)
    for accepted in (True, False):
        if valid_only is not None and accepted != valid_only:
            continue
        mask = finite & known & (approved if accepted else ~approved)
        if not mask.any():
            continue
        masked = coordinates.copy()
        masked[~mask] = np.nan
        alpha = 0.8 if accepted else 0.5
        if theme.trajectory_gradient:
            plot_gradient_line(ax, masked, object_index=object_index, alpha=alpha, linestyle="-" if accepted else "--")
        else:
            plot_temporal_shaded_line(
                ax,
                masked,
                np.arange(len(state)),
                len(state),
                approved=accepted,
                object_index=object_index,
                n_objects=n_objects,
                alpha=alpha,
            )
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=4,
            color=theme.video_true_color if accepted else theme.video_false_color,
            alpha=0.45,
            marker="o" if accepted else "x",
        )
    unknown = finite & ~known
    if valid_only is not False and unknown.any():
        ax.scatter(coordinates[unknown, 0], coordinates[unknown, 1], s=5, color="0.6", marker="+", label="unknown")


def draw_ray_field(
    ax: Axes,
    manifest: InspectionManifest,
    component: object,
    frame_shape: tuple[int, ...],
    *,
    result_column: str | None = None,
    valid_only: bool | None = None,
) -> None:
    theme = get_active_theme()
    origin = getattr(component, "ray_origin_label", None)
    dx, dy = (getattr(component, name, None) for name in ("ray_direction_x_column", "ray_direction_y_column"))
    column = result_column or getattr(component, "result_column")
    required = [dx, dy, column]
    if origin not in manifest.coordinate_columns or any(c not in manifest.df.columns for c in required):
        ax.text(0.5, 0.5, "Full-trial ray inputs unavailable", transform=ax.transAxes, ha="center")
        return
    raw_origins = manifest.get_coordinates(origin)
    raw_directions = np.column_stack([manifest.df[dx].to_numpy(), manifest.df[dy].to_numpy()]).astype(float)
    origins = _metric_to_frame_pixels(
        raw_origins, manifest.settings.meters_per_pixel, _calibration_resolution(manifest), frame_shape
    )
    directions = _metric_to_frame_pixels(
        raw_directions, manifest.settings.meters_per_pixel, _calibration_resolution(manifest), frame_shape
    )
    norms = np.linalg.norm(directions, axis=1)
    finite = np.isfinite(origins).all(axis=1) & np.isfinite(norms) & (norms > 0)
    state = manifest.df[column]
    known = ~state.is_null().to_numpy()
    approved = state.fill_null(False).to_numpy().astype(bool)
    arrow_length = min(frame_shape[:2]) * 0.045
    directions[finite] *= (arrow_length / norms[finite])[:, None]
    geometry = getattr(component, "evaluated_perimeter", None)
    target = getattr(component, "perimeter_label", None)
    # Short labels are presentation only; evaluated geometry and columns above
    # retain their exact native bindings, including erosion/expansion labels.
    target_label = target.split("_expanded_", 1)[0] if target else None
    gate_label = component.label.replace("_hits_object", "").replace("_", " ")
    binding_label = f"{target_label}/{gate_label}" if target_label else gate_label
    if geometry is not None:
        plot_perimeter_on_ax(
            ax,
            geometry,
            color=theme.video_perimeter_color,
            meters_per_pixel=manifest.settings.meters_per_pixel,
            as_pixels=True,
            video_resolution=_calibration_resolution(manifest),
            frame_shape=frame_shape,
        )
    tx, ty = (getattr(component, name, None) for name in ("ray_target_x_column", "ray_target_y_column"))
    if tx in manifest.df.columns and ty in manifest.df.columns:
        targets = np.column_stack([manifest.df[tx].to_numpy(), manifest.df[ty].to_numpy()]).astype(float)
        targets = _metric_to_frame_pixels(
            targets, manifest.settings.meters_per_pixel, _calibration_resolution(manifest), frame_shape
        )
        selection = (
            known & (approved if valid_only else ~approved)
            if valid_only is not None
            else np.ones(len(finite), dtype=bool)
        )
        # Historical boundary examples are about two seconds apart. The
        # native targets remain unchanged; only drawing is decimated.
        candidates = np.flatnonzero(finite & selection & np.isfinite(targets).all(axis=1))
        clock = "elapsed_s" if "elapsed_s" in manifest.df.columns else "elapsed"
        elapsed = (
            manifest.df[clock].to_numpy()
            if clock in manifest.df.columns
            else np.arange(len(finite)) / (manifest.video.fps or 1.0)
        )
        candidates = candidates[np.isfinite(elapsed[candidates])]
        bins = np.floor(elapsed[candidates] / 2.0)
        keep = np.r_[True, bins[1:] != bins[:-1]] if candidates.size else np.array([], dtype=bool)
        indices = sample_indices(candidates[keep])
        if indices.size:
            ax.scatter(
                targets[indices, 0],
                targets[indices, 1],
                s=5,
                color=theme.video_point_color,
                alpha=0.45,
                label=f"{binding_label}: targets (2 s)",
            )
            ax.plot(
                np.column_stack([origins[indices, 0], targets[indices, 0]]).T,
                np.column_stack([origins[indices, 1], targets[indices, 1]]).T,
                color="0.4",
                alpha=0.15,
                linewidth=0.5,
            )
    for mask, color, label in [
        (known & approved, theme.video_true_color, "accepted"),
        (known & ~approved, theme.video_false_color, "rejected"),
        (~known, "0.6", "unknown"),
    ]:
        if valid_only is not None:
            displayed = {"accepted", "unknown"} if valid_only else {"rejected"}
            if label not in displayed:
                continue
        indices = sample_indices(np.flatnonzero(finite & mask))
        if indices.size:
            ax.quiver(
                origins[indices, 0],
                origins[indices, 1],
                directions[indices, 0],
                directions[indices, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                alpha=theme.quiver_alpha,
                label=f"{binding_label} {label}: {int((finite & mask).sum())} ({len(indices)} shown)",
            )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            fontsize=max(6, theme.annotation_fontsize - 1),
            frameon=False,
        )
    # Multiple targets/whiskers share an axis; retain one compact annotation
    # with each binding's missingness instead of painting text over itself.
    annotation = next((text for text in ax.texts if text.get_gid() == "native_ray_summary"), None)
    missing_count = int((~finite).sum())
    missing_row = f"{binding_label}: {missing_count} missing/zero" if missing_count else "0 missing/zero directions"
    if annotation is None:
        annotation = ax.text(
            0.01,
            0.99,
            f"Full trial · {len(origins)} frames\n{missing_row}",
            transform=ax.transAxes,
            va="top",
            fontsize=theme.annotation_fontsize,
            bbox={"facecolor": theme.colors()["surface"], "alpha": 0.7, "edgecolor": "none"},
            clip_on=True,
        )
        annotation.set_gid("native_ray_summary")
    elif missing_count:
        existing = annotation.get_text().replace("\n0 missing/zero directions", "")
        annotation.set_text(f"{existing}\n{missing_row}")
    ax.set_xlim(-0.5, frame_shape[1] - 0.5)
    ax.set_ylim(frame_shape[0] - 0.5, -0.5)
    metric_axes(ax, manifest, frame_shape)


def draw_scalar_histogram(ax: Axes, manifest: InspectionManifest, component: object) -> None:
    """Histogram of the native diagnostic, with all finite observations retained."""
    column = getattr(component, "value_column", None)
    if column not in manifest.df.columns:
        ax.text(0.5, 0.5, "Native scalar unavailable", ha="center", transform=ax.transAxes)
        return
    values = manifest.df[column].to_numpy().astype(float)
    mask_column = getattr(component, "result_column")
    state = manifest.df[mask_column] if mask_column in manifest.df.columns else None
    if state is None:
        return
    known = ~state.is_null().to_numpy()
    mask = state.fill_null(False).to_numpy().astype(bool)
    finite = np.isfinite(values)
    theme = get_active_theme()
    edges = (
        np.linspace(0, 180, 19)
        if getattr(component, "value_unit", None) == "degrees"
        else np.histogram_bin_edges(values[finite], bins=20)
    )
    for selection, color, label, style in [
        (known & mask, theme.video_true_color, "accepted", "-"),
        (known & ~mask, theme.video_false_color, "rejected", "--"),
        (~known, "0.6", "unknown", ":"),
    ]:
        ax.hist(values[finite & selection], bins=edges, histtype="step", color=color, label=label, linestyle=style)
    ax.set_xlabel(getattr(component, "value_unit", None) or "native value")
    ax.set_ylabel("Frames")
    ax.legend(fontsize="small")
    ax.set_title(f"{int(finite.sum())} finite / {len(values)} frames")
