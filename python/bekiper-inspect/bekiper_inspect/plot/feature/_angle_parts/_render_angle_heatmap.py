from bekiper_inspect.plot.feature._helpers import _calibration_resolution
from typing import Any, cast

from .._helpers import _metric_to_frame_pixels, _resolve_metric_perimeter
from ._best_angle_diag_for_frame import *
from ._first_ray_circle_hit import *
from ._select_display_bin_px import *
from .angleoverlaysample import *
from .diagnostic_values import _float_or_nan, _float_or_none, _heuristic_name_from_feature, _parse_diag_result_column


def _render_angle_heatmap(
    ax: Axes,
    heatmap: AngleHeatmap,
    *,
    theme: PlotTheme,
    title: str,
    overlay_samples: Sequence[AngleOverlaySample] | None,
) -> None:
    """Render a precomputed median-angle heatmap."""
    masked = np.ma.masked_invalid(heatmap.values)
    x_min, x_max, y_max, y_min = heatmap.extent
    ax.set_facecolor("#2b2b2b")
    ax.imshow(
        masked,
        extent=(x_min, x_max, y_max, y_min),
        cmap="viridis",
        vmin=0.0,
        vmax=180.0,
        alpha=heatmap.alpha,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=theme.annotation_fontsize, pad=4)
    if overlay_samples:
        draw_angle_samples(
            ax,
            overlay_samples,
            theme=theme,
            comparison_color=theme.video_point_color,
        )
        draw_angle_pair_labels(
            ax,
            overlay_samples,
            theme=theme,
        )


def build_feature_angle_sample(
    manifest: InspectionManifest,
    feature: FeatureAngle,
    frame_idx: int,
    frame_shape: tuple[int, ...],
    *,
    color: str,
) -> AngleOverlaySample | None:
    """Build the shared angle sample used by ``FeatureAngle`` plots."""
    if (
        feature.from_coordinate_label not in manifest.coordinate_columns
        or feature.to_coordinate_label not in manifest.coordinate_columns
    ):
        return None

    from_coords = manifest.get_coordinates(feature.from_coordinate_label)
    to_coords = manifest.get_coordinates(feature.to_coordinate_label)
    if frame_idx >= len(from_coords) or frame_idx >= len(to_coords):
        return None

    mpp = manifest.settings.meters_per_pixel
    origin = _metric_to_frame_pixels(
        from_coords[frame_idx : frame_idx + 1],
        mpp,
        _calibration_resolution(manifest),
        frame_shape,
    )[0]
    target = _metric_to_frame_pixels(
        to_coords[frame_idx : frame_idx + 1],
        mpp,
        _calibration_resolution(manifest),
        frame_shape,
    )[0]
    ox, oy = float(origin[0]), float(origin[1])
    tx, ty = float(target[0]), float(target[1])
    if not all(np.isfinite([ox, oy, tx, ty])):
        return None

    vector = (tx - ox, ty - oy)
    return AngleOverlaySample(
        frame_index=frame_idx,
        origin_xy=(ox, oy),
        primary_vector=vector,
        comparison_vector=None,
        rendered_angle_deg=float(np.degrees(np.arctan2(vector[1], vector[0]))),
        display_color=color,
    )


def build_heuristic_angle_overlay(
    manifest: InspectionManifest,
    feature: FeatureHeuristic,
    frame_idx: int,
    frame_shape: tuple[int, ...],
    *,
    theme: PlotTheme,
    heuristic_name: str | None = None,
    truth_filter: str = "true",
    primary_color: str | None = None,
) -> AngleOverlayData | None:
    """Build the sampled quiver overlay and optional sidecar data."""
    if feature.result_column not in manifest.df.columns:
        return None

    bool_index = manifest.df[feature.result_column].to_numpy().astype(bool)
    diag_groups = _angle_diagnostic_groups(feature, heuristic_name=heuristic_name)
    fallback_candidates = _fallback_ray_candidates(
        feature,
        heuristic_name=heuristic_name,
    )
    fallback_perimeter = (
        _resolve_metric_perimeter(manifest, feature.perimeter_label) if feature.perimeter_label else None
    )

    sample_indices = _heuristic_window_indices(len(bool_index), frame_idx)
    raw_samples: list[AngleOverlaySample] = []
    margins: list[float] = []
    thresholds: list[float] = []
    for idx in sample_indices:
        if truth_filter == "true" and not bool_index[idx]:
            continue
        if truth_filter == "false" and bool_index[idx]:
            continue

        diag = _best_angle_diag_for_frame(
            manifest,
            idx,
            diag_groups,
            frame_shape=frame_shape,
        )
        if diag is not None:
            diag_any = cast(dict[str, Any], diag)
            ox_px, oy_px = cast(tuple[float, float], diag_any["origin_px"])
            hdx, hdy = cast(tuple[float, float], diag_any["heading_unit"])
            obj_dx, obj_dy = cast(tuple[float, float], diag_any["target_unit"])
            angle_deg = float(diag_any["angle_deg"])
            margin_deg = diag_any.get("margin_deg")
            threshold_deg = diag_any.get("threshold_deg")
        else:
            fallback = _fallback_angle_diag_for_frame(
                manifest,
                idx,
                fallback_candidates,
                perimeter=fallback_perimeter,
                frame_shape=frame_shape,
            )
            if fallback is None:
                continue
            fallback_any = cast(dict[str, Any], fallback)
            ox_px, oy_px = cast(tuple[float, float], fallback_any["origin_px"])
            hdx, hdy = cast(tuple[float, float], fallback_any["heading_unit"])
            obj_dx, obj_dy = cast(tuple[float, float], fallback_any["target_unit"])
            angle_deg = float(fallback_any["angle_deg"])
            margin_deg = None
            threshold_deg = None

        if not all(np.isfinite([ox_px, oy_px, hdx, hdy, obj_dx, obj_dy, angle_deg])):
            continue
        if np.hypot(hdx, hdy) < 1e-9:
            continue
        if np.hypot(obj_dx, obj_dy) < 1e-9:
            continue

        display_color = primary_color or (theme.video_true_color if bool_index[idx] else theme.video_false_color)
        if margin_deg is not None and np.isfinite(float(margin_deg)):
            margins.append(float(margin_deg))
        if threshold_deg is not None and np.isfinite(float(threshold_deg)):
            thresholds.append(float(threshold_deg))
        raw_samples.append(
            AngleOverlaySample(
                frame_index=idx,
                origin_xy=(ox_px, oy_px),
                primary_vector=(
                    hdx * ANGLE_HEURISTIC_VECTOR_PX,
                    hdy * ANGLE_HEURISTIC_VECTOR_PX,
                ),
                comparison_vector=(
                    obj_dx * ANGLE_HEURISTIC_VECTOR_PX,
                    obj_dy * ANGLE_HEURISTIC_VECTOR_PX,
                ),
                rendered_angle_deg=angle_deg,
                display_color=display_color,
            )
        )

    if not raw_samples:
        return None

    representative_sample = min(
        raw_samples,
        key=lambda sample: abs(sample.frame_index - frame_idx),
    )
    display_samples, display_bin_px = build_display_angle_samples(raw_samples)
    removed_fraction = 1.0 - (len(display_samples) / len(raw_samples))
    angles = np.array([sample.rendered_angle_deg for sample in raw_samples], dtype=float)
    stats = AngleOverlayStats(
        representative_angle_deg=representative_sample.rendered_angle_deg,
        median_angle_deg=float(np.median(angles)),
        p10_angle_deg=float(np.percentile(angles, 10)),
        p90_angle_deg=float(np.percentile(angles, 90)),
        threshold_angle_deg=(float(np.median(np.asarray(thresholds, dtype=float))) if thresholds else None),
        median_margin_deg=(float(np.median(np.asarray(margins, dtype=float))) if margins else None),
    )
    heatmap = None
    if len(raw_samples) >= ANGLE_DENSE_MIN_SAMPLES or removed_fraction >= ANGLE_DENSE_REMOVED_FRACTION:
        heatmap = _build_angle_heatmap(raw_samples, pad_px=display_bin_px)

    return AngleOverlayData(
        raw_samples=tuple(raw_samples),
        display_samples=tuple(display_samples),
        representative_sample=representative_sample,
        stats=stats,
        heatmap=heatmap,
        removed_fraction=removed_fraction,
        display_bin_px=display_bin_px,
    )


def _diag_column_name(heuristic_name: str, component: str) -> str:
    return f"{DIAGNOSTIC_COLUMN_PREFIX}{heuristic_name}__{component}"


def _olfaction_group(heuristic_name: str) -> AngleDiagnosticGroup:
    return AngleDiagnosticGroup(
        origin_label="nose",
        direction_x_col="snout_dx",
        direction_y_col="snout_dy",
        min_angle_col=_diag_column_name(heuristic_name, "ray_min_angle_deg"),
        margin_col=_diag_column_name(heuristic_name, "ray_angle_margin_deg"),
        target_x_col=_diag_column_name(heuristic_name, "ray_target_x"),
        target_y_col=_diag_column_name(heuristic_name, "ray_target_y"),
    )


def _whisker_group(heuristic_name: str, prefix: str, direction_stem: str) -> AngleDiagnosticGroup:
    return AngleDiagnosticGroup(
        origin_label="center_ear",
        direction_x_col=f"{direction_stem}_dx",
        direction_y_col=f"{direction_stem}_dy",
        min_angle_col=_diag_column_name(heuristic_name, f"{prefix}_min_angle_deg"),
        margin_col=_diag_column_name(heuristic_name, f"{prefix}_angle_margin_deg"),
        target_x_col=_diag_column_name(heuristic_name, f"{prefix}_target_x"),
        target_y_col=_diag_column_name(heuristic_name, f"{prefix}_target_y"),
    )


def _selected_whisker_group(heuristic_name: str) -> AngleDiagnosticGroup:
    return AngleDiagnosticGroup(
        origin_label="center_ear",
        direction_x_col=_diag_column_name(heuristic_name, "selected_ray_dx"),
        direction_y_col=_diag_column_name(heuristic_name, "selected_ray_dy"),
        min_angle_col=_diag_column_name(heuristic_name, "selected_ray_min_angle_deg"),
        margin_col=_diag_column_name(heuristic_name, "selected_ray_angle_margin_deg"),
        target_x_col=_diag_column_name(heuristic_name, "selected_ray_target_x"),
        target_y_col=_diag_column_name(heuristic_name, "selected_ray_target_y"),
        priority=0,
    )


def _exploration_group(heuristic_name: str) -> AngleDiagnosticGroup:
    return AngleDiagnosticGroup(
        origin_label="nose",
        direction_x_col="snout_dx",
        direction_y_col="snout_dy",
        min_angle_col=_diag_column_name(heuristic_name, "within_angle_deg"),
        margin_col=_diag_column_name(heuristic_name, "within_angle_margin_deg"),
        target_x_col=_diag_column_name(heuristic_name, "within_angle_target_x"),
        target_y_col=_diag_column_name(heuristic_name, "within_angle_target_y"),
    )


def _angle_diagnostic_groups(
    feature: FeatureHeuristic,
    *,
    heuristic_name: str | None = None,
) -> tuple[AngleDiagnosticGroup, ...]:
    parsed = _parse_diag_result_column(feature.result_column)
    if parsed is not None:
        parsed_name, component = parsed
        if component == "ray_hits_object":
            return (_olfaction_group(parsed_name),)
        if component == "left_ray_hits_object":
            return (_whisker_group(parsed_name, "left_ray", "left_ear"),)
        if component == "right_ray_hits_object":
            return (_whisker_group(parsed_name, "right_ray", "right_ear"),)
        if component == "within_angle":
            return (_exploration_group(parsed_name),)

    inferred_name = _heuristic_name_from_feature(feature, heuristic_name=heuristic_name)
    if inferred_name is None:
        return ()

    haystacks = (
        heuristic_name,
        feature.label,
        feature.result_column,
    )
    if any("whisker" in text for text in haystacks if text):
        return (
            _selected_whisker_group(inferred_name),
            _whisker_group(inferred_name, "left_ray", "left_ear"),
            _whisker_group(inferred_name, "right_ray", "right_ear"),
        )
    if any("olfaction" in text for text in haystacks if text):
        return (_olfaction_group(inferred_name),)
    if any("exploring" in text for text in haystacks if text):
        return (_exploration_group(inferred_name),)
    return ()


__all__ = [
    "_angle_diagnostic_groups",
    "_diag_column_name",
    "_exploration_group",
    "_float_or_nan",
    "_float_or_none",
    "_heuristic_name_from_feature",
    "_olfaction_group",
    "_parse_diag_result_column",
    "_render_angle_heatmap",
    "_selected_whisker_group",
    "_whisker_group",
    "build_feature_angle_sample",
    "build_heuristic_angle_overlay",
]
