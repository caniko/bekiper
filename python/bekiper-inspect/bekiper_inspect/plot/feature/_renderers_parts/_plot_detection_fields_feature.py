from bekiper_inspect.plot.feature._helpers import (
    _calibration_resolution,
    _detection_hits,
    _metric_to_frame_pixels,
)
from ._draw_perimeter_overlay import *
from bekiper_inspect.manifest import FeatureDetectionFields

# Frame overlays keep fixed colours across night/day mode (AGENTS.md Theming).
_CONE_COLOR = "#E69F00"
_FAN_COLORS = {"left": "#56B4E9", "right": "#CC79A7"}
_HIT_OUTLINE = "#D55E00"


def _sector(apex: np.ndarray, axis: float, half_deg: float, radius: float) -> np.ndarray:
    """Metric polygon of a circular sector (apex + arc), angles in stored coordinates."""
    angles = axis + np.radians(np.linspace(-half_deg, half_deg, 33))
    arc = apex + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return np.vstack([apex, arc])


def _plot_detection_fields_feature(
    ax: Axes,
    manifest: InspectionManifest,
    feature: FeatureDetectionFields,
    frame_idx: int,
    frame_shape: tuple[int, ...] | None = None,
) -> None:
    """Draw the nose cone, contact circle, whisker apex and both whisker fans.

    Geometry is built in metric space from the nose/head coordinates at
    *frame_idx* and mapped through the shared metric-to-frame-pixel path (so
    per-axis scale is honoured). Fields that hit on this frame are filled;
    the others stay faint. Left fan = +90 degrees from the head axis.
    """
    cols = manifest.coordinate_columns
    if feature.nose_label not in cols or feature.head_label not in cols:
        return
    nose = manifest.get_coordinates(feature.nose_label)
    head = manifest.get_coordinates(feature.head_label)
    if frame_idx >= len(nose) or frame_idx >= len(head):
        return
    n, h = nose[frame_idx].astype(float), head[frame_idx].astype(float)
    if not (np.isfinite(n).all() and np.isfinite(h).all()):
        return
    axis = float(np.arctan2(n[1] - h[1], n[0] - h[0]))

    mpp = manifest.settings.meters_per_pixel
    res = _calibration_resolution(manifest)
    fs = frame_shape or (0, 0)

    def px(points: np.ndarray) -> np.ndarray:
        return _metric_to_frame_pixels(np.atleast_2d(points), mpp, res, fs)

    nose_hit, whisker_hit = _detection_hits(manifest, feature)
    nose_active = bool(nose_hit[:, frame_idx].any()) if nose_hit.size else False
    whisker_cols = feature.whisker_hit_columns
    sides: set[str] = set()
    for i, hit_col in enumerate(whisker_cols):
        if i < len(whisker_hit) and whisker_hit[i, frame_idx]:
            side_col = hit_col.replace("__whisker_hit", "__whisker_side")
            side = manifest.df[side_col][frame_idx] if side_col in manifest.df.columns else None
            sides |= {"left", "right"} if side in (None, "both") else {str(side)}

    theme = get_active_theme()

    def draw(target: Axes) -> None:
        def add(points: np.ndarray, color: str, active: bool, label: str | None = None) -> None:
            target.add_patch(
                patches.Polygon(
                    px(points),
                    closed=True,
                    facecolor=color,
                    alpha=0.45 if active else 0.12,
                    edgecolor=color,
                    linewidth=1.5,
                    label=label,
                )
            )

        add(_sector(n, axis, feature.cone_half_angle_deg, feature.nose_reach_m), _CONE_COLOR, nose_active, "nose cone")
        circle = np.radians(np.linspace(0, 360, 49))
        target.add_patch(
            patches.Polygon(
                px(n + feature.contact_distance_m * np.column_stack([np.cos(circle), np.sin(circle)])),
                closed=True,
                fill=False,
                edgecolor="white",
                linestyle=":",
                linewidth=1.2,
                label="contact",
            )
        )

        fraction = feature.whisker_apex_fraction
        if fraction is not None and feature.whisker_half_angle_deg and feature.whisker_length_m:
            apex = h + fraction * (n - h)
            for side, turn in (("left", np.pi / 2), ("right", -np.pi / 2)):
                add(
                    _sector(apex, axis + turn, feature.whisker_half_angle_deg, feature.whisker_length_m),
                    _FAN_COLORS[side],
                    side in sides,
                    f"{side} whisker fan",
                )
            apex_px = px(apex)[0]
            target.scatter([apex_px[0]], [apex_px[1]], color="white", edgecolors="black", s=36, zorder=6)
        nose_px, head_px = px(n)[0], px(h)[0]
        target.plot([head_px[0], nose_px[0]], [head_px[1], nose_px[1]], color="white", linewidth=1.0, zorder=5)
        target.scatter([nose_px[0]], [nose_px[1]], color=_CONE_COLOR, edgecolors="black", s=40, zorder=6)

        for i, label in enumerate(feature.object_labels):
            hit = bool(
                (i < len(nose_hit) and nose_hit[i, frame_idx]) or (i < len(whisker_hit) and whisker_hit[i, frame_idx])
            )
            try:
                _draw_perimeter_overlay(
                    target,
                    manifest,
                    label,
                    color=_HIT_OUTLINE if hit else theme.video_perimeter_color,
                    frame_shape=frame_shape,
                    linewidth=3.0 if hit else None,
                )
            except StopIteration:
                continue

    draw(ax)
    ax.legend(loc="lower right", fontsize=theme.annotation_fontsize, framealpha=theme.bbox_alpha)

    # Centimetre fields are a few pixels on a whole-arena frame: repeat them in
    # a zoomed inset around the head.
    if ax.images:
        image = ax.images[0]
        inset = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
        inset.imshow(image.get_array(), cmap=image.get_cmap(), extent=image.get_extent())
        draw(inset)
        reach = 1.5 * max(feature.nose_reach_m, feature.whisker_length_m or 0.0)
        corners = px(np.array([n - reach, n + reach]))
        inset.set_xlim(corners[:, 0].min(), corners[:, 0].max())
        lo, hi = corners[:, 1].min(), corners[:, 1].max()
        inset.set_ylim((hi, lo) if ax.yaxis_inverted() else (lo, hi))
        inset.set_xticks([])
        inset.set_yticks([])
        ax.indicate_inset_zoom(inset, edgecolor="white")


__all__ = ["_plot_detection_fields_feature"]
