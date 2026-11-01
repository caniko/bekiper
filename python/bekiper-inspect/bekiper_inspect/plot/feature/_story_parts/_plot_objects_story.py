from bekiper_inspect.plot.feature._helpers import _calibration_resolution
from .plot_story import *
from .._diagnostic import bound_ray_components, draw_ray_field, metric_axes


def _heuristic_type_from_name(name: str, perim_labels: list[str]) -> str:
    """Extract the heuristic type prefix from a result column name."""
    name = name.removesuffix("_raw")
    for plabel in sorted(perim_labels, key=len, reverse=True):
        suffix = f"_{plabel}"
        if name.endswith(suffix):
            return name[: -len(suffix)].removesuffix("_raw")
    return name.removesuffix("_raw")


def _plot_objects_story(
    manifest: InspectionManifest,
    group: StoryGroup,
    frame: np.ndarray | None,
) -> Figure | None:
    """Render a unified story with all objects overlaid.

    Each family occupies a valid/rejected pair, with at most two pairs per row.
    All objects' trajectories are drawn on every panel, each in its
    own gradient color. Olfaction/whisker valid panels include angle
    vector overlays.
    """
    if frame is None:
        return None

    theme = get_active_theme()
    fs = frame.shape
    mpp = manifest.settings.meters_per_pixel

    # ── Discover heuristic types and object labels ──────────────────
    perim_labels = [p.label for p in manifest.perimeters]
    heuristic_features = [f for f in group.features if isinstance(f, FeatureHeuristic)]
    names_by_column = {meta.result_column: meta.name for meta in manifest.heuristics}

    def family(feature: FeatureHeuristic) -> str:
        return _heuristic_type_from_name(
            names_by_column.get(feature.result_column, feature.result_column), perim_labels
        )

    # Determine unique heuristic types in stable order.
    seen_types: list[str] = []
    for feat in heuristic_features:
        ht = family(feat)
        if ht not in seen_types:
            seen_types.append(ht)

    # Determine object labels (non-arena perimeters that appear in features).
    obj_labels: list[str] = []
    for feat in heuristic_features:
        pl = feat.perimeter_label
        if pl and pl != "arena" and pl not in obj_labels:
            obj_labels.append(pl)

    # Map (heuristic_type, object_label) → feature.
    feature_map: dict[tuple[str, str], FeatureHeuristic] = {}
    for feat in heuristic_features:
        ht = family(feat)
        pl = feat.perimeter_label or ""
        feature_map[(ht, pl)] = feat

    # Object index: position in non-arena perimeter list.
    non_arena = [p.label for p in manifest.perimeters if p.label != "arena"]

    def obj_idx(label: str) -> int:
        try:
            return non_arena.index(label)
        except ValueError:
            return 0

    # ── Layout ──────────────────────────────────────────────────────
    n_types = len(seen_types)
    if n_types == 0:
        return None

    pairs_per_row = min(2, n_types)
    n_cols = pairs_per_row * 2
    spatial_rows = math.ceil(n_types / pairs_per_row)
    has_timelines = bool(group.heuristic_names)
    timeline_height = max(1.0, len(group.heuristic_names) * 0.22)
    panel_height = 4.7 * fs[0] / fs[1] + 1.8
    height_ratios = [panel_height] * spatial_rows + ([timeline_height] if has_timelines else [])
    n_rows = spatial_rows + int(has_timelines)
    fig_w = n_cols * 5.0
    fig_h = sum(height_ratios) + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.15,
        wspace=0.05,
    )

    # ── Panels: one valid + rejected pair per heuristic type ───────
    for j, ht in enumerate(seen_types):
        row, pair = divmod(j, pairs_per_row)
        angle_data_by_label: dict[str, AngleOverlayData] = {}
        show_sidecar = False
        for ol in obj_labels:
            feat_opt = feature_map.get((ht, ol))
            if (
                feat_opt is None
                or not _is_angle_heuristic(feat_opt)
                or bound_ray_components(manifest, feat_opt.result_column)
            ):
                continue
            data = _build_angle_heuristic_data(
                manifest,
                feat_opt,
                _pick_frame_for_feature(manifest, feat_opt),
                fs,
            )
            if data is not None:
                angle_data_by_label[ol] = data
            if data is not None and data.heatmap is not None:
                show_sidecar = True

        valid_slot = gs[row, pair * 2]
        ax_sidecar = None
        footer_ax = None
        if angle_data_by_label:
            valid_grid = valid_slot.subgridspec(
                2,
                1,
                height_ratios=[4.0, 0.72],
                hspace=0.02,
            )
            valid_visual = valid_grid[0, 0]
            footer_ax = fig.add_subplot(valid_grid[1, 0])
            if show_sidecar:
                visual_grid = valid_visual.subgridspec(
                    1,
                    2,
                    width_ratios=[3, 2],
                    wspace=0.05,
                )
                ax_valid = fig.add_subplot(visual_grid[0, 0])
                ax_sidecar = fig.add_subplot(visual_grid[0, 1])
            else:
                ax_valid = fig.add_subplot(valid_visual)
        else:
            ax_valid = fig.add_subplot(valid_slot)
        gray_frame = frame.mean(axis=2) if frame.ndim == 3 else frame
        ax_valid.imshow(gray_frame, cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax_reject = fig.add_subplot(gs[row, pair * 2 + 1])
        ax_reject.imshow(gray_frame, cmap="gray", vmin=0, vmax=255, aspect="equal")

        # Draw all object perimeters on both axes.
        for ol in obj_labels:
            try:
                perim = manifest.get_perimeter(ol)
                perimeter_linewidth = (
                    _angle_object_perimeter_linewidth(ol, theme=theme) if angle_data_by_label else None
                )
                for ax in (ax_valid, ax_reject):
                    plot_perimeter_on_ax(
                        ax,
                        perim,
                        color=theme.video_perimeter_color,
                        meters_per_pixel=mpp,
                        as_pixels=True,
                        video_resolution=_calibration_resolution(manifest),
                        frame_shape=fs,
                        linewidth=perimeter_linewidth,
                    )
            except StopIteration:
                pass

        # Overlay each object's trajectory.
        n_objs = len(obj_labels)
        for ol in obj_labels:
            feat_opt = feature_map.get((ht, ol))
            if feat_opt is None:
                continue
            oi = obj_idx(ol)
            _plot_heuristic_trajectory_only(
                ax_valid,
                manifest,
                feat_opt,
                fs,
                object_index=oi,
                n_objects=n_objs,
                valid_only=True,
            )
            _plot_heuristic_trajectory_only(
                ax_reject,
                manifest,
                feat_opt,
                fs,
                object_index=oi,
                n_objects=n_objs,
                valid_only=False,
            )

            # Angle vectors on valid panel for olfaction/whisker.
            if _is_angle_heuristic(feat_opt):
                for component in bound_ray_components(manifest, feat_opt.result_column):
                    draw_ray_field(
                        ax_valid, manifest, component, fs, result_column=feat_opt.result_column, valid_only=True
                    )
                    draw_ray_field(
                        ax_reject, manifest, component, fs, result_column=feat_opt.result_column, valid_only=False
                    )
                angle_data = angle_data_by_label.get(ol)
                if angle_data is not None:
                    _draw_angle_overlay(
                        ax_valid,
                        angle_data,
                        line_width=_resolve_linewidth(feat_opt, theme.linewidth),
                    )

        # Also overlay arena-targeted features if they exist.
        arena_feat = feature_map.get((ht, "arena"))
        if arena_feat is not None:
            _plot_heuristic_trajectory_only(
                ax_valid,
                manifest,
                arena_feat,
                fs,
                object_index=len(non_arena),
                n_objects=n_objs + 1,
                valid_only=True,
            )

        summary_rows: list[str] = []
        merged_samples: list[AngleOverlaySample] = []
        for ol in obj_labels:
            angle_data = angle_data_by_label.get(ol)
            if angle_data is None:
                continue
            summary_rows.append(angle_summary_row(angle_data, label=ol))
            merged_samples.extend(angle_data.raw_samples)
        if footer_ax is not None:
            draw_angle_metric_footer(
                footer_ax,
                left_rows=summary_rows,
                right_rows=[],
                theme=theme,
            )
        if ax_sidecar is not None and merged_samples:
            overlay_samples, _ = build_display_angle_samples(merged_samples)
            draw_angle_heatmap_from_samples(
                ax_sidecar,
                merged_samples,
                theme=theme,
                overlay_samples=overlay_samples,
            )

        # Titles.
        readable = ht.replace("_", " ").title()
        fontsize = theme.annotation_fontsize
        ax_valid.set_title(readable, fontsize=fontsize, pad=4)
        ax_reject.set_title("Rejected", fontsize=fontsize, pad=4)
        for ax in (ax_valid, ax_reject):
            metric_axes(ax, manifest, fs)
            ax.set_xlim(-0.5, fs[1] - 0.5)
            ax.set_ylim(fs[0] - 0.5, -0.5)
        if ax_sidecar is not None:
            ax_sidecar.set_xticks([])
            ax_sidecar.set_yticks([])

    # ── Timeline strip ──────────────────────────────────────────────
    if has_timelines:
        ax_tl = fig.add_subplot(gs[-1, :])
        n_heur = len(group.heuristic_names)
        for k, hname in enumerate(group.heuristic_names):
            draw_timeline_on_ax(
                ax_tl,
                manifest,
                hname,
                label=hname.replace("_", " "),
                y_offset=k,
                row_height=1.0,
                show_stats=True,
            )
        ax_tl.set_ylim(0, max(n_heur, 1))
        ax_tl.set_yticks([])
        ax_tl.set_xlabel("Time (s)", fontsize=theme.fontsize)

    fig.suptitle(group.title, fontsize=theme.title_fontsize)
    return fig


def _set_panel_title(ax: Axes, feat: FeatureInspection, theme: object) -> None:
    """Set a concise panel title from the feature label."""
    label = feat.label
    # Strip common prefixes for brevity.
    for prefix in ("heuristic_", "zone_", "proximity_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    title = getattr(feat, "title_override", None) or label.replace("_", " ").title()
    fontsize = getattr(theme, "annotation_fontsize", 10)
    ax.set_title(title, fontsize=fontsize, pad=4)


def _short_heuristic_label(heuristic_name: str, story_key: str) -> str:
    """Strip the target suffix from a heuristic name for the timeline legend.

    E.g. ``olfaction_object_1`` with key ``object_1`` → ``olfaction``.
    """
    suffix = f"_{story_key}"
    if heuristic_name.endswith(suffix):
        return heuristic_name[: -len(suffix)]
    return heuristic_name


# ── Layout: geometry story ───────────────────────────────────────────


def _plot_geometry_story(
    manifest: InspectionManifest,
    group: StoryGroup,
    frame: np.ndarray | None,
) -> Figure | None:
    if frame is None:
        return None

    theme = get_active_theme()
    fig, ax = plt.subplots(figsize=_figsize_from_frame(frame))
    ax.imshow(frame, aspect="equal")

    fs = frame.shape

    # Overlay centroid trajectory so the figure shows how the animal
    # used the space relative to the geometry.
    centroid_label = _pick_centroid_label(manifest)
    if centroid_label is not None:
        coords = manifest.get_coordinates(centroid_label)
        coords_px = _metric_to_frame_pixels(
            coords,
            manifest.settings.meters_per_pixel,
            _calibration_resolution(manifest),
            fs,
        )
        plot_gradient_line(ax, coords_px, alpha=0.45)

    for feat in group.features:
        if isinstance(feat, FeaturePerimeter):
            _plot_perimeter_feature(ax, manifest, feat, frame_shape=fs)
        elif isinstance(feat, FeatureAngle):
            frame_idx = _pick_frame_for_feature(manifest, feat)
            _plot_angle_feature(ax, manifest, feat, frame_idx, frame_shape=fs)

    ax.set_title(group.title, fontsize=theme.title_fontsize)
    metric_axes(ax, manifest, fs)
    fig.tight_layout()
    return fig


def _pick_centroid_label(manifest: InspectionManifest) -> str | None:
    """Pick the best centroid coordinate label from the manifest.

    Prefers labels containing ``"center"`` that aren't specific body parts
    (e.g. ``mouse_center`` over ``center_ear``).  Falls back to the first
    available label.
    """
    labels = list(manifest.coordinate_columns.keys())
    if not labels:
        return None
    # Prefer body-centroid labels (e.g. "mouse_center", "body_center").
    body_parts = {"ear", "nose", "paw", "tail", "snout", "eye"}
    for label in labels:
        if "center" in label and not any(bp in label for bp in body_parts):
            return label
    # Fall back to any "center" label.
    for label in labels:
        if "center" in label:
            return label
    return labels[0]


# ── Layout: task-specific (already composite) ────────────────────────


#: Single-panel renderers of the self-contained task features (``TASK_STORY_KEYS``), by ``kind``.
_TASK_RENDERERS = {
    "object_discrimination": _plot_nort_discrimination_feature,
    "arm_sequence": _plot_arm_sequence_feature,
    "ymaze_arm_sections": _plot_ymaze_arm_sections_feature,
    "center_periphery": _plot_center_periphery_feature,
}


def _plot_task_story(
    manifest: InspectionManifest,
    group: StoryGroup,
    frame: np.ndarray | None,
) -> Figure | None:
    """Delegate to existing single-panel renderers for task-specific features."""
    if frame is None or not group.features:
        return None

    theme = get_active_theme()
    feat = group.features[0]
    if group.key == "arm_sequence":
        feat = next((item for item in group.features if item.kind == "arm_sequence"), feat)
    render = _TASK_RENDERERS.get(feat.kind)
    if render is None:
        return None
    fig, ax = plt.subplots(figsize=_figsize_from_frame(frame))
    ax.imshow(frame, aspect="equal")
    render(ax, manifest, feat, frame_shape=frame.shape)

    ax.set_title(group.title, fontsize=theme.title_fontsize)
    fig.tight_layout()
    return fig


# ── Fallback: features without spatial panels ────────────────────────


def _plot_fallback(
    manifest: InspectionManifest,
    group: StoryGroup,
    frame: np.ndarray | None,
) -> Figure | None:
    """Render non-spatial features as a simple grid."""
    if not group.features or (frame is None and any(f.kind != "time_series" for f in group.features)):
        return None

    theme = get_active_theme()
    n = len(group.features)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        squeeze=False,
        figsize=(cols * 5.5, rows * (5.0 * frame.shape[0] / frame.shape[1] + 0.8) if frame is not None else rows * 3.8),
        layout="constrained",
        sharex=group.key.startswith("tracking_timeseries_"),
        sharey="row" if group.key.startswith("tracking_timeseries_") else False,
    )

    for ax, feat in zip(axes.flat, group.features):
        from bekiper_inspect.manifest import FeatureTimeSeries, FeatureTrajectory
        from .._renderers import _draw_timeseries_on_ax, _plot_trajectory_feature
        from .._diagnostic import metric_axes

        if isinstance(feat, FeatureTimeSeries):
            frame_idx = _pick_frame_for_feature(manifest, feat)
            _draw_timeseries_on_ax(ax, manifest, feat, frame_idx)
            continue
        current_frame, frame_idx = _resolve_frame_for_feature(manifest, feat)
        current_frame = current_frame if current_frame is not None else frame
        fs = current_frame.shape
        ax.imshow(
            current_frame.mean(axis=2) if current_frame.ndim == 3 else current_frame,
            cmap="gray",
            vmin=0,
            vmax=255,
            aspect="equal",
        )
        if isinstance(feat, FeaturePerimeter):
            _plot_perimeter_feature(ax, manifest, feat, frame_shape=fs)
        elif isinstance(feat, FeatureTrajectory):
            for perimeter in manifest.perimeters:
                plot_perimeter_on_ax(
                    ax,
                    perimeter,
                    meters_per_pixel=manifest.settings.meters_per_pixel,
                    as_pixels=True,
                    video_resolution=_calibration_resolution(manifest),
                    frame_shape=fs,
                )
            available = all(label in manifest.coordinate_columns for label in feat.coordinate_labels)
            if available:
                _plot_trajectory_feature(ax, manifest, feat, frame_shape=fs, frame_idx=frame_idx)
            else:
                ax.text(0.5, 0.5, "Retained stage unavailable", transform=ax.transAxes, ha="center")
            ax.set_title(feat.title_override or f"Frame {manifest.source_frame_index(frame_idx)}")
            metric_axes(ax, manifest, fs)
            if not feat.label.startswith("skeleton_"):
                ax.set_xlim(-0.5, fs[1] - 0.5)
                ax.set_ylim(fs[0] - 0.5, -0.5)

    for ax in list(axes.flat)[n:]:
        ax.axis("off")

    fig.suptitle(group.title, fontsize=theme.title_fontsize)
    return fig
