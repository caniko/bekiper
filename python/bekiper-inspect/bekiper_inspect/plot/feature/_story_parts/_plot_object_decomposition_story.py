from .._renderers import _draw_heuristic_decomposition_panels, _decomposition_columns
from .plot_story import *


def _plot_object_decomposition_story(
    manifest: InspectionManifest,
    group: StoryGroup,
) -> Figure | None:
    """Render one merged decomposition figure with one column per object target."""
    theme = get_active_theme()
    perim_labels = [p.label for p in manifest.perimeters]
    layouts: list[
        tuple[
            str,
            list[tuple[str, FeatureHeuristic | FeatureHeuristicCloud, str | None]],
            int,
            int,
            np.ndarray,
            int,
            FeatureHeuristicDecomposition,
        ]
    ] = []

    for feature in group.features:
        if not isinstance(feature, FeatureHeuristicDecomposition):
            continue

        frame, frame_idx = _resolve_frame_for_feature(manifest, feature)
        if frame is None:
            continue

        panels = _build_heuristic_decomposition_panels(feature)
        n_panels = max(1, len(panels))
        n_cols = _decomposition_columns(feature, n_panels)
        n_rows = int(math.ceil(n_panels / n_cols))
        target_label = _heuristic_target_label(feature.heuristic_name, perim_labels)
        layouts.append(
            (
                _humanize_story_label(target_label or feature.heuristic_name),
                panels,
                n_rows,
                n_cols,
                frame,
                frame_idx,
                feature,
            )
        )

    if not layouts:
        return None

    fig = plt.figure(
        figsize=(
            sum(n_cols * 5.5 for _, _, _, n_cols, _, _, _ in layouts),
            max(n_rows * (4.8 * frame.shape[0] / frame.shape[1] + 1.6) for _, _, n_rows, _, frame, _, _ in layouts)
            + 0.5,
        ),
        layout="constrained",
    )
    gs = GridSpec(
        1,
        len(layouts),
        figure=fig,
        width_ratios=[n_cols for _, _, _, n_cols, _, _, _ in layouts],
        wspace=0.12,
    )

    for column, (target_label, panels, n_rows, n_cols, frame, frame_idx, feature) in enumerate(layouts):
        subgs = gs[0, column].subgridspec(
            n_rows + 1,
            n_cols,
            height_ratios=[0.07] + [1] * n_rows,
            hspace=0.04,
            wspace=0.08,
        )

        ax_title = fig.add_subplot(subgs[0, :])
        ax_title.axis("off")
        ax_title.text(
            0.5,
            0.4,
            target_label,
            ha="center",
            va="center",
            fontsize=theme.annotation_fontsize + 1,
            fontweight="bold",
            transform=ax_title.transAxes,
        )

        slots = [subgs[index // n_cols + 1, index % n_cols] for index in range(len(panels))]
        _draw_heuristic_decomposition_panels(
            fig, slots, panels, manifest, frame, frame_idx, frame.shape, diagnostics=feature.components
        )

        for index in range(len(panels), n_rows * n_cols):
            row = (index // n_cols) + 1
            col = index % n_cols
            ax_empty = fig.add_subplot(subgs[row, col])
            ax_empty.axis("off")

    fig.suptitle(group.title, fontsize=theme.title_fontsize)
    return fig
