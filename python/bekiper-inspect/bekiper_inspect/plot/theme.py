"""Plot theme: centralized visual defaults for inspection figures and videos.

Every color, font size, DPI, format, and ffmpeg encoder preset used by the
inspection plots lives here in a single :class:`PlotTheme` dataclass. The
goal is to have one knob per visual decision, loadable from (in precedence
order): constructor overrides (e.g. CLI flags) → environment → ``[inspect.plot]`` section
of ``bekiper_config.toml`` → built-in defaults.

Effective settings are retained in render receipts. Captured native settings
are replayed with an empty injected environment.

Colours that depend on the page (background, ink, masks, points) come from the
shared ``themekit`` palette for :attr:`PlotTheme.color_mode` (``dark`` by
default, or ``light``); activating a theme applies its matplotlib rcParams, so
every figure created afterwards follows the mode. Colours drawn on video frames
stay frame-relative and do not change with the mode.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields, replace
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

from bekiper_rs import theme_palette, theme_rc, theme_resolve

logger = logging.getLogger(__name__)

FigFormat = Literal[".svgz", ".jpg", ".png", ".pdf", ".webp"]
DEFAULT_CONFIG_FILENAME = "bekiper_config.toml"


# ── Built-in defaults ─────────────────────────────────────────────────────

#: Colormap name used for true-state coloring (matplotlib colormap registry).
DEFAULT_TRUE_COLORMAP: str = "cool"

#: Colormap name used for false-state coloring.
DEFAULT_FALSE_COLORMAP: str = "Wistia"

#: Seaborn palette name for categorical color lookups.
DEFAULT_PALETTE: str = "cool"

#: Scatter color for "valid" (boolean-true) points in generic plots.
DEFAULT_VALID_SCATTER_COLOR: str = "dodgerblue"

#: Scatter color for "invalid" (boolean-false) points in generic plots.
DEFAULT_INVALID_SCATTER_COLOR: str = "crimson"

#: Default video-overlay color for unlabelled keypoints.
DEFAULT_VIDEO_POINT_COLOR: str = "cyan"

#: Color used when a boolean index marks a frame as "true" (e.g. confined).
DEFAULT_VIDEO_TRUE_COLOR: str = "lime"

#: Color used when a boolean index marks a frame as "false".
DEFAULT_VIDEO_FALSE_COLOR: str = "red"

#: Default perimeter color for video overlays (no boolean index).
DEFAULT_VIDEO_PERIMETER_COLOR: str = "b"

#: Line width for trajectory segments in generic plots.
DEFAULT_LINEWIDTH: float = 1.5

#: Primary annotation / body-text font size.
DEFAULT_FONTSIZE: int = 10

#: Font size used for in-figure titles.
DEFAULT_TITLE_FONTSIZE: int = 13

#: Font size used for the dimension-line labels in scale plots.
DEFAULT_ANNOTATION_FONTSIZE: int = 9

#: Alpha applied to the black background of annotation bboxes.
DEFAULT_BBOX_ALPHA: float = 0.7

#: Alpha of the quiver arrows drawn on video frames.
DEFAULT_QUIVER_ALPHA: float = 0.60

#: Time-graded colormaps of the two classes of a binary quiver figure.
DEFAULT_QUIVER_BINARY_CMAPS: tuple[str, str] = ("Reds", "Blues")

#: Time-graded colormap of a single-case quiver figure.
DEFAULT_QUIVER_SINGLE_CMAP: str = "RdPu"

#: One quiver arrow per this many selected frames.
DEFAULT_QUIVER_STRIDE: int = 15

#: Minimum DPI used when saving inspection figures.
DEFAULT_MIN_DPI: int = 300
# One mega-composite per trial: ~2,400 px wide at this DPI (reviewer package).
DEFAULT_COMPOSITE_DPI: int = 110

#: ``dark`` (night mode, the default) or ``light``; selects the themekit palette.
DEFAULT_COLOR_MODE: Literal["dark", "light"] = "dark"

#: File format used for saved inspection figures.
DEFAULT_FIG_FORMAT: FigFormat = ".webp"

#: ``ffmpeg`` encoder preset used by the Rust-native inspection-video renderer.
DEFAULT_FFMPEG_PRESET: str = "slower"

#: Bar color for the "primary" metric (percentages) in heuristic summaries.
DEFAULT_BAR_PRIMARY_COLOR: str = "steelblue"

#: Bar color for the "secondary" metric (durations) in heuristic summaries.
DEFAULT_BAR_SECONDARY_COLOR: str = "darkorange"

#: Alpha applied to the timeline spans that mark heuristic-true intervals.
DEFAULT_TIMELINE_ALPHA: float = 0.4

#: Color used for the soft "hint" text layer on heuristic plots.
DEFAULT_HINT_TEXT_COLOR: str = ""  # empty: the palette's muted ink

#: Default figsize for :func:`plot_heuristic_summary`.
DEFAULT_SUMMARY_FIGSIZE: tuple[float, float] = (12.0, 5.0)

#: Default figsize for :func:`plot_heuristic_timeline`.
DEFAULT_TIMELINE_FIGSIZE: tuple[float, float] = (14.0, 4.0)

#: Default figsize for :func:`plot_heuristic_on_coordinates`.
DEFAULT_COORDINATES_FIGSIZE: tuple[float, float] = (10.0, 10.0)

#: Whether to use gradient-colored trajectory lines (start→end color).
DEFAULT_TRAJECTORY_GRADIENT: bool = False

#: Colormap used to pick per-object gradient start colors.
DEFAULT_TRAJECTORY_GRADIENT_COLORMAP: str = "tab10"

#: Render mode for trajectory panels prone to overplotting (e.g. tightly
#: cropped Y-maze arm sections). ``"line"`` keeps the legacy single-line
#: render; ``"density"`` replaces it with a 2D occupancy heatmap; ``"both"``
#: draws the heatmap behind the line as a legibility aid.
DEFAULT_TRAJECTORY_RENDER_MODE: Literal["line", "density", "both"] = "line"

#: Colormap used for occupancy heatmaps drawn by
#: :func:`plot_occupancy_heatmap` (and the center/periphery panel).
DEFAULT_DENSITY_COLORMAP: str = "hot"

#: Alpha applied to occupancy heatmaps. Low enough that an underlying
#: trajectory or video frame remains visible.
DEFAULT_DENSITY_ALPHA: float = 0.4

#: Number of bins per axis for occupancy heatmaps.
DEFAULT_DENSITY_BINS: int = 40

#: How occupancy heatmap counts are scaled before normalisation.
#: ``"linear"`` is the raw histogram; ``"log"`` applies ``log1p`` so that
#: brief visits to rare locations remain visible against long dwell zones
#: (useful for >10-minute sessions in small arenas).
DEFAULT_DENSITY_SCALE: Literal["linear", "log"] = "linear"

#: Minimum centroid speed (m/s) to draw on trajectory panels. Frames
#: below this threshold are NaN-masked before rendering, removing the
#: "sitting still" scribble that dominates static-period overlays.
#: ``0.0`` disables the filter.
DEFAULT_MIN_SPEED_FILTER: float = 0.0

#: Whether to overlay a quiver of mean velocity per spatial bin on top
#: of the trajectory layer. Useful when occupancy alone hides direction.
DEFAULT_SHOW_VELOCITY_FIELD: bool = False

#: Number of bins per axis for the velocity-field quiver overlay.
DEFAULT_VELOCITY_FIELD_BINS: int = 20

#: Number of side-by-side temporal panels to split a long trajectory
#: into. ``1`` keeps the single-panel render. Currently honoured by the
#: Y-maze arm-sections feature; the dispatcher short-circuits the
#: standard single-axes path and builds an N-panel figure instead.
DEFAULT_SMALL_MULTIPLES_N: int = 1

#: Mapping ``DataFrame column → matplotlib color`` that colours each
#: trajectory segment by behavioural state. The first column whose value
#: is True for a given frame wins, so dict-insertion order acts as
#: priority. Empty (default) keeps the legacy single-colour render.
DEFAULT_STATE_COLOR_COLUMNS: dict[str, str] = {}

#: Robust outlier factor for splitting trajectory plots on tracking jumps.
#: Per-step distances above ``median + factor * MAD`` are treated as
#: tracking artifacts and break the rendered line, preventing matplotlib
#: from drawing spurious bridges across far-apart samples (e.g. the Y-maze
#: arm-tip "triangles"). Set to ``math.inf`` to disable.
DEFAULT_TRAJECTORY_JUMP_MAD_FACTOR: float = 12.0


# ── Skip configuration ───────────────────────────────────────────────────


@dataclass
class SkipConfig:
    """Per-feature suppression rules loaded from ``[inspect.plot.skip]``."""

    kinds: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


# ── Profile configuration ────────────────────────────────────────────────


@dataclass
class FigureProfileRule:
    """Routing rule mapping a feature ``(kind, label)`` to a profile name.

    Both *kind* and *label* are fnmatch patterns; an empty pattern (``""``)
    matches anything. Rules are evaluated in declaration order and the
    first match wins.
    """

    profile: str
    kind: str = ""
    label: str = ""

    def matches(self, kind: str, label: str) -> bool:
        kind_ok = not self.kind or fnmatch(kind, self.kind)
        label_ok = not self.label or fnmatch(label, self.label)
        return kind_ok and label_ok


@dataclass
class ProfileConfig:
    """Container for named profile presets and their figure routing rules.

    A *preset* is a flat ``dict[field_name, value]`` overlaid on the base
    theme via :meth:`PlotTheme.with_profile`. Built-in presets ship with
    the package; users can add or override entries via the
    ``[inspect.plot.profiles.<name>]`` TOML section.
    """

    presets: dict[str, dict[str, Any]] = field(default_factory=dict)
    rules: list[FigureProfileRule] = field(default_factory=list)

    def resolve_name(self, kind: str, label: str) -> str | None:
        """Return the first profile whose rule matches ``(kind, label)``."""
        for rule in self.rules:
            if rule.matches(kind, label):
                return rule.profile
        return None


#: Built-in profile presets shipped with the package. Each entry is a
#: ``dict[field_name, value]`` overlaid on the base theme. Users can add
#: more or override these in ``[inspect.plot.profiles.<name>]``.
BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    # Single-line render — explicit form of the legacy default. Useful for
    # opting a specific panel out of a project-wide density default.
    "line": {
        "trajectory_render_mode": "line",
    },
    # Pure occupancy heatmap — best for very long sessions where the path
    # itself carries no information beyond "where time was spent".
    "density": {
        "trajectory_render_mode": "density",
        "density_alpha": 0.65,
    },
    # Heatmap behind a translucent trajectory — recommended for tightly
    # cropped panels (Y-maze arms, narrow zones) where the line tangle
    # would otherwise be illegible but path order still matters.
    "dense_trajectory": {
        "trajectory_render_mode": "both",
        "density_alpha": 0.45,
        "density_bins": 50,
    },
    # Time-coloured trajectory — colour evolves over the session so the
    # eye can follow path order in a busy panel without a heatmap.
    "temporal": {
        "trajectory_gradient": True,
    },
    # Larger fonts and thicker lines for figures destined for slides or
    # printed reports rather than browser inspection.
    "print_friendly": {
        "linewidth": 2.0,
        "fontsize": 12,
        "title_fontsize": 16,
        "annotation_fontsize": 11,
    },
    # Drop frames where the animal is essentially stationary (default
    # 2 cm/s, matches the freezing threshold). Removes the static scribble
    # that dominates long-session panels.
    "active_only": {
        "min_speed_filter": 0.02,
        "trajectory_render_mode": "line",
    },
    # Logarithmic occupancy heatmap — brief visits stay visible alongside
    # long dwell zones. Ideal for >10-min sessions in small arenas.
    "density_log": {
        "trajectory_render_mode": "density",
        "density_scale": "log",
        "density_alpha": 0.65,
    },
    # Mean-velocity quiver layered on top of a faint occupancy heatmap.
    # Reveals habitual movement direction (corridor traversal, edge
    # following) that occupancy alone hides.
    "velocity_field": {
        "trajectory_render_mode": "density",
        "density_alpha": 0.35,
        "show_velocity_field": True,
    },
    # Colour the trajectory by behavioural state. The state→colour mapping
    # itself is project-specific, so this preset is empty by design;
    # users override it under ``[inspect.plot.profiles.state_colored]`` —
    # for example:
    #   [inspect.plot.profiles.state_colored]
    #   state_color_columns = { freezing = "deepskyblue", grooming = "magenta" }
    "state_colored": {
        "trajectory_render_mode": "line",
    },
    # Split a long session into 3 side-by-side temporal panels. Currently
    # honoured by the Y-maze arm-sections feature.
    "small_multiples_3": {
        "small_multiples_n": 3,
    },
    "small_multiples_4": {
        "small_multiples_n": 4,
    },
}


# ── Dataclass ─────────────────────────────────────────────────────────────


@dataclass
class PlotTheme:
    """Visual configuration for the :mod:`bekiper_inspect.plot` modules.

    Use :meth:`PlotTheme.load` to construct an instance from the full
    precedence chain (defaults → TOML → env → overrides). Importing a
    single shared instance avoids threading the theme through every
    plot-function signature; modules can call :func:`get_active_theme`
    to retrieve the most recent :meth:`PlotTheme.activate` value.
    """

    color_mode: Literal["dark", "light"] = DEFAULT_COLOR_MODE
    true_colormap: str = DEFAULT_TRUE_COLORMAP
    false_colormap: str = DEFAULT_FALSE_COLORMAP
    palette: str = DEFAULT_PALETTE
    valid_scatter_color: str = DEFAULT_VALID_SCATTER_COLOR
    invalid_scatter_color: str = DEFAULT_INVALID_SCATTER_COLOR
    video_point_color: str = DEFAULT_VIDEO_POINT_COLOR
    video_true_color: str = DEFAULT_VIDEO_TRUE_COLOR
    video_false_color: str = DEFAULT_VIDEO_FALSE_COLOR
    video_perimeter_color: str = DEFAULT_VIDEO_PERIMETER_COLOR
    linewidth: float = DEFAULT_LINEWIDTH
    fontsize: int = DEFAULT_FONTSIZE
    title_fontsize: int = DEFAULT_TITLE_FONTSIZE
    annotation_fontsize: int = DEFAULT_ANNOTATION_FONTSIZE
    bbox_alpha: float = DEFAULT_BBOX_ALPHA
    quiver_alpha: float = DEFAULT_QUIVER_ALPHA
    quiver_binary_cmaps: tuple[str, str] = DEFAULT_QUIVER_BINARY_CMAPS
    quiver_single_cmap: str = DEFAULT_QUIVER_SINGLE_CMAP
    quiver_stride: int = DEFAULT_QUIVER_STRIDE
    min_dpi: int = DEFAULT_MIN_DPI
    composite_dpi: int = DEFAULT_COMPOSITE_DPI
    fig_format: FigFormat = DEFAULT_FIG_FORMAT
    ffmpeg_preset: str = DEFAULT_FFMPEG_PRESET
    bar_primary_color: str = DEFAULT_BAR_PRIMARY_COLOR
    bar_secondary_color: str = DEFAULT_BAR_SECONDARY_COLOR
    timeline_alpha: float = DEFAULT_TIMELINE_ALPHA
    hint_text_color: str = DEFAULT_HINT_TEXT_COLOR
    summary_figsize: tuple[float, float] = DEFAULT_SUMMARY_FIGSIZE
    timeline_figsize: tuple[float, float] = DEFAULT_TIMELINE_FIGSIZE
    coordinates_figsize: tuple[float, float] = DEFAULT_COORDINATES_FIGSIZE
    trajectory_gradient: bool = DEFAULT_TRAJECTORY_GRADIENT
    trajectory_gradient_colormap: str = DEFAULT_TRAJECTORY_GRADIENT_COLORMAP
    trajectory_jump_mad_factor: float = DEFAULT_TRAJECTORY_JUMP_MAD_FACTOR
    trajectory_render_mode: Literal["line", "density", "both"] = DEFAULT_TRAJECTORY_RENDER_MODE
    density_colormap: str = DEFAULT_DENSITY_COLORMAP
    density_alpha: float = DEFAULT_DENSITY_ALPHA
    density_bins: int = DEFAULT_DENSITY_BINS
    density_scale: Literal["linear", "log"] = DEFAULT_DENSITY_SCALE
    min_speed_filter: float = DEFAULT_MIN_SPEED_FILTER
    show_velocity_field: bool = DEFAULT_SHOW_VELOCITY_FIELD
    velocity_field_bins: int = DEFAULT_VELOCITY_FIELD_BINS
    small_multiples_n: int = DEFAULT_SMALL_MULTIPLES_N
    state_color_columns: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_STATE_COLOR_COLUMNS),
    )
    skip: SkipConfig = field(default_factory=SkipConfig)
    profiles: ProfileConfig = field(
        default_factory=lambda: ProfileConfig(
            presets={name: dict(preset) for name, preset in BUILTIN_PROFILES.items()},
        ),
    )

    def __post_init__(self) -> None:
        if isinstance(self.min_dpi, bool) or not isinstance(self.min_dpi, int) or self.min_dpi <= 0:
            raise ValueError("Inspection DPI must be a positive integer")
        if isinstance(self.quiver_stride, bool) or not isinstance(self.quiver_stride, int) or self.quiver_stride < 1:
            raise ValueError("Quiver stride must be a positive integer")
        self.quiver_binary_cmaps = tuple(self.quiver_binary_cmaps)  # type: ignore[assignment]
        if len(self.quiver_binary_cmaps) != 2:
            raise ValueError("quiver_binary_cmaps needs exactly two colormap names")
        self.color_mode = theme_resolve(str(self.color_mode))  # type: ignore[assignment]

    def colors(self) -> dict[str, Any]:
        """Semantic colour tokens of :attr:`color_mode` (``ink``, ``true_mask``, ...)."""
        return dict(_palette(self.color_mode))

    @property
    def hint_color(self) -> str:
        return self.hint_text_color or _palette(self.color_mode)["muted"]

    # ── Construction helpers ────────────────────────────────────────

    @classmethod
    def load(
        cls,
        config_path: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        **overrides: Any,
    ) -> PlotTheme:
        """Build a :class:`PlotTheme` using the full precedence chain.

        Parameters
        ----------
        config_path:
            Path to a ``bekiper_config.toml`` file. Its ``[inspect.plot]``
            section is loaded as the file layer.
        **overrides:
            Highest-priority per-field overrides (typically CLI flags).
        """
        values: dict[str, Any] = {}

        if config_path is not None:
            file_values = _load_theme_toml(Path(config_path))
            values.update(file_values)

        environment = os.environ if env is None else env
        defaults = cls()
        for spec in fields(cls):
            if overrides.get(spec.name) is not None:
                continue
            alias = {"min_dpi": "DPI", "fig_format": "FORMAT"}.get(spec.name, spec.name.upper())
            raw = environment.get(f"BEKIPER_PLOT_{alias}", environment.get(f"BEKIPER_PLOT_{spec.name.upper()}"))
            if raw is None:
                continue
            default = getattr(defaults, spec.name)
            if isinstance(default, str):
                values[spec.name] = raw
            elif isinstance(default, (bool, int, float, tuple, dict)):
                parsed = tomllib.loads(f"value = {raw}")["value"]
                if isinstance(default, tuple):
                    parsed = tuple(parsed)
                elif not isinstance(parsed, type(default)):
                    raise ValueError(f"Invalid BEKIPER_PLOT_{alias} type")
                values[spec.name] = parsed
        values.update({k: v for k, v in overrides.items() if v is not None})

        # Build ProfileConfig from the ``profiles`` table and ``figure_profiles``
        # array of tables. Built-in presets are seeded first, so user entries
        # add new profiles or override individual fields of existing ones.
        profiles_raw = values.pop("profiles", None)
        figure_profiles_raw = values.pop("figure_profiles", None)
        profile_cfg = ProfileConfig(
            presets={name: dict(preset) for name, preset in BUILTIN_PROFILES.items()},
        )
        if isinstance(profiles_raw, dict):
            for name, preset in profiles_raw.items():
                if not isinstance(preset, dict):
                    logger.warning("ignoring [inspect.plot.profiles.%s]: not a table", name)
                    continue
                merged = dict(profile_cfg.presets.get(name, {}))
                merged.update(preset)
                profile_cfg.presets[name] = merged
        if isinstance(figure_profiles_raw, list):
            for idx, rule_raw in enumerate(figure_profiles_raw):
                if not isinstance(rule_raw, dict):
                    continue
                profile_name = rule_raw.get("profile")
                if not isinstance(profile_name, str) or not profile_name:
                    logger.warning(
                        "ignoring figure_profiles[%d]: missing 'profile' name",
                        idx,
                    )
                    continue
                profile_cfg.rules.append(
                    FigureProfileRule(
                        profile=profile_name,
                        kind=str(rule_raw.get("kind", "")),
                        label=str(rule_raw.get("label", "")),
                    )
                )

        # Build SkipConfig from the nested ``skip`` dict (TOML).
        skip_raw = values.pop("skip", None)
        skip_cfg = SkipConfig()
        if isinstance(skip_raw, dict):
            skip_cfg = SkipConfig(
                kinds=skip_raw.get("kinds", []),
                labels=skip_raw.get("labels", []),
            )

        # Drop keys that are not dataclass fields (keeps extra="ignore"
        # semantics without requiring pydantic).
        valid = {f.name for f in fields(cls)}
        values = {k: v for k, v in values.items() if k in valid}
        values["skip"] = skip_cfg
        values["profiles"] = profile_cfg

        return cls(**values)

    def with_overrides(self, **overrides: Any) -> PlotTheme:
        """Return a copy of the theme with the given fields replaced."""
        return replace(self, **overrides)

    def with_profile(self, name: str) -> PlotTheme:
        """Return a copy of the theme with profile ``name`` applied.

        Unknown names are a soft error: a warning is logged and the
        unmodified theme is returned, so a typo or stale config does not
        break rendering.
        """
        preset = self.profiles.presets.get(name)
        if preset is None:
            logger.warning("plot profile %r not found; using base theme", name)
            return self
        # Filter to known fields so an old preset referencing a removed
        # field cannot crash dataclass replacement.
        valid = {f.name for f in fields(self)}
        cleaned = {k: v for k, v in preset.items() if k in valid}
        if len(cleaned) != len(preset):
            unknown = sorted(set(preset) - set(cleaned))
            logger.warning(
                "profile %r references unknown theme fields: %s",
                name,
                unknown,
            )
        return replace(self, **cleaned)

    def resolve_for_figure(self, kind: str, label: str) -> PlotTheme:
        """Return the theme to use for a feature with the given ``kind``/``label``.

        Returns ``self`` when no routing rule matches, so the active theme
        is unchanged for features without a configured profile.
        """
        name = self.profiles.resolve_name(kind, label)
        if name is None:
            return self
        return self.with_profile(name)

    def should_skip_feature(self, kind: str, label: str) -> bool:
        """Return True if this feature should be suppressed by config."""
        if kind in self.skip.kinds:
            return True
        return any(fnmatch(label, pat) for pat in self.skip.labels)

    def quiver_kwargs(self) -> dict[str, Any]:
        """Return the kwargs dict expected by ``matplotlib.Axes.quiver``."""
        return {"alpha": self.quiver_alpha, "angles": "xy"}


# ── Active-theme registry ─────────────────────────────────────────────────

_active_theme: PlotTheme = PlotTheme()


def get_active_theme() -> PlotTheme:
    """Return the currently active :class:`PlotTheme`.

    Defaults to :class:`PlotTheme` with built-in defaults until something
    calls :func:`set_active_theme`.
    """
    return _active_theme


def set_active_theme(theme: PlotTheme) -> None:
    """Install ``theme`` as the process-wide active theme and apply its colour mode."""
    global _active_theme
    _active_theme = theme
    apply_color_mode(theme.color_mode)


@functools.lru_cache(maxsize=2)
def _palette(mode: str) -> dict[str, Any]:
    return dict(theme_palette(mode))


def apply_color_mode(mode: str) -> None:
    """Point matplotlib's defaults (background, ink, ticks, cycle) at ``mode``'s palette."""
    import matplotlib

    matplotlib.rcParams.update(theme_rc(mode))


@contextlib.contextmanager
def active_theme_for_figure(kind: str, label: str) -> Iterator[PlotTheme]:
    """Temporarily swap the active theme to the profile resolved for ``kind``/``label``.

    On entry the current active theme's :meth:`PlotTheme.resolve_for_figure`
    is consulted; if a profile matches, its overrides are layered on top
    and the result is installed as the active theme. The previous active
    theme is restored on exit (even if the body raises).

    Yields the resolved theme so callers can inspect it without a second
    :func:`get_active_theme` call.
    """
    previous = get_active_theme()
    resolved = previous.resolve_for_figure(kind, label)
    if resolved is previous:
        yield previous
        return
    try:
        set_active_theme(resolved)
        yield resolved
    finally:
        set_active_theme(previous)


# ── Internal helpers ──────────────────────────────────────────────────────


def _load_theme_toml(config_path: Path) -> dict[str, Any]:
    """Extract the ``[inspect.plot]`` section of a TOML file."""
    if not config_path.is_file():
        return {}
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        logger.warning("failed to parse %s: %s", config_path, e)
        return {}
    inspect = data.get("inspect")
    if not isinstance(inspect, dict):
        return {}
    plot = inspect.get("plot")
    if not isinstance(plot, dict):
        return {}
    return plot


# Night mode is the default for every figure, even before a CLI loads a theme.
apply_color_mode(_active_theme.color_mode)
