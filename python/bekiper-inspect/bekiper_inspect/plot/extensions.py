"""Exploratory research-extension figures (reviewer package, "Research extensions" page).

Everything is read from outputs a run already wrote under ``<run>/data/<trial>/``
(``result.toml``, ``trial_context.json``, ``<trial>_evaluated.parquet``) and the
effective parameters recorded in ``result.toml [detailed.effective_parameters.values]``.
Nothing is rescored. A figure that cannot be computed is not an error: it is
recorded in ``index.json`` with a status ("needs cohort", "needs animal metadata",
...) and a one-sentence reason.

Per-frame columns used (prefix ``__bk_diag__exploring_<object>__``, object in
``novel``/``familiar``; coordinates are metres):

* ``final_toleranced`` - the scored exploration frames (seconds = frames x frame step);
* ``outside_object``, ``nose_near``, ``contact_or_angle`` - the nose rule
  (``outside & nose_near & contact_or_angle``); a scored frame that fails it is
  whisker-only when ``whisker_candidate`` holds, otherwise a bridged gap;
* ``within_angle_target_x/y`` with ``nose_x/y`` - nose to nearest object-perimeter
  point (distance in metres);
* ``whisker_nearest_hit_distance_m`` - distance along the whisker fan to the hit;
* ``elapsed_s`` - frame clock.

Per-trial values: NORT ``detailed.indices.discrimination_index``, OFT
``detailed.center_time_pct``, Y-maze ``detailed.analysis`` (``spontaneous_alternation_pct``,
``arm_sequence``, ``correct/incorrect_alternations``) of the principal contexts named by
``analysis_role`` (``ymaze__<variant>``). The Y-maze contexts, their labels and order, the
principal one, the NORT test step and the figures each experiment owns all come from the
experiment descriptors (``bekiper_inspect.experiments``), not from this module.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from bekiper_rs import theme_palette, theme_rc, theme_resolve
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image

from bekiper_inspect.experiments import (
    context_label,
    context_order_key,
    descriptor_for,
    descriptors,
    generic,
    is_comparison,
    section,
    step_for,
    variant_of,
)

DPI = 110
OBJECTS = ("familiar", "novel")
NOSE_ZONE_M = 0.03  # fallback when the run recorded no exploration.nose_max_distance
MIN_CUM_S = 2.0
LOW_ENTRIES, LOW_TRIADS = 8, 5  # behavioral_analysis_requirements.md: low quality flag
FOOTER = "Exploratory — not part of the registered analysis. Computed from existing run outputs; nothing rescored."


class Needs(Exception):
    """A figure cannot be computed from this run; ``status`` and ``reason`` go to index.json."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


@dataclass
class Trial:
    dir: Path
    task: str
    role: str
    ctx: dict[str, Any]
    detailed: dict[str, Any]
    animal: str | None
    time: str | None  # "pre" / "post"
    group: str | None = None
    descriptor: dict[str, Any] = field(default_factory=generic)
    step: str | None = None  # step id of the descriptor, None when no step matches
    _frames: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.dir.name

    def param(self, key: str, default: Any = None) -> Any:
        return self.detailed.get("effective_parameters", {}).get("values", {}).get(key, default)


def _time(ctx: dict[str, Any], name: str) -> str | None:
    """pre/post from ``phase_part``; archived Y-maze trials record no session_context, so fall back to the name."""
    part = {"1": "pre", "2": "post"}.get(str(ctx.get("phase_part", "")))
    return part or ("pre" if "_pre_" in name else "post" if "_post_" in name else None)


def load_trials(run: Path) -> list[Trial]:
    """Every readable ``<run>/data/<trial>/result.toml``; unreadable trials are skipped."""
    trials = []
    for result in sorted((run / "data").glob("*/result.toml")):
        try:
            res = tomllib.loads(result.read_text())
            tc_path = result.with_name("trial_context.json")
            tc = json.loads(tc_path.read_text()) if tc_path.exists() else {}
        except (OSError, ValueError):
            continue
        ctx = res.get("session_context") or tc.get("session_context") or {}
        # Only the recorded animal_id identifies an animal; session metadata or trial names never pair trials.
        animal = tc.get("animal_id")
        task = res.get("task_type", "")
        d = descriptor_for(task)
        time = _time(ctx, result.parent.name)
        facts = {
            "recipe": (((tc.get("experiment") or {}).get("detection") or {}).get("recipe")),
            "stage": str(ctx.get("stage") or ""),
            "phase_part": str(ctx.get("phase_part") or ""),
            "timepoint": time,
            # Session-context `phase` is the metadata batch letter (e.g. "A"), not the protocol phase.
            "phase": None,
        }
        step = step_for(d, facts)
        trials.append(
            Trial(
                result.parent,
                task,
                res.get("analysis_role", ""),
                ctx,
                res.get("detailed", {}),
                str(animal) if animal not in (None, "") else None,
                time,
                descriptor=d,
                step=step["id"] if step else None,
            )
        )
    # Y-maze sessions carry no group: borrow it from the same animal's other assays.
    meta = {t.animal: t.ctx for t in trials if t.animal and t.ctx.get("group") not in (None, "")}
    for t in trials:
        t.group = str(t.ctx["group"]) if t.ctx.get("group") not in (None, "") else None
        if t.group is None and t.animal in meta:
            t.group = str(meta[t.animal]["group"])
    return trials


# ── helpers ───────────────────────────────────────────────────────────────


def _frames(trial: Trial) -> dict[str, Any]:
    """Per-frame boolean masks and distances of one NORT trial (cached on the trial)."""
    if trial._frames:
        return trial._frames
    path = trial.dir / f"{trial.name}_evaluated.parquet"
    if not path.exists():
        raise Needs("needs frame tables", f"{trial.name} has no evaluated frame table.")
    have = set(pl.read_parquet_schema(path))
    cols = ["elapsed_s", "nose_x", "nose_y"]
    for o in OBJECTS:
        p = f"__bk_diag__exploring_{o}__"
        cols += [p + s for s in ("final_toleranced", "outside_object", "nose_near", "contact_or_angle")]
        cols += [p + s for s in ("within_angle_target_x", "within_angle_target_y")]
    missing = [c for c in cols if c not in have]
    if missing:
        raise Needs("needs frame tables", f"{trial.name} lacks evaluated column {missing[0]}.")
    whisker = all(f"__bk_diag__exploring_{o}__whisker_candidate" in have for o in OBJECTS)
    if not whisker:
        raise Needs("needs whisker scoring", f"{trial.name} was scored without the whisker fan.")
    cols += [
        f"__bk_diag__exploring_{o}__{s}"
        for o in OBJECTS
        for s in ("whisker_candidate", "whisker_nearest_hit_distance_m")
    ]
    df = pl.read_parquet(path, columns=[c for c in cols if c in have])
    t = df["elapsed_s"].to_numpy()
    out: dict[str, Any] = {"t": t, "dt": float(np.median(np.diff(t))) if len(t) > 1 else 0.0, "o": {}}
    for o in OBJECTS:
        p = f"__bk_diag__exploring_{o}__"
        flag = lambda s: df[p + s].fill_null(False).to_numpy()  # noqa: E731
        out["o"][o] = {
            "scored": flag("final_toleranced"),
            "nose": flag("outside_object") & flag("nose_near") & flag("contact_or_angle"),
            "whisker": flag("whisker_candidate"),
            "dist": np.hypot(
                df["nose_x"].to_numpy() - df[p + "within_angle_target_x"].to_numpy(),
                df["nose_y"].to_numpy() - df[p + "within_angle_target_y"].to_numpy(),
            ),
            "hit": df[p + "whisker_nearest_hit_distance_m"].to_numpy(),
        }
    trial._frames = out
    return out


def _di(novel: float, familiar: float) -> float:
    total = novel + familiar
    return (novel - familiar) / total if total > 0 else math.nan


def _test_step(d: dict[str, Any]) -> str | None:
    """The step id of the descriptor's endpoint that is tied to a step (NORT DI -> the test)."""
    return next((e["step"] for e in d.get("endpoints") or [] if e.get("step")), None)


def _nort_tests(trials: list[Trial]) -> list[Trial]:
    tests = [t for t in trials if _test_step(t.descriptor) and t.step == _test_step(t.descriptor)]
    if not tests:
        d = next(d for d in descriptors() if _test_step(d))
        label = next(s["label"] for s in d["steps"] if s["id"] == _test_step(d))
        raise Needs("needs cohort", f"The run has no {d['short_name']} {label} trial.")
    return tests


def _group_style(trials: list[Trial], groups: dict[str, str], pal: dict[str, Any]) -> dict[str | None, tuple[str, str]]:
    """{group code: (colour, display name)}; colours come from the palette's categorical cycle."""
    codes = sorted({t.group for t in trials if t.group}, key=lambda c: (len(c), c))
    cycle = pal["categorical"]
    style = {c: (cycle[i % len(cycle)], groups.get(c, f"Group {c}")) for i, c in enumerate(codes)}
    style[None] = (pal["muted"], "Unrecorded group")
    return style


def _legend(fig: Figure, style: dict[str | None, tuple[str, str]], used: set[str | None], pal: dict[str, Any]) -> None:
    handles = [Line2D([], [], color=c, marker="o", lw=1.5, label=n) for k, (c, n) in style.items() if k in used]
    if len(handles) > 1 or (handles and None not in used):
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 4),
            fontsize=8,
            frameon=False,
            labelcolor=pal["ink"],
            bbox_to_anchor=(0.5, 0.035),
        )


def _finish(fig: Figure, pal: dict[str, Any], title: str, legend: bool = False) -> Figure:
    fig.suptitle(title, fontsize=12, color=pal["ink"])
    fig.text(0.5, 0.008, FOOTER, ha="center", va="bottom", fontsize=7.5, color=pal["muted"])
    fig.tight_layout(rect=(0, 0.1 if legend else 0.04, 1, 0.95))
    return fig


def _style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _empty(ax: Any, text: str, pal: dict[str, Any]) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=9, color=pal["muted"])
    ax.set_xticks([])
    ax.set_yticks([])


# ── figures ───────────────────────────────────────────────────────────────


def whisker_contribution(trials: list[Trial], groups: dict[str, str], pal: dict[str, Any]) -> Figure:
    """Scored seconds per object split by nose rule / whisker-only / bridged gap, and DI with vs without whisker-only frames."""
    tests = _nort_tests(trials)
    rows = []
    for t in tests:
        f = _frames(t)
        sec = {}
        for o in OBJECTS:
            m = f["o"][o]
            scored, nose = m["scored"], m["nose"]
            n_nose = scored[nose].sum() * f["dt"] if scored.any() else 0.0
            n_w = (scored & ~nose & m["whisker"]).sum() * f["dt"]
            total = scored.sum() * f["dt"]
            sec[o] = (n_nose, n_w, max(total - n_nose - n_w, 0.0), total)
        rows.append((t, sec))
    style = _group_style(tests, groups, pal)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    # (a) pooled seconds per object
    ax = axes[0]
    colours = (pal["categorical"][0], pal["categorical"][1], pal["unavailable"])
    labels = ("nose rule", "whisker-only", "bridged gap")
    for x, o in enumerate(OBJECTS):
        bottom = 0.0
        for i, (c, lab) in enumerate(zip(colours, labels, strict=True)):
            v = float(np.mean([s[o][i] for _, s in rows]))
            ax.bar(x, v, bottom=bottom, color=c, width=0.55, label=lab if x == 0 else None)
            bottom += v
    ax.set_xticks(range(2), OBJECTS)
    ax.set_ylabel("scored exploration (s, mean per trial)", fontsize=9)
    ax.set_title(f"Seconds per object (n = {len(rows)} trials)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=pal["ink"])
    # (b) DI all vs nose-only
    ax = axes[1]
    di_all = np.array([_di(s["novel"][3], s["familiar"][3]) for _, s in rows])
    di_nose = np.array([_di(s["novel"][0], s["familiar"][0]) for _, s in rows])
    ax.axline((0, 0), slope=1, color=pal["reference"], ls="--", lw=1)
    ax.axhline(0, color=pal["border"], lw=0.8)
    ax.axvline(0, color=pal["border"], lw=0.8)
    ax.scatter(
        di_nose, di_all, s=36, c=[style[t.group][0] for t, _ in rows], edgecolors=pal["ink"], linewidths=0.5, zorder=3
    )
    ax.set_xlabel("DI, nose-rule frames only", fontsize=9)
    ax.set_ylabel("DI, all scored frames", fontsize=9)
    ax.set_title("DI with vs without whisker-only frames", fontsize=10, loc="left")
    # (c) whisker-only share per object
    ax = axes[2]
    share = np.array([[s[o][1] / s[o][3] * 100 if s[o][3] > 0 else math.nan for o in OBJECTS] for _, s in rows])
    for (t, _), pair in zip(rows, share, strict=True):
        ax.plot([0, 1], pair, color=style[t.group][0], marker="o", lw=1, alpha=0.8)
    ax.plot([0, 1], np.nanmean(share, axis=0), color=pal["ink"], marker="_", markersize=22, lw=2.2, zorder=4)
    ax.set_xticks([0, 1], OBJECTS)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("whisker-only share of scored time (%)", fontsize=9)
    ax.set_title("Whisker-only share: familiar vs novel (bold = mean)", fontsize=10, loc="left")
    for a in axes:
        _style(a)
    _legend(fig, style, {t.group for t, _ in rows}, pal)
    return _finish(fig, pal, "Whisker fan versus nose: what touch adds to novelty discrimination", legend=True)


def whisker_reach(trials: list[Trial], groups: dict[str, str], pal: dict[str, Any]) -> Figure:
    """Nose-to-perimeter distance and fan-hit distance in whisker-only scored frames."""
    tests = _nort_tests(trials)
    near, hit = {o: [] for o in OBJECTS}, {o: [] for o in OBJECTS}
    for t in tests:
        f = _frames(t)
        for o in OBJECTS:
            m = f["o"][o]
            only = m["scored"] & ~m["nose"] & m["whisker"]
            near[o].append(m["dist"][only])
            hit[o].append(m["hit"][only])
    near = {o: np.concatenate(v) for o, v in near.items()}
    hit = {o: np.concatenate(v) for o, v in hit.items()}
    if sum(len(v) for v in near.values()) == 0:
        raise Needs("needs cohort", "No whisker-only scored frames in the NORT test trials of this run.")
    reach = float(np.median([t.param("exploration.whisker.length_m", math.nan) for t in tests]))
    zone = float(np.median([t.param("exploration.nose_max_distance", NOSE_ZONE_M) for t in tests]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    top = 1000 * max(zone * 1.15, max((float(np.nanmax(v)) for v in near.values() if len(v)), default=0.0) * 1.05)
    for ax, data, xlabel, xmax, title in (
        (axes[0], near, "nose to nearest object perimeter (mm)", top, "Where whisker-only frames sit"),
        (
            axes[1],
            hit,
            "fan-hit distance (mm)",
            1000 * max(reach * 1.2, 0.001) if reach == reach else None,
            "Distance along the fan to the hit",
        ),
    ):
        vals = [data[o][np.isfinite(data[o])] * 1000 for o in OBJECTS]
        ax.hist(
            vals,
            bins=np.linspace(0, xmax or max((v.max() for v in vals if len(v)), default=1.0), 31),
            stacked=True,
            color=pal["categorical"][:2],
            label=[f"{o} (n = {len(v)} frames)" for o, v in zip(OBJECTS, vals, strict=True)],
        )
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("whisker-only frames", fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        if reach == reach:
            ax.axvline(reach * 1000, color=pal["highlight"], lw=1.6, label=f"configured reach {reach * 1000:.1f} mm")
        _style(ax)
    axes[0].axvline(zone * 1000, color=pal["negative"], ls="--", lw=1.4, label=f"nose zone {zone * 1000:.0f} mm")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False, labelcolor=pal["ink"])
    return _finish(fig, pal, "Whisker reach: how close to the edge are whisker-only hits?")


def nort_time_course(trials: list[Trial], groups: dict[str, str], pal: dict[str, Any]) -> Figure:
    """Cumulative DI and cumulative exploration over the test, one line per trial."""
    tests = _nort_tests(trials)
    times = [x for x in ("pre", "post") if any(t.time == x for t in tests)] or [None]
    style = _group_style(tests, groups, pal)
    fig, axes = plt.subplots(2, len(times), figsize=(6.5 * len(times) + 1, 7), squeeze=False, sharex="col")
    for j, tp in enumerate(times):
        sel = [t for t in tests if t.time == tp]
        for t in sel:
            f = _frames(t)
            n, fam = (np.cumsum(f["o"][o]["scored"]) * f["dt"] for o in ("novel", "familiar"))
            tot = n + fam
            # ponytail: DI is shown from 2 s of cumulative exploration; earlier values are +-1 noise.
            di = np.where(tot >= MIN_CUM_S, (n - fam) / np.where(tot > 0, tot, 1), np.nan)
            step = max(1, len(tot) // 800)
            x = f["t"][::step] / 60
            colour = style[t.group][0]
            axes[0, j].plot(x, di[::step], color=colour, lw=1.2, alpha=0.9)
            axes[1, j].plot(x, tot[::step], color=colour, lw=1.2, alpha=0.9)
        axes[0, j].axhline(0, color=pal["reference"], ls="--", lw=1)
        axes[0, j].set_ylim(-1, 1)
        axes[0, j].set_title(
            f"{tp or 'test'} · cumulative DI from {MIN_CUM_S:g} s of exploration (n = {len(sel)} trials)",
            fontsize=10,
            loc="left",
        )
        axes[1, j].set_title("cumulative scored exploration", fontsize=10, loc="left")
        axes[1, j].set_xlabel("test time (min)", fontsize=9)
        axes[0, j].set_ylabel("cumulative DI", fontsize=9)
        axes[1, j].set_ylabel("exploration (s)", fontsize=9)
    for a in axes.flat:
        _style(a)
    _legend(fig, style, {t.group for t in tests}, pal)
    return _finish(fig, pal, "Time course of novelty preference within the test", legend=True)


CONCORDANCE = "ymaze_threshold_concordance"


def _threshold_owners() -> list[dict[str, Any]]:
    """Descriptors that own the threshold-concordance figure and define scoring contexts."""
    return [d for d in descriptors() if CONCORDANCE in (d.get("figures") or []) and section(d, "contexts")]


def _ymaze_rows(trials: list[Trial]) -> tuple[dict[tuple[str, str | None], dict[str, dict[str, float]]], list[str]]:
    """({(animal, time): {context label: {sa, entries, triads, low}}}, labels in context order), non-comparison contexts."""
    owners = {d["id"] for d in _threshold_owners()}
    out: dict[tuple[str, str | None], dict[str, dict[str, float]]] = {}
    rank: dict[str, tuple[int, str]] = {}
    names: dict[str, str] = {}
    for t in trials:
        contexts, variant = section(t.descriptor, "contexts"), variant_of(t.role)
        if t.descriptor.get("id") not in owners or not variant or is_comparison(contexts, variant) or not t.animal:
            continue
        rank[variant] = context_order_key(contexts, variant)
        names[variant] = context_label(contexts, variant)
        a = t.detailed.get("analysis") or {}
        entries = len(a.get("arm_sequence") or [])
        triads = (a.get("correct_alternations") or 0) + (a.get("incorrect_alternations") or 0)
        sa = a.get("spontaneous_alternation_pct")
        out.setdefault((t.animal, t.time), {})[variant] = {
            "sa": math.nan if sa is None else float(sa),
            "entries": entries,
            "low": entries < LOW_ENTRIES or triads < LOW_TRIADS,
        }
    # Keyed by variant so arm25 and archived proximal25 stay distinct; equal labels get the variant appended.
    shown = {v: f"{n} [{v}]" if list(names.values()).count(n) > 1 else n for v, n in names.items()}
    out = {k: {shown[v]: c for v, c in ctxs.items()} for k, ctxs in out.items()}
    return out, [shown[v] for v in sorted(rank, key=rank.__getitem__)]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return math.nan
    rx, ry = pl.Series(x[ok]).rank().to_numpy(), pl.Series(y[ok]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else math.nan


def ymaze_threshold_concordance(trials: list[Trial], groups: dict[str, str], pal: dict[str, Any]) -> Figure:
    """Per-animal primary SA% and valid entries across the arm-start thresholds."""
    owners = _threshold_owners()
    ids, short = {d["id"] for d in owners}, ", ".join(d["short_name"] for d in owners)
    mine = [t for t in trials if t.descriptor.get("id") in ids]
    if not mine:
        raise Needs("needs cohort", f"The run has no {short} trials.")
    if not any(t.animal for t in mine):
        raise Needs(
            "needs animal metadata",
            f"{short} trials record no animal_id; animals are never inferred from session metadata or trial names.",
        )
    rows, labels = _ymaze_rows(trials)
    rows = {k: v for k, v in rows.items() if len(v) >= 2}
    if not rows:
        raise Needs("needs sensitivity runs", "Fewer than two principal arm-start thresholds exist for any animal.")
    group_of = {t.animal: t.group for t in trials if t.animal}
    style = _group_style([t for t in trials if t.animal in group_of], groups, pal)
    times = [x for x in ("pre", "post") if any(k[1] == x for k in rows)] or [None]
    fig, axes = plt.subplots(2, len(times), figsize=(6.5 * len(times) + 1, 7.5), squeeze=False, sharex="col")
    for j, tp in enumerate(times):
        sel = {k: v for k, v in rows.items() if k[1] == tp}
        for (animal, _), ctxs in sel.items():
            colour = style[group_of.get(animal)][0]
            xs = [i for i, lab in enumerate(labels) if lab in ctxs]
            for r, field_ in enumerate(("sa", "entries")):
                ys = [ctxs[labels[i]][field_] for i in xs]
                axes[r, j].plot(xs, ys, color=colour, lw=1.1, alpha=0.8)
                for i, y in zip(xs, ys, strict=True):
                    low = ctxs[labels[i]]["low"]
                    axes[r, j].scatter(
                        [i],
                        [y],
                        s=40,
                        marker="X" if low else "o",
                        zorder=3,
                        facecolors=pal["background"] if low else colour,
                        edgecolors=colour,
                        linewidths=1.2,
                    )
        axes[1, j].axhline(LOW_ENTRIES, color=pal["negative"], ls="--", lw=1.2)
        axes[1, j].text(
            len(labels) - 0.55,
            LOW_ENTRIES,
            " 8 valid entries",
            va="bottom",
            ha="right",
            fontsize=8,
            color=pal["negative"],
        )
        matrix = np.array([[v.get(lab, {}).get("sa", math.nan) for lab in labels] for v in sel.values()])
        rho = [_spearman(matrix[:, a], matrix[:, b]) for a in range(len(labels)) for b in range(a + 1, len(labels))]
        rho = [r for r in rho if r == r]
        note = (
            f"Spearman ρ between thresholds: median {np.median(rho):.2f}, min {min(rho):.2f}"
            if rho
            else "rank agreement needs ≥ 3 animals with varying SA"
        )
        axes[0, j].set_title(f"{tp or 'session'} · SA% (n = {len(sel)} animals) · {note}", fontsize=9, loc="left")
        axes[1, j].set_title("valid arm entries", fontsize=10, loc="left")
        axes[0, j].set_ylabel("primary SA (%)", fontsize=9)
        axes[1, j].set_ylabel("valid_total_arm_entries", fontsize=9)
        axes[1, j].set_xticks(range(len(labels)), labels)
        axes[1, j].set_xlabel("arm-start threshold", fontsize=9)
    for a in axes.flat:
        _style(a)
    if any(c["low"] for v in rows.values() for c in v.values()):
        axes[0, 0].legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="X",
                    color=pal["muted"],
                    ls="",
                    mfc=pal["background"],
                    label="low-quality trial (marked, not excluded)",
                )
            ],
            fontsize=8,
            frameon=False,
            labelcolor=pal["ink"],
            loc="lower right",
        )
    for ax in axes[0]:
        ax.set_ylim(0, 100)
    _legend(fig, style, {group_of.get(k[0]) for k in rows}, pal)
    return _finish(fig, pal, "Y-maze alternation across arm-start thresholds", legend=True)


def _trajectory_series(trials: list[Trial]) -> list[tuple[str, dict[str, Any], dict[str, Any], str | None]]:
    """[(series name, descriptor, endpoint, principal variant)] for every endpoint listing pre_post_trajectories."""
    out = []
    for d in descriptors():
        if d["id"] == "generic":
            continue
        contexts = section(d, "contexts")
        variants = {v for t in trials if t.descriptor["id"] == d["id"] if (v := variant_of(t.role))}
        principal = next((v for v in contexts.get("principal") or [] if v in variants), None)
        if principal is None and contexts:
            principal = min(
                (v for v in variants if not is_comparison(contexts, v)),
                key=lambda v: context_order_key(contexts, v),
                default=None,
            )
        for e in d.get("endpoints") or []:
            if "pre_post_trajectories" not in (e.get("figures") or []):
                continue
            step = next((s["label"] for s in d.get("steps") or [] if s["id"] == e.get("step")), None)
            suffix = step or (context_label(contexts, principal) if contexts else None)
            name = f"{d['short_name']} {e.get('label') or e['name']}" + (f" ({suffix})" if suffix else "")
            out.append((name, d, e, principal))
    return out


def _dotted(data: dict[str, Any], path: str) -> Any:
    for key in path.split("."):
        data = data.get(key) if isinstance(data, dict) else None
    return data


def pre_post_trajectories(trials: list[Trial], groups: dict[str, str], pal: dict[str, Any]) -> Figure:
    """Pre and post values per animal for every endpoint a descriptor lists for this figure, coloured by group."""
    if not any(t.animal for t in trials):
        raise Needs(
            "needs animal metadata",
            "No trial records an animal_id; pre/post pairs are never inferred from session metadata or trial names.",
        )
    spec = _trajectory_series(trials)
    series: dict[str, dict[tuple[str, str], list[float]]] = {name: {} for name, *_ in spec}
    for t in trials:
        if not t.animal or t.time not in ("pre", "post"):
            continue
        variant = variant_of(t.role)
        for name, d, e, principal in spec:
            if t.descriptor["id"] != d["id"] or (e.get("step") and t.step != e["step"]):
                continue
            if section(d, "contexts") and (principal is None or variant != principal):
                continue
            v = _dotted(t.detailed, e["detail"])
            if v is not None:
                series[name].setdefault((t.animal, t.time), []).append(float(v))  # repeated sessions average
    pairs = {n: {a for (a, tp) in s if tp == "pre"} & {a for (a, tp) in s if tp == "post"} for n, s in series.items()}
    if not any(pairs.values()):
        names = ", ".join(dict.fromkeys(d["short_name"] for _, d, _, _ in spec))
        raise Needs("needs cohort", f"No animal has both a pre and a post session for {names} in this run.")
    group_of = {t.animal: t.group for t in trials if t.animal}
    style = _group_style([t for t in trials if t.animal], groups, pal)
    fig, axes = plt.subplots(1, len(series), figsize=(14 * len(series) / 3, 5.2), squeeze=False)
    for ax, (name, s) in zip(axes[0], series.items(), strict=True):
        animals = sorted({a for a, _ in s})
        if not animals:
            _empty(ax, "no recorded values", pal)
        for a in animals:
            pts = [(x, float(np.mean(s[(a, tp)]))) for x, tp in enumerate(("pre", "post")) if (a, tp) in s]
            ax.plot(*zip(*pts, strict=True), color=style[group_of.get(a)][0], marker="o", lw=1.1, alpha=0.85)
        for g in sorted({group_of.get(a) for a in pairs[name]}, key=str):
            both = [a for a in pairs[name] if group_of.get(a) == g]
            means = [np.mean([np.mean(s[(a, tp)]) for a in both]) for tp in ("pre", "post")]
            ax.plot([0, 1], means, color=style[g][0], lw=3.2, alpha=0.5, zorder=1)
        if animals:
            ax.set_xticks([0, 1], ["pre", "post"])
            ax.set_xlim(-0.3, 1.3)
            ax.set_title(f"{name} · {len(pairs[name])} paired of {len(animals)} animals", fontsize=10, loc="left")
            _style(ax)
    _legend(fig, style, {group_of.get(a) for p in pairs.values() for a in p}, pal)
    return _finish(
        fig, pal, "Individual pre→post trajectories (thick line = mean of paired animals per group)", legend=True
    )


FIGURES: dict[str, tuple[str, Callable[..., Figure]]] = {
    "whisker_contribution": ("Whisker fan versus nose", whisker_contribution),
    "whisker_reach": ("Whisker reach", whisker_reach),
    "nort_time_course": ("Time course of novelty preference", nort_time_course),
    CONCORDANCE: ("Y-maze alternation across arm-start thresholds", ymaze_threshold_concordance),
    "pre_post_trajectories": ("Individual pre→post trajectories", pre_post_trajectories),
}


def _figure_ids(trials: list[Trial]) -> list[str]:
    """Figure ids of every experiment: their own, then those of their endpoints.

    A registered renderer is always listed (``needs cohort`` when its experiment is absent);
    an id without one only when its experiment is in the run.
    """
    own: list[str] = []
    endpoints: list[str] = []
    for d in descriptors():
        here = any(t.descriptor["id"] == d["id"] for t in trials)
        own += [f for f in d.get("figures") or [] if here or f in FIGURES]
        endpoints += [f for e in d.get("endpoints") or [] for f in e.get("figures") or [] if here or f in FIGURES]
    ids = own + endpoints
    return list(dict.fromkeys(ids))


def write_extension_figures(
    run: Path, output: Path, groups: dict[str, str] | None = None, mode: str | None = None
) -> dict[str, Any]:
    """Render every figure it can into ``output`` and write ``index.json`` (descriptor order)."""
    resolved = theme_resolve(mode, None)
    pal = dict(theme_palette(resolved))
    output.mkdir(parents=True, exist_ok=True)
    trials = load_trials(run)
    entries = []
    with plt.rc_context(dict(theme_rc(resolved))):
        for fid in _figure_ids(trials):
            title, fn = FIGURES.get(fid, (fid, None))
            entry: dict[str, Any] = {
                "id": fid,
                "title": title,
                "status": "ok",
                "reason": "",
                "file": None,
                "width": None,
                "height": None,
            }
            try:
                if fn is None:
                    raise Needs("needs renderer", f"No renderer is registered for figure {fid}.")
                fig = fn(trials, groups or {}, pal)
                path = output / f"{fid}.png"
                fig.savefig(path, dpi=DPI)
                with Image.open(path) as image:
                    entry.update(file=path.name, width=image.width, height=image.height)
            except Needs as need:
                entry.update(status=need.status, reason=need.reason)
            except Exception as exc:  # one broken figure must not hide the others
                entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            finally:
                plt.close("all")
            entries.append(entry)
    index = {"color_mode": resolved, "figures": entries}
    (output / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    return index
