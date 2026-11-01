"""Individual-animal panels with exact plotting data and registered uncertainty."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .inputs import KEY, choice, subgroup_plan, validate_sensitivity_thresholds
from .metrics import OFT_PRIMARY, Record
from .statistics import pairing, verification_reason


def draw_panel(
    ax: Any, data: list[Record], *, label: str, reference: float | None, uncertainty: str | None
) -> list[Record]:
    ax.set_title(label)
    ax.set_ylabel(label)
    if reference is not None:
        ax.axhline(reference, linestyle="--", color="0.5")
    categories = sorted({row["category"] for row in data})
    summaries = []
    for x, category in enumerate(categories):
        group = sorted([r for r in data if r["category"] == category], key=lambda r: r["animal_id"])
        if len({r["animal_id"] for r in group}) != len(group):
            raise ValueError("Plot category contains repeated animal replicates")
        values = [row["value"] for row in group]
        offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0]
        ax.scatter([x + d for d in offsets], values, s=16, alpha=0.8)
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
        error = sem
        if uncertainty == "95% CI" and sem is not None:
            error = sem * float(stats.t.ppf(0.975, len(values) - 1))
        if uncertainty in ["SEM", "95% CI"] and error is not None:
            ax.errorbar(x, mean, yerr=error, fmt="_", color="black", capsize=3)
        else:
            ax.plot(x, mean, "_", color="black")
        summaries.append(
            {
                "category": category,
                "n_animals": len(values),
                "mean": mean,
                "sem": sem,
                "uncertainty": uncertainty,
                "error_half_width": error if uncertainty else None,
                "status": "available" if uncertainty else "blocked:D09:summary_uncertainty",
            }
        )
    # Explicit pairing field connects only known same-animal observations.
    series = {r["plot_pair_id"] for r in data if r.get("pair_series") and r.get("plot_pair_id")}
    for identity in sorted(series):
        paired = [r for r in data if r.get("plot_pair_id") == identity and r.get("pair_series")]
        if len(paired) == 2:
            paired.sort(key=lambda r: r.get("pair_order", 0))
            ax.plot(
                [categories.index(r["category"]) for r in paired],
                [r["value"] for r in paired],
                color="0.6",
                alpha=0.4,
                linewidth=0.7,
            )
    ax.set_xticks(
        range(len(categories)),
        [f"{category}\nn={sum(r['category'] == category for r in data)}" for category in categories],
        rotation=35,
        ha="right",
    )
    if not data:
        ax.text(0.5, 0.5, "Unavailable / blocked\nsee inclusion ledger", transform=ax.transAxes, ha="center")
    return summaries


def make_figures(
    rows: list[Record], config: Record, directory: Path
) -> tuple[list[Record], list[Record], list[Record]]:
    requirement = choice(config, "inclusion.upstream_verification_requirement", "D12")
    population_policy = requirement or "descriptive_metadata_nonexcluded"
    rows = [
        {**row, "plot_eligibility_reason": verification_reason(row, requirement) if requirement is not None else None}
        for row in rows
    ]
    uncertainty = choice(config, "statistics.summary_uncertainty", "D09")
    if uncertainty not in [None, "SEM", "95% CI"]:
        raise ValueError("Summary uncertainty must be registered SEM or 95% CI")
    formats = config["outputs"].get("figure_formats") or ["svg"]
    if not set(formats) <= {"svg", "png", "pdf"}:
        raise ValueError("Unsupported figure format")
    figures: list[Record] = []
    exported: list[Record] = []
    summaries: list[Record] = []
    for assay in ["oft", "ymaze", "nort"]:
        subset = [r for r in rows if r["assay"] == assay]
        if not subset:
            continue
        post = [r for r in subset if r["timepoint"] == "post"]
        specs: list[tuple[str, list[Record], str, float | None]] = []
        if assay == "oft":
            specs = [
                (field, post, label, None)
                for field, label in [
                    ("total_distance_m", "Total distance (m)"),
                    ("mean_speed_m_per_s", "Mean speed (m/s)"),
                    ("center_time_pct", "Center time (%) / 300 s"),
                    ("center_entries", "Center entries"),
                    ("latency_to_first_center_entry_s", "Latency to center (s)"),
                ]
            ]
        elif assay == "ymaze":
            primary = [r for r in subset if r.get("primary_threshold") is True]
            post = [r for r in primary if r["timepoint"] == "post"]
            paired, _ = pairing(
                primary, "spontaneous_alternation_pct", verification_requirement=requirement or "allow_unverified"
            )
            paired_points = []
            for pair in paired:
                pre = next(
                    r
                    for r in primary
                    if r["animal_id"] == pair["animal_id"]
                    and r["session_id"] == pair["pre_session_id"]
                    and r["timepoint"] == "pre"
                )
                identity = json.dumps([pair["animal_id"], pair.get("pair_id"), pre["session_id"], pair["session_id"]])
                paired_points.extend([{**r, "plot_pair": True, "plot_pair_id": identity} for r in [pre, pair]])
            specs = [
                ("spontaneous_alternation_pct", post, "Post-treatment SA (%) — primary", None),
                ("DeltaSA", paired, "DeltaSA (post − pre; secondary)", None),
                ("spontaneous_alternation_pct", paired_points, "Paired pre/post SA (%)", None),
            ]
            specs += [
                (field, post, label, None)
                for field, label in [
                    ("valid_total_arm_entries", "Valid arm entries"),
                    ("total_distance_m", "Distance (m)"),
                    ("mean_speed_m_per_s", "Mean speed (m/s)"),
                    ("neutral_triads_pct", "Neutral triads (%)"),
                ]
            ]
        else:
            training = [r for r in post if r["phase"] == "training"]
            test = [r for r in post if r["phase"] == "test"]
            training_pairs = [
                {
                    **r,
                    "exploration_s": r.get(field),
                    "plot_object": role,
                    "plot_pair": True,
                    "plot_pair_id": json.dumps([r[k] for k in KEY]),
                    "pair_order": order,
                }
                for r in training
                for order, (field, role) in enumerate([("TA", "A"), ("TB", "B")])
            ]
            test_pairs = [
                {
                    **r,
                    "exploration_s": r.get(field),
                    "plot_object": role,
                    "plot_pair": True,
                    "plot_pair_id": json.dumps([r[k] for k in KEY]),
                    "pair_order": order,
                }
                for r in test
                for order, (field, role) in enumerate([("TF", "familiar"), ("TN", "novel")])
            ]
            specs = [
                ("exploration_s", training_pairs, "Training A vs B exploration (s)", None),
                ("exploration_s", test_pairs, "Test familiar vs novel exploration (s)", None),
                ("DI", test, "DI (primary)", 0),
                ("percent_novel", test, "Percent novel (%)", 50),
                ("total_test_exploration", test, "Total test exploration (s)", None),
            ]
        figures.extend(render_specs(f"figure_{assay}", specs, directory, formats, uncertainty, exported, summaries))
        supplement: list[tuple[str, list[Record], str, float | None]] = []
        if assay == "ymaze":
            thresholds = config["assays"]["ymaze"].get("requested_sensitivity_thresholds_arm_length_pct", [25, 45, 65])
            validate_sensitivity_thresholds(thresholds)
            for threshold in thresholds:
                threshold_rows = [r for r in subset if r.get("threshold_pct") == threshold and r["timepoint"] == "post"]
                supplement.append(
                    (
                        "spontaneous_alternation_pct",
                        threshold_rows,
                        f"Sensitivity SA — {threshold}% imen radial vertex span",
                        None,
                    )
                )
                supplement.append(
                    (
                        "spontaneous_alternation_pct_standard",
                        threshold_rows,
                        f"Standard SA — {threshold}% (secondary)",
                        None,
                    )
                )
        if assay == "nort":
            supplement = [
                ("object_bias", [r for r in post if r["phase"] == "training"], "Object bias", 0),
                (
                    "side_bias",
                    [r for r in post if r["phase"] == "training"],
                    "Source algebraic side_bias (not spatial)",
                    0,
                ),
            ]
            for field in ["valid_bouts", "removed_duration_bouts", "removed_speed_bouts", "removed_angle_bouts"]:
                supplement.append((field, [r for r in post if r["phase"] == "test"], field.replace("_", " "), None))
        if supplement:
            figures.extend(
                render_specs(f"supplement_{assay}", supplement, directory, formats, uncertainty, exported, summaries)
            )
    plan = subgroup_plan(config)
    for index, stratum in enumerate(plan["strata"] if plan else []):
        for assay in stratum["assays"]:
            subgroup_specs: list[tuple[str, list[Record], str, float | None]] = []
            for level in stratum["levels"]:
                members = [
                    r
                    for r in rows
                    if r["assay"] == assay
                    and r["timepoint"] == "post"
                    and r.get(stratum["field"]) == level
                    and (assay != "nort" or r["phase"] == "test")
                    and (assay != "ymaze" or r.get("primary_threshold") is True)
                ]
                fields = (
                    OFT_PRIMARY if assay == "oft" else ["DI"] if assay == "nort" else ["spontaneous_alternation_pct"]
                )
                subgroup_specs.extend(
                    (field, members, f"{field} — {stratum['field']}={level}", 0 if field == "DI" else None)
                    for field in fields
                )
            figures.extend(
                render_specs(
                    f"subgroup_{assay}_{stratum['field']}_{index}",
                    subgroup_specs,
                    directory,
                    formats,
                    uncertainty,
                    exported,
                    summaries,
                )
            )
    # QC scatter panels are pairwise animal data, not significance annotations.
    for assay, xfield, yfields in [
        (
            "ymaze",
            "spontaneous_alternation_pct",
            ["valid_total_arm_entries", "total_distance_m", "mean_speed_m_per_s", "neutral_triads_pct"],
        ),
        ("nort", "total_test_exploration", ["DI"]),
    ]:
        source = [
            r
            for r in rows
            if r["assay"] == assay
            and r["timepoint"] == "post"
            and not r["excluded"]
            and not r["plot_eligibility_reason"]
            and (assay != "ymaze" or r.get("primary_threshold") is True)
            and (assay != "nort" or r["phase"] == "test")
        ]
        if not source:
            continue
        fig, axes = plt.subplots(1, len(yfields), figsize=(4 * len(yfields), 3.5), squeeze=False)
        for ax, yfield in zip(axes.flat, yfields, strict=True):
            data = [r for r in source if r.get(xfield) is not None and r.get(yfield) is not None]
            ax.scatter([r[xfield] for r in data], [r[yfield] for r in data], s=16)
            ax.set(xlabel=xfield, ylabel=yfield, title=f"n={len(data)} animals")
            exported.extend(
                [
                    {
                        "figure": f"qc_{assay}",
                        "panel": yfield,
                        "animal_id": r["animal_id"],
                        "session_id": r["session_id"],
                        "x": r[xfield],
                        "value": r[yfield],
                        "input_verification_status": r["input_verification"]["status"],
                    }
                    for r in data
                ]
            )
        fig.tight_layout()
        save_figure(fig, directory, f"qc_{assay}", formats)
        plt.close(fig)
        figures.append({"name": f"qc_{assay}", "status": "rendered", "reference_lines": []})
    for figure in figures:
        figure["population_policy"] = population_policy
    for point in exported:
        point["population_policy"] = population_policy
    return figures, exported, summaries


def render_specs(
    name: str,
    specs: list[tuple[str, list[Record], str, float | None]],
    directory: Path,
    formats: list[str],
    uncertainty: str | None,
    exported: list[Record],
    summaries: list[Record],
) -> list[Record]:
    fig, axes = plt.subplots(
        math.ceil(len(specs) / 3), min(3, len(specs)), figsize=(12, 3.8 * math.ceil(len(specs) / 3)), squeeze=False
    )
    populated = 0
    panels = []
    for ax, (field, source, label, reference) in zip(axes.flat, specs):
        data = []
        for row in source:
            if row.get(field) is None or row["excluded"] or row.get("plot_eligibility_reason"):
                continue
            category = row["group"]
            if row.get("plot_object"):
                category += ":" + row["plot_object"]
            elif row.get("plot_pair"):
                category += ":" + row["timepoint"]
            data.append(
                {
                    "animal_id": row["animal_id"],
                    "session_id": row["session_id"],
                    "category": category,
                    "sex": row.get("sex"),
                    "age_treatment_window": row.get("age_treatment_window"),
                    "five_xfad": row.get("five_xfad"),
                    "hcar1": row.get("hcar1"),
                    "treatment": row.get("treatment"),
                    "value": row[field],
                    "timepoint": row["timepoint"],
                    "phase": row["phase"],
                    "pre_session_id": row.get("pre_session_id"),
                    "input_verification_status": row["input_verification"]["status"],
                    "threshold_pct": row.get("threshold_pct"),
                    "pair_series": row.get("plot_pair", False),
                    "pair_id": row.get("pair_id"),
                    "plot_pair_id": row.get("plot_pair_id"),
                    "pair_order": row.get("pair_order", 0 if row["timepoint"] == "pre" else 1),
                }
            )
        exported.extend(
            [{**r, "figure": name, "panel": field, "label": label, "reference_line": reference} for r in data]
        )
        summaries.extend(
            [
                {**summary, "figure": name, "panel": field}
                for summary in draw_panel(ax, data, label=label, reference=reference, uncertainty=uncertainty)
            ]
        )
        panels.append(
            {
                "endpoint": field,
                "label": label,
                "reference_line": reference,
                "status": "rendered" if data else "unavailable_or_blocked",
            }
        )
        populated += bool(data)
    for ax in list(axes.flat)[len(specs) :]:
        ax.set_visible(False)
    fig.tight_layout()
    save_figure(fig, directory, name, formats)
    plt.close(fig)
    return [
        {
            "name": name,
            "status": "rendered" if populated == len(specs) and uncertainty else "partial",
            "panels": panels,
            "summary_uncertainty": uncertainty,
        }
    ]


def save_figure(fig: Any, directory: Path, name: str, formats: list[str]) -> None:
    for extension in formats:
        fig.savefig(directory / f"{name}.{extension}", dpi=150)
