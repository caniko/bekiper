"""Exploratory research-extension figures on synthetic run outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from bekiper_rs import theme_palette
from matplotlib import pyplot as plt

from bekiper_inspect.plot.extensions import (
    FIGURES,
    Trial,
    _ymaze_rows,
    load_trials,
    pre_post_trajectories,
    write_extension_figures,
    ymaze_threshold_concordance,
)

IDS = [
    "whisker_contribution",
    "whisker_reach",
    "nort_time_course",
    "ymaze_threshold_concordance",
    "pre_post_trajectories",
]
N = 900  # 30 s at 30 fps


def _nort(
    root: Path,
    name: str,
    animal: str | None,
    group: str,
    phase_part: str,
    whisker: bool = True,
    ids: bool = True,
    stage: str = "2",
) -> None:
    d = root / "data" / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(len(name) + int(animal or 0))
    cols: dict[str, np.ndarray] = {
        "elapsed_s": np.arange(N) / 30.0,
        "nose_x": rng.uniform(0, 0.4, N),
        "nose_y": rng.uniform(0, 0.4, N),
    }
    for obj, rate in (("familiar", 0.15), ("novel", 0.3)):
        p = f"__bk_diag__exploring_{obj}__"
        near = rng.random(N) < rate
        cols[p + "final_toleranced"] = near | (rng.random(N) < 0.05)
        cols[p + "outside_object"] = np.ones(N, bool)
        cols[p + "nose_near"] = near & (rng.random(N) < 0.7)
        cols[p + "contact_or_angle"] = np.ones(N, bool)
        cols[p + "within_angle_target_x"] = cols["nose_x"] + rng.uniform(0, 0.012, N)
        cols[p + "within_angle_target_y"] = cols["nose_y"]
        if whisker:
            cols[p + "whisker_candidate"] = near
            cols[p + "whisker_nearest_hit_distance_m"] = np.where(near, rng.uniform(0.002, 0.0117, N), np.nan)
    pl.DataFrame(cols).write_parquet(d / f"{name}_evaluated.parquet")
    novel, fam = (
        cols["__bk_diag__exploring_novel__final_toleranced"].sum(),
        cols["__bk_diag__exploring_familiar__final_toleranced"].sum(),
    )
    (d / "result.toml").write_text(
        f'task_type = "object_recognition"\n[session_context]\nanimal = "{animal}"\ngroup = "{group}"\nstage = "{stage}"\n'
        f'phase_part = "{phase_part}"\n[detailed.indices]\ndiscrimination_index = {(novel - fam) / (novel + fam)}\n'
        '[detailed.effective_parameters.values]\n"exploration.whisker.length_m" = 0.0117\n"exploration.nose_max_distance" = 0.03\n'
    )
    (d / "trial_context.json").write_text(json.dumps({"animal_id": animal if ids else None}))


def _ymaze(root: Path, animal: str, label: str, seq: str, phase_part: str = "2") -> None:
    d = root / "data" / f"y_{animal}_{label}_{phase_part}"
    d.mkdir(parents=True)
    arms = list(seq)
    d.joinpath("result.toml").write_text(
        f'task_type = "y_maze"\nanalysis_role = "ymaze__{label}"\n[session_context]\nanimal = "{animal}"\nphase_part = "{phase_part}"\n'
        f"[detailed.analysis]\nspontaneous_alternation_pct = {40 + len(seq) % 7}\ncorrect_alternations = {len(seq) // 4}\n"
        f"incorrect_alternations = {len(seq) // 3}\narm_sequence = {json.dumps(arms)}\n"
    )
    d.joinpath("trial_context.json").write_text(json.dumps({"animal_id": animal}))


def _oft(root: Path, animal: str, group: str, phase_part: str, pct: float) -> None:
    d = root / "data" / f"oft_{animal}_{phase_part}"
    d.mkdir(parents=True)
    d.joinpath("result.toml").write_text(
        f'task_type = "open_field"\n[session_context]\nanimal = "{animal}"\ngroup = "{group}"\nphase_part = "{phase_part}"\n'
        f"[detailed]\ncenter_time_pct = {pct}\n"
    )
    d.joinpath("trial_context.json").write_text(json.dumps({"animal_id": animal}))


def _cohort(root: Path) -> None:
    for i, animal in enumerate(("1", "2", "3", "4")):
        group = "5" if i % 2 else "6"
        for part, tag in (("1", "pre"), ("2", "post")):
            _nort(root, f"nort_{tag}_t2_{animal}", animal, group, part)
            _oft(root, animal, group, part, 20.0 + 3 * i + int(part))
        for label, seq in (
            ("arm0_dwell1s", "ABCABCABCAB"),
            ("arm25", "ABCABCABCABC" + "AB" * i),
            ("arm45", "ABCACBABCAB"),
            ("arm65", "ABCABCB"),
        ):
            _ymaze(root, animal, label, seq)
            _ymaze(root, animal, label, seq[:-1], phase_part="1")


def _status(index: dict) -> dict[str, dict]:
    return {f["id"]: f for f in index["figures"]}


def test_cohort_renders_every_figure_and_writes_index(tmp_path: Path) -> None:
    _cohort(tmp_path / "run")
    out = tmp_path / "out"
    index = write_extension_figures(tmp_path / "run", out, {"5": "G5 · lactate"}, "light")
    assert json.loads((out / "index.json").read_text()) == index
    assert index["color_mode"] == "light"
    assert [f["id"] for f in index["figures"]] == IDS == list(FIGURES)
    for figure in index["figures"]:
        assert figure["status"] == "ok", figure
        assert (out / figure["file"]).exists() and figure["width"] > 400 and figure["height"] > 300


def test_missing_pairs_and_columns_degrade_with_a_reason(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _nort(run, "nort_post_t2_1", "1", "6", "2", whisker=False)  # one animal, no pre, no whisker columns
    _ymaze(run, "1", "arm45", "ABCABC")  # a single threshold
    index = write_extension_figures(run, tmp_path / "out")
    got = _status(index)
    assert got["whisker_contribution"]["status"] == "needs whisker scoring"
    assert got["whisker_reach"]["status"] == "needs whisker scoring"
    assert got["ymaze_threshold_concordance"]["status"] == "needs sensitivity runs"
    assert got["pre_post_trajectories"]["status"] == "needs cohort"
    assert got["nort_time_course"]["status"] == "needs whisker scoring"
    assert all(f["reason"] and f["file"] is None for f in index["figures"] if f["status"] != "ok")
    assert not list((tmp_path / "out").glob("*.png"))


def test_missing_animal_metadata_and_empty_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    d = run / "data" / "y_x_arm45"
    d.mkdir(parents=True)
    (d / "result.toml").write_text('task_type = "y_maze"\nanalysis_role = "ymaze__arm45"\n')
    got = _status(write_extension_figures(run, tmp_path / "out"))
    assert got["ymaze_threshold_concordance"]["status"] == "needs animal metadata"
    assert got["pre_post_trajectories"]["status"] == "needs animal metadata"
    empty = _status(write_extension_figures(tmp_path / "none", tmp_path / "out2"))
    assert all(f["status"] != "ok" and f["reason"] for f in empty.values())


def test_null_animal_id_is_never_inferred(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for part, tag in (("1", "pre"), ("2", "post")):  # session metadata names animal 7, trial_context does not
        _nort(run, f"nort_{tag}_t2_7", "7", "6", part, ids=False)
    got = _status(write_extension_figures(run, tmp_path / "out"))
    assert got["pre_post_trajectories"]["status"] == "needs animal metadata"
    assert "animal_id" in got["pre_post_trajectories"]["reason"]
    assert got["whisker_contribution"]["status"] == "ok"


def test_pre_post_trajectories_png_is_deterministic(tmp_path: Path) -> None:
    _cohort(tmp_path / "run")
    first = write_extension_figures(tmp_path / "run", tmp_path / "a", {"5": "G5"}, "light")
    write_extension_figures(tmp_path / "run", tmp_path / "b", {"5": "G5"}, "light")
    name = _status(first)["pre_post_trajectories"]["file"]
    assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def _figure(fn, run: Path):
    return fn(load_trials(run), {}, dict(theme_palette("light")))


def test_cohort_ids_titles_and_ticks_come_from_the_descriptors(tmp_path: Path) -> None:
    _cohort(tmp_path / "run")
    index = write_extension_figures(tmp_path / "run", tmp_path / "out")
    assert [f["id"] for f in index["figures"]] == IDS
    assert {f["status"] for f in index["figures"]} == {"ok"}
    fig = _figure(ymaze_threshold_concordance, tmp_path / "run")
    assert [t.get_text() for t in fig.axes[-1].get_xticklabels()] == ["0% (1 s dwell)", "25%", "45%", "65%"]
    fig = _figure(pre_post_trajectories, tmp_path / "run")
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    plt.close("all")
    assert [t.split(" · ")[0] for t in titles] == ["OFT centre time %", "NORT DI (T2 test)", "Y-maze SA% (45%)"]


def test_t2_in_the_name_is_not_a_nort_test_when_the_stage_says_otherwise(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _nort(run, "nort_post_t2_1", "1", "6", "2", stage="1")
    (trial,) = load_trials(run)
    assert trial.step == "t1"
    got = _status(write_extension_figures(run, tmp_path / "out"))
    assert got["whisker_contribution"]["status"] == "needs cohort"
    assert "T2 test" in got["whisker_contribution"]["reason"]


def test_ymaze_only_run_excludes_the_comparison_context(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for animal in ("1", "2", "3"):
        for label in ("proximal25", "proximal45", "proximal65", "compare_0pct_dwell0s"):
            _ymaze(run, animal, label, "ABCABCABC" + "AB" * int(animal))
            _ymaze(run, animal, label, "ABCABCAB" + "AB" * int(animal), phase_part="1")
    index = write_extension_figures(run, tmp_path / "out")
    assert [f["id"] for f in index["figures"]] == IDS
    assert {f["id"] for f in index["figures"] if f["status"] == "ok"} == {
        "ymaze_threshold_concordance",
        "pre_post_trajectories",
    }
    assert _status(index)["whisker_contribution"]["status"] == "needs cohort"
    fig = _figure(ymaze_threshold_concordance, run)
    assert [t.get_text() for t in fig.axes[-1].get_xticklabels()] == ["25%", "45%", "65%"]
    plt.close("all")


def test_oft_only_run_renders_only_the_trajectories(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for i, animal in enumerate(("1", "2", "3")):
        for part in ("1", "2"):
            _oft(run, animal, "5", part, 20.0 + i + int(part))
    got = write_extension_figures(run, tmp_path / "out")["figures"]
    assert [f["id"] for f in got] == IDS
    assert [f["id"] for f in got if f["status"] == "ok"] == ["pre_post_trajectories"]
    assert {f["status"] for f in got if f["status"] != "ok"} == {"needs cohort"}


def test_unknown_task_type_lists_every_figure_as_needing_a_cohort(tmp_path: Path) -> None:
    d = tmp_path / "run" / "data" / "w_1"
    d.mkdir(parents=True)
    (d / "result.toml").write_text('task_type = "water_maze"\n')
    got = write_extension_figures(tmp_path / "run", tmp_path / "out")["figures"]
    assert [f["id"] for f in got] == IDS and all(f["status"].startswith("needs") for f in got)


def test_trial_descriptor_defaults_to_generic() -> None:
    from bekiper_inspect.experiments import generic

    t = Trial(Path("x"), "", "", {}, {}, None, None)
    assert t.descriptor["id"] == generic()["id"]


def test_archived_proximal_and_arm_contexts_stay_distinct(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for animal in ("1", "2", "3"):
        for label in ("proximal25", "arm25", "arm45"):
            _ymaze(run, animal, label, "ABCABCABC" + "AB" * int(animal))
            _ymaze(run, animal, label, "ABCABCAB" + "AB" * int(animal), phase_part="1")
    rows, labels = _ymaze_rows(load_trials(run))
    assert len(labels) == 3 and len(set(labels)) == 3, labels
    assert "25% [proximal25]" in labels and "25% [arm25]" in labels and "45%" in labels
    assert all(len(v) == 3 for v in rows.values())


def test_time_falls_back_to_the_trial_name_without_session_context():
    from bekiper_inspect.plot.extensions import _time

    assert _time({"phase_part": 2}, "ym_pre_a1") == "post"
    assert _time({}, "ym_pre_a1") == "pre" and _time({}, "ym_post_a1") == "post"
    assert _time({}, "ym_a1") is None
