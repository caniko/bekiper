"""Descriptor loader and the helpers that mirror the Rust descriptor methods."""

from __future__ import annotations

import pytest

from bekiper_inspect import experiments as ex


def test_descriptors_load_in_order_with_generic_last() -> None:
    ids = [d["id"] for d in ex.descriptors()]
    assert ids == ["open_field", "object_recognition", "y_maze", "cheeseboard", "generic"]
    assert ex.descriptor_for("nort")["id"] == "object_recognition"
    assert ex.descriptor_for("radial_arm")["id"] == "y_maze"
    assert ex.descriptor_for("water_maze")["id"] == "generic"
    assert ex.descriptor_for("")["id"] == "generic"


def test_context_labels_and_order_match_rust() -> None:
    # Same cases as rust/bekiper-core/src/experiment/tests.rs::context_labels_and_order.
    c = ex.descriptor_for("y_maze")["contexts"]
    for variant, label in [
        ("proximal25", "25%"),
        ("arm0_dwell1s", "0% (1 s dwell)"),
        ("arm45", "45%"),
        ("arm0", "0%"),
        ("compare_0pct_dwell0s", "0% (no dwell)"),
        ("compare_x", "compare_x"),
        ("foo", "foo"),
        (None, "recorded"),
    ]:
        assert ex.context_label(c, variant) == label, variant
    variants = ["proximal65", "arm100", "compare_0pct_dwell0s", None, "arm45", "proximal25", "arm0_dwell1s", "arm25"]
    variants.sort(key=lambda v: ex.context_order_key(c, v))
    assert variants == [
        None,
        "arm0_dwell1s",
        "compare_0pct_dwell0s",
        "arm25",
        "arm45",
        "arm100",
        "proximal25",
        "proximal65",
    ]
    assert ex.is_comparison(c, "compare_0pct_dwell0s") and not ex.is_comparison(c, "arm0_dwell1s")
    assert ex.variant_of("ymaze__arm45") == "arm45" and ex.variant_of("nort_habituation") is None


def test_steps_match_recorded_fields_not_names() -> None:
    nort = ex.descriptor_for("object_recognition")
    step = lambda **facts: (ex.step_for(nort, facts) or {}).get("id")  # noqa: E731
    assert step(stage="2") == "t2"
    assert step(phase="test", stage="1") == "t2"
    assert step(stage="1") == "t1"
    assert step(recipe="nort_novelty") == "t2"
    assert step(recipe="oul_update", stage="2") == "oul_update"
    assert step(stage="0") is None
    assert ex.step_for(ex.descriptor_for("y_maze"), {})["id"] == "ymaze"


def test_objects_from_any_descriptor() -> None:
    assert all(ex.is_object(p) for p in ("familiar", "Novel", "object_3", "reward", "start"))
    assert not any(ex.is_object(p) for p in ("arena", "center_zone", "arm_A"))


def _subset(toml: object, py: object, path: str = "") -> list[str]:
    """Paths where a TOML value is missing or different in the PyO3 export (serde defaults may add keys)."""
    if isinstance(toml, dict):
        if not isinstance(py, dict):
            return [path]
        return [p for k, v in toml.items() for p in _subset(v, py.get(k), f"{path}.{k}")]
    if isinstance(toml, list):
        if not isinstance(py, list) or len(py) != len(toml):
            return [path]
        return [p for i, (a, b) in enumerate(zip(toml, py, strict=True)) for p in _subset(a, b, f"{path}[{i}]")]
    return [] if toml == py else [path]


def test_pyo3_descriptors_mirror_the_toml() -> None:
    """A stale committed _bekiper_rs.abi3.so would silently drop descriptor fields."""
    import tomllib

    import bekiper_rs

    assert hasattr(bekiper_rs, "experiment_descriptors"), "stale _bekiper_rs.abi3.so: rebuild and copy it"
    sources = sorted(ex._SOURCES.glob("*.toml"))
    assert sources, f"no descriptor TOMLs under {ex._SOURCES}"
    pyo3 = {d["id"]: d for d in bekiper_rs.experiment_descriptors()}
    assert {tomllib.loads(p.read_text())["id"] for p in sources} == set(pyo3)
    for source in sources:
        toml = tomllib.loads(source.read_text())
        assert _subset(toml, pyo3[toml["id"]]) == [], source.name
    assert "whisker_interaction_" in ex.all_features("diagnostic_prefixes")


def test_missing_descriptors_raise_and_are_never_cached(monkeypatch, tmp_path) -> None:
    import builtins

    real = builtins.__import__

    def absent(name, *a, **k):
        if name == "bekiper_rs":
            raise ModuleNotFoundError("No module named 'bekiper_rs'", name="bekiper_rs")
        return real(name, *a, **k)

    def broken(name, *a, **k):
        if name == "bekiper_rs":
            raise ImportError("libfoo.so: cannot open shared object file")
        return real(name, *a, **k)

    ex.descriptors.cache_clear()
    try:
        monkeypatch.setattr(builtins, "__import__", absent)
        monkeypatch.setattr(ex, "_SOURCES", tmp_path)  # empty glob
        for _ in range(2):  # a failed load is not cached
            with pytest.raises(RuntimeError, match="no descriptor TOMLs"):
                ex.descriptors()
        monkeypatch.undo()
        monkeypatch.setattr(builtins, "__import__", broken)
        with pytest.raises(ImportError, match="libfoo"):
            ex.descriptors()
    finally:
        monkeypatch.undo()
        ex.descriptors.cache_clear()
    assert ex.generic()["id"] == "generic"
