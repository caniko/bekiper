"""Experiment descriptors (single source: ``rust/bekiper-core/experiments/*.toml``).

``descriptors()`` returns one plain dict per descriptor, the same keys and nesting
as the TOML, sorted by ``order`` (``generic`` last). It reads them from
``bekiper_rs.experiment_descriptors()``; the helpers mirror the Rust methods of
the same names (``Contexts::label``/``order_key``/``is_comparison``,
``Matcher::matches``, ``Objects::is_object``, ``descriptor_for``).
"""

from __future__ import annotations

import json
import re
import tomllib
from functools import cache
from pathlib import Path
from typing import Any

Descriptor = dict[str, Any]

_SOURCES = Path(__file__).resolve().parents[3] / "rust" / "bekiper-core" / "experiments"


@cache
def descriptors() -> tuple[Descriptor, ...]:
    """Every built-in descriptor, sorted by ``order``."""
    try:
        from bekiper_rs import experiment_descriptors  # a broken .so or stale export raises
    except ModuleNotFoundError as e:
        if e.name != "bekiper_rs":
            raise
        # bekiper_rs genuinely absent: dev-checkout fallback to the TOMLs the Rust crate embeds.
        files = sorted(_SOURCES.glob("*.toml"))
        if not files:
            raise RuntimeError(f"bekiper_rs is not installed and no descriptor TOMLs under {_SOURCES}") from e
        loaded = [tomllib.loads(p.read_text()) for p in files]
    else:
        loaded = list(experiment_descriptors())
    if not loaded:
        raise RuntimeError("no experiment descriptors loaded")
    return tuple(sorted(loaded, key=lambda d: d["order"]))


def generic() -> Descriptor:
    return next(d for d in descriptors() if d["id"] == "generic")


def descriptor_for(task_type: str) -> Descriptor:
    """The descriptor of a task type or alias; ``generic`` otherwise."""
    for d in descriptors():
        if d["task_type"] and task_type in (d["task_type"], *(d.get("aliases") or [])):
            return d
    return generic()


def section(d: Descriptor, name: str) -> Any:
    """A descriptor table or ``{}`` (``None`` and a missing key are the same)."""
    return d.get(name) or {}


def all_features(name: str) -> list[Any]:
    """``features.<name>`` across every descriptor, in descriptor order, deduplicated."""
    out: list[Any] = []
    for d in descriptors():
        out += [v for v in section(d, "features").get(name) or [] if v not in out]
    return out


# ── contexts ──────────────────────────────────────────────────────────────


def _capture(pattern: str, variant: str) -> str | None:
    """The ``{n}`` digits when ``pattern`` matches (``""`` without ``{n}``)."""
    any_suffix = pattern.endswith("*")
    pattern = pattern.removesuffix("*")
    head, has_n, tail = pattern.partition("{n}")
    if not has_n:
        return "" if (variant.startswith(pattern) if any_suffix else variant == pattern) else None
    if not variant.startswith(head):
        return None
    match = re.match(r"\d+", variant[len(head) :])
    if not match:
        return None
    after = variant[len(head) + match.end() :]
    return match.group() if (after.startswith(tail) if any_suffix else after == tail) else None


def _first(contexts: dict[str, Any], variant: str, field: str) -> Any:
    for rule in contexts.get("rules") or []:
        n = _capture(rule["pattern"], variant)
        if n is not None and rule.get(field) is not None:
            return (n, rule[field])
    return None


def context_label(contexts: dict[str, Any], variant: str | None) -> str:
    """Display label of a variant (``arm45`` -> ``45%``; none -> ``recorded``)."""
    if not variant:
        return "recorded"
    hit = _first(contexts, variant, "label")
    return hit[1].replace("{n}", hit[0]) if hit else variant


def context_order_key(contexts: dict[str, Any], variant: str | None) -> tuple[int, str]:
    """Rank, then the name with ``{n}`` zero-padded (``arm25`` < ``arm100``)."""
    if not variant:
        return (0, "")
    hit = _first(contexts, variant, "rank")
    rank = hit[1] if hit else 1
    for rule in contexts.get("rules") or []:
        n = _capture(rule["pattern"], variant)
        if n and "{n}" in rule["pattern"]:
            head = rule["pattern"].split("{n}")[0]
            return rank, f"{head}{int(n):05}{variant[len(head) + len(n) :]}"
    return rank, variant


def is_comparison(contexts: dict[str, Any], variant: str) -> bool:
    hit = _first(contexts, variant, "comparison")
    return bool(hit and hit[1])


def variant_of(analysis_role: str | None, variant: str | None = None) -> str | None:
    """The scoring variant, else the ``analysis_role`` suffix (``ymaze__arm45`` -> ``arm45``)."""
    if variant:
        return variant
    _, sep, suffix = (analysis_role or "").partition("__")
    return suffix if sep and suffix else None


def context_identity(context: Path | str) -> tuple[str, str | None]:
    """``(id, assay)`` of a post-analysis context directory.

    The run manifest records ``primary_context`` (``id``, ``assay``); older runs
    only the threshold, and a bare name falls back to the legacy ``primary<pct>``
    directory naming (``arm<pct>``), so the directory name is never the identity.
    """
    path = Path(context)
    try:
        record = json.loads((path / "run_manifest.json").read_text())["resolved_config"]["primary_context"]
    except (OSError, ValueError, KeyError, TypeError):
        record = {}
    ident = record.get("id") or (
        f"arm{record['threshold_pct']}"
        if record.get("threshold_pct") is not None
        else re.sub(r"^primary(\d+)$", r"arm\1", path.name)
    )
    return ident, record.get("assay") or ("ymaze" if record.get("threshold_pct") is not None else None)


def context_assays(context: Path | str, present: set[str]) -> set[str] | None:
    """Assays a post-analysis context draws (``None`` = all of ``present``).

    The principal cohort (``contexts.principal_cohort``, matched by context id)
    carries every experiment; another context only the assay it varies (its
    recorded ``assay``), else the experiments that name a principal cohort.
    """
    ident, assay = context_identity(context)
    cohorts = {section(d, "contexts").get("principal_cohort") for d in descriptors()}
    contexted = (
        {assay}
        if assay
        else {
            section(d, "post_analysis").get("assay")
            for d in descriptors()
            if section(d, "contexts").get("principal_cohort")
        }
    )
    own = present & contexted
    return None if ident in cohorts or not own else own


# ── protocol steps ────────────────────────────────────────────────────────


def matches(matcher: dict[str, Any], facts: dict[str, str | None]) -> bool:
    """All set fields equal the trial's value; ``""`` means the field is unrecorded."""
    for name, want in matcher.items():
        if want is None:
            continue
        have = facts.get(name) or None
        if (have is not None) if want == "" else (have != want):
            return False
    return True


def step_for(d: Descriptor, facts: dict[str, str | None]) -> Descriptor | None:
    """The first step (file order) one of whose matchers holds; no matchers match every trial.

    ``facts`` holds ``recipe``, ``phase`` (protocol phase, not a batch letter),
    ``stage``, ``timepoint`` and ``phase_part``.
    """
    for step in d.get("steps") or []:
        alternatives = step.get("match") or []
        if not alternatives or any(matches(m, facts) for m in alternatives):
            return step
    return None


# ── objects ───────────────────────────────────────────────────────────────


def is_object(perimeter: str) -> bool:
    """A perimeter label that is an object or goal in any descriptor."""
    for d in descriptors():
        objects = section(d, "objects")
        if any(r.lower() == perimeter.strip().lower() for r in objects.get("roles") or []):
            return True
        prefix = objects.get("perimeter_prefix")
        if prefix and perimeter.startswith(prefix):
            return True
    return False
