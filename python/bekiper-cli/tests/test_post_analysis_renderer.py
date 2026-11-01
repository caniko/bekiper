"""The renderer must draw native quantities, never recompute statistics."""

import copy
import json

import matplotlib.pyplot as plt
import pytest

from bekiper_cli.post_analysis.render_bundle import draw_panel, render_bundle, validate_bundle


def categorical_panel():
    return {
        "endpoint": "DI",
        "label": "DI (primary)",
        "kind": "categorical",
        "reference_line": 0,
        "status": "rendered",
        "n_animals": 2,
        "points": [
            {
                "animal_id": "a1",
                "session_id": "s1",
                "category": "G5",
                "value": 0.2,
                "pair_series": False,
                "plot_pair_id": None,
                "pair_order": 0,
            },
            {
                "animal_id": "a2",
                "session_id": "s2",
                "category": "G5",
                "value": 0.4,
                "pair_series": False,
                "plot_pair_id": None,
                "pair_order": 0,
            },
        ],
        # Deliberately different from the observations: drawing this proves the
        # Python boundary does not calculate or replace Rust's mean/error.
        "summaries": [
            {
                "category": "G5",
                "n_animals": 2,
                "mean": 0.8,
                "sem": 0.1,
                "error_half_width": 0.15,
                "uncertainty": "SEM",
                "status": "available",
            }
        ],
    }


def bundle():
    return {
        "schema_version": "1.0",
        "figures": [
            {"name": "figure_nort", "status": "rendered", "formats": ["svg", "pdf"], "panels": [categorical_panel()]}
        ],
        "plotting_data": [],
        "summaries": [],
    }


def test_descriptive_provenance_references_resolve_exact_analysis_identities():
    value = bundle()
    value["scoring_provenance_by_analysis"] = {"analysis_s1": {"minimum_dwell_seconds": 0.0}, "analysis_s2": None}
    panel = value["figures"][0]["panels"][0]
    for point in panel["points"]:
        point["analysis_id"] = f"analysis_{point['session_id']}"
        point["scoring_provenance_ref"] = point["analysis_id"]
    panel["eligibility_ledger"] = [copy.deepcopy(point) for point in panel["points"]]
    value["plotting_data"] = copy.deepcopy(panel["points"])
    value["eligibility_ledger"] = copy.deepcopy(panel["eligibility_ledger"])
    original = copy.deepcopy(value)
    validate_bundle(value)
    assert value == original
    for records in [panel["points"], panel["eligibility_ledger"], value["plotting_data"], value["eligibility_ledger"]]:
        reference = records[0]["scoring_provenance_ref"]
        for bad_reference in ["absent_analysis", "analysis_s2", None]:
            records[0]["scoring_provenance_ref"] = bad_reference
            with pytest.raises(ValueError, match="provenance"):
                validate_bundle(value)
        records[0]["scoring_provenance_ref"] = reference


def test_recorded_descriptive_retains_repeated_recordings_and_observation_counts():
    value = bundle()
    panel = value["figures"][0]["panels"][0]
    panel.update(population_policy="recorded_descriptive", n_animals=1, style="histogram")
    for index, point in enumerate(panel["points"]):
        point.update(animal_id="a1", analysis_id=f"analysis_{index}")
    summary = panel["summaries"][0]
    summary.update(n_animals=1, n_observations=2, uncertainty=None, error_half_width=None)
    summary["distribution"] = {
        "status": "available",
        "box": {"whislo": 0.2, "q1": 0.25, "med": 0.3, "q3": 0.35, "whishi": 0.4, "fliers": []},
        "histogram": {"edges": [0.1, 0.3, 0.5], "counts": [1, 1]},
        "density": {"status": "unavailable:insufficient_variation", "x": [], "y": []},
    }
    validate_bundle(value)
    fig, ax = plt.subplots()
    try:
        draw_panel(ax, panel)
        assert ax.get_ylabel() == "Observations"
    finally:
        plt.close(fig)
    panel["points"][1].update(session_id="s1", analysis_id="analysis_0")
    with pytest.raises(ValueError, match="repeated"):
        validate_bundle(value)


def test_draws_supplied_mean_error_and_dashed_reference():
    panel = categorical_panel()
    fig, ax = plt.subplots()
    try:
        draw_panel(ax, panel)
        assert ax.lines[0].get_linestyle() == "--"
        assert list(ax.lines[0].get_ydata()) == [0, 0]
        assert any(list(line.get_ydata()) == [0.8] for line in ax.lines)
        error_segments = [
            segment
            for collection in ax.collections
            if hasattr(collection, "get_segments")
            for segment in collection.get_segments()
        ]
        assert any(segment[:, 1].tolist() == pytest.approx([0.65, 0.95]) for segment in error_segments)
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["G5\nn=2"]
    finally:
        plt.close(fig)


def test_het_category_labels_do_not_overlap_at_three_column_panel_width():
    from copy import deepcopy
    from itertools import pairwise

    panel = categorical_panel()
    template = deepcopy(panel)
    categories = ["G1", "G2", "G3", "G4", "G5", "G5 [Het]", "G6", "G7", "G8"]
    panel["points"] = []
    panel["summaries"] = []
    panel["n_animals"] = len(categories) * 2
    for index, category in enumerate(categories):
        summary = deepcopy(template["summaries"][0])
        summary["category"] = category
        panel["summaries"].append(summary)
        for point_index, original in enumerate(template["points"]):
            point = deepcopy(original)
            point.update(category=category, animal_id=f"animal_{index}_{point_index}")
            panel["points"].append(point)
    original = deepcopy(panel)
    fig, ax = plt.subplots(figsize=(4, 3.8))
    try:
        draw_panel(ax, panel)
        fig.canvas.draw()
        labels = ax.get_xticklabels()
        bounds = sorted((label.get_window_extent() for label in labels), key=lambda box: box.x0)
        assert all(left.x1 < right.x0 for left, right in pairwise(bounds))
        assert any("G5 [Het]" in label.get_text() and "n=2" in label.get_text() for label in labels)
        assert panel == original
    finally:
        plt.close(fig)


def test_draws_only_explicit_same_animal_pairing():
    panel = categorical_panel()
    panel["points"] = [
        {
            "animal_id": "a1",
            "session_id": "pre",
            "category": "G5:pre",
            "value": 10,
            "pair_series": True,
            "plot_pair_id": "registered_pair",
            "pair_order": 0,
        },
        {
            "animal_id": "a1",
            "session_id": "post",
            "category": "G5:post",
            "value": 20,
            "pair_series": True,
            "plot_pair_id": "registered_pair",
            "pair_order": 1,
        },
    ]
    panel["summaries"] = [
        {
            "category": category,
            "n_animals": 1,
            "mean": mean,
            "error_half_width": None,
            "uncertainty": "SEM",
            "status": "unavailable",
        }
        for category, mean in [("G5:pre", 10), ("G5:post", 20)]
    ]
    panel["n_animals"] = 1
    fig, ax = plt.subplots()
    try:
        draw_panel(ax, panel)
        assert any(list(line.get_ydata()) == [10, 20] for line in ax.lines)
    finally:
        plt.close(fig)


def test_pair_lines_end_at_the_individual_points_after_jitter():
    panel = categorical_panel()
    panel["points"] = [
        {
            "animal_id": animal,
            "session_id": f"{animal}_{phase}",
            "category": f"G5:{phase}",
            "value": value,
            "pair_series": True,
            "plot_pair_id": animal,
            "pair_order": order,
        }
        for animal, values in [("a1", [10, 20]), ("a2", [30, 40])]
        for order, (phase, value) in enumerate(zip(["pre", "post"], values, strict=True))
    ]
    panel["summaries"] = [
        {
            "category": f"G5:{phase}",
            "n_animals": 2,
            "mean": mean,
            "error_half_width": None,
            "uncertainty": None,
            "status": "blocked:D09:summary_uncertainty",
        }
        for phase, mean in [("pre", 20), ("post", 30)]
    ]
    fig, ax = plt.subplots()
    try:
        draw_panel(ax, panel)
        locations = {float(y): float(x) for collection in ax.collections for x, y in collection.get_offsets()}
        pair_lines = [line for line in ax.lines if len(line.get_xdata()) == 2 and line.get_linestyle() == "-"]
        assert len(pair_lines) == 2
        for line in pair_lines:
            for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True):
                assert x == pytest.approx(locations[float(y)])
    finally:
        plt.close(fig)


def test_empty_categorical_panel_reports_zero_animals():
    panel = categorical_panel()
    panel.update(points=[], summaries=[], n_animals=0, status="unavailable_or_blocked")
    fig, ax = plt.subplots()
    try:
        draw_panel(ax, panel)
        assert "n=0 animals" in ax.get_title()
    finally:
        plt.close(fig)


def test_unavailable_single_animal_uncertainty_cannot_be_drawn(tmp_path):
    value = bundle()
    panel = value["figures"][0]["panels"][0]
    panel["points"] = panel["points"][:1]
    panel["n_animals"] = 1
    panel["summaries"][0].update(n_animals=1, status="unavailable")
    with pytest.raises(ValueError, match="Uncertainty requires at least two animals"):
        render_bundle(value, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_censored_bound_markers_use_native_null_summary_without_calculation():
    value = bundle()
    panel = value["figures"][0]["panels"][0]
    panel["points"][0]["censored"] = True
    panel["summaries"][0].update(
        mean=None, sem=None, error_half_width=None, status="blocked:D09:censored_latency_summary"
    )
    validate_bundle(value)
    fig, ax = plt.subplots()
    try:
        draw_panel(ax, panel)
        assert len(ax.collections) == 2
        assert any(collection.get_offsets()[:, 1].tolist() == [0.2] for collection in ax.collections)
        assert not any(line.get_marker() == "_" for line in ax.lines)
    finally:
        plt.close(fig)


def test_empty_and_scatter_panels_use_native_counts_and_labels():
    panel = {
        "endpoint": "DI",
        "label": "DI vs exploration",
        "kind": "scatter",
        "x_label": "Exploration (s)",
        "y_label": "DI",
        "reference_line": 0,
        "points": [],
        "summaries": [],
        "n_animals": 0,
        "status": "unavailable_or_blocked",
    }
    fig, ax = plt.subplots()
    try:
        draw_panel(ax, panel)
        assert ax.get_xlabel() == "Exploration (s)"
        assert "n=0 animals" in ax.get_title()
        assert any("Unavailable / blocked" in text.get_text() for text in ax.texts)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mutation", ["version", "path", "format", "nonfinite", "repeat", "count", "pair"])
def test_invalid_bundle_rejected_before_writing(tmp_path, mutation):
    value = bundle()
    figure, panel = value["figures"][0], value["figures"][0]["panels"][0]
    if mutation == "version":
        value["schema_version"] = "future"
    elif mutation == "path":
        figure["name"] = "../outside"
    elif mutation == "format":
        figure["formats"] = ["tex"]
    elif mutation == "nonfinite":
        panel["points"][0]["value"] = float("nan")
    elif mutation == "repeat":
        panel["points"][1]["animal_id"] = "a1"
    elif mutation == "count":
        panel["summaries"][0]["n_animals"] = 3
    else:
        for point in panel["points"]:
            point.update(pair_series=True, plot_pair_id="different_animals")
    with pytest.raises(ValueError):
        render_bundle(value, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_writes_only_requested_vector_formats_and_does_not_overwrite(tmp_path):
    value = bundle()
    original = copy.deepcopy(value)
    paths = render_bundle(value, tmp_path)
    assert {path.name for path in paths} == {"figure_nort.svg", "figure_nort.pdf"}
    assert (tmp_path / "figure_nort.pdf").read_bytes().startswith(b"%PDF")
    assert "DI (primary)" in (tmp_path / "figure_nort.svg").read_text()
    assert value == original
    before = (tmp_path / "figure_nort.svg").read_bytes()
    with pytest.raises(FileExistsError):
        render_bundle(value, tmp_path)
    assert (tmp_path / "figure_nort.svg").read_bytes() == before


def test_refuses_symlink_targets(tmp_path):
    outside = tmp_path / "outside.svg"
    outside.write_text("preserve")
    (tmp_path / "figure_nort.svg").symlink_to(outside)
    with pytest.raises(FileExistsError):
        render_bundle(bundle(), tmp_path)
    assert outside.read_text() == "preserve"


def test_renderer_validation_accepts_native_missing_uncertainty():
    value = bundle()
    summary = value["figures"][0]["panels"][0]["summaries"][0]
    summary.update(uncertainty=None, error_half_width=None, status="blocked:D09:summary_uncertainty")
    validate_bundle(value)


def test_cli_renders_json_bundle(tmp_path):
    from bekiper_cli.post_analysis.render_bundle import main

    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle()))
    assert main(["--bundle", str(path), "--output", str(tmp_path / "rendered")]) == 0
    assert (tmp_path / "rendered/figure_nort.pdf").is_file()


def test_parallel_render_matches_serial_bytes(tmp_path):
    value = bundle()
    second = copy.deepcopy(value["figures"][0])
    second["name"] = "figure_nort_copy"
    value["figures"].append(second)
    serial = render_bundle(value, tmp_path / "serial", workers=1)
    parallel = render_bundle(value, tmp_path / "parallel", workers=2)
    for left, right in zip(serial, parallel, strict=True):
        assert left.name == right.name and left.read_bytes() == right.read_bytes()


def test_vector_outputs_are_byte_reproducible(tmp_path):
    first = render_bundle(bundle(), tmp_path / "first")
    second = render_bundle(bundle(), tmp_path / "second")
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()


def test_native_raster_dpi_controls_pixels_and_png_metadata(tmp_path):
    from PIL import Image

    value = bundle()
    value["figures"][0].update(formats=["png"], dpi=180)
    render_bundle(value, tmp_path)
    with Image.open(tmp_path / "figure_nort.png") as image:
        assert image.size == (720, 684)
        assert image.info["dpi"] == pytest.approx((180, 180), abs=0.02)


@pytest.mark.parametrize("dpi", [0, -1, True, 1.5, "180"])
def test_invalid_raster_dpi_is_rejected_before_writing(tmp_path, dpi):
    value = bundle()
    value["figures"][0]["dpi"] = dpi
    with pytest.raises(ValueError, match="dpi"):
        render_bundle(value, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_runtime_save_failure_is_reported_and_figures_are_closed(tmp_path, monkeypatch):
    import matplotlib.figure

    from bekiper_cli.post_analysis.render_bundle import main

    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle()))
    before = plt.get_fignums()

    def fail(*args, **kwargs):
        raise OSError("simulated figure-write failure")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail)
    assert main(["--bundle", str(path), "--output", str(tmp_path / "rendered")]) == 2
    assert plt.get_fignums() == before


def test_markers_follow_the_selected_colour_mode():
    from matplotlib.colors import to_hex

    from bekiper_cli.post_analysis.render_bundle import use_color_mode

    inks = {}
    try:
        for mode in ("dark", "light"):
            with plt.rc_context(use_color_mode(mode)):
                fig, ax = plt.subplots()
                try:
                    draw_panel(ax, categorical_panel())
                    colours = {to_hex(line.get_color()) for line in ax.get_lines()}
                    background = to_hex(fig.get_facecolor())
                finally:
                    plt.close(fig)
            inks[mode] = (colours, background)
    finally:
        use_color_mode(None)
    assert "#e6e9ee" in inks["dark"][0] and inks["dark"][1] == "#0f1318"
    assert "#1d2733" in inks["light"][0] and inks["light"][1] == "#ffffff"
