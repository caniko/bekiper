import copy

import matplotlib.pyplot as plt
import pytest

from bekiper_cli.post_analysis.render_bundle import draw_panel, validate_bundle
from test_post_analysis_renderer import bundle
import numpy as np


def distribution_bundle(style):
    value = bundle()
    panel = value["figures"][0]["panels"][0]
    panel["style"] = style
    # Deliberately distinct from point quantiles: the renderer must draw Rust's
    # values instead of deriving a different distribution from observations.
    panel["summaries"][0]["distribution"] = {
        "status": "available",
        "box": {"whislo": 0.0, "q1": 0.5, "med": 0.6, "q3": 0.7, "whishi": 1.0, "fliers": []},
        "histogram": {"edges": [0.0, 0.5, 1.0], "counts": [2, 0]},
        "density": {"status": "available", "x": [0.0, 0.5, 1.0], "y": [0.0, 2.0, 0.0]},
    }
    return value


@pytest.mark.parametrize("style", ["violin", "boxplot", "histogram"])
def test_distribution_artist_uses_native_payload(style):
    value = distribution_bundle(style)
    validate_bundle(value)
    fig, ax = plt.subplots()
    try:
        draw_panel(ax, value["figures"][0]["panels"][0])
        if style == "boxplot":
            assert any(list(line.get_ydata()) == [0.6, 0.6] for line in ax.lines)
        elif style == "histogram":
            assert ax.get_ylabel() == "Animals"
            assert ax.patches[0].get_data().values.tolist() == [2, 0]
        else:
            assert any(collection.get_paths() for collection in ax.collections)
    finally:
        plt.close(fig)


def test_distribution_boundary_rejects_mismatched_population_and_negative_density():
    for key, bad in [("histogram", {"edges": [0, 1], "counts": [3]}), ("density", {"x": [0, 1], "y": [0, -1]})]:
        value = copy.deepcopy(distribution_bundle("violin"))
        value["figures"][0]["panels"][0]["summaries"][0]["distribution"][key] = bad
        with pytest.raises(ValueError):
            validate_bundle(value)


@pytest.mark.parametrize("style", ["boxplot", "violin", "histogram"])
@pytest.mark.parametrize(
    "observations, box_values, fliers",
    [
        ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.25, 0.25], [1.0]),
        ([0.0, 1.0, 1.0, 1.0], [0.75, 0.75, 1.0, 1.0, 1.0], [0.0]),
        ([2.0], [2.0] * 5, []),
        ([0.0] * 4, [0.0] * 5, []),
        ([0.0, 0.0, 0.0, 0.0, 20.0], [0.0] * 5, [20.0]),
    ],
)
def test_tied_and_sparse_native_box_contract(style, observations, box_values, fliers):
    # These independent expectations also gate Rust's distributions::attach.
    # The distribution_smoke edge_cases bundle exercises the full live boundary.
    value = distribution_bundle(style)
    panel = value["figures"][0]["panels"][0]
    prototype = panel["points"][0]
    panel["points"] = [
        dict(prototype, animal_id=f"mouse-{index}", value=observation) for index, observation in enumerate(observations)
    ]
    panel["n_animals"] = len(observations)
    summary = panel["summaries"][0]
    summary.update(n_animals=len(observations), uncertainty=None, error_half_width=None)
    distribution = summary["distribution"]
    distribution["box"] = dict(zip(["whislo", "q1", "med", "q3", "whishi"], box_values, strict=True), fliers=fliers)
    distribution["histogram"] = {"edges": [-1.0, 21.0], "counts": [len(observations)]}
    distribution["density"] = {"status": "unavailable:constant_or_singleton", "x": [], "y": []}
    validate_bundle(value)
    fig, ax = plt.subplots()
    try:
        draw_panel(ax, panel)
        fig.canvas.draw()
        if style == "boxplot":
            assert any(list(line.get_ydata()) == [box_values[1], box_values[0]] for line in ax.lines)
            assert any(list(line.get_ydata()) == [box_values[3], box_values[4]] for line in ax.lines)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("key, invalid", [("whislo", 0.55), ("whishi", 0.65)])
def test_distribution_boundary_still_rejects_whiskers_inside_the_box(key, invalid):
    value = distribution_bundle("boxplot")
    value["figures"][0]["panels"][0]["summaries"][0]["distribution"]["box"][key] = invalid
    with pytest.raises(ValueError, match="Native box quantiles/whiskers must be ordered"):
        validate_bundle(value)


def test_swarm_is_collision_aware_and_identity_stable_under_reordering():
    from bekiper_cli.post_analysis.render_bundle import swarm_coordinates

    points = [{"animal_id": f"mouse-{i}", "session_id": "test", "value": 0.0 if i < 8 else i / 10} for i in range(12)]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.1, 1.3)
    coordinates, size = swarm_coordinates(ax, points, 0)
    shuffled, shuffled_size = swarm_coordinates(ax, list(reversed(points)), 0)
    np.testing.assert_allclose(coordinates, list(reversed(shuffled)))
    assert size == shuffled_size
    pixels = ax.transData.transform(np.column_stack([coordinates, [p["value"] for p in points]]))
    diameter = np.sqrt(size) * fig.dpi / 72 * 1.15
    distances = np.linalg.norm(pixels[:, None] - pixels[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= diameter - 1e-6
    assert max(abs(x) for x in coordinates) <= 0.38
