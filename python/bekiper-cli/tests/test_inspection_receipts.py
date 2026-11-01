from pathlib import Path

import pytest
from matplotlib import pyplot as plt

from bekiper_cli.inspection_export import RenderJob, execute_plan
from bekiper_inspect.plot.theme import PlotTheme


def test_failed_declared_renderer_cannot_publish_a_completed_receipt(tmp_path: Path):
    import json

    output = tmp_path / "figures"
    with pytest.raises(RuntimeError, match="produced no figure"):
        list(execute_plan([RenderJob("required", lambda: None, True)], PlotTheme(), tmp_path, output, {}))
    assert json.loads((output / "render_receipt.json").read_text())["status"] == "failed"


def test_receipt_retains_decimal_label_and_resolved_profile(tmp_path: Path):
    import json

    theme = PlotTheme(min_dpi=48, fig_format=".png")

    def draw():
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1])
        fig._bekiper_plot_theme = theme.with_overrides(min_dpi=72)
        return fig

    output = tmp_path / "figures"
    paths = list(execute_plan([RenderJob("distance_-0.01", draw, True)], theme, tmp_path, output, {}))
    assert paths == [output / "distance_-0.01.png"]
    receipt = json.loads((output / "render_receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["artifacts"][0]["effective_theme"]["min_dpi"] == 72
