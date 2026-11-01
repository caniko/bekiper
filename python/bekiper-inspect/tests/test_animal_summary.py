from __future__ import annotations

import io

from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from PIL import Image

from bekiper_inspect.plot.animal import render_animal_summary
from bekiper_inspect.plot.theme import PlotTheme, get_active_theme, set_active_theme

JOB = {
    "title": "Animal 7",
    "subtitle": "G2 · AD · HCAR1-KO · vehicle",
    "panels": [
        {
            "assay": "oft",
            "step": "Habituation (OFT)",
            "endpoint": "center_time_pct",
            "animal": {"pre": 20.0, "post": 25.0},
            "group": {"pre": [10.0, 20.0, 40.0], "post": [12.0, 25.0]},
        },
        {"assay": "nort", "step": "T2 test", "endpoint": "DI", "animal": {"post": 0.5}, "group": {"post": [0.1, 0.5]}},
    ],
}


def test_summary_highlights_the_animal_in_both_modes() -> None:
    previous = get_active_theme()
    try:
        for mode in ("dark", "light"):
            theme = PlotTheme(color_mode=mode)
            set_active_theme(theme)
            fig = render_animal_summary(JOB)
            try:
                visible = [ax for ax in fig.axes if ax.get_visible()]
                assert len(visible) == 2
                lines = [to_hex(line.get_color()) for ax in visible for line in ax.get_lines()]
                assert theme.colors()["highlight"] in lines
                assert to_hex(fig.get_facecolor()) == theme.colors()["background"]
                buffer = io.BytesIO()
                fig.savefig(buffer, format="webp", dpi=60)
                assert Image.open(io.BytesIO(buffer.getvalue())).format == "WEBP"
            finally:
                plt.close(fig)
        empty = render_animal_summary({"title": "Animal 9", "subtitle": "Unrecorded group", "panels": []})
        assert any("No recorded" in t.get_text() for ax in empty.axes for t in ax.texts)
        plt.close(empty)
    finally:
        set_active_theme(previous)
