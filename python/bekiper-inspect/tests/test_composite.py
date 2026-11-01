from __future__ import annotations

import io

from PIL import Image

from bekiper_inspect.plot.composite import Receipt, Variant, _matrix_groups, render_composite


def _cells(groups) -> int:
    return sum(len(cells) for _, _, cells in groups)


def test_golden_manifest_renders_one_webp_with_a_complete_receipt(sample_manifest) -> None:
    fig, receipt = render_composite([Variant("recorded", sample_manifest)], {"title": "Animal 7", "trial_id": "t"})
    buffer = io.BytesIO()
    fig.savefig(buffer, format="webp", dpi=60, pil_kwargs={"quality": 85})
    image = Image.open(io.BytesIO(buffer.getvalue()))
    assert image.format == "WEBP" and image.width > 1000
    assert len(buffer.getvalue()) < 2_000_000
    accounted = set(receipt.drawn) | set(receipt.folded) | set(receipt.omitted)
    assert {f.label for f in sample_manifest.feature_inspections} <= accounted
    # Every omission names its reason.
    assert all(receipt.omitted.values())


def test_identical_variants_share_component_cells(sample_manifest) -> None:
    one = _cells(_matrix_groups([Variant("25%", sample_manifest)], Receipt()))
    two = _cells(_matrix_groups([Variant("25%", sample_manifest), Variant("45%", sample_manifest)], Receipt()))
    assert one > 0
    # Final masks repeat per variant; components are drawn once "(all variants)".
    assert two < 2 * one


def test_variant_endpoints_get_one_header_line_each() -> None:
    from bekiper_inspect.plot.composite import _endpoint_lines, _header_height

    table = {"endpoints": {"25%": {"sa": 57.14, "n": 12}, "45%": {"sa": -0.0, "n": None}}}
    assert _endpoint_lines(table) == ["25%: sa = 57.1   n = 12", "45%: sa = 0   n = NA"]
    assert _endpoint_lines({"endpoints": {"DI": 0.25}}) == ["DI = 0.25"]
    assert _header_height(table) > _header_height({"endpoints": {"DI": 0.25}})
