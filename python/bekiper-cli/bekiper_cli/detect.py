"""bekiper detect — arena/object detection commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import tomli_w
import typer

app = typer.Typer(help="Arena and object detection for behavioral experiments.")
logger = logging.getLogger(__name__)
DEFAULT_CONFIG_FILENAME = "bekiper_config.toml"
# Keep this lightweight alias list in sync with `RECIPE_REGISTRY` so
# `bekiper detect list` still works when auto-detect deps are unavailable.
AVAILABLE_RECIPES = (
    "nort",
    "nort_novelty",
    "nort_training",
    "oft",
    "open_field",
    "oul_test",
    "oul_training",
    "oul_update",
    "y_maze",
    "ymaze",
)


@app.command("list")
def list_recipes() -> None:
    """List available detection recipes."""
    for name in AVAILABLE_RECIPES:
        typer.echo(name)


@app.command()
def run(
    video: Path = typer.Option(..., help="Path to video file."),
    recipe: str = typer.Option(..., help="Detection recipe name."),
    output: Path = typer.Option(..., help="Output perimeters.toml path."),
    frame: int = typer.Option(30, help="Frame number to extract."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    config: Path | None = typer.Option(None, "--config", help="Path to bekiper_config.toml (reads [detect] section)."),
    device: str | None = typer.Option(None, "--device", help="Compute backend: auto|cuda|mps|cpu."),
    sam2_model: str | None = typer.Option(None, "--sam2-model", help="Override SAM2 checkpoint."),
    grounding_dino_model: str | None = typer.Option(
        None, "--grounding-dino-model", help="Override GroundingDINO model ID."
    ),
    yoloe_model: str | None = typer.Option(None, "--yoloe-model", help="Override YOLOE checkpoint."),
    confidence: float | None = typer.Option(None, "--confidence", help="Detection confidence threshold (0–1)."),
    box_threshold: float | None = typer.Option(
        None, "--box-threshold", help="GroundingDINO box-score threshold (0–1)."
    ),
) -> None:
    """Run detection on a video frame and output perimeters."""
    import cv2
    from bekiper_auto_detect.config import DetectorConfig
    from bekiper_auto_detect.mask_to_perimeter import mask_to_perimeter
    from bekiper_auto_detect.recipe import RECIPE_REGISTRY

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s")

    if recipe not in RECIPE_REGISTRY:
        logger.error(f"Unknown recipe '{recipe}'. Available: {', '.join(sorted(RECIPE_REGISTRY))}")
        raise typer.Exit(1)

    if not video.exists():
        logger.error(f"Video not found: {video}")
        raise typer.Exit(1)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video}")
        raise typer.Exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    cap.release()

    if not ret:
        logger.error(f"Could not read frame {frame} from {video}")
        raise typer.Exit(1)

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detector_config = DetectorConfig.load(
        config_path=config,
        device=device,
        sam2_model=sam2_model,
        grounding_dino_model=grounding_dino_model,
        yoloe_model=yoloe_model,
        confidence=confidence,
        box_threshold=box_threshold,
    )
    recipe_fn = RECIPE_REGISTRY[recipe](config=detector_config)
    try:
        result = recipe_fn.detect(frame_rgb)
    except ValueError as e:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame_path = output.with_name(output.stem + "_frame.png")
        cv2.imwrite(str(frame_path), img)
        logger.error(f"{e}\nSaved detection frame to {frame_path}")
        raise typer.Exit(1)
    labeled_masks = recipe_fn.assign_labels(result.masks)

    perimeters: list[dict[str, Any]] = []
    for mask in labeled_masks:
        perim = mask_to_perimeter(mask)
        perimeters.append(_perimeter_to_dict(mask.label, perim))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(tomli_w.dumps({"perimeters": perimeters}))

    logger.info(f"Wrote {len(perimeters)} perimeters to {output}")


def _perimeter_to_dict(label: str, perimeter: Any) -> dict[str, Any]:
    from bekiper_auto_detect.mask_to_perimeter import CirclePerimeter, PolygonPerimeter

    if isinstance(perimeter, CirclePerimeter):
        cx, cy = perimeter.center_pixels
        return {
            "shape": "circle",
            "label": label,
            "center_x": float(cx),
            "center_y": float(cy),
            "radius": float(perimeter.radius_pixels),
        }
    elif isinstance(perimeter, PolygonPerimeter):
        vertices = [(float(x), float(y)) for x, y in perimeter.vertices_pixels]
        return {
            "shape": "polygon",
            "label": label,
            "vertices": vertices,
        }
    else:
        raise TypeError(f"Unknown perimeter type: {type(perimeter)}")
