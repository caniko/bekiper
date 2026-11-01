"""CLI entry point for bekiper_auto_detect, invoked via `python -m bekiper_auto_detect`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tomli_w

from bekiper_auto_detect.config import DetectorConfig
from bekiper_auto_detect.mask_to_perimeter import CirclePerimeter, PolygonPerimeter, mask_to_perimeter
from bekiper_auto_detect.model import DetectionResult
from bekiper_auto_detect.recipe import RECIPE_REGISTRY, ExperimentDetectionRecipe

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_frame(video_path: str, frame_index: int = 0) -> np.ndarray:
    """Extract a single frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def perimeter_to_spec(perimeter: CirclePerimeter | PolygonPerimeter) -> dict[str, Any]:
    """Convert a Perimeter to a PerimeterSpec dict matching the Rust serde format."""
    if isinstance(perimeter, CirclePerimeter):
        return {
            "shape": "circle",
            "label": perimeter.label,
            "center_x": float(perimeter.center_pixels[0]),
            "center_y": float(perimeter.center_pixels[1]),
            "radius": float(perimeter.radius_pixels),
        }
    else:
        return {
            "shape": "polygon",
            "label": perimeter.label,
            "vertices": [[float(x), float(y)] for x, y in perimeter.vertices_pixels],
        }


def _get_raw_result(
    image: np.ndarray,
    backend: str,
    recipe: ExperimentDetectionRecipe,
) -> DetectionResult:
    """Get raw (unfiltered) detection result for debug purposes.

    Model is already loaded from the failed detect() call, so re-running is cheap.
    """
    from bekiper_auto_detect.detect import (
        auto_detect_perimeters,
        auto_detect_perimeters_with_text,
        auto_detect_perimeters_yoloe,
    )

    if backend == "sam2_auto":
        return auto_detect_perimeters(image, config=recipe.config)
    elif backend == "sam2_text" or backend == "grounding_dino_sam2":
        return auto_detect_perimeters_with_text(image, recipe.text_prompt, config=recipe.config)
    elif backend == "yoloe":
        return auto_detect_perimeters_yoloe(image, [recipe.text_prompt], config=recipe.config)
    return DetectionResult(masks=[])


def save_detection_debug(
    debug_dir: Path,
    image: np.ndarray,
    backend: str,
    recipe_name: str,
    raw_result: DetectionResult | None = None,
    error: str | None = None,
) -> None:
    """Save detection debug artifacts (always, not just on failure)."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Save detection frame
    cv2.imwrite(str(debug_dir / "detection_frame.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if raw_result is None or not raw_result.masks:
        meta: dict[str, Any] = {
            "backend": backend,
            "recipe": recipe_name,
            "raw_mask_count": 0,
            "masks": [],
        }
        if error:
            meta["error"] = error
        with open(debug_dir / "debug.toml", "w") as f:
            f.write(tomli_w.dumps(meta))
        return

    # Save individual mask PNGs
    masks_dir = debug_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    mask_meta = []
    for i, mask in enumerate(raw_result.masks):
        mask_img = (mask.mask.astype(np.uint8)) * 255
        cv2.imwrite(str(masks_dir / f"mask_{i:03d}.png"), mask_img)
        mask_meta.append(
            {
                "index": i,
                "score": float(mask.score),
                "area": int(mask.area),
                "bbox": list(mask.bbox),
                "label": mask.label or "",
            }
        )

    # Save overlay: all masks on the detection frame with different colors
    overlay = image.copy()
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for i, mask in enumerate(raw_result.masks):
        color = colors[i % len(colors)]
        colored = np.zeros_like(overlay)
        colored[:] = color
        mask_bool = mask.mask.astype(bool)
        overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.5, colored[mask_bool], 0.5, 0)
        # Draw contour
        contours, _ = cv2.findContours(mask.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    cv2.imwrite(str(debug_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    meta = {
        "backend": backend,
        "recipe": recipe_name,
        "raw_mask_count": len(raw_result.masks),
        "masks": mask_meta,
    }
    if error:
        meta["error"] = error
    with open(debug_dir / "debug.toml", "w") as f:
        f.write(tomli_w.dumps(meta))

    logger.info(f"Saved {len(raw_result.masks)} debug masks to {debug_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect arena objects from video")
    parser.add_argument("--video", default=None, help="Path to input video file")
    parser.add_argument("--recipe", required=True, help="Detection recipe name")
    parser.add_argument("--backend", required=True, help="Detection backend name")
    parser.add_argument("--output", default=None, help="Path to output perimeters.toml")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to extract")
    parser.add_argument("--frame-output", default=None, help="Path to save detection frame PNG")
    parser.add_argument("--debug-output", default=None, help="Directory for detection debug artifacts")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Validate recipe/backend/config imports without reading video or writing outputs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to bekiper_config.toml (reads [detect] section).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help="Compute backend override (auto|cuda|mps|cpu).",
    )
    parser.add_argument("--sam2-model", default=None, help="Override SAM2 checkpoint.")
    parser.add_argument("--grounding-dino-model", default=None, help="Override GroundingDINO model ID.")
    parser.add_argument("--yoloe-model", default=None, help="Override YOLOE checkpoint.")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=None,
        help="GroundingDINO box-score threshold.",
    )
    args = parser.parse_args()

    if args.recipe not in RECIPE_REGISTRY:
        logger.error(f"Unknown recipe: {args.recipe}. Available: {list(RECIPE_REGISTRY.keys())}")
        sys.exit(1)
    valid_backends = {
        "classical_cv",
        "sam2_auto",
        "sam2_text",
        "grounding_dino_sam2",
        "yoloe",
    }
    if args.backend not in valid_backends:
        logger.error(f"Unknown backend: {args.backend}. Available: {sorted(valid_backends)}")
        sys.exit(1)

    detector_config = DetectorConfig.load(
        config_path=args.config,
        device=args.device,
        sam2_model=args.sam2_model,
        grounding_dino_model=args.grounding_dino_model,
        yoloe_model=args.yoloe_model,
        confidence=args.confidence,
        box_threshold=args.box_threshold,
    )
    recipe = RECIPE_REGISTRY[args.recipe](config=detector_config)
    # Override the backend so the Rust fallback chain works: when Rust calls us
    # with e.g. backend=sam2_auto, we should use that backend even if the recipe
    # normally uses classical CV in its detect() override.
    recipe.backend = args.backend
    # For recipes that override detect() with classical CV (e.g. _NORTBase),
    # bypass their override and use the base class detect() which respects self.backend.
    from bekiper_auto_detect.recipe import _NORTBase

    if isinstance(recipe, _NORTBase) and args.backend != "classical_cv":
        recipe.detect = lambda image: ExperimentDetectionRecipe.detect(recipe, image)  # type: ignore[method-assign]

    if args.smoke:
        logger.info(f"Smoke OK recipe={args.recipe} backend={args.backend}")
        return

    if args.video is None:
        logger.error("--video is required unless --smoke is set")
        sys.exit(2)
    if args.output is None:
        logger.error("--output is required unless --smoke is set")
        sys.exit(2)

    logger.info(f"Extracting frame {args.frame} from {args.video}")
    image = extract_frame(args.video, args.frame)

    logger.info(f"Running recipe={args.recipe} backend={args.backend}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the detection frame for visual inspection
    if args.frame_output:
        frame_path = Path(args.frame_output)
    else:
        frame_path = output_path.with_name("detection_frame.png")
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved detection frame to {frame_path}")

    # Run detection — always save debug artifacts
    debug_dir = Path(args.debug_output) if args.debug_output else None
    try:
        result = recipe.detect(image)
    except (ValueError, Exception) as e:
        if debug_dir:
            # Re-run without count validation to capture raw masks for debugging.
            # The model is already loaded so this is fast.
            raw_result = None
            try:
                raw_result = _get_raw_result(image, args.backend, recipe)
            except Exception:
                pass
            save_detection_debug(debug_dir, image, args.backend, args.recipe, raw_result, error=str(e))
        logger.error(f"Detection failed: {e}")
        sys.exit(1)

    # Save debug artifacts on success too
    if debug_dir:
        save_detection_debug(debug_dir, image, args.backend, args.recipe, result)

    labeled_masks = recipe.assign_labels(result.masks)

    specs = []
    for mask in labeled_masks:
        perimeter = mask_to_perimeter(mask)
        specs.append(perimeter_to_spec(perimeter))

    with open(args.output, "w") as f:
        f.write(tomli_w.dumps({"perimeters": specs}))

    logger.info(f"Wrote {len(specs)} perimeters to {args.output}")


if __name__ == "__main__":
    main()
