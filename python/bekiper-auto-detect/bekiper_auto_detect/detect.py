"""Detection backends: SAM2 (automatic), GroundingDINO+SAM2 (text-prompted), YOLOE (lightweight)."""

from __future__ import annotations

import functools
import logging
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from bekiper_auto_detect.config import DetectorConfig
from bekiper_auto_detect.model import DetectedMask, DetectionResult

logger = logging.getLogger(__name__)

# Lazy-loaded model caches — keyed by model identifier so switching backends
# mid-session doesn't silently reuse the wrong weights.
_sam2_cache: dict[str, Any] = {}
_grounding_cache: dict[str, tuple[Any, Any]] = {}


@functools.cache
def _ensure_gpu_nms() -> None:
    """Give ``torchvision::nms`` a GPU kernel when torchvision ships CPU-only ops.

    The Hydra-cached ROCm torchvision is built without HIP kernels, but ultralytics
    calls ``torchvision.ops.nms`` on GPU tensors. ``TorchNMS.nms`` is pure torch and
    matches torchvision's result.
    """
    import torchvision  # noqa: F401  — defines the torchvision::nms schema
    from ultralytics.utils.nms import TorchNMS

    if not torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "CUDA"):
        torch.library.impl("torchvision::nms", "CUDA", TorchNMS.nms)


def _bbox_tuple(seq: Any) -> tuple[int, int, int, int]:
    """Convert any sequence to a 4-int bbox tuple."""
    vals = [int(x) for x in seq]
    return (vals[0], vals[1], vals[2], vals[3])


def _get_sam2_model(model_name: str) -> Any:
    """Lazily load (and cache) a SAM2 model via ultralytics."""
    if model_name not in _sam2_cache:
        from ultralytics import SAM  # type: ignore[attr-defined]

        _ensure_gpu_nms()

        _sam2_cache[model_name] = SAM(model_name)
        logger.info(f"SAM2 loaded via ultralytics ({model_name})")
    return _sam2_cache[model_name]


def _get_grounding_model(model_id: str, device: str) -> tuple[Any, Any]:
    """Lazily load (and cache) a GroundingDINO processor + model."""
    key = f"{model_id}@{device}"
    if key not in _grounding_cache:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)  # type: ignore[no-untyped-call]
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        _grounding_cache[key] = (processor, model)
        logger.info(f"GroundingDINO loaded on {device} ({model_id})")
    return _grounding_cache[key]


def auto_detect_perimeters(
    image: np.ndarray[Any, Any],
    *,
    config: DetectorConfig | None = None,
) -> DetectionResult:
    """Prompt-free detection using SAM2 automatic mask generator."""
    cfg = config or DetectorConfig()
    sam2 = _get_sam2_model(cfg.sam2_model)

    # Call with no prompts to trigger automatic segment-everything mode.
    # The generate() method's defaults (points_stride=32, conf_thres=0.88,
    # stability_score_thresh=0.95) are used internally by the predictor.
    results = sam2(
        image,
        verbose=False,
    )

    masks = []
    for result in results:
        if result.masks is None:
            continue
        h, w = image.shape[:2]
        for i, seg_mask in enumerate(result.masks.data.cpu().numpy()):
            full_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            box = result.boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            masks.append(
                DetectedMask(
                    mask=full_mask,
                    score=float(box.conf[0]) if box.conf is not None else 1.0,
                    area=int(full_mask.sum()),
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                )
            )

    return DetectionResult(masks=masks)


def auto_detect_perimeters_with_text(
    image: np.ndarray[Any, Any],
    text_prompt: str,
    *,
    config: DetectorConfig | None = None,
) -> DetectionResult:
    """Text-prompted detection using GroundingDINO + SAM2."""
    cfg = config or DetectorConfig()
    device = cfg.resolve_device()

    processor, gd_model = _get_grounding_model(cfg.grounding_dino_model, device)
    sam2 = _get_sam2_model(cfg.sam2_model)

    pil_image = Image.fromarray(image)
    # GroundingDINO uses "." as class delimiter; ensure prompt ends with one
    if not text_prompt.endswith("."):
        text_prompt = text_prompt + "."
    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = gd_model(**inputs)

    gd_results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=cfg.box_threshold,
        target_sizes=[pil_image.size[::-1]],
    )[0]

    if len(gd_results["boxes"]) == 0:
        return DetectionResult(masks=[])

    boxes_xyxy = gd_results["boxes"].cpu().numpy().tolist()

    results = sam2(image, bboxes=boxes_xyxy, verbose=False)

    masks = []
    for result in results:
        if result.masks is None:
            continue
        h, w = image.shape[:2]
        for i, seg_mask in enumerate(result.masks.data.cpu().numpy()):
            full_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            box = result.boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = gd_results["labels"][i] if i < len(gd_results["labels"]) else ""
            masks.append(
                DetectedMask(
                    mask=full_mask.astype(bool),
                    score=float(box.conf[0]) if box.conf is not None else 1.0,
                    label=str(label),
                    area=int(full_mask.sum()),
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                )
            )

    return DetectionResult(masks=masks)


def auto_detect_perimeters_yoloe(
    image: np.ndarray[Any, Any],
    class_names: list[str],
    *,
    config: DetectorConfig | None = None,
) -> DetectionResult:
    """Lightweight open-vocabulary detection using YOLOE (ultralytics)."""
    from ultralytics import YOLO  # type: ignore[attr-defined]

    _ensure_gpu_nms()

    cfg = config or DetectorConfig()
    model = YOLO(cfg.yoloe_model)
    model.set_classes(class_names)

    results = model.predict(image, conf=cfg.confidence, verbose=False)

    masks = []
    for result in results:
        if result.masks is None:
            continue
        for i, seg_mask in enumerate(result.masks.data.cpu().numpy()):  # type: ignore[union-attr]
            h, w = image.shape[:2]
            full_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            box = result.boxes[i]  # type: ignore[index]
            cls_idx = int(box.cls[0])
            masks.append(
                DetectedMask(
                    mask=full_mask,
                    score=float(box.conf[0]),
                    label=class_names[cls_idx] if cls_idx < len(class_names) else "",
                    area=int(full_mask.sum()),
                    bbox=_bbox_tuple(box.xywh[0].cpu().numpy()),
                )
            )

    return DetectionResult(masks=masks)
