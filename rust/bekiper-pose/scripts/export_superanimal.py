#!/usr/bin/env python3
"""Download and export DLC SuperAnimal-TopViewMouse to ONNX for tract inference.

Usage:
    python export_superanimal.py [--model hrnet_w32] [--output models/]

Requirements (install with `uv pip install`):
    deeplabcut>=3.0.0rc1 (--no-deps --prerelease allow)
    huggingface_hub
    torch (system torch via nix is fine)
    timm
    albumentations
    onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SUPERANIMAL = "superanimal_topviewmouse"

MODEL_FILENAMES = {
    "hrnet_w32": "superanimal_topviewmouse_hrnet_w32.pt",
    "hrnet_w48": "superanimal_topviewmouse_hrnet_w48.pt",
    "resnet_50": "superanimal_topviewmouse_resnet_50.pt",
    "resnet_101": "superanimal_topviewmouse_resnet_101.pt",
    "rtmpose_s": "superanimal_topviewmouse_rtmpose_s.pt",
    "rtmpose_m": "superanimal_topviewmouse_rtmpose_m.pt",
    "rtmpose_x": "superanimal_topviewmouse_rtmpose_x.pt",
}

HF_REPO = "mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse"


def download_checkpoint(model_name: str, output_dir: Path) -> Path:
    """Download the PyTorch checkpoint from HuggingFace."""
    from huggingface_hub import hf_hub_download

    filename = MODEL_FILENAMES[model_name]
    dest = output_dir / filename
    if dest.exists():
        print(f"Checkpoint already exists: {dest}")
        return dest

    print(f"Downloading {filename} from {HF_REPO}...")
    path = hf_hub_download(HF_REPO, filename, local_dir=str(output_dir))
    print(f"Downloaded to: {path}")
    return Path(path)


def _patch_dlc_imports():
    """Stub out DLC's heavy import chain so we can import the pytorch subpackage.

    DLC's import graph pulls in filterpy, numba, tensorflow, etc. through
    deeplabcut.utils -> deeplabcut.core.trackingutils. We stub these out
    before any DLC import.
    """
    import importlib.util
    import sys
    import types

    def _make_stub(name, attrs=None):
        """Create a stub module with optional attributes."""
        mod = types.ModuleType(name)
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        # Set __spec__ so torch._dynamo.trace_rules doesn't crash
        mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    # Stub out heavy dependencies that DLC imports transitively
    for mod_name in [
        "filterpy",
        "filterpy.common",
        "filterpy.kalman",
        "numba",
        "tensorflow",
        "tensorflow.python",
        "tables",
    ]:
        if mod_name not in sys.modules:
            _make_stub(mod_name)

    # numba.jit needs to be a no-op decorator, and numba needs to be a package
    numba_mod = sys.modules["numba"]
    numba_mod.__path__ = []  # make it a package
    numba_mod.jit = lambda *a, **kw: lambda fn: fn
    numba_mod.njit = numba_mod.jit

    # numba.core.errors needs NumbaPerformanceWarning
    _make_stub("numba.core")
    sys.modules["numba.core"].__path__ = []
    _make_stub(
        "numba.core.errors",
        {
            "NumbaPerformanceWarning": type("NumbaPerformanceWarning", (UserWarning,), {}),
        },
    )

    # filterpy stubs
    sys.modules["filterpy.common"].kinematic_kf = lambda *a, **kw: None
    sys.modules["filterpy.kalman"].KalmanFilter = type("KalmanFilter", (), {})

    # Make stub packages (need __path__ for submodule imports)
    for pkg in ["filterpy", "numba", "tensorflow", "tables"]:
        if pkg in sys.modules:
            if not hasattr(sys.modules[pkg], "__path__"):
                sys.modules[pkg].__path__ = []

    # Find deeplabcut package location without importing it
    spec = importlib.util.find_spec("deeplabcut")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("deeplabcut package not found")

    dlc_dir = spec.submodule_search_locations[0]

    # Replace the deeplabcut package with a namespace-only module so
    # subpackage imports work without triggering __init__.py side-effects.
    ns = types.ModuleType("deeplabcut")
    ns.__path__ = [str(dlc_dir)]
    ns.__package__ = "deeplabcut"
    init_file = str(Path(dlc_dir) / "__init__.py")
    ns.__file__ = init_file
    ns.__spec__ = importlib.util.spec_from_file_location(
        "deeplabcut", init_file, submodule_search_locations=[str(dlc_dir)]
    )
    sys.modules["deeplabcut"] = ns


# Patch before any DLC subpackage import
_patch_dlc_imports()


def build_model(model_name: str):
    """Build the DLC PoseModel architecture using the DLC 3.0 config system."""
    from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import load_super_animal_config

    config = load_super_animal_config(
        super_animal=SUPERANIMAL,
        model_name=model_name,
        detector_name=None,
        max_individuals=30,
        device="cpu",
    )

    model = PoseModel.build(cfg=config["model"], weight_init=None, pretrained_backbone=False)
    return model, config


def get_labels() -> list[str]:
    """Return the ordered keypoint labels for SuperAnimal-TopViewMouse."""
    return [
        "nose",
        "left_ear",
        "right_ear",
        "left_ear_tip",
        "right_ear_tip",
        "left_eye",
        "right_eye",
        "neck",
        "mid_back",
        "mouse_center",
        "mid_backend",
        "mid_backend2",
        "mid_backend3",
        "tail_base",
        "tail1",
        "tail2",
        "tail3",
        "tail4",
        "tail5",
        "left_shoulder",
        "left_midside",
        "left_hip",
        "right_shoulder",
        "right_midside",
        "right_hip",
        "tail_end",
        "head_midpoint",
    ]


class HeatmapWrapper(torch.nn.Module):
    """Wrap PoseModel to output a single heatmap tensor for ONNX."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs["bodypart"]["heatmap"]


def export_onnx(
    model: torch.nn.Module,
    checkpoint_path: Path,
    onnx_path: Path,
    input_size: tuple[int, int] = (256, 256),
    opset: int = 14,
) -> None:
    """Load weights and export to ONNX."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    wrapper = HeatmapWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, input_size[0], input_size[1])

    print(f"Exporting ONNX to {onnx_path} (opset {opset})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["image"],
            output_names=["heatmap"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes={
                "image": {0: "batch_size"},
                "heatmap": {0: "batch_size"},
            },
        )

    print(f"ONNX export complete: {onnx_path}")

    # Write model spec JSON for bekiper-pose
    spec = {
        "backend": "deeplabcut",
        "labels": get_labels(),
        "input_height": input_size[0],
        "input_width": input_size[1],
        "input_channels": 3,
        "channels_first": True,
        "normalization": {
            "ImageNet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
        },
        "output_layout": "Heatmaps",
    }
    spec_path = onnx_path.with_suffix(".json")
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"Model spec written to: {spec_path}")


def validate_onnx(onnx_path: Path, input_size: tuple[int, int] = (256, 256)) -> None:
    """Validate ONNX model with onnxruntime if available."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        dummy = torch.randn(1, 3, input_size[0], input_size[1]).numpy()
        output = session.run(None, {"image": dummy})
        print(f"ONNX validation OK — output shape: {output[0].shape}")
    except ImportError:
        print("(onnxruntime not installed, skipping validation)")


def main():
    parser = argparse.ArgumentParser(description="Export DLC SuperAnimal to ONNX")
    parser.add_argument(
        "--model",
        choices=list(MODEL_FILENAMES.keys()),
        default="hrnet_w32",
        help="Model variant (default: hrnet_w32)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Output directory (default: models/)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Model input size H W (default: 256 256)",
    )
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Download
    checkpoint_path = download_checkpoint(args.model, args.output)

    # Build model architecture
    model, _ = build_model(args.model)

    # Export
    onnx_name = f"superanimal_topviewmouse_{args.model}.onnx"
    onnx_path = args.output / onnx_name
    export_onnx(model, checkpoint_path, onnx_path, tuple(args.input_size), args.opset)

    # Validate
    validate_onnx(onnx_path, tuple(args.input_size))


if __name__ == "__main__":
    main()
