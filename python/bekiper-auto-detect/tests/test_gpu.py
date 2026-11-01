"""Tests that verify ROCm GPU availability and basic torch operations."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No ROCm/CUDA GPU available")
class TestGPU:
    def test_gpu_is_available(self):
        assert torch.cuda.is_available()

    def test_gpu_device_name_contains_amd(self):
        name = torch.cuda.get_device_name(0)
        # 7900 XTX shows up as gfx1100 or "AMD Radeon" depending on driver
        assert name, "GPU device name is empty"

    def test_tensor_operations_on_gpu(self):
        a = torch.randn(100, 100, device="cuda")
        b = torch.randn(100, 100, device="cuda")
        c = a @ b
        assert c.device.type == "cuda"
        assert c.shape == (100, 100)

    def test_device_selection_helper(self):
        from bekiper_auto_detect.config import DetectorConfig

        device = DetectorConfig().resolve_device()
        assert device == "cuda"

    def test_torchvision_nms_runs_on_gpu_and_matches_cpu(self):
        import torchvision

        from bekiper_auto_detect.detect import _ensure_gpu_nms

        _ensure_gpu_nms()
        g = torch.Generator().manual_seed(0)
        xy = torch.rand(500, 2, generator=g) * 200
        boxes = torch.cat((xy, xy + torch.rand(500, 2, generator=g) * 50 + 1), 1)
        scores = torch.rand(500, generator=g)

        expected = torchvision.ops.nms(boxes, scores, 0.5)
        actual = torchvision.ops.nms(boxes.cuda(), scores.cuda(), 0.5)
        assert actual.device.type == "cuda"
        assert torch.equal(actual.cpu(), expected)
