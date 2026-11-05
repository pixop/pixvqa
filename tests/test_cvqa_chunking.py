"""Isolated tests for CVQA adapter temporal sampling and chunked pooling.

Run with: python tests/test_cvqa_chunking.py
These tests do not require a CVQA checkpoint or GPU.
"""
import importlib.util
import math
import os
import sys
import types
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("pixvqa_cvqahdr")
pkg.PINNED_COMMIT = "test"
pkg.PINNED_UPSTREAM = "test"
sys.modules["pixvqa_cvqahdr"] = pkg
device = types.ModuleType("pixvqa_device")
device.DEVICE_CHOICES = {"cpu", "cuda", "auto"}
device.DeviceError = RuntimeError
for name in ("auto_notice", "device_record", "peak_rss_bytes", "resolve_device"):
    setattr(device, name, lambda *args, **kwargs: None)
sys.modules["pixvqa_device"] = device
spec = importlib.util.spec_from_file_location("cvqa_adapter_under_test", root / "python/pixvqa_cvqahdr/adapter.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SamplingTests(unittest.TestCase):
    def test_original_midpoints(self):
        p = module.sampling_plan(2200, 60.0)
        self.assertEqual(p["sampled_frames"], 36)
        self.assertEqual(p["sample_interval_frames"], 60)
        self.assertEqual(module.sample_schedule(p)[-1]["frame_index"], 2130)

    def test_two_second_intervals(self):
        p = module.sampling_plan(3600, 60.0, 2.0)
        self.assertEqual(p["sampled_frames"], 30)
        self.assertEqual([x["frame_index"] for x in module.sample_schedule(p)[:3]], [60, 180, 300])

    def test_bad_interval(self):
        with self.assertRaises(module.PreflightError):
            module.sampling_plan(0, 60.0)

    def test_chunked_matches_full_mean(self):
        import torch
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quality1 = torch.nn.Linear(3, 1, bias=False)
                with torch.no_grad():
                    self.quality1.weight[:] = torch.tensor([[0.25, 0.5, -0.25]])
            def forward(self, x):
                z = x.reshape(-1, 3)
                scores = self.quality1(z).reshape(x.shape[0], x.shape[1]).mean(dim=1)
                return (scores,) * 5
        model = TinyModel().eval()
        video = torch.randn(60, 3)
        with torch.no_grad():
            expected, _, _, _ = module.forward_with_samples(model, video.unsqueeze(0), torch)
            score, scores, parity = module.forward_chunked(model, video, torch.device("cpu"), torch, 8)
        self.assertEqual(len(scores), 60)
        self.assertTrue(parity)
        self.assertTrue(math.isclose(score, expected[0].item(), abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
