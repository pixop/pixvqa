#!/usr/bin/env python3
"""CVQA-HDR-NR production decoder/sampler contract without GPU inference."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import cv2
from pixvqa_cvqahdr import adapter


def main() -> int:
    original = cv2.VideoCapture
    sampler_source = inspect.getsource(adapter.preprocess_custom_interval)
    runner_source = inspect.getsource(adapter.run)
    preflight_source = inspect.getsource(adapter.preflight_video)
    assert "FFmpegCapture(test)" in sampler_source
    assert "FFmpegCapture(path)" in preflight_source
    assert "preprocess_custom_interval(test, sampling, torch)" in runner_source
    assert "_CaptureOverride" not in runner_source
    assert "VQA_NR.video_processing(test)" not in runner_source
    assert "PIXVQA_LEARNED_DECODER" not in inspect.getsource(adapter)
    assert 'get("PIXVQA_LEARNED_DETERMINISTIC", "1")' in inspect.getsource(adapter)
    assert cv2.VideoCapture is original, "CVQA must not globally monkeypatch OpenCV"

    plan = adapter.sampling_plan(1199, 60.0)
    selected = adapter.sample_schedule(plan)
    assert len(selected) == 19, len(selected)
    assert selected[0]["frame_index"] == 30
    assert selected[-1]["frame_index"] == 1110
    print("PASS: CVQA FFmpeg-only sampling contract; 19 1-Hz midpoints and no OpenCV decoder override")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
