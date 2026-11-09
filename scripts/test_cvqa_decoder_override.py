#!/usr/bin/env python3
"""Verify CVQA's scoped FFmpeg VideoCapture substitution and OpenCV fallback."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import cv2
from pixvqa_cvqahdr.adapter import _CaptureOverride
from pixvqa_video.ffmpeg_bgr import FFmpegCapture


def main() -> int:
    original = cv2.VideoCapture
    previous = os.environ.get("PIXVQA_LEARNED_DECODER")
    try:
        os.environ["PIXVQA_LEARNED_DECODER"] = "ffmpeg"
        with _CaptureOverride(cv2):
            assert cv2.VideoCapture is FFmpegCapture
        assert cv2.VideoCapture is original

        os.environ["PIXVQA_LEARNED_DECODER"] = "opencv"
        with _CaptureOverride(cv2):
            assert cv2.VideoCapture is original
        assert cv2.VideoCapture is original

        os.environ["PIXVQA_LEARNED_DECODER"] = "ffmpeg"
        try:
            with _CaptureOverride(cv2):
                raise RuntimeError("intentional")
        except RuntimeError as exc:
            assert str(exc) == "intentional"
        assert cv2.VideoCapture is original
        print("PASS: CVQA FFmpeg capture override/restoration and OpenCV fallback")
        return 0
    finally:
        if previous is None:
            os.environ.pop("PIXVQA_LEARNED_DECODER", None)
        else:
            os.environ["PIXVQA_LEARNED_DECODER"] = previous
        cv2.VideoCapture = original


if __name__ == "__main__":
    raise SystemExit(main())
