#!/usr/bin/env python3
"""Smoke-test deterministic FFmpegCapture behavior on a generated short MP4."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pixvqa_video.ffmpeg_bgr import FFmpegCapture


def decode(path: Path) -> tuple[list[str], float]:
    import cv2

    cap = FFmpegCapture(str(path))
    assert cap.isOpened()
    assert cap.getBackendName() == "FFMPEG_PIPE"
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 96
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 64
    assert int(cap.get(cv2.CAP_PROP_FPS)) == 6
    fingerprints = []
    try:
        for _ in range(12):
            ok, frame = cap.read()
            assert ok and frame.shape == (64, 96, 3)
            fingerprints.append(hashlib.sha256(frame.tobytes()).hexdigest())
        assert cap.read()[0] is False
    finally:
        cap.release()
    return fingerprints, 6.0


def main() -> int:
    ffmpeg = os.environ.get("PIXVQA_FFMPEG_BIN", "ffmpeg")
    ffprobe = os.environ.get("PIXVQA_FFPROBE_BIN", "ffprobe")
    if not shutil.which(ffmpeg) or not shutil.which(ffprobe):
        print("SKIP: ffmpeg/ffprobe unavailable")
        return 0
    with tempfile.TemporaryDirectory(prefix="pixvqa_ffmpeg_capture_test_") as root:
        path = Path(root) / "test.mp4"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=6:duration=2",
             "-frames:v", "12", "-c:v", "mpeg4", str(path)],
            check=True,
        )
        first, _ = decode(path)
        second, _ = decode(path)
        assert first == second, "FFmpegCapture decoded different pixels"
        with FFmpegCapture(str(path)) as early:
            assert early.read()[0]
        print("PASS: FFmpegCapture sequential decode, frame metadata and reproducibility")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
