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
    from pixvqa_video.ffmpeg_bins import FFmpegBinError, resolve_ffmpeg_bin, resolve_ffprobe_bin

    try:
        ffmpeg = resolve_ffmpeg_bin()
        ffprobe = resolve_ffprobe_bin()
    except FFmpegBinError as exc:
        print(f"SKIP: {exc}")
        return 0
    # Fixture encoding uses a host encoder (lavfi); FFmpegCapture decode stays on pinned bins.
    fixture_ffmpeg = os.environ.get("PIXVQA_TEST_FIXTURE_FFMPEG", "").strip()
    if not fixture_ffmpeg:
        learned_bin = os.environ.get("PIXVQA_LEARNED_ENV_DIR", "").strip()
        learned_bin = str(Path(learned_bin).expanduser() / "bin") if learned_bin else None
        search_path = os.pathsep.join(
            entry for entry in os.environ.get("PATH", "").split(os.pathsep)
            if entry and entry != learned_bin
        )
        fixture_ffmpeg = shutil.which("ffmpeg", path=search_path or None)
    if not fixture_ffmpeg:
        print("SKIP: host ffmpeg unavailable for lavfi test fixture")
        return 0
    with tempfile.TemporaryDirectory(prefix="pixvqa_ffmpeg_capture_test_") as root:
        path = Path(root) / "test.mp4"
        subprocess.run(
            [fixture_ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=6:duration=2",
             "-frames:v", "12", "-c:v", "mpeg4", str(path)],
            check=True,
        )
        _ = ffprobe  # resolved above; FFmpegCapture uses the same pinned pair internally
        first, _ = decode(path)
        second, _ = decode(path)
        assert first == second, "FFmpegCapture decoded different pixels"
        with FFmpegCapture(str(path)) as early:
            assert early.read()[0]
        print("PASS: FFmpegCapture sequential decode, frame metadata and reproducibility")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
