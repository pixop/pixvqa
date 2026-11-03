#!/usr/bin/env python3
"""FFmpeg atomic .tmp container writes (explicit muxer)."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from numerical_conformance.ffmpeg_util import (  # noqa: E402
    atomic_tmp_path,
    container_format_flags,
    ffprobe_stream,
    run_ffmpeg,
)


def test_container_format_flags() -> None:
    assert container_format_flags(Path("x.mp4")) == ["-f", "mp4"]
    assert container_format_flags(Path("x.mkv")) == ["-f", "matroska"]
    tmp = atomic_tmp_path(Path("distorted_lossy_60f.mp4"))
    assert tmp.name == "distorted_lossy_60f.mp4.tmp"


def test_ffmpeg_atomic_tmp_mp4_and_mkv() -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        print("SKIP ffmpeg-atomic-tmp (ffmpeg/ffprobe not on PATH)")
        return
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mp4_final = root / "clip.mp4"
        mkv_final = root / "clip.mkv"
        mp4_tmp = atomic_tmp_path(mp4_final)
        mkv_tmp = atomic_tmp_path(mkv_final)
        lavfi = ["-f", "lavfi", "-i", "color=c=black:s=64x64:r=60:d=0.05"]
        run_ffmpeg(
            ffmpeg,
            [
                "-y",
                *lavfi,
                "-frames:v",
                "3",
                "-c:v",
                "libx265",
                "-tag:v",
                "hvc1",
                *container_format_flags(mp4_final),
                str(mp4_tmp),
            ],
            label="atomic-mp4",
        )
        mp4_tmp.replace(mp4_final)
        run_ffmpeg(
            ffmpeg,
            [
                "-y",
                *lavfi,
                "-frames:v",
                "3",
                "-c:v",
                "libx265",
                "-x265-params",
                "lossless=1",
                *container_format_flags(mkv_final),
                str(mkv_tmp),
            ],
            label="atomic-mkv",
        )
        mkv_tmp.replace(mkv_final)
        for path, frames in ((mp4_final, 3), (mkv_final, 3)):
            info = ffprobe_stream(ffmpeg, path)
            assert info.get("width") == "64"
            assert info.get("height") == "64"
            nb = info.get("nb_read_frames") or info.get("nb_frames") or ""
            if nb.isdigit():
                assert int(nb) == frames
            assert info.get("codec_name") == "hevc"


def main() -> int:
    test_container_format_flags()
    print("OK   container-format-flags")
    test_ffmpeg_atomic_tmp_mp4_and_mkv()
    print("OK   ffmpeg-atomic-tmp-mp4-mkv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
